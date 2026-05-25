"""
LoRA fine-tuning of Depth Anything V2 on the competition training set.

Applies LoRA to the query and value projections of the DINOv2 backbone.
The DPT head is fully unfrozen (small, so cheap).

Usage:
    uv run python src/train_lora.py --config configs/config_lora.yaml
    SCRIPT=train_lora sbatch submit.sh
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DepthDataset
from src.loss import silog_loss
from src.metrics import si_rmse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device):
    from transformers import AutoModelForDepthEstimation
    from peft import LoraConfig, get_peft_model

    size_map = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    hf_name = size_map[cfg["model"]["encoder_size"]]
    model   = AutoModelForDepthEstimation.from_pretrained(hf_name)

    # LoRA on backbone attention q and v projections only
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model.to(device)


def predict(model, images: torch.Tensor, output_size: int) -> torch.Tensor:
    out   = model(pixel_values=images)
    depth = out.predicted_depth.unsqueeze(1)  # (B, 1, H', W')
    return F.interpolate(depth, size=(output_size, output_size),
                         mode="bilinear", align_corners=False).clamp(min=1e-3)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, log_interval):
    model.train()
    output_size = cfg["model"]["output_size"]
    amp         = cfg["training"]["amp"]
    grad_clip   = cfg["training"]["grad_clip"]

    total_loss = 0.0
    for i, (images, depths) in enumerate(loader):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=amp):
            preds = predict(model, images, output_size)
            loss  = silog_loss(preds, depths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print(f"    [{i+1}/{len(loader)}] loss={loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    output_size = cfg["model"]["output_size"]
    total = 0.0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        preds = predict(model, images, output_size)
        total += si_rmse(preds, depths).item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lora.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  encoder={cfg['model']['encoder_size']}")

    image_dir = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_image_dir"])
    depth_dir  = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_depth_dir"])
    img_size   = cfg["model"]["img_size"]

    train_ds = DepthDataset(image_dir, depth_dir, img_size=img_size, augment=True)
    val_ds   = DepthDataset(image_dir, depth_dir, img_size=img_size, augment=False)

    n       = len(train_ds)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["training"]["seed"])).tolist()
    val_n   = int(n * cfg["data"]["val_split"])
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    nw = cfg["data"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=bs,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=True)
    print(f"train={len(train_idx)}  val={len(val_idx)}")

    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["lr"] * 0.01
    )
    scaler = GradScaler("cuda", enabled=cfg["training"]["amp"])

    job_id   = os.environ.get("SLURM_JOB_ID")
    run_name = f"da2_lora_{cfg['model']['encoder_size']}" + (f"_job{job_id}" if job_id else "")
    ckpt_dir = os.path.join(cfg["logging"]["checkpoint_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                     device, cfg, cfg["logging"]["log_interval"])
        val_metric = validate(model, val_loader, device, cfg)
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_si_rmse": val_metric,
            "cfg":         cfg,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pth"))
        if val_metric < best_val:
            best_val = val_metric
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            print(f"  --> new best (si_rmse={best_val:.4f})")

    print(f"\nDone. Best val SI-RMSE: {best_val:.4f}")
    print(f"Checkpoint dir: {ckpt_dir}")


if __name__ == "__main__":
    main()
