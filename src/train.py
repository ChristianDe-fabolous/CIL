import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DepthDataset
from src.models.model import DepthModel
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


def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip, log_interval):
    model.train()
    total_loss = 0.0
    for i, (images, depths) in enumerate(loader):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=amp):
            preds = model(images)
            loss = silog_loss(preds, depths)
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
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        preds = model(images)
        total += si_rmse(preds, depths).item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--decoder", choices=["transformer", "conv"], help="override decoder_type")
    parser.add_argument("--pretrained", type=lambda x: x.lower() == "true", help="override encoder_pretrained")
    parser.add_argument("--freeze_encoder", action="store_true", help="freeze ViT encoder weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.decoder:
        cfg["model"]["decoder_type"] = args.decoder
    if args.pretrained is not None:
        cfg["model"]["encoder_pretrained"] = args.pretrained
    if args.freeze_encoder:
        cfg["training"]["freeze_encoder"] = True

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  decoder={cfg['model']['decoder_type']}  pretrained={cfg['model']['encoder_pretrained']}")

    # --- data ---
    image_dir = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_image_dir"])
    depth_dir  = os.path.join(cfg["data"]["data_root"], cfg["data"]["train_depth_dir"])
    img_size   = cfg["model"]["img_size"]

    train_dataset = DepthDataset(image_dir, depth_dir, img_size=img_size, augment=True)
    val_dataset   = DepthDataset(image_dir, depth_dir, img_size=img_size, augment=False)

    n = len(train_dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["training"]["seed"])).tolist()
    val_size = int(n * cfg["data"]["val_split"])
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_set = Subset(train_dataset, train_idx)
    val_set   = Subset(val_dataset,   val_idx)

    nw = cfg["data"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    print(f"train={len(train_set)}  val={len(val_set)}")

    # --- model ---
    model = DepthModel(
        decoder_type=cfg["model"]["decoder_type"],
        pretrained=cfg["model"]["encoder_pretrained"],
        img_size=img_size,
        patch_size=cfg["model"]["patch_size"],
        decoder_blocks=cfg["model"]["decoder_blocks"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)

    freeze_encoder = cfg["training"].get("freeze_encoder", False)
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("encoder frozen")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"params={n_params:.1f}M  trainable={n_trainable:.1f}M")

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["lr"] * 0.01
    )
    scaler = GradScaler("cuda", enabled=cfg["training"]["amp"])

    # --- checkpoint dir ---
    job_id = os.environ.get("SLURM_JOB_ID")
    run_name = (
        f"{cfg['logging']['experiment_name']}"
        f"_{cfg['model']['decoder_type']}"
        f"_pretrained{cfg['model']['encoder_pretrained']}"
        + (f"_job{job_id}" if job_id else "")
    )
    ckpt_dir = os.path.join(cfg["logging"]["checkpoint_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            cfg["training"]["amp"], cfg["training"]["grad_clip"], cfg["logging"]["log_interval"],
        )
        val_metric = validate(model, val_loader, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}  lr={lr_now:.2e}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_si_rmse": val_metric,
            "cfg": cfg,
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "last.pth"))

        if val_metric < best_val:
            best_val = val_metric
            torch.save(checkpoint, os.path.join(ckpt_dir, "best.pth"))
            print(f"  --> new best (si_rmse={best_val:.4f})")

    print(f"\nDone. Best val SI-RMSE: {best_val:.4f}")
    print(f"Checkpoint dir: {ckpt_dir}")


if __name__ == "__main__":
    main()
