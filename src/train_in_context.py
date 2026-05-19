import os
import sys
import argparse
import random
import glob
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.in_context_dataset import InContextDataset
from src.models.encoder import ViTEncoder
from src.models.in_context_model import InContextDepthModel
from src.retrieval.random_retriever import RandomRetriever
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


def collect_paths(data_root: str, image_dir: str, depth_dir: str):
    img_dir = os.path.join(data_root, image_dir)
    dep_dir = os.path.join(data_root, depth_dir)
    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png")) +
        glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images in {img_dir}")
    depth_paths = [
        os.path.join(dep_dir, os.path.splitext(os.path.basename(p))[0].replace("_rgb", "_depth") + ".npy")
        for p in image_paths
    ]
    return image_paths, depth_paths


def build_retriever(cfg: dict, n_train: int):
    strategy = cfg["retrieval"]["strategy"]
    if strategy == "random":
        return RandomRetriever(n_train, seed=cfg["training"]["seed"])
    elif strategy == "patch_sim":
        from src.retrieval.patch_sim_retriever import PatchSimRetriever
        return PatchSimRetriever(cfg["retrieval"]["index_path"])
    elif strategy in ("gt_depth_sim", "learned"):
        from src.retrieval.graph_retriever import GraphRetriever
        return GraphRetriever(cfg["retrieval"]["graph_path"])
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy!r}")


def make_optimizer(model: InContextDepthModel, cfg: dict):
    lr      = cfg["training"]["lr"]
    enc_lr  = lr / cfg["training"].get("encoder_lr_divisor", 50)
    enc_params  = list(model.encoder.parameters())
    rest_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    return torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr},
         {"params": rest_params, "lr": lr}],
        weight_decay=cfg["training"]["weight_decay"],
    )


def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip, log_interval):
    model.train()
    total_loss = 0.0
    for i, (q_img, q_depth, ctx_imgs, ctx_depths) in enumerate(loader):
        q_img, q_depth     = q_img.to(device), q_depth.to(device)
        ctx_imgs            = ctx_imgs.to(device)
        ctx_depths          = ctx_depths.to(device)

        optimizer.zero_grad()
        with autocast(enabled=amp):
            preds = model(q_img, ctx_imgs, ctx_depths)
            loss  = silog_loss(preds, q_depth)
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
    for q_img, q_depth, ctx_imgs, ctx_depths in loader:
        q_img, q_depth = q_img.to(device), q_depth.to(device)
        ctx_imgs        = ctx_imgs.to(device)
        ctx_depths      = ctx_depths.to(device)
        preds = model(q_img, ctx_imgs, ctx_depths)
        total += si_rmse(preds, q_depth).item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config_in_context.yaml")
    parser.add_argument("--strategy", choices=["random", "patch_sim", "gt_depth_sim", "learned"],
                        help="override retrieval.strategy from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.strategy:
        cfg["retrieval"]["strategy"] = args.strategy

    set_seed(cfg["training"]["seed"])
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = cfg["retrieval"]["strategy"]
    K        = cfg["data"]["num_context"]
    print(f"device={device}  strategy={strategy}  K={K}")

    # --- paths ---
    all_image_paths, all_depth_paths = collect_paths(
        cfg["data"]["data_root"],
        cfg["data"]["train_image_dir"],
        cfg["data"]["train_depth_dir"],
    )
    n = len(all_image_paths)

    # --- deterministic split ---
    g       = torch.Generator().manual_seed(cfg["training"]["seed"])
    indices = torch.randperm(n, generator=g).tolist()
    val_n   = int(n * cfg["data"]["val_split"])
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    train_image_paths = [all_image_paths[i] for i in train_idx]
    train_depth_paths = [all_depth_paths[i] for i in train_idx]
    val_image_paths   = [all_image_paths[i] for i in val_idx]
    val_depth_paths   = [all_depth_paths[i] for i in val_idx]

    print(f"train={len(train_idx)}  val={len(val_idx)}")

    # --- retrievers ---
    train_retriever = build_retriever(cfg, len(train_idx))
    # val queries always retrieve context from the training pool (no leakage)
    val_retriever   = RandomRetriever(len(train_idx), seed=0)

    # --- datasets ---
    img_size = cfg["model"]["img_size"]
    train_ds = InContextDataset(
        train_image_paths, train_depth_paths,
        train_image_paths, train_depth_paths,
        retriever=train_retriever, num_context=K,
        img_size=img_size, augment=True,
    )
    val_ds = InContextDataset(
        val_image_paths, val_depth_paths,
        train_image_paths, train_depth_paths,  # context always from train
        retriever=val_retriever, num_context=K,
        img_size=img_size, augment=False,
    )

    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # --- model ---
    encoder = ViTEncoder(
        pretrained=cfg["model"]["encoder_pretrained"],
        img_size=img_size,
        patch_size=cfg["model"]["patch_size"],
    )
    model = InContextDepthModel(
        encoder=encoder,
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_cross_blocks=cfg["model"]["num_cross_blocks"],
        num_self_blocks=cfg["model"]["num_self_blocks"],
        patch_grid=img_size // cfg["model"]["patch_size"],
        patch_size=cfg["model"]["patch_size"],
    ).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # --- optimiser with differential LR (encoder much lower than decoder) ---
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["lr"] * 0.01,
    )
    scaler = GradScaler(enabled=cfg["training"]["amp"])

    # --- checkpoint dir ---
    run_name = f"in_context_{strategy}_K{K}"
    ckpt_dir = os.path.join(cfg["logging"]["checkpoint_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            cfg["training"]["amp"], cfg["training"]["grad_clip"],
            cfg["logging"]["log_interval"],
        )
        val_metric = validate(model, val_loader, device)
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}"
              f"  lr={optimizer.param_groups[1]['lr']:.2e}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_si_rmse": val_metric, "cfg": cfg,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pth"))
        if val_metric < best_val:
            best_val = val_metric
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            print(f"  --> new best (si_rmse={best_val:.4f})")

    print(f"\nDone. Best val SI-RMSE: {best_val:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
