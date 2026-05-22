import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TestDataset
from src.models.model import DepthModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to best.pth")
    parser.add_argument("--test_dir",   required=True, help="directory of test images")
    parser.add_argument("--output_dir", default="predictions/", help="where to save .npy files")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["cfg"]

    model = DepthModel(
        decoder_type=cfg["model"]["decoder_type"],
        pretrained=False,
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        decoder_blocks=cfg["model"]["decoder_blocks"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_si_rmse={ckpt.get('val_si_rmse', 'n/a'):.4f})")

    dataset = TestDataset(args.test_dir, img_size=cfg["model"]["img_size"])
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg["data"].get("num_workers", 3), pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for images, stems in loader:
            images = images.to(device)
            with autocast("cuda", enabled=cfg["training"]["amp"]):
                preds = model(images)  # (B, 1, H, W)
            preds = preds.squeeze(1).cpu().numpy()  # (B, H, W)
            for pred, stem in zip(preds, stems):
                out_stem = stem.replace("_rgb", "_depth")
                np.save(os.path.join(args.output_dir, f"{out_stem}.npy"), pred)

    print(f"Saved {len(dataset)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
