"""
Zero-shot Depth Anything V2 inference on the competition test set.

Usage:
    uv run python src/predict_da2.py \
        --test_dir  /path/to/test/images \
        --output_dir predictions/da2_base \
        --size      base   # small | base | large
        --output_size 560
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TestDataset


def load_da2(size: str, device: torch.device):
    from transformers import AutoModelForDepthEstimation
    names = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    assert size in names, f"size must be one of {list(names)}"
    model = AutoModelForDepthEstimation.from_pretrained(names[size])
    model.eval().to(device)
    print(f"Loaded Depth Anything V2 {size}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",    required=True)
    parser.add_argument("--output_dir",  default="predictions/da2")
    parser.add_argument("--size",        default="base", choices=["small", "base", "large"])
    parser.add_argument("--img_size",    type=int, default=518,
                        help="resize input to this before passing to DA2 (518 = DA2 native)")
    parser.add_argument("--output_size", type=int, default=560,
                        help="upsample prediction to this size before saving")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_da2(args.size, device)

    dataset = TestDataset(args.test_dir, img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for images, stems in loader:
            images = images.to(device)

            # DA2 forward — returns a DepthEstimatorOutput with .predicted_depth
            out    = model(pixel_values=images)
            depth  = out.predicted_depth.unsqueeze(1)  # (B, 1, H', W')

            if depth.shape[-1] != args.output_size:
                depth = F.interpolate(
                    depth, size=(args.output_size, args.output_size),
                    mode="bilinear", align_corners=False,
                )

            depth = depth.squeeze(1).cpu().numpy()  # (B, H, W)
            for pred, stem in zip(depth, stems):
                out_stem = stem.replace("_rgb", "_depth")
                np.save(os.path.join(args.output_dir, f"{out_stem}.npy"), pred)

    print(f"Saved {len(dataset)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
