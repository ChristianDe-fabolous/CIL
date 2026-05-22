import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import base64
import zlib


def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="folder with test_*.npy predictions")
    parser.add_argument("--output",   required=True, help="path for output submission.csv")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_csv  = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pred_files = sorted(pred_dir.glob("test_*.npy"))

    for pred_path in pred_files:
        depth = np.load(pred_path)
        img_id = pred_path.stem  # already test_000001_depth
        rows.append({"id": img_id, "Depths": encode_depth(depth)})

    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(out_csv, index=False)

    print(f"Saved submission to {out_csv}")
    print(f"Number of predictions: {len(df)}")


if __name__ == "__main__":
    main()