"""
Precompute retrieval indices before training the in-context model.

Examples
--------
# Path 1 — patch similarity using vit_base (or vit_large_patch16_224)
python -m src.retrieval.build_index patch_sim \\
    --data_root data/ --save_path data/retrieval/patch_sim.pt \\
    --encoder vit_base_patch16_224 --device cuda

# Path 3a — GT depth histogram similarity
python -m src.retrieval.build_index gt_depth \\
    --data_root data/ --save_path data/retrieval/gt_depth_graph.pt

# Path 3b — learned graph from a fine-tuned baseline checkpoint
python -m src.retrieval.build_index learned \\
    --data_root data/ --save_path data/retrieval/learned_graph.pt \\
    --checkpoint checkpoints/vit_depth_transformer_pretrainedTrue/best.pth
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _collect_paths(data_root: str):
    img_dir = os.path.join(data_root, "train")
    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*_rgb.png")) +
        glob.glob(os.path.join(img_dir, "*_rgb.jpg"))
    )
    depth_paths = [
        os.path.join(img_dir, os.path.splitext(os.path.basename(p))[0].replace("_rgb", "_depth") + ".npy")
        for p in image_paths
    ]
    return image_paths, depth_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["patch_sim", "gt_depth", "learned"])
    parser.add_argument("--data_root",   default="data/")
    parser.add_argument("--save_path",   required=True)
    parser.add_argument("--encoder",     default="vit_base_patch16_224",
                        help="ViT encoder for patch_sim (e.g. vit_large_patch16_224)")
    parser.add_argument("--img_size",    type=int, default=560)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--k",           type=int, default=50, help="neighbours stored per node")
    parser.add_argument("--checkpoint",  default=None,
                        help="path to DepthModel .pth for the learned strategy")
    args = parser.parse_args()

    image_paths, depth_paths = _collect_paths(args.data_root)
    print(f"Found {len(image_paths)} training images")

    if args.strategy == "patch_sim":
        from src.retrieval.patch_sim_retriever import PatchSimRetriever
        PatchSimRetriever.build_index(
            image_paths, args.save_path,
            encoder_name=args.encoder,
            img_size=args.img_size,
            device=args.device,
        )

    elif args.strategy == "gt_depth":
        from src.retrieval.graph_retriever import GraphRetriever
        GraphRetriever.build_gt_graph(depth_paths, args.save_path, k=args.k)

    elif args.strategy == "learned":
        if not args.checkpoint:
            parser.error("--checkpoint required for the learned strategy")
        from src.retrieval.graph_retriever import GraphRetriever
        GraphRetriever.build_learned_graph(
            image_paths, args.save_path,
            checkpoint_path=args.checkpoint,
            img_size=args.img_size,
            device=args.device,
            k=args.k,
        )


if __name__ == "__main__":
    main()
