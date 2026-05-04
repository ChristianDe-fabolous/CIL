import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .base import BaseRetriever


class GraphRetriever(BaseRetriever):
    """
    Path 3 — retrieve from a precomputed KNN graph.

    Two build modes (both stored in the same format):

    gt_depth_sim  — cosine similarity of depth log-histograms.
                    "Naive" ground-truth matching: images with similar depth
                    distributions (similar scene geometry) are good context.

    learned        — cosine similarity of task-specific encoder features
                    extracted from a fine-tuned DepthModel checkpoint.
                    Captures what the model has learned about scene structure
                    and surfaces pairs that were implicitly useful during training.

    Usage:
        GraphRetriever.build_gt_graph(depth_paths, save_path)
        GraphRetriever.build_learned_graph(image_paths, save_path, checkpoint_path)
        retriever = GraphRetriever(graph_path)
    """

    def __init__(self, graph_path: str):
        data = torch.load(graph_path, map_location="cpu")
        # graph[i] = list of pool indices sorted best-first
        self.graph: List[List[int]] = data["graph"]

    def retrieve(self, query_idx: int, k: int) -> List[int]:
        neighbors = [n for n in self.graph[query_idx] if n != query_idx]
        return neighbors[:k]

    # ------------------------------------------------------------------
    # Build: ground-truth depth histogram similarity
    # ------------------------------------------------------------------
    @classmethod
    def build_gt_graph(
        cls,
        depth_paths: List[str],
        save_path: str,
        n_bins: int = 64,
        k: int = 50,
    ) -> None:
        """
        Compute a depth log-histogram for every training image and build a KNN
        graph by cosine similarity. Histograms capture the global depth distribution
        (scene type) — near-zero distance means similar object layout and scale.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        histograms = []
        for p in tqdm(depth_paths, desc="gt-depth histograms"):
            d = np.load(p).astype(np.float32).ravel()
            d = d[d > 1e-6]
            d = np.log(d + 1e-6)
            hist, _ = np.histogram(d, bins=n_bins, density=True)
            histograms.append(hist.astype(np.float32))

        H   = F.normalize(torch.tensor(np.stack(histograms)), dim=-1)  # (N, n_bins)
        sim = H @ H.T                                                    # (N, N)
        N   = sim.shape[0]
        k   = min(k, N - 1)

        graph = []
        for i in range(N):
            s    = sim[i].clone()
            s[i] = -2.0
            graph.append(torch.topk(s, k).indices.tolist())

        torch.save({"graph": graph, "depth_paths": depth_paths, "mode": "gt_depth_sim"}, save_path)
        print(f"GT-depth graph saved: {N} nodes, k={k} → {save_path}")

    # ------------------------------------------------------------------
    # Build: task-encoder feature similarity (learned graph)
    # ------------------------------------------------------------------
    @classmethod
    def build_learned_graph(
        cls,
        image_paths: List[str],
        save_path: str,
        checkpoint_path: str,
        img_size: int = 560,
        device: str = "cuda",
        batch_size: int = 8,
        k: int = 50,
    ) -> None:
        """
        Extract mean-pooled encoder features from a fine-tuned DepthModel checkpoint
        and build a KNN graph by cosine similarity.  Unlike patch_sim (generic ViT),
        these features encode what the task encoder considers structurally similar —
        pairs that are close here were implicitly beneficial during training.
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        import torchvision.transforms as T
        from src.models.model import DepthModel

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        ckpt  = torch.load(checkpoint_path, map_location="cpu")
        cfg   = ckpt["cfg"]
        model = DepthModel(
            decoder_type=cfg["model"]["decoder_type"],
            pretrained=False,
            img_size=img_size,
            patch_size=cfg["model"]["patch_size"],
            embed_dim=cfg["model"]["embed_dim"],
            num_heads=cfg["model"]["num_heads"],
        )
        model.load_state_dict(ckpt["model"])
        encoder = model.encoder.to(device).eval()

        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="task-encoder features"):
                batch = [transform(Image.open(p).convert("RGB")) for p in image_paths[i:i + batch_size]]
                imgs  = torch.stack(batch).to(device)
                feats = encoder(imgs).mean(dim=1).cpu()  # mean-pool patches
                all_feats.append(feats)

        feats = F.normalize(torch.cat(all_feats, dim=0), dim=-1)  # (N, D)
        sim   = feats @ feats.T
        N     = sim.shape[0]
        k     = min(k, N - 1)

        graph = []
        for i in range(N):
            s    = sim[i].clone()
            s[i] = -2.0
            graph.append(torch.topk(s, k).indices.tolist())

        torch.save({"graph": graph, "image_paths": image_paths, "mode": "learned",
                    "checkpoint": checkpoint_path}, save_path)
        print(f"Learned graph saved: {N} nodes, k={k} → {save_path}")
