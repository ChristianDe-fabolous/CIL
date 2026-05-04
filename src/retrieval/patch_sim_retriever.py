import os
from typing import List

import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from .base import BaseRetriever

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class PatchSimRetriever(BaseRetriever):
    """
    Path 1 — retrieve by cosine similarity of mean-pooled patch embeddings
    from a large pretrained ViT (e.g. vit_base_patch16_224 or vit_large_patch16_224).

    Build the index once with build_index(), then instantiate from the saved file.

    Usage:
        PatchSimRetriever.build_index(image_paths, save_path, encoder_name, device)
        retriever = PatchSimRetriever(save_path)
    """

    def __init__(self, index_path: str):
        data = torch.load(index_path, map_location="cpu")
        feats = F.normalize(data["feats"].float(), dim=-1)  # (N, D)
        # Precompute full similarity matrix — fast at retrieval time.
        # For large N (>50k) consider approximate nearest neighbours instead.
        self.sim = feats @ feats.T  # (N, N)

    def retrieve(self, query_idx: int, k: int) -> List[int]:
        sims = self.sim[query_idx].clone()
        sims[query_idx] = -2.0  # exclude self
        return torch.topk(sims, k).indices.tolist()

    @classmethod
    def build_index(
        cls,
        image_paths: List[str],
        save_path: str,
        encoder_name: str = "vit_base_patch16_224",
        img_size: int = 560,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        """
        Encode all images with a large ViT, mean-pool patch tokens, save to disk.
        The encoder is only used here — it is NOT the training encoder.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        model = timm.create_model(
            encoder_name, pretrained=True, img_size=img_size, num_classes=0
        ).to(device).eval()

        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        all_feats: List[torch.Tensor] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="patch-sim index"):
                batch = [transform(Image.open(p).convert("RGB")) for p in image_paths[i:i + batch_size]]
                imgs  = torch.stack(batch).to(device)
                # forward_features returns (B, 1+N, D); drop CLS then mean-pool
                tokens = model.forward_features(imgs)[:, 1:, :]
                all_feats.append(tokens.mean(dim=1).cpu())

        feats = torch.cat(all_feats, dim=0)  # (N_train, D)
        torch.save({"feats": feats, "image_paths": image_paths, "encoder": encoder_name}, save_path)
        print(f"Patch-sim index saved: {feats.shape} → {save_path}")
