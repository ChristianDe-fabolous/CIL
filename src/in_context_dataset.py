import random
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.retrieval.base import BaseRetriever

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class InContextDataset(Dataset):
    """
    Returns (query_img, query_depth, ctx_imgs, ctx_depths).

    query_img   : (3, H, W)
    query_depth : (1, H, W)
    ctx_imgs    : (K, 3, H, W)
    ctx_depths  : (K, 1, H, W)

    Context images are always drawn from ctx_image_paths / ctx_depth_paths
    (typically the training pool) via the retriever, which maps query_idx →
    a list of indices into that pool.  This lets the val split use a
    query set that is disjoint from the context pool.
    """

    def __init__(
        self,
        query_image_paths: List[str],
        query_depth_paths: List[str],
        ctx_image_paths: List[str],
        ctx_depth_paths: List[str],
        retriever: BaseRetriever,
        num_context: int = 4,
        img_size: int = 560,
        augment: bool = False,
    ):
        self.query_image_paths = query_image_paths
        self.query_depth_paths = query_depth_paths
        self.ctx_image_paths   = ctx_image_paths
        self.ctx_depth_paths   = ctx_depth_paths
        self.retriever   = retriever
        self.num_context = num_context
        self.img_size    = img_size
        self.augment     = augment
        self.normalize   = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.query_image_paths)

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)  # (3, H, W), not yet normalized

    def _load_depth(self, path: str) -> torch.Tensor:
        d = np.load(path).astype(np.float32)
        if d.shape[0] != self.img_size or d.shape[1] != self.img_size:
            d = np.array(Image.fromarray(d).resize((self.img_size, self.img_size), Image.NEAREST))
        return torch.from_numpy(d).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int):
        q_img   = self._load_img(self.query_image_paths[idx])
        q_depth = self._load_depth(self.query_depth_paths[idx])

        if self.augment and random.random() > 0.5:
            q_img   = TF.hflip(q_img)
            q_depth = torch.from_numpy(np.fliplr(q_depth.numpy()).copy())

        q_img = self.normalize(q_img)

        ctx_indices = self.retriever.retrieve(idx, self.num_context)
        ctx_imgs    = torch.stack([
            self.normalize(self._load_img(self.ctx_image_paths[i])) for i in ctx_indices
        ])
        ctx_depths = torch.stack([
            self._load_depth(self.ctx_depth_paths[i]) for i in ctx_indices
        ])

        return q_img, q_depth, ctx_imgs, ctx_depths
