import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class DepthDataset(Dataset):
    def __init__(self, image_dir: str, depth_dir: str, img_size: int = 560, augment: bool = False):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.png")) +
            glob.glob(os.path.join(image_dir, "*.jpg"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.depth_dir = depth_dir
        self.img_size = img_size
        self.augment = augment
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def _depth_path(self, image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        stem = stem.replace("_rgb", "_depth")
        return os.path.join(self.depth_dir, stem + ".npy")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        depth = np.load(self._depth_path(img_path)).astype(np.float32)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        depth_h, depth_w = depth.shape[:2]
        if depth_h != self.img_size or depth_w != self.img_size:
            depth = np.array(
                Image.fromarray(depth).resize((self.img_size, self.img_size), Image.NEAREST)
            )

        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            depth = np.fliplr(depth).copy()

        image = TF.to_tensor(image)
        image = self.normalize(image)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        return image, depth


class TestDataset(Dataset):
    def __init__(self, image_dir: str, img_size: int = 560):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.png")) +
            glob.glob(os.path.join(image_dir, "*.jpg"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.img_size = img_size
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = TF.to_tensor(image)
        image = self.normalize(image)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return image, stem
