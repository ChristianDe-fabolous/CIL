import random
from typing import List

from .base import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Path 2 — draw K random images from the context pool, excluding self."""

    def __init__(self, pool_size: int, seed: int = 42):
        self.pool_size = pool_size
        self.rng = random.Random(seed)

    def retrieve(self, query_idx: int, k: int) -> List[int]:
        pool = [i for i in range(self.pool_size) if i != query_idx]
        return self.rng.sample(pool, min(k, len(pool)))
