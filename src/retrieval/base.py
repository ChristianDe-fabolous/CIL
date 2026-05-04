from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_idx: int, k: int) -> List[int]:
        """Return k indices into the context pool for the given query index."""
        ...
