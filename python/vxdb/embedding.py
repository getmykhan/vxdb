"""Pluggable embedding interface for vxdb."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingFunction(ABC):
    """Base class for embedding functions.

    Subclass this and implement `embed()` to provide custom embeddings.

    Example::

        class MyEmbedder(EmbeddingFunction):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

        collection.upsert(
            ids=["a"],
            documents=["hello"],
            embedding_fn=my_embedder,
        )
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of text strings into embedding vectors."""
        ...
