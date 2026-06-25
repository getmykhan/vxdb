"""Pluggable embedding interface for vxdb."""

from abc import ABC, abstractmethod


class EmbeddingFunction(ABC):
    """Base class for embedding functions.

    Subclass this and implement `embed()` to provide custom embeddings.

    Example::

        class MyEmbedder(EmbeddingFunction):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

        embedder = MyEmbedder()
        docs = ["hello"]
        collection.upsert(ids=["a"], vectors=embedder.embed(docs), documents=docs)
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text strings into embedding vectors."""
        ...
