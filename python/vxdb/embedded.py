"""Embedded-mode ``Database`` / ``Collection`` with optional auto-embedding.

Thin Python wrappers over the native ``_vxdb`` engine. When a collection is
created with an ``embedding_function``, you may pass raw ``documents`` to
``upsert`` and text to ``query`` / ``hybrid_query`` and the vectors are computed
for you. With no embedding function, every call behaves exactly like the native
classes (the native ``vector`` / ``vectors`` argument stays first and positional,
so existing code is unaffected).

The ``embedding_function`` may be an :class:`vxdb.EmbeddingFunction` instance or
any plain callable ``list[str] -> list[list[float]]``.
"""

from vxdb._vxdb import Database as _NativeDatabase


def _embed(embedding_function, texts):
    if embedding_function is None:
        raise ValueError(
            "no embedding_function is set on this collection; either pass "
            "`vectors=` explicitly or create the collection with an "
            "`embedding_function`"
        )
    if hasattr(embedding_function, "embed"):
        return embedding_function.embed(texts)
    if callable(embedding_function):
        return embedding_function(texts)
    raise TypeError("embedding_function must be an EmbeddingFunction (with .embed) or a callable")


class Collection:
    """A collection with optional automatic embedding.

    Wraps the native collection; delegates everything, adding text-in
    convenience when an ``embedding_function`` is present.
    """

    def __init__(self, native, name, embedding_function=None):
        self._native = native
        self.name = name
        self.embedding_function = embedding_function

    def upsert(self, ids, vectors=None, metadata=None, documents=None):
        if vectors is None:
            if documents is None:
                raise ValueError(
                    "upsert requires `vectors`, or `documents` together with an embedding_function on the collection"
                )
            vectors = _embed(self.embedding_function, documents)
        self._native.upsert(ids=ids, vectors=vectors, metadata=metadata, documents=documents)

    def query(self, vector=None, top_k=10, filter=None, query_text=None, ef_search=None):
        if vector is None:
            if query_text is None:
                raise ValueError(
                    "query requires `vector`, or `query_text` together with an embedding_function on the collection"
                )
            vector = _embed(self.embedding_function, [query_text])[0]
        return self._native.query(vector=vector, top_k=top_k, filter=filter, ef_search=ef_search)

    def hybrid_query(self, vector=None, query=None, top_k=10, alpha=0.5):
        # `vector` stays first to match the native positional order
        # `hybrid_query(vector, query, ...)`. `query` (the BM25 text) is required;
        # `vector` is embedded from it when omitted.
        if query is None:
            raise ValueError("hybrid_query requires `query` text (the keyword/BM25 component)")
        if vector is None:
            vector = _embed(self.embedding_function, [query])[0]
        return self._native.hybrid_query(vector=vector, query=query, top_k=top_k, alpha=alpha)

    def keyword_search(self, query, top_k=10):
        return self._native.keyword_search(query=query, top_k=top_k)

    def delete(self, ids):
        return self._native.delete(ids)

    def count(self):
        return self._native.count()

    def __repr__(self):
        return repr(self._native)


class Database:
    """An embedded database. Drop-in for the native ``Database`` plus
    per-collection ``embedding_function`` support."""

    def __init__(self, path=None):
        self._native = _NativeDatabase(path)
        self._embedders = {}

    def create_collection(
        self,
        name,
        dimension=None,
        metric="cosine",
        index="flat",
        embedding_function=None,
    ):
        if dimension is None:
            if embedding_function is None:
                raise ValueError("create_collection requires `dimension`, or an `embedding_function` to infer it from")
            # Infer the dimension once from a probe embedding.
            dimension = len(_embed(embedding_function, ["dimension probe"])[0])
        native = self._native.create_collection(name, dimension, metric, index)
        if embedding_function is not None:
            self._embedders[name] = embedding_function
        return Collection(native, name, embedding_function)

    def get_collection(self, name):
        native = self._native.get_collection(name)
        return Collection(native, name, self._embedders.get(name))

    def list_collections(self):
        return self._native.list_collections()

    def delete_collection(self, name):
        self._embedders.pop(name, None)
        return self._native.delete_collection(name)

    def __repr__(self):
        return repr(self._native)
