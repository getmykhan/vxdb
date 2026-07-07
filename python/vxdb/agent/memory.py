"""Ephemeral, in-process semantic working memory for agents.

The defining idea: this is a vector store you *allocate* like a data structure, not
one you *connect to* like a database. Because it lives in-process and creates/destroys
instantly, an agent can consult it on **every step** of its loop — something that is
prohibitively slow and costly against a networked store.

    from vxdb.agent import scratch

    wm = scratch(embed)                       # in-memory, instant
    wm.add("checkout depends on tax-service", metadata={"hop": 1})
    hits = wm.recall("what does checkout use for tax", k=3)   # working-memory recall
    dup = wm.match("checkout tax dependency", threshold=0.85) # dedup / loop guard
    wm.close()                                # instant destroy

``WorkingMemory`` is also a context manager. It is built entirely on the embedded
``vxdb.Database`` (zero Rust changes); the only net-new behavior over the raw API is
auto-embedding, namespacing, and — importantly — **normalizing the engine's distance
into an intuitive similarity** (the raw ``query`` returns ``score`` as a *distance*,
where smaller is closer; here ``similarity`` is higher-is-better).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from vxdb import Database


def _summary(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0}
    s = sorted(samples)
    return {
        "count": len(s),
        "p50_us": s[len(s) // 2],
        "p99_us": s[min(len(s) - 1, int(len(s) * 0.99))],
        "mean_us": sum(s) / len(s),
        "max_us": s[-1],
    }


# An embedder is either an EmbeddingFunction (has ``.embed``) or a plain callable.
Embedder = Callable[[list[str]], list[list[float]]] | Any


class Recall(dict):
    """A recall hit. A dict with convenience attributes: ``text``, ``similarity``,
    ``metadata``, ``id`` (so it prints nicely and is JSON-friendly)."""

    @property
    def text(self) -> str | None:
        return self.get("text")

    @property
    def similarity(self) -> float:
        return self["similarity"]

    @property
    def metadata(self) -> dict:
        return self.get("metadata") or {}

    @property
    def id(self) -> str:
        return self["id"]


class WorkingMemory:
    """An ephemeral, in-process semantic store scoped to a single agent run.

    Allocate it, add to it, recall/match against it, drop it. No disk, no network.
    """

    def __init__(
        self,
        embed: Embedder,
        *,
        metric: str = "cosine",
        index: str = "flat",
        name: str | None = None,
    ) -> None:
        self._embed = embed
        self._metric = metric
        self._seq = 0
        self._closed = False
        # Per-op latency samples (microseconds). The store op is the in-process vxdb cost;
        # embed is the (often networked) embedder cost, tracked separately so callers can
        # see exactly how fast the vxdb operation itself returns. Set before the dim probe.
        self.timing: dict[str, list[float]] = {"embed_us": [], "write_us": [], "query_us": []}
        self._db = Database()  # path=None -> in-memory; instant create
        self._name = name or f"wm-{uuid.uuid4().hex[:8]}"
        dim = len(self._embed_one("dimension probe"))
        self._col = self._db.create_collection(self._name, dimension=dim, metric=metric, index=index)

    # -- embedding helpers -------------------------------------------------
    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        fn = self._embed
        t = perf_counter()
        vecs = fn.embed(list(texts)) if hasattr(fn, "embed") else fn(list(texts))
        self.timing["embed_us"].append((perf_counter() - t) * 1e6)
        # Accept numpy arrays transparently.
        return [list(v) for v in vecs]

    def _embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def timing_summary(self) -> dict:
        """Latency summary (us) for embed vs the vxdb store ops, measured live."""
        return {k: _summary(v) for k, v in self.timing.items()}

    def _similarity(self, score: float) -> float:
        """Map the engine's distance ``score`` to higher-is-better similarity."""
        if self._metric == "cosine":
            return 1.0 - score  # cosine distance = 1 - cos_sim
        if self._metric == "dot":
            return -score  # engine stores -dot_product as the distance
        return -score  # euclidean: smaller distance is better

    def _fmt(self, hit: dict) -> Recall:
        return Recall(
            id=hit["id"],
            text=hit.get("document"),
            similarity=self._similarity(hit["score"]),
            metadata=hit.get("metadata") or {},
        )

    # -- writes ------------------------------------------------------------
    def add(
        self,
        text: str,
        metadata: dict | None = None,
        *,
        id: str | None = None,
    ) -> str:
        """Embed and store ``text``. Returns its id (auto-assigned if not given)."""
        self._guard()
        vid = id or str(self._seq)
        vec = self._embed_one(text)
        t = perf_counter()
        self._col.upsert(
            ids=[vid],
            vectors=[vec],
            metadata=[metadata or {}],
            documents=[text],
        )
        self.timing["write_us"].append((perf_counter() - t) * 1e6)
        self._seq += 1
        return vid

    def add_many(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        *,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Batch ``add`` (one embedding call for the whole batch)."""
        self._guard()
        texts = list(texts)
        if ids is None:
            ids = [str(self._seq + i) for i in range(len(texts))]
        self._col.upsert(
            ids=list(ids),
            vectors=self._embed_batch(texts),
            metadata=list(metadatas) if metadatas else [{} for _ in texts],
            documents=texts,
        )
        self._seq += len(texts)
        return list(ids)

    # -- reads -------------------------------------------------------------
    def recall(self, query: str, k: int = 5, *, where: dict | None = None) -> list[Recall]:
        """Working-memory recall: the top-``k`` most relevant items (best first)."""
        self._guard()
        vec = self._embed_one(query)
        t = perf_counter()
        hits = self._col.query(vector=vec, top_k=k, filter=where)
        self.timing["query_us"].append((perf_counter() - t) * 1e6)
        return [self._fmt(h) for h in hits]

    def match(self, text: str, threshold: float = 0.85, *, where: dict | None = None) -> Recall | None:
        """Dedup / loop guard: the closest item if its similarity ≥ ``threshold``."""
        self._guard()
        vec = self._embed_one(text)
        t = perf_counter()
        hits = self._col.query(vector=vec, top_k=1, filter=where)
        self.timing["query_us"].append((perf_counter() - t) * 1e6)
        if hits:
            hit = self._fmt(hits[0])
            if hit["similarity"] >= threshold:
                return hit
        return None

    def seen(self, text: str, threshold: float = 0.85) -> bool:
        """Boolean sugar over :meth:`match` for the ``have I done this?`` case."""
        return self.match(text, threshold) is not None

    # -- lifecycle ---------------------------------------------------------
    def __len__(self) -> int:
        return 0 if self._closed else self._col.count()

    def close(self) -> None:
        """Drop the underlying collection. Idempotent."""
        if not self._closed:
            try:
                self._db.delete_collection(self._name)
            finally:
                self._closed = True

    def _guard(self) -> None:
        if self._closed:
            raise RuntimeError("WorkingMemory is closed")

    def __enter__(self) -> WorkingMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self)} items"
        return f"WorkingMemory(name={self._name!r}, {state})"


def scratch(embed: Embedder, **kwargs: Any) -> WorkingMemory:
    """Allocate a fresh ephemeral :class:`WorkingMemory`. Sugar for the constructor."""
    return WorkingMemory(embed, **kwargs)
