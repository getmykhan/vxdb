# Changelog

All notable changes to vxdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-06

### Added

- **`vxdb.agent`: ephemeral working memory for agents.** `WorkingMemory` (and
  the `scratch()` allocator) is an in-process semantic store an agent can
  consult on every step of its loop: `add`/`add_many` to write, `recall` for
  top-k reads, `match`/`seen` as a dedup and loop guard, context-manager
  lifecycle, and live per-op timing via `timing_summary()`. Store operations
  measure in the ~100 microsecond range, so in-loop memory costs nothing next
  to the LLM and embedder calls. Pure Python over the embedded `Database()`;
  no Rust changes.
- `examples/agent_working_memory.ipynb`: a with/without-memory A/B in notebook
  form, plus the system prompt and `remember`/`recall` tool wiring for a real
  OpenAI Agents SDK agent.
- `tests/test_agent_memory.py`: deterministic, offline coverage for the new
  module.

### Fixed

- `WorkingMemory` similarity was inverted for `metric="dot"`: the engine
  reports `-dot_product` as the distance, and the passthrough mapping returned
  that negated value, so `match`/`seen` thresholds could never fire. Recall
  ordering was unaffected. Similarity now negates the score back to the raw
  dot product.
- `import vxdb` no longer imports `httpx` eagerly. With httpx's CLI extras
  installed, the old module-level import dragged in rich and pygments (~100 ms
  and an `importlib.metadata` load, defeating 0.3.3's lazy-import work). The
  import now happens inside `Client()`, which is the only place it is needed.

### Changed

- README hero is dual-track: use vxdb as a database, or as agent working
  memory.

## [0.3.3] - 2026-06-25

### Changed

- `__version__` now resolves lazily on first access via `importlib.metadata`,
  keeping `import vxdb` startup under 10 ms. The value still single-sources
  from package metadata, so it cannot drift from pyproject/Cargo.

## [0.3.2] - 2026-06-25

### Added

- **`vxdb-server` is now a pip-installable package.** The standalone HTTP server
  ships as a separate, optional wheel built from the same workspace (maturin
  "bin" bindings), so `pip install vxdb-server` puts the `vxdb-server` binary on
  your `PATH` — making the README's "Server Mode" section true for pip users for
  the first time. The core `vxdb` wheel is unchanged (still dependency-free,
  ~1.4 MB); the server wheel (~2.6 MB) is only installed by those who want it.
  Versions stay in lockstep via the shared workspace version.

### Changed

- `tests/test_server_client.py` now discovers the `vxdb-server` binary on `PATH`
  (i.e. an installed wheel) before falling back to the cargo target dir, so the
  suite validates the shipped artifact.
- README "Server Mode" documents `pip install vxdb-server` and notes the server
  is in-memory only.

## [0.3.1] - 2026-06-24

### Fixed

- `Collection.hybrid_query` positional argument order regressed in 0.3.0: the
  wrapper declared `hybrid_query(query, vector=...)` instead of the native
  `hybrid_query(vector, query, ...)`, so positional calls like
  `hybrid_query(vec, "text")` raised `TypeError`. Restored the native order
  (`query` is still required; `vector` is embedded from it when omitted) and
  added a positional-call regression test.

## [0.3.0] - 2026-06-24

### Added

- Automatic embedding: create a collection with an `embedding_function` and pass
  raw `documents` to `upsert` / text via `query_text=` (and `hybrid_query`) — the
  vectors are computed for you. The embedding function may be an
  `EmbeddingFunction` instance or any callable `list[str] -> list[list[float]]`,
  and the collection dimension is inferred from it when omitted. Implemented as a
  thin Python wrapper over the native engine — no new dependencies, and fully
  backward compatible (passing `vectors`/`vector` behaves exactly as before).

### Fixed

- `upsert` no longer panics on a zero-width (`(n, 0)`) NumPy array; it now raises
  a clean `ValueError`.

### Changed

- CI now installs NumPy and runs the full `tests/` suite (previously only
  `test_embedded.py` ran, and the NumPy buffer-ingest path was untested).
- Added direct unit tests for the SIMD distance kernels across a range of
  dimensions (the chunked path was only covered indirectly before).

## [0.2.1] - 2026-06-24

### Fixed

- `vxdb.__version__` now derives from installed package metadata via
  `importlib.metadata` instead of a hardcoded string. It previously reported
  `0.1.0` even on the `0.2.0` release; it can no longer drift from the package
  version.
- `EmbeddingFunction` is now exported from the top-level `vxdb` package
  (`from vxdb import EmbeddingFunction`). Its docstring also showed a
  non-existent `collection.upsert(..., embedding_fn=...)` call that raised
  `TypeError`; it now shows the correct workflow — call `embed()` yourself and
  pass the result as `vectors`.

## [0.2.0] - 2026-06-24

### Added

- Zero-copy NumPy ingest: `Collection.upsert(vectors=...)` now accepts a 2-D
  `float32` NumPy array (or any object exposing the Python buffer protocol —
  torch, jax, `array.array`) and reads its memory directly. NumPy is never
  imported or required, so the zero-dependency guarantee is unchanged. Roughly
  halves peak memory during ingest.

### Changed

- HNSW distance computation rewritten for auto-vectorization: the
  `DistanceMetric` trait object was replaced with a monomorphized `Metric`
  enum, and the L2/cosine/dot kernels use multiple accumulators so the compiler
  emits SIMD (NEON on arm64, SSE on x86-64). ~2x faster HNSW build and ~2x
  faster queries — no new dependencies, no `unsafe`.
- HNSW search reuses a version-stamped visited buffer instead of allocating a
  `HashSet` per layer, further reducing build time.
- Default `ef_search` raised from 50 to 150, materially improving recall on
  high-dimensional data in exchange for a small amount of query latency.

### Removed

- Dropped support for Python 3.9 and 3.10 (both end-of-life). vxdb now requires
  **Python 3.11+** and ships an `abi3-py311` wheel. This is what enables the
  buffer-protocol NumPy ingest, which the older limited ABI does not expose.

## [0.1.0] - 2026-04-07

### Added

- Rust core engine with HNSW and flat (exact) vector indexes
- Distance metrics: cosine, euclidean, dot product
- PyO3 Python bindings with zero-copy in-process execution
- Persistent storage: mmap vectors, SQLite metadata, write-ahead log (WAL)
- In-memory (ephemeral) mode for prototyping
- Metadata filtering with 10 operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or)
- Built-in BM25 keyword search index
- Hybrid search (vector + BM25) fused via Reciprocal Rank Fusion with tunable alpha
- Standalone HTTP server (Axum) with REST API
- Python HTTP client for remote server access
- Pluggable embedding interface (`EmbeddingFunction` base class)
- Dockerfile for server deployment (~145 MB Debian-based image)
- Jupyter notebook examples for OpenAI, Sentence Transformers, LangChain, Cohere, and hybrid search

### Supported Platforms

- macOS (arm64, x86_64)
- Linux (x86_64, aarch64)
- Windows (x86_64)
- Python 3.9 - 3.13
