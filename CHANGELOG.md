# Changelog

All notable changes to vxdb will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Dockerfile for server deployment (~10 MB image)
- Jupyter notebook examples for OpenAI, Sentence Transformers, LangChain, Cohere, and hybrid search

### Supported Platforms

- macOS (arm64, x86_64)
- Linux (x86_64, aarch64)
- Windows (x86_64)
- Python 3.9 - 3.13
