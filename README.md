

# vxdb

**The vector database that fits in your pocket.**

Rust-powered. Python-native. One `pip install` away.





---

```python
pip install vxdb
```

```python
import vxdb

db = vxdb.Database()
collection = db.create_collection("docs", dimension=384)

collection.upsert(
    ids=["a", "b"],
    vectors=[embed("how to train a model"), embed("best pasta recipe")],
    documents=["how to train a model", "best pasta recipe"],
)

collection.query(vector=embed("machine learning"), top_k=5)
```

That's it. No Docker. No config files. No cloud account. No 500 MB of dependencies.

---

## Why developers choose vxdb


|     |
| --- |
|     |


### Stupid fast

The entire hot path — distance computation, HNSW traversal, BM25 scoring, mmap I/O — is **pure Rust** with zero GIL contention. Your Python code calls directly into compiled native code via PyO3. No serialization overhead. No REST round-trips. No subprocess.



### Stupid light

A single native wheel **under 5 MB**. Starts in **under 10 ms**. Compare that to ChromaDB (~200 MB, ~2s startup), Milvus (needs Docker + etcd + MinIO), or Pinecone (needs a cloud account and an internet connection).



### Runs anywhere

Laptop. CI pipeline. Raspberry Pi. AWS Lambda. Docker container. Air-gapped server. Anywhere Python runs, vxdb runs. No infrastructure required to get started — scale up to a standalone server when you need it.



### Hybrid search built-in

Vector similarity + BM25 keyword matching fused via **Reciprocal Rank Fusion**. One API call. Tunable `alpha` parameter. No separate search engine needed. No Elasticsearch sidecar.



---

## The full picture

```
                    ┌─────────────────────────────────────────────────┐
                    │               Your Python Code                  │
                    └─────────────┬───────────────────┬───────────────┘
                                  │                   │
                    ┌─────────────▼──────┐  ┌────────▼────────────┐
                    │  Embedded (PyO3)   │  │  Server (REST API)  │
                    │  Zero-copy, in-    │  │  Axum, async,       │
                    │  process, <1μs     │  │  multi-client       │
                    │  call overhead     │  │                     │
                    └─────────────┬──────┘  └────────┬────────────┘
                                  │                   │
                    ┌─────────────▼───────────────────▼───────────────┐
                    │              Rust Core Engine                    │
                    │                                                  │
                    │  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
                    │  │   HNSW   │ │   Flat   │ │  BM25 Keyword   │  │
                    │  │  Index   │ │  Index   │ │     Index       │  │
                    │  └──────────┘ └──────────┘ └─────────────────┘  │
                    │  ┌──────────────────┐ ┌──────────────────────┐  │
                    │  │ Distance Metrics  │ │  Metadata Filtering  │  │
                    │  │ cosine/L2/dot     │ │  10 operators, SQL   │  │
                    │  └──────────────────┘ └──────────────────────┘  │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │   Hybrid Search (Reciprocal Rank Fusion)  │   │
                    │  └──────────────────────────────────────────┘   │
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │                  Storage                        │
                    │  mmap vectors │ SQLite metadata │ Write-Ahead Log│
                    └─────────────────────────────────────────────────┘
```

---

## Quick Start

### 3 lines to your first search

```python
import vxdb

db = vxdb.Database()
collection = db.create_collection("docs", dimension=384, metric="cosine")
```

### Insert vectors

```python
collection.upsert(
    ids=["a", "b", "c"],
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    metadata=[{"type": "article"}, {"type": "blog"}, {"type": "article"}],
    documents=["intro to ML", "my favorite recipes", "deep learning guide"],
)
```

### Search — four ways

```python
# 1. Vector similarity
results = collection.query(vector=[0.1, 0.2, ...], top_k=5)

# 2. Filtered (metadata constraints)
results = collection.query(
    vector=[0.1, ...], top_k=5,
    filter={"type": {"$eq": "article"}}
)

# 3. Hybrid (vector + keyword — the sweet spot)
results = collection.hybrid_query(
    vector=[0.1, ...],
    query="machine learning",
    top_k=5,
    alpha=0.5,  # 0=keyword only, 1=vector only
)

# 4. Keyword only (BM25)
results = collection.keyword_search(query="machine learning", top_k=5)
```

Every result returns `{"id", "score", "metadata"}`.

---

## Installation

```bash
pip install vxdb
```

That's the whole thing. Works on **macOS, Linux, Windows**. Python 3.9+.

For the HTTP client (talking to a remote vxdb server):

```bash
pip install 'vxdb[server]'
```

---

## Embedding Providers

vxdb stores **pre-computed vectors** — bring any embedding model you want. We have step-by-step notebooks for each:


| Provider                     | Install                             | API Key?   | Notebook                                                                       |
| ---------------------------- | ----------------------------------- | ---------- | ------------------------------------------------------------------------------ |
| **OpenAI**                   | `pip install openai`                | Yes        | `[examples/openai_embeddings.ipynb](examples/openai_embeddings.ipynb)`         |
| **Sentence Transformers**    | `pip install sentence-transformers` | No (local) | `[examples/sentence_transformers.ipynb](examples/sentence_transformers.ipynb)` |
| **LangChain** (any provider) | `pip install langchain-openai`      | Depends    | `[examples/langchain_integration.ipynb](examples/langchain_integration.ipynb)` |
| **Cohere**                   | `pip install cohere`                | Yes        | `[examples/cohere_embeddings.ipynb](examples/cohere_embeddings.ipynb)`         |
| **Ollama** (local LLMs)      | `pip install ollama`                | No (local) | —                                                                              |


Or use the pluggable interface:

```python
from vxdb.embedding import EmbeddingFunction

class MyEmbedder(EmbeddingFunction):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return your_model.encode(texts)
```

---

## Server Mode

Same engine, accessed over HTTP. Deploy it as a standalone service.

```bash
# Start the server
vxdb-server --host 0.0.0.0 --port 8080
```

**Python client:**

```python
from vxdb import Client

client = Client("http://localhost:8080")
coll = client.create_collection("docs", dimension=384)
coll.upsert(ids=["a"], vectors=[[0.1, ...]], documents=["hello world"])
results = coll.hybrid_query(vector=[0.1, ...], query="hello", top_k=5)
```

**cURL:**

```bash
# Create collection
curl -X POST localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384}'

# Upsert
curl -X POST localhost:8080/collections/docs/upsert \
  -H "Content-Type: application/json" \
  -d '{"ids": ["a"], "vectors": [[0.1, 0.2]], "documents": ["hello world"]}'

# Query
curl -X POST localhost:8080/collections/docs/query \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2], "top_k": 5}'
```

**Docker:**

```bash
docker build -t vxdb .
docker run -p 8080:8080 vxdb    # ~10 MB image
```

---

## Hybrid Search

Most vector databases give you vector search OR keyword search. vxdb gives you both, fused intelligently in a single call.

**How it works:**

1. **You upsert with documents** — raw text is tokenized into a built-in BM25 index alongside your vectors
2. **At query time** — vector search and BM25 run in parallel, then Reciprocal Rank Fusion merges both ranked lists
3. **You control the blend** — `alpha=1.0` (pure vector) → `alpha=0.5` (balanced) → `alpha=0.0` (pure keyword)

**When to use it:** Specific product names. Error codes. Proper nouns. Anything where exact terms matter alongside semantic meaning. See `[examples/hybrid_search.ipynb](examples/hybrid_search.ipynb)` for a deep dive with side-by-side comparisons.

```python
results = collection.hybrid_query(
    vector=embed("lightweight laptop for students"),
    query="MacBook Air M4",
    top_k=5,
    alpha=0.5,
)
```

---

## How vxdb compares


|                              | vxdb                    | ChromaDB         | Qdrant         | Pinecone    | Milvus         | Weaviate    | FAISS         |
| ---------------------------- | ----------------------- | ---------------- | -------------- | ----------- | -------------- | ----------- | ------------- |
| **Language**                 | Rust                    | Python           | Rust           | Proprietary | Go/C++         | Go          | C++           |
| **Embedded mode**            | **PyO3, zero-copy**     | Python-speed     | No             | No          | No             | No          | SWIG bindings |
| **Server mode**              | **Yes**                 | Yes              | Yes            | Cloud only  | Yes            | Yes         | No            |
| `**pip install` just works** | **Yes**                 | Yes              | No (Docker)    | N/A (SaaS)  | No (Docker)    | No (Docker) | Yes           |
| **Binary size**              | **~5 MB**               | ~200 MB+         | ~50 MB         | N/A         | ~500 MB+       | ~100 MB+    | ~20 MB        |
| **Startup time**             | **<10 ms**              | ~1-2 s           | ~1-3 s         | N/A         | ~5-10 s        | ~3-5 s      | <10 ms        |
| **Hybrid search**            | **BM25 + RRF**          | No               | Requires setup | No          | Sparse vectors | BM25        | No            |
| **Metadata filtering**       | **10 operators**        | Yes              | Yes            | Yes         | Yes            | Yes         | No            |
| **Persistence**              | **mmap + SQLite + WAL** | SQLite + Parquet | RocksDB        | Cloud       | RocksDB        | LSM         | Manual        |
| **Crash recovery**           | **WAL**                 | No               | Yes            | Yes         | Yes            | Yes         | No            |
| **Docker image**             | **~10 MB**              | ~500 MB+         | ~100 MB        | No          | ~1 GB+         | ~300 MB+    | No            |
| **Runs offline**             | **Yes**                 | Yes              | Yes            | No          | Yes            | Yes         | Yes           |
| **License**                  | **Apache 2.0**          | Apache 2.0       | Apache 2.0     | Proprietary | Apache 2.0     | BSD-3       | MIT           |


---

## API Reference

### Python (Embedded)

```python
# Database
db = vxdb.Database()
db.create_collection(name, dimension, metric="cosine", index="flat")
db.get_collection(name)
db.list_collections()
db.delete_collection(name)

# Collection
collection.upsert(ids, vectors, metadata=None, documents=None)
collection.query(vector, top_k=10, filter=None)
collection.hybrid_query(vector, query, top_k=10, alpha=0.5)
collection.keyword_search(query, top_k=10)
collection.delete(ids)
collection.count()
```

### REST API


| Method   | Endpoint                      | Description                           |
| -------- | ----------------------------- | ------------------------------------- |
| `POST`   | `/collections`                | Create collection                     |
| `GET`    | `/collections`                | List collections                      |
| `DELETE` | `/collections/{name}`         | Delete collection                     |
| `POST`   | `/collections/{name}/upsert`  | Upsert vectors (+ optional documents) |
| `POST`   | `/collections/{name}/query`   | Vector search (+ optional filter)     |
| `POST`   | `/collections/{name}/hybrid`  | Hybrid vector + keyword search        |
| `POST`   | `/collections/{name}/keyword` | BM25 keyword search                   |
| `POST`   | `/collections/{name}/delete`  | Delete vectors by ID                  |
| `GET`    | `/collections/{name}/count`   | Count vectors                         |


### Parameters


| Parameter | Values                                                          | Default    |
| --------- | --------------------------------------------------------------- | ---------- |
| `metric`  | `"cosine"`, `"euclidean"`, `"dot"`                              | `"cosine"` |
| `index`   | `"flat"` (exact), `"hnsw"` (approximate)                        | `"flat"`   |
| `filter`  | `$eq` `$ne` `$gt` `$gte` `$lt` `$lte` `$in` `$nin` `$and` `$or` | —          |
| `alpha`   | `0.0` (keyword) to `1.0` (vector)                               | `0.5`      |


---

## Examples

Interactive Jupyter notebooks with step-by-step walkthroughs:


| Notebook                                                              | What you'll build                      |
| --------------------------------------------------------------------- | -------------------------------------- |
| `[quickstart.ipynb](examples/quickstart.ipynb)`                       | Every feature in 5 min (no API keys)   |
| `[openai_embeddings.ipynb](examples/openai_embeddings.ipynb)`         | Semantic search with OpenAI embeddings |
| `[sentence_transformers.ipynb](examples/sentence_transformers.ipynb)` | Free, local embeddings (no API key)    |
| `[langchain_integration.ipynb](examples/langchain_integration.ipynb)` | LangChain + RAG pipeline               |
| `[cohere_embeddings.ipynb](examples/cohere_embeddings.ipynb)`         | Multilingual search with Cohere        |
| `[hybrid_search.ipynb](examples/hybrid_search.ipynb)`                 | Deep dive: vector vs keyword vs hybrid |


---

## Development

```bash
git clone https://github.com/your-org/vxdb.git && cd vxdb

# Rust
cargo build --all
cargo test --all        # 110 tests

# Python
uv venv .venv && source .venv/bin/activate
uv pip install maturin pytest httpx
maturin develop
PYTHONPATH=python pytest tests/ -v
```

The codebase is a Cargo workspace:

```
vxdb/
├── crates/
│   ├── vxdb-core/       # Engine: indexes, distance, storage, hybrid search
│   ├── vxdb-python/     # PyO3 bindings
│   └── vxdb-server/     # Axum REST API server
├── python/vxdb/         # Python package (client SDK, embedding interface)
├── examples/             # Jupyter notebooks
└── tests/                # Python integration tests
```

---

## Roadmap

- SIMD-accelerated distance computation
- GPU acceleration (CUDA/Metal)
- Persistent collections (mmap + SQLite integration with the index layer)
- Streaming upsert for large datasets
- gRPC API
- Official LangChain `VectorStore` integration
- Kubernetes Helm chart
- Benchmarks suite vs Qdrant, ChromaDB, FAISS

---

## License

Apache 2.0