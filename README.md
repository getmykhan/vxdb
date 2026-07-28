<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/getmykhan/vxdb/main/docs/logo/dark.svg">
    <img src="https://raw.githubusercontent.com/getmykhan/vxdb/main/docs/logo/light.svg" alt="vxdb" width="280">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vxdb/"><img src="https://img.shields.io/pypi/v/vxdb" alt="PyPI"></a>
  <a href="https://pypi.org/project/vxdb-server/"><img src="https://img.shields.io/pypi/v/vxdb-server?label=vxdb-server" alt="PyPI: vxdb-server"></a>
  <a href="https://github.com/getmykhan/vxdb/actions/workflows/ci.yml"><img src="https://github.com/getmykhan/vxdb/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/vxdb/"><img src="https://img.shields.io/pypi/pyversions/vxdb" alt="Python"></a>
  <a href="https://github.com/getmykhan/vxdb/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

<p align="center"><strong>Memory in the loop: an in-process vector database fast enough for your agent to consult on every step.</strong></p>

<p align="center">One Rust engine, two modes. Ephemeral agent working memory, or a persistent vector database. One <code>pip install</code> away.</p>

```bash
pip install vxdb
```

## Two modes

| [Ephemeral: agent working memory](#memory-in-the-loop)                                                 | [Persistent: vector database](#the-vector-database-persistent)                                    |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Working memory your agent reads and writes on every step. Create it at the start of a run, drop it at the end. | Persistent collections: mmap + SQLite + WAL, four ways to search, optional HTTP server. |
| `scratch(embed)`, then `recall` in about 100 microseconds.                                | `Database(path=...)` and the data survives restarts.                                  |
| [Jump to working memory](#memory-in-the-loop)                                             | [Jump to the database](#the-vector-database-persistent)                                  |

```python
# Ephemeral mode: working memory for an agent run
from vxdb.agent import scratch

wm = scratch(embed)
wm.add("user prefers concise answers")
wm.recall("how should I answer?", k=3)
```

```python
# Persistent mode: a vector database on disk
import vxdb

db = vxdb.Database(path="./my_data")
docs = db.create_collection("docs", dimension=384)
docs.upsert(ids=["a"], vectors=[embed("hello")], documents=["hello"])
```

`embed()` is any function that turns text into vectors: OpenAI, Sentence Transformers, Cohere, or your own. See [Embedding Providers](#embedding-providers).

## Memory in the loop

A query returns in about 100 microseconds in-process, three orders of magnitude below a networked database. At that latency your agent can check memory before every action and write memory after every observation. `scratch()` creates a semantic scratchpad in memory with no setup; it lasts for the run and `close()` drops it.

### Create a scratchpad

```python
from vxdb.agent import scratch

wm = scratch(embed)   # in-memory, scoped to this run
wm.add("checkout depends on tax-service", metadata={"hop": 1})

hits = wm.recall("what does checkout use for tax", k=3)
hits[0].text        # "checkout depends on tax-service"
hits[0].similarity  # higher is better

wm.close()          # drop it
```

`WorkingMemory` is also a context manager: `with scratch(embed) as wm:` closes on exit. Hits carry `.text`, `.similarity`, `.metadata`, and `.id`, with `similarity` normalized so higher is better (the raw `query` API returns distance). `len(wm)` counts stored items, and `add_many(texts)` batches writes through a single embedding call.

### Tool integration

Register `remember` and `recall` as tools; the model decides what to store. The `seen` check rejects near-duplicates:

```python
from agents import Agent, function_tool  # OpenAI Agents SDK; any tool-calling framework works
from vxdb.agent import scratch

wm = scratch(embed)  # ephemeral scratchpad for this run

@function_tool
def remember(fact: str) -> str:
    """Save one durable fact, preference, or constraint."""
    if wm.seen(fact, threshold=0.9):  # loop guard: near-duplicates are rejected
        return "already known"
    wm.add(fact)
    return "stored"

@function_tool
def recall(query: str) -> list[str]:
    """Fetch the stored facts most relevant to the query."""
    return [hit.text for hit in wm.recall(query, k=5)]

agent = Agent(
    name="assistant",
    instructions=(
        "The moment the user states a durable fact, preference, or constraint, "
        "call remember() with it. Never store small talk. Call recall() before "
        "answering anything about earlier context."
    ),
    tools=[remember, recall],
)
```

Each `remember` call costs microseconds of store time on top of the embedding call.

### Loop guards

Use `match` and `seen` to check whether a planned action is a near-duplicate of a stored one:

```python
plan = "re-run the failing test with verbose logging"

if wm.seen(plan, threshold=0.9):
    pick_another_action()

hit = wm.match(plan, threshold=0.85)  # the near-duplicate memory, or None
if hit:
    print(f"tried before: {hit.text} ({hit.similarity:.2f})")
```

`match` returns the closest stored item when its similarity clears the threshold; `seen` is the boolean version. `match` and `recall` also take a `where=` metadata filter using the same operators as collection queries.

### Timing

`timing_summary()` reports per-operation latency for the current run and breaks out embedder time from store time:

```python
wm.timing_summary()
# {"embed_us": {...}, "write_us": {...}, "query_us": {...}}
# each with count / p50_us / p99_us / mean_us / max_us
```

The [agent working memory notebook](examples/agent_working_memory.ipynb) runs the full loop and a with/without A/B where a bounded-window agent fails without memory and succeeds with it.

## The vector database (persistent)

Pass a `path` and the same engine persists everything to disk:

```python
import vxdb

db = vxdb.Database(path="./my_data")  # data survives restarts
collection = db.create_collection("docs", dimension=384, metric="cosine")

collection.upsert(
    ids=["a", "b", "c"],
    vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    metadata=[{"type": "article"}, {"type": "blog"}, {"type": "article"}],
    documents=["intro to ML", "my favorite recipes", "deep learning guide"],
)
```

No Docker. No config files. No cloud account. Drop the `path` argument for an in-memory database.

### Search

```python
# 1. Vector similarity
results = collection.query(vector=[0.1, 0.2, ...], top_k=5)

# Trade recall for latency on HNSW: raise ef_search for more accurate results,
# lower it for faster ones. Defaults to the index setting when omitted.
results = collection.query(vector=[0.1, ...], top_k=5, ef_search=200)

# 2. Filtered (metadata constraints)
results = collection.query(
    vector=[0.1, ...], top_k=5,
    filter={"type": {"$eq": "article"}}
)

# 3. Hybrid (vector + keyword, the sweet spot)
results = collection.hybrid_query(
    vector=[0.1, ...],
    query="machine learning",
    top_k=5,
    alpha=0.5,  # 0=keyword only, 1=vector only
)

# 4. Keyword only (BM25)
results = collection.keyword_search(query="machine learning", top_k=5)
```

Every result returns `{"id", "score", "metadata", "document"}`.

### The persistence stack

`path=` turns on three layers: vectors in a memory-mapped file, metadata and documents in SQLite, and a write-ahead log that replays on open for crash recovery. The entire hot path (distance computation, HNSW traversal, BM25 scoring, mmap I/O) is pure Rust, called from Python via PyO3 with no serialization and no subprocess. Search releases the GIL, so concurrent queries run in parallel across cores.

### Attaching an embedder

Attach an `embedding_function` to a collection and pass `documents` to `upsert` and `query_text` to `query`:

```python
from vxdb import Database, EmbeddingFunction

class MyEmbedder(EmbeddingFunction):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return your_model.encode(texts)

db = Database()
docs = db.create_collection("docs", embedding_function=MyEmbedder())  # dimension inferred

docs.upsert(ids=["a", "b"], documents=["how to train a model", "best pasta recipe"])
docs.query(query_text="machine learning", top_k=5)
```

The `embedding_function` can be an `EmbeddingFunction` subclass or any callable `list[str] -> list[list[float]]`. Passing `vectors`/`vector` still works and bypasses embedding; vxdb does not require or import your model library.

### HTTP server

The server exposes the same engine over HTTP to multiple clients.

The server ships as a **separate, optional package**: `pip install vxdb-server` adds the `vxdb-server` binary without touching the core `vxdb` wheel.

```bash
# Install the standalone server (separate package, no extra deps)
pip install vxdb-server

# Start it
vxdb-server --host 0.0.0.0 --port 8080
```

The Python `Client` lives in the core package. Install it with the `server` extra (which pulls in `httpx`):

```bash
pip install 'vxdb[server]'
```

> **Note:** the server is **in-memory only** for now. Data does not persist
> across restarts. For persistence, run vxdb in-process (`vxdb.Database(path=...)`).

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
docker run -p 8080:8080 vxdb    # ~145 MB Debian-based image
```

## Architecture

![vxdb architecture](docs/vxdb-architecture.png)

## Installation

```bash
pip install vxdb
```

One native wheel **under 2 MB** with **zero Python dependencies**: no numpy, no scipy, no protobuf, no grpcio version conflicts. Starts in **under 10 ms**. Works on **macOS, Linux, Windows**, Python 3.11+.

No infrastructure, no network calls at query time. vxdb runs on a laptop, a CI runner, a Raspberry Pi, an AWS Lambda, or an air-gapped server.

For the HTTP client (talking to a remote vxdb server):

```bash
pip install 'vxdb[server]'
```

## Embedding Providers

vxdb stores **pre-computed vectors**: bring any embedding model. Pass the same function to `scratch(embed)` or call it yourself before `upsert`. Step-by-step notebooks for each provider:

| Provider                     | Install                             | API Key?   | Notebook                                                                     |
| ---------------------------- | ----------------------------------- | ---------- | ---------------------------------------------------------------------------- |
| **OpenAI**                   | `pip install openai`                | Yes        | [examples/openai_embeddings.ipynb](examples/openai_embeddings.ipynb)         |
| **Sentence Transformers**    | `pip install sentence-transformers` | No (local) | [examples/sentence_transformers.ipynb](examples/sentence_transformers.ipynb) |
| **LangChain** (any provider) | `pip install langchain-openai`      | Depends    | [examples/langchain_integration.ipynb](examples/langchain_integration.ipynb) |
| **Cohere**                   | `pip install cohere`                | Yes        | [examples/cohere_embeddings.ipynb](examples/cohere_embeddings.ipynb)         |
| **Ollama** (local LLMs)      | `pip install ollama`                | No (local) | n/a                                                                          |

To skip the manual embedding step, [attach an embedder to a collection](#attaching-an-embedder).

## Hybrid Search

`hybrid_query` fuses vector search and keyword search in a single call. vxdb computes BM25 from the documents you already upserted: you run no separate sparse encoder and pass no pre-computed sparse vectors.

**How it works:**

1. **You upsert with documents.** vxdb tokenizes the raw text into a built-in BM25 index alongside your vectors.
2. **At query time,** vector search and BM25 run in parallel, then Reciprocal Rank Fusion merges both ranked lists.
3. **You control the blend:** `alpha=1.0` (pure vector) → `alpha=0.5` (balanced) → `alpha=0.0` (pure keyword).

**When to use it:** Specific product names, error codes, proper nouns: anything where exact terms matter alongside semantic meaning. See [examples/hybrid_search.ipynb](examples/hybrid_search.ipynb) for side-by-side comparisons.

```python
results = collection.hybrid_query(
    vector=embed("lightweight laptop for students"),
    query="MacBook Air M4",
    top_k=5,
    alpha=0.5,
)
```

## How vxdb compares

|                                  | vxdb                        | Zvec (Alibaba)              | ChromaDB                | Qdrant                    | Pinecone      | Milvus                  | Weaviate                    | FAISS         |
| -------------------------------- | --------------------------- | --------------------------- | ----------------------- | ------------------------- | ------------- | ----------------------- | --------------------------- | ------------- |
| **Language**                     | Rust                        | C++ (Proxima)               | Rust (v1.0+)            | Rust                      | Proprietary   | Go/C++                  | Go                          | C++           |
| **Embedded mode**                | **PyO3, true in-process**   | In-process                  | In-process              | Python-only local mode    | No            | Milvus Lite             | Subprocess (downloads Go binary) | SWIG bindings |
| **Server mode**                  | **Yes**                     | No                          | Yes                     | Yes                       | Cloud only    | Yes                     | Yes                         | No            |
| **`pip install` just works**     | **Yes**                     | Yes                         | Yes                     | Yes (local mode)          | N/A (SaaS)    | Yes (Milvus Lite)       | Yes (Linux/macOS)           | Yes           |
| **Python dependencies**          | **None (zero)**             | None (zero-dep)             | Several                 | numpy, grpcio, etc.       | N/A           | grpcio, protobuf, etc.  | grpcio, etc.                | numpy         |
| **Wheel size**                   | **~1.5 MB**                 | ~17 MB                      | ~20 MB                  | ~50 MB                    | N/A           | ~50 MB+                 | ~100 MB+ (downloads binary) | ~20 MB        |
| **Startup time**                 | **<10 ms**                  | <100 ms                     | <500 ms                 | ~1-3 s (server)           | N/A           | ~5-10 s (server)        | ~3-5 s (server)             | <10 ms        |
| **Hybrid search**                | **Built-in BM25 + RRF**    | BM25 + RRF + weighted       | RRF (dense+sparse)      | RRF, DBSF                 | Sparse+dense  | Sparse vectors          | BM25 + RRF                  | No            |
| **BM25 without external encoder** | **Yes (automatic)**        | Yes (native FTS)            | Yes                     | Requires sparse encoder   | No            | Requires sparse encoder | Yes                         | No            |
| **Sparse vectors**               | No                          | Yes                         | Yes                     | Yes                       | Yes           | Yes                     | No                          | No            |
| **Multi-vector queries**         | No                          | Yes                         | No                      | Yes                       | No            | No                      | No                          | No            |
| **Metadata filtering**           | **10 operators**            | Structured filters          | Yes                     | Yes                       | Yes           | Yes                     | Yes                         | No            |
| **Persistence**                  | **mmap + SQLite + WAL**     | Custom engine               | SQLite                  | Gridstore                 | Cloud         | RocksDB                 | LSM                         | Manual        |
| **Crash recovery**               | **WAL**                     | Yes                         | Yes (v1.0)              | Yes                       | Yes           | Yes                     | Yes                         | No            |
| **Quantization**                 | No (planned)                | FP16, INT8, INT4, RaBitQ    | No                      | Scalar/PQ                 | Yes           | Yes                     | PQ/BQ                       | PQ/SQ         |
| **Docker image**                 | ~145 MB                     | N/A (no server)             | ~200 MB+                | ~100 MB                   | No            | ~1 GB+                  | ~300 MB+                    | No            |
| **Runs offline**                 | **Yes**                     | Yes                         | Yes                     | Yes                       | No            | Yes                     | Yes                         | Yes           |
| **License**                      | **Apache 2.0**              | Apache 2.0                  | Apache 2.0              | Apache 2.0                | Proprietary   | Apache 2.0              | BSD-3                       | MIT           |

## API Reference

### Agent working memory

```python
from vxdb.agent import scratch

wm = scratch(embed)  # embed: EmbeddingFunction or callable list[str] -> list[list[float]]

wm.add(text, metadata=None, *, id=None)           # embed + store one item; returns its id
wm.add_many(texts, metadatas=None, *, ids=None)   # batch add; one embedding call
wm.recall(query, k=5, *, where=None)              # top-k hits, best first
wm.match(text, threshold=0.85, *, where=None)     # closest hit above threshold, or None
wm.seen(text, threshold=0.85)                     # True if match() finds one
wm.timing_summary()                               # live latency: embed vs store, p50/p99
len(wm)                                           # items currently stored
wm.close()                                        # drop the scratchpad (idempotent)
```

`WorkingMemory` is a context manager: `with scratch(embed) as wm:` closes on exit. Recall hits expose `.text`, `.similarity` (higher is better), `.metadata`, and `.id`, and are plain dicts underneath, so they serialize to JSON as-is. `where=` takes the same filter operators as `collection.query`.

### Python (Embedded)

```python
# Database
db = vxdb.Database()                  # in-memory (ephemeral)
db = vxdb.Database(path="./my_data")  # persistent (data survives restarts)
db.create_collection(name, dimension, metric="cosine", index="flat")
db.get_collection(name)
db.list_collections()
db.delete_collection(name)

# Collection
collection.upsert(ids, vectors, metadata=None, documents=None)
collection.query(vector, top_k=10, filter=None, ef_search=None)
collection.hybrid_query(vector, query, top_k=10, alpha=0.5)
collection.keyword_search(query, top_k=10)
collection.delete(ids)
collection.count()
```

`vectors` accepts a `list[list[float]]` or a 2-D `float32` NumPy array. vxdb reads NumPy arrays zero-copy via the buffer protocol and does not import or require NumPy.

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
| `filter`  | `$eq` `$ne` `$gt` `$gte` `$lt` `$lte` `$in` `$nin` `$and` `$or` | none       |
| `alpha`   | `0.0` (keyword) to `1.0` (vector)                               | `0.5`      |
| `ef_search` | HNSW candidates explored per query; higher lifts recall, costs latency | `150` (index default) |

## Examples

Interactive Jupyter notebooks with step-by-step walkthroughs:

| Notebook                                                            | What you'll build                          |
| ------------------------------------------------------------------- | ------------------------------------------ |
| [agent_working_memory.ipynb](examples/agent_working_memory.ipynb)   | An agent with memory in the loop, plus a with/without A/B |
| [quickstart.ipynb](examples/quickstart.ipynb)                       | Every feature in 5 min (no API keys)       |
| [openai_embeddings.ipynb](examples/openai_embeddings.ipynb)         | Semantic search with OpenAI embeddings     |
| [sentence_transformers.ipynb](examples/sentence_transformers.ipynb) | Free, local embeddings (no API key)        |
| [langchain_integration.ipynb](examples/langchain_integration.ipynb) | LangChain + RAG pipeline                   |
| [cohere_embeddings.ipynb](examples/cohere_embeddings.ipynb)         | Multilingual search with Cohere            |
| [hybrid_search.ipynb](examples/hybrid_search.ipynb)                 | Deep dive: vector vs keyword vs hybrid     |

## Development

```bash
git clone https://github.com/getmykhan/vxdb.git && cd vxdb

# Rust
cargo build --all
cargo test --all        # 120+ tests

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
├── python/vxdb/         # Python package (agent memory, client SDK, embedding interface)
├── examples/             # Jupyter notebooks
└── tests/                # Python integration tests
```

## Roadmap

- ~~Persistent collections (mmap + SQLite + WAL)~~ **Done**
- ~~SIMD-accelerated distance computation~~ **Done** (v0.5.1: NEON on arm64, AVX2 on x86_64)
- Quantization (int8/binary) for reduced memory
- GPU acceleration (CUDA/Metal)
- HNSW graph serialization (fast restart for large indexes)
- Streaming upsert for large datasets
- Sparse vector support
- gRPC API
- Official LangChain `VectorStore` integration
- Kubernetes Helm chart
- Benchmarks suite vs Qdrant, ChromaDB, Zvec, FAISS

## License

Apache 2.0
