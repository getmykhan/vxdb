# vxdb-server

The standalone HTTP server for [**vxdb**](https://github.com/getmykhan/vxdb) — the
same Rust engine that powers vxdb's embedded mode, exposed over a REST API.

```bash
pip install vxdb-server
vxdb-server --host 0.0.0.0 --port 8080
```

This package ships only the compiled `vxdb-server` binary. To talk to it from
Python, install the core library and use its HTTP client:

```bash
pip install "vxdb[server]"   # core engine + httpx for the client
```

```python
from vxdb import Client

client = Client("http://localhost:8080")
coll = client.create_collection("docs", dimension=384)
coll.upsert(ids=["a"], vectors=[[0.1, 0.2, ...]], documents=["hello world"])
results = coll.query(vector=[0.1, 0.2, ...], top_k=5)
```

## Configuration

| Flag / env | Default | Description |
| --- | --- | --- |
| `--host` / `VXDB_HOST` | `0.0.0.0` | Bind address |
| `--port` / `VXDB_PORT` | `8080` | Bind port |

> **Note:** The server is currently **in-memory only** — data does not persist
> across restarts. For persistent storage, use vxdb's embedded mode
> (`vxdb.Database(path=...)`). Server-side persistence is on the roadmap.

See the [main vxdb README](https://github.com/getmykhan/vxdb#server-mode) for the
full REST API and `curl` examples.

Licensed under Apache-2.0.
