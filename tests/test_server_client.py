"""Integration tests: Python client against vxdb HTTP server.

These tests start the server binary, run client operations, and verify results.
They require the vxdb-server binary to be built (cargo build -p vxdb-server).
"""

import os
import signal
import subprocess
import sys
import time

import pytest

# Adjust path so we can import vxdb from the workspace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def find_server_binary():
    """Find the vxdb-server binary in the cargo target directory."""
    base = os.path.join(os.path.dirname(__file__), "..")
    candidates = [
        os.path.join(base, "target", "debug", "vxdb-server"),
        # sandboxed builds use a different target dir
    ]
    # Also check CARGO_TARGET_DIR env var
    cargo_target = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target:
        candidates.insert(0, os.path.join(cargo_target, "debug", "vxdb-server"))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


SERVER_PORT = 18932  # random high port to avoid conflicts


@pytest.fixture(scope="module")
def server():
    """Start the vxdb server and yield. Kill it after tests."""
    binary = find_server_binary()
    if binary is None:
        pytest.skip("vxdb-server binary not found; run 'cargo build -p vxdb-server' first")

    proc = subprocess.Popen(
        [binary, "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    import httpx
    for _ in range(30):
        try:
            httpx.get(f"http://127.0.0.1:{SERVER_PORT}/collections", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Server did not start in time")

    yield proc

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture
def client(server):
    from vxdb.client import Client
    c = Client(f"http://127.0.0.1:{SERVER_PORT}")
    yield c
    # Clean up collections after each test
    for name in c.list_collections():
        c.delete_collection(name)
    c.close()


def test_create_and_list(client):
    client.create_collection("test_col", dimension=3)
    names = client.list_collections()
    assert "test_col" in names


def test_upsert_and_query(client):
    coll = client.create_collection("docs", dimension=3, metric="cosine")
    coll.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.1, 0.0]],
    )
    assert coll.count() == 3

    results = coll.query(vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"


def test_upsert_with_metadata(client):
    coll = client.create_collection("meta", dimension=3)
    coll.upsert(
        ids=["a"],
        vectors=[[1.0, 0.0, 0.0]],
        metadata=[{"color": "red", "price": 42}],
    )
    results = coll.query(vector=[1.0, 0.0, 0.0], top_k=1)
    assert results[0]["metadata"]["color"] == "red"
    assert results[0]["metadata"]["price"] == 42


def test_filtered_query(client):
    coll = client.create_collection("filtered", dimension=3)
    coll.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        metadata=[{"color": "red"}, {"color": "blue"}, {"color": "red"}],
    )
    results = coll.query(
        vector=[1.0, 0.0, 0.0],
        top_k=10,
        filter={"color": {"$eq": "red"}},
    )
    assert len(results) == 2
    assert all(r["metadata"]["color"] == "red" for r in results)


def test_delete_vectors(client):
    coll = client.create_collection("deltest", dimension=3)
    coll.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    deleted = coll.delete(ids=["a"])
    assert deleted == [True]
    assert coll.count() == 1


def test_delete_collection(client):
    client.create_collection("to_delete", dimension=3)
    client.delete_collection("to_delete")
    assert "to_delete" not in client.list_collections()
