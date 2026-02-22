"""End-to-end tests for vxdb embedded mode (PyO3 bindings)."""

import pytest
import vxdb


def test_create_database():
    db = vxdb.Database()
    assert repr(db) == "Database(collections=0)"


def test_create_collection():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")
    assert repr(coll) == "Collection(name='docs')"
    assert coll.count() == 0


def test_list_collections():
    db = vxdb.Database()
    db.create_collection("a", dimension=3)
    db.create_collection("b", dimension=3)
    names = sorted(db.list_collections())
    assert names == ["a", "b"]


def test_delete_collection():
    db = vxdb.Database()
    db.create_collection("docs", dimension=3)
    db.delete_collection("docs")
    assert db.list_collections() == []


def test_duplicate_collection_raises():
    db = vxdb.Database()
    db.create_collection("docs", dimension=3)
    with pytest.raises(ValueError, match="already exists"):
        db.create_collection("docs", dimension=3)


def test_upsert_and_query():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")

    coll.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.1, 0.0]],
    )
    assert coll.count() == 3

    results = coll.query(vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"
    assert results[0]["score"] < 0.01  # near-zero distance


def test_upsert_with_metadata():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3)

    coll.upsert(
        ids=["a"],
        vectors=[[1.0, 0.0, 0.0]],
        metadata=[{"color": "red", "price": 42}],
    )

    results = coll.query(vector=[1.0, 0.0, 0.0], top_k=1)
    assert results[0]["metadata"]["color"] == "red"
    assert results[0]["metadata"]["price"] == 42


def test_filtered_query():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3)

    coll.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        metadata=[
            {"color": "red"},
            {"color": "blue"},
            {"color": "red"},
        ],
    )

    results = coll.query(
        vector=[1.0, 0.0, 0.0],
        top_k=10,
        filter={"color": {"$eq": "red"}},
    )
    assert len(results) == 2
    assert all(r["metadata"]["color"] == "red" for r in results)


def test_filtered_query_numeric():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3)

    coll.upsert(
        ids=["cheap", "mid", "expensive"],
        vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        metadata=[{"price": 10}, {"price": 50}, {"price": 200}],
    )

    results = coll.query(
        vector=[1.0, 0.0, 0.0],
        top_k=10,
        filter={"price": {"$lte": 50}},
    )
    assert len(results) == 2
    assert all(r["metadata"]["price"] <= 50 for r in results)


def test_delete_vectors():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3)

    coll.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert coll.count() == 2

    deleted = coll.delete(ids=["a"])
    assert deleted == [True]
    assert coll.count() == 1


def test_upsert_overwrites():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3)

    coll.upsert(ids=["a"], vectors=[[1.0, 0.0, 0.0]])
    coll.upsert(ids=["a"], vectors=[[0.0, 1.0, 0.0]])
    assert coll.count() == 1

    results = coll.query(vector=[0.0, 1.0, 0.0], top_k=1)
    assert results[0]["id"] == "a"
    assert results[0]["score"] < 0.01


def test_euclidean_metric():
    db = vxdb.Database()
    coll = db.create_collection("euc", dimension=2, metric="euclidean")

    coll.upsert(
        ids=["origin", "near", "far"],
        vectors=[[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]],
    )

    results = coll.query(vector=[0.0, 0.0], top_k=3)
    assert results[0]["id"] == "origin"
    assert results[1]["id"] == "near"
    assert results[2]["id"] == "far"


def test_dot_product_metric():
    db = vxdb.Database()
    coll = db.create_collection("dot", dimension=2, metric="dot")

    coll.upsert(
        ids=["aligned", "orthogonal"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )

    results = coll.query(vector=[1.0, 0.0], top_k=2)
    assert results[0]["id"] == "aligned"


def test_hnsw_index():
    db = vxdb.Database()
    coll = db.create_collection("hnsw_test", dimension=3, index="hnsw")

    coll.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.1, 0.0]],
    )

    results = coll.query(vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"


def test_get_collection():
    db = vxdb.Database()
    db.create_collection("docs", dimension=3)
    coll = db.get_collection("docs")
    assert repr(coll) == "Collection(name='docs')"


def test_get_nonexistent_collection_raises():
    db = vxdb.Database()
    with pytest.raises(ValueError, match="not found"):
        db.get_collection("nope")


def test_invalid_metric_raises():
    db = vxdb.Database()
    with pytest.raises(ValueError, match="unknown metric"):
        db.create_collection("docs", dimension=3, metric="invalid")


def test_embedding_function_interface():
    from vxdb.embedding import EmbeddingFunction

    class MockEmbedder(EmbeddingFunction):
        def embed(self, texts):
            return [[float(i)] * 3 for i, _ in enumerate(texts)]

    embedder = MockEmbedder()
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert vectors[0] == [0.0, 0.0, 0.0]
    assert vectors[1] == [1.0, 1.0, 1.0]


def test_upsert_with_documents():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")

    coll.upsert(
        ids=["ml", "cook", "dl"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        documents=[
            "machine learning for image recognition",
            "cooking recipes for pasta and pizza",
            "deep learning neural networks",
        ],
    )
    assert coll.count() == 3


def test_keyword_search():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")

    coll.upsert(
        ids=["ml", "cook", "dl"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        documents=[
            "machine learning for image recognition",
            "cooking recipes for pasta and pizza",
            "deep learning neural networks",
        ],
    )

    results = coll.keyword_search(query="machine learning", top_k=10)
    assert len(results) > 0
    assert results[0]["id"] == "ml"


def test_hybrid_query():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")

    coll.upsert(
        ids=["vec_close", "text_match", "both"],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        documents=[
            "unrelated content about cooking",
            "machine learning and artificial intelligence",
            "machine learning for image processing",
        ],
    )

    # Pure vector search: vec_close should win
    vec_results = coll.query(vector=[1.0, 0.0, 0.0], top_k=3)
    assert vec_results[0]["id"] == "vec_close"

    # Hybrid (alpha=0.5): "both" should rank well
    hybrid = coll.hybrid_query(
        vector=[1.0, 0.0, 0.0],
        query="machine learning",
        top_k=3,
        alpha=0.5,
    )
    assert len(hybrid) == 3
    both_rank = next(i for i, r in enumerate(hybrid) if r["id"] == "both")
    assert both_rank <= 1, f"expected 'both' in top 2, got rank {both_rank}"


def test_hybrid_alpha_extremes():
    db = vxdb.Database()
    coll = db.create_collection("docs", dimension=3, metric="cosine")

    coll.upsert(
        ids=["vec_only", "text_only"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["unrelated document", "quantum computing research"],
    )

    # alpha=1.0 (pure vector): vec_only should win
    results = coll.hybrid_query(
        vector=[1.0, 0.0, 0.0], query="quantum computing", top_k=2, alpha=1.0
    )
    assert results[0]["id"] == "vec_only"

    # alpha=0.0 (pure keyword): text_only should win
    results = coll.hybrid_query(
        vector=[1.0, 0.0, 0.0], query="quantum computing", top_k=2, alpha=0.0
    )
    assert results[0]["id"] == "text_only"
