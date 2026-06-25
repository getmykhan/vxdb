"""End-to-end tests for automatic embedding integration.

A collection created with an `embedding_function` lets you pass raw `documents`
to `upsert` and text to `query`/`hybrid_query`; vectors are computed for you.
Without an embedding function, everything behaves like the native classes.
"""

import pytest
import vxdb
from vxdb import EmbeddingFunction


class WordEmbedder(EmbeddingFunction):
    """Deterministic bag-of-words embedder so semantic ranking is testable."""

    VOCAB = ["cat", "dog", "car", "bike", "food", "code"]

    def embed(self, texts):
        return [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]


def make_db():
    return vxdb.Database()


# --- creation / dimension inference -----------------------------------------


def test_create_collection_infers_dimension_from_embedder():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    # VOCAB has 6 words -> dimension 6, inferred without passing `dimension`.
    coll.upsert(ids=["a"], documents=["cat"])
    assert coll.count() == 1


def test_create_collection_explicit_dimension_with_embedder():
    db = make_db()
    coll = db.create_collection("docs", dimension=6, metric="l2", embedding_function=WordEmbedder())
    coll.upsert(ids=["a"], documents=["dog"])
    assert coll.count() == 1


# --- upsert auto-embed -------------------------------------------------------


def test_upsert_documents_auto_embeds():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(
        ids=["pet", "vehicle"],
        documents=["i love my cat and dog", "fast car and bike"],
    )
    assert coll.count() == 2


def test_explicit_vectors_bypass_embedding():
    db = make_db()
    coll = db.create_collection("docs", dimension=3, metric="l2", embedding_function=WordEmbedder())
    # Explicit vectors win even when an embedder is set (and dim can differ).
    coll.upsert(ids=["a", "b"], vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert coll.count() == 2
    res = coll.query(vector=[1.0, 0.0, 0.0], top_k=1)
    assert res[0]["id"] == "a"


# --- query auto-embed --------------------------------------------------------


def test_query_text_auto_embeds():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(
        ids=["pet", "vehicle", "tech"],
        documents=["my cat eats food", "car and bike", "i write code"],
    )
    res = coll.query(query_text="hungry cat food", top_k=1)
    assert res[0]["id"] == "pet"


def test_query_explicit_vector_still_works():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(ids=["pet"], documents=["cat"])
    # cat -> [1,0,0,0,0,0]
    res = coll.query(vector=[1.0, 0, 0, 0, 0, 0], top_k=1)
    assert res[0]["id"] == "pet"


# --- hybrid auto-embed -------------------------------------------------------


def test_hybrid_query_positional_arg_order():
    # Back-compat: the native order is hybrid_query(vector, query, ...). Calling
    # it positionally must keep working (regression guard for the wrapper).
    db = make_db()
    coll = db.create_collection("docs", dimension=3, metric="l2")
    coll.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["hello world", "foo bar"],
    )
    res = coll.hybrid_query([1.0, 0.0, 0.0], "hello", top_k=1)
    assert res[0]["id"] == "a"


def test_hybrid_query_requires_query_text():
    db = make_db()
    coll = db.create_collection("docs", dimension=3, metric="l2")
    coll.upsert(ids=["a"], vectors=[[1.0, 0.0, 0.0]], documents=["hello"])
    with pytest.raises(ValueError):
        coll.hybrid_query(vector=[1.0, 0.0, 0.0])  # no query text


def test_hybrid_query_auto_embeds_vector():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(
        ids=["pet", "tech"],
        documents=["cat food for my cat", "code and more code"],
    )
    res = coll.hybrid_query(query="cat", top_k=1)
    assert res[0]["id"] == "pet"


# --- callable (non-class) embedder ------------------------------------------


def test_plain_callable_embedding_function():
    db = make_db()

    def embed(texts):
        return [[float(len(t))] * 2 for t in texts]

    coll = db.create_collection("docs", metric="l2", embedding_function=embed)
    coll.upsert(ids=["a", "b"], documents=["hi", "hello"])  # len 2 and 5
    res = coll.query(query_text="yo", top_k=1)  # len 2 -> closest to "hi"
    assert res[0]["id"] == "a"


# --- error handling ----------------------------------------------------------


def test_upsert_without_vectors_or_embedder_raises():
    db = make_db()
    coll = db.create_collection("docs", dimension=3)
    with pytest.raises(ValueError):
        coll.upsert(ids=["a"], documents=["hello"])  # no embedder, no vectors


def test_query_text_without_embedder_raises():
    db = make_db()
    coll = db.create_collection("docs", dimension=3)
    coll.upsert(ids=["a"], vectors=[[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        coll.query(query_text="hello")


def test_create_collection_no_dimension_no_embedder_raises():
    db = make_db()
    with pytest.raises(ValueError):
        db.create_collection("docs")


# --- persistence of embedder across get_collection ---------------------------


def test_get_collection_retains_embedder():
    db = make_db()
    db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    again = db.get_collection("docs")
    again.upsert(ids=["a"], documents=["cat"])  # auto-embed works via re-fetch
    assert again.count() == 1


# --- realistic end-to-end semantic search ------------------------------------


def test_end_to_end_semantic_search():
    db = make_db()
    coll = db.create_collection("library", metric="cosine", embedding_function=WordEmbedder())
    coll.upsert(
        ids=["animals", "transport", "programming"],
        documents=[
            "the cat and the dog are pets",
            "i drive a car and ride a bike",
            "i love to code and write code",
        ],
    )
    assert coll.count() == 3
    # Query by text; the animal doc should win.
    res = coll.query(query_text="my cat and my dog", top_k=1)
    assert res[0]["id"] == "animals"
    # And a transport query should win for the transport doc.
    res = coll.query(query_text="car bike ride", top_k=1)
    assert res[0]["id"] == "transport"


# --- backward compatibility (no embedding function) --------------------------


def test_backward_compatible_without_embedder():
    db = make_db()
    coll = db.create_collection("docs", dimension=3, metric="cosine")
    coll.upsert(ids=["a", "b"], vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert coll.count() == 2
    res = coll.query(vector=[1.0, 0.0, 0.0], top_k=1)
    assert res[0]["id"] == "a"
    assert repr(coll) == "Collection(name='docs')"


def test_auto_embed_with_metadata_and_filter():
    db = make_db()
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(
        ids=["a", "b"],
        documents=["cat food", "cat code"],
        metadata=[{"kind": "pet"}, {"kind": "tech"}],
    )
    res = coll.query(query_text="cat", top_k=5, filter={"kind": "tech"})
    assert len(res) == 1 and res[0]["id"] == "b"


def test_persistent_reopen_without_embedder_raises(tmp_path):
    # The embedding function is code, not data: it is not persisted. A reopened
    # collection has no embedder, so a text-only query must error clearly rather
    # than silently misbehave.
    path = str(tmp_path / "db")
    db = vxdb.Database(path=path)
    coll = db.create_collection("docs", metric="l2", embedding_function=WordEmbedder())
    coll.upsert(ids=["a"], documents=["cat"])

    db2 = vxdb.Database(path=path)
    reopened = db2.get_collection("docs")
    assert reopened.embedding_function is None
    with pytest.raises(ValueError):
        reopened.query(query_text="cat")
    # explicit vectors still work on the reopened collection
    assert reopened.query(vector=[1.0, 0, 0, 0, 0, 0], top_k=1)[0]["id"] == "a"
