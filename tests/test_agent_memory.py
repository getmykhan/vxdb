"""Tests for vxdb.agent.WorkingMemory, the ephemeral working-memory primitive.

Uses a deterministic vocabulary-count embedder (no network, no model downloads)
so similarities are hand-computable: texts become L2-normalized counts over a
four-word vocabulary, and cosine similarity between "apple" and "apple banana"
is exactly 1/sqrt(2).
"""

import math

import pytest
from vxdb.agent import Recall, WorkingMemory, scratch

VOCAB = ["apple", "banana", "cherry", "tax"]


def embed(texts):
    out = []
    for t in texts:
        words = t.lower().split()
        v = [float(words.count(w)) for w in VOCAB]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def wm():
    m = scratch(embed)
    yield m
    m.close()


# -- add / recall ----------------------------------------------------------


def test_add_recall_roundtrip(wm):
    wm.add("apple", metadata={"hop": 1})
    wm.add("banana")
    hits = wm.recall("apple", k=1)
    assert len(hits) == 1
    assert hits[0].text == "apple"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[0].metadata == {"hop": 1}


def test_recall_orders_best_first(wm):
    wm.add("apple")
    wm.add("apple banana")
    wm.add("banana")
    hits = wm.recall("apple", k=3)
    assert [h.text for h in hits] == ["apple", "apple banana", "banana"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[1].similarity == pytest.approx(1 / math.sqrt(2), abs=1e-5)
    assert hits[2].similarity == pytest.approx(0.0, abs=1e-5)


def test_recall_respects_k(wm):
    for text in ("apple", "apple apple", "apple banana", "banana"):
        wm.add(text)
    assert len(wm.recall("apple", k=2)) == 2


def test_recall_where_filter(wm):
    wm.add("apple", metadata={"kind": "fruit"})
    wm.add("apple banana", metadata={"kind": "salad"})
    hits = wm.recall("apple", k=5, where={"kind": {"$eq": "salad"}})
    assert [h.text for h in hits] == ["apple banana"]


def test_recall_on_empty_store(wm):
    assert wm.recall("apple") == []
    assert wm.match("apple") is None
    assert wm.seen("apple") is False


def test_recall_hits_are_dicts(wm):
    wm.add("apple")
    hit = wm.recall("apple", k=1)[0]
    assert isinstance(hit, Recall)
    assert isinstance(hit, dict)
    assert set(hit) == {"id", "text", "similarity", "metadata"}
    assert hit.id == "0"


# -- similarity normalization per metric ------------------------------------
# The engine returns distance (smaller = closer); WorkingMemory flips it into
# higher-is-better similarity. This mapping is the subtlest logic in the module.


def test_similarity_cosine_identical_is_one(wm):
    wm.add("apple")
    assert wm.recall("apple", k=1)[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_similarity_cosine_orthogonal_is_zero(wm):
    wm.add("banana")
    assert wm.recall("apple", k=1)[0].similarity == pytest.approx(0.0, abs=1e-5)


def test_similarity_dot_passthrough():
    m = scratch(embed, metric="dot")
    try:
        m.add("apple")
        m.add("banana")
        hits = m.recall("apple", k=2)
        assert hits[0].text == "apple"
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
        assert hits[1].similarity == pytest.approx(0.0, abs=1e-5)
    finally:
        m.close()


def test_similarity_l2_identical_is_zero_and_orders():
    m = scratch(embed, metric="l2")
    try:
        m.add("apple")
        m.add("banana")
        hits = m.recall("apple", k=2)
        assert hits[0].text == "apple"
        assert hits[0].similarity == pytest.approx(0.0, abs=1e-5)
        # farther item maps to a strictly lower (negative) similarity
        assert hits[1].similarity < hits[0].similarity
    finally:
        m.close()


# -- match / seen ------------------------------------------------------------


def test_match_above_threshold(wm):
    wm.add("apple banana")
    hit = wm.match("apple", threshold=0.6)  # cosine sim = 0.7071
    assert hit is not None
    assert hit.text == "apple banana"


def test_match_below_threshold(wm):
    wm.add("apple banana")
    assert wm.match("apple", threshold=0.8) is None


def test_seen(wm):
    wm.add("apple")
    assert wm.seen("apple") is True
    assert wm.seen("cherry") is False


# -- add_many ----------------------------------------------------------------


def test_add_many_continues_id_sequence(wm):
    assert wm.add("apple") == "0"
    assert wm.add_many(["banana", "cherry"]) == ["1", "2"]
    assert wm.add("tax") == "3"
    assert len(wm) == 4


def test_add_many_explicit_ids_and_metadata(wm):
    ids = wm.add_many(
        ["apple", "banana"],
        metadatas=[{"n": 1}, {"n": 2}],
        ids=["a", "b"],
    )
    assert ids == ["a", "b"]
    hit = wm.recall("banana", k=1)[0]
    assert hit.id == "b"
    assert hit.metadata == {"n": 2}


def test_add_many_embeds_once():
    m = scratch(embed)
    try:
        probes = len(m.timing["embed_us"])  # constructor dimension probe
        m.add_many(["apple", "banana", "cherry"])
        assert len(m.timing["embed_us"]) == probes + 1
    finally:
        m.close()


# -- lifecycle ----------------------------------------------------------------


def test_close_is_idempotent_and_guards():
    m = scratch(embed)
    m.add("apple")
    m.close()
    m.close()  # second close must not raise
    assert len(m) == 0
    for op in (
        lambda: m.add("x"),
        lambda: m.add_many(["x"]),
        lambda: m.recall("x"),
        lambda: m.match("x"),
    ):
        with pytest.raises(RuntimeError):
            op()


def test_context_manager_closes():
    with scratch(embed) as m:
        m.add("apple")
        assert len(m) == 1
    with pytest.raises(RuntimeError):
        m.recall("apple")


def test_named_instance_repr():
    m = scratch(embed, name="wm-test")
    try:
        assert "wm-test" in repr(m)
    finally:
        m.close()
    assert "closed" in repr(m)


def test_working_memory_class_is_exported():
    m = WorkingMemory(embed, name="wm-direct")
    try:
        assert isinstance(m, WorkingMemory)
    finally:
        m.close()


# -- timing -------------------------------------------------------------------


def test_timing_summary_populated(wm):
    wm.add("apple")
    wm.recall("apple")
    summary = wm.timing_summary()
    assert set(summary) == {"embed_us", "write_us", "query_us"}
    for key in ("embed_us", "write_us", "query_us"):
        assert summary[key]["count"] >= 1
        assert summary[key]["p50_us"] > 0
        assert set(summary[key]) == {"count", "p50_us", "p99_us", "mean_us", "max_us"}
