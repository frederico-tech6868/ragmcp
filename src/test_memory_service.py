import hashlib
import json
import math
import re

import pytest

from ragmcp.memory_service import MemoryStore, build_server


class WordEmbedder:
    """Bag of words hashed into 64 dimensions: similar texts share words."""

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem", WordEmbedder())


def test_remember_recall_and_merge(store):
    a = store.remember("lesson", "tests in test_memory.py must use the monkeypatch_env fixture",
                       "ragmcp", files=["src/test_memory.py"])
    assert a["merged"] is False
    again = store.remember("lesson", "tests in test_memory.py must use the monkeypatch_env fixture",
                           "ragmcp")
    assert again == {"id": a["id"], "merged": True}
    other = store.remember("lesson", "tests in test_memory.py must use the monkeypatch_env fixture",
                           "textkit")           # the same text in another project is its own
    assert other["merged"] is False
    store.remember("skill", "1. add the function to stats.py\n2. add a test", "textkit",
                   title="add-text-stat")
    found = store.recall("monkeypatch_env fixture in test_memory.py", "ragmcp")
    assert {f["project"] for f in found[:2]} == {"ragmcp", "textkit"}
    assert found[0]["kind"] == "lesson" and found[0]["score"] > 0.7
    mine = store.recall("monkeypatch_env fixture", "ragmcp", other_projects=False)
    assert {f["project"] for f in mine} == {"ragmcp"}
    skills = store.recall("add a test function stats", kinds=["skill"])
    assert skills[0]["title"] == "add-text-stat" and skills[0]["text"].startswith("1. add")
    assert store.recall("anything", min_score=0.99) == []


def test_grade_retires_failing_memories(store):
    mid = store.remember("lesson", "use split_whitespace for words", "rust-demo")["id"]
    for outcome in ("failed", "clean", "failed"):
        assert store.grade(mid, outcome)["retired"] is False
    assert store.grade(mid, "failed") == {"id": mid, "uses": 4, "retired": True}
    assert store.recall("split_whitespace words") == []
    with pytest.raises(ValueError):
        store.grade(mid, "great")
    assert store.stats()["by_kind"] == {"lessons (retired)": 1}


def test_validation_and_forget(store):
    with pytest.raises(ValueError, match="kind"):
        store.remember("rule", "x", "p")
    with pytest.raises(ValueError, match="empty"):
        store.remember("lesson", "  ", "p")
    mid = store.remember("note", "remember this", "p")["id"]
    store.forget(mid)
    assert store.stats()["total"] == 0


def test_persists(tmp_path):
    MemoryStore(tmp_path / "m", WordEmbedder()).remember("lesson", "kept across restarts", "p")
    assert MemoryStore(tmp_path / "m", WordEmbedder()).recall("kept across restarts")[0]["text"] == "kept across restarts"


def test_mcp_tools(tmp_path):
    import asyncio
    from fastmcp import Client
    server = build_server(MemoryStore(tmp_path / "m", WordEmbedder()))

    async def go():
        async with Client(server) as c:
            names = {t.name for t in await c.list_tools()}
            assert names == {"remember", "recall", "grade", "forget", "stats"}
            r = await c.call_tool("remember", {"kind": "lesson", "text": "x y z", "project": "p"})
            mid = json.loads(r.content[0].text)["id"]
            r = await c.call_tool("recall", {"query": "x y z"})
            assert json.loads(r.content[0].text)[0]["id"] == mid
    asyncio.run(go())
