"""A shared memory service over MCP: lessons and skills that coding agents learned.

localcoder learns two kinds of things in a project: lessons (one line about a fix,
.localcoder/lessons.md) and skills (a procedure that worked, .localcoder/skills/). Kept
per project, a lesson about pytest fixtures learned in one project never helps another.
This server keeps them in one place for every project and every machine that points at
it, and finds them by meaning (embeddings), not only by shared words.

Tools (all return JSON text):
- remember(kind, text, project, title, files): store a lesson, skill or note; a
  near-identical one from the same project is updated instead of added twice.
- recall(query, project, kinds, limit, other_projects, min_score): the memories closest to
  the query, best first, with their grades. Retired ones are left out.
- grade(id, outcome): "clean", "fixed" or "failed" for a request that used the memory;
  one that goes with mostly failures (3 of 4 or worse) is retired.
- forget(id), stats().

Storage: a persistent Chroma collection in $RAGMCP_MEMORY_DIR (default
~/.local/share/ragmcp/memory), embeddings from sentence-transformers all-MiniLM-L6-v2.
Run it as an MCP server over stdio: `ragmcp-memory` (or `python -m ragmcp.memory_service`).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

KINDS = ("lesson", "skill", "note")
OUTCOMES = ("clean", "fixed", "failed")
SAME = 0.92            # cosine similarity above which a new memory updates an old one
RETIRE_AFTER = 4
RETIRE_FAIL_RATE = 0.5


def default_dir() -> Path:
    base = os.environ.get("RAGMCP_MEMORY_DIR")
    if base:
        return Path(base).expanduser()
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data / "ragmcp" / "memory"


class SentenceEmbedder:
    """all-MiniLM-L6-v2, loaded on first use (a few seconds)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.name = model
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


class MemoryStore:
    def __init__(self, path: Path | None = None, embedder=None):
        import chromadb
        from chromadb.config import Settings
        self.path = Path(path) if path else default_dir()
        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.path),
                                           settings=Settings(anonymized_telemetry=False))
        self.col = client.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"})
        self.embedder = embedder or SentenceEmbedder()

    # -- writing ---------------------------------------------------------

    def remember(self, kind: str, text: str, project: str, title: str = "",
                 files: list[str] | None = None) -> dict:
        kind = kind.strip().lower()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        text = text.strip()
        if not text:
            raise ValueError("text is empty")
        doc = f"{title}\n{text}" if title else text
        vec = self.embedder.encode([doc])[0]
        now = time.time()
        if self.col.count():
            near = self.col.query(query_embeddings=[vec], n_results=1,
                                  where={"$and": [{"kind": kind}, {"project": project}]})
            if near["ids"][0] and 1 - near["distances"][0][0] >= SAME:
                mid = near["ids"][0][0]
                meta = dict(near["metadatas"][0][0])
                meta.update(title=title or meta.get("title", ""), updated=now,
                            files=",".join(files or []) or meta.get("files", ""))
                self.col.update(ids=[mid], documents=[doc], embeddings=[vec], metadatas=[meta])
                return {"id": mid, "merged": True}
        mid = uuid.uuid4().hex[:16]
        meta = {"kind": kind, "project": project, "title": title, "files": ",".join(files or []),
                "created": now, "updated": now, "uses": 0, "clean": 0, "fixed": 0, "failed": 0,
                "retired": False}
        self.col.add(ids=[mid], documents=[doc], embeddings=[vec], metadatas=[meta])
        return {"id": mid, "merged": False}

    def grade(self, id: str, outcome: str) -> dict:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
        got = self.col.get(ids=[id])
        if not got["ids"]:
            raise ValueError(f"no memory {id}")
        meta = dict(got["metadatas"][0])
        meta["uses"] = int(meta.get("uses", 0)) + 1
        meta[outcome] = int(meta.get(outcome, 0)) + 1
        if meta["uses"] >= RETIRE_AFTER and meta["failed"] / meta["uses"] >= RETIRE_FAIL_RATE:
            meta["retired"] = True
        self.col.update(ids=[id], metadatas=[meta])
        return {"id": id, "uses": meta["uses"], "retired": meta["retired"]}

    def forget(self, id: str) -> dict:
        self.col.delete(ids=[id])
        return {"id": id, "forgotten": True}

    # -- reading ---------------------------------------------------------

    def recall(self, query: str, project: str = "", kinds: list[str] | None = None,
               limit: int = 5, other_projects: bool = True, min_score: float = 0.0) -> list[dict]:
        if not self.col.count() or not query.strip():
            return []
        conds = [{"retired": False}]
        if kinds:
            conds.append({"kind": {"$in": list(kinds)}})
        if project and not other_projects:
            conds.append({"project": project})
        where = conds[0] if len(conds) == 1 else {"$and": conds}
        vec = self.embedder.encode([query])[0]
        res = self.col.query(query_embeddings=[vec], n_results=max(1, min(limit * 3, 50)), where=where)
        out = []
        for mid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                        res["distances"][0]):
            score = round(1 - dist, 3)
            if score < min_score:
                continue
            title = meta.get("title", "")
            text = doc[len(title) + 1:] if title and doc.startswith(title + "\n") else doc
            out.append({"id": mid, "kind": meta["kind"], "project": meta["project"], "title": title,
                        "text": text, "files": [f for f in meta.get("files", "").split(",") if f],
                        "score": score, "uses": meta.get("uses", 0), "clean": meta.get("clean", 0),
                        "fixed": meta.get("fixed", 0), "failed": meta.get("failed", 0)})
            if len(out) >= limit:
                break
        return out

    def stats(self) -> dict:
        got = self.col.get(include=["metadatas"])
        by: dict = {}
        for meta in got["metadatas"]:
            key = f"{meta['kind']}s" + (" (retired)" if meta.get("retired") else "")
            by[key] = by.get(key, 0) + 1
        projects = sorted({m["project"] for m in got["metadatas"]})
        return {"path": str(self.path), "total": len(got["ids"]), "by_kind": by,
                "projects": projects}


def build_server(store: MemoryStore | None = None):
    from fastmcp import FastMCP
    mcp = FastMCP("ragmcp-memory")
    state = {"store": store}

    def s() -> MemoryStore:
        if state["store"] is None:
            state["store"] = MemoryStore()
        return state["store"]

    @mcp.tool
    def remember(kind: str, text: str, project: str, title: str = "",
                 files: list[str] | None = None) -> str:
        """Store a lesson, skill or note a coding agent learned in a project. A near-identical
        one from the same project is updated instead of added twice. Returns {id, merged}."""
        return json.dumps(s().remember(kind, text, project, title, files))

    @mcp.tool
    def recall(query: str, project: str = "", kinds: list[str] | None = None, limit: int = 5,
               other_projects: bool = True, min_score: float = 0.0) -> str:
        """The stored memories closest in meaning to the query, best first, as a JSON list of
        {id, kind, project, title, text, files, score, uses, clean, fixed, failed}."""
        return json.dumps(s().recall(query, project, kinds, limit, other_projects, min_score))

    @mcp.tool
    def grade(id: str, outcome: str) -> str:
        """Record how a request that used this memory ended: clean, fixed or failed."""
        return json.dumps(s().grade(id, outcome))

    @mcp.tool
    def forget(id: str) -> str:
        """Delete a memory."""
        return json.dumps(s().forget(id))

    @mcp.tool
    def stats() -> str:
        """How many memories of each kind, and from which projects."""
        return json.dumps(s().stats())

    return mcp


def main() -> None:
    build_server().run()       # stdio


if __name__ == "__main__":
    main()
