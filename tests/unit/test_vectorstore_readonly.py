"""Read-only access to the committed vector store.

The knowledge base is committed to git so a hosted deploy ships with it. But the
first chromadb PersistentClient a process opens on a path writes a lock row into
that store's `acquire_write` table, even if nothing is ever added — measured on
the real store, where one app start changed chroma.sqlite3. Reading it in place
would dirty a 33 MB tracked binary on every run.

VectorStoreManager(read_only=True) opens a private per-process snapshot instead.

Why the key tests use subprocesses: chromadb caches one client per path per
process, and only the FIRST client for a path writes. An in-process test that has
already touched the store would see no change whether or not read-only mode
worked — it would pass vacuously. A fresh process per open is also exactly what a
real app start is. The writable-mode contrast test exists to prove the harness
can detect a write at all.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.retrieval import vectorstore as vs
from src.retrieval.vectorstore import VectorStoreManager

COLLECTION = "test_kb"
REPO = Path(__file__).resolve().parents[2]


def _digest(directory: Path) -> dict:
    return {str(p.relative_to(directory)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _run(code: str) -> str:
    """Run `code` in a fresh interpreter from the repo root; return its stdout."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env={**os.environ, "GEMINI_API_KEY": "unused",
             "PYTHONIOENCODING": "utf-8"},
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


@pytest.fixture
def source(tmp_path):
    """A tiny real chromadb store with explicit 3-d vectors, built in a separate
    process so this test process never holds a client on it."""
    path = tmp_path / "store"
    _run(f"""
        import chromadb
        from chromadb.config import Settings
        c = chromadb.PersistentClient(path={str(path)!r},
                settings=Settings(anonymized_telemetry=False, allow_reset=True))
        col = c.get_or_create_collection({COLLECTION!r},
                                         metadata={{"hnsw:space": "cosine"}})
        col.add(ids=["a", "b", "c"], documents=["theis", "hantush", "boulton"],
                embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                metadatas=[{{"m": 1}}, {{"m": 2}}, {{"m": 3}}])
    """)
    vs._SNAPSHOTS.clear()
    return path


def _open_in_fresh_process(path: Path, read_only: bool) -> list:
    """Open the store the way the app does at start-up, query it, report ids."""
    out = _run(f"""
        import json
        from src.retrieval.vectorstore import VectorStoreManager
        s = VectorStoreManager({str(path)!r}, {COLLECTION!r}, read_only={read_only})
        r = s.query(query_embeddings=[[1, 0, 0]], n_results=1)
        print(json.dumps(r["ids"][0]))
    """)
    return json.loads(out.strip().splitlines()[-1])


class TestTheSourceIsNeverWritten:

    def test_read_only_open_leaves_the_source_byte_identical(self, source):
        before = _digest(source)
        ids = _open_in_fresh_process(source, read_only=True)

        assert ids == ["a"], "the snapshot must contain the real data"
        assert _digest(source) == before

    def test_writable_open_by_contrast_does_write_to_the_source(self, source):
        """Proves the harness can see a write at all, so the test above cannot
        pass vacuously. This is the behaviour read-only mode exists to avoid."""
        before = _digest(source)
        _open_in_fresh_process(source, read_only=False)

        assert _digest(source) != before

    def test_read_only_mode_opens_a_different_directory(self, source):
        store = VectorStoreManager(str(source), COLLECTION, read_only=True)

        assert store.persist_directory != source
        assert store.source_directory == source


class TestSnapshotLifecycle:

    def test_the_snapshot_is_reused_within_a_process(self, source):
        a = VectorStoreManager(str(source), COLLECTION, read_only=True)
        b = VectorStoreManager(str(source), COLLECTION, read_only=True)

        assert a.persist_directory == b.persist_directory

    def test_the_snapshot_is_refreshed_when_the_source_changes(self, source):
        """If ingestion rewrites the store, later readers must see the new data."""
        import time

        first = VectorStoreManager(str(source), COLLECTION, read_only=True)
        later = time.time() + 5
        os.utime(source / "chroma.sqlite3", (later, later))

        second = VectorStoreManager(str(source), COLLECTION, read_only=True)
        assert second.persist_directory != first.persist_directory

    def test_a_missing_source_is_not_created(self, tmp_path):
        """A read path must never create an empty store in the repository."""
        missing = tmp_path / "no_store_here"
        vs._SNAPSHOTS.clear()

        VectorStoreManager(str(missing), COLLECTION, read_only=True)
        assert not missing.exists()


class TestWritesAreRefused:
    """Writing to a throwaway snapshot would appear to succeed and then vanish."""

    def test_add_documents_is_refused(self, source):
        store = VectorStoreManager(str(source), COLLECTION, read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            store.add_documents(["x"], [{"m": 9}], ["x"])

    def test_reset_is_refused(self, source):
        store = VectorStoreManager(str(source), COLLECTION, read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            store.reset_collection()


class TestTheAppReadsReadOnly:

    def test_the_retriever_opens_the_store_read_only(self):
        from src.core.config import load_config
        from src.retrieval.retriever import Retriever

        assert Retriever(load_config()).vectorstore.read_only is True

    @pytest.mark.skipif(not (REPO / "data/vectordb/chroma.sqlite3").exists(),
                        reason="committed store not present")
    def test_an_app_start_leaves_the_committed_store_untouched(self):
        """The whole point: a fresh process running real retrievals — which is what
        starting the app does — must not change the committed files."""
        committed = REPO / "data/vectordb"
        before = _digest(committed)

        _run("""
            from src.core.config import load_config
            from src.retrieval.retriever import Retriever
            r = Retriever(load_config())
            for q in ("Theis assumptions", "leakage factor", "delayed yield"):
                r.retrieve(q, top_k=5)
        """)

        assert _digest(committed) == before
