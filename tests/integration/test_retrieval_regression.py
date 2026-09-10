"""Retrieval regression across the embedding-backend switch.

Queries used to be embedded by sentence-transformers on torch; they are now
embedded by ChromaDB's ONNX DefaultEmbeddingFunction — the same function that
embedded every stored chunk in the first place. The switch removes about 1.7 GB of
dependencies and is what lets the app fit a free-tier host.

`tests/fixtures/retrieval_baseline.json` holds the top-5 chunk ids and scores for a
fixed set of hydrogeology questions, captured with the OLD torch embedder before
the switch. These tests assert the new embedder returns the same chunks in the same
order. At the time of the switch all 12 queries matched exactly, with the largest
score change 6.9e-7.

They use the real committed vector store, so they are skipped where it is absent.
"""

import json
import sys
from pathlib import Path

import pytest

BASELINE = Path("tests/fixtures/retrieval_baseline.json")
VECTORDB = Path("data/vectordb/chroma.sqlite3")

pytestmark = pytest.mark.skipif(
    not VECTORDB.exists(), reason="knowledge-base vector store not present")


@pytest.fixture(scope="module")
def retriever():
    from src.core.config import load_config
    from src.retrieval.retriever import Retriever
    return Retriever(load_config())


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text())["queries"]


def _ids(hits):
    return [h["metadata"].get("chunk_id") for h in hits]


class TestSameResultsAsTheTorchEmbedder:

    def test_baseline_covers_every_aquifer_type(self, baseline):
        text = " ".join(baseline).lower()
        for term in ("theis", "leak", "unconfined", "fracture", "step-drawdown"):
            assert term in text, term

    def test_every_query_returns_the_same_top_five_in_order(self, retriever, baseline):
        mismatches = []
        for query, expected in baseline.items():
            got = _ids(retriever.retrieve(query, top_k=5, similarity_threshold=0.0))
            want = [e["chunk_id"] for e in expected]
            if got != want:
                mismatches.append((query, want, got))
        assert not mismatches, mismatches

    def test_scores_are_unchanged(self, retriever, baseline):
        """Cosine similarity against the same stored vectors; any drift beyond
        floating-point noise would mean the query embedding changed."""
        for query, expected in baseline.items():
            hits = retriever.retrieve(query, top_k=5, similarity_threshold=0.0)
            for hit, exp in zip(hits, expected):
                assert hit["score"] == pytest.approx(exp["score"], abs=1e-5), query


class TestTheHeavyStackIsGone:

    def test_the_embedder_is_onnx(self, retriever):
        from src.retrieval.embeddings import OnnxEmbeddings
        assert isinstance(retriever.embedding_model, OnnxEmbeddings)

    def test_a_real_retrieval_does_not_import_torch(self):
        """Run in a subprocess so modules already imported by other tests cannot
        mask a regression."""
        import subprocess

        code = (
            "import sys\n"
            "from src.core.config import load_config\n"
            "from src.retrieval.retriever import Retriever\n"
            "Retriever(load_config()).retrieve('Theis assumptions', top_k=3)\n"
            "heavy = {'torch','transformers','sentence_transformers',"
            "'langchain_huggingface'} & {m.split('.')[0] for m in sys.modules}\n"
            "print(sorted(heavy))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=300,
                             env={**__import__("os").environ,
                                  "GEMINI_API_KEY": "unused"})
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == "[]"

    def test_requirements_no_longer_pull_in_torch(self):
        reqs = Path("requirements.txt").read_text().lower()
        for pkg in ("sentence-transformers", "langchain-huggingface", "torch"):
            assert not any(line.strip().startswith(pkg)
                           for line in reqs.splitlines()), pkg


class TestEmbedderContract:

    def test_query_vectors_are_384_dimensional(self, retriever):
        """all-MiniLM-L6-v2 — the dimension the stored index was built with."""
        assert len(retriever.embedding_model.embed_query("drawdown")) == 384

    def test_a_different_model_is_refused_rather_than_substituted(self):
        """Querying with any other model would produce vectors incomparable with
        the stored index, so it must fail loudly."""
        from src.retrieval.embeddings import OnnxEmbeddings

        with pytest.raises(ValueError, match="incomparable"):
            OnnxEmbeddings("sentence-transformers/all-mpnet-base-v2")

    def test_the_configured_model_name_is_accepted(self):
        """The configs still say "sentence-transformers/all-MiniLM-L6-v2"."""
        from src.core.config import load_config
        from src.retrieval.embeddings import embeddings_manager

        model = embeddings_manager.get_embeddings(
            load_config().retrieval.embedding_model)
        assert model.model_name == "all-MiniLM-L6-v2"
