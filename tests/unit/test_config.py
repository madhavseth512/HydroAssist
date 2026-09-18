"""Every configuration the repository ships must be one the app was tested with.

`configs/` once held `production.yaml` and `development.yaml`. Nothing loaded
them, nothing tested them, and they disagreed with `default.yaml` on the
retrieval similarity threshold — 0.75 against 0.2 — which would have quietly cut
the sources the consultant finds. Deploying with `HYDROASSIST_ENV=production`
would have looked like a broken knowledge base, not a configuration change.

These checks keep that from coming back: a config file may exist only if it
loads and agrees with the settings the suite actually ran against.
"""

from pathlib import Path

import pytest
import yaml

from src.core.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# What the whole suite — including the retrieval regression — ran against.
TESTED_THRESHOLD = 0.2
TESTED_TOP_K = 5
TESTED_STORE = "./data/vectordb"


# The sections an application config defines. `configs/` also holds data that is
# not configuration — corpus_methods.yaml, which says what each paper is about —
# and feeding that to load_config() would produce an AppConfig of pure defaults
# rather than an error.
APP_SECTIONS = {"retrieval", "vectorstore", "llm", "ingestion", "logging"}


def _config_names():
    """The files in configs/ that are application configuration."""
    names = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and APP_SECTIONS & set(data):
            names.append(path.stem)
    return names


class TestWhatShips:

    def test_default_is_present(self):
        assert "default" in _config_names()

    def test_no_untested_variants(self):
        """A new application config is only safe once it is in TESTED_* below.
        Data files in configs/ — the corpus manifest — are not configuration and
        are not counted here; `tests/unit/test_corpus_methods.py` covers that."""
        assert _config_names() == ["default"], (
            "configs/ holds a variant the suite does not exercise; either test it "
            "or delete it")


class TestEveryConfigLoads:

    @pytest.mark.parametrize("name", _config_names())
    def test_it_parses_into_the_dataclasses(self, name):
        config = load_config(name)
        assert config.retrieval.top_k > 0
        assert config.vectorstore.collection_name
        assert config.llm.provider

    @pytest.mark.parametrize("name", _config_names())
    def test_its_vector_store_exists(self, name):
        """A config pointing at a directory that was never built retrieves
        nothing, and the consultant reports that as missing knowledge."""
        path = (CONFIG_DIR.parent / load_config(name).vectorstore.persist_directory)
        assert path.is_dir(), f"{name}.yaml points at {path}, which does not exist"


class TestTheDefaultMatchesWhatWasTested:

    def test_retrieval_settings(self):
        retrieval = load_config("default").retrieval
        assert retrieval.similarity_threshold == TESTED_THRESHOLD
        assert retrieval.top_k == TESTED_TOP_K

    def test_the_vector_store_is_the_committed_one(self):
        assert load_config("default").vectorstore.persist_directory == TESTED_STORE

    def test_an_unknown_name_says_so(self):
        with pytest.raises(FileNotFoundError, match="production"):
            load_config("production")
