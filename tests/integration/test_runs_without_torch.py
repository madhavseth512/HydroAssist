"""The app must run with torch and transformers absent.

The deploy target (Streamlit Community Cloud) installs only requirements.txt,
which no longer includes sentence-transformers, torch or transformers. But a
developer's local venv may still have them from before, and one dependency imports
them opportunistically: langchain_core/language_models/base.py does
`from transformers import GPT2TokenizerFast` inside a try/except, for a
token-counting fallback, and transformers in turn probes for torch. Locally that
loads ~210 MB of torch for nothing; on the host it is simply skipped.

So "does anything import torch" is the wrong question to test locally — the answer
depends on what happens to be installed. The right question is whether the app
still works when torch and transformers CANNOT be imported. These tests block both
at the import system, in a fresh subprocess, and then build and use the real
components. That makes the check meaningful on any machine, whether or not torch
is installed.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Installed as the first meta-path finder, so any attempt to import these
# packages — directly or transitively — raises ImportError exactly as it would on
# a host where they are not installed.
_BLOCK_TORCH = """
import sys, importlib.abc
BLOCKED = ("torch", "transformers", "sentence_transformers", "langchain_huggingface")
class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked to simulate a torch-free host")
        return None
sys.meta_path.insert(0, _Absent())
"""


def _run_without_torch(body: str) -> str:
    code = _BLOCK_TORCH + textwrap.dedent(body)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
        timeout=600,
        env={**os.environ, "GEMINI_API_KEY": "unused", "PYTHONIOENCODING": "utf-8"},
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


class TestTheAppWorksWithoutTorch:

    def test_the_blocker_actually_blocks(self):
        """Guards the harness: if the blocker failed open, every test below would
        pass for the wrong reason."""
        out = _run_without_torch("""
            try:
                import torch
                print("IMPORTED")
            except ImportError:
                print("BLOCKED")
        """)
        assert out.strip().splitlines()[-1] == "BLOCKED"

    def test_the_chat_graph_builds(self):
        """The import chain that opportunistically reaches for transformers."""
        out = _run_without_torch("""
            from src.core.config import load_config
            from src.core.graph import HydroAssistChat
            HydroAssistChat(load_config())
            print("OK")
        """)
        assert out.strip().splitlines()[-1] == "OK"

    @pytest.mark.skipif(not (REPO / "data/vectordb/chroma.sqlite3").exists(),
                        reason="committed store not present")
    def test_retrieval_returns_results(self):
        out = _run_without_torch("""
            from src.core.config import load_config
            from src.retrieval.retriever import Retriever
            hits = Retriever(load_config()).retrieve(
                "What are the assumptions of the Theis method?", top_k=5)
            print(len(hits))
        """)
        assert int(out.strip().splitlines()[-1]) == 5

    def test_every_calculator_runs(self):
        out = _run_without_torch("""
            import pandas as pd
            from src.calculator import registry
            from src.calculator.base import CalculationInput
            df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
            ok = 0
            for spec in registry.all_specs():
                kw = dict(df=df, Q=500 / 86400, r=50.0)
                if spec.data_source == registry.PUMPED_WELL:
                    kw.update(r=0.30, r_w=0.30, r_c=0.20)
                ok += spec.build().calculate(CalculationInput(**kw)).success
            print(ok, len(registry.all_specs()))
        """)
        ok, total = map(int, out.strip().splitlines()[-1].split())
        assert ok == total

    def test_the_pdf_report_generates(self):
        out = _run_without_torch("""
            import pandas as pd
            from src.calculator.base import CalculationInput
            from src.calculator.report import generate_pdf
            from src.calculator.theis import TheisCalculator
            df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
            res = TheisCalculator().calculate(
                CalculationInput(df=df, Q=500 / 86400, r=50.0))
            print(generate_pdf(res, 500 / 86400, 50.0, "Theis (1935)", df)[:5])
        """)
        assert "%PDF-" in out
