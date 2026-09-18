"""The manifest that says which method each paper is about.

Nearly every chunk in the knowledge base was tagged "multiple", so a search
could not be narrowed to the papers discussing the method in question. The
manifest fixes that per document, and these checks keep it honest: a tag that
does not match the calculator's own method id would filter to nothing, and a
paper nobody tagged would quietly stop being findable by method.
"""

from pathlib import Path

import pytest

from src.calculator import registry
from src.ingestion import corpus

KNOWLEDGE_SOURCE = Path(__file__).resolve().parents[2] / "knowledge source"


@pytest.fixture(scope="module")
def manifest():
    return corpus.load_manifest()


class TestTheVocabulary:

    def test_the_implemented_ids_are_exactly_the_calculator_s(self, manifest):
        """A tag of 'theis-1935' or 'Theis_1935' would silently match nothing
        when the workspace filters by the method being run."""
        assert sorted(manifest.implemented) == sorted(s.id for s in registry.all_specs())

    def test_no_paper_uses_an_undeclared_id(self, manifest):
        assert manifest.unknown_ids() == {}

    def test_ids_are_lower_case_with_underscores(self, manifest):
        for method in manifest.vocabulary:
            assert method == method.lower().strip()
            assert " " not in method and "-" not in method

    def test_the_two_halves_do_not_overlap(self, manifest):
        assert not set(manifest.implemented) & set(manifest.other)


class TestCoverage:

    def test_every_paper_in_the_corpus_is_named(self, manifest):
        """A paper added to `knowledge source/` without a manifest entry keeps
        the old 'multiple' tag, so this fails rather than letting it disappear
        from method searches."""
        if not KNOWLEDGE_SOURCE.exists():
            pytest.skip("knowledge source/ is not present in this checkout")

        on_disk = {p.name for p in KNOWLEDGE_SOURCE.rglob("*.pdf")}
        missing = sorted(on_disk - set(manifest.papers))

        assert not missing, f"not in configs/corpus_methods.yaml: {missing}"

    def test_the_manifest_names_no_paper_that_is_gone(self, manifest):
        if not KNOWLEDGE_SOURCE.exists():
            pytest.skip("knowledge source/ is not present in this checkout")

        on_disk = {p.name for p in KNOWLEDGE_SOURCE.rglob("*.pdf")}
        assert not sorted(set(manifest.papers) - on_disk)

    def test_every_paper_has_at_least_one_method(self, manifest):
        for name, entry in manifest.papers.items():
            assert entry.get("methods"), name

    @pytest.mark.parametrize("method_id", [
        "theis_1935", "theis_recovery_1935", "cooper_jacob_1946",
        "papadopulos_cooper_1967", "hantush_jacob_1955", "warren_root_1963",
    ])
    def test_each_implemented_method_has_a_paper_behind_it(self, manifest, method_id):
        """The consultant is meant to justify a method from the literature; a
        method with no tagged paper cannot be."""
        tagged = [name for name, entry in manifest.papers.items()
                  if method_id in entry.get("methods", [])]
        assert tagged, f"no paper tagged {method_id}"

    def test_boulton_is_covered_only_at_second_hand(self, manifest):
        """Boulton (1963) is implemented, but his own paper is not in the
        corpus: the consultant can cite only Neuman's 1979 perspective on
        delayed yield and Moench's 1995 combination of the two models.

        Recorded so the gap stays deliberate. If Boulton (1963) is ever added,
        this fails and the list gets updated.
        """
        tagged = sorted(name for name, entry in manifest.papers.items()
                        if "boulton_1963" in entry.get("methods", []))

        assert tagged == [
            "Groundwater - May 1995 - Moench - Combining the Neuman and Boulton "
            "Models for Flow to a Well in an Unconfined Aquifer.pdf",
            "Water Resources Research - August 1979 - Neuman - Perspective on  "
            "Delayed yield.pdf",
        ]


class TestLookups:

    def test_a_named_paper_gives_its_methods(self, manifest):
        methods = manifest.methods_for(Path("knowledge source/Confined Aquifer/Theis-1935.pdf"))
        assert methods == ["theis_1935"]

    def test_an_unnamed_paper_gives_nothing_rather_than_a_guess(self, manifest):
        assert manifest.methods_for(Path("knowledge source/General/new-paper.pdf")) == []

    def test_the_folder_decides_the_aquifer_unless_the_manifest_says_otherwise(self, manifest):
        theis = Path("knowledge source/Confined Aquifer/Theis-1935.pdf")
        assert manifest.aquifer_for(theis) == "confined"

        # Filed in two folders; the manifest pins it rather than the read order.
        general = Path("knowledge source/General/wri99-4225.pdf")
        unconfined = Path("knowledge source/Unconfined Aquifer/wri99-4225.pdf")
        assert manifest.aquifer_for(general) == manifest.aquifer_for(unconfined) == "unconfined"

    def test_a_missing_manifest_is_not_fatal(self, tmp_path):
        empty = corpus.load_manifest(tmp_path / "absent.yaml")

        assert empty.papers == {}
        assert empty.methods_for(Path("anything.pdf")) == []
        assert empty.aquifer_for(Path("knowledge source/General/x.pdf")) == "general"

    def test_the_scanned_papers_are_tagged_with_what_they_cover(self, manifest):
        """Both were absent from the index until OCR; tagging is what makes them
        findable now."""
        agarwal = manifest.methods_for(
            Path("knowledge source/General/52485_SPE-9289-Agarwal-buildup.pdf"))
        assert "theis_recovery_1935" in agarwal


class TestWhatReachesTheIndex:

    def _pages(self):
        return [{"page": 1, "text": "Hantush. " * 50, "has_equations": True, "ocr": False}]

    def test_the_method_field_lists_them_for_a_reader(self):
        documents = corpus.build_documents(
            Path("knowledge source/Leaky Confined Aquifer/hantush.pdf"), self._pages(),
            methods=["hantush_jacob_1955", "hantush_1960"])

        assert documents[0]["metadata"]["method"] == "hantush_jacob_1955,hantush_1960"

    def test_each_method_also_becomes_a_flag_that_a_filter_can_match(self):
        """ChromaDB matches metadata values exactly, so the comma-joined string
        could never match a search for one method. The flags can."""
        documents = corpus.build_documents(
            Path("knowledge source/Leaky Confined Aquifer/hantush.pdf"), self._pages(),
            methods=["hantush_jacob_1955", "hantush_1960"])
        metadata = documents[0]["metadata"]

        assert metadata["method_hantush_jacob_1955"] is True
        assert metadata["method_hantush_1960"] is True

    def test_an_untagged_paper_keeps_the_old_value_and_gains_no_flags(self):
        documents = corpus.build_documents(
            Path("knowledge source/General/unknown.pdf"), self._pages())
        metadata = documents[0]["metadata"]

        assert metadata["method"] == corpus.UNTAGGED
        assert not [key for key in metadata if key.startswith(corpus.METHOD_FLAG_PREFIX)]

    def test_the_flags_are_plain_scalars(self):
        """ChromaDB rejects a list, and a nested value would fail at write time
        after an hour of embedding."""
        documents = corpus.build_documents(
            Path("knowledge source/Confined Aquifer/theis.pdf"), self._pages(),
            methods=["theis_1935"])

        for value in documents[0]["metadata"].values():
            assert isinstance(value, (str, int, float, bool))
