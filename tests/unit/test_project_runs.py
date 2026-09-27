"""The log of what was run against a test.

`analyses` holds the fits deliberately saved for comparison, with everything
needed to trust them later. This is the lighter record of what was *run*, and it
exists because those were two different questions and only one of them had an
answer: a project with two tests showed an analysis for neither, because a result
lived for one session and one test.
"""

import pandas as pd
import pytest

from src.project.model import Project, Readings, RunRecord, RUNS_KEPT


def readings(n=5, offset=0.0):
    return Readings.from_frame(pd.DataFrame({
        "time_s": [60.0 * (i + 1) for i in range(n)],
        "drawdown_m": [0.1 * (i + 1) + offset for i in range(n)],
    }))


def fit(T_day=215.29, S=5e-5, method_id="theis_1935", success=True):
    return {"success": success, "T_day": T_day, "S": S, "r_squared": 0.999,
            "rmse": 0.002, "method_id": method_id}


@pytest.fixture
def test_with_readings():
    project = Project()
    test = project.add_test()
    test.set_readings(readings())
    return test


class TestRecordingARun:

    def test_a_run_is_kept_with_its_numbers(self, test_with_readings):
        test_with_readings.record_run(fit(), "Theis (1935)")

        run = test_with_readings.runs[0]
        assert run.method_label == "Theis (1935)"
        assert run.T_day == 215.29
        assert run.n_points == 5
        assert run.readings_fingerprint == test_with_readings.readings.fingerprint

    def test_runs_are_kept_in_order(self, test_with_readings):
        test_with_readings.record_run(fit(), "Theis (1935)")
        test_with_readings.record_run(fit(64.11), "Cooper-Jacob (1946)")

        assert [run.method_label for run in test_with_readings.runs] == [
            "Theis (1935)", "Cooper-Jacob (1946)"]

    def test_re_running_a_method_replaces_its_entry(self, test_with_readings):
        """Two rows for one method invite a comparison that is not one."""
        test_with_readings.record_run(fit(215.29), "Theis (1935)")
        test_with_readings.record_run(fit(216.00), "Theis (1935)")

        assert len(test_with_readings.runs) == 1
        assert test_with_readings.runs[0].T_day == 216.00

    def test_the_same_method_on_new_readings_does_not_replace_anything(
            self, test_with_readings):
        """Different data, different result — both are true."""
        test_with_readings.record_run(fit(215.29), "Theis (1935)")
        test_with_readings.set_readings(readings(offset=0.05))
        test_with_readings.record_run(fit(180.0), "Theis (1935)")

        assert len(test_with_readings.runs) == 2

    def test_a_failed_fit_is_recorded_as_one(self, test_with_readings):
        test_with_readings.record_run(
            {"success": False, "error_message": "did not converge"},
            "Papadopulos-Cooper (1967)")

        run = test_with_readings.runs[0]
        assert run.success is False
        assert "converge" in run.error_message

    def test_the_log_does_not_grow_without_bound(self, test_with_readings):
        for index in range(RUNS_KEPT + 5):
            test_with_readings.record_run(fit(), f"Method {index}")

        assert len(test_with_readings.runs) == RUNS_KEPT
        assert test_with_readings.runs[-1].method_label == \
            f"Method {RUNS_KEPT + 4}"

    def test_a_run_without_readings_is_still_recorded(self):
        """A fit cannot happen without readings, but nothing here should raise if
        one is recorded oddly — the log is not a gate."""
        test = Project().add_test()

        test.record_run(fit(), "Theis (1935)")

        assert test.runs[0].readings_fingerprint == ""


class TestWhichRunsStillDescribeTheData:

    def test_runs_on_the_current_readings_are_offered(self, test_with_readings):
        test_with_readings.record_run(fit(), "Theis (1935)")

        assert len(test_with_readings.current_runs()) == 1

    def test_runs_on_replaced_readings_are_kept_but_not_offered(
            self, test_with_readings):
        """They happened, so the file keeps them; they do not describe this data,
        so they are not shown beside it."""
        test_with_readings.record_run(fit(), "Theis (1935)")
        test_with_readings.set_readings(readings(offset=0.05))

        assert len(test_with_readings.runs) == 1
        assert test_with_readings.current_runs() == []

    def test_a_test_with_no_readings_offers_none(self):
        test = Project().add_test()
        test.record_run(fit(), "Theis (1935)")

        assert test.current_runs() == []


class TestThroughTheProjectFile:
    """Which is what makes any of this survive a reload."""

    def test_runs_round_trip(self, test_with_readings):
        test_with_readings.record_run(fit(), "Theis (1935)")
        test_with_readings.record_run(fit(64.11), "Cooper-Jacob (1946)")
        project = Project(tests=[test_with_readings])

        reopened = Project.from_json(project.to_json())

        assert [(run.method_label, run.T_day)
                for run in reopened.tests[0].runs] == [
            ("Theis (1935)", 215.29), ("Cooper-Jacob (1946)", 64.11)]

    def test_a_file_written_before_runs_existed_still_opens(self):
        """The field is additive; an older project simply has none."""
        project = Project()
        project.add_test()
        data = project.to_dict()
        for test in data["tests"]:
            del test["runs"]

        reopened = Project.from_dict(data)

        assert reopened.tests[0].runs == []

    def test_a_damaged_run_is_reported_like_any_other_damage(self):
        from src.project.model import ProjectFileError

        project = Project()
        project.add_test()
        data = project.to_dict()
        data["tests"][0]["runs"] = [{"not": "a run"}]

        with pytest.raises(ProjectFileError):
            Project.from_dict(data)

    def test_the_fingerprint_travels_with_the_run(self, test_with_readings):
        """Without it a reopened project could not tell which runs still describe
        its readings."""
        test_with_readings.record_run(fit(), "Theis (1935)")
        project = Project(tests=[test_with_readings])

        reopened = Project.from_json(project.to_json())

        assert len(reopened.tests[0].current_runs()) == 1
