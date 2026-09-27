"""Running, matching, saving, comparing and reporting over HTTP.

The behaviours the Streamlit AppTests covered for Steps 2-5, plus a
registry-driven pass that runs every method through the API — the API-level
equivalent of the conformance suite, so a new method cannot reach the browser
broken.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import PC_SAMPLE, RECOVERY_SAMPLE, THEIS_SAMPLE, run, setup_payload, upload

from src.calculator import registry

RECOVERY_SETUP = dict(test_type="recovery", t_p_hours=24.0)
PC_SETUP = dict(q_value=300.0, r_m=0.30, r_w_m=0.30, r_c_m=0.20)


def _leaky_csv() -> bytes:
    from src.calculator.methods.hantush_jacob import hantush_jacob_drawdown

    t = np.logspace(1, 6, 70)
    s = hantush_jacob_drawdown(t, 2.5e-3, 5e-5, 300.0, 500 / 86400, 50.0)
    s = s + np.random.default_rng(2).normal(0, 0.002, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s}).to_csv(index=False).encode()


def _upload_bytes(client, content: bytes, name="data.csv"):
    test_id = client.get("/api/project").json()["active_test_id"]
    return client.post(f"/api/tests/{test_id}/readings",
                       files={"file": (name, content, "text/csv")})


class TestRunning:

    def test_fits_the_active_tests_readings(self, client):
        upload(client)
        body = run(client).json()

        assert body["success"] is True
        assert body["T_day"] == pytest.approx(215.29, rel=1e-3)
        assert body["estimates_storativity"] is True
        assert [p["name"] for p in body["parameters"]] == ["T", "S"]
        assert body["figure"].startswith("{")
        assert body["gof"]["aicc"] is not None

    def test_the_setup_is_stored_on_the_test(self, client):
        upload(client)
        run(client, q_value=300.0, r_m=42.0)

        setup = client.get("/api/project").json()["tests"][0]["setup"]
        assert setup["q_value"] == 300.0 and setup["r_m"] == 42.0

    def test_without_readings_it_says_what_to_do(self, client):
        response = run(client)

        assert response.status_code == 400
        assert "Load readings" in response.json()["detail"]

    def test_an_unknown_method_is_rejected(self, client):
        upload(client)
        assert run(client, method="not_a_method").status_code == 400

    def test_a_failed_fit_is_reported_not_raised(self, client):
        """Too few points to constrain the model: a result that explains itself,
        not a 500."""
        # Below the ten-point minimum a fit is not meaningfully constrained.
        _upload_bytes(client, pd.read_csv(THEIS_SAMPLE).head(8)
                      .to_csv(index=False).encode())
        body = run(client).json()

        assert body["success"] is False
        assert body["error_message"]

    def test_a_recovery_fit_reports_no_storativity(self, client):
        upload(client, RECOVERY_SAMPLE)
        body = run(client, method="theis_recovery_1935", **RECOVERY_SETUP).json()

        assert body["success"] is True
        assert body["T"] == pytest.approx(2.5e-3, rel=0.03)
        assert body["S"] is None and body["estimates_storativity"] is False
        assert [p["name"] for p in body["parameters"]] == ["T", "S_ratio"]

    def test_recovery_without_a_duration_is_explained(self, client):
        upload(client, RECOVERY_SAMPLE)
        response = run(client, method="theis_recovery_1935", test_type="recovery")

        assert response.status_code == 400
        assert "pumping duration" in response.json()["detail"]

    def test_the_last_result_survives_a_reload(self, client):
        assert client.get("/api/analyses/current").status_code == 404

        upload(client)
        expected = run(client).json()["T"]

        assert client.get("/api/analyses/current").json()["T"] == expected


class TestManualMatching:

    def test_matching_needs_a_result_first(self, client):
        response = client.post("/api/analyses/manual-match", json={"T": 1e-3})
        assert response.status_code == 400

    def test_at_the_fitted_values_it_reproduces_the_fit(self, client):
        upload(client)
        result = run(client).json()

        body = client.post("/api/analyses/manual-match",
                           json={"T": result["T"], "S": result["S"]}).json()

        assert body["rmse_mm"] == pytest.approx(body["auto_rmse_mm"], rel=1e-3)
        assert body["figure"].startswith("{")

    def test_away_from_the_fit_it_is_worse(self, client):
        upload(client)
        result = run(client).json()

        at_fit = client.post("/api/analyses/manual-match",
                             json={"T": result["T"], "S": result["S"]}).json()
        off = client.post("/api/analyses/manual-match",
                          json={"T": result["T"] * 4, "S": result["S"]}).json()

        assert off["rmse_mm"] > at_fit["rmse_mm"]

    def test_a_method_without_storativity_matches_on_T_alone(self, client):
        upload(client, RECOVERY_SAMPLE)
        result = run(client, method="theis_recovery_1935", **RECOVERY_SETUP).json()

        body = client.post("/api/analyses/manual-match",
                           json={"T": result["T"]}).json()

        assert body["rmse_mm"] == pytest.approx(body["auto_rmse_mm"], rel=1e-3)


class TestSavingAndComparing:

    def test_a_saved_analysis_is_ranked(self, client):
        upload(client)
        run(client)

        body = client.post("/api/analyses/save").json()

        assert len(body["rows"]) == 1
        assert body["rows"][0]["rank"] == 1
        assert body["rows"][0]["evidence"] == "best"
        assert body["rows"][0]["method"] == "Theis (1935)"

    def test_the_same_analysis_cannot_be_saved_twice(self, client):
        upload(client)
        run(client)
        client.post("/api/analyses/save")

        assert client.post("/api/analyses/save").status_code == 409

    def test_saving_needs_a_result(self, client):
        assert client.post("/api/analyses/save").status_code == 400

    def test_methods_are_ranked_by_aicc(self, client):
        """On leaky data the leaky model wins, and the ranking says by how much."""
        _upload_bytes(client, _leaky_csv(), "leaky.csv")
        run(client, method="theis_1935")
        client.post("/api/analyses/save")
        run(client, method="hantush_jacob_1955")
        body = client.post("/api/analyses/save").json()

        assert [row["method"] for row in body["rows"]] == [
            "Hantush-Jacob (1955)", "Theis (1935)"]
        assert body["rows"][0]["delta_aicc"] == 0
        assert body["rows"][1]["delta_aicc"] > 10
        assert body["rows"][1]["evidence"] == "essentially no support"

    def test_an_analysis_can_be_removed(self, client):
        upload(client)
        run(client)
        saved = client.post("/api/analyses/save").json()
        test_id = saved["test_id"]

        body = client.delete(
            f"/api/tests/{test_id}/analyses/{saved['rows'][0]['analysis_id']}").json()

        assert body["rows"] == []

    def test_comparison_can_be_read_for_a_test(self, client):
        upload(client)
        run(client)
        client.post("/api/analyses/save")
        test_id = client.get("/api/project").json()["active_test_id"]

        body = client.get(f"/api/tests/{test_id}/comparison").json()
        assert len(body["rows"]) == 1 and body["n_stale"] == 0


class TestReport:

    def test_the_report_needs_a_result(self, client):
        assert client.get("/api/report.pdf").status_code == 400

    def test_the_report_is_a_pdf_named_for_the_method(self, client):
        upload(client)
        run(client)

        response = client.get("/api/report.pdf")

        assert response.content.startswith(b"%PDF-")
        assert "theis_report.pdf" in response.headers["content-disposition"]

    def test_a_pumped_well_report_is_named_for_its_method(self, client):
        upload(client, PC_SAMPLE)
        run(client, method="papadopulos_cooper_1967", **PC_SETUP)

        response = client.get("/api/report.pdf")

        assert response.content.startswith(b"%PDF-")
        assert "papadopulos-cooper_report.pdf" in response.headers["content-disposition"]


@pytest.mark.slow
@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
class TestEveryMethodThroughTheApi:
    """Registry-driven: a method that cannot be run and drawn over HTTP is
    broken for the browser, however well its own tests pass."""

    def _run(self, client, spec):
        if registry.RECOVERY in spec.test_types:
            upload(client, RECOVERY_SAMPLE)
            return run(client, method=spec.id, **RECOVERY_SETUP)
        if spec.data_source == registry.PUMPED_WELL:
            upload(client, PC_SAMPLE)
            return run(client, method=spec.id, **PC_SETUP)
        upload(client, THEIS_SAMPLE)
        return run(client, method=spec.id)

    def test_it_fits_and_returns_a_figure(self, client, spec):
        body = self._run(client, spec).json()

        assert body["success"] is True, body["error_message"]
        assert body["figure"].startswith("{")
        assert body["parameters"], "a method must report what it estimated"

    def test_it_can_be_saved_matched_and_reported(self, client, spec):
        result = self._run(client, spec).json()

        assert client.post("/api/analyses/save").status_code == 200
        assert client.post("/api/analyses/manual-match",
                           json={"T": result["T"], "S": result["S"]}).status_code == 200
        assert client.get("/api/report.pdf").content.startswith(b"%PDF-")


class TestComingBackToATest:
    """Reported from the deployed site: two tests in the project panel, and
    clicking either showed no analysis at all.

    The readings were kept per test; the result was kept once per session, and
    switching test emptied it deliberately — so that a fit was never shown against
    another test's data. Correct, and it meant an afternoon's work left nothing
    behind. A test's result belongs to that test now, and a saved analysis is
    recovered from the project file by fitting it again.
    """

    def _test_with_a_fit(self, client, method, name=None):
        """Add a test, give it readings, fit it. Returns its id."""
        test_id = client.post("/api/tests").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)
        client.put(f"/api/tests/{test_id}/setup", json=setup_payload())
        response = client.post("/api/analyses/run",
                               json={"method": method, "setup": setup_payload()})
        assert response.status_code == 200, response.json()
        return test_id

    def test_each_test_keeps_its_own_result(self, client):
        first = self._test_with_a_fit(client, "theis_1935")
        self._test_with_a_fit(client, "cooper_jacob_1946")

        client.post(f"/api/tests/{first}/activate")

        body = client.get("/api/analyses/current").json()
        assert "Theis" in body["method"]

    def test_and_its_own_readings(self, client):
        first = self._test_with_a_fit(client, "theis_1935")
        self._test_with_a_fit(client, "cooper_jacob_1946")

        client.post(f"/api/tests/{first}/activate")

        assert client.get(f"/api/tests/{first}/readings").status_code == 200

    def test_the_other_test_still_has_its_own(self, client):
        """The guard the original wiping was there for: a result must never be
        shown against readings it was not fitted to."""
        self._test_with_a_fit(client, "theis_1935")
        second = self._test_with_a_fit(client, "cooper_jacob_1946")

        client.post(f"/api/tests/{second}/activate")

        assert "Cooper-Jacob" in client.get("/api/analyses/current").json()["method"]

    def test_the_run_log_follows_the_test(self, client):
        first = self._test_with_a_fit(client, "theis_1935")
        self._test_with_a_fit(client, "cooper_jacob_1946")

        client.post(f"/api/tests/{first}/activate")
        history = client.get("/api/analyses/history").json()

        assert [row["method"] for row in history] == ["Theis (1935)"]

    def test_a_test_never_fitted_has_nothing_to_show(self, client):
        self._test_with_a_fit(client, "theis_1935")
        empty = client.post("/api/tests").json()["active_test_id"]

        client.post(f"/api/tests/{empty}/activate")

        assert client.get("/api/analyses/current").status_code == 404
        assert client.get("/api/analyses/history").json() == []

    def test_new_readings_retire_the_remembered_result(self, client):
        """They describe data the test no longer holds."""
        test_id = self._test_with_a_fit(client, "theis_1935")

        upload(client, RECOVERY_SAMPLE, test_id)

        assert client.get("/api/analyses/current").status_code == 404
        assert client.get("/api/analyses/history").json() == []


class TestAnalysesSurviveAReload:
    """The other half of the same report: "there is no persistence of analysed
    results so far".

    A reload starts a new session. The project file holds the readings, the saved
    analyses and — now — the log of what was run, so everything can be recovered.
    The fit itself is not stored: it is run again, which is deterministic and takes
    milliseconds, and keeps every project file from carrying a copy of curves that
    the readings already imply.
    """

    def _worked_then_saved(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)
        client.put(f"/api/tests/{test_id}/setup", json=setup_payload())
        client.post("/api/analyses/run",
                    json={"method": "theis_1935", "setup": setup_payload()})
        client.post("/api/analyses/save")
        client.post("/api/analyses/run",
                    json={"method": "cooper_jacob_1946", "setup": setup_payload()})
        return client.get("/api/project/file").content

    def _reopen(self, client, file):
        response = client.post(
            "/api/project/open",
            files={"file": ("p.hydroassist.json", file, "application/json")})
        assert response.status_code == 200, response.json()
        return response.json()["active_test_id"]

    def test_the_saved_analysis_comes_back_whole(self, client, second_client):
        """A different browser entirely, which is what a reload amounts to."""
        file = self._worked_then_saved(client)

        self._reopen(second_client, file)
        body = second_client.get("/api/analyses/current").json()

        assert "Theis" in body["method"]
        assert body["T_day"] == pytest.approx(215.29, rel=1e-3)
        # Including what a re-run reproduces and a stored copy would not: the
        # figures and the notes that make a result readable.
        assert body["figure"]
        assert body["validity_notes"]

    def test_the_runs_come_back_too(self, client, second_client):
        file = self._worked_then_saved(client)

        self._reopen(second_client, file)

        assert [row["method"] for row in
                second_client.get("/api/analyses/history").json()] == [
            "Theis (1935)", "Cooper-Jacob (1946)"]

    def test_the_conversation_is_told_about_the_recovered_fit(self, client,
                                                             second_client):
        """Otherwise the consultant would discuss an analysis the user can see and
        it cannot."""
        file = self._worked_then_saved(client)

        self._reopen(second_client, file)
        second_client.get("/api/analyses/current")

        context = second_client.get("/api/chat").json()["context"]
        assert context["has_result"] is True

    def test_a_project_with_no_saved_analysis_recovers_nothing(self, client,
                                                              second_client):
        """Runs are a log; only a saved analysis carries the setup needed to
        reproduce the fit."""
        test_id = client.get("/api/project").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)
        client.put(f"/api/tests/{test_id}/setup", json=setup_payload())
        client.post("/api/analyses/run",
                    json={"method": "theis_1935", "setup": setup_payload()})
        file = client.get("/api/project/file").content

        self._reopen(second_client, file)

        assert second_client.get("/api/analyses/current").status_code == 404
        # The log still says it was run, which is the honest thing to show.
        assert [row["method"] for row in
                second_client.get("/api/analyses/history").json()] == ["Theis (1935)"]
