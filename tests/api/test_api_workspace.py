"""Session, project and readings over HTTP.

These cover what the Streamlit AppTests covered for the sidebar and Step 1:
sessions never share state, a project round-trips through its file, and an
uploaded CSV comes back inspected, diagnosed and validated.
"""

import pytest

from conftest import RECOVERY_SAMPLE, THEIS_SAMPLE, setup_payload, upload

from src.calculator import registry


class TestSession:

    def test_health_names_the_method_catalogue(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["methods"] == len(registry.all_specs())

    def test_a_session_is_created_on_first_contact(self, client):
        body = client.get("/api/session").json()

        assert body["session_id"]
        assert len(body["project"]["tests"]) == 1
        assert body["chat"]["messages"] == []

    def test_the_same_browser_keeps_its_session(self, client):
        first = client.get("/api/session").json()["session_id"]
        assert client.get("/api/session").json()["session_id"] == first

    def test_two_visitors_never_share_a_workspace(self, client, second_client):
        """The bug that shipped once already: one cached chat for everyone."""
        first = client.get("/api/session").json()["session_id"]
        second = second_client.get("/api/session").json()["session_id"]
        assert first != second

        client.post("/api/chat/confirm", json={"method": "Theis (1935)"})

        assert client.get("/api/chat").json()["context"]["suggestion_confirmed"]
        assert not second_client.get("/api/chat").json()["context"]["suggestion_confirmed"]


class TestMethodCatalogue:

    def test_only_test_types_with_a_method_are_offered(self, client):
        ids = [t["id"] for t in client.get("/api/test-types").json()]
        assert ids == ["constant_rate", "recovery", "tracer"]

    def test_methods_are_listed_for_a_test_type(self, client):
        body = client.get("/api/methods", params={"test_type": "constant_rate"}).json()
        labels = [m["label"] for m in body["offered"]]

        assert labels == [s.label for s in registry.for_test_type("constant_rate")]
        assert "Theis Recovery (1935)" not in labels

    def test_a_method_declares_what_the_form_needs(self, client):
        body = client.get("/api/methods", params={"test_type": "constant_rate"}).json()
        by_id = {m["id"]: m for m in body["offered"]}

        assert by_id["papadopulos_cooper_1967"]["needs_well_geometry"] is True
        assert by_id["theis_1935"]["needs_well_geometry"] is False
        assert by_id["theis_1935"]["estimates_storativity"] is True

    def test_recovery_reports_that_it_estimates_no_storativity(self, client):
        body = client.get("/api/methods", params={"test_type": "recovery"}).json()

        assert [m["label"] for m in body["offered"]] == ["Theis Recovery (1935)"]
        assert body["offered"][0]["estimates_storativity"] is False

    def test_an_unknown_test_type_is_rejected(self, client):
        assert client.get("/api/methods", params={"test_type": "sideways"}).status_code == 400

    def test_a_short_test_gates_cooper_jacob(self, client):
        """Cooper-Jacob needs late-time data; the validator says so before the
        user picks it."""
        import pandas as pd
        from scipy.special import expn
        import numpy as np

        t = np.linspace(10, 600, 30)
        s = (500 / 86400) / (4 * np.pi * 2.5e-3) * expn(1, 50.0**2 * 5e-5 / (4 * 2.5e-3 * t))
        csv = pd.DataFrame({"time_s": t, "drawdown_m": s}).to_csv(index=False).encode()
        test_id = client.get("/api/project").json()["active_test_id"]
        client.post(f"/api/tests/{test_id}/readings",
                    files={"file": ("short.csv", csv, "text/csv")})

        body = client.get("/api/methods", params={"test_type": "constant_rate"}).json()

        assert "Cooper-Jacob (1946)" in body["gated_out"]
        assert "Theis (1935)" in [m["label"] for m in body["offered"]]
        assert body["help"]

    def test_a_validation_for_one_test_type_does_not_gate_another(self, client):
        """The readings are judged as a recovery record, which says nothing
        about which pumping methods suit a constant-rate test."""
        test_id = client.get("/api/project").json()["active_test_id"]
        client.put(f"/api/tests/{test_id}/setup",
                   json=setup_payload(test_type="recovery", t_p_hours=24.0))
        upload(client, RECOVERY_SAMPLE)

        recovery = client.get("/api/methods", params={"test_type": "recovery"}).json()
        pumping = client.get("/api/methods", params={"test_type": "constant_rate"}).json()

        assert [m["label"] for m in recovery["offered"]] == ["Theis Recovery (1935)"]
        assert pumping["gated_out"] == []


class TestProject:

    def test_metadata_can_be_edited(self, client):
        body = client.patch("/api/project/info", json={
            "title": "Campus well field", "client": "IIT Kharagpur",
            "project_number": "BTP-26-04", "location": "Kharagpur",
            "test_date": "2026-09-01", "analyst": "M. Seth", "notes": "",
        }).json()

        assert body["info"]["title"] == "Campus well field"
        assert body["filename"] == "Campus_well_field.hydroassist.json"

    def test_a_project_round_trips_through_its_file(self, client):
        client.patch("/api/project/info", json={"title": "Round trip"})
        upload(client)

        saved = client.get("/api/project/file")
        assert saved.headers["content-disposition"].endswith('.hydroassist.json"')

        client.post("/api/project/new")
        assert client.get("/api/project").json()["info"]["title"] == "Untitled project"

        reopened = client.post("/api/project/open",
                               files={"file": ("p.json", saved.content,
                                               "application/json")}).json()

        assert reopened["info"]["title"] == "Round trip"
        assert reopened["tests"][0]["n_readings"] == 50

    def test_a_file_that_is_not_a_project_is_explained(self, client):
        response = client.post("/api/project/open",
                               files={"file": ("x.json", b"{}", "application/json")})

        assert response.status_code == 400
        assert "not a HydroAssist project file" in response.json()["detail"]

    def test_tests_can_be_added_renamed_and_activated(self, client):
        first = client.get("/api/project").json()["active_test_id"]

        added = client.post("/api/tests").json()
        second = added["active_test_id"]
        assert second != first and len(added["tests"]) == 2

        renamed = client.patch(f"/api/tests/{second}", json={"name": "PW-2"}).json()
        assert [t["name"] for t in renamed["tests"]] == ["Test 1", "PW-2"]

        back = client.post(f"/api/tests/{first}/activate").json()
        assert back["active_test_id"] == first

    def test_an_unknown_test_is_a_404(self, client):
        assert client.post("/api/tests/nope/activate").status_code == 404


class TestReadings:

    def test_an_uploaded_csv_comes_back_prepared(self, client):
        body = upload(client).json()

        assert body["n_points"] == 50
        assert body["inspection"]["time_col"] == "time_s"
        assert body["validation"]["is_usable"] is True
        assert len(body["preview"]) == 10
        assert "50 data points" in body["summary"]

    def test_the_derivative_diagnosis_travels_with_it(self, client):
        body = upload(client).json()

        assert body["diagnosis"]["regime"]
        assert "Theis (1935)" in body["diagnosis"]["consistent_methods"]
        assert body["derivative_figure"].startswith("{")

    def test_the_readings_are_recorded_against_the_test(self, client):
        upload(client)
        test = client.get("/api/project").json()["tests"][0]

        assert test["n_readings"] == 50
        assert test["readings_source"].endswith(".csv")

    def test_saved_readings_can_be_read_back(self, client):
        upload(client)
        test_id = client.get("/api/project").json()["active_test_id"]

        body = client.get(f"/api/tests/{test_id}/readings").json()
        assert body["n_points"] == 50

    def test_a_test_without_readings_says_so(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        response = client.get(f"/api/tests/{test_id}/readings")

        assert response.status_code == 404

    def test_a_file_that_is_not_csv_is_explained(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        response = client.post(f"/api/tests/{test_id}/readings",
                               files={"file": ("x.csv", b"\x00\x01 not a csv",
                                               "text/csv")})

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_recovery_readings_are_not_flagged_for_falling_drawdown(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        client.put(f"/api/tests/{test_id}/setup",
                   json=setup_payload(test_type="recovery", t_p_hours=24.0))

        body = upload(client, RECOVERY_SAMPLE).json()

        assert not any("recovery" in note.lower()
                       for note in body["validation"]["info"])
        assert body["validation"]["suitable_methods"] == ["Theis Recovery (1935)"]


class TestTheFormatsPeopleArrriveWith:
    """CSV was the only accepted format, which is not how field data travels.

    The reader is unit-tested in tests/unit/test_table_reader.py; these check
    that every path reaches the same intake and produces the same readings, so
    a pasted table and an uploaded file cannot diverge.
    """

    def _readings(self):
        import pandas as pd
        return pd.read_csv(THEIS_SAMPLE)

    def _post_file(self, client, name, data, test_id=None):
        if test_id is None:
            test_id = client.get("/api/project").json()["active_test_id"]
        return client.post(f"/api/tests/{test_id}/readings",
                           files={"file": (name, data, "application/octet-stream")})

    def test_a_tab_delimited_text_export(self, client):
        response = self._post_file(
            client, "field.txt",
            self._readings().to_csv(sep="\t", index=False).encode())

        assert response.status_code == 200
        assert response.json()["n_points"] == 50

    def test_an_excel_workbook(self, client):
        import io

        import pandas as pd

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            self._readings().to_excel(writer, index=False, sheet_name="Test 1")

        response = self._post_file(client, "field.xlsx", buffer.getvalue())

        assert response.status_code == 200
        assert response.json()["n_points"] == 50

    def test_columns_pasted_from_a_spreadsheet(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        pasted = self._readings().to_csv(sep="\t", index=False)

        response = client.post(f"/api/tests/{test_id}/readings/text",
                               json={"text": pasted})

        assert response.status_code == 200
        assert response.json()["n_points"] == 50

    def test_pasted_readings_are_recorded_like_uploaded_ones(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]

        client.post(f"/api/tests/{test_id}/readings/text",
                    json={"text": self._readings().to_csv(index=False)})

        test = client.get("/api/project").json()["tests"][0]
        assert test["n_readings"] == 50
        assert test["readings_source"] == "pasted"

    def test_every_route_produces_the_same_readings(self, client):
        """The point of one reader behind all of them."""
        import io

        import pandas as pd

        readings = self._readings()
        project = client.get("/api/project").json()
        test_id = project["active_test_id"]

        by_csv = self._post_file(client, "a.csv",
                                 readings.to_csv(index=False).encode(), test_id).json()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            readings.to_excel(writer, index=False, sheet_name="Sheet1")
        by_excel = self._post_file(client, "a.xlsx", buffer.getvalue(), test_id).json()
        by_paste = client.post(f"/api/tests/{test_id}/readings/text",
                               json={"text": readings.to_csv(index=False)}).json()

        assert by_csv["n_points"] == by_excel["n_points"] == by_paste["n_points"]
        assert by_csv["preview"] == by_excel["preview"] == by_paste["preview"]

    def test_an_image_is_refused_with_the_formats_that_work(self, client):
        response = self._post_file(client, "photo.png", b"\x89PNG\r\n\x1a\n")

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert ".png" in detail and ".csv" in detail

    def test_empty_pasted_text_is_refused_by_the_schema(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]

        response = client.post(f"/api/tests/{test_id}/readings/text",
                               json={"text": ""})

        assert response.status_code == 422

    def test_a_sheet_can_be_named(self, client):
        import io

        import pandas as pd

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"Site": ["Gridley"]}).to_excel(
                writer, index=False, sheet_name="Cover")
            self._readings().to_excel(writer, index=False, sheet_name="Drawdown")

        test_id = client.get("/api/project").json()["active_test_id"]
        response = client.post(
            f"/api/tests/{test_id}/readings",
            files={"file": ("field.xlsx", buffer.getvalue(),
                            "application/octet-stream")},
            data={"sheet": "Drawdown"})

        assert response.status_code == 200
        assert response.json()["n_points"] == 50
        assert any("Drawdown" in note for note in response.json()["conversions"])


class TestAPhotographedFieldSheet:
    """A photograph proposes a grid; it never becomes readings on its own.

    The reading itself is unit-tested in tests/unit/test_table_ocr.py. What
    matters here is the boundary: this route stores nothing, and the corrected
    grid re-enters by the same door as typed text, so no image can reach a
    calculator without a person having looked at the numbers.
    """

    @pytest.fixture(scope="class")
    def photograph(self):
        """A printed field sheet, rendered and committed as a fixture."""
        from pathlib import Path

        from src.services.table_ocr import ocr_available

        if not ocr_available():
            pytest.skip("Tesseract is not installed on this machine")
        return Path("tests/fixtures/field_data/field_sheet.png").read_bytes()

    def _post(self, client, raw, test_id=None):
        if test_id is None:
            test_id = client.get("/api/project").json()["active_test_id"]
        return client.post(f"/api/tests/{test_id}/readings/image",
                           files={"file": ("sheet.png", raw, "image/png")})

    def test_the_grid_comes_back_cell_by_cell(self, client, photograph):
        body = self._post(client, photograph).json()

        assert body["columns"] == 2
        assert len(body["rows"]) == 13
        assert [cell["value"] for cell in body["rows"][0]] == [1.0, 0.22]
        assert [cell["value"] for cell in body["rows"][-1]] == [120.0, 1.07]

    def test_each_cell_carries_what_was_read_and_how_surely(self, client, photograph):
        cell = self._post(client, photograph).json()["rows"][0][0]

        assert cell["text"] == "1"
        assert 0.0 < cell["confidence"] <= 100.0
        assert cell["uncertain"] is False

    def test_a_cell_it_was_unsure_of_reaches_the_browser_marked(self, client,
                                                                photograph):
        """Tesseract reads one printed digit on this sheet at 71%. The mark is
        the only thing standing between that and a wrong transmissivity."""
        body = self._post(client, photograph).json()
        flagged = [cell for row in body["rows"] for cell in row
                   if cell and cell["uncertain"]]

        assert [cell["value"] for cell in flagged] == [5.0]
        assert body["uncertain_cells"] == 1
        assert body["mostly_uncertain"] is False

    def test_nothing_is_stored_against_the_test(self, client, photograph):
        """The rule the route exists for."""
        test_id = client.get("/api/project").json()["active_test_id"]

        self._post(client, photograph, test_id)

        assert client.get("/api/project").json()["tests"][0]["n_readings"] == 0
        assert client.get(f"/api/tests/{test_id}/readings").status_code == 404

    def test_the_confirmed_grid_becomes_readings_through_the_text_route(
            self, client, photograph):
        """The whole journey, as the browser makes it: propose, then confirm."""
        test_id = client.get("/api/project").json()["active_test_id"]
        grid = self._post(client, photograph, test_id).json()

        text = "time_min\tdrawdown_m\n" + "\n".join(
            "\t".join("" if cell is None else cell["text"] for cell in row)
            for row in grid["rows"])
        response = client.post(f"/api/tests/{test_id}/readings/text",
                               json={"text": text})

        assert response.status_code == 200
        assert response.json()["n_points"] == 13

    def test_a_file_that_is_not_an_image_is_refused(self, client):
        from src.services.table_ocr import ocr_available

        if not ocr_available():
            pytest.skip("Tesseract is not installed on this machine")

        response = self._post(client, b"time_s,drawdown_m\n60,0.12\n")

        assert response.status_code == 400
        assert "could not be opened" in response.json()["detail"]

    def test_an_unknown_test_is_a_404(self, client, photograph):
        assert self._post(client, photograph, "no-such-test").status_code == 404

    def test_a_server_without_tesseract_says_what_is_missing(self, client,
                                                             monkeypatch):
        """Tesseract is a program, not a package, so a host may well lack it.
        That is a 503 about the server, not a 400 about the image."""
        from src.services import table_ocr

        def missing(*_args, **_kwargs):
            raise table_ocr.OCRNotAvailable(
                "Reading a photograph needs the Tesseract program itself")

        monkeypatch.setattr(table_ocr, "read_image_table", missing)

        response = self._post(client, b"whatever")

        assert response.status_code == 503
        assert "Tesseract" in response.json()["detail"]


class TestHowMuchOfTheTableComesBack:
    """The preview was ten rows with no way to see the rest, so a user could
    check the first few readings of their file and nothing else."""

    def _upload(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)
        return test_id

    def test_a_preview_by_default(self, client):
        test_id = self._upload(client)

        body = client.get(f"/api/tests/{test_id}/readings").json()

        assert body["n_points"] == 50
        assert len(body["preview"]) == 10

    def test_the_whole_table_when_asked(self, client):
        test_id = self._upload(client)

        body = client.get(f"/api/tests/{test_id}/readings?rows=50").json()

        assert len(body["preview"]) == 50

    def test_asking_for_more_rows_than_exist_is_not_an_error(self, client):
        test_id = self._upload(client)

        body = client.get(f"/api/tests/{test_id}/readings?rows=5000").json()

        assert len(body["preview"]) == 50

    def test_a_nonsensical_row_count_is_refused_by_the_schema(self, client):
        test_id = self._upload(client)

        assert client.get(f"/api/tests/{test_id}/readings?rows=0").status_code == 422


class TestWhoOwnsTheMethod:
    """The workspace owns the method; the conversation only proposes one.

    Reported from the deployed site: Theis was selected and run, then a message
    about another method changed what the chat said the method was. The rule
    taken from that — and already the rule for the guide's offers — is that
    nothing in a conversation moves the analysis without the user pressing
    something.
    """

    def _ran_theis(self, client):
        test_id = client.get("/api/project").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)
        client.put(f"/api/tests/{test_id}/setup", json=setup_payload())
        response = client.post("/api/analyses/run", json={
            "method": "theis_1935", "setup": setup_payload()})
        assert response.status_code == 200
        return test_id

    def _preselected(self, client):
        body = client.get("/api/methods").json()
        return body["offered"][body["default_index"]]["label"]

    def test_the_form_opens_on_the_method_that_was_run(self, client):
        self._ran_theis(client)

        assert "Theis" in self._preselected(client)

    def test_the_conversation_does_not_move_it(self, client, store):
        """The case as reported: a fit exists, and the chat starts discussing
        something else."""
        self._ran_theis(client)

        # What a message naming another method leaves in the conversation.
        _discussing(store, "Cooper-Jacob (1946)")

        assert "Theis" in self._preselected(client)

    def test_accepting_a_suggestion_does_move_it(self, client):
        """Because that is the user pressing a button, which is the whole
        difference."""
        self._ran_theis(client)

        response = client.post("/api/chat/confirm",
                               json={"method": "Cooper-Jacob (1946)"})
        assert response.status_code == 200

        assert "Cooper-Jacob" in self._preselected(client)

    def test_before_any_fit_the_conversation_still_leads(self, client, store):
        """Nothing has been run, so the suggestion is the best guess there is."""
        test_id = client.get("/api/project").json()["active_test_id"]
        upload(client, THEIS_SAMPLE, test_id)

        _discussing(store, "Cooper-Jacob (1946)")

        assert "Cooper-Jacob" in self._preselected(client)


def _discussing(store, method: str) -> None:
    """Put a method into the conversation, as a message naming one would.

    The conversation is normally written by the model, which these tests do not
    call; setting the field directly is how "the user mentioned Cooper-Jacob" is
    expressed without spending a request from the daily allowance.
    """
    for session in store._sessions.values():
        session.chat.state["discussed_method"] = method
        if not session.chat.state.get("suggestion_confirmed"):
            session.chat.state["selected_method"] = method


class TestAFieldFileWithAProvenanceHeader:
    """Reported from the deployed site: uploading this repository's own Gridley
    fixture failed with `Failed to process the file: "[''] not in index"`.

    Its first four lines are `#` comments naming the test and what Walton
    published from it. They were read as the header row, so the readings never
    arrived. Driven over HTTP here because that is how it was found.
    """

    GRIDLEY = "tests/fixtures/field_data/gridley_well1.csv"

    def _upload(self, client, name="gridley1.csv"):
        test_id = client.get("/api/project").json()["active_test_id"]
        with open(self.GRIDLEY, "rb") as handle:
            return client.post(
                f"/api/tests/{test_id}/readings",
                files={"file": (name, handle.read(), "text/csv")})

    def test_it_uploads(self, client):
        response = self._upload(client)

        assert response.status_code == 200, response.json()
        assert response.json()["n_points"] == 22

    def test_the_columns_and_units_are_read(self, client):
        body = self._upload(client).json()

        assert body["inspection"]["time_unit"] == "minutes"
        assert body["inspection"]["drawdown_unit"] == "feet"

    def test_where_the_readings_came_from_reaches_the_user(self, client):
        body = self._upload(client).json()
        notes = " ".join(body["conversions"])

        assert "comment line" in notes
        assert "Bulletin 49" in notes

    def test_it_can_then_be_fitted(self, client):
        """Walton's own analysis of this test was Theis, and his published
        transmissivity is 11,000 gpd/ft — about 137 m²/day."""
        self._upload(client)
        setup = dict(setup_payload(), q_value=1199.2, q_unit="m³/day", r_m=251.2)
        client.put(f"/api/tests/"
                   f"{client.get('/api/project').json()['active_test_id']}/setup",
                   json=setup)

        response = client.post("/api/analyses/run",
                               json={"method": "theis_1935", "setup": setup})

        assert response.status_code == 200, response.json()
        assert response.json()["success"] is True
        assert 100 < response.json()["T_day"] < 200
