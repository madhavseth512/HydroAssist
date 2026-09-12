"""Serving the built single-page app.

One origin serves the API and the SPA, which is what removes CORS and makes the
deployment a single container. Two things have to hold: a client-side route
must survive a reload (so unknown paths return index.html rather than 404), and
that catch-all must not become a way to read files off the server.

Skipped unless the frontend has been built, since `frontend/dist` is generated
at deploy time and not committed.
"""

from pathlib import Path

import pytest

DIST = Path("frontend/dist")

pytestmark = pytest.mark.skipif(
    not (DIST / "index.html").is_file(),
    reason="frontend/dist is not built (npm run build)",
)


def _index_html(response) -> bool:
    return '<div id="root">' in response.text


class TestTheAppIsServed:

    def test_the_root_serves_the_app(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert _index_html(response)

    def test_a_client_side_route_survives_a_reload(self, client):
        """The browser's own routes are not server routes; a refresh on one must
        still load the app."""
        response = client.get("/workspace/step-2")

        assert response.status_code == 200
        assert _index_html(response)

    def test_the_bundle_is_served(self, client):
        import re

        match = re.search(r'/assets/([\w.-]+\.js)', client.get("/").text)
        assert match, "index.html should reference a built bundle"

        response = client.get(f"/assets/{match.group(1)}")
        assert response.status_code == 200

    def test_the_api_still_wins_over_the_catch_all(self, client):
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/session").json()["session_id"]


class TestTheCatchAllIsNotAFileServer:
    """The SPA route resolves a path under the build directory, so it must
    never return a file from the repository around it."""

    @pytest.mark.parametrize("path", [
        "/../pyproject.toml",
        "/%2e%2e/pyproject.toml",
        "/..%2fpyproject.toml",
        "/assets/../../pyproject.toml",
        "/../../.env",
        "/../requirements.txt",
    ])
    def test_it_refuses_to_climb_out_of_the_build(self, client, path):
        response = client.get(path)

        # Either the app itself or a refusal — never the file being asked for.
        assert response.status_code in (200, 404)
        for secret in ("[tool.pytest", "[project]", "GEMINI_API_KEY",
                       "langchain", "streamlit"):
            assert secret not in response.text, f"{path} leaked {secret!r}"
