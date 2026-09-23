"""The password that keeps the site private.

On Oracle, Caddy did this. On a host that terminates TLS itself there is no
Caddy, so this middleware is the only thing between the knowledge base — which
quotes paywalled papers — and anyone who has the URL. It therefore gets tested
like a lock, not like a feature: every route closed, the health check open, a
wrong password refused, and a malformed configuration failing shut.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from src.api import auth
from src.api.app import create_app

USER = "kgp"
PASSWORD = "a-long-shared-demo-password"


def _header(user: str = USER, password: str = PASSWORD) -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def protected(monkeypatch, store):
    monkeypatch.setenv(auth.USER_ENV, USER)
    monkeypatch.setenv(auth.HASH_ENV, auth.hash_password(PASSWORD, iterations=1000))
    return TestClient(create_app(store))


@pytest.fixture
def open_site(monkeypatch, store):
    monkeypatch.delenv(auth.USER_ENV, raising=False)
    monkeypatch.delenv(auth.HASH_ENV, raising=False)
    return TestClient(create_app(store))


class TestHashing:

    def test_a_password_verifies_against_its_own_hash(self):
        stored = auth.hash_password(PASSWORD, iterations=1000)
        assert auth.verify_password(PASSWORD, stored) is True

    def test_a_different_password_does_not(self):
        stored = auth.hash_password(PASSWORD, iterations=1000)
        assert auth.verify_password(PASSWORD + "!", stored) is False

    def test_the_password_is_not_recoverable_from_the_hash(self):
        assert PASSWORD not in auth.hash_password(PASSWORD, iterations=1000)

    def test_the_same_password_hashes_differently_every_time(self):
        """A per-password salt, so two people choosing the same password do not
        share a hash."""
        first = auth.hash_password(PASSWORD, iterations=1000)
        second = auth.hash_password(PASSWORD, iterations=1000)

        assert first != second
        assert auth.verify_password(PASSWORD, first)
        assert auth.verify_password(PASSWORD, second)

    @pytest.mark.parametrize("stored", [
        "", "nonsense", "pbkdf2_sha256$notanumber$aa$bb",
        "md5$1000$aa$bb", "pbkdf2_sha256$1000$zz$bb", None,
    ])
    def test_a_malformed_hash_denies_rather_than_crashes(self, stored):
        """A truncated or mistyped environment variable must fail shut. Raising
        here would 500 every request, which is a different kind of outage."""
        assert auth.verify_password(PASSWORD, stored) is False


class TestWithAPasswordSet:

    @pytest.mark.parametrize("path", ["/", "/api/session", "/api/methods"])
    def test_every_route_refuses_without_credentials(self, protected, path):
        assert protected.get(path).status_code == 401

    def test_the_browser_is_asked_for_a_password(self, protected):
        """Without this header the browser shows a bare 401 instead of a login
        box."""
        response = protected.get("/api/session")

        assert "Basic" in response.headers.get("www-authenticate", "")
        assert "HydroAssist" in response.headers.get("www-authenticate", "")

    def test_the_right_credentials_get_in(self, protected):
        assert protected.get("/api/session", headers=_header()).status_code == 200

    @pytest.mark.parametrize("user,password", [
        (USER, "wrong"), ("wrong", PASSWORD), ("wrong", "wrong"), ("", ""),
    ])
    def test_wrong_credentials_are_refused(self, protected, user, password):
        response = protected.get("/api/session", headers=_header(user, password))
        assert response.status_code == 401

    @pytest.mark.parametrize("header", [
        "Basic not-base64!!", "Basic " + base64.b64encode(b"nocolon").decode(),
        "Bearer sometoken", "Basic", "",
    ])
    def test_a_malformed_header_is_refused(self, protected, header):
        response = protected.get("/api/session", headers={"Authorization": header})
        assert response.status_code == 401

    def test_the_health_check_stays_open(self, protected):
        """The host polls this before anyone can type a password, and a 401
        there reads as a dead service."""
        response = protected.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_a_posted_upload_is_refused_too(self, protected):
        """Not just GETs: an unauthenticated POST must not reach the API."""
        response = protected.post("/api/chat/messages", json={"message": "hello"})
        assert response.status_code == 401


class TestWithNothingSet:

    def test_the_site_is_open(self, open_site):
        """What local development and the suite need."""
        assert open_site.get("/api/session").status_code == 200

    def test_half_a_configuration_is_not_protection(self, monkeypatch, store):
        """A user with no hash cannot be verified, so it must not be mistaken
        for protection: it stays open, and the start-up warning stands."""
        monkeypatch.setenv(auth.USER_ENV, USER)
        monkeypatch.delenv(auth.HASH_ENV, raising=False)

        assert auth.credentials_configured() is False

    def test_it_says_so_loudly(self, monkeypatch, caplog):
        monkeypatch.delenv(auth.USER_ENV, raising=False)
        monkeypatch.delenv(auth.HASH_ENV, raising=False)

        with caplog.at_level("WARNING"):
            auth.announce_state()

        assert "OFF" in caplog.text
        assert "paywalled" in caplog.text
