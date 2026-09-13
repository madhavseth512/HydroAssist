"""The deployment files must describe the application they deploy.

A Dockerfile that copies a directory the app no longer reads, or forgets one it
does, builds cleanly and fails only on the server. These checks run with the
rest of the suite, so a change to the app that the deployment has not caught up
with fails here instead.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _ignored(pattern: str) -> bool:
    lines = [line.strip() for line in _read(".dockerignore").splitlines()]
    return pattern in lines


class TestTheImage:

    def test_it_runs_the_api_with_a_single_worker(self):
        """Sessions live in memory; a second worker would split them."""
        dockerfile = _read("Dockerfile")
        assert "src.api.app:app" in dockerfile
        assert '"--workers", "1"' in dockerfile

    @pytest.mark.parametrize("path", ["src/", "configs/", "data/vectordb/"])
    def test_it_copies_what_the_app_reads_at_runtime(self, path):
        assert re.search(rf"^COPY .*\s{re.escape(path)}\s", _read("Dockerfile"), re.M), path
        assert (ROOT / path).is_dir(), f"{path} is copied but does not exist"

    def test_it_ships_the_built_frontend(self):
        dockerfile = _read("Dockerfile")
        assert "--from=frontend" in dockerfile and "frontend/dist" in dockerfile
        assert (ROOT / "frontend" / "package-lock.json").is_file()

    def test_it_uses_the_configuration_the_tests_ran_against(self):
        assert "HYDROASSIST_ENV=default" in _read("Dockerfile")
        assert (ROOT / "configs" / "default.yaml").is_file()

    def test_it_runs_as_an_unprivileged_user(self):
        users = re.findall(r"^USER (\S+)", _read("Dockerfile"), re.M)
        assert users and users[-1] != "root"

    def test_the_embedding_model_is_baked_in(self):
        """Otherwise the first visitor waits for an 80 MB download."""
        assert "DefaultEmbeddingFunction" in _read("Dockerfile")


class TestTheBuildContext:

    def test_secrets_stay_out(self):
        assert _ignored(".env")

    def test_the_publisher_pdfs_stay_out(self):
        assert _ignored("knowledge source")
        assert "knowledge source" not in "\n".join(
            line for line in _read("Dockerfile").splitlines() if line.startswith("COPY"))

    def test_local_environments_stay_out(self):
        for pattern in (".venv", "frontend/node_modules", ".git"):
            assert _ignored(pattern), pattern


class TestCompose:

    @pytest.fixture(scope="class")
    def compose(self):
        return yaml.safe_load(_read("docker-compose.yml"))

    def test_the_session_cookie_is_secure_behind_https(self, compose):
        assert compose["services"]["app"]["environment"]["HYDROASSIST_COOKIE_SECURE"] == "1"

    def test_the_app_is_not_published_directly(self, compose):
        assert "ports" not in compose["services"]["app"]

    def test_caddy_waits_for_a_healthy_app(self, compose):
        assert compose["services"]["caddy"]["depends_on"]["app"]["condition"] == "service_healthy"

    def test_caddy_serves_http_and_https(self, compose):
        ports = compose["services"]["caddy"]["ports"]
        assert "80:80" in ports and "443:443" in ports

    def test_certificates_persist(self, compose):
        volumes = compose["services"]["caddy"]["volumes"]
        assert any(v.startswith("caddy_data:") for v in volumes)
        assert any("Caddyfile" in v for v in volumes)


class TestCaddy:

    def test_the_site_is_private(self):
        assert "basic_auth" in _read("Caddyfile")

    def test_it_proxies_to_the_app(self):
        assert "reverse_proxy app:8000" in _read("Caddyfile")

    def test_it_accepts_the_largest_upload_the_app_allows(self):
        from src.api.deps import MAX_UPLOAD_BYTES

        match = re.search(r"max_size\s+(\d+)MB", _read("Caddyfile"))
        assert match
        assert int(match.group(1)) * 1024 * 1024 > MAX_UPLOAD_BYTES


class TestStreamlitIsGone:

    def test_it_is_not_a_dependency(self):
        requirements = _read("requirements.txt")
        assert not re.search(r"^\s*streamlit", requirements, re.M)
        assert re.search(r"^fastapi", requirements, re.M)
        assert re.search(r"^uvicorn", requirements, re.M)

    def test_the_script_is_gone(self):
        assert not (ROOT / "app.py").exists()
        assert not (ROOT / ".streamlit").exists()

    def test_nothing_in_src_imports_it(self):
        for path in (ROOT / "src").rglob("*.py"):
            assert not re.search(r"^\s*(import|from)\s+streamlit",
                                 path.read_text(encoding="utf-8"), re.M), path

    def test_the_env_example_documents_every_setting_compose_reads(self):
        compose_vars = set(re.findall(r"\$\{(\w+)\}", _read("docker-compose.yml")))
        example = _read(".env.example")
        for name in compose_vars | {"GEMINI_API_KEY"}:
            assert re.search(rf"^{name}=", example, re.M), name
