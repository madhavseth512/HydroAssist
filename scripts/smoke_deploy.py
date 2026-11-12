"""Check a running HydroAssist end to end.

Health, a session, a CSV upload, a Theis fit checked against the golden value, a
photographed field sheet, the PDF report, and a tracer breakthrough curve fitted
with Sauty (1980) — the path a hydrogeologist takes,
over HTTP. Standard library only, so it runs on the server without the project's
virtual environment:

    python3 scripts/smoke_deploy.py https://yourname.duckdns.org --user hydroassist --password '...'
    python3 scripts/smoke_deploy.py http://127.0.0.1:8000

Exits 0 when every step passes and 1 at the first that does not, saying which.
The consultant is not exercised: it spends model quota, and the workspace is
designed to work without it.
"""

import argparse
import base64
import http.cookiejar
import json
import ssl
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "theis_cooper_jacob_sample.csv"
# A printed field sheet, for the OCR route: 13 readings, one cell of which
# Tesseract reads at 71% and the grid therefore marks for checking.
FIELD_SHEET = (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
               / "field_data" / "field_sheet.png")

# A forced-gradient breakthrough curve generated with t_c = 67.86 h, alpha_L = 1 m,
# n_e = 0.2 and 80% recovery of 5 kg, at r = 15 m, b = 10 m, Q = 500 m3/day.
TRACER_SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "tracer_convergent_sample.csv"
EXPECTED_N_E = 0.2

# The fit tests/fixtures/golden_fits.json pins for this sample.
EXPECTED_T_DAY = 215.29
TOLERANCE = 1e-3


class Client:
    """Cookies, basic auth, and nothing else."""

    def __init__(self, base_url: str, user: str = None, password: str = None,
                 insecure: bool = False):
        self.base_url = base_url.rstrip("/")
        handlers = [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())]
        if insecure:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self.opener = urllib.request.build_opener(*handlers)
        self.auth = None
        if user is not None:
            token = base64.b64encode(f"{user}:{password or ''}".encode()).decode()
            self.auth = f"Basic {token}"

    def request(self, method: str, path: str, body: bytes = None,
                content_type: str = None):
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        if content_type:
            request.add_header("Content-Type", content_type)
        if self.auth:
            request.add_header("Authorization", self.auth)
        try:
            with self.opener.open(request, timeout=180) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def json(self, method: str, path: str, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        status, raw = self.request(method, path, body,
                                   "application/json" if body is not None else None)
        try:
            return status, json.loads(raw.decode("utf-8") or "null")
        except ValueError:
            return status, raw.decode("utf-8", "replace")

    def upload(self, path: str, file_path: Path):
        boundary = uuid.uuid4().hex
        # The server chooses its reader by the filename, not by this, but a PNG
        # announced as text/csv is a lie that would confuse anyone reading a log.
        content_type = ("image/png" if file_path.suffix.lower() == ".png"
                        else "text/csv")
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode() + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        status, raw = self.request("POST", path, body,
                                   f"multipart/form-data; boundary={boundary}")
        try:
            return status, json.loads(raw.decode("utf-8"))
        except ValueError:
            return status, raw.decode("utf-8", "replace")


def check(ok: bool, step: str, success: str = "", failure: str = "") -> None:
    """One line per step. The detail on failure can be long — it is the server's
    whole answer — so it is only printed when something is wrong."""
    detail = success if ok else failure
    print(f"  {'ok  ' if ok else 'FAIL'}  {step}{' - ' + detail if detail else ''}")
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="e.g. https://yourname.duckdns.org")
    parser.add_argument("--user", help="basic-auth user, if the site has one")
    parser.add_argument("--password", help="basic-auth password")
    parser.add_argument("--insecure", action="store_true",
                        help="skip TLS verification (a certificate still being issued)")
    args = parser.parse_args()

    client = Client(args.base_url, args.user, args.password, args.insecure)
    print(f"HydroAssist at {client.base_url}")

    status, body = client.json("GET", "/api/health")
    check(status == 200 and isinstance(body, dict) and body.get("status") == "ok",
          "health", success=f"{body.get('methods') if isinstance(body, dict) else '?'} methods",
          failure=f"HTTP {status}: {body}")

    status, _ = client.request("GET", "/")
    check(status == 200, "the workspace page is served", failure=f"HTTP {status}")

    status, session = client.json("GET", "/api/session")
    check(status == 200 and isinstance(session, dict), "session",
          failure=f"HTTP {status}: {session}")
    test_id = session["project"]["active_test_id"]

    status, prepared = client.upload(f"/api/tests/{test_id}/readings", SAMPLE)
    check(status == 200 and isinstance(prepared, dict) and prepared.get("n_points") == 50,
          "upload the sample readings",
          success=f"{prepared.get('n_points') if isinstance(prepared, dict) else '?'} points",
          failure=f"HTTP {status}: {prepared}")

    status, result = client.json("POST", "/api/analyses/run", {
        "method": "theis_1935",
        "setup": {"q_value": 500, "q_unit": "m³/day", "r_m": 50},
    })
    fitted = isinstance(result, dict) and result.get("success")
    check(status == 200 and fitted, "run Theis (1935)",
          failure=f"HTTP {status}: {result}")

    t_day = result["T_day"]
    # ASCII units: this may print to a console that cannot encode "²".
    check(abs(t_day - EXPECTED_T_DAY) / EXPECTED_T_DAY < TOLERANCE,
          "transmissivity matches the golden fit",
          success=f"T = {t_day:.2f} m^2/day",
          failure=f"T = {t_day:.2f} m^2/day, expected {EXPECTED_T_DAY}")

    # The one feature that needs a program rather than a wheel. It is checked
    # here because a missing apt layer in the image is invisible until someone
    # photographs a field sheet, and the answer then looks like their fault.
    status, grid = client.upload(f"/api/tests/{test_id}/readings/image", FIELD_SHEET)
    rows = len(grid.get("rows", [])) if isinstance(grid, dict) else 0
    check(status == 200 and rows == 13,
          "read a photographed field sheet",
          success=f"{rows} rows, "
                  f"{grid.get('uncertain_cells')} of {grid.get('total_cells')} "
                  f"cells uncertain" if isinstance(grid, dict) else "",
          failure=(f"HTTP {status}: {grid}. A 503 means the image lacks "
                   f"Tesseract — see the apt-get layer in the Dockerfile."))

    status, pdf = client.request("GET", "/api/report.pdf")
    check(status == 200 and pdf.startswith(b"%PDF-"), "PDF report",
          success=f"{len(pdf)} bytes", failure=f"HTTP {status}, {len(pdf)} bytes")

    # Tracer tests: a concentration column switches the test over by itself.
    status, prepared = client.upload(f"/api/tests/{test_id}/readings", TRACER_SAMPLE)
    check(status == 200 and isinstance(prepared, dict) and prepared.get("n_points") == 40,
          "upload a tracer breakthrough curve", failure=f"HTTP {status}: {prepared}")

    status, result = client.json("POST", "/api/analyses/run", {
        "method": "sauty_1980",
        "setup": {"test_type": "tracer", "q_value": 500, "q_unit": "m³/day", "r_m": 15,
                  "tracer_b_m": 10, "tracer_M_kg": 5},
    })
    params = ({p["name"]: p["value"] for p in result.get("parameters", [])}
              if isinstance(result, dict) and result.get("success") else {})
    n_e = params.get("n_e")
    check(status == 200 and n_e is not None and abs(n_e - EXPECTED_N_E) < 0.01,
          "run Sauty (1980) on the tracer curve",
          success=f"n_e = {n_e:.3f}, alpha_L = {params.get('alpha_L', 0):.2f} m" if n_e else "",
          failure=f"HTTP {status}: {result}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
