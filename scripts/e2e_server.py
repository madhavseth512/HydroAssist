"""Serve the built app for the browser end-to-end tests.

The same FastAPI application and the same built SPA as production, with one
difference: the consultant is switched off. An end-to-end run must not spend
the free-tier model quota or depend on the network, and switching the
consultant off exercises the fallback every deployment relies on when the model
is unavailable — the workspace keeps working and the chat says why it cannot
answer.

Started by frontend/playwright.config.ts; build the frontend first
(`npm run build`), since this serves frontend/dist.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The app resolves frontend/dist relative to the working directory at import.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402  (after the path fix)

from src.api.app import create_app  # noqa: E402
from src.api.sessions import SessionStore  # noqa: E402


def offline_consultant():
    raise RuntimeError("the consultant is switched off for end-to-end tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()

    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        sys.exit("frontend/dist is not built. Run `npm run build` in frontend/ first.")

    app = create_app(SessionStore(chat_factory=offline_consultant))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
