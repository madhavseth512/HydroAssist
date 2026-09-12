"""The HydroAssist HTTP application.

One origin serves both halves: `/api/*` is this API, and everything else is the
built React app. That removes CORS entirely and makes the deployment a single
container behind Caddy, which only terminates TLS.

Run it with:  uvicorn src.api.app:app --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api import routes_analysis, routes_chat, routes_workspace
from src.api.deps import set_store
from src.api.sessions import SessionStore

# Written by `npm run build` in frontend/. Absent during backend development and
# in the API tests, which is why serving it is optional.
FRONTEND_DIST = Path(
    os.getenv("HYDROASSIST_FRONTEND_DIST", "frontend/dist")).resolve()


def create_app(store: SessionStore = None) -> FastAPI:
    """Build the application.

    `store` is injectable so tests can run against a stub conversation instead
    of building a language-model client and opening the vector store.
    """
    if store is not None:
        set_store(store)

    app = FastAPI(
        title="HydroAssist",
        version="5.0.0",
        summary="Pumping test analysis that explains itself.",
        description=(
            "The calculators, the project model and the consultant, over HTTP. "
            "Sessions are per visitor: a conversation is never shared."
        ),
    )

    app.include_router(routes_workspace.router)
    app.include_router(routes_analysis.router)
    app.include_router(routes_chat.router)

    @app.get("/api/health", tags=["meta"])
    def health():
        """Liveness for the container smoke test and the reverse proxy."""
        from src.calculator import registry

        return {"status": "ok", "methods": len(registry.all_specs())}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA, when it has been built.

    Unknown paths return index.html rather than 404 so client-side routes
    survive a reload; `/api/*` is already claimed by the routers above.
    """
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        return

    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_root():
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = (FRONTEND_DIST / path).resolve()
        if (candidate.is_file()
                and candidate.is_relative_to(FRONTEND_DIST)):  # no path traversal
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
