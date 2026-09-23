"""Password protection for the whole site.

On Oracle this was Caddy's job: `basic_auth` in the Caddyfile kept the site
private while the licensing question about the publisher PDFs is open, because
the consultant quotes them. A host like Render has no Caddy — the platform
terminates TLS and routes straight to the container — so the same protection has
to live in the application, or the corpus is one URL away from being public.

Deliberately stdlib-only. The runtime image already carries scipy, chromadb and
onnxruntime into 512 MB of RAM, and `bcrypt` or `passlib` would buy little here:
one shared password, behind HTTPS, guarding a demo.

The hash format is `pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`, which
`python scripts/make_auth_hash.py` prints. A plaintext password is never stored,
and the comparison is timing-safe.
"""

import base64
import binascii
import hashlib
import hmac
import os
import secrets
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

USER_ENV = "HYDROASSIST_AUTH_USER"
HASH_ENV = "HYDROASSIST_AUTH_PASSWORD_HASH"

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 240_000

# Reachable without credentials: the host's own health check runs before anyone
# can type a password, and a health check that 401s reads as a dead service.
OPEN_PATHS = frozenset({"/api/health"})


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS,
                  salt: Optional[bytes] = None) -> str:
    """A storable hash of `password`, in the format the middleware expects."""
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Whether `password` produces `stored`. False for anything malformed, so a
    corrupted or truncated environment variable denies access rather than
    crashing every request."""
    try:
        algorithm, iterations, salt_hex, expected_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        derived = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                      bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, AttributeError, binascii.Error):
        return False

    return hmac.compare_digest(derived.hex(), expected_hex)


def credentials_configured() -> bool:
    return bool(os.getenv(USER_ENV)) and bool(os.getenv(HASH_ENV))


def _unauthorised() -> Response:
    return JSONResponse(
        {"detail": "Authentication required."},
        status_code=401,
        # Without this the browser never offers a login box.
        headers={"WWW-Authenticate": 'Basic realm="HydroAssist", charset="UTF-8"'},
    )


def _credentials_from(header: Optional[str]):
    """The user and password in an Authorization header, or None."""
    if not header or not header.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1], validate=True).decode()
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    if ":" not in decoded:
        return None
    user, password = decoded.split(":", 1)
    return user, password


async def basic_auth_middleware(request: Request, call_next):
    """Require the configured user and password on every request but the health
    check.

    With nothing configured the site is open, which is what local development
    and the test suite need. That is announced loudly at start-up rather than
    left to be discovered.
    """
    if not credentials_configured() or request.url.path in OPEN_PATHS:
        return await call_next(request)

    credentials = _credentials_from(request.headers.get("authorization"))
    if credentials is None:
        return _unauthorised()

    user, password = credentials
    expected_user = os.getenv(USER_ENV, "")

    # Both checks always run: returning early on a wrong username would leak
    # which half was wrong through the response time.
    user_ok = hmac.compare_digest(user, expected_user)
    password_ok = verify_password(password, os.getenv(HASH_ENV, ""))

    if not (user_ok and password_ok):
        logger.warning(f"Rejected sign-in for {user!r} on {request.url.path}")
        return _unauthorised()

    return await call_next(request)


def announce_state() -> None:
    """Say at start-up whether the site is protected."""
    if credentials_configured():
        logger.info("Password protection is ON "
                    f"(user {os.getenv(USER_ENV)!r}).")
    else:
        logger.warning(
            f"Password protection is OFF: {USER_ENV} and {HASH_ENV} are not "
            "both set. Anyone with the URL can read the knowledge base, which "
            "quotes paywalled papers. Set them before sharing the link.")
