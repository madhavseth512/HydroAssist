#!/usr/bin/env python3
"""Print the password hash the site checks against.

    python scripts/make_auth_hash.py 'the password you chose'

Put the output in HYDROASSIST_AUTH_PASSWORD_HASH, alongside
HYDROASSIST_AUTH_USER, wherever the app's environment is configured — Render's
dashboard, or the .env on a VM. The password itself is never stored.

Without both variables the site is open to anyone with the URL, and says so in
its start-up log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.auth import hash_password


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1]:
        print(__doc__.strip())
        return 1

    print(hash_password(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
