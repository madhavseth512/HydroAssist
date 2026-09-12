"""Write the API's OpenAPI schema to openapi.json.

The frontend's TypeScript types are generated from this file
(`npm run gen:types`), so it is committed: the browser's idea of a response and
the server's Pydantic models come from one source, and a mismatch is a compile
error rather than a runtime surprise.

Run after changing anything in src/api/schemas.py:

    python scripts/export_openapi.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.app import create_app  # noqa: E402  (after the path fix)

OUTPUT = Path(__file__).resolve().parents[1] / "openapi.json"


def main() -> None:
    schema = create_app().openapi()
    OUTPUT.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"{OUTPUT.name}: {len(schema['paths'])} paths, "
          f"{len(schema['components']['schemas'])} schemas")


if __name__ == "__main__":
    main()
