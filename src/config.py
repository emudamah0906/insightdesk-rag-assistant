"""
Lightweight .env loader (zero dependencies).

Importing this module loads key=value pairs from a project-root `.env` file into
os.environ (without overwriting variables already set in the shell). This lets you keep
secrets like ANTHROPIC_API_KEY in a local, gitignored `.env` file instead of exporting
them every session. Real secrets stay out of version control (.env is in .gitignore;
only .env.example is committed).
"""
from __future__ import annotations
import os, pathlib

ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / ".env"


def load_env(path: pathlib.Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # shell-set values win; .env only fills what's missing
        if key and key not in os.environ:
            os.environ[key] = value


load_env()
