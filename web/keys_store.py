"""
Secure persistence of API keys for the web UI.

Keys are stored in a single file with restricted permissions (0o600).
Values are never returned to the client; only presence is reported.
"""

from __future__ import annotations

import os
from pathlib import Path

# Whitelist: only these env var names can be set via the UI
ALLOWED_KEY_NAMES = frozenset({
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "SERPAPI_API_KEY",
})

# File lives under project root so it can be gitignored and permission-restricted
_KEYS_FILE = Path(__file__).resolve().parent.parent / "data" / "web_api_keys.env"


def _ensure_dir() -> None:
    _KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_raw() -> dict[str, str]:
    """Read KEY=value pairs from file. No value is ever exposed outside this module."""
    out: dict[str, str] = {}
    if not _KEYS_FILE.exists():
        return out
    with open(_KEYS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key in ALLOWED_KEY_NAMES and value:
                    out[key] = value
    return out


def load_into_env() -> None:
    """Load stored keys into os.environ so LLM clients can use them."""
    for key, value in _read_raw().items():
        os.environ[key] = value


def get_status() -> dict[str, bool]:
    """Return which allowed keys are set (true/false). Never return values."""
    raw = _read_raw()
    # Also consider already-set env (e.g. from shell)
    return {name: (name in raw or bool(os.getenv(name))) for name in sorted(ALLOWED_KEY_NAMES)}


def save_key(key_name: str, value: str) -> None:
    """Store one key. key_name must be in ALLOWED_KEY_NAMES. Value is stripped."""
    if key_name not in ALLOWED_KEY_NAMES:
        raise ValueError(f"Key not allowed: {key_name}")
    value = (value or "").strip()
    if not value:
        raise ValueError("Value cannot be empty")
    _ensure_dir()
    raw = _read_raw()
    raw[key_name] = value
    with open(_KEYS_FILE, "w") as f:
        for k, v in raw.items():
            # Store as KEY=value; avoid newlines in value
            v_flat = v.replace("\n", " ").replace("\r", " ")
            f.write(f"{k}={v_flat}\n")
    try:
        os.chmod(_KEYS_FILE, 0o600)
    except OSError:
        pass
    os.environ[key_name] = value


def remove_key(key_name: str) -> None:
    """Remove a stored key (delete from file and env)."""
    if key_name not in ALLOWED_KEY_NAMES:
        raise ValueError(f"Key not allowed: {key_name}")
    _ensure_dir()
    raw = _read_raw()
    raw.pop(key_name, None)
    if raw:
        with open(_KEYS_FILE, "w") as f:
            for k, v in raw.items():
                v_flat = v.replace("\n", " ").replace("\r", " ")
                f.write(f"{k}={v_flat}\n")
    else:
        if _KEYS_FILE.exists():
            _KEYS_FILE.unlink()
    os.environ.pop(key_name, None)
