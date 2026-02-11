"""
Tools callable from the web UI: web browse, bash, file read/write, code snippet runner.

All paths and cwd are restricted to the project root (or a temp dir for snippets).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Project root: web/tools.py -> web/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MAX_BROWSE_BYTES = 512 * 1024  # 512 KiB
_MAX_READ_BYTES = 256 * 1024    # 256 KiB
_BASH_TIMEOUT = 60              # seconds
_SNIPPET_TIMEOUT = 15           # seconds
_MAX_SNIPPET_OUTPUT = 50 * 1024 # 50 KiB total stdout+stderr


def _resolve_path(path: str) -> Path:
    """Resolve path to project root; raise ValueError if outside."""
    p = (_PROJECT_ROOT / path).resolve()
    try:
        p.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Path must be inside project root: {path}")
    return p


def browse_web(url: str) -> dict[str, Any]:
    """Fetch URL (http/https only), return content and status. Content truncated if large."""
    url = (url or "").strip()
    if not url:
        return {"success": False, "error": "URL is empty"}
    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "Only http:// and https:// URLs are allowed"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LisaSimpson-WebUI/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(_MAX_BROWSE_BYTES)
            decoded = raw.decode("utf-8", errors="replace")
            return {
                "success": True,
                "status_code": getattr(resp, "status", 200),
                "content": decoded,
                "truncated": len(raw) >= _MAX_BROWSE_BYTES,
            }
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}", "status_code": e.code}
    except urllib.error.URLError as e:
        return {"success": False, "error": str(e.reason or e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_bash(command: str, cwd: str | None = None) -> dict[str, Any]:
    """Run a shell command with cwd restricted to project root. Timeout 60s."""
    command = (command or "").strip()
    if not command:
        return {"success": False, "error": "Command is empty"}
    work_dir = _PROJECT_ROOT
    if cwd:
        work_dir = _resolve_path(cwd)
        if not work_dir.is_dir():
            return {"success": False, "error": f"Not a directory: {cwd}"}
    try:
        r = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT,
        )
        return {
            "success": r.returncode == 0,
            "stdout": r.stdout or "",
            "stderr": r.stderr or "",
            "returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {_BASH_TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(path: str) -> dict[str, Any]:
    """Read file under project root. Max 256 KiB."""
    path = (path or "").strip()
    if not path:
        return {"success": False, "error": "Path is empty"}
    try:
        p = _resolve_path(path)
        if not p.is_file():
            return {"success": False, "error": "Not a file or does not exist"}
        content = p.read_bytes()
        if len(content) > _MAX_READ_BYTES:
            return {"success": False, "error": f"File too large (max {_MAX_READ_BYTES} bytes)"}
        return {"success": True, "content": content.decode("utf-8", errors="replace")}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def web_search(query: str, api_key: str | None = None) -> dict[str, Any]:
    """Run a web search via Serp API (Google). Uses SERPAPI_API_KEY from env if api_key not passed."""
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "Query is empty"}
    key = (api_key or os.getenv("SERPAPI_API_KEY") or "").strip()
    if not key:
        return {"success": False, "error": "SERPAPI_API_KEY not set. Add it in API Keys."}
    url = "https://serpapi.com/search?" + urllib.parse.urlencode({
        "engine": "google",
        "q": query,
        "api_key": key,
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LisaSimpson-WebUI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        error = data.get("error")
        if error:
            return {"success": False, "error": error}
        results = []
        for r in data.get("organic_results") or []:
            results.append({
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })
        return {"success": True, "query": query, "results": results}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"success": False, "error": str(e.reason or e)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write file under project root. Fails if path would escape project."""
    path = (path or "").strip()
    if not path:
        return {"success": False, "error": "Path is empty"}
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_snippet(language: str, code: str) -> dict[str, Any]:
    """Run a code snippet in a sandboxed subprocess: temp cwd, timeout, limited output. Supports python."""
    language = (language or "python").strip().lower()
    code = (code or "").strip()
    if not code:
        return {"success": False, "error": "Code is empty"}
    if language != "python":
        return {"success": False, "error": "Only python is supported. Use language='python'."}
    tmpdir = tempfile.mkdtemp(prefix="snippet_")
    try:
        r = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-c", code],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=_SNIPPET_TIMEOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        stdout = (r.stdout or "")[: _MAX_SNIPPET_OUTPUT]
        stderr = (r.stderr or "")[: _MAX_SNIPPET_OUTPUT]
        if len(r.stdout or "") > _MAX_SNIPPET_OUTPUT or len(r.stderr or "") > _MAX_SNIPPET_OUTPUT:
            stdout += "\n... (output truncated)"
        return {
            "success": r.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Run timed out after {_SNIPPET_TIMEOUT}s"}
    except FileNotFoundError:
        return {"success": False, "error": "python3 not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


TOOLS = {
    "browse_web": (browse_web, ["url"]),
    "run_bash": (run_bash, ["command"]),  # cwd optional
    "read_file": (read_file, ["path"]),
    "write_file": (write_file, ["path", "content"]),
    "web_search": (web_search, ["query"]),  # api_key optional, uses env
    "run_snippet": (run_snippet, ["language", "code"]),
}


def run_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to tool by name. params must contain required keys."""
    if tool_name not in TOOLS:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    func, required = TOOLS[tool_name]
    for k in required:
        if k not in params:
            return {"success": False, "error": f"Missing parameter: {k}"}
    kwargs = {k: params[k] for k in params if params.get(k) is not None or k in required}
    try:
        result = func(**kwargs)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
