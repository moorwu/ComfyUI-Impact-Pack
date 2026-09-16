"""Minimal ComfyUI HTTP API helpers (stdlib only)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
CLIENT_ID = str(uuid.uuid4())


def _open(req_or_url, timeout: float) -> bytes:
    """Opens a URL or request, returns response bytes, or raises RuntimeError with HTTP error details.

    Args:
        req_or_url: urllib.request.Request object or URL string
        timeout: timeout in seconds

    Returns:
        Response body as bytes

    Raises:
        RuntimeError: if HTTP error occurs, includes method, path, status code, and response body
    """
    try:
        with urllib.request.urlopen(req_or_url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        # Determine method and path for error message
        if isinstance(req_or_url, urllib.request.Request):
            method = req_or_url.get_method()
            full_url = req_or_url.full_url
        else:
            method = "GET"
            full_url = req_or_url

        # Extract relative path from full URL
        path = full_url.removeprefix(BASE)

        body = e.read().decode(errors='replace')
        raise RuntimeError(f"{method} {path} -> {e.code}: {body}") from e


def _post_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    data = _open(req, timeout=60)
    return json.loads(data or b"{}")


def _get_json(path: str) -> dict:
    data = _open(BASE + path, timeout=60)
    return json.loads(data)


def queue(workflow: dict) -> str:
    return _post_json("/prompt", {"prompt": workflow, "client_id": CLIENT_ID})["prompt_id"]


def wait(prompt_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = _get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"prompt {prompt_id} failed: {json.dumps(status.get('messages'), ensure_ascii=False)[:4000]}")
            if status.get("completed", True):
                return entry["outputs"]
        time.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} not finished after {timeout}s")


def download(filename: str, subfolder: str, folder_type: str, dest: Path) -> Path:
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _open(f"{BASE}/view?{query}", timeout=120)
    dest.write_bytes(data)
    return dest


def upload_image(path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(BASE + "/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    data = _open(req, timeout=120)
    return json.loads(data)["name"]
