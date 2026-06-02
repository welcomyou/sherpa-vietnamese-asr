"""ASGI app for the browser-only offline PWA.

The server does not run ASR. It serves:
- the installable PWA shell
- a model manifest
- same-origin model file downloads from the bundled local models/

Keeping model downloads same-origin is important for COOP/COEP isolation and
WASM threading.
"""

import json
import logging
import os
from urllib.parse import quote, urlparse
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import BASE_DIR
from offline_pwa.config import MANIFEST_FILE, STATIC_DIR, offline_pwa_config

logger = logging.getLogger("asr.offline_pwa")

app = FastAPI(title="Sherpa Vietnamese ASR Offline PWA", docs_url=None, redoc_url=None, openapi_url=None)
SHARED_STATIC_DIR = os.path.join(BASE_DIR, "shared_ui", "static")
_ALLOWED_REMOTE_MODEL_HOSTS = {"huggingface.co", "github.com", "raw.githubusercontent.com"}


class OfflinePWASecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'wasm-unsafe-eval'; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "media-src 'self' blob:; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'"
        )
        return response


app.add_middleware(OfflinePWASecurityMiddleware)


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        raise HTTPException(500, "Offline PWA model manifest is missing")
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    ready = 0
    missing = []
    optional_missing = []
    for pack in data.get("packs", []):
        optional_pack = pack.get("required") is False or pack.get("optional") is True
        pack_ready = 0
        pack_missing = []
        for item in pack.get("files", []):
            total += 1
            local_status = _local_file_status(item)
            item["download_url"] = f"/api/model-files/{quote(item['id'], safe='')}"
            item["available_local"] = bool(local_status["ready"])
            item["server_size"] = local_status.get("size", 0)
            item["server_status"] = local_status["reason"]
            if local_status["ready"]:
                ready += 1
                pack_ready += 1
            else:
                pack_missing.append(item.get("id", "unknown"))
                if optional_pack:
                    optional_missing.append(item.get("id", "unknown"))
                else:
                    missing.append(item.get("id", "unknown"))
        pack["server_ready"] = not pack_missing
        pack["server_available"] = pack_ready
        pack["server_missing"] = pack_missing
    data["server_model_bundle"] = {
        "source": offline_pwa_config.model_source,
        "ready": not missing,
        "available": ready,
        "total": total,
        "missing": missing,
        "missing_optional": optional_missing,
        "remote_downloads_enabled": offline_pwa_config.remote_model_downloads_enabled,
    }
    return data


def _local_file_status(item: dict) -> dict:
    local_path = item.get("local_path") or item.get("target_path")
    if not local_path:
        return {"ready": False, "path": None, "size": 0, "reason": "no local path"}
    candidate = os.path.realpath(os.path.join(BASE_DIR, local_path))
    models_root = os.path.realpath(os.path.join(BASE_DIR, "models"))
    if not candidate.startswith(models_root + os.sep) and candidate != models_root:
        return {"ready": False, "path": None, "size": 0, "reason": "outside models root"}
    if not os.path.exists(candidate) or not os.path.isfile(candidate):
        return {"ready": False, "path": candidate, "size": 0, "reason": "missing on server"}
    size = os.path.getsize(candidate)
    expected = item.get("bytes")
    if expected and size != expected:
        return {
            "ready": False,
            "path": candidate,
            "size": size,
            "reason": f"server size mismatch ({size} != {expected})",
        }
    return {"ready": True, "path": candidate, "size": size, "reason": "bundled on server"}


def _resolve_local_path(item: dict) -> str | None:
    status = _local_file_status(item)
    return status["path"] if status["ready"] else None


def _validate_remote_model_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _ALLOWED_REMOTE_MODEL_HOSTS:
        raise HTTPException(400, "Remote model URL is not allowed")
    return url


def _iter_hf_file(url: str, expected_bytes: int | None = None):
    req = UrlRequest(url, headers={"User-Agent": "asr-vn-offline-pwa/1.0"})
    with urlopen(req, timeout=60) as resp:
        max_bytes = offline_pwa_config.max_model_download_bytes
        header_size = resp.headers.get("Content-Length")
        if header_size:
            try:
                remote_size = int(header_size)
                if remote_size > max_bytes:
                    raise RuntimeError("Remote model file exceeds configured size limit")
                if expected_bytes and remote_size != expected_bytes:
                    raise RuntimeError("Remote model file size does not match manifest")
            except ValueError:
                pass
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise RuntimeError("Remote model file exceeds configured size limit")
            if expected_bytes and downloaded > expected_bytes:
                raise RuntimeError("Remote model file exceeds manifest size")
            yield chunk
        if expected_bytes and downloaded != expected_bytes:
            raise RuntimeError("Remote model file size does not match manifest")


def _find_manifest_item(file_id: str) -> dict:
    manifest = _load_manifest()
    for pack in manifest.get("packs", []):
        for item in pack.get("files", []):
            if item.get("id") == file_id:
                return item
    raise HTTPException(404, "Model file id not found")


def _static_file(root: str, path: str, media_type: str | None = None):
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, path))
    if not candidate.startswith(root_real + os.sep) and candidate != root_real:
        raise HTTPException(404, "Static path not found")
    if not os.path.exists(candidate) or not os.path.isfile(candidate):
        raise HTTPException(404, "Static file not found")
    return FileResponse(candidate, media_type=media_type)


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/index.html")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/manifest.json")
async def web_manifest():
    return FileResponse(os.path.join(STATIC_DIR, "manifest.json"), media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker():
    return FileResponse(os.path.join(STATIC_DIR, "sw.js"), media_type="application/javascript")


@app.get("/hotword.txt")
async def hotword_txt():
    return FileResponse(os.path.join(STATIC_DIR, "hotword.txt"), media_type="text/plain; charset=utf-8")


@app.get("/shared/{path:path}")
async def shared(path: str):
    return _static_file(SHARED_STATIC_DIR, path)


@app.get("/install-cert")
async def download_cert():
    from web_service.ssl_utils import get_install_cert_path

    cert_path = get_install_cert_path()
    if not cert_path:
        raise HTTPException(404, "SSL certificate not found")
    return FileResponse(
        cert_path,
        filename="sherpa-asr-vn.crt",
        media_type="application/x-x509-ca-cert",
    )


@app.get("/css/{path:path}")
async def css(path: str):
    return _static_file(os.path.join(STATIC_DIR, "css"), path)


@app.get("/js/{path:path}")
async def js(path: str):
    return _static_file(os.path.join(STATIC_DIR, "js"), path, media_type="application/javascript")


@app.get("/vendor/{path:path}")
async def vendor(path: str):
    suffix = os.path.splitext(path)[1].lower()
    media_type = None
    if suffix in (".js", ".mjs"):
        media_type = "application/javascript"
    elif suffix == ".wasm":
        media_type = "application/wasm"
    return _static_file(os.path.join(STATIC_DIR, "vendor"), path, media_type=media_type)


@app.get("/calibration/{path:path}")
async def calibration(path: str):
    return _static_file(os.path.join(STATIC_DIR, "calibration"), path, media_type="application/octet-stream")


@app.get("/icons/{filename}")
async def icons(filename: str):
    icon_path = os.path.join(STATIC_DIR, "icons", filename)
    if not os.path.exists(icon_path):
        icon_path = os.path.join(BASE_DIR, "web_service", "static", "icons", filename)
    if not os.path.exists(icon_path):
        raise HTTPException(404, "Icon not found")
    return FileResponse(icon_path)


@app.get("/health")
async def health():
    return {"ok": True, "offline_pwa": True, "config": offline_pwa_config.to_dict()}


@app.get("/api/config")
async def config():
    return JSONResponse(offline_pwa_config.to_dict())


@app.get("/api/version")
async def version():
    from core.version import get_version
    return {"version": get_version()}


@app.get("/api/model-manifest")
async def model_manifest():
    return JSONResponse(_load_manifest())


@app.get("/api/model-bundle-status")
async def model_bundle_status():
    manifest = _load_manifest()
    return JSONResponse(manifest["server_model_bundle"])


@app.get("/api/model-files/{file_id}")
async def model_file(file_id: str):
    item = _find_manifest_item(file_id)
    local_path = _resolve_local_path(item)
    if local_path:
        headers = {}
        if item.get("sha256"):
            headers["X-Content-SHA256"] = str(item["sha256"])
        return FileResponse(
            local_path,
            filename=os.path.basename(local_path),
            media_type="application/octet-stream",
            headers=headers,
        )

    local_status = _local_file_status(item)
    if not offline_pwa_config.remote_model_downloads_enabled:
        raise HTTPException(
            503,
            (
                f"Bundled model file is not ready on server: {file_id} "
                f"({local_status['reason']}). Rebuild or prepare the server package "
                "with the offline PWA model bundle."
            ),
        )

    repo = item.get("repo")
    path = item.get("path")
    if not repo or not path:
        raise HTTPException(404, "No HuggingFace repo/path for model file")
    if not item.get("bytes") or not item.get("sha256"):
        raise HTTPException(500, "Remote model manifest must include bytes and sha256")
    if int(item["bytes"]) > offline_pwa_config.max_model_download_bytes:
        raise HTTPException(413, "Model file exceeds configured size limit")

    remote_url = _validate_remote_model_url(
        item.get("url") or f"https://huggingface.co/{repo}/resolve/main/{quote(path)}"
    )
    if not offline_pwa_config.model_proxy_enabled:
        return RedirectResponse(remote_url)
    logger.info("[OfflinePWA] Proxying model file %s from %s", file_id, remote_url)
    headers = {
        "Content-Length": str(item["bytes"]),
        "X-Content-SHA256": str(item["sha256"]),
    }
    return StreamingResponse(
        _iter_hf_file(remote_url, int(item["bytes"])),
        media_type="application/octet-stream",
        headers=headers,
    )
