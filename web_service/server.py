"""
FastAPI server chính - routes, WebSocket, static files.
"""

import os
import re
import json
import uuid
import time
import asyncio
import logging
from typing import Optional
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote, urlparse

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form,
    Depends, HTTPException, Request, Response, Cookie, Query,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from web_service.config import server_config, UPLOAD_DIR, DATA_DIR
from web_service.database import db
from web_service.auth import (
    authenticate_user, create_token, decode_token, hash_password,
    change_password, ensure_admin,
)
from web_service.session_manager import session_manager, ws_manager
from web_service.queue_manager import queue_manager

logger = logging.getLogger("asr.server")

# === JWT revocation list (A07: revoke tokens on logout) ===
# Lưu (jti/token, exp_timestamp) — tự cleanup khi hết hạn
_revoked_tokens: dict[str, float] = {}


def _revoke_token(token: str, exp: float):
    """Thêm token vào danh sách bị thu hồi."""
    _revoked_tokens[token] = exp
    # Cleanup tokens đã hết hạn để tránh memory leak
    now = time.time()
    expired = [t for t, e in _revoked_tokens.items() if e < now]
    for t in expired:
        del _revoked_tokens[t]


def is_token_revoked(token: str) -> bool:
    return token in _revoked_tokens


# === Login rate limiting (in-memory) ===
_login_attempts: dict[str, list[float]] = defaultdict(list)
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300  # 5 phút


def _check_login_rate(ip: str):
    """Kiểm tra rate limit login. Raise 429 nếu quá giới hạn."""
    now = time.time()
    _login_attempts[ip] = [t for t in _login_attempts[ip]
                           if now - t < _LOGIN_WINDOW_SECONDS]
    if len(_login_attempts[ip]) >= _LOGIN_MAX_ATTEMPTS:
        oldest = min(_login_attempts[ip])
        remain = int(_LOGIN_WINDOW_SECONDS - (now - oldest)) + 1
        raise HTTPException(429, f"Quá nhiều lần thử. Vui lòng đợi {remain} giây.")


def _record_failed_login(ip: str):
    _login_attempts[ip].append(time.time())


def _clear_login_rate(ip: str = None):
    """Xóa rate limit. ip=None xóa tất cả."""
    if ip:
        _login_attempts.pop(ip, None)
    else:
        _login_attempts.clear()


def _get_locked_ips() -> list:
    """Trả về danh sách IP đang bị khóa login. Đồng thời cleanup entries hết hạn."""
    now = time.time()
    result = []
    # A04: Cleanup để tránh memory leak khi có nhiều IP
    for ip in list(_login_attempts.keys()):
        valid = [t for t in _login_attempts[ip] if now - t < _LOGIN_WINDOW_SECONDS]
        if not valid:
            del _login_attempts[ip]
        else:
            _login_attempts[ip] = valid
            if len(valid) >= _LOGIN_MAX_ATTEMPTS:
                oldest = min(valid)
                remain = int(_LOGIN_WINDOW_SECONDS - (now - oldest)) + 1
                result.append({"ip": ip, "attempts": len(valid), "unlock_in_seconds": remain})
    # Cleanup upload attempts
    for sid in list(_upload_attempts.keys()):
        valid = [t for t in _upload_attempts[sid] if now - t < _UPLOAD_WINDOW_SECONDS]
        if not valid:
            del _upload_attempts[sid]
        else:
            _upload_attempts[sid] = valid
    # Cleanup account lockouts
    for uname in list(_account_failures.keys()):
        valid = [t for t in _account_failures[uname] if now - t < _LOCKOUT_DURATION_SECONDS]
        if not valid:
            del _account_failures[uname]
        else:
            _account_failures[uname] = valid
    return result

# === Upload rate limiting (per session, cho phép tải nhiều nhưng chống spam) ===
_upload_attempts: dict[str, list[float]] = defaultdict(list)
_UPLOAD_MAX_PER_MINUTE = 10  # 10 upload/phút — đủ cho người dùng bình thường
_UPLOAD_WINDOW_SECONDS = 60


def _check_upload_rate(session_id: str):
    now = time.time()
    _upload_attempts[session_id] = [t for t in _upload_attempts[session_id]
                                     if now - t < _UPLOAD_WINDOW_SECONDS]
    if len(_upload_attempts[session_id]) >= _UPLOAD_MAX_PER_MINUTE:
        raise HTTPException(429, "Upload quá nhanh. Vui lòng đợi 1 phút.")
    _upload_attempts[session_id].append(now)


# === Account lockout (per username, chống brute force) ===
_account_failures: dict[str, list[float]] = defaultdict(list)
_LOCKOUT_THRESHOLD = 10        # Lock sau 10 lần sai
_LOCKOUT_DURATION_SECONDS = 900  # Khóa 15 phút


def _check_account_lockout(username: str):
    now = time.time()
    _account_failures[username] = [t for t in _account_failures[username]
                                    if now - t < _LOCKOUT_DURATION_SECONDS]
    if len(_account_failures[username]) >= _LOCKOUT_THRESHOLD:
        remain = int(_LOCKOUT_DURATION_SECONDS - (now - min(_account_failures[username]))) + 1
        raise HTTPException(429, f"Tài khoản tạm khóa do đăng nhập sai quá nhiều. Thử lại sau {remain // 60} phút.")


def _record_failed_account(username: str):
    _account_failures[username].append(time.time())


def _clear_account_lockout(username: str):
    _account_failures.pop(username, None)


# === FastAPI App ===

app = FastAPI(title="Sherpa Vietnamese ASR", docs_url=None, redoc_url=None)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# === Middleware ===

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers + CSRF Origin check."""

    async def dispatch(self, request: Request, call_next):
        # CSRF: kiểm tra Origin header trên state-changing methods
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin", "")
            if origin:
                origin_host = urlparse(origin).netloc
                request_host = request.headers.get("host", "")
                if origin_host and request_host and origin_host != request_host:
                    return JSONResponse(
                        {"detail": "Origin không hợp lệ"}, status_code=403
                    )

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # A05: 'unsafe-inline' giữ cho scripts vì PWA dùng inline handlers.
        # connect-src giới hạn về 'self' thay vì wss:/ws: mở toàn bộ.
        host = request.headers.get("host", "").split(":")[0] or "localhost"
        _ws_scheme = "ws" if server_config.http_mode else "wss"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data:; "
            f"connect-src 'self' {_ws_scheme}://{host} {_ws_scheme}://{host}:*; "
            "media-src 'self' blob:; "
            "font-src 'self' https://fonts.gstatic.com"
        )
        if not server_config.http_mode:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"

        return response


class SessionMiddleware(BaseHTTPMiddleware):
    """Read-only middleware: chỉ đọc session cookie, KHÔNG tự động tạo.
    Session được tạo qua POST /api/session (gọi 1 lần từ frontend).
    """

    async def dispatch(self, request: Request, call_next):
        session_id = request.cookies.get("session_id", "")
        # Validate session tồn tại trong DB
        if session_id and db.get_session(session_id):
            request.state.session_id = session_id
        else:
            request.state.session_id = ""
        request.state.new_session = False
        return await call_next(request)


# Middleware thêm theo thứ tự ngược (middleware cuối chạy trước)
app.add_middleware(SessionMiddleware)
app.add_middleware(SecurityMiddleware)


# === Dependencies ===

def get_session_id(request: Request) -> str:
    return getattr(request.state, "session_id", request.cookies.get("session_id", ""))


def get_current_user(request: Request) -> Optional[dict]:
    """Lấy user từ JWT token (nếu có). Trả về None nếu anonymous, deactivated, hoặc token bị revoke."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    # A07: Kiểm tra token đã bị revoke (logout)
    if is_token_revoked(token):
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user = db.get_user_by_id(int(payload["sub"]))
    if not user or not user.get("is_active", True):
        return None
    return user


def require_auth(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Authentication required")
    return user


def require_admin(user: Optional[dict] = Depends(get_current_user)):
    if not user or user["role"] != "admin":
        raise HTTPException(403, "Admin required")
    return user


def check_file_access(file_record: dict | None, session_id: str, user: Optional[dict] = None):
    """Kiểm tra quyền truy cập file: session owner hoặc user owner (cho meetings)."""
    if not file_record:
        raise HTTPException(404, "File not found")
    # Session owner
    if file_record["session_id"] == session_id:
        return
    # User owner (logged-in user loading from meetings)
    if user and file_record.get("user_id") and file_record["user_id"] == user["id"]:
        return
    raise HTTPException(404, "File not found")


# === Static files & index ===

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Download SSL cert de cai tren dien thoai
@app.get("/install-cert")
async def download_cert():
    cert_path = os.path.join(os.path.dirname(__file__), "certs", "server.crt")
    if os.path.exists(cert_path):
        return FileResponse(cert_path, filename="sherpa-asr-vn.crt",
                            media_type="application/x-x509-ca-cert")
    return {"error": "Certificate not found"}


# Mount static AFTER specific routes
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# === Startup ===

@app.on_event("startup")
async def startup():
    ensure_admin()
    # Lưu reference tới event loop chính để worker threads có thể gửi WS messages
    queue_manager._event_loop = asyncio.get_running_loop()
    # Cleanup stale 'processing' items left from previous run
    stale = db.cleanup_stale_queue()
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale queue items (file_ids: {stale})")
    # Startup: xóa mọi anonymous session cũ + files vật lý
    session_manager.cleanup_on_startup(kill_processes_callback=queue_manager.cancel)
    # Start cleanup loop
    asyncio.create_task(
        session_manager.start_cleanup_loop(
            kill_processes_callback=queue_manager.cancel
        )
    )
    # Resume queue neu co items dang doi
    queue_manager.process_next()
    # Warm-up: pre-import heavy modules (librosa, soundfile, pydub, speaker_diarization)
    # trong thread riêng để không block event loop, giúp /api/config/models respond nhanh
    asyncio.create_task(_warmup_heavy_imports())
    logger.info("Server started")


async def _warmup_heavy_imports():
    """Pre-import heavy modules in background thread so first API call is fast."""
    def _do_imports():
        try:
            from core.config import get_speaker_embedding_models, is_diarization_available
            is_diarization_available()
            get_speaker_embedding_models()
            logger.info("Warmup: speaker diarization modules loaded")
        except Exception as e:
            logger.warning(f"Warmup: speaker diarization not available: {e}")

    await asyncio.to_thread(_do_imports)


# === Config API (public) ===

@app.get("/api/config/models")
async def get_models():
    """Danh sách models ASR và speaker có sẵn (chỉ trả về models đã tải)"""
    # Chạy trong thread riêng vì các import/file check có thể block event loop
    return await asyncio.to_thread(_get_models_sync)


def _get_models_sync():
    """Synchronous helper - chạy trong thread để không block event loop."""
    from core.config import BASE_DIR, get_speaker_embedding_models, is_diarization_available

    models_dir = os.path.join(BASE_DIR, "models")
    asr_models = []
    from core.asr_engine import ROVER_MODEL_ID, ROVER_MODEL_IDS

    # Zipformer 30M (nhanh)
    path_30m = os.path.join(models_dir, "zipformer-30m-rnnt-6000h")
    if os.path.isdir(path_30m):
        asr_models.append({"id": "zipformer-30m-rnnt-6000h", "name": "hynt/Zipformer-30M (nhanh)"})

    # Zipformer 2025 68M (mặc định, chính xác hơn)
    path_68m = os.path.join(models_dir, "sherpa-onnx-zipformer-vi-2025-04-20")
    if os.path.isdir(path_68m):
        asr_models.append({"id": "sherpa-onnx-zipformer-vi-2025-04-20", "name": "Zipformer-Vi 2025 (68M)"})

    # ROVER (kết hợp 2 model, chậm nhưng chính xác nhất)
    rover_available = all(
        os.path.isdir(os.path.join(models_dir, mid)) for mid in ROVER_MODEL_IDS
    )
    if rover_available:
        asr_models.append({"id": ROVER_MODEL_ID, "name": "ROVER (chậm, chính xác)"})

    speaker_models = []
    if is_diarization_available():
        from core.speaker_diarization import get_available_models, SpeakerDiarizer
        available = get_available_models(BASE_DIR)
        all_models = get_speaker_embedding_models()
        for model_id in available:
            info = all_models.get(model_id, {})
            speaker_models.append({
                "id": model_id,
                "name": info.get("name", model_id),
                "default_threshold": int(SpeakerDiarizer.get_default_threshold(model_id) * 100),
                "has_threshold": info.get("has_threshold", True),
            })

    return {
        "asr_models": asr_models,
        "speaker_models": speaker_models,
        "diarization_available": is_diarization_available(),
    }


@app.get("/api/version")
async def get_version_info():
    """Trả về version hiện tại."""
    from core.version import get_version
    return {"version": get_version()}


@app.get("/api/config/defaults")
async def get_defaults():
    """Cấu hình mặc định"""
    return {
        "asr_model": server_config.get("default_asr_model"),
        "speaker_model": server_config.get("default_speaker_model"),
        "punctuation_confidence": int(server_config.get("default_punctuation_confidence")),
        "case_confidence": int(server_config.get("default_case_confidence")),
        "diarization_threshold": int(server_config.get("default_diarization_threshold")),
        # merge_short_speaker removed — NaturalTurn luôn bật
        "max_upload_mb": int(server_config.get("max_upload_mb")),
        "offline_download_url": server_config.get("offline_download_url"),
    }


# === Session API ===

@app.post("/api/session")
async def create_session(request: Request):
    """Tạo session mới (gọi 1 lần khi trang load).
    Nếu đã có session cookie hợp lệ, trả về session hiện tại."""
    session_id = request.cookies.get("session_id", "")
    if session_id:
        session = db.get_session(session_id)  # cached - chỉ gọi 1 lần
        if session:
            return JSONResponse({
                "session_id": session["id"],
                "is_anonymous": session["is_anonymous"],
                "user_id": session["user_id"],
            })

    # Kiem tra max_sessions truoc khi tao moi
    max_sessions = server_config.max_sessions
    active_count = db.get_active_session_count()
    if active_count >= max_sessions:
        # Thu kick session anonymous idle lau nhat de nhuong cho
        oldest = db.get_oldest_idle_anonymous_session()
        if oldest and not ws_manager.is_connected(oldest["id"]):
            logger.info(f"Max sessions reached ({active_count}/{max_sessions}), "
                        f"evicting idle anonymous session: {oldest['id']}")
            session_manager.kill_session(
                oldest["id"],
                kill_processes_callback=queue_manager.cancel,
            )
        else:
            raise HTTPException(503, "Server đang đầy, vui lòng thử lại sau")

    # Tao session moi
    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")
    session_id = session_manager.create_session(ip_address=ip, user_agent=ua)

    response = JSONResponse({
        "session_id": session_id,
        "is_anonymous": True,
        "user_id": None,
    })
    response.set_cookie(
        "session_id", session_id,
        httponly=True, samesite="lax", max_age=86400,
        secure=not server_config.http_mode,
    )
    return response


@app.get("/api/session")
async def get_session_info(session_id: str = Depends(get_session_id)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session["id"],
        "is_anonymous": session["is_anonymous"],
        "user_id": session["user_id"],
    }


@app.post("/api/session/heartbeat")
async def session_heartbeat(session_id: str = Depends(get_session_id)):
    """HTTP heartbeat - dùng khi WS chưa reconnect (mobile browser quay lại từ background)."""
    if not session_id or not db.get_session(session_id):
        raise HTTPException(404, "Session not found")
    session_manager.heartbeat(session_id)
    return {"ok": True}


@app.get("/api/session/status")
async def get_session_status(session_id: str = Depends(get_session_id)):
    """Trạng thái chi tiết session: anonymous/login, queue, connected."""
    status = session_manager.get_session_status(session_id)
    if not status:
        raise HTTPException(404, "Session not found")
    return status


# === Upload API ===

_MAX_SPEAKER_ID = 99       # speaker_id hợp lệ: số nguyên 0-99
_MAX_TEXT_LEN   = 50_000   # ký tự tối đa mỗi segment text
_MAX_SEGMENTS   = 100_000  # số segment tối đa trong một file


def _sanitize_asr_json(data: dict) -> dict:
    """
    P1 XSS: Sanitize JSON ASR trước khi lưu DB.
    - Chỉ giữ các key được phép ở top-level.
    - Validate và ép kiểu từng segment (type, speaker_id, text, start, end).
    - Loại bỏ mọi giá trị kiểu HTML/JS có thể gây stored XSS khi frontend render.
    """
    import html as _html

    allowed_top = {"segments", "speaker_names", "model", "duration_sec",
                   "speaker_colors", "language", "sample_rate"}
    cleaned: dict = {}

    # Top-level metadata
    for k in allowed_top:
        if k in data:
            cleaned[k] = data[k]

    # segments: validate từng phần tử
    raw_segs = data.get("segments", [])
    if not isinstance(raw_segs, list):
        raise HTTPException(400, "segments phải là array")
    if len(raw_segs) > _MAX_SEGMENTS:
        raise HTTPException(400, f"Quá nhiều segments (tối đa {_MAX_SEGMENTS})")

    safe_segs = []
    for seg in raw_segs:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type", ""))
        if seg_type not in ("text", "speaker", "gap"):
            continue
        s: dict = {"type": seg_type}

        # speaker_id: phải là số nguyên hợp lệ
        if "speaker_id" in seg:
            try:
                spk = int(seg["speaker_id"])
                if 0 <= spk <= _MAX_SPEAKER_ID:
                    s["speaker_id"] = spk
            except (ValueError, TypeError):
                pass  # bỏ qua speaker_id không hợp lệ

        # text: giới hạn độ dài, strip HTML tags
        if "text" in seg:
            txt = str(seg["text"])[:_MAX_TEXT_LEN]
            s["text"] = _html.escape(txt, quote=False)

        # timing: chỉ chấp nhận số
        for tf in ("start", "end", "duration"):
            if tf in seg:
                try:
                    s[tf] = float(seg[tf])
                except (ValueError, TypeError):
                    pass

        # confidence: 0.0-1.0
        if "confidence" in seg:
            try:
                c = float(seg["confidence"])
                s["confidence"] = max(0.0, min(1.0, c))
            except (ValueError, TypeError):
                pass

        safe_segs.append(s)

    cleaned["segments"] = safe_segs

    # speaker_names: {str_key: str_value}, escape HTML
    raw_names = cleaned.get("speaker_names", {})
    if isinstance(raw_names, dict):
        cleaned["speaker_names"] = {
            str(k)[:20]: _html.escape(str(v)[:200], quote=False)
            for k, v in list(raw_names.items())[:_MAX_SPEAKER_ID + 1]
        }
    else:
        cleaned["speaker_names"] = {}

    # speaker_colors: {str_key: CSS color string}, chỉ cho phép #hex hoặc rgb(...)
    raw_colors = cleaned.get("speaker_colors", {})
    if isinstance(raw_colors, dict):
        import re as _re2
        _color_re = _re2.compile(r'^(#[0-9a-fA-F]{3,8}|rgb\(\d{1,3},\s*\d{1,3},\s*\d{1,3}\))$')
        cleaned["speaker_colors"] = {
            str(k)[:20]: v
            for k, v in list(raw_colors.items())[:_MAX_SPEAKER_ID + 1]
            if isinstance(v, str) and _color_re.match(v.strip())
        }
    else:
        cleaned.pop("speaker_colors", None)

    # Scalar fields
    if "model" in cleaned:
        cleaned["model"] = str(cleaned["model"])[:200]
    if "duration_sec" in cleaned:
        try:
            cleaned["duration_sec"] = float(cleaned["duration_sec"])
        except (ValueError, TypeError):
            cleaned.pop("duration_sec", None)

    return cleaned


ALLOWED_EXTENSIONS = {
    "mp3", "wav", "m4a", "flac", "aac", "wma", "ogg", "opus",
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    # Anonymous user phải có session hợp lệ
    if not session_id and not user:
        raise HTTPException(400, "Session hết hạn. Vui lòng tải lại trang.")

    # Rate limit upload (chống spam, không ảnh hưởng sử dụng bình thường)
    _check_upload_rate(session_id or (user["id"] if user else "unknown"))

    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Định dạng không hỗ trợ: .{ext}")

    max_size = server_config.max_upload_bytes

    # Anonymous: xóa file cũ trước khi upload mới (chỉ giữ 1 file)
    user_id = user["id"] if user else None
    if not user_id and session_id:
        old_files = db.delete_session_files(session_id)
        for fname in old_files:
            import glob as _glob
            for f_path in _glob.glob(os.path.join(UPLOAD_DIR, f"{fname}*")):
                try:
                    os.remove(f_path)
                except OSError:
                    pass
        if old_files:
            logger.info(f"Cleaned {len(old_files)} old file(s) for anonymous session {session_id[:8]}")

    # Sanitize filename (chống path traversal)
    import re as _re
    safe_name = _re.sub(r'[^\w\s.\-]', '', file.filename.replace('..', '').replace('/', '_').replace('\\', '_'))
    if not safe_name:
        safe_name = "upload"

    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    if not os.path.realpath(stored_path).startswith(os.path.realpath(UPLOAD_DIR)):
        raise HTTPException(400, "Tên file không hợp lệ")

    # P2 DoS: Stream file lên disk từng chunk thay vì đọc toàn bộ vào RAM
    # Giới hạn size được kiểm tra trong khi ghi — tránh DoS bộ nhớ khi upload song song
    _CHUNK = 1024 * 1024  # 1 MB
    written = 0
    try:
        with open(stored_path, "wb") as f:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_size:
                    f.close()
                    os.remove(stored_path)
                    raise HTTPException(400, f"File quá lớn. Tối đa {server_config.get('max_upload_mb')}MB")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(stored_path):
            os.remove(stored_path)
        raise HTTPException(500, f"Lỗi lưu file: {e}")

    # Check storage limit nếu user đã login
    if user and user["storage_limit_gb"] > 0:
        limit = int(user["storage_limit_gb"] * 1024 * 1024 * 1024)
        if user["storage_used_bytes"] + written > limit:
            os.remove(stored_path)
            raise HTTPException(400, "Vượt quá giới hạn lưu trữ")

    file_id = db.create_file(
        session_id=session_id,
        original_filename=file.filename,
        stored_filename=stored_name,
        file_size_bytes=written,
        user_id=user_id,
    )

    # Cập nhật storage
    if user_id:
        db.update_user_storage(user_id)

    logger.info(f"File uploaded: {file.filename} ({written} bytes) file_id={file_id}")

    return {"file_id": file_id, "filename": file.filename, "size": written}


@app.post("/api/upload-json/{file_id}")
async def upload_json(
    file_id: int,
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Upload JSON ASR đã có trước đó"""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    # P2 DoS: Giới hạn JSON upload tối đa 50 MB (đủ cho file transcript rất dài)
    _JSON_MAX = 50 * 1024 * 1024
    content = await file.read(_JSON_MAX + 1)
    if len(content) > _JSON_MAX:
        raise HTTPException(400, "JSON file quá lớn (tối đa 50 MB)")
    try:
        json_data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "JSON không hợp lệ")

    # Validate + sanitize JSON structure (P1 XSS: loại bỏ payload độc hại)
    if "segments" not in json_data:
        raise HTTPException(400, "JSON không đúng cấu trúc ASR (thiếu 'segments')")
    json_data = _sanitize_asr_json(json_data)

    # Lưu vào DB
    speaker_names = json_data.get("speaker_names", {})
    db.update_file(
        file_id,
        status="completed",
        asr_result_json=json.dumps(json_data, ensure_ascii=False),
        speaker_names_json=json.dumps(speaker_names, ensure_ascii=False),
        model_used=json_data.get("model", "imported"),
        duration_sec=json_data.get("duration_sec", 0),
    )

    return {"success": True, "message": "JSON loaded"}


# === Process API ===

@app.post("/api/process/{file_id}")
async def process_file(
    file_id: int,
    request: Request,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    if file_record["status"] in ("processing", "queued"):
        raise HTTPException(400, "File đang được xử lý")

    # Lấy config từ request body
    try:
        body = await request.json()
    except Exception:
        body = {}

    config = {
        "model": body.get("model", server_config.get("default_asr_model")),
        "speaker_diarization": body.get("speaker_diarization", True),
        "speaker_model": body.get("speaker_model", server_config.get("default_speaker_model")),
        "num_speakers": body.get("num_speakers", 0),
        "punctuation_confidence": body.get("punctuation_confidence",
                                           int(server_config.get("default_punctuation_confidence"))),
        "case_confidence": body.get("case_confidence",
                                    int(server_config.get("default_case_confidence"))),
        "diarization_threshold": body.get("diarization_threshold",
                                          int(server_config.get("default_diarization_threshold"))),

        "rms_normalize": body.get("rms_normalize", False),
    }

    # Tạo meeting record cho user đã login (skip nếu đã có)
    if user and not db.get_meeting_by_file_id(file_id):
        meeting_name = body.get("meeting_name", "").strip()
        if not meeting_name:
            meeting_name = file_record["original_filename"]
        db.create_meeting(
            user_id=user["id"],
            file_id=file_id,
            meeting_name=meeting_name,
            original_filename=file_record["original_filename"],
            stored_filename=file_record["stored_filename"],
            file_size=file_record["file_size_bytes"],
        )

    result = queue_manager.add_to_queue(file_id, session_id, config)
    if "error" in result:
        raise HTTPException(400, result["error"])

    return result


@app.post("/api/cancel/{file_id}")
async def cancel_processing(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    ok = queue_manager.cancel(file_id)
    return {"success": ok}


# === File status & results ===

@app.get("/api/files/{file_id}/status")
async def file_status(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    position = db.get_queue_position(file_id)
    total_waiting = db.get_queue_total_waiting()

    return {
        "file_id": file_id,
        "status": file_record["status"],
        "queue_position": position,
        "queue_total": total_waiting,
    }


@app.get("/api/files/{file_id}/result")
async def file_result(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    if not file_record["asr_result_json"]:
        raise HTTPException(404, "Chưa có kết quả ASR")

    return json.loads(file_record["asr_result_json"])


# === Summarization API ===

@app.post("/api/files/{file_id}/summarize")
async def summarize_file(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Trigger tóm tắt cho file đã có ASR result. Đưa vào queue chung."""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    if not file_record.get("asr_result_json"):
        raise HTTPException(400, "Chưa có kết quả ASR để tóm tắt")

    # Kiểm tra summarizer có sẵn không
    if server_config.get("summarizer_enabled") != "1":
        raise HTTPException(404, "Chức năng tóm tắt chưa được bật")

    from web_service.summarizer import is_summarizer_available
    if not is_summarizer_available():
        raise HTTPException(404, "Model tóm tắt chưa được cấu hình hoặc không tồn tại")

    result = queue_manager.add_summarize_to_queue(file_id, session_id)
    if "error" in result:
        raise HTTPException(400, result["error"])

    return result


@app.get("/api/files/{file_id}/summary")
async def get_summary(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Lấy kết quả tóm tắt đã có."""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    if not file_record.get("summary_json"):
        raise HTTPException(404, "Chưa có tóm tắt")

    return json.loads(file_record["summary_json"])


@app.get("/api/summarizer/status")
async def summarizer_status():
    """Kiểm tra summarizer có sẵn không (cho frontend show/hide nút)."""
    enabled = server_config.get("summarizer_enabled") == "1"
    if not enabled:
        return {"available": False}

    from web_service.summarizer import is_summarizer_available
    available = is_summarizer_available()
    return {"available": available and enabled}


@app.post("/api/files/{file_id}/save-result")
async def save_file_result(
    file_id: int,
    request: Request,
    user: dict = Depends(require_auth),
    session_id: str = Depends(get_session_id),
):
    """Save edited ASR result from client (logged-in users only)"""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    body = await request.json()
    asr_data = body.get("asr_result")
    if not asr_data or not isinstance(asr_data, dict):
        raise HTTPException(400, "Missing asr_result")

    # P1 XSS: sanitize trước khi lưu (save-result từ client cũng cần validate)
    if "segments" not in asr_data:
        raise HTTPException(400, "asr_result thiếu 'segments'")
    asr_data = _sanitize_asr_json(asr_data)
    result_json = json.dumps(asr_data, ensure_ascii=False)
    db.update_file(file_id, asr_result_json=result_json)

    # Also update meeting if exists
    try:
        meeting = db.get_meeting_by_file_id(file_id)
        if meeting:
            db.update_meeting(meeting["id"], asr_result_json=result_json)
    except Exception:
        pass  # meeting update is best-effort

    return {"ok": True}


@app.get("/api/files/{file_id}/audio")
async def file_audio(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Stream audio file để phát trên web"""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    file_path = os.path.join(UPLOAD_DIR, file_record["stored_filename"])

    # Ưu tiên file WAV đã convert
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(wav_path):
        return FileResponse(wav_path, media_type="audio/wav")

    if os.path.exists(file_path):
        # Tra ve file goc
        ext = file_record["original_filename"].rsplit(".", 1)[-1].lower()
        media_types = {
            "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
            "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac",
            "wma": "audio/x-ms-wma", "opus": "audio/opus",
            "mp4": "video/mp4", "webm": "video/webm",
        }
        return FileResponse(file_path, media_type=media_types.get(ext, "application/octet-stream"))

    raise HTTPException(404, "Audio file not found")


@app.get("/api/files/{file_id}/download-json")
async def download_json(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    if not file_record["asr_result_json"]:
        raise HTTPException(404, "Chưa có kết quả ASR")

    original_name = file_record["original_filename"].rsplit(".", 1)[0]
    # RFC 5987: URL-encode filename để tránh header injection
    safe_filename = quote(f"{original_name}.asr.json")
    return Response(
        content=file_record["asr_result_json"],
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"},
    )


# === Queue status ===

@app.get("/api/queue/position/{file_id}")
async def queue_position(
    file_id: int,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    # A01: Kiểm tra quyền truy cập — chỉ owner mới xem được position
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)
    position = db.get_queue_position(file_id)
    total = db.get_queue_total_waiting()
    return {"position": position, "total": total}


# === Speaker operations ===

@app.post("/api/files/{file_id}/speakers")
async def update_speakers(
    file_id: int,
    request: Request,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Cập nhật speaker names"""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    body = await request.json()
    speaker_names = body.get("speaker_names", {})

    # Cập nhật speaker names trong kết quả JSON
    if file_record["asr_result_json"]:
        asr_data = json.loads(file_record["asr_result_json"])
        asr_data["speaker_names"] = speaker_names
        db.update_file(
            file_id,
            asr_result_json=json.dumps(asr_data, ensure_ascii=False),
            speaker_names_json=json.dumps(speaker_names, ensure_ascii=False),
        )

    return {"success": True}


@app.post("/api/files/{file_id}/split-speaker")
async def split_speaker(
    file_id: int,
    request: Request,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Tách người nói tại vị trí segment"""
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    body = await request.json()
    seg_index = body.get("seg_index")
    new_speaker = body.get("new_speaker", "")
    scope = body.get("scope", "to_end")  # "to_end" hoặc "single"

    if seg_index is None or not file_record["asr_result_json"]:
        raise HTTPException(400, "Invalid request")

    asr_data = json.loads(file_record["asr_result_json"])
    segments = asr_data.get("segments", [])

    # Tìm speaker_id mới (hoặc dùng tên đã có)
    speaker_names = asr_data.get("speaker_names", {})

    # Tìm speaker_id lớn nhất hiện có
    max_spk_id = -1
    for seg in segments:
        if seg.get("type") == "speaker":
            spk_id = seg.get("speaker_id", 0)
            if spk_id > max_spk_id:
                max_spk_id = spk_id

    # Kiểm tra new_speaker đã có chưa
    new_speaker_id = None
    for sid, name in speaker_names.items():
        if name == new_speaker:
            new_speaker_id = int(sid)
            break

    if new_speaker_id is None:
        new_speaker_id = max_spk_id + 1
        speaker_names[str(new_speaker_id)] = new_speaker

    # Áp dụng split
    # Tìm vị trí segment và speaker block hiện tại
    text_idx = 0
    target_pos = None
    current_block_sep = None  # speaker separator của block hiện tại

    for i, seg in enumerate(segments):
        if seg.get("type") == "speaker":
            current_block_sep = seg
        elif seg.get("type") == "text":
            if text_idx == seg_index:
                target_pos = i
                break
            text_idx += 1

    if target_pos is not None:
        # Chèn speaker separator trước segment này
        new_sep = {
            "type": "speaker",
            "speaker": new_speaker,
            "speaker_id": new_speaker_id,
            "start_time": segments[target_pos].get("start_time", 0),
        }
        segments.insert(target_pos, new_sep)

        if scope == "single" and current_block_sep:
            # Chèn separator phục hồi speaker gốc sau segment đơn
            # target_pos + 1 = new_sep, target_pos + 2 = the target text segment
            restore_pos = target_pos + 2
            # Tìm segment tiếp theo sau target để lấy start_time
            restore_time = 0
            if restore_pos < len(segments):
                restore_time = segments[restore_pos].get("start_time", 0)
            restore_sep = {
                "type": "speaker",
                "speaker": current_block_sep.get("speaker", ""),
                "speaker_id": current_block_sep.get("speaker_id", 0),
                "start_time": restore_time,
            }
            segments.insert(restore_pos, restore_sep)

    asr_data["segments"] = segments
    asr_data["speaker_names"] = speaker_names

    db.update_file(
        file_id,
        asr_result_json=json.dumps(asr_data, ensure_ascii=False),
        speaker_names_json=json.dumps(speaker_names, ensure_ascii=False),
    )

    return {"success": True, "result": asr_data}


@app.post("/api/files/{file_id}/merge-speaker")
async def merge_speaker(
    file_id: int,
    request: Request,
    session_id: str = Depends(get_session_id),
    user: Optional[dict] = Depends(get_current_user),
):
    """Gộp người nói với block trước/sau (hỗ trợ partial merge như desktop)

    Nếu seg_index được cung cấp:
      - merge up: gộp từ đầu block đến seg_index vào người nói trước
      - merge down: gộp từ seg_index đến cuối block vào người nói sau
    Nếu không có seg_index: gộp toàn bộ block.
    """
    file_record = db.get_file(file_id)
    check_file_access(file_record, session_id, user)

    body = await request.json()
    block_index = body.get("block_index")
    direction = body.get("direction", "up")  # "up" hoặc "down"
    seg_index = body.get("seg_index")  # text segment index (optional)

    if block_index is None or not file_record["asr_result_json"]:
        raise HTTPException(400, "Invalid request")

    asr_data = json.loads(file_record["asr_result_json"])
    segments = asr_data.get("segments", [])

    # Tìm speaker separators
    speaker_indices = [i for i, s in enumerate(segments) if s.get("type") == "speaker"]

    if block_index < 0 or block_index >= len(speaker_indices):
        raise HTTPException(400, "Invalid block index")

    current_sep_idx = speaker_indices[block_index]
    current_sep = segments[current_sep_idx]

    # Tìm vị trí thực của seg_index trong segments array
    actual_seg_pos = None
    if seg_index is not None:
        text_count = 0
        for i, s in enumerate(segments):
            if s.get("type") == "text":
                if text_count == seg_index:
                    actual_seg_pos = i
                    break
                text_count += 1

    # Tìm phạm vi text segments trong block hiện tại
    next_sep_idx = speaker_indices[block_index + 1] if block_index + 1 < len(speaker_indices) else len(segments)
    block_text_indices = [i for i in range(current_sep_idx + 1, next_sep_idx) if segments[i].get("type") == "text"]

    if direction == "up" and block_index > 0:
        # Gộp với block trước
        # Kiểm tra xem seg_index có phải segment cuối của block không
        is_last_in_block = (actual_seg_pos is None or
                            len(block_text_indices) == 0 or
                            actual_seg_pos >= block_text_indices[-1])

        if is_last_in_block:
            # Gộp toàn bộ block: xóa separator hiện tại
            segments.pop(current_sep_idx)
        else:
            # Partial merge: gộp từ đầu block đến seg_index vào người nói trước,
            # phần còn lại giữ nguyên speaker
            # 1. Xóa separator hiện tại (merge phần đầu vào block trước)
            segments.pop(current_sep_idx)
            # 2. Chèn separator mới sau seg_index để giữ phần còn lại
            # actual_seg_pos giảm 1 vì đã pop separator phía trước nó
            new_sep_pos = actual_seg_pos - 1
            insert_pos = new_sep_pos + 1
            restore_time = segments[insert_pos].get("start_time", 0) if insert_pos < len(segments) else 0
            segments.insert(insert_pos, {
                "type": "speaker",
                "speaker": current_sep.get("speaker", ""),
                "speaker_id": current_sep.get("speaker_id", 0),
                "start_time": restore_time,
            })

    elif direction == "down" and block_index < len(speaker_indices) - 1:
        # Gộp với block sau
        next_block_sep_idx = speaker_indices[block_index + 1]
        next_sep = segments[next_block_sep_idx]

        # Kiểm tra xem seg_index có phải segment đầu của block không
        is_first_in_block = (actual_seg_pos is None or
                             len(block_text_indices) == 0 or
                             actual_seg_pos <= block_text_indices[0])

        if is_first_in_block:
            # Gộp toàn bộ block: xóa separator của block sau
            segments.pop(next_block_sep_idx)
        else:
            # Partial merge: gộp từ seg_index đến cuối block vào người nói sau,
            # phần trước giữ nguyên speaker
            # 1. Xóa separator block sau
            segments.pop(next_block_sep_idx)
            # 2. Chèn separator block sau trước seg_index
            segments.insert(actual_seg_pos, {
                "type": "speaker",
                "speaker": next_sep.get("speaker", ""),
                "speaker_id": next_sep.get("speaker_id", 0),
                "start_time": segments[actual_seg_pos].get("start_time", 0) if actual_seg_pos < len(segments) else 0,
            })

    asr_data["segments"] = segments

    db.update_file(
        file_id,
        asr_result_json=json.dumps(asr_data, ensure_ascii=False),
    )

    return {"success": True, "result": asr_data}


# === Auth API ===

@app.post("/api/auth/login")
async def login(request: Request, session_id: str = Depends(get_session_id)):
    ip = request.client.host if request.client else ""
    _check_login_rate(ip)

    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    # Account lockout (chống brute force per username)
    _check_account_lockout(username)

    user = authenticate_user(username, password)
    if not user:
        _record_failed_login(ip)
        _record_failed_account(username)
        # A09: Không log username để tránh confirm username hợp lệ qua log
        logger.warning(f"Failed login attempt from {ip}")
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu")

    _clear_account_lockout(username)

    token = create_token(user["id"], user["username"], user["role"])
    # A09: Log user_id thay vì username
    logger.info(f"User id={user['id']} logged in from {ip}")

    # Session fixation prevention: tạo session mới thay vì giữ session cũ
    ua = request.headers.get("user-agent", "")
    new_session_id = session_manager.create_session(
        ip_address=ip, user_agent=ua, user_id=user["id"]
    )
    # Expire session anonymous cũ
    if session_id:
        old_session = db.get_session(session_id)
        if old_session and old_session["is_anonymous"]:
            db.expire_session(session_id)

    response = JSONResponse({
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        },
    })
    response.set_cookie(
        "session_id", new_session_id,
        httponly=True, samesite="lax", max_age=86400,
        secure=not server_config.http_mode,
    )
    return response


@app.get("/api/auth/me")
async def auth_me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "storage_limit_gb": user["storage_limit_gb"],
        "storage_used_bytes": user["storage_used_bytes"],
    }


@app.post("/api/auth/change-password")
async def auth_change_password(
    request: Request,
    user: Optional[dict] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(401, "Not authenticated")

    body = await request.json()
    old_password = body.get("old_password", "")
    new_password = body.get("new_password", "")

    if len(new_password) < 8:
        raise HTTPException(400, "Mật khẩu mới phải có ít nhất 8 ký tự")

    from web_service.auth import verify_password
    if not verify_password(old_password, user["password_hash"]):
        raise HTTPException(400, "Mật khẩu cũ không đúng")

    change_password(user["id"], new_password)
    return {"success": True}


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    """Logout: tạo session anonymous mới, tách khỏi session cũ (vẫn giữ processing)."""
    ip = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")

    # A07: Revoke JWT token hiện tại để không còn dùng được sau khi logout
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = decode_token(token)
        if payload and "exp" in payload:
            _revoke_token(token, float(payload["exp"]))

    new_session_id = session_manager.create_session(ip_address=ip, user_agent=ua)

    response = JSONResponse({
        "session_id": new_session_id,
        "is_anonymous": True,
    })
    response.set_cookie(
        "session_id", new_session_id,
        httponly=True, samesite="lax", max_age=86400,
        secure=not server_config.http_mode,
    )
    return response


# === User files (logged in) ===

@app.get("/api/user/files")
async def user_files(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Not authenticated")
    files = db.get_user_files(user["id"])
    return [
        {
            "id": f["id"],
            "filename": f["original_filename"],
            "size": f["file_size_bytes"],
            "status": f["status"],
            "created_at": f["created_at"],
            "duration_sec": f["duration_sec"],
        }
        for f in files
    ]


@app.delete("/api/user/files/{file_id}")
async def delete_user_file(
    file_id: int,
    user: Optional[dict] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(401, "Not authenticated")

    file_record = db.get_file(file_id)
    if not file_record or file_record["user_id"] != user["id"]:
        raise HTTPException(404, "File not found")

    # Xóa file vật lý
    file_path = os.path.join(UPLOAD_DIR, file_record["stored_filename"])
    for path in [file_path, file_path.rsplit(".", 1)[0] + ".wav"]:
        if os.path.exists(path):
            os.remove(path)

    db.delete_file(file_id)
    db.update_user_storage(user["id"])

    return {"success": True}


# === Meetings API ===

@app.get("/api/meetings")
async def list_meetings(
    search: str = Query(default=""),
    user: dict = Depends(require_auth),
):
    meetings = db.get_user_meetings(user["id"], search=search or None)
    return [
        {
            "id": m["id"],
            "meeting_name": m["meeting_name"],
            "original_filename": m["original_filename"],
            "status": m["status"],
            "file_size": m["file_size"],
            "created_at": m["created_at"],
            "updated_at": m["updated_at"],
            "error_message": m["error_message"],
        }
        for m in meetings
    ]


@app.get("/api/meetings/{meeting_id}")
async def get_meeting_detail(meeting_id: int, user: dict = Depends(require_auth)):
    meeting = db.get_meeting(meeting_id)
    if not meeting or meeting["user_id"] != user["id"]:
        raise HTTPException(404, "Meeting not found")

    # Đọc kết quả ASR mới nhất từ bảng files (luôn đồng bộ)
    file_record = db.get_file(meeting["file_id"])
    asr_result = None
    if file_record and file_record["asr_result_json"]:
        asr_result = json.loads(file_record["asr_result_json"])

    return {
        "id": meeting["id"],
        "meeting_name": meeting["meeting_name"],
        "original_filename": meeting["original_filename"],
        "file_id": meeting["file_id"],
        "status": meeting["status"],
        "file_size": meeting["file_size"],
        "created_at": meeting["created_at"],
        "error_message": meeting["error_message"],
        "asr_result": asr_result,
    }


@app.put("/api/meetings/{meeting_id}")
async def rename_meeting(meeting_id: int, request: Request, user: dict = Depends(require_auth)):
    meeting = db.get_meeting(meeting_id)
    if not meeting or meeting["user_id"] != user["id"]:
        raise HTTPException(404, "Meeting not found")

    body = await request.json()
    new_name = body.get("meeting_name", "").strip()
    if not new_name:
        raise HTTPException(400, "Tên cuộc họp không được để trống")

    db.update_meeting(meeting_id, meeting_name=new_name)
    return {"success": True}


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int, user: dict = Depends(require_auth)):
    meeting = db.get_meeting(meeting_id)
    if not meeting or meeting["user_id"] != user["id"]:
        raise HTTPException(404, "Meeting not found")

    # Không cho xóa nếu đang xử lý
    if meeting["status"] == "processing":
        raise HTTPException(400, "Không thể xóa cuộc họp đang xử lý")

    file_id = meeting["file_id"]

    # Xóa file vật lý
    import glob as glob_mod
    file_path = os.path.join(UPLOAD_DIR, meeting["stored_filename"])
    for f in glob_mod.glob(file_path + "*"):
        try:
            os.remove(f)
        except OSError:
            pass
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    # Xóa DB records
    db.delete_meeting(meeting_id)
    db.delete_file(file_id)
    db.update_user_storage(user["id"])

    return {"success": True}


@app.get("/api/meetings/{meeting_id}/audio")
async def meeting_audio(meeting_id: int, user: dict = Depends(require_auth)):
    meeting = db.get_meeting(meeting_id)
    if not meeting or meeting["user_id"] != user["id"]:
        raise HTTPException(404, "Meeting not found")

    file_path = os.path.join(UPLOAD_DIR, meeting["stored_filename"])

    # Ưu tiên file WAV đã convert
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(wav_path):
        return FileResponse(wav_path, media_type="audio/wav")

    if os.path.exists(file_path):
        ext = meeting["original_filename"].rsplit(".", 1)[-1].lower()
        media_types = {
            "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
            "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac",
            "wma": "audio/x-ms-wma", "opus": "audio/opus",
            "mp4": "video/mp4", "webm": "video/webm",
        }
        return FileResponse(file_path, media_type=media_types.get(ext, "application/octet-stream"))

    raise HTTPException(404, "Audio file not found")


# === Admin API ===

@app.get("/api/admin/stats")
async def admin_stats(request: Request, admin=Depends(require_admin)):
    return db.get_stats()


@app.get("/api/stats")
async def server_stats(request: Request):
    """Stats endpoint cho admin GUI (chỉ cho phép từ localhost)."""
    _require_localhost(request)
    return db.get_stats()


def _require_localhost(request: Request):
    """Chỉ cho phép từ localhost hoặc từ chính server bind IP (GUI admin cùng máy).
    Khi server bind vào 192.168.x.x, GUI kết nối qua IP đó → client_ip = bind IP."""
    client_ip = request.client.host if request.client else ""
    allowed = {"127.0.0.1", "::1", "localhost"}
    bind_host = server_config.get("host")
    if bind_host and bind_host not in ("0.0.0.0", "", "::"):
        allowed.add(bind_host)
    if client_ip not in allowed:
        raise HTTPException(403, "Only accessible from localhost")


# --- Localhost-only API cho GUI admin (không cần JWT) ---

@app.get("/api/local/sessions")
async def local_sessions(request: Request):
    _require_localhost(request)
    return db.get_all_sessions()


@app.delete("/api/local/sessions/{session_id}")
async def local_kill_session(session_id: str, request: Request):
    _require_localhost(request)
    session_manager.kill_session(session_id, kill_processes_callback=queue_manager.cancel)
    return {"success": True}


@app.post("/api/local/sessions/cleanup")
async def local_cleanup_sessions(request: Request):
    _require_localhost(request)
    cleaned = await session_manager.cleanup_expired(kill_processes_callback=queue_manager.cancel)
    return {"success": True, "cleaned_count": cleaned}


@app.get("/api/local/rate-limits")
async def local_rate_limits(request: Request):
    _require_localhost(request)
    return _get_locked_ips()


@app.post("/api/local/rate-limits/clear")
async def local_clear_rate_limits(request: Request):
    """Xóa tất cả IP bị khóa login."""
    _require_localhost(request)
    _clear_login_rate()
    return {"success": True}


@app.get("/api/local/queue")
async def local_queue(request: Request):
    _require_localhost(request)
    return db.get_all_queue()


@app.post("/api/local/queue/pause")
async def local_pause_queue(request: Request):
    _require_localhost(request)
    queue_manager.pause()
    return {"success": True}


@app.post("/api/local/queue/resume")
async def local_resume_queue(request: Request):
    _require_localhost(request)
    queue_manager.resume()
    return {"success": True}


@app.post("/api/local/queue/cancel/{file_id}")
async def local_cancel_queue_item(file_id: int, request: Request):
    _require_localhost(request)
    ok = queue_manager.cancel(file_id)
    return {"success": ok}


@app.get("/api/local/users")
async def local_list_users(request: Request):
    _require_localhost(request)
    return db.get_all_users()


@app.post("/api/local/users")
async def local_create_user(request: Request):
    _require_localhost(request)
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    storage_gb = float(body.get("storage_limit_gb", 5.0))
    if not username or len(username) < 2:
        raise HTTPException(400, "Username phải có ít nhất 2 ký tự")
    if len(password) < 8:
        raise HTTPException(400, "Mật khẩu phải có ít nhất 8 ký tự")
    existing = db.get_user_by_username(username)
    if existing:
        raise HTTPException(400, f"Username '{username}' đã tồn tại")
    user_id = db.create_user(username, hash_password(password), "user", storage_gb)
    logger.info(f"[local] Created user '{username}' (id={user_id})")
    return {"user_id": user_id, "username": username}


@app.post("/api/local/users/{user_id}/reset-password")
async def local_reset_password(user_id: int, request: Request):
    _require_localhost(request)
    body = await request.json()
    new_password = body.get("password", "")
    if len(new_password) < 8:
        raise HTTPException(400, "Mật khẩu phải có ít nhất 8 ký tự")
    change_password(user_id, new_password)
    logger.info(f"[local] Reset password for user_id={user_id}")
    return {"success": True}


@app.delete("/api/local/users/{user_id}")
async def local_delete_user(user_id: int, request: Request):
    _require_localhost(request)
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user["role"] == "admin":
        raise HTTPException(400, "Không thể xóa admin")
    db.delete_user(user_id)
    return {"success": True}


@app.put("/api/local/users/{user_id}")
async def local_update_user(user_id: int, request: Request):
    _require_localhost(request)
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    body = await request.json()
    updates = {}
    if "storage_limit_gb" in body:
        updates["storage_limit_gb"] = float(body["storage_limit_gb"])
    if "is_active" in body:
        # Không cho phép vô hiệu hóa admin
        if user["role"] == "admin" and not body["is_active"]:
            raise HTTPException(400, "Không thể vô hiệu hóa tài khoản admin")
        updates["is_active"] = bool(body["is_active"])
    if updates:
        db.update_user(user_id, **updates)
    return {"success": True}


@app.get("/api/admin/sessions")
async def admin_sessions(admin=Depends(require_admin)):
    return db.get_all_sessions()


@app.delete("/api/admin/sessions/{session_id}")
async def admin_kill_session(session_id: str, admin=Depends(require_admin)):
    session_manager.kill_session(session_id, kill_processes_callback=queue_manager.cancel)
    return {"success": True}


@app.post("/api/admin/sessions/cleanup")
async def admin_cleanup_sessions(admin=Depends(require_admin)):
    cleaned = await session_manager.cleanup_expired(kill_processes_callback=queue_manager.cancel)
    return {"success": True, "cleaned_count": cleaned}


@app.get("/api/admin/queue")
async def admin_queue(admin=Depends(require_admin)):
    return db.get_all_queue()


@app.post("/api/admin/queue/pause")
async def admin_pause_queue(admin=Depends(require_admin)):
    queue_manager.pause()
    return {"success": True}


@app.post("/api/admin/queue/resume")
async def admin_resume_queue(admin=Depends(require_admin)):
    queue_manager.resume()
    return {"success": True}


@app.post("/api/admin/queue/cancel/{file_id}")
async def admin_cancel_queue_item(file_id: int, admin=Depends(require_admin)):
    ok = queue_manager.cancel(file_id)
    return {"success": ok}


@app.get("/api/admin/users")
async def admin_list_users(admin=Depends(require_admin)):
    return db.get_all_users()


@app.post("/api/admin/users")
async def admin_create_user(request: Request, admin=Depends(require_admin)):
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    storage_gb = float(body.get("storage_limit_gb", 5.0))

    if not username or len(username) < 2:
        raise HTTPException(400, "Username phải có ít nhất 2 ký tự")
    if len(password) < 8:
        raise HTTPException(400, "Mật khẩu phải có ít nhất 8 ký tự")

    existing = db.get_user_by_username(username)
    if existing:
        raise HTTPException(400, f"Username '{username}' đã tồn tại")

    user_id = db.create_user(username, hash_password(password), "user", storage_gb)
    logger.info(f"Admin {admin['id']} created user '{username}' (id={user_id})")
    return {"user_id": user_id, "username": username}


@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: int, request: Request, admin=Depends(require_admin)):
    body = await request.json()
    updates = {}
    if "storage_limit_gb" in body:
        updates["storage_limit_gb"] = float(body["storage_limit_gb"])
    if "is_active" in body:
        updates["is_active"] = bool(body["is_active"])
    if updates:
        db.update_user(user_id, **updates)
    return {"success": True}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: int, request: Request, admin=Depends(require_admin)):
    body = await request.json()
    new_password = body.get("password", "")
    if len(new_password) < 8:
        raise HTTPException(400, "Mật khẩu phải có ít nhất 8 ký tự")
    change_password(user_id, new_password)
    logger.info(f"Admin {admin['id']} reset password for user_id={user_id}")
    return {"success": True}


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, admin=Depends(require_admin)):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user["role"] == "admin":
        raise HTTPException(400, "Khong the xoa admin")
    db.delete_user(user_id)
    logger.info(f"Admin {admin['id']} deleted user_id={user_id} ('{user['username']}')")
    return {"success": True}


@app.get("/api/admin/rate-limits")
async def admin_rate_limits(admin=Depends(require_admin)):
    return _get_locked_ips()


@app.post("/api/admin/rate-limits/clear")
async def admin_clear_rate_limits(admin=Depends(require_admin)):
    _clear_login_rate()
    return {"success": True}


@app.get("/api/admin/config")
async def admin_get_config(admin=Depends(require_admin)):
    return server_config.to_dict()


_CONFIG_VALIDATORS = {
    "port": lambda v: 1 <= int(v) <= 65535,
    "cpu_threads": lambda v: 1 <= int(v) <= 128,
    "max_upload_mb": lambda v: 1 <= int(v) <= 10000,
    "anonymous_timeout_minutes": lambda v: 1 <= int(v) <= 1440,
    "storage_per_user_gb": lambda v: 0 <= float(v) <= 1000,
    "max_sessions": lambda v: 1 <= int(v) <= 10000,
    "jwt_expire_minutes": lambda v: 5 <= int(v) <= 43200,
    "summarizer_threads": lambda v: 1 <= int(v) <= 128,
    "summarizer_context_size": lambda v: 1024 <= int(v) <= 262144,
    "summarizer_enabled": lambda v: v in ("0", "1"),
}
_CONFIG_READONLY = {"admin_password_hash", "host"}


@app.put("/api/admin/config")
async def admin_update_config(request: Request, admin=Depends(require_admin)):
    body = await request.json()
    changed = []
    for key, value in body.items():
        if key not in server_config.DEFAULTS or key in _CONFIG_READONLY:
            continue
        validator = _CONFIG_VALIDATORS.get(key)
        if validator:
            try:
                if not validator(value):
                    raise HTTPException(400, f"Giá trị không hợp lệ cho {key}")
            except (ValueError, TypeError):
                raise HTTPException(400, f"Giá trị không hợp lệ cho {key}")
        server_config.set(key, value)
        changed.append(key)
    server_config.save()
    # A09: Audit log — ghi nhận ai thay đổi config gì
    if changed:
        logger.info(f"Admin {admin['id']} updated config keys: {changed}")
    return {"success": True}


@app.post("/api/admin/download-summarizer-model")
async def admin_download_model(admin=Depends(require_admin)):
    """Tải model summarizer GGUF từ HuggingFace (chạy background)."""
    import asyncio
    from web_service.summarizer import download_model, get_default_model_path

    # Kiểm tra đã có chưa
    default_path = get_default_model_path()
    if os.path.isfile(default_path):
        return {"success": True, "path": default_path, "message": "Model đã tồn tại"}

    # Chạy download trong thread pool
    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, download_model)
        # Auto-set config
        server_config.set("summarizer_model_path", path)
        server_config.save()
        return {"success": True, "path": path}
    except Exception as e:
        raise HTTPException(500, f"Lỗi tải model: {str(e)}")


# === WebSocket ===

@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
    session_id: str = Query(default=""),
):
    if not session_id:
        session_id = ws.cookies.get("session_id", "")

    if not session_id or not db.get_session(session_id):
        await ws.close(code=4001, reason="Invalid session")
        return

    await ws_manager.connect(session_id, ws)

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "heartbeat":
                session_manager.heartbeat(session_id)
                await ws.send_json({"type": "heartbeat_ack"})

            elif msg_type == "subscribe_queue":
                file_id = data.get("file_id")
                if file_id:
                    pos = db.get_queue_position(file_id)
                    total = db.get_queue_total_waiting()
                    await ws.send_json({
                        "type": "queue_position",
                        "file_id": file_id,
                        "position": pos,
                        "total": total,
                    })

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id, ws)
        # Xu ly disconnect cho anonymous
        asyncio.create_task(
            session_manager.handle_disconnect(
                session_id,
                kill_processes_callback=queue_manager.cancel,
            )
        )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(session_id, ws)
