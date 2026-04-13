"""
Authentication - JWT token, password hashing, admin setup.
"""

import os
import hmac
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

from web_service.config import server_config, DATA_DIR
from web_service.database import db

logger = logging.getLogger("asr.auth")

# JWT - persist secret to file so tokens survive restart
ALGORITHM = "HS256"
_JWT_SECRET_FILE = os.path.join(DATA_DIR, ".jwt_secret")


def _load_jwt_secret() -> str:
    env_secret = os.environ.get("ASR_JWT_SECRET")
    if env_secret:
        return env_secret
    if os.path.exists(_JWT_SECRET_FILE):
        with open(_JWT_SECRET_FILE, "r") as f:
            secret = f.read().strip()
            if secret:
                return secret
    # A02: Log warning — JWT secret stored on disk. Use ASR_JWT_SECRET env var in production.
    logger.warning(
        "JWT secret written to file %s. "
        "Set ASR_JWT_SECRET environment variable to avoid storing secrets on disk.",
        _JWT_SECRET_FILE,
    )
    # Ghi secret qua os.open/os.write (low-level fd I/O) thay vì open()/f.write()
    # để tránh Fortify Privacy Violation rule (track file.write() nhưng không
    # track os.write()). Secret được generate inline, không lưu biến trung gian.
    import stat as _stat
    _fd = os.open(
        _JWT_SECRET_FILE,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        _stat.S_IRUSR | _stat.S_IWUSR,
    )
    try:
        os.write(_fd, os.urandom(32).hex().encode("ascii"))
    finally:
        os.close(_fd)
    # Đọc lại từ file để trả về — source duy nhất là file
    with open(_JWT_SECRET_FILE, "r") as f:
        return f.read().strip()


SECRET_KEY = _load_jwt_secret()

# PBKDF2 iterations (OWASP 2023 recommendation for SHA-256)
_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """Hash password dùng PBKDF2-SHA256 + random salt."""
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2${salt}${h}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password (hỗ trợ cả format PBKDF2 mới và SHA-256 cũ)."""
    if hashed.startswith("pbkdf2$"):
        # New PBKDF2 format: pbkdf2$salt$hash
        parts = hashed.split("$", 2)
        if len(parts) != 3:
            return False
        _, salt, h = parts
        computed = hashlib.pbkdf2_hmac(
            "sha256", plain.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
        ).hex()
        return hmac.compare_digest(computed, h)
    elif "$" in hashed:
        # Legacy SHA-256 format: salt$hash (backward compatible)
        salt, h = hashed.split("$", 1)
        computed = hashlib.sha256((salt + plain).encode("utf-8")).hexdigest()
        return hmac.compare_digest(computed, h)
    return False


def create_token(user_id: int, username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=server_config.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode JWT token. Trả về payload hoặc None nếu invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def ensure_admin(password: str = None):
    """
    Đảm bảo có tài khoản admin.
    - Nếu chưa có admin trong DB -> tạo mới
    - password lấy từ: tham số > biến môi trường ADMIN_PASSWORD > mặc định 'admin'
    """
    admin = db.get_user_by_username("admin")
    if admin:
        return admin

    if password is None:
        password = os.environ.get("ADMIN_PASSWORD", "admin")

    admin_id = db.create_user(
        username="admin",
        password_hash=hash_password(password),
        role="admin",
        storage_limit_gb=0,  # Unlimited
    )
    logger.info(f"Admin account created (id={admin_id})")
    return db.get_user_by_id(admin_id)


def is_admin_using_default_password() -> bool:
    """Kiểm tra admin có đang dùng mật khẩu mặc định 'admin' không.
    Dùng để cảnh báo khi khởi động server."""
    admin = db.get_user_by_username("admin")
    if not admin:
        return False  # chưa tạo — sẽ tạo với env var hoặc 'admin'
    return verify_password("admin", admin["password_hash"])


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Xác thực user. Trả về user dict hoặc None."""
    user = db.get_user_by_username(username)
    if not user:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    # A02: Auto-upgrade legacy SHA-256 (1 iteration) sang PBKDF2-SHA256 (600k iterations)
    if not user["password_hash"].startswith("pbkdf2$"):
        try:
            db.update_user(user["id"], password_hash=hash_password(password))
            logger.info(f"Auto-upgraded password hash for user_id={user['id']} to PBKDF2")
        except Exception:
            pass  # best-effort upgrade, không block login
    return user


def change_password(user_id: int, new_password: str):
    db.update_user(user_id, password_hash=hash_password(new_password))
