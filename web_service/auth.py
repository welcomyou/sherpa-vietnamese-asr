"""
Authentication - JWT token, password hashing, admin setup.
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

from web_service.config import server_config
from web_service.database import db

# JWT
SECRET_KEY = os.environ.get("ASR_JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Hash password dùng SHA-256 + salt."""
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${h}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password."""
    if "$" not in hashed:
        return False
    salt, h = hashed.split("$", 1)
    return hashlib.sha256((salt + plain).encode("utf-8")).hexdigest() == h


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
    print(f"[Auth] Admin account created (id={admin_id})")
    return db.get_user_by_id(admin_id)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Xác thực user. Trả về user dict hoặc None."""
    user = db.get_user_by_username(username)
    if not user:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def change_password(user_id: int, new_password: str):
    db.update_user(user_id, password_hash=hash_password(new_password))
