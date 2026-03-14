"""
SQLite database - users, sessions, files, queue.
Dùng sqlite3 thuần (sync) với connection pool + in-memory cache.
"""

import os
import sqlite3
import json
import time
import threading
from queue import Queue, Empty
from contextlib import contextmanager
from datetime import datetime, timedelta

from web_service.config import DB_PATH

# === Schema ===

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    storage_limit_gb REAL DEFAULT 5.0,
    storage_used_bytes INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NULL,
    ip_address TEXT,
    user_agent TEXT,
    is_anonymous BOOLEAN DEFAULT 1,
    last_heartbeat TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    expired_at TEXT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id INTEGER NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    duration_sec REAL NULL,
    status TEXT DEFAULT 'uploaded',
    asr_result_json TEXT NULL,
    speaker_names_json TEXT NULL,
    model_used TEXT NULL,
    config_json TEXT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    priority REAL NOT NULL,
    status TEXT DEFAULT 'waiting',
    progress_percent INTEGER DEFAULT 0,
    progress_message TEXT DEFAULT '',
    config_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT NULL,
    completed_at TEXT NULL,
    FOREIGN KEY (file_id) REFERENCES files(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_anonymous ON sessions(is_anonymous, expired_at);
CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_queue_session ON queue(session_id);

CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    meeting_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    asr_result_json TEXT NULL,
    status TEXT DEFAULT 'waiting',
    error_message TEXT NULL,
    file_size INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (file_id) REFERENCES files(id)
);
CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_file ON meetings(file_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_sessions_id_active ON sessions(id, expired_at);
"""


class Database:
    """Thread-safe SQLite database wrapper with connection pool + cache."""

    POOL_SIZE = 8  # Số connection tối đa trong pool

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._pool: Queue = Queue(maxsize=self.POOL_SIZE)
        self._pool_lock = threading.Lock()
        # In-memory cache cho session lookups (hot path)
        self._session_cache: dict[str, tuple[dict, float]] = {}  # session_id -> (data, expire_ts)
        self._session_cache_lock = threading.Lock()
        self._SESSION_CACHE_TTL = 5.0  # seconds
        # User cache
        self._user_cache: dict = {}  # user_id -> (data, expire_ts)
        self._user_cache_lock = threading.Lock()
        self._USER_CACHE_TTL = 10.0  # seconds
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Tạo connection mới với PRAGMAs đã cấu hình."""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")  # Nhanh hơn FULL, vẫn safe với WAL
        conn.execute("PRAGMA cache_size=-8000")  # 8MB cache thay vì 2MB mặc định
        conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables trong RAM
        conn.execute("PRAGMA mmap_size=67108864")  # 64MB mmap
        return conn

    def _init_db(self):
        conn = self._create_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def connect(self):
        """Lấy connection từ pool, trả lại sau khi dùng."""
        conn = None
        try:
            conn = self._pool.get_nowait()
        except Empty:
            conn = self._create_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if conn:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    conn.close()  # Pool đầy, đóng connection

    # === Cache helpers ===

    def _invalidate_session_cache(self, session_id: str):
        with self._session_cache_lock:
            self._session_cache.pop(session_id, None)

    def _invalidate_user_cache(self, user_id: int = None):
        with self._user_cache_lock:
            if user_id:
                self._user_cache.pop(user_id, None)
            else:
                self._user_cache.clear()

    # === Users ===

    def create_user(self, username: str, password_hash: str, role: str = "user",
                    storage_limit_gb: float = 5.0) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, storage_limit_gb) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, storage_limit_gb),
            )
            return cursor.lastrowid

    def get_user_by_username(self, username: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                user = dict(row)
                with self._user_cache_lock:
                    self._user_cache[user["id"]] = (user, time.monotonic() + self._USER_CACHE_TTL)
                return user
            return None

    def get_user_by_id(self, user_id: int) -> dict | None:
        # Check cache
        now = time.monotonic()
        with self._user_cache_lock:
            cached = self._user_cache.get(user_id)
            if cached and cached[1] > now:
                return cached[0]
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row:
                user = dict(row)
                with self._user_cache_lock:
                    self._user_cache[user_id] = (user, now + self._USER_CACHE_TTL)
                return user
            return None

    def get_all_users(self) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT u.*, "
                "(SELECT COUNT(*) FROM files f WHERE f.user_id = u.id) as file_count "
                "FROM users u ORDER BY u.created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_user(self, user_id: int, **kwargs):
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [user_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE users SET {sets} WHERE id = ?", vals)
        self._invalidate_user_cache(user_id)

    def delete_user(self, user_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM users WHERE id = ? AND role != 'admin'", (user_id,))
        self._invalidate_user_cache(user_id)

    def update_user_storage(self, user_id: int):
        """Tính lại storage_used_bytes từ bảng files"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(file_size_bytes), 0) as total FROM files WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            conn.execute(
                "UPDATE users SET storage_used_bytes = ?, updated_at = ? WHERE id = ?",
                (row["total"], datetime.now().isoformat(), user_id),
            )
        self._invalidate_user_cache(user_id)

    # === Sessions ===

    def create_session(self, session_id: str, ip_address: str = "",
                       user_agent: str = "", user_id: int = None) -> str:
        is_anonymous = user_id is None
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, ip_address, user_agent, is_anonymous, last_heartbeat) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, user_id, ip_address, user_agent, is_anonymous,
                 datetime.now().isoformat()),
            )
        return session_id

    def update_heartbeat(self, session_id: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_heartbeat = ? WHERE id = ? AND expired_at IS NULL",
                (datetime.now().isoformat(), session_id),
            )

    def get_session(self, session_id: str) -> dict | None:
        # Hot path - check cache first
        now = time.monotonic()
        with self._session_cache_lock:
            cached = self._session_cache.get(session_id)
            if cached and cached[1] > now:
                return cached[0]
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND expired_at IS NULL", (session_id,)
            ).fetchone()
            if row:
                session = dict(row)
                with self._session_cache_lock:
                    self._session_cache[session_id] = (session, now + self._SESSION_CACHE_TTL)
                return session
            return None

    def get_active_session_count(self) -> int:
        """Đếm số sessions đang active."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE expired_at IS NULL"
            ).fetchone()
            return row["cnt"]

    def get_oldest_idle_anonymous_session(self) -> dict | None:
        """Lấy session anonymous idle lâu nhất (để kick nhường chỗ)."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE is_anonymous = 1 AND expired_at IS NULL "
                "ORDER BY last_heartbeat ASC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_all_sessions(self) -> list:
        """Lấy tất cả sessions đang hoạt động (chưa hết hạn)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT s.*, u.username, "
                "(SELECT COUNT(*) FROM files f WHERE f.session_id = s.id) as file_count "
                "FROM sessions s LEFT JOIN users u ON s.user_id = u.id "
                "WHERE s.expired_at IS NULL "
                "ORDER BY s.created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def expire_session(self, session_id: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET expired_at = ? WHERE id = ?",
                (datetime.now().isoformat(), session_id),
            )
        self._invalidate_session_cache(session_id)

    def get_expired_anonymous_sessions(self, timeout_minutes: int) -> list:
        """Lấy sessions anonymous đã quá timeout"""
        cutoff = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE is_anonymous = 1 AND expired_at IS NULL "
                "AND last_heartbeat < ?",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_orphaned_login_sessions(self, idle_minutes: int = 60, max_age_hours: int = 24) -> list:
        """Lấy sessions login mồ côi: idle > idle_minutes VÀ không còn queue active.
        Hoặc sessions login quá max_age_hours bất kể trạng thái."""
        idle_cutoff = (datetime.now() - timedelta(minutes=idle_minutes)).isoformat()
        age_cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT s.* FROM sessions s "
                "WHERE s.is_anonymous = 0 AND s.expired_at IS NULL AND ("
                "  (s.last_heartbeat < ? AND NOT EXISTS ("
                "    SELECT 1 FROM queue q WHERE q.session_id = s.id "
                "    AND q.status IN ('waiting', 'processing')"
                "  ))"
                "  OR s.created_at < ?"
                ")",
                (idle_cutoff, age_cutoff),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_old_expired_sessions(self, days: int = 7) -> int:
        """Xóa hẳn sessions đã expired quá X ngày khỏi DB."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self.connect() as conn:
            # Xóa queue records liên quan trước (FK constraint)
            conn.execute(
                "DELETE FROM queue WHERE session_id IN ("
                "  SELECT id FROM sessions WHERE expired_at IS NOT NULL AND expired_at < ?"
                ")", (cutoff,),
            )
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expired_at IS NOT NULL AND expired_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def link_session_to_user(self, session_id: str, user_id: int):
        """Liên kết session với user khi login"""
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET user_id = ?, is_anonymous = 0 WHERE id = ?",
                (user_id, session_id),
            )
            # Chuyển files của session sang user
            conn.execute(
                "UPDATE files SET user_id = ? WHERE session_id = ? AND user_id IS NULL",
                (user_id, session_id),
            )
        self._invalidate_session_cache(session_id)

    # === Files ===

    def create_file(self, session_id: str, original_filename: str, stored_filename: str,
                    file_size_bytes: int, user_id: int = None) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO files (session_id, user_id, original_filename, stored_filename, file_size_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, original_filename, stored_filename, file_size_bytes),
            )
            return cursor.lastrowid

    def get_file(self, file_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
            return dict(row) if row else None

    def get_session_files(self, session_id: str) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user_files(self, user_id: int) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_file(self, file_id: int, **kwargs):
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [file_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE files SET {sets} WHERE id = ?", vals)

    def delete_file(self, file_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM queue WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def delete_session_files(self, session_id: str) -> list:
        """Xóa tất cả files của session, trả về danh sách stored_filename để xóa file vật lý"""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT stored_filename FROM files WHERE session_id = ?", (session_id,)
            ).fetchall()
            filenames = [r["stored_filename"] for r in rows]
            conn.execute("DELETE FROM queue WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM files WHERE session_id = ?", (session_id,))
            return filenames

    # === Queue ===

    def add_to_queue(self, file_id: int, session_id: str, config: dict) -> int:
        priority = time.time()  # FIFO theo thời gian
        with self.connect() as conn:
            # Remove old queue record if exists (reprocess)
            conn.execute("DELETE FROM queue WHERE file_id = ?", (file_id,))
            cursor = conn.execute(
                "INSERT INTO queue (file_id, session_id, priority, config_json) VALUES (?, ?, ?, ?)",
                (file_id, session_id, priority, json.dumps(config)),
            )
            # Cập nhật status file
            conn.execute("UPDATE files SET status = 'queued' WHERE id = ?", (file_id,))
            return cursor.lastrowid

    def get_next_queue_item(self) -> dict | None:
        """Lấy item đầu tiên trong queue đang waiting"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT q.*, f.stored_filename, f.original_filename "
                "FROM queue q JOIN files f ON q.file_id = f.id "
                "WHERE q.status = 'waiting' ORDER BY q.priority ASC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_queue_position(self, file_id: int) -> int:
        """Vị trí trong hàng đợi (1-based). 0 = đang processing, -1 = không trong queue"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status, priority FROM queue WHERE file_id = ?", (file_id,)
            ).fetchone()
            if not row:
                return -1
            if row["status"] == "processing":
                return 0
            if row["status"] != "waiting":
                return -1
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM queue WHERE status = 'waiting' AND priority < ?",
                (row["priority"],),
            ).fetchone()
            return count["cnt"] + 1

    def get_queue_total_waiting(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM queue WHERE status = 'waiting'").fetchone()
            return row["cnt"]

    def update_queue_progress(self, file_id: int, percent: int, message: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE queue SET progress_percent = ?, progress_message = ? WHERE file_id = ?",
                (percent, message, file_id),
            )

    def set_queue_processing(self, file_id: int):
        now = datetime.now().isoformat()
        with self.connect() as conn:
            conn.execute(
                "UPDATE queue SET status = 'processing', started_at = ? WHERE file_id = ?",
                (now, file_id),
            )
            conn.execute("UPDATE files SET status = 'processing' WHERE id = ?", (file_id,))

    def set_queue_completed(self, file_id: int):
        now = datetime.now().isoformat()
        with self.connect() as conn:
            conn.execute(
                "UPDATE queue SET status = 'completed', completed_at = ?, progress_percent = 100 "
                "WHERE file_id = ?",
                (now, file_id),
            )
            conn.execute(
                "UPDATE files SET status = 'completed', completed_at = ? WHERE id = ?",
                (now, file_id),
            )

    def set_queue_error(self, file_id: int, error_msg: str):
        now = datetime.now().isoformat()
        with self.connect() as conn:
            conn.execute(
                "UPDATE queue SET status = 'error', progress_message = ?, completed_at = ? "
                "WHERE file_id = ?",
                (error_msg, now, file_id),
            )
            conn.execute("UPDATE files SET status = 'error' WHERE id = ?", (file_id,))

    def set_queue_cancelled(self, file_id: int):
        now = datetime.now().isoformat()
        with self.connect() as conn:
            conn.execute(
                "UPDATE queue SET status = 'cancelled', completed_at = ? WHERE file_id = ?",
                (now, file_id),
            )
            conn.execute("UPDATE files SET status = 'cancelled' WHERE id = ?", (file_id,))

    def remove_from_queue(self, file_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM queue WHERE file_id = ?", (file_id,))
            conn.execute(
                "UPDATE files SET status = 'uploaded' WHERE id = ? AND status = 'queued'",
                (file_id,),
            )

    def has_session_in_queue(self, session_id: str) -> bool:
        """Kiểm tra session có file đang waiting/processing trong queue không"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM queue WHERE session_id = ? AND status IN ('waiting', 'processing')",
                (session_id,),
            ).fetchone()
            return row["cnt"] > 0

    def cleanup_stale_queue(self):
        """Reset 'processing' items to 'error' after server restart (stale from previous run)."""
        now = datetime.now().isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT file_id FROM queue WHERE status = 'processing'"
            ).fetchall()
            stale_ids = [r["file_id"] for r in rows]
            if stale_ids:
                conn.execute(
                    "UPDATE queue SET status = 'error', progress_message = 'Server restarted', completed_at = ? "
                    "WHERE status = 'processing'",
                    (now,),
                )
                conn.execute(
                    f"UPDATE files SET status = 'error' WHERE id IN ({','.join('?' for _ in stale_ids)})",
                    stale_ids,
                )
        return stale_ids

    def get_all_queue(self) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT q.*, f.original_filename, f.stored_filename, "
                "s.ip_address, s.is_anonymous, u.username "
                "FROM queue q "
                "JOIN files f ON q.file_id = f.id "
                "LEFT JOIN sessions s ON q.session_id = s.id "
                "LEFT JOIN users u ON s.user_id = u.id "
                "ORDER BY q.priority ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def cancel_session_queue(self, session_id: str) -> list:
        """Hủy tất cả queue items của session, trả về file_ids đang processing"""
        with self.connect() as conn:
            # Lay file_ids dang processing
            rows = conn.execute(
                "SELECT file_id FROM queue WHERE session_id = ? AND status = 'processing'",
                (session_id,),
            ).fetchall()
            processing_file_ids = [r["file_id"] for r in rows]

            # Hủy tất cả waiting
            conn.execute(
                "UPDATE queue SET status = 'cancelled' WHERE session_id = ? AND status = 'waiting'",
                (session_id,),
            )
            conn.execute(
                "UPDATE files SET status = 'cancelled' WHERE session_id = ? AND status = 'queued'",
                (session_id,),
            )
            return processing_file_ids

    def get_all_waiting_queue_items(self) -> list:
        """Lấy tất cả queue items đang waiting (để broadcast position)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT q.file_id, q.session_id FROM queue q "
                "WHERE q.status = 'waiting' ORDER BY q.priority ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_queue_status_for_session(self, session_id: str) -> dict | None:
        """Trả về queue item đang active (waiting/processing) của session."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT q.*, f.original_filename FROM queue q "
                "JOIN files f ON q.file_id = f.id "
                "WHERE q.session_id = ? AND q.status IN ('waiting', 'processing') "
                "ORDER BY q.priority ASC LIMIT 1",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_latest_file_for_session(self, session_id: str) -> dict | None:
        """Lấy file gần nhất của session (bất kể status) để restore UI."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, original_filename, status, asr_result_json IS NOT NULL as has_result "
                "FROM files WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    # === Meetings ===

    def create_meeting(self, user_id: int, file_id: int, meeting_name: str,
                       original_filename: str, stored_filename: str, file_size: int = 0) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO meetings (user_id, file_id, meeting_name, original_filename, "
                "stored_filename, file_size) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, file_id, meeting_name, original_filename, stored_filename, file_size),
            )
            return cursor.lastrowid

    def get_meeting(self, meeting_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
            return dict(row) if row else None

    def get_meeting_by_file_id(self, file_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM meetings WHERE file_id = ?", (file_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_user_meetings(self, user_id: int, search: str = None) -> list:
        with self.connect() as conn:
            if search:
                pattern = f"%{search}%"
                rows = conn.execute(
                    "SELECT * FROM meetings WHERE user_id = ? "
                    "AND (meeting_name LIKE ? OR original_filename LIKE ?) "
                    "ORDER BY created_at DESC",
                    (user_id, pattern, pattern),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM meetings WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def update_meeting(self, meeting_id: int, **kwargs):
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [meeting_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE meetings SET {sets} WHERE id = ?", vals)

    def delete_meeting(self, meeting_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))

    # === Stats ===

    def get_stats(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM sessions WHERE expired_at IS NULL) as active_sessions, "
                "(SELECT COUNT(*) FROM queue WHERE status = 'waiting') as queue_waiting, "
                "(SELECT COUNT(*) FROM queue WHERE status = 'processing') as queue_processing, "
                "(SELECT COUNT(*) FROM files WHERE status = 'completed' AND completed_at >= ?) as completed_today, "
                "(SELECT COUNT(*) FROM users) as total_users",
                (today,),
            ).fetchone()

        return dict(row)


# Singleton
db = Database()
