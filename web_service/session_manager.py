"""
Session lifecycle management - tạo, heartbeat, cleanup, WebSocket tracking.
"""

import os
import glob
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set

from fastapi import WebSocket

from web_service.config import server_config, UPLOAD_DIR
from web_service.database import db

logger = logging.getLogger("asr.session")


class WebSocketManager:
    """Quản lý tất cả WebSocket connections."""

    def __init__(self):
        # session_id -> set of WebSocket
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        if session_id not in self.connections:
            self.connections[session_id] = set()
        self.connections[session_id].add(ws)
        logger.info(f"WebSocket connected: session={session_id}")

    def disconnect(self, session_id: str, ws: WebSocket):
        if session_id in self.connections:
            self.connections[session_id].discard(ws)
            if not self.connections[session_id]:
                del self.connections[session_id]
        logger.info(f"WebSocket disconnected: session={session_id}")

    def is_connected(self, session_id: str) -> bool:
        return session_id in self.connections and len(self.connections[session_id]) > 0

    async def send_to_session(self, session_id: str, data: dict):
        """Gửi message đến tất cả WS của 1 session"""
        if session_id not in self.connections:
            return
        dead = []
        for ws in self.connections[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections[session_id].discard(ws)

    async def broadcast(self, data: dict):
        """Gửi đến tất cả sessions"""
        for session_id in list(self.connections.keys()):
            await self.send_to_session(session_id, data)

    def get_connected_session_ids(self) -> set:
        return set(self.connections.keys())


# Singleton
ws_manager = WebSocketManager()


class SessionManager:
    """Quản lý session lifecycle."""

    def __init__(self):
        self._cleanup_task = None

    def create_session(self, ip_address: str = "", user_agent: str = "",
                       user_id: int = None) -> str:
        session_id = str(uuid.uuid4())
        db.create_session(session_id, ip_address, user_agent, user_id)
        logger.info(f"Session created: {session_id} ip={ip_address} anonymous={user_id is None}")
        return session_id

    def heartbeat(self, session_id: str):
        db.update_heartbeat(session_id)

    def get_session(self, session_id: str) -> dict | None:
        return db.get_session(session_id)

    def kill_session(self, session_id: str, kill_processes_callback=None):
        """Kill session: hủy queue, xóa files (nếu anonymous), expire session"""
        session = db.get_session(session_id)
        if not session:
            return

        # Huy queue items + kill processing
        processing_file_ids = db.cancel_session_queue(session_id)
        if kill_processes_callback:
            for file_id in processing_file_ids:
                kill_processes_callback(file_id)

        # Nếu anonymous: xóa files vật lý + cache + DB records
        if session["is_anonymous"]:
            stored_files = db.delete_session_files(session_id)
            for filename in stored_files:
                base_path = os.path.join(UPLOAD_DIR, filename)
                # Xóa file gốc + mọi file phái sinh (*.wav, *.tmp, etc.) bằng glob
                try:
                    for f in glob.glob(base_path + "*"):
                        os.remove(f)
                        logger.debug(f"Deleted cache file: {f}")
                    # Xóa file gốc nếu chưa match glob (trường hợp tên đặc biệt)
                    if os.path.exists(base_path):
                        os.remove(base_path)
                except OSError as e:
                    logger.error(f"Error deleting file {base_path}: {e}")

        # Expire session
        db.expire_session(session_id)
        logger.info(f"Session killed: {session_id} anonymous={session['is_anonymous']}")

    def cleanup_on_startup(self, kill_processes_callback=None):
        """Server restart: xóa mọi anonymous session + files vật lý.
        Sau restart không ai còn kết nối nên an toàn xóa hết."""
        sessions = db.get_active_anonymous_sessions()
        for session in sessions:
            self.kill_session(session["id"], kill_processes_callback)
        if sessions:
            logger.info(f"Startup cleanup: killed {len(sessions)} anonymous sessions, files deleted")

    async def cleanup_expired(self, kill_processes_callback=None):
        """Quet va don dep sessions anonymous da timeout"""
        timeout = server_config.anonymous_timeout_minutes
        expired = db.get_expired_anonymous_sessions(timeout)

        cleaned_count = 0
        for session in expired:
            sid = session["id"]
            # Kiem tra con WS connection khong (phong truong hop heartbeat DB tre)
            if ws_manager.is_connected(sid):
                # Van con ket noi, cap nhat heartbeat
                db.update_heartbeat(sid)
                continue

            # Tinh thoi gian idle
            last_hb = session.get("last_heartbeat", "")
            try:
                hb_time = datetime.fromisoformat(last_hb) if last_hb else None
                idle_minutes = (datetime.now() - hb_time).total_seconds() / 60 if hb_time else 0
            except Exception:
                idle_minutes = 0

            logger.info(f"Cleaning up expired session: {sid} (idle {idle_minutes:.1f} minutes)")
            # Gửi thông báo expired cho client trước khi kill
            await ws_manager.send_to_session(sid, {
                "type": "session_expired", "reason": "timeout"
            })
            self.kill_session(sid, kill_processes_callback)
            cleaned_count += 1

        return cleaned_count

    async def cleanup_orphaned_login_sessions(self, kill_processes_callback=None):
        """Cleanup session login mồ côi:
        - Idle > 1h + không còn queue active -> expire (không xóa file)
        - Quá 24h -> force expire + cancel processing
        """
        orphaned = db.get_orphaned_login_sessions(idle_minutes=60, max_age_hours=24)
        cleaned_count = 0

        for session in orphaned:
            sid = session["id"]
            # Vẫn còn WS -> skip
            if ws_manager.is_connected(sid):
                db.update_heartbeat(sid)
                continue

            created = session.get("created_at", "")
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(created)).total_seconds() / 3600
            except Exception:
                age_hours = 0

            if age_hours >= 24:
                # Quá 24h -> force cancel processing + expire
                logger.info(f"Force expiring login session (age={age_hours:.1f}h): {sid}")
                self.kill_session(sid, kill_processes_callback)
            else:
                # Idle + no queue -> chỉ expire, không cancel gì
                logger.info(f"Expiring idle orphaned login session: {sid}")
                db.expire_session(sid)

            cleaned_count += 1

        return cleaned_count

    async def start_cleanup_loop(self, kill_processes_callback=None):
        """Chay cleanup moi 60 giay"""
        cycle = 0
        while True:
            try:
                # Moi 60 giay: cleanup anonymous
                cleaned = await self.cleanup_expired(kill_processes_callback)
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired anonymous sessions")

                # Moi 60 giay: cleanup login sessions mo coi
                cleaned_login = await self.cleanup_orphaned_login_sessions(kill_processes_callback)
                if cleaned_login > 0:
                    logger.info(f"Cleaned up {cleaned_login} orphaned login sessions")

                # Moi 60 chu ky (~1 gio): xoa row DB expired cu > 7 ngay
                cycle += 1
                if cycle >= 60:
                    cycle = 0
                    deleted = db.delete_old_expired_sessions(days=7)
                    if deleted > 0:
                        logger.info(f"Purged {deleted} old expired session records from DB")

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(60)

    def get_session_status(self, session_id: str) -> dict | None:
        """Trả về trạng thái chi tiết của session."""
        session = db.get_session(session_id)
        if not session:
            return None
        queue_item = db.get_queue_status_for_session(session_id)
        latest_file = db.get_latest_file_for_session(session_id)
        return {
            "session_id": session_id,
            "is_anonymous": bool(session["is_anonymous"]),
            "user_id": session.get("user_id"),
            "connected": ws_manager.is_connected(session_id),
            "queue_item": {
                "file_id": queue_item["file_id"],
                "status": queue_item["status"],
                "progress_percent": queue_item["progress_percent"],
                "progress_message": queue_item["progress_message"],
                "original_filename": queue_item["original_filename"],
            } if queue_item else None,
            "latest_file": {
                "file_id": latest_file["id"],
                "original_filename": latest_file["original_filename"],
                "status": latest_file["status"],
                "has_result": bool(latest_file["has_result"]),
            } if latest_file else None,
        }

    async def handle_disconnect(self, session_id: str, kill_processes_callback=None):
        """Xử lý khi WebSocket disconnect - đợi rồi cleanup nếu anonymous.
        Mobile browser suspend JS khi ẩn app nên cần grace period đủ dài."""
        session = db.get_session(session_id)
        if not session or not session["is_anonymous"]:
            return

        # Grace period = anonymous_timeout (10 phut) de dong bo voi cleanup loop
        # Mobile browser suspend JS khi an app, WS bi dong ngay lap tuc
        grace_seconds = server_config.anonymous_timeout_minutes * 60
        check_interval = 10
        elapsed = 0

        while elapsed < grace_seconds:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            # Da reconnect?
            if ws_manager.is_connected(session_id):
                return

        # Het grace period, chua reconnect -> kill
        # Kiem tra lan cuoi truoc khi kill
        if ws_manager.is_connected(session_id):
            return

        logger.info(f"Anonymous session disconnected for {grace_seconds}s, killing: {session_id}")
        self.kill_session(session_id, kill_processes_callback)


# Singleton
session_manager = SessionManager()
