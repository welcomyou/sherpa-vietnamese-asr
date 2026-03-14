"""
Server Admin GUI - PyQt6, 7 Tabs.
Quan ly FastAPI server process, sessions, queue, users, config, logs, Windows Service.
"""

import os
import sys
import subprocess
import json
import logging
import time
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QComboBox, QSpinBox,
    QGroupBox, QFormLayout, QCheckBox, QMessageBox, QHeaderView,
    QDialog, QDialogButtonBox, QProgressBar, QDoubleSpinBox,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.config import ALLOWED_THREADS, COLORS

logger = logging.getLogger("asr.gui")


class _LocalAPI:
    """Helper gọi localhost-only API từ GUI (không cần JWT auth)."""

    def __init__(self, port=8443):
        self._port = port
        self._host = "127.0.0.1"

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        # 0.0.0.0 listens on all interfaces → connect via 127.0.0.1
        self._host = "127.0.0.1" if value == "0.0.0.0" else value

    def _ssl_ctx(self):
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def request(self, method, path, body=None):
        """Gọi API localhost, trả về dict/list hoặc raise Exception."""
        import urllib.request
        url = f"https://{self._host}:{self._port}{path}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        resp = urllib.request.urlopen(req, timeout=10, context=self._ssl_ctx())
        return json.loads(resp.read())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body=None):
        return self.request("POST", path, body)

    def put(self, path, body=None):
        return self.request("PUT", path, body)

    def delete(self, path):
        return self.request("DELETE", path)


class ServerProcess:
    """Quan ly FastAPI server subprocess."""

    def __init__(self):
        self.process = None
        self._start_time = None

    def start(self, host="0.0.0.0", port=8443):
        if self.is_running():
            return
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "server_launcher.py"),
            "--host", host, "--port", str(port), "--no-gui",
        ]
        # KHÔNG dùng subprocess.PIPE vì không có ai đọc pipe -> buffer đầy -> block forever
        # Logs da ghi vao file (server.log), LogsTab doc tu file do
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=BASE_DIR, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._start_time = time.time()
        logger.info(f"Server started PID={self.process.pid}")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self._start_time = None
            logger.info("Server stopped")

    def restart(self, host, port):
        self.stop()
        time.sleep(1)
        self.start(host, port)

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def uptime(self):
        if not self._start_time or not self.is_running():
            return ""
        secs = int(time.time() - self._start_time)
        hours, rem = divmod(secs, 3600)
        mins, _ = divmod(rem, 60)
        return f"{hours} giờ {mins} phút"


class BaseTab(QWidget):
    """Base class cho tat ca tabs."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        raise NotImplementedError

    def refresh(self):
        pass


class _StatsFetcher(QThread):
    """Background thread de fetch stats tu API, khong block GUI."""
    result = pyqtSignal(dict)

    def __init__(self, api: _LocalAPI):
        super().__init__()
        self._api = api

    def run(self):
        try:
            data = self._api.get("/api/stats")
            self.result.emit(data)
        except Exception:
            pass


class _ApiFetcher(QThread):
    """Background thread gọi local API, trả về list."""
    result = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api: _LocalAPI, path: str):
        super().__init__()
        self._api = api
        self._path = path

    def run(self):
        try:
            data = self._api.get(self._path)
            if isinstance(data, list):
                self.result.emit(data)
            else:
                self.result.emit([data])
        except Exception as e:
            self.error.emit(str(e))


class StatusTab(BaseTab):
    """Tab 1: Server Status & Control"""

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Status
        status_group = QGroupBox("Trạng thái Server")
        sg = QFormLayout()

        self.lbl_status = QLabel("Đang dừng")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #dc3545;")
        sg.addRow("Trạng thái:", self.lbl_status)

        self.lbl_uptime = QLabel("")
        sg.addRow("Thời gian chạy (Uptime):", self.lbl_uptime)

        status_group.setLayout(sg)
        layout.addWidget(status_group)

        # Controls
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Bắt đầu Server")
        self.btn_start.setStyleSheet("background: #28a745; color: white; padding: 8px 16px;")
        self.btn_start.clicked.connect(self.start_server)

        self.btn_stop = QPushButton("Dừng Server")
        self.btn_stop.setStyleSheet("background: #dc3545; color: white; padding: 8px 16px;")
        self.btn_stop.clicked.connect(self.stop_server)

        self.btn_restart = QPushButton("Khởi động lại")
        self.btn_restart.setStyleSheet("padding: 8px 16px;")
        self.btn_restart.clicked.connect(self.restart_server)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_restart)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Config
        config_group = QGroupBox("Cấu hình nhanh")
        cg = QFormLayout()

        self.edit_port = QSpinBox()
        self.edit_port.setRange(1, 65535)
        self.edit_port.setValue(8443)
        cg.addRow("Port:", self.edit_port)

        self.spin_cpu = QSpinBox()
        self.spin_cpu.setRange(1, ALLOWED_THREADS)
        self.spin_cpu.setValue(min(4, ALLOWED_THREADS))
        cg.addRow(f"CPU Threads (max {ALLOWED_THREADS}):", self.spin_cpu)

        self.combo_bind = QComboBox()
        self.combo_bind.addItems(["0.0.0.0", "127.0.0.1"])
        cg.addRow("Địa chỉ Bind:", self.combo_bind)

        config_group.setLayout(cg)
        layout.addWidget(config_group)

        # Stats
        stats_group = QGroupBox("Thống kê")
        stats_layout = QHBoxLayout()
        self.lbl_sessions = QLabel("0")
        self.lbl_queue = QLabel("0")
        self.lbl_processed = QLabel("0")

        for label, value_lbl, title in [
            (self.lbl_sessions, "Sessions", "đang hoạt động"),
            (self.lbl_queue, "Queue", "đang chờ"),
            (self.lbl_processed, "Processed", "hôm nay"),
        ]:
            card = QVBoxLayout()
            v = QLabel("0")
            v.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff;")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t = QLabel(f"{title}")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet("color: #999;")
            card.addWidget(v)
            card.addWidget(t)
            stats_layout.addLayout(card)
            if title == "đang hoạt động":
                self.lbl_sessions = v
            elif title == "đang chờ":
                self.lbl_queue = v
            else:
                self.lbl_processed = v

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

    def start_server(self):
        host = self.combo_bind.currentText()
        port = self.edit_port.value()
        self.main_window.local_api.port = port
        self.main_window.local_api.host = host
        self.main_window.server.start(host, port)
        self.refresh()

    def stop_server(self):
        self.main_window.server.stop()
        self.refresh()

    def restart_server(self):
        host = self.combo_bind.currentText()
        port = self.edit_port.value()
        self.main_window.local_api.port = port
        self.main_window.local_api.host = host
        self.main_window.server.restart(host, port)
        self.refresh()

    def refresh(self):
        running = self.main_window.server.is_running()
        if running:
            self.lbl_status.setText("Đang chạy")
            self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #28a745;")
            self.lbl_uptime.setText(self.main_window.server.uptime())
            # Fetch stats in background thread (không block GUI)
            self._fetch_stats()
        else:
            self.lbl_status.setText("Đang dừng")
            self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #dc3545;")
            self.lbl_uptime.setText("")

    def _fetch_stats(self):
        """Fetch stats từ API trong background thread."""
        if hasattr(self, '_stats_thread') and self._stats_thread.isRunning():
            return  # Đang fetch, bỏ qua
        self._stats_thread = _StatsFetcher(self.main_window.local_api)
        self._stats_thread.result.connect(self._on_stats)
        self._stats_thread.start()

    def _on_stats(self, data):
        self.lbl_sessions.setText(str(data.get("active_sessions", 0)))
        self.lbl_queue.setText(str(data.get("queue_waiting", 0)))
        self.lbl_processed.setText(str(data.get("completed_today", 0)))


class SessionsTab(BaseTab):
    """Tab 2: Sessions"""

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Sessions đang hoạt động"))
        toolbar.addStretch()
        btn_refresh = QPushButton("Làm mới")
        btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID Phiên", "IP", "Người dùng", "Trạng thái", "Heartbeat cuối"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.chk_select_all = QCheckBox("Chọn tất cả")
        self.chk_select_all.stateChanged.connect(self._toggle_select_all)
        btn_layout.addWidget(self.chk_select_all)

        btn_kill = QPushButton("Ngắt phiên đã chọn")
        btn_kill.setStyleSheet("background: #dc3545; color: white;")
        btn_kill.clicked.connect(self.kill_selected)
        btn_layout.addWidget(btn_kill)

        btn_cleanup = QPushButton("Dọn dẹp phiên hết hạn")
        btn_cleanup.clicked.connect(self.cleanup_expired)
        btn_layout.addWidget(btn_cleanup)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def refresh(self):
        if not self.main_window.server.is_running():
            return
        api = self.main_window.local_api
        if hasattr(self, '_fetcher') and self._fetcher.isRunning():
            return
        self._fetcher = _ApiFetcher(api, "/api/local/sessions")
        self._fetcher.result.connect(self._on_sessions)
        self._fetcher.error.connect(lambda e: self.main_window.statusBar().showMessage(f"Lỗi: {e}", 5000))
        self._fetcher.start()

    def _toggle_select_all(self, state):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(
                    Qt.CheckState.Checked if state == Qt.CheckState.Checked.value
                    else Qt.CheckState.Unchecked
                )

    def _on_sessions(self, sessions: list):
        from datetime import datetime
        from web_service.config import server_config

        self.table.setRowCount(len(sessions))

        try:
            anon_timeout = int(server_config.get("anonymous_timeout_minutes", 10))
        except Exception:
            anon_timeout = 10
        logged_in_timeout = 60

        now = datetime.now()

        for i, s in enumerate(sessions):
            sid = s.get("id", "")
            # Cột 0: ID Phiên - hiện đầy đủ + checkbox
            id_item = QTableWidgetItem(sid)
            id_item.setData(Qt.ItemDataRole.UserRole, sid)
            id_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, id_item)

            self.table.setItem(i, 1, QTableWidgetItem(s.get("ip_address", "")))

            is_anonymous = s.get("is_anonymous", False)
            user = s.get("username") or ("Ẩn danh" if is_anonymous else "N/A")
            self.table.setItem(i, 2, QTableWidgetItem(user))

            # Trạng thái dựa trên heartbeat
            expired_at = s.get("expired_at")
            last_hb_str = s.get("last_heartbeat", "")
            status_text = "Hoạt động"
            status_color = "#28a745"

            if expired_at:
                status_text = "Hết hạn"
                status_color = "#dc3545"
            elif last_hb_str:
                try:
                    last_hb = datetime.fromisoformat(last_hb_str)
                    if last_hb.tzinfo is not None:
                        last_hb = last_hb.replace(tzinfo=None)
                    idle_minutes = (now - last_hb).total_seconds() / 60
                    timeout_threshold = anon_timeout if is_anonymous else logged_in_timeout

                    if idle_minutes > timeout_threshold:
                        status_text = f"Mất kết nối ({int(idle_minutes)} phút)"
                        status_color = "#dc3545"
                    elif idle_minutes > (timeout_threshold / 2):
                        status_text = f"Chậm ({int(idle_minutes)} phút)"
                        status_color = "#ffc107"
                    else:
                        status_text = f"Hoạt động ({int(idle_minutes)} phút)"
                        status_color = "#28a745"
                except Exception:
                    status_text = "Hoạt động"
            else:
                status_text = "Không xác định"
                status_color = "#6c757d"

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor(status_color))
            self.table.setItem(i, 3, status_item)

            hb = s.get("last_heartbeat", "")
            hb_display = hb[:19].replace("T", " ") if len(hb) >= 19 else hb
            self.table.setItem(i, 4, QTableWidgetItem(hb_display))

    def kill_selected(self):
        checked_sids = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                checked_sids.append(item.data(Qt.ItemDataRole.UserRole))
        if not checked_sids:
            QMessageBox.warning(self, "Chú ý", "Chọn phiên cần ngắt.")
            return
        reply = QMessageBox.question(self, "Xác nhận",
            f"Ngắt {len(checked_sids)} phiên đã chọn?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        api = self.main_window.local_api
        try:
            for sid in checked_sids:
                api.delete(f"/api/local/sessions/{sid}")
            QMessageBox.information(self, "OK", f"Đã ngắt {len(checked_sids)} phiên.")
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def cleanup_expired(self):
        api = self.main_window.local_api
        try:
            result = api.post("/api/local/sessions/cleanup")
            cleaned = result.get("cleaned_count", 0)
            QMessageBox.information(self, "OK", f"Đã dọn dẹp {cleaned} phiên hết hạn.")
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


class QueueTab(BaseTab):
    """Tab 3: Queue"""

    def init_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Hàng đợi xử lý"))
        toolbar.addStretch()
        btn_refresh = QPushButton("Làm mới")
        btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Tập tin", "IP / Người dùng", "Trạng thái", "Tiến độ", "Thời gian"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_pause = QPushButton("Tạm dừng hàng đợi")
        btn_pause.clicked.connect(self.pause_queue)
        btn_layout.addWidget(btn_pause)

        btn_resume = QPushButton("Tiếp tục")
        btn_resume.clicked.connect(self.resume_queue)
        btn_layout.addWidget(btn_resume)

        btn_cancel = QPushButton("Hủy mục đã chọn")
        btn_cancel.setStyleSheet("background: #dc3545; color: white;")
        btn_cancel.clicked.connect(self.cancel_selected)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def refresh(self):
        if not self.main_window.server.is_running():
            return
        api = self.main_window.local_api
        if hasattr(self, '_fetcher') and self._fetcher.isRunning():
            return
        self._fetcher = _ApiFetcher(api, "/api/local/queue")
        self._fetcher.result.connect(self._on_queue)
        self._fetcher.error.connect(lambda e: self.main_window.statusBar().showMessage(f"Lỗi: {e}", 5000))
        self._fetcher.start()

    def _on_queue(self, items: list):
        # Chỉ hiển thị các file đang waiting hoặc processing
        active_items = [q for q in items if q.get("status") in ("waiting", "processing")]
        self.table.setRowCount(len(active_items))
        for i, q in enumerate(active_items):
            self.table.setItem(i, 0, QTableWidgetItem(str(q.get("file_id", ""))))
            self.table.item(i, 0).setData(Qt.ItemDataRole.UserRole, q.get("file_id"))
            self.table.setItem(i, 1, QTableWidgetItem(q.get("original_filename", "")))
            # IP/User: nếu có username (login) thì show username, nếu anonymous thì show IP
            username = q.get("username", "")
            ip = q.get("ip_address", "")
            display_user = username if username else ip or "Ẩn danh"
            self.table.setItem(i, 2, QTableWidgetItem(display_user))
            status = q.get("status", "")
            self.table.setItem(i, 3, QTableWidgetItem(status))
            percent = q.get("progress_percent", 0) or 0
            msg = q.get("progress_message", "") or ""
            self.table.setItem(i, 4, QTableWidgetItem(f"{percent}% {msg}"))
            created = q.get("created_at", "")
            if created:
                created_display = created[:19].replace("T", " ")
            else:
                created_display = ""
            self.table.setItem(i, 5, QTableWidgetItem(created_display))

    def pause_queue(self):
        try:
            api = self.main_window.local_api
            api.post("/api/local/queue/pause")
            self.main_window.statusBar().showMessage("Hàng đợi đã tạm dừng.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def resume_queue(self):
        try:
            api = self.main_window.local_api
            api.post("/api/local/queue/resume")
            self.main_window.statusBar().showMessage("Hàng đợi tiếp tục.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def cancel_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.warning(self, "Chú ý", "Chọn mục cần hủy.")
            return
        try:
            api = self.main_window.local_api
            for row in rows:
                file_id = self.table.item(row.row(), 0).data(Qt.ItemDataRole.UserRole)
                api.post(f"/api/local/queue/cancel/{file_id}")
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


class UsersTab(BaseTab):
    """Tab 4: Users"""

    def init_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Quản lý người dùng"))
        toolbar.addStretch()

        btn_refresh = QPushButton("Làm mới")
        btn_refresh.clicked.connect(self.refresh)
        toolbar.addWidget(btn_refresh)

        btn_add = QPushButton("Tạo Người dùng")
        btn_add.setStyleSheet("background: #28a745; color: white;")
        btn_add.clicked.connect(self.create_user_dialog)
        toolbar.addWidget(btn_add)
        layout.addLayout(toolbar)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ID", "Tên đăng nhập", "Quyền", "Số file", "Dung lượng (GB)", "Ngày tạo", "Trạng thái"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.chk_select_all = QCheckBox("Chọn tất cả")
        self.chk_select_all.stateChanged.connect(self._toggle_select_all)
        btn_layout.addWidget(self.chk_select_all)

        btn_edit = QPushButton("Chỉnh sửa")
        btn_edit.clicked.connect(self.edit_user_dialog)
        btn_layout.addWidget(btn_edit)

        btn_reset = QPushButton("Đặt lại mật khẩu")
        btn_reset.clicked.connect(self.reset_password_dialog)
        btn_layout.addWidget(btn_reset)

        btn_delete = QPushButton("Xóa người dùng")
        btn_delete.setStyleSheet("background: #dc3545; color: white;")
        btn_delete.clicked.connect(self.delete_selected)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def refresh(self):
        if not self.main_window.server.is_running():
            return
        api = self.main_window.local_api
        if hasattr(self, '_fetcher') and self._fetcher.isRunning():
            return
        self._fetcher = _ApiFetcher(api, "/api/local/users")
        self._fetcher.result.connect(self._on_users)
        self._fetcher.error.connect(lambda e: self.main_window.statusBar().showMessage(f"Lỗi: {e}", 5000))
        self._fetcher.start()

    def _toggle_select_all(self, state):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setCheckState(
                    Qt.CheckState.Checked if state == Qt.CheckState.Checked.value
                    else Qt.CheckState.Unchecked
                )

    def _on_users(self, users: list):
        self.table.setRowCount(len(users))
        for i, u in enumerate(users):
            uid = u.get("id", "")
            id_item = QTableWidgetItem(str(uid))
            id_item.setData(Qt.ItemDataRole.UserRole, uid)
            id_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, id_item)
            self.table.setItem(i, 1, QTableWidgetItem(u.get("username", "")))
            self.table.setItem(i, 2, QTableWidgetItem(u.get("role", "")))
            file_count = u.get("file_count", 0) or 0
            self.table.setItem(i, 3, QTableWidgetItem(str(file_count)))
            storage = u.get("storage_limit_gb", 0)
            used_bytes = u.get("storage_used_bytes", 0) or 0
            used_gb = used_bytes / (1024 * 1024 * 1024)
            self.table.setItem(i, 4, QTableWidgetItem(f"{used_gb:.2f} / {storage}"))
            created = u.get("created_at", "")
            created_display = created[:19].replace("T", " ") if created else ""
            self.table.setItem(i, 5, QTableWidgetItem(created_display))
            active = u.get("is_active", True)
            self.table.setItem(i, 6, QTableWidgetItem("Hoạt động" if active else "Vô hiệu"))

    def create_user_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Tạo người dùng mới")
        dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)

        edit_user = QLineEdit()
        edit_user.setPlaceholderText("Tối thiểu 2 ký tự")
        form.addRow("Tên đăng nhập:", edit_user)

        edit_pass = QLineEdit()
        edit_pass.setEchoMode(QLineEdit.EchoMode.Password)
        edit_pass.setPlaceholderText("Tối thiểu 6 ký tự")
        form.addRow("Mật khẩu:", edit_pass)

        spin_storage = QDoubleSpinBox()
        spin_storage.setRange(0.1, 100)
        spin_storage.setValue(5.0)
        spin_storage.setSuffix(" GB")
        form.addRow("Giới hạn dung lượng:", spin_storage)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            username = edit_user.text().strip()
            password = edit_pass.text()
            storage = spin_storage.value()
            if len(username) < 2:
                QMessageBox.warning(self, "Lỗi", "Tên đăng nhập phải có ít nhất 2 ký tự.")
                return
            if len(password) < 6:
                QMessageBox.warning(self, "Lỗi", "Mật khẩu phải có ít nhất 6 ký tự.")
                return
            try:
                api = self.main_window.local_api
                api.post("/api/local/users", {
                    "username": username,
                    "password": password,
                    "storage_limit_gb": storage,
                })
                QMessageBox.information(self, "OK", f"Đã tạo user '{username}'.")
                self.refresh()  # Tự động refresh sau khi thêm
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def _get_selected_user_row(self):
        """Lấy row index của user đang được check (ưu tiên checkbox, fallback selection)."""
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                return i
        rows = self.table.selectionModel().selectedRows()
        if rows:
            return rows[0].row()
        return None

    def reset_password_dialog(self):
        row = self._get_selected_user_row()
        if row is None:
            QMessageBox.warning(self, "Chú ý", "Chọn người dùng cần đặt lại mật khẩu.")
            return
        user_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        username = self.table.item(row, 1).text()

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Đặt lại mật khẩu - {username}")
        form = QFormLayout(dlg)

        edit_pass = QLineEdit()
        edit_pass.setEchoMode(QLineEdit.EchoMode.Password)
        edit_pass.setPlaceholderText("Tối thiểu 6 ký tự")
        form.addRow("Mật khẩu mới:", edit_pass)

        edit_confirm = QLineEdit()
        edit_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Xác nhận:", edit_confirm)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            pw = edit_pass.text()
            if pw != edit_confirm.text():
                QMessageBox.warning(self, "Lỗi", "Mật khẩu xác nhận không khớp.")
                return
            if len(pw) < 6:
                QMessageBox.warning(self, "Lỗi", "Mật khẩu phải có ít nhất 6 ký tự.")
                return
            try:
                api = self.main_window.local_api
                api.post(f"/api/local/users/{user_id}/reset-password", {"password": pw})
                QMessageBox.information(self, "OK", f"Đã đặt lại mật khẩu cho '{username}'.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def edit_user_dialog(self):
        row = self._get_selected_user_row()
        if row is None:
            QMessageBox.warning(self, "Chú ý", "Chọn người dùng cần chỉnh sửa.")
            return
        user_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        username = self.table.item(row, 1).text()
        role = self.table.item(row, 2).text()
        current_storage = self.table.item(row, 4).text().split(" / ")[-1]  # Cột 4 = Dung lượng
        try:
            current_storage = float(current_storage)
        except ValueError:
            current_storage = 5.0

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Chỉnh sửa ngườii dùng - {username}")
        dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)

        spin_storage = QDoubleSpinBox()
        spin_storage.setRange(0.1, 1000)
        spin_storage.setValue(current_storage)
        spin_storage.setSuffix(" GB")
        form.addRow("Giới hạn dung lượng:", spin_storage)

        chk_active = QCheckBox("Hoạt động")
        chk_active.setChecked(self.table.item(row, 6).text() == "Hoạt động")
        if role == "admin":
            chk_active.setEnabled(False)  # Không cho vô hiệu hóa admin
            chk_active.setToolTip("Không thể vô hiệu hóa tài khoản admin")
        form.addRow("Trạng thái:", chk_active)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                api = self.main_window.local_api
                api.put(f"/api/local/users/{user_id}", {
                    "storage_limit_gb": spin_storage.value(),
                    "is_active": chk_active.isChecked(),
                })
                QMessageBox.information(self, "OK", f"Đã cập nhật ngườii dùng '{username}'.")
                self.refresh()  # Tự động refresh sau khi chỉnh sửa
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def delete_selected(self):
        checked = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                uid = item.data(Qt.ItemDataRole.UserRole)
                username = self.table.item(i, 1).text()
                role = self.table.item(i, 2).text()
                if role == "admin":
                    QMessageBox.warning(self, "Lỗi", f"Không thể xóa tài khoản admin '{username}'.")
                    return
                checked.append((uid, username))
        if not checked:
            QMessageBox.warning(self, "Chú ý", "Chọn người dùng cần xóa.")
            return

        names = ", ".join(u[1] for u in checked)
        reply = QMessageBox.question(self, "Xác nhận", f"Xóa {len(checked)} người dùng: {names}?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            api = self.main_window.local_api
            for uid, _ in checked:
                api.delete(f"/api/local/users/{uid}")
            QMessageBox.information(self, "OK", f"Đã xóa {len(checked)} người dùng.")
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


class ConfigTab(BaseTab):
    """Tab 1: Config & Status (gộp)"""

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ===== TRẠNG THÁI SERVER (từ StatusTab) =====
        status_group = QGroupBox("Trạng thái Server")
        sg = QFormLayout()

        self.lbl_status = QLabel("Đang dừng")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #dc3545;")
        sg.addRow("Trạng thái:", self.lbl_status)

        self.lbl_uptime = QLabel("")
        sg.addRow("Thờii gian chạy (Uptime):", self.lbl_uptime)

        status_group.setLayout(sg)
        layout.addWidget(status_group)

        # Server control buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Bắt đầu Server")
        self.btn_start.setStyleSheet("background: #28a745; color: white; padding: 8px 16px;")
        self.btn_start.clicked.connect(self.start_server)

        self.btn_stop = QPushButton("Dừng Server")
        self.btn_stop.setStyleSheet("background: #dc3545; color: white; padding: 8px 16px;")
        self.btn_stop.clicked.connect(self.stop_server)

        self.btn_restart = QPushButton("Khởi động lại")
        self.btn_restart.setStyleSheet("padding: 8px 16px;")
        self.btn_restart.clicked.connect(self.restart_server)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_restart)

        # Stats inline cùng hàng với buttons
        btn_layout.addStretch()
        self.lbl_sessions = QLabel("0")
        self.lbl_sessions.setStyleSheet("font-size: 18px; font-weight: bold; color: #007bff;")
        lbl_sessions_title = QLabel("Phiên hoạt động")
        lbl_sessions_title.setStyleSheet("color: #999; font-size: 11px;")
        stats_card1 = QVBoxLayout()
        stats_card1.addWidget(self.lbl_sessions, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_card1.addWidget(lbl_sessions_title, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addLayout(stats_card1)

        btn_layout.addSpacing(24)

        self.lbl_queue = QLabel("0")
        self.lbl_queue.setStyleSheet("font-size: 18px; font-weight: bold; color: #007bff;")
        lbl_queue_title = QLabel("Tập tin đang chờ")
        lbl_queue_title.setStyleSheet("color: #999; font-size: 11px;")
        stats_card2 = QVBoxLayout()
        stats_card2.addWidget(self.lbl_queue, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_card2.addWidget(lbl_queue_title, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addLayout(stats_card2)

        layout.addLayout(btn_layout)

        # ===== CẤU HÌNH =====
        config_group = QGroupBox("Cấu hình Server")
        cg = QFormLayout()

        self.edit_host = QComboBox()
        self.edit_host.setEditable(True)
        self._populate_host_ips()
        cg.addRow("Địa chỉ Host:", self.edit_host)

        self.edit_port = QSpinBox()
        self.edit_port.setRange(1, 65535)
        self.edit_port.setValue(8443)
        cg.addRow("Cổng (Port):", self.edit_port)

        self.spin_cpu = QSpinBox()
        self.spin_cpu.setRange(1, ALLOWED_THREADS)
        self.spin_cpu.setValue(min(4, ALLOWED_THREADS))
        cg.addRow("Số luồng CPU:", self.spin_cpu)

        # Nút tạo cert SSL
        btn_cert = QPushButton("Tạo cert SSL")
        btn_cert.setStyleSheet("background: #007bff; color: white; padding: 6px 16px;")
        btn_cert.clicked.connect(self._generate_ssl_cert)
        cg.addRow("", btn_cert)

        config_group.setLayout(cg)
        layout.addWidget(config_group)

        # Limits
        limits_group = QGroupBox("Giới hạn")
        lg = QFormLayout()

        self.spin_max_upload = QSpinBox()
        self.spin_max_upload.setRange(10, 2000)
        self.spin_max_upload.setValue(500)
        self.spin_max_upload.setSuffix(" MB")
        lg.addRow("Dung lượng upload tối đa:", self.spin_max_upload)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(1, 1440)
        self.spin_timeout.setValue(10)
        self.spin_timeout.setSuffix(" phút")
        lg.addRow("Thời gian chờ (Anonymous):", self.spin_timeout)

        self.spin_storage = QDoubleSpinBox()
        self.spin_storage.setRange(0.1, 100)
        self.spin_storage.setValue(5.0)
        self.spin_storage.setSuffix(" GB")
        lg.addRow("Bộ nhớ trên mỗi User:", self.spin_storage)

        self.spin_max_sessions = QSpinBox()
        self.spin_max_sessions.setRange(1, 1000)
        self.spin_max_sessions.setValue(100)
        lg.addRow("Số phiên tối đa:", self.spin_max_sessions)

        limits_group.setLayout(lg)
        layout.addWidget(limits_group)

        # Security
        security_group = QGroupBox("Bảo mật")
        sec = QFormLayout()

        btn_change_pw = QPushButton("Đổi mật khẩu Admin")
        btn_change_pw.clicked.connect(self.change_admin_password)
        sec.addRow(btn_change_pw)

        self.spin_jwt = QSpinBox()
        self.spin_jwt.setRange(60, 43200)
        self.spin_jwt.setValue(1440)
        self.spin_jwt.setSuffix(" phút")
        sec.addRow("Thời hạn JWT:", self.spin_jwt)

        security_group.setLayout(sec)
        layout.addWidget(security_group)

        # Save button
        btn_save = QPushButton("Áp dụng")
        btn_save.setStyleSheet("background: #007bff; color: white; padding: 8px 20px;")
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        layout.addStretch()

        # Load saved config values
        self._load_config()

    def _load_config(self):
        """Đọc config.ini và set giá trị vào UI widgets."""
        try:
            from web_service.config import server_config
            # Host - tìm trong combo hoặc thêm mới
            saved_host = server_config.get("host")
            idx = self.edit_host.findText(saved_host)
            if idx >= 0:
                self.edit_host.setCurrentIndex(idx)
            else:
                self.edit_host.addItem(saved_host)
                self.edit_host.setCurrentText(saved_host)

            self.edit_port.setValue(server_config.port)
            self.spin_cpu.setValue(server_config.cpu_threads)
            self.spin_max_upload.setValue(server_config.get_int("max_upload_mb"))
            self.spin_timeout.setValue(server_config.anonymous_timeout_minutes)
            self.spin_storage.setValue(server_config.get_float("storage_per_user_gb"))
            self.spin_max_sessions.setValue(server_config.max_sessions)
            self.spin_jwt.setValue(server_config.jwt_expire_minutes)

            # Cập nhật local_api port + host
            self.main_window.local_api.port = server_config.port
            self.main_window.local_api.host = server_config.get("host")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def start_server(self):
        host = self.edit_host.currentText()
        port = self.edit_port.value()
        self.main_window.local_api.port = port
        self.main_window.local_api.host = host
        self.main_window.server.start(host, port)
        self.refresh()

    def stop_server(self):
        self.main_window.server.stop()
        self.refresh()

    def restart_server(self):
        host = self.edit_host.currentText()
        port = self.edit_port.value()
        self.main_window.local_api.port = port
        self.main_window.local_api.host = host
        self.main_window.server.restart(host, port)
        self.refresh()

    def refresh(self):
        running = self.main_window.server.is_running()
        if running:
            self.lbl_status.setText("Đang chạy")
            self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #28a745;")
            self.lbl_uptime.setText(self.main_window.server.uptime())
            # Fetch stats in background thread (không block GUI)
            self._fetch_stats()
        else:
            self.lbl_status.setText("Đang dừng")
            self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #dc3545;")
            self.lbl_uptime.setText("")
            self.lbl_sessions.setText("0")
            self.lbl_queue.setText("0")

    def _fetch_stats(self):
        """Fetch stats từ API trong background thread."""
        if hasattr(self, '_stats_thread') and self._stats_thread.isRunning():
            return  # Đang fetch, bỏ qua
        self._stats_thread = _StatsFetcher(self.main_window.local_api)
        self._stats_thread.result.connect(self._on_stats)
        self._stats_thread.start()

    def _on_stats(self, data):
        self.lbl_sessions.setText(str(data.get("active_sessions", 0)))
        self.lbl_queue.setText(str(data.get("queue_waiting", 0)))

    def change_admin_password(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Đổi mật khẩu Admin")
        dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)

        edit_new = QLineEdit()
        edit_new.setEchoMode(QLineEdit.EchoMode.Password)
        edit_new.setPlaceholderText("Tối thiểu 6 ký tự")
        form.addRow("Mật khẩu mới:", edit_new)

        edit_confirm = QLineEdit()
        edit_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Xác nhận:", edit_confirm)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            pw = edit_new.text()
            if pw != edit_confirm.text():
                QMessageBox.warning(self, "Lỗi", "Mật khẩu xác nhận không khớp.")
                return
            if len(pw) < 6:
                QMessageBox.warning(self, "Lỗi", "Mật khẩu phải có ít nhất 6 ký tự.")
                return
            try:
                api = self.main_window.local_api
                # Tìm admin user ID
                users = api.get("/api/local/users")
                admin_user = next((u for u in users if u.get("role") == "admin"), None)
                if not admin_user:
                    QMessageBox.critical(self, "Lỗi", "Không tìm thấy tài khoản admin.")
                    return
                api.post(f"/api/local/users/{admin_user['id']}/reset-password", {"password": pw})
                QMessageBox.information(self, "OK", "Đã đổi mật khẩu admin thành công.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def _populate_host_ips(self):
        """Liet ke IP cua may chu (v4 + v6) de chon lam host."""
        import socket
        self.edit_host.clear()
        self.edit_host.addItems(["0.0.0.0", "127.0.0.1"])
        seen = {"0.0.0.0", "127.0.0.1", "::1"}
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                ip = info[4][0]
                if ip not in seen and not ip.startswith("fe80"):
                    seen.add(ip)
                    self.edit_host.addItem(ip)
        except Exception:
            pass
        # Them phuong phap backup: dung netifaces-style
        try:
            import psutil
            for name, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    ip = addr.address
                    if ip not in seen and not ip.startswith("fe80") and '%' not in ip:
                        if addr.family in (socket.AF_INET, socket.AF_INET6):
                            seen.add(ip)
                            self.edit_host.addItem(ip)
        except ImportError:
            pass

    def _generate_ssl_cert(self):
        """Tao self-signed cert voi IP/domain tu truong host va cac IP hien co."""
        import ipaddress as ipamod
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        # Thu thap tat ca IP va domain can them vao SAN
        san_entries = set()
        san_entries.add(("dns", "localhost"))
        san_entries.add(("ip", "127.0.0.1"))

        # Lay IP tu host field hien tai
        current_host = self.edit_host.currentText().strip()
        if current_host and current_host != "0.0.0.0":
            try:
                ipamod.ip_address(current_host)
                san_entries.add(("ip", current_host))
            except ValueError:
                san_entries.add(("dns", current_host))

        # Them tat ca IP tu dropdown
        for i in range(self.edit_host.count()):
            ip_text = self.edit_host.itemText(i).strip()
            if ip_text and ip_text not in ("0.0.0.0",):
                try:
                    ipamod.ip_address(ip_text)
                    san_entries.add(("ip", ip_text))
                except ValueError:
                    if ip_text != "":
                        san_entries.add(("dns", ip_text))

        # Build SAN list
        san_list = []
        for entry_type, value in sorted(san_entries):
            if entry_type == "dns":
                san_list.append(x509.DNSName(value))
            else:
                san_list.append(x509.IPAddress(ipamod.ip_address(value)))

        # Tao RSA key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        import datetime
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Sherpa Vietnamese ASR"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ASR VN"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        cert_dir = os.path.join(BASE_DIR, "web_service", "certs")
        os.makedirs(cert_dir, exist_ok=True)
        cert_file = os.path.join(cert_dir, "server.crt")
        key_file = os.path.join(cert_dir, "server.key")

        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Hien thi danh sach IP/domain da them
        ip_list = ", ".join(v for _, v in sorted(san_entries))
        QMessageBox.information(self, "Tạo cert SSL",
            f"Đã tạo chứng chỉ SSL thành công!\n\n"
            f"SAN: {ip_list}\n\n"
            f"Vui lòng khởi động lại server để áp dụng.")

    def _check_cert_covers_host(self, host: str) -> bool:
        """Kiểm tra cert SSL hiện tại có SAN chứa host không."""
        cert_file = os.path.join(BASE_DIR, "web_service", "certs", "server.crt")
        if not os.path.exists(cert_file):
            return False
        try:
            from cryptography import x509
            with open(cert_file, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            # Lấy tất cả IP và DNS trong SAN
            all_ips = [str(ip) for ip in san.value.get_values_for_type(x509.IPAddress)]
            all_dns = san.value.get_values_for_type(x509.DNSName)
            return host in all_ips or host in all_dns or host == "0.0.0.0"
        except Exception:
            return False

    def save_config(self):
        # Hiện dialog xác nhận
        reply = QMessageBox.question(
            self, "Xác nhận áp dụng",
            "Áp dụng cấu hình sẽ:\n"
            "• Lưu các thay đổi cấu hình\n"
            "• Hủy tất cả các phiên làm việc hiện tại\n"
            "• Khởi động lại dịch vụ\n\n"
            "Bạn có chắc chắn muốn tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from web_service.config import server_config
            new_host = self.edit_host.currentText()
            server_config.set("host", new_host)
            server_config.set("port", str(self.edit_port.value()))
            server_config.set("cpu_threads", str(self.spin_cpu.value()))
            server_config.set("max_upload_mb", str(self.spin_max_upload.value()))
            server_config.set("anonymous_timeout_minutes", str(self.spin_timeout.value()))
            server_config.set("storage_per_user_gb", str(self.spin_storage.value()))
            server_config.set("max_sessions", str(self.spin_max_sessions.value()))
            server_config.set("jwt_expire_minutes", str(self.spin_jwt.value()))
            server_config.save()

            # Kiểm tra cert có cover host mới không
            if not self._check_cert_covers_host(new_host):
                regen = QMessageBox.question(
                    self, "Chứng chỉ SSL",
                    f"Chứng chỉ SSL hiện tại không chứa IP '{new_host}'.\n"
                    "Tạo lại chứng chỉ SSL mới?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if regen == QMessageBox.StandardButton.Yes:
                    self._generate_ssl_cert()

            # Thực hiện restart server để áp dụng
            if self.main_window.server.is_running():
                self.restart_server()
                QMessageBox.information(self, "OK", "Đã áp dụng cấu hình và khởi động lại dịch vụ.")
            else:
                QMessageBox.information(self, "OK", "Đã lưu cấu hình. Server sẽ sử dụng cấu hình mới khi khởi động.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


class LogsTab(BaseTab):
    """Tab 6: Logs"""

    MAX_LOG_LINES = 5000  # Giới hạn dòng hiển thị

    def init_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("System Logs"))
        toolbar.addStretch()

        self.chk_auto_scroll = QCheckBox("Tự động cuộn")
        self.chk_auto_scroll.setChecked(True)
        toolbar.addWidget(self.chk_auto_scroll)

        btn_clear = QPushButton("Xóa")
        btn_clear.clicked.connect(self._clear_log)
        toolbar.addWidget(btn_clear)

        btn_open_dir = QPushButton("Mở thư mục Log")
        btn_open_dir.clicked.connect(self.open_log_dir)
        toolbar.addWidget(btn_open_dir)

        layout.addLayout(toolbar)

        # QPlainTextEdit nhanh hơn QTextEdit rất nhiều cho log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        self.log_view.setStyleSheet("background: #1e1e1e; color: #d4d4d4; border: 1px solid #555;")
        self.log_view.setMaximumBlockCount(self.MAX_LOG_LINES)
        layout.addWidget(self.log_view)

        # Timer đọc log file
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.read_log)
        self.log_timer.start(2000)
        self._log_pos = 0

    def _clear_log(self):
        self.log_view.clear()
        # Reset position để không đọc lại log cũ
        from web_service.config import LOG_DIR
        log_file = os.path.join(LOG_DIR, "server.log")
        if os.path.exists(log_file):
            self._log_pos = os.path.getsize(log_file)

    def read_log(self):
        # Chỉ đọc khi tab Logs đang hiển thị
        if self.main_window.tabs.currentWidget() is not self:
            return

        from web_service.config import LOG_DIR
        log_file = os.path.join(LOG_DIR, "server.log")
        if not os.path.exists(log_file):
            return
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                f.seek(self._log_pos)
                new_text = f.read()
                if not new_text:
                    return
                self._log_pos = f.tell()

            # Batch append tất cả dòng mới 1 lần (nhanh hơn append từng dòng)
            self.log_view.appendPlainText(new_text.rstrip('\n'))

            if self.chk_auto_scroll.isChecked():
                self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            pass

    def open_log_dir(self):
        from web_service.config import LOG_DIR
        os.makedirs(LOG_DIR, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(LOG_DIR)


class ServiceTab(BaseTab):
    """Tab 7: Windows Service"""

    SERVICE_NAME = "SherpaASRVNOnline"
    DISPLAY_NAME = "Sherpa ASR Vietnamese Online Service"

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Status
        status_group = QGroupBox("Trạng thái Service")
        sg = QFormLayout()

        self.lbl_service_status = QLabel("Chưa kiểm tra")
        sg.addRow("Trạng thái:", self.lbl_service_status)
        sg.addRow("Tên Service:", QLabel(self.SERVICE_NAME))
        sg.addRow("Tên hiển thị:", QLabel(self.DISPLAY_NAME))

        status_group.setLayout(sg)
        layout.addWidget(status_group)

        # Control buttons
        btn_layout = QHBoxLayout()

        btn_install = QPushButton("Cài đặt Service")
        btn_install.setStyleSheet("background: #007bff; color: white; padding: 8px 16px;")
        btn_install.clicked.connect(self.install_service)
        btn_layout.addWidget(btn_install)

        btn_uninstall = QPushButton("Gỡ bỏ Service")
        btn_uninstall.setStyleSheet("background: #dc3545; color: white; padding: 8px 16px;")
        btn_uninstall.clicked.connect(self.uninstall_service)
        btn_layout.addWidget(btn_uninstall)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Service controls
        svc_layout = QHBoxLayout()
        for name, method in [("Bắt đầu", self.start_service), ("Dừng", self.stop_service), ("Khởi động lại", self.restart_service)]:
            btn = QPushButton(name)
            btn.clicked.connect(method)
            svc_layout.addWidget(btn)
        svc_layout.addStretch()
        layout.addLayout(svc_layout)

        # Auto start
        self.chk_auto_start = QCheckBox("Tự động bắt đầu service khi Windows khởi động")
        layout.addWidget(self.chk_auto_start)

        # CLI commands
        cli_group = QGroupBox("Lệnh CLI (Command Prompt với quyền Admin)")
        cl = QVBoxLayout()
        commands = [
            f"sc start {self.SERVICE_NAME}",
            f"sc stop {self.SERVICE_NAME}",
            f"sc query {self.SERVICE_NAME}",
            f"sc config {self.SERVICE_NAME} start= auto",
        ]
        cli_text = QTextEdit()
        cli_text.setReadOnly(True)
        cli_text.setPlainText("\n".join(commands))
        cli_text.setMaximumHeight(100)
        cli_text.setFont(QFont("Consolas", 10))
        cl.addWidget(cli_text)
        cli_group.setLayout(cl)
        layout.addWidget(cli_group)

        layout.addStretch()

        # Kiểm tra trạng thái ban đầu
        self.refresh()

    def _run_sc(self, args):
        try:
            result = subprocess.run(
                ["sc"] + args, capture_output=True, text=True, timeout=10,
            )
            return result.returncode, result.stdout + result.stderr
        except Exception as e:
            return -1, str(e)

    def refresh(self):
        code, output = self._run_sc(["query", self.SERVICE_NAME])
        if code != 0 or "does not exist" in output.lower() or "1060" in output:
            self.lbl_service_status.setText("Chưa cài đặt")
            self.lbl_service_status.setStyleSheet("color: #999;")
        elif "RUNNING" in output:
            self.lbl_service_status.setText("Đang chạy")
            self.lbl_service_status.setStyleSheet("color: #28a745; font-weight: bold;")
        elif "STOPPED" in output:
            self.lbl_service_status.setText("Đã dừng")
            self.lbl_service_status.setStyleSheet("color: #dc3545;")
        else:
            self.lbl_service_status.setText(output.strip()[:50])

    def install_service(self):
        QMessageBox.information(self, "Cài đặt",
            "Chạy lệnh sau với quyền Admin:\n\n"
            f'python service_installer.py install\n\n'
            "Hoặc dùng nssm:\n"
            f'nssm install {self.SERVICE_NAME} "{sys.executable}" "{os.path.join(BASE_DIR, "server_launcher.py")}" --no-gui'
        )

    def uninstall_service(self):
        reply = QMessageBox.question(self, "Xác nhận", "Bạn có chắc muốn gỡ bỏ service?")
        if reply == QMessageBox.StandardButton.Yes:
            code, output = self._run_sc(["delete", self.SERVICE_NAME])
            QMessageBox.information(self, "Kết quả", output)
            self.refresh()

    def start_service(self):
        code, output = self._run_sc(["start", self.SERVICE_NAME])
        QMessageBox.information(self, "Kết quả", output)
        self.refresh()

    def stop_service(self):
        code, output = self._run_sc(["stop", self.SERVICE_NAME])
        QMessageBox.information(self, "Kết quả", output)
        self.refresh()

    def restart_service(self):
        self.stop_service()
        time.sleep(2)
        self.start_service()


class ServerGUI(QMainWindow):
    """Main window - 7 tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sherpa Vietnamese ASR - Server Admin")
        self.resize(1000, 750)

        self.server = ServerProcess()
        self.local_api = _LocalAPI(port=8443)
        # host + port sẽ được cập nhật từ config trong ConfigTab._load_config()
        try:
            from web_service.config import server_config
            self.local_api.port = server_config.port
            self.local_api.host = server_config.get("host")
        except Exception:
            pass
        self.init_ui()
        self.init_timers()
        self.center_on_screen()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()

        self.tab_config = ConfigTab(self)
        self.tab_sessions = SessionsTab(self)
        self.tab_queue = QueueTab(self)
        self.tab_users = UsersTab(self)
        self.tab_logs = LogsTab(self)
        self.tab_service = ServiceTab(self)

        # Cấu hình (gộp Trạng thái) đưa lên đầu tiên
        self.tabs.addTab(self.tab_config, "Cấu hình")
        self.tabs.addTab(self.tab_sessions, "Phiên")
        self.tabs.addTab(self.tab_queue, "Hàng đợi")
        self.tabs.addTab(self.tab_users, "Ngườii dùng")
        self.tabs.addTab(self.tab_logs, "Nhật ký")
        self.tabs.addTab(self.tab_service, "Dịch vụ")

        # Kết nối signal để refresh tab khi được chọn
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        self.statusBar().showMessage(f"Ready | Server: Stopped | Max CPU: {ALLOWED_THREADS}")

    def _on_tab_changed(self, index):
        """Tự động refresh tab khi được chọn."""
        current_tab = self.tabs.widget(index)
        if current_tab and hasattr(current_tab, 'refresh'):
            current_tab.refresh()

    def init_timers(self):
        # Refresh config tab (chứa trạng thái) mỗi 5 giây
        self.timer_status = QTimer()
        self.timer_status.timeout.connect(self.tab_config.refresh)
        self.timer_status.start(5000)

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def closeEvent(self, event):
        if self.server.is_running():
            reply = QMessageBox.question(
                self, "Xác nhận",
                "Server đang chạy. Bạn có muốn dừng server và thoát?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.server.stop()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    window = ServerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
