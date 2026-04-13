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
    QDialog, QDialogButtonBox, QProgressBar, QDoubleSpinBox, QFileDialog,
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
        # P3 MITM: cảnh báo nếu GUI kết nối tới host không phải loopback
        _loopback = {"127.0.0.1", "::1", "localhost"}
        if self._host not in _loopback:
            logger.warning(
                "P3 MITM risk: Admin GUI kết nối tới %s (không phải localhost). "
                "Traffic quản trị có thể bị chặn trên LAN. "
                "Khuyến nghị: chỉ dùng 127.0.0.1 hoặc dùng VPN/SSH tunnel.",
                self._host,
            )

    def _ssl_ctx(self):
        import ssl
        ctx = ssl.create_default_context()
        # A02: Chỉ disable cert verification cho loopback (self-signed cert).
        # Với remote host, giữ verification để tránh MITM.
        _loopback = {"127.0.0.1", "::1", "localhost"}
        if self._host in _loopback:
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


class CollapsibleSection(QWidget):
    """Section bấm tiêu đề (▶/▼) để bung/thu nội dung."""

    def __init__(self, title: str, collapsed: bool = True):
        super().__init__()
        self._title = title
        self._collapsed = collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header — flat button với tam giác
        self._btn = QPushButton()
        self._btn.setFlat(True)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setFixedHeight(32)
        self._btn.setStyleSheet(
            f"QPushButton {{ text-align: left; padding: 6px 12px; font-weight: bold;"
            f"  background-color: {COLORS['bg_card']}; color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['border']}; border-radius: 4px; }}"
            f"QPushButton:hover {{ background-color: {COLORS['bg_elevated']}; }}"
        )
        self._btn.clicked.connect(self.toggle)
        outer.addWidget(self._btn)

        # Content area
        self._content = QWidget()
        self._content.setStyleSheet(
            f"QWidget {{ background-color: {COLORS['bg_dark']}; margin-left: 4px; }}"
        )
        outer.addWidget(self._content)

        self._update()

    def setLayout(self, layout):
        """Redirect layout vào content area (không phải self)."""
        self._content.setLayout(layout)

    def toggle(self):
        self._collapsed = not self._collapsed
        self._update()

    def _update(self):
        arrow = "▾" if not self._collapsed else "▸"
        self._btn.setText(f"  {arrow}   {self._title}")
        self._content.setVisible(not self._collapsed)
        # Resize main window height to fit content
        def _resize():
            QApplication.processEvents()
            main_win = self.window()
            if not isinstance(main_win, QMainWindow):
                return
            current_w = main_win.width()
            central = main_win.centralWidget()
            if central and central.layout():
                needed_h = central.layout().minimumSize().height()
                frame_extra = main_win.frameGeometry().height() - main_win.height()
                new_h = min(needed_h + frame_extra + 8, 900)
                main_win.resize(current_w, new_h)
        QTimer.singleShot(10, _resize)


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
        self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['danger']};")
        sg.addRow("Trạng thái:", self.lbl_status)

        self.lbl_uptime = QLabel("")
        sg.addRow("Thời gian chạy (Uptime):", self.lbl_uptime)

        status_group.setLayout(sg)
        layout.addWidget(status_group)

        # Controls
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Bắt đầu Server")
        self.btn_start.setStyleSheet(f"background: {COLORS['success']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
        self.btn_start.clicked.connect(self.start_server)

        self.btn_stop = QPushButton("Dừng Server")
        self.btn_stop.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
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
            v.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLORS['accent']};")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t = QLabel(f"{title}")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet(f"color: {COLORS['text_secondary']};")
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

        # Cảnh báo nếu admin chưa đổi mật khẩu mặc định
        try:
            import sys as _sys
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _base not in _sys.path:
                _sys.path.insert(0, _base)
            from web_service.auth import is_admin_using_default_password
            if is_admin_using_default_password():
                reply = QMessageBox.warning(
                    self,
                    "⚠️ Cảnh báo bảo mật",
                    "Tài khoản <b>admin</b> đang dùng mật khẩu mặc định <b>'admin'</b>.<br><br>"
                    "Điều này có thể gây rủi ro bảo mật nếu server được truy cập từ mạng ngoài.<br><br>"
                    "Bạn có muốn tiếp tục khởi động không?<br>"
                    "<i>(Khuyến nghị: đổi mật khẩu ngay sau khi đăng nhập)</i>",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
        except Exception:
            pass  # Không chặn start nếu kiểm tra lỗi

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
            self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['success']};")
            self.lbl_uptime.setText(self.main_window.server.uptime())
            # Fetch stats in background thread (không block GUI)
            self._fetch_stats()
        else:
            self.lbl_status.setText("Đang dừng")
            self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['danger']};")
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
        btn_kill.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']};")
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
            status_color = COLORS['success']

            if expired_at:
                status_text = "Hết hạn"
                status_color = COLORS['danger']
            elif last_hb_str:
                try:
                    last_hb = datetime.fromisoformat(last_hb_str)
                    if last_hb.tzinfo is not None:
                        last_hb = last_hb.replace(tzinfo=None)
                    idle_minutes = (now - last_hb).total_seconds() / 60
                    timeout_threshold = anon_timeout if is_anonymous else logged_in_timeout

                    if idle_minutes > timeout_threshold:
                        status_text = f"Mất kết nối ({int(idle_minutes)} phút)"
                        status_color = COLORS['danger']
                    elif idle_minutes > (timeout_threshold / 2):
                        status_text = f"Chậm ({int(idle_minutes)} phút)"
                        status_color = COLORS['warning']
                    else:
                        status_text = f"Hoạt động ({int(idle_minutes)} phút)"
                        status_color = COLORS['success']
                except Exception:
                    status_text = "Hoạt động"
            else:
                status_text = "Không xác định"
                status_color = COLORS['text_secondary']

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
        btn_cancel.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']};")
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
        btn_add.setStyleSheet(f"background: {COLORS['success']}; color: {COLORS['text_primary']};")
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
        btn_delete.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']};")
        btn_delete.clicked.connect(self.delete_selected)
        btn_layout.addWidget(btn_delete)

        btn_unlock = QPushButton("🔓 Xóa khóa đăng nhập")
        btn_unlock.setToolTip("Xóa tất cả IP bị khóa do đăng nhập sai nhiều lần")
        btn_unlock.clicked.connect(self.clear_rate_limits)
        btn_layout.addWidget(btn_unlock)

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

    def clear_rate_limits(self):
        """Xóa tất cả IP bị khóa đăng nhập."""
        if not self.main_window.server.is_running():
            QMessageBox.warning(self, "Chú ý", "Server chưa chạy.")
            return
        try:
            api = self.main_window.local_api
            locked = api.get("/api/local/rate-limits")
            if not locked:
                QMessageBox.information(self, "Thông báo", "Không có IP nào đang bị khóa.")
                return
            info = "\n".join(f"  • {x['ip']} ({x['attempts']} lần sai, mở khóa sau {x['unlock_in_seconds']}s)"
                             for x in locked)
            reply = QMessageBox.question(
                self, "Xác nhận",
                f"Có {len(locked)} IP đang bị khóa:\n{info}\n\nXóa khóa tất cả?"
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            api.post("/api/local/rate-limits/clear")
            QMessageBox.information(self, "OK", f"Đã mở khóa {len(locked)} IP.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


class ConfigTab(BaseTab):
    """Tab 1: Config & Status (gộp)"""

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ===== TRẠNG THÁI SERVER (từ StatusTab) =====
        status_group = QGroupBox("Trạng thái Server")
        sg = QFormLayout()

        self.lbl_status = QLabel("Đang dừng")
        self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['danger']};")
        sg.addRow("Trạng thái:", self.lbl_status)

        self.lbl_uptime = QLabel("")
        sg.addRow("Thờii gian chạy (Uptime):", self.lbl_uptime)

        status_group.setLayout(sg)
        layout.addWidget(status_group)

        # Server control buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Bắt đầu Server")
        self.btn_start.setStyleSheet(f"background: {COLORS['success']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
        self.btn_start.clicked.connect(self.start_server)

        self.btn_stop = QPushButton("Dừng Server")
        self.btn_stop.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
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
        self.lbl_sessions.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']};")
        lbl_sessions_title = QLabel("Phiên hoạt động")
        lbl_sessions_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        stats_card1 = QVBoxLayout()
        stats_card1.addWidget(self.lbl_sessions, alignment=Qt.AlignmentFlag.AlignCenter)
        stats_card1.addWidget(lbl_sessions_title, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_layout.addLayout(stats_card1)

        btn_layout.addSpacing(24)

        self.lbl_queue = QLabel("0")
        self.lbl_queue.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']};")
        lbl_queue_title = QLabel("Tập tin đang chờ")
        lbl_queue_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
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
        self.spin_cpu.setRange(1, 128)
        self.spin_cpu.setValue(min(4, ALLOWED_THREADS))
        self.lbl_cpu_warn = QLabel()
        self.lbl_cpu_warn.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        self.lbl_cpu_warn.hide()
        self.spin_cpu.valueChanged.connect(self._on_cpu_changed)
        cpu_layout = QVBoxLayout()
        cpu_layout.setSpacing(2)
        cpu_layout.addWidget(self.spin_cpu)
        cpu_layout.addWidget(self.lbl_cpu_warn)
        cg.addRow("Số luồng CPU:", cpu_layout)

        self.edit_offline_url = QLineEdit()
        self.edit_offline_url.setPlaceholderText("https://drive.google.com/... (để trống nếu chưa có)")
        cg.addRow("Link tải bản cài offline:", self.edit_offline_url)

        config_group.setLayout(cg)
        layout.addWidget(config_group)

        # === Summarizer (Tóm tắt cuộc họp) ===
        summ_group = CollapsibleSection("📝 Tóm tắt cuộc họp (LLM)", collapsed=True)
        sg = QFormLayout()
        sg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Dòng 1: checkbox + model status cùng hàng
        row_enable = QHBoxLayout()
        self.chk_summarizer_enabled = QCheckBox("Bật tóm tắt cuộc họp")
        self.chk_summarizer_enabled.toggled.connect(self._on_summarizer_toggled)
        row_enable.addWidget(self.chk_summarizer_enabled)
        self.lbl_summarizer_status = QLabel("")
        self.lbl_summarizer_status.setStyleSheet(f"color: {COLORS['text_secondary']}")
        row_enable.addWidget(self.lbl_summarizer_status, 1)
        sg.addRow("", row_enable)

        # Dòng 2: đường dẫn model + browse + tải
        row_model = QHBoxLayout()
        row_model.setContentsMargins(0, 0, 0, 0)
        self.edit_summarizer_gguf = QLineEdit()
        self.edit_summarizer_gguf.setPlaceholderText("Đường dẫn file model GGUF...")
        btn_browse = QPushButton("Chọn file...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_gguf_model)
        self.btn_download_model = QPushButton("Tải model")
        self.btn_download_model.setFixedWidth(90)
        self.btn_download_model.setToolTip("Tải Qwen3-4B-Instruct-2507 Q4_K_M (~2.5 GB) từ HuggingFace")
        self.btn_download_model.clicked.connect(self._download_summarizer_model)
        row_model.addWidget(self.edit_summarizer_gguf, 1)
        row_model.addWidget(btn_browse)
        row_model.addWidget(self.btn_download_model)
        sg.addRow("Model GGUF:", row_model)

        # Dòng 3: số luồng LLM
        row_threads = QHBoxLayout()
        self.spin_llm_threads = QSpinBox()
        self.spin_llm_threads.setRange(1, 64)
        self.spin_llm_threads.setValue(ALLOWED_THREADS)
        self.spin_llm_threads.setFixedWidth(80)
        row_threads.addWidget(self.spin_llm_threads)
        lbl_threads_hint = QLabel(f"(vật lý: {ALLOWED_THREADS} cores — khuyến nghị = số core vật lý)")
        lbl_threads_hint.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        row_threads.addWidget(lbl_threads_hint)
        row_threads.addStretch()
        sg.addRow("Số luồng LLM:", row_threads)

        summ_group.setLayout(sg)
        layout.addWidget(summ_group)

        # === SSL Certificate ===
        ssl_group = CollapsibleSection("Chứng chỉ SSL", collapsed=True)
        ssl_layout = QVBoxLayout()

        # HTTP mode checkbox
        self.chk_http_mode = QCheckBox("Không chạy HTTPS, chạy HTTP thôi (WAF sẽ dùng SSL Offloading)")
        self.chk_http_mode.setStyleSheet(f"color: {COLORS['warning']}; font-weight: bold; padding: 4px 0;")
        self.chk_http_mode.toggled.connect(self._on_http_mode_toggled)
        ssl_layout.addWidget(self.chk_http_mode)

        # Container cho tất cả SSL widgets (ẩn khi HTTP mode)
        self.ssl_widgets_container = QWidget()
        ssl_inner = QVBoxLayout()
        ssl_inner.setContentsMargins(0, 0, 0, 0)

        # Trạng thái cert hiện tại
        self.lbl_cert_status = QLabel("Đang kiểm tra...")
        self.lbl_cert_status.setWordWrap(True)
        ssl_inner.addWidget(self.lbl_cert_status)

        # Self-signed
        btn_cert = QPushButton("Tạo cert Self-signed")
        btn_cert.setStyleSheet(f"background: {COLORS['accent']}; color: {COLORS['text_primary']}; padding: 6px 16px;")
        btn_cert.clicked.connect(self._generate_ssl_cert)
        cert_btn_row = QHBoxLayout()
        cert_btn_row.addWidget(btn_cert)
        cert_btn_row.addStretch()
        ssl_inner.addLayout(cert_btn_row)

        # Import custom cert
        import_group = QGroupBox("Import chứng chỉ (Let's Encrypt, v.v.)")
        ig = QFormLayout()

        self.edit_cert_file = QLineEdit()
        self.edit_cert_file.setPlaceholderText("fullchain.pem hoặc certificate.crt")
        self.edit_cert_file.setReadOnly(True)
        btn_browse_cert = QPushButton("Chọn...")
        btn_browse_cert.clicked.connect(lambda: self._browse_cert_file(self.edit_cert_file))
        cert_row = QHBoxLayout()
        cert_row.addWidget(self.edit_cert_file, 1)
        cert_row.addWidget(btn_browse_cert)
        ig.addRow("Cert file:", cert_row)

        self.edit_key_file = QLineEdit()
        self.edit_key_file.setPlaceholderText("privkey.pem hoặc private.key")
        self.edit_key_file.setReadOnly(True)
        btn_browse_key = QPushButton("Chọn...")
        btn_browse_key.clicked.connect(lambda: self._browse_cert_file(self.edit_key_file))
        key_row = QHBoxLayout()
        key_row.addWidget(self.edit_key_file, 1)
        key_row.addWidget(btn_browse_key)
        ig.addRow("Key file:", key_row)

        import_group.setLayout(ig)
        ssl_inner.addWidget(import_group)

        # Buttons: Import + Xóa
        ssl_btn_layout = QHBoxLayout()
        btn_import = QPushButton("Import chứng chỉ")
        btn_import.setStyleSheet(f"background: {COLORS['success']}; color: {COLORS['text_primary']}; padding: 6px 16px;")
        btn_import.clicked.connect(self._import_custom_cert)
        ssl_btn_layout.addWidget(btn_import)

        btn_remove = QPushButton("Xóa cert custom")
        btn_remove.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']}; padding: 6px 16px;")
        btn_remove.clicked.connect(self._remove_custom_cert)
        ssl_btn_layout.addWidget(btn_remove)

        ssl_btn_layout.addStretch()
        ssl_inner.addLayout(ssl_btn_layout)

        self.ssl_widgets_container.setLayout(ssl_inner)
        ssl_layout.addWidget(self.ssl_widgets_container)

        ssl_group.setLayout(ssl_layout)
        layout.addWidget(ssl_group)

        # Limits
        limits_group = CollapsibleSection("Giới hạn", collapsed=True)
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
        security_group = CollapsibleSection("Bảo mật", collapsed=True)
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
        btn_save.setStyleSheet(f"background: {COLORS['accent']}; color: {COLORS['text_primary']}; padding: 8px 20px;")
        btn_save.clicked.connect(self.save_config)
        save_btn_row = QHBoxLayout()
        save_btn_row.addStretch()
        save_btn_row.addWidget(btn_save)
        save_btn_row.addStretch()
        layout.addLayout(save_btn_row)

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
            self.edit_offline_url.setText(server_config.get("offline_download_url"))

            # Summarizer
            summ_path = server_config.get("summarizer_model_path") or ""
            if summ_path:
                self.edit_summarizer_gguf.setText(summ_path)
            llm_threads = int(server_config.get("summarizer_threads") or ALLOWED_THREADS)
            self.spin_llm_threads.setValue(llm_threads)
            self.chk_summarizer_enabled.setChecked(server_config.get("summarizer_enabled") == "1")
            self._check_summarizer_status()

            # HTTP mode
            self.chk_http_mode.setChecked(server_config.http_mode)

            # Cập nhật local_api port + host
            self.main_window.local_api.port = server_config.port
            self.main_window.local_api.host = server_config.get("host")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

        self._on_cpu_changed(self.spin_cpu.value())
        self._refresh_cert_status()

    def _on_summarizer_toggled(self, checked):
        """Khi toggle summarizer — check model có sẵn không."""
        self._check_summarizer_status()

    def _browse_gguf_model(self):
        """Mở file dialog chọn file GGUF."""
        from core.config import BASE_DIR
        start_dir = os.path.join(BASE_DIR, "models")
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file model GGUF",
            start_dir,
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if path:
            self.edit_summarizer_gguf.setText(path)
            self._check_summarizer_status()

    def _check_summarizer_status(self):
        """Kiểm tra model và cập nhật status label."""
        import os
        lbl = self.lbl_summarizer_status

        if not self.chk_summarizer_enabled.isChecked():
            lbl.setText("")
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}")
            return

        path = self.edit_summarizer_gguf.text().strip()

        # Nếu chưa chọn → check model mặc định
        if not path:
            try:
                from web_service.summarizer import get_default_model_path
                default = get_default_model_path()
                if os.path.isfile(default):
                    self.edit_summarizer_gguf.setText(default)
                    path = default
            except Exception:
                pass

        if not path:
            lbl.setText("Chưa chọn model")
            lbl.setStyleSheet(f"color: {COLORS['warning']}")
        elif os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1e6
            lbl.setText(f"{os.path.basename(path)} ({size_mb:.0f} MB)")
            lbl.setStyleSheet(f"color: {COLORS['success']}")
        else:
            # Thử resolve relative path
            from web_service.summarizer import _resolve_model_path
            resolved = _resolve_model_path(path)
            if os.path.isfile(resolved):
                size_mb = os.path.getsize(resolved) / 1e6
                lbl.setText(f"{os.path.basename(resolved)} ({size_mb:.0f} MB)")
                lbl.setStyleSheet(f"color: {COLORS['success']}")
            else:
                lbl.setText("File không tồn tại")
                lbl.setStyleSheet(f"color: {COLORS['danger']}")

    def _download_summarizer_model(self):
        """Tải model GGUF — hỏi xác nhận, chọn thư mục, chạy background."""
        from web_service.summarizer import get_default_model_path, DEFAULT_GGUF_FILE
        import os

        default_path = get_default_model_path()
        default_dir = os.path.dirname(default_path)

        # Hỏi chọn thư mục tải về
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu model GGUF",
            default_dir,
        )
        if not dest_dir:
            return  # User cancel

        dest_file = os.path.join(dest_dir, DEFAULT_GGUF_FILE)

        # Nếu file đã tồn tại → hỏi ghi đè
        if os.path.isfile(dest_file):
            size_mb = os.path.getsize(dest_file) / 1e6
            reply = QMessageBox.question(
                self, "File đã tồn tại",
                f"File đã có:\n{dest_file}\n({size_mb:.0f} MB)\n\nGhi đè?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                # Dùng file có sẵn
                self.edit_summarizer_gguf.setText(dest_file)
                self._check_summarizer_status()
                return
            # Xóa file cũ để tải lại
            os.remove(dest_file)

        self.btn_download_model.setEnabled(False)
        self.btn_download_model.setText("Đang tải...")
        self.lbl_summarizer_status.setText("Đang tải ~2.7 GB...")
        self.lbl_summarizer_status.setStyleSheet(f"color: {COLORS['warning']}")

        import threading
        from PyQt6.QtCore import QTimer

        def _do_download():
            try:
                from web_service.summarizer import download_model_to
                path = download_model_to(dest_dir)
                # Update UI từ main thread qua QTimer (an toàn hơn QMetaObject)
                QTimer.singleShot(0, lambda: self._on_download_done(path))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_download_error(str(e)))

        threading.Thread(target=_do_download, daemon=True).start()

    def _on_download_done(self, path):
        """Callback khi tải model xong (chạy trên main thread)."""
        import os
        self.btn_download_model.setEnabled(True)
        self.btn_download_model.setText("Tải model")
        self.edit_summarizer_gguf.setText(path)
        size_mb = os.path.getsize(path) / 1e6
        self.lbl_summarizer_status.setText(f"{os.path.basename(path)} ({size_mb:.0f} MB)")
        self.lbl_summarizer_status.setStyleSheet(f"color: {COLORS['success']}")

    def _on_download_error(self, error_msg):
        """Callback khi tải model lỗi (chạy trên main thread)."""
        self.btn_download_model.setEnabled(True)
        self.btn_download_model.setText("Tải model")
        self.lbl_summarizer_status.setText(f"Lỗi: {error_msg[:80]}")
        self.lbl_summarizer_status.setStyleSheet(f"color: {COLORS['danger']}")

    def _on_http_mode_toggled(self, checked):
        """Ẩn/hiện SSL widgets khi toggle HTTP mode."""
        self.ssl_widgets_container.setVisible(not checked)

    def _on_cpu_changed(self, value):
        """Cảnh báo khi số luồng CPU > số core vật lý."""
        if value > ALLOWED_THREADS:
            self.lbl_cpu_warn.setText(
                f"⚠ Vượt quá số core vật lý ({ALLOWED_THREADS}). "
                f"Có thể giảm hiệu suất do context switching."
            )
            self.lbl_cpu_warn.show()
        else:
            self.lbl_cpu_warn.hide()

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
            self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['success']};")
            uptime = self.main_window.server.uptime()
            self.lbl_uptime.setText(uptime)
            self.main_window.statusBar().showMessage(
                f"Server: Running ({uptime}) | Max CPU: {ALLOWED_THREADS}"
            )
            # Fetch stats in background thread (không block GUI)
            self._fetch_stats()
        else:
            self.lbl_status.setText("Đang dừng")
            self.lbl_status.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['danger']};")
            self.lbl_uptime.setText("")
            self.lbl_sessions.setText("0")
            self.lbl_queue.setText("0")
            self.main_window.statusBar().showMessage(
                f"Server: Stopped | Max CPU: {ALLOWED_THREADS}"
            )

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

        # A02: RSA 3072-bit (đồng bộ với ssl_utils.py, NIST recommends >=3072)
        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)

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
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=730))
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
        # A02: Restrict key file permissions (owner-only)
        try:
            import stat as _stat
            os.chmod(key_file, _stat.S_IRUSR | _stat.S_IWUSR)
        except OSError:
            pass  # Windows ACL không set qua chmod
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Hien thi danh sach IP/domain da them
        ip_list = ", ".join(v for _, v in sorted(san_entries))
        QMessageBox.information(self, "Tạo cert SSL",
            f"Đã tạo chứng chỉ SSL thành công!\n\n"
            f"SAN: {ip_list}\n\n"
            f"Vui lòng khởi động lại server để áp dụng.")
        self._refresh_cert_status()

    def _browse_cert_file(self, target_edit: QLineEdit):
        """Mở dialog chọn file cert/key."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file chứng chỉ", "",
            "PEM / CRT files (*.pem *.crt *.key *.cer);;All files (*)",
        )
        if path:
            target_edit.setText(path)

    def _refresh_cert_status(self):
        """Hiển thị thông tin cert đang được server sử dụng."""
        cert_dir = os.path.join(BASE_DIR, "web_service", "certs")
        custom_cert = os.path.join(cert_dir, "custom.crt")
        self_signed = os.path.join(cert_dir, "server.crt")

        # Xác định cert đang dùng
        if os.path.exists(custom_cert):
            cert_path = custom_cert
            cert_type = "Custom (import)"
        elif os.path.exists(self_signed):
            cert_path = self_signed
            cert_type = "Self-signed"
        else:
            self.lbl_cert_status.setText("Chưa có chứng chỉ SSL")
            self.lbl_cert_status.setStyleSheet(f"color: {COLORS['danger']};")
            return

        try:
            from cryptography import x509
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            # CN
            cn = ""
            for attr in cert.subject:
                if attr.oid == x509.oid.NameOID.COMMON_NAME:
                    cn = attr.value
                    break

            # SAN domains/IPs
            san_names = []
            try:
                san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                san_names.extend(san.value.get_values_for_type(x509.DNSName))
                san_names.extend(str(ip) for ip in san.value.get_values_for_type(x509.IPAddress))
            except x509.ExtensionNotFound:
                pass

            # Expiry
            import datetime
            expiry = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
            days_left = (expiry - now).days
            expiry_str = expiry.strftime("%Y-%m-%d")

            if days_left < 0:
                expiry_color = COLORS['danger']
                expiry_text = f"ĐÃ HẾT HẠN ({expiry_str})"
            elif days_left < 30:
                expiry_color = COLORS['warning']
                expiry_text = f"{expiry_str} (còn {days_left} ngày)"
            else:
                expiry_color = COLORS['success']
                expiry_text = f"{expiry_str} (còn {days_left} ngày)"

            san_str = ", ".join(san_names[:5])
            if len(san_names) > 5:
                san_str += f" (+{len(san_names) - 5})"

            self.lbl_cert_status.setText(
                f"Loại: {cert_type}  |  CN: {cn}\n"
                f"SAN: {san_str}\n"
                f"Hết hạn: {expiry_text}"
            )
            self.lbl_cert_status.setStyleSheet(f"color: {expiry_color};")
        except Exception as e:
            self.lbl_cert_status.setText(f"Lỗi đọc cert: {e}")
            self.lbl_cert_status.setStyleSheet(f"color: {COLORS['danger']};")

    def _import_custom_cert(self):
        """Validate và import cert + key custom vào certs/custom.crt + custom.key."""
        cert_path = self.edit_cert_file.text().strip()
        key_path = self.edit_key_file.text().strip()

        if not cert_path or not key_path:
            QMessageBox.warning(self, "Thiếu file", "Vui lòng chọn cả file Cert và Key.")
            return

        if not os.path.exists(cert_path):
            QMessageBox.warning(self, "Lỗi", f"File cert không tồn tại:\n{cert_path}")
            return
        if not os.path.exists(key_path):
            QMessageBox.warning(self, "Lỗi", f"File key không tồn tại:\n{key_path}")
            return

        # Validate cert + key
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            with open(cert_path, "rb") as f:
                cert_data = f.read()
            with open(key_path, "rb") as f:
                key_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data)
            key = load_pem_private_key(key_data, password=None)

            # Kiểm tra key khớp cert
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            cert_pubkey = cert.public_key().public_bytes(
                encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo,
            )
            key_pubkey = key.public_key().public_bytes(
                encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo,
            )
            if cert_pubkey != key_pubkey:
                QMessageBox.critical(self, "Lỗi", "Private key không khớp với certificate!")
                return

            # Lấy thông tin cert để hiển thị xác nhận
            cn = ""
            for attr in cert.subject:
                if attr.oid == x509.oid.NameOID.COMMON_NAME:
                    cn = attr.value
                    break

            san_names = []
            try:
                san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                san_names.extend(san.value.get_values_for_type(x509.DNSName))
                san_names.extend(str(ip) for ip in san.value.get_values_for_type(x509.IPAddress))
            except x509.ExtensionNotFound:
                pass

            import datetime
            expiry = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
            days_left = (expiry - now).days

            if days_left < 0:
                QMessageBox.critical(self, "Lỗi", f"Chứng chỉ đã hết hạn! ({expiry.strftime('%Y-%m-%d')})")
                return

            # Xác nhận import
            san_str = ", ".join(san_names) if san_names else "(không có)"
            reply = QMessageBox.question(
                self, "Xác nhận import",
                f"CN: {cn}\n"
                f"SAN: {san_str}\n"
                f"Hết hạn: {expiry.strftime('%Y-%m-%d')} (còn {days_left} ngày)\n\n"
                f"Import chứng chỉ này?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            # Copy vào certs/
            import shutil
            cert_dir = os.path.join(BASE_DIR, "web_service", "certs")
            os.makedirs(cert_dir, exist_ok=True)
            shutil.copy2(cert_path, os.path.join(cert_dir, "custom.crt"))
            shutil.copy2(key_path, os.path.join(cert_dir, "custom.key"))

            self._refresh_cert_status()
            QMessageBox.information(self, "Thành công",
                "Đã import chứng chỉ SSL!\n"
                "Vui lòng khởi động lại server để áp dụng.")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể import chứng chỉ:\n{e}")

    def _remove_custom_cert(self):
        """Xóa custom cert, server quay lại dùng self-signed."""
        cert_dir = os.path.join(BASE_DIR, "web_service", "certs")
        custom_cert = os.path.join(cert_dir, "custom.crt")
        custom_key = os.path.join(cert_dir, "custom.key")

        if not os.path.exists(custom_cert):
            QMessageBox.information(self, "Thông báo", "Không có cert custom nào để xóa.")
            return

        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            "Xóa chứng chỉ custom?\n"
            "Server sẽ quay lại dùng cert self-signed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if os.path.exists(custom_cert):
                os.remove(custom_cert)
            if os.path.exists(custom_key):
                os.remove(custom_key)
            self._refresh_cert_status()
            QMessageBox.information(self, "Thành công",
                "Đã xóa cert custom.\n"
                "Vui lòng khởi động lại server để áp dụng.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa: {e}")

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
            server_config.set("http_mode", "1" if self.chk_http_mode.isChecked() else "0")
            server_config.set("max_upload_mb", str(self.spin_max_upload.value()))
            server_config.set("anonymous_timeout_minutes", str(self.spin_timeout.value()))
            server_config.set("storage_per_user_gb", str(self.spin_storage.value()))
            server_config.set("max_sessions", str(self.spin_max_sessions.value()))
            server_config.set("jwt_expire_minutes", str(self.spin_jwt.value()))
            server_config.set("offline_download_url", self.edit_offline_url.text().strip())

            # Summarizer
            server_config.set("summarizer_enabled", "1" if self.chk_summarizer_enabled.isChecked() else "0")
            server_config.set("summarizer_model_path", self.edit_summarizer_gguf.text().strip())
            server_config.set("summarizer_threads", str(self.spin_llm_threads.value()))

            server_config.save()

            # Kiểm tra cert có cover host mới không (bỏ qua nếu HTTP mode)
            if not self.chk_http_mode.isChecked() and not self._check_cert_covers_host(new_host):
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
        self.log_view.setStyleSheet(f"background: {COLORS['bg_dark']}; color: {COLORS['text_secondary']}; border: 1px solid {COLORS['border']};")
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
        btn_install.setStyleSheet(f"background: {COLORS['accent']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
        btn_install.clicked.connect(self.install_service)
        btn_layout.addWidget(btn_install)

        btn_uninstall = QPushButton("Gỡ bỏ Service")
        btn_uninstall.setStyleSheet(f"background: {COLORS['danger']}; color: {COLORS['text_primary']}; padding: 8px 16px;")
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
            self.lbl_service_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        elif "RUNNING" in output:
            self.lbl_service_status.setText("Đang chạy")
            self.lbl_service_status.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")
        elif "STOPPED" in output:
            self.lbl_service_status.setText("Đã dừng")
            self.lbl_service_status.setStyleSheet(f"color: {COLORS['danger']};")
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

        # Chặn scroll wheel trên spinbox/combobox
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        # ── Global dark-theme stylesheet ──
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
            }}
            QTabWidget::pane {{
                border: none;
                background-color: {COLORS['bg_dark']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['accent']};
                color: {COLORS['text_primary']};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {COLORS['border']};
                color: {COLORS['text_primary']};
            }}
            QGroupBox {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: bold;
                color: {COLORS['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {COLORS['text_secondary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
            QPushButton:disabled {{
                opacity: 0.4;
                color: {COLORS['text_secondary']};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']};
            }}
            QTextEdit, QPlainTextEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QTableWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                gridline-color: {COLORS['border']};
                selection-background-color: {COLORS['accent']};
            }}
            QTableWidget QHeaderView::section {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                padding: 4px 8px;
                font-weight: bold;
            }}
            QCheckBox {{
                color: {COLORS['text_primary']};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QProgressBar {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                color: {COLORS['text_primary']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
            QStatusBar {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border']};
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_dark']};
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['border_light']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                height: 0px;
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 0px;
                height: 0px;
                border: none;
            }}
        """)

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

    def eventFilter(self, obj, event):
        """Chặn scroll wheel trên SpinBox/ComboBox."""
        if event.type() == event.Type.Wheel:
            if isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)):
                event.ignore()
                return True
        return super().eventFilter(obj, event)

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

        self.statusBar().showMessage(f"Server: Stopped | Max CPU: {ALLOWED_THREADS}")

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
