# common.py - Chứa các thành phần dùng chung
import sys
import os
import json
import unicodedata
import re
import multiprocessing
import numpy as np
from difflib import SequenceMatcher

import psutil

# === CPU Affinity / Thread Limiting Logic ===
def get_allowed_cpu_count():
    """Detects the number of allowed CPUs (respecting affinity/container limits)."""
    try:
        p = psutil.Process(os.getpid())
        affinity = p.cpu_affinity()
        if affinity:
            return len(affinity)
    except Exception as e:
        print(f"[Init] Could not get CPU affinity: {e}")
    
    try:
        return os.cpu_count() or multiprocessing.cpu_count()
    except:
        return 4

ALLOWED_THREADS = get_allowed_cpu_count()

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit, 
                             QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout, QMessageBox, 
                             QFrame, QStyle, QTabWidget, QToolButton, QLineEdit, QDialog, 
                             QMenu, QListWidget, QRadioButton, QInputDialog)
from PyQt6.QtCore import Qt, QMimeData, pyqtSignal, QUrl, QTime, QTimer, QThread
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor, QTextBlockFormat, QColor, QTextCharFormat, QPalette
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QAudioSource, QMediaDevices, QAudioFormat

# Import transcriber
from transcriber import TranscriberThread

# Import streaming ASR
from streaming_asr import StreamingASRManager
from streaming_asr_online import OnlineStreamingASRManager

# Import Speaker Diarization
try:
    from speaker_diarization import SpeakerDiarizer, SPEAKER_EMBEDDING_MODELS
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    SPEAKER_EMBEDDING_MODELS = {}

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Config file path
CONFIG_FILE = os.path.join(BASE_DIR, "config.ini")

# Color scheme
COLORS = {
    'bg_dark': '#2b2b2b',
    'bg_card': '#3a3a3a',
    'bg_input': '#f5f5f5',
    'text_primary': '#ffffff',
    'text_secondary': '#cccccc',
    'text_dark': '#222222',
    'accent': '#007bff',
    'accent_hover': '#0056b3',
    'border': '#555555',
    'border_light': '#aaaaaa',
    'highlight': '#ffd700',
    'success': '#28a745',
    'warning': '#ffc107',
    'search_match': '#00ced1',
    'search_current': '#ff4500',
}

# Helper functions for fuzzy search
def normalize_vietnamese(text):
    """Chuyển text về dạng không dấu, lowercase để search không dấu"""
    if not text:
        return ""
    text = text.lower()
    # Thay thế 'đ' thủ công vì Unicode NFD không decompose 'đ' thành 'd'
    text = text.replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

def fuzzy_score(query, text):
    """Tính độ tương đồng fuzzy giữa query và text (0.0 - 1.0)"""
    if not query or not text:
        return 0.0
    if query.lower() in text.lower():
        return 1.0
    query_norm = normalize_vietnamese(query)
    text_norm = normalize_vietnamese(text)
    if query_norm in text_norm:
        return 0.9
    return SequenceMatcher(None, query_norm, text_norm).ratio()

def find_fuzzy_matches(query, text, threshold=0.6):
    """Tìm tất cả các vị trí fuzzy match trong text"""
    matches = []
    if not query or not text:
        return matches
    
    query_lower = query.lower()
    text_lower = text.lower()
    query_len = len(query)
    
    # First: try exact match (case insensitive)
    start = 0
    while True:
        idx = text_lower.find(query_lower, start)
        if idx == -1:
            break
        matches.append((idx, idx + query_len, text[idx:idx + query_len], 1.0))
        start = idx + 1
    
    # Second: try normalized (no accent) match
    query_norm = normalize_vietnamese(query)
    text_norm = normalize_vietnamese(text)
    start = 0
    while True:
        idx = text_norm.find(query_norm, start)
        if idx == -1:
            break
        orig_start = idx
        orig_end = idx + query_len
        matches.append((orig_start, orig_end, text[orig_start:orig_end], 0.9))
        start = idx + 1
    
    # Remove duplicate positions (keep highest score)
    seen = set()
    unique_matches = []
    for start, end, matched_text, score in matches:
        key = (start, end)
        if key not in seen:
            seen.add(key)
            unique_matches.append((start, end, matched_text, score))
    
    return unique_matches


class DragDropLabel(QLabel):
    fileDropped = pyqtSignal(str)
    clicked = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.has_file = False
        self.setDefaultText()
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {COLORS['border_light']};
                border-radius: 8px;
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                font-size: 13px;
                padding: 10px;
            }}
            QLabel:hover {{
                border-color: {COLORS['accent']};
                background-color: #e8f4ff;
            }}
        """)
        self.setAcceptDrops(True)

    def setDefaultText(self):
        self.has_file = False
        self.setText("📁 Kéo thả file âm thanh/video (mp3, wav, m4a, mp4, mkv, avi...) hoặc bấm để chọn")
        self.setToolTip("")

    def setFileText(self, filename):
        self.has_file = True
        self.setText(f"📄 <b>{filename}</b><br><span style='font-size:11px;color:#666;'>Bấm để đổi file | Kéo thả file khác</span>")
        self.setToolTip(f"File đã chọn: {filename}\nBấm để chọn file khác")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.fileDropped.emit(files[0])

    def mousePressEvent(self, event):
        self.clicked.emit()


class SearchWidget(QWidget):
    searchRequested = pyqtSignal(str)
    nextRequested = pyqtSignal()
    prevRequested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.input = QLineEdit()
        self.input.setPlaceholderText("Tìm...")
        self.input.setFixedWidth(100)
        self.input.setFixedHeight(22)
        self.input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {COLORS['border']};
                border-radius: 3px 0 0 3px;
                padding: 1px 4px;
                font-size: 11px;
                color: {COLORS['text_dark']};
                background-color: {COLORS['bg_input']};
                border-right: none;
                margin: 0px;
            }}
            QLineEdit:focus {{
                border: 1px solid {COLORS['accent']};
                border-right: none;
            }}
        """)
        self.input.returnPressed.connect(self.on_return_pressed)
        
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._do_search)
        self.input.textChanged.connect(self._on_text_changed)
        
        self.label_count = QLabel("0/0")
        self.label_count.setFixedWidth(28)
        self.label_count.setFixedHeight(22)
        self.label_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_count.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 9px;
                background-color: {COLORS['bg_input']};
                padding: 0px;
                margin: 0px;
                border: 1px solid {COLORS['border']};
                border-left: none;
                border-right: none;
            }}
        """)
        
        self.btn_up = self.create_nav_button("▲", self.prevRequested)
        self.btn_down = self.create_nav_button("▼", self.nextRequested)
        self.btn_close = self.create_nav_button("✕", self.closed, last=True)
        
        self.layout.addWidget(self.input)
        self.layout.addWidget(self.label_count)
        self.layout.addWidget(self.btn_up)
        self.layout.addWidget(self.btn_down)
        self.layout.addWidget(self.btn_close)
        self.layout.addStretch(0)
        
    def _on_text_changed(self, text):
        self._search_timer.stop()
        self._search_timer.start(300)
    
    def _do_search(self):
        text = self.input.text()
        self.searchRequested.emit(text)
        
    def set_input_width(self, width):
        """Set the width of the search input field"""
        self.input.setFixedWidth(width)
    
    def create_nav_button(self, text, signal, last=False):
        btn = QToolButton()
        btn.setText(text)
        btn.setFixedSize(18, 22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        radius = "0 3px 3px 0" if last else "0"
        border_right = "1px" if last else "none"
        btn.setStyleSheet(f"""
            QToolButton {{
                border: 1px solid {COLORS['border']};
                border-left: none;
                border-right: {border_right} solid {COLORS['border']};
                border-radius: {radius};
                background: {COLORS['bg_input']};
                color: {COLORS['text_secondary']};
                font-size: 9px;
                margin: 0px;
                padding: 0px;
            }}
            QToolButton:hover {{
                color: {COLORS['text_primary']};
                background-color: {COLORS['border']};
            }}
        """)
        btn.clicked.connect(signal.emit)
        return btn

    def on_return_pressed(self):
        text = self.input.text()
        if text:
            self.searchRequested.emit(text)
    
    def clear_search(self):
        self._search_timer.stop()
        self.input.clear()
        self.searchRequested.emit("")


class ClickableTextEdit(QTextEdit):
    sentenceClicked = pyqtSignal(int)
    speakerLabelClicked = pyqtSignal(str, int)
    splitSpeakerRequested = pyqtSignal(int)
    mergeSpeakerRequested = pyqtSignal(int, str)  # sentence_idx, direction ('prev' or 'next')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                font-size: 14px;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def mousePressEvent(self, event):
        anchor = self.anchorAt(event.pos())
        
        # Fallback: dùng cursorForPosition khi anchorAt không tìm thấy anchor
        # anchorAt() có thể trả về rỗng khi click vào khoảng trống giữa các anchor
        # hoặc vào <span> lồng bên trong <a>
        if not anchor:
            cursor = self.cursorForPosition(event.pos())
            fmt = cursor.charFormat()
            if fmt.isAnchor():
                anchor = fmt.anchorHref()
        
        if anchor:
            if anchor.startswith("s_"):
                try:
                    idx = int(anchor.split("_")[1])
                    self.sentenceClicked.emit(idx)
                except Exception:
                    pass
            elif anchor.startswith("spk_"):
                try:
                    parts = anchor.split("_")
                    speaker_id = parts[1]
                    block_index = int(parts[2]) if len(parts) > 2 else 0
                    self.speakerLabelClicked.emit(speaker_id, block_index)
                except Exception:
                    pass
            elif anchor.startswith("livespk_"):
                try:
                    name = anchor.split("livespk_")[1]
                    self.speakerLabelClicked.emit(name, -1)
                except Exception:
                    pass
        super().mousePressEvent(event)
    
    def show_context_menu(self, position):
        cursor = self.cursorForPosition(position)
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        
        anchor = self.anchorAt(position)
        sent_idx = None
        
        if anchor and anchor.startswith("s_"):
            try:
                sent_idx = int(anchor.split("_")[1])
            except:
                pass
        else:
            html = self.toHtml()
            block = cursor.block()
            block_text = block.text()
            
            anchors_in_block = re.findall(r'href=\'s_(\d+)\'', block_text)
            if anchors_in_block:
                sent_idx = int(anchors_in_block[0])
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS['accent']};
                color: white;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {COLORS['border']};
                margin: 4px 8px;
            }}
        """)
        
        if sent_idx is not None:
            split_action = menu.addAction(f"🔀 Tách người nói từ câu này")
            split_action.triggered.connect(lambda: self.splitSpeakerRequested.emit(sent_idx))
            
            # Thêm tùy chọn gộp với người nói trước/sau
            merge_menu = menu.addMenu("⬆️⬇️ Gộp với người nói...")
            merge_prev = merge_menu.addAction("⬆️ Gộp với người nói phía trước")
            merge_prev.triggered.connect(lambda: self.mergeSpeakerRequested.emit(sent_idx, 'prev'))
            merge_next = merge_menu.addAction("⬇️ Gộp với người nói phía sau")
            merge_next.triggered.connect(lambda: self.mergeSpeakerRequested.emit(sent_idx, 'next'))
            
            menu.addSeparator()
        
        copy_action = menu.addAction("📋 Sao chép")
        copy_action.triggered.connect(self.copy)
        
        select_all_action = menu.addAction("📄 Chọn tất cả")
        select_all_action.triggered.connect(self.selectAll)
        
        menu.exec(self.viewport().mapToGlobal(position))


class SpeakerRenameDialog(QDialog):
    """Dialog để đổi tên người nói"""
    
    def __init__(self, current_speaker_id, current_name, custom_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Đổi tên người nói")
        self.setFixedSize(400, 350)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['accent']};
            }}
            QListWidget {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
                font-size: 13px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-radius: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['accent']};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['border']};
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
            QPushButton#cancelBtn {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton#cancelBtn:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        
        self.current_speaker_id = current_speaker_id
        self.current_name = current_name
        self.custom_names = custom_names
        self.selected_name = None
        self.apply_to_all = False
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        info_label = QLabel(f"<b>người nói:</b> {self.current_name}")
        layout.addWidget(info_label)
        
        name_label = QLabel("Tên mới:")
        layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập tên mới...")
        layout.addWidget(self.name_input)
        
        if self.custom_names:
            list_label = QLabel("Hoặc chọn từ danh sách đã có:")
            layout.addWidget(list_label)
            
            self.names_list = QListWidget()
            for name in sorted(self.custom_names):
                self.names_list.addItem(name)
            self.names_list.itemClicked.connect(self.on_name_selected)
            layout.addWidget(self.names_list)
        else:
            self.names_list = None
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        
        self.btn_apply_all = QPushButton("🔄 Sửa tất cả")
        self.btn_apply_all.setMinimumWidth(140)
        self.btn_apply_all.setToolTip("Đổi tên hiển thị cho tất cả đoạn có cùng người nói này")
        self.btn_apply_all.clicked.connect(self.on_apply_all)
        self.btn_apply_all.setEnabled(False)
        
        self.btn_select_only = QPushButton("✓ Sửa tên này")
        self.btn_select_only.setToolTip("Gán lại đoạn này cho người nói khác")
        self.btn_select_only.clicked.connect(self.on_select_only)
        self.btn_select_only.setEnabled(False)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setObjectName("cancelBtn")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_apply_all)
        btn_layout.addWidget(self.btn_select_only)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        self.name_input.textChanged.connect(self.on_input_changed)
    
    def on_input_changed(self, text):
        has_text = bool(text.strip())
        self.btn_apply_all.setEnabled(has_text)
        self.btn_select_only.setEnabled(has_text)
    
    def on_name_selected(self, item):
        self.name_input.setText(item.text())
    
    def on_apply_all(self):
        self.selected_name = self.name_input.text().strip()
        self.apply_to_all = True
        self.accept()
    
    def on_select_only(self):
        self.selected_name = self.name_input.text().strip()
        self.apply_to_all = False
        self.accept()
    
    def get_result(self):
        return self.selected_name, self.apply_to_all


class SpeakerHotkeyDialog(QDialog):
    """Dialog để cấu hình Hotkey người nói"""
    def __init__(self, hotkey_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cấu hình Hotkey Người nói")
        self.setFixedSize(450, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['accent']};
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)
        
        self.hotkey_config = hotkey_config.copy() if hotkey_config else {}
        for i in range(1, 10):
            if str(i) not in self.hotkey_config:
                self.hotkey_config[str(i)] = ""
                
        self.inputs = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("<h2>Cấu hình Phím tắt Người nói</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>STT</b>"))
        header_layout.addWidget(QLabel("<b>Phím</b>"))
        header_layout.addWidget(QLabel("<b>Tên người nói (Để trống = Ẩn)</b>"))
        layout.addLayout(header_layout)
        
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(8)
        
        for i in range(1, 10):
            row_layout = QHBoxLayout()
            
            lbl_stt = QLabel(str(i))
            lbl_stt.setFixedWidth(30)
            lbl_stt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            lbl_key = QLabel(f"Num {i}")
            lbl_key.setFixedWidth(50)
            lbl_key.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
            lbl_key.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            inp = QLineEdit()
            inp.setPlaceholderText(f"Nhập tên cho phím {i}...")
            inp.setText(self.hotkey_config.get(str(i), ""))
            self.inputs[str(i)] = inp
            
            row_layout.addWidget(lbl_stt)
            row_layout.addWidget(lbl_key)
            row_layout.addWidget(inp)
            
            content_layout.addLayout(row_layout)
            
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        info = QLabel("<i>Lưu ý: Trong khi dịch trực tiếp, bấm phím số tương ứng để chèn tên người nói.</i>")
        info.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Lưu cấu hình")
        btn_save.clicked.connect(self.save_config)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setStyleSheet(f"background-color: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; color: {COLORS['text_primary']};")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        layout.addLayout(btn_layout)
        
    def save_config(self):
        for i in range(1, 10):
            self.hotkey_config[str(i)] = self.inputs[str(i)].text().strip()
        self.accept()
        
    def get_config(self):
        return self.hotkey_config


class SplitSpeakerDialog(QDialog):
    """Dialog để tách người nói"""
    
    def __init__(self, current_speaker, custom_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tách người nói")
        self.setFixedSize(380, 280)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['accent']};
            }}
            QComboBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 10px;
                font-size: 14px;
                min-height: 24px;
            }}
            QComboBox:focus {{
                border: 2px solid {COLORS['accent']};
            }}
            QComboBox::drop-down {{
                border-left: 2px solid {COLORS['border']};
                width: 32px;
                background-color: {COLORS['bg_dark']};
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border-radius: 0 3px 3px 0;
            }}
            QComboBox::drop-down:hover {{
                background-color: {COLORS['accent']};
                border-left: 2px solid {COLORS['accent']};
            }}
            QComboBox::down-arrow {{
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjMiPjxwYXRoIGQ9Ik02IDlsNiA2IDYtNiIvPjwvc3ZnPg==);
                width: 14px;
                height: 14px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']};
                border: 1px solid {COLORS['border']};
                padding: 4px;
            }}
            QComboBox QLineEdit {{
                padding: 4px;
                background: transparent;
                border: none;
            }}
            QRadioButton {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLORS['border']};
                border-radius: 9px;
                background-color: {COLORS['bg_card']};
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid {COLORS['accent']};
                background-color: {COLORS['accent']};
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNiIvPjwvc3ZnPg==);
            }}
            QRadioButton::indicator:hover {{
                border: 2px solid {COLORS['accent_hover']};
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton#cancelBtn {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton#cancelBtn:hover {{
                background-color: {COLORS['border']};
            }}
        """)
        
        self.current_speaker = current_speaker
        self.custom_names = custom_names
        self.new_speaker_name = None
        self.split_scope = "to_end"
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        info_label = QLabel(f"<b>người nói hiện tại:</b> {self.current_speaker}")
        layout.addWidget(info_label)
        
        layout.addSpacing(8)
        new_speaker_label = QLabel("Chọn người nói mới:")
        layout.addWidget(new_speaker_label)
        
        self.speaker_combo = QComboBox()
        self.speaker_combo.setEditable(True)
        self.speaker_combo.setPlaceholderText("Nhập tên người nói mới...")
        self.speaker_combo.setMinimumHeight(32)
        
        all_speakers = set(self.custom_names)
        
        for name in sorted(all_speakers):
            if name != self.current_speaker:
                self.speaker_combo.addItem(name)
        
        # Khi chọn từ dropdown, line edit tự động cập nhật (do setEditable=True)
        # Không cần connect signal để tránh lỗi
        
        layout.addWidget(self.speaker_combo)
        
        layout.addSpacing(12)
        scope_label = QLabel("Phạm vi tách:")
        layout.addWidget(scope_label)
        
        self.radio_to_end = QRadioButton("Từ câu này đến hết đoạn")
        self.radio_to_end.setChecked(True)
        self.radio_only_this = QRadioButton("Chỉ câu này")
        
        layout.addWidget(self.radio_to_end)
        layout.addWidget(self.radio_only_this)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        
        btn_ok = QPushButton("✓ Xác nhận")
        btn_ok.clicked.connect(self.on_confirm)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setObjectName("cancelBtn")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
    
    def on_confirm(self):
        # Lấy giá trị hiện tại từ combo box (dù nhập tay hay chọn từ dropdown)
        # Đảm bảo lấy đúng giá trị từ line edit (tránh lỗi khi nhập nhanh)
        if self.speaker_combo.lineEdit():
            self.new_speaker_name = self.speaker_combo.lineEdit().text().strip()
        else:
            self.new_speaker_name = self.speaker_combo.currentText().strip()
            
        if not self.new_speaker_name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tên người nói mới!")
            return
        
        self.split_scope = "to_end" if self.radio_to_end.isChecked() else "only_this"
        self.accept()
    
    def get_result(self):
        return self.new_speaker_name, self.split_scope


class SpeakerDiarizationThread(QThread):
    """Worker thread for running speaker diarization only (without transcription)"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, float, list)
    error = pyqtSignal(str)
    
    def __init__(self, audio_file, segments, speaker_model_id, num_speakers, num_threads, **kwargs):
        super().__init__()
        self.audio_file = audio_file
        self.segments = segments
        self.speaker_model_id = speaker_model_id
        self.num_speakers = num_speakers
        self.num_threads = num_threads
        self.threshold = kwargs.get('threshold', 0.6)
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            import time
            start_time = time.time()
            
            try:
                from speaker_diarization import _setup_ffmpeg_path
                _setup_ffmpeg_path()
            except:
                pass
            
            self.progress.emit("PHASE:Diarization|Đang khởi tạo model|0")
            
            diarizer = SpeakerDiarizer(
                embedding_model_id=self.speaker_model_id,
                num_clusters=self.num_speakers,
                num_threads=self.num_threads,
                threshold=self.threshold
            )
            diarizer.initialize()
            
            self.progress.emit("PHASE:Diarization|Đang phân tách Người nói|10")
            
            def progress_callback(num_processed, num_total):
                progress = int(10 + (num_processed / num_total) * 80)
                self.progress.emit(f"PHASE:Diarization|Đang phân tách Người nói|{progress}")
                return 0 if self.is_running else 1
            
            raw_segments = diarizer.process(self.audio_file, progress_callback=progress_callback)
            
            speaker_segments_raw = [
                {
                    "speaker": f"Người nói {seg.speaker + 1}",
                    "speaker_id": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": seg.duration
                }
                for seg in raw_segments
            ]
            
            self.progress.emit("PHASE:Diarization|Đang gán nhãn Người nói|90")
            
            results = diarizer.process_with_transcription(
                audio_file=self.audio_file,
                transcribed_segments=self.segments,
                speaker_segments=raw_segments
            )
            
            elapsed = time.time() - start_time
            self.progress.emit("PHASE:Diarization|Hoàn thành|100")
            
            self.finished.emit(speaker_segments_raw, elapsed, results)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MicrophoneRecordThread(QThread):
    """Thread for recording audio from microphone"""
    volume_changed = pyqtSignal(float)
    chunk_ready = pyqtSignal(bytes)
    error = pyqtSignal(str)
    
    def __init__(self, device_index=None, sample_rate=16000):
        super().__init__()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_source = None
        self.io_device = None
        
    def stop(self):
        self.is_recording = False
        if self.io_device:
            self.io_device.close()
        
    def _resample_audio(self, audio_data, src_rate, dst_rate):
        """Simple resampling using linear interpolation (Issue #11)"""
        if src_rate == dst_rate:
            return audio_data
        
        # Calculate resampling ratio
        ratio = dst_rate / src_rate
        new_length = int(len(audio_data) * ratio)
        
        # Use scipy if available, otherwise simple linear interpolation
        try:
            from scipy import signal
            resampled = signal.resample(audio_data, new_length)
            return resampled.astype(np.int16)
        except ImportError:
            # Simple linear interpolation fallback
            old_indices = np.arange(len(audio_data))
            new_indices = np.linspace(0, len(audio_data) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, audio_data)
            return resampled.astype(np.int16)
    
    def run(self):
        try:
            self.is_recording = True
            
            devices = QMediaDevices.audioInputs()
            if not devices:
                self.error.emit("Không tìm thấy microphone")
                return
                
            device = devices[self.device_index] if self.device_index is not None and self.device_index < len(devices) else devices[0]
            
            format = QAudioFormat()
            format.setSampleRate(self.sample_rate)
            format.setChannelCount(1)
            format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            
            print(f"[MicrophoneRecordThread] Requested format: {self.sample_rate}Hz, Mono, 16-bit")
            
            actual_channels = 1
            actual_sample_rate = self.sample_rate
            if not device.isFormatSupported(format):
                print(f"[MicrophoneRecordThread] Warning: Format not supported, using preferred")
                format = device.preferredFormat()
                actual_channels = format.channelCount()
                actual_sample_rate = format.sampleRate()
                print(f"[MicrophoneRecordThread] Preferred: {actual_sample_rate}Hz, {actual_channels}ch")
            
            self.audio_source = QAudioSource(device, format)
            self.io_device = self.audio_source.start()
            
            print(f"[MicrophoneRecordThread] Recording started, format: {actual_sample_rate}Hz, {actual_channels}ch, {format.sampleFormat()}")
            
            # Calculate bytes to read based on actual sample rate (Issue #11)
            bytes_per_read = int(actual_sample_rate * 2 * actual_channels * 0.05)  # 50ms worth of data
            
            while self.is_recording:
                if self.io_device and self.io_device.bytesAvailable() > 0:
                    data = self.io_device.readAll()
                    data_bytes = bytes(data)
                    
                    if len(data_bytes) == 0:
                        continue
                    
                    # Convert to numpy for processing
                    audio_data = np.frombuffer(data_bytes, dtype=np.int16)
                    
                    # Stereo to mono conversion if needed
                    if actual_channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    
                    # Resample if needed (Issue #11: sample rate mismatch)
                    if actual_sample_rate != self.sample_rate:
                        audio_data = self._resample_audio(audio_data, actual_sample_rate, self.sample_rate)
                    
                    # Accumulate vào buffer để đảm bảo chunk size đều 50ms (800 samples @ 16kHz)
                    if not hasattr(self, '_audio_accumulator'):
                        self._audio_accumulator = np.array([], dtype=np.int16)
                    
                    self._audio_accumulator = np.concatenate([self._audio_accumulator, audio_data])
                    
                    samples_per_chunk = int(self.sample_rate * 0.05)  # 50ms = 800 samples
                    
                    # Emit các chunk đủ 50ms
                    while len(self._audio_accumulator) >= samples_per_chunk:
                        chunk = self._audio_accumulator[:samples_per_chunk]
                        self._audio_accumulator = self._audio_accumulator[samples_per_chunk:]
                        
                        chunk_bytes = chunk.tobytes()
                        
                        if not hasattr(self, '_chunk_count'):
                            self._chunk_count = 0
                        self._chunk_count += 1
                        
                        if self._chunk_count <= 3:
                            print(f"[MicrophoneRecordThread] Chunk #{self._chunk_count}: {len(chunk_bytes)} bytes, {len(chunk)} samples @ {self.sample_rate}Hz")
                        
                        self.chunk_ready.emit(chunk_bytes)
                        
                        # Tính volume cho từng chunk
                        volume = np.abs(chunk).mean() / 32768.0
                        self.volume_changed.emit(min(volume * 3, 1.0))
                        
                QThread.msleep(20)
                
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# === Model Download Information ===
MODEL_DOWNLOAD_INFO = {
    "sherpa-onnx-zipformer-vi-2025-04-20": {
        "name": "Sherpa-ONNX Zipformer Vietnamese",
        "hf_url": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20",
        "description": "Model ASR chính cho tiếng Việt",
        "files": ["encoder-epoch-12-avg-8.onnx", "decoder-epoch-12-avg-8.onnx", "joiner-epoch-12-avg-8.onnx"]
    },
    "zipformer-30m-rnnt-6000h": {
        "name": "Zipformer-30M-RNNT-6000h",
        "hf_url": "https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h",
        "description": "Model ASR nhẹ, nhanh",
        "files": ["encoder-epoch-20-avg-10.onnx", "decoder-epoch-20-avg-10.onnx", "joiner-epoch-20-avg-10.onnx"]
    },
    "zipformer-30m-rnnt-streaming-6000h": {
        "name": "Zipformer-30M-RNNT-Streaming-6000h",
        "hf_url": "https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h",
        "description": "Model ASR streaming cho thu âm trực tiếp (chunk 64)",
        "files": ["encoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx", "decoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx", "joiner-epoch-31-avg-11-chunk-64-left-128.fp16.onnx"]
    },
    "sat-12l-sm": {
        "name": "SAT (Segment Any Text)",
        "hf_url": "https://huggingface.co/segment-any-text/sat-12l-sm",
        "description": "Model tách câu/tách đoạn",
        "files": ["model_optimized.onnx"]
    },
    "vibert-capu": {
        "name": "ViBERT-capu",
        "hf_url": "https://huggingface.co/dragonSwing/vibert-capu",
        "description": "Model thêm dấu câu tiếng Việt",
        "files": ["pytorch_model.bin"]
    }
}


def show_missing_model_dialog(parent, model_name: str, model_path: str):
    """
    Hiển thị thông báo thiếu model với hướng dẫn tải về.
    
    Args:
        parent: QWidget parent cho QMessageBox
        model_name: Tên model (key trong MODEL_DOWNLOAD_INFO)
        model_path: Đường dẫn local mong đợi
    """
    if model_name not in MODEL_DOWNLOAD_INFO:
        # Model không có trong danh sách - hiển thị thông báo chung
        QMessageBox.critical(
            parent,
            "Lỗi - Không tìm thấy Model",
            f"Không tìm thấy model: {model_name}\n\n"
            f"Thư mục mong đợi: {model_path}\n\n"
            f"Vui lòng tải model và giải nén vào thư mục models/"
        )
        return
    
    info = MODEL_DOWNLOAD_INFO[model_name]
    
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Thiếu Model - Cần tải về")
    msg.setText(f"<b>{info['name']}</b>")
    msg.setInformativeText(
        f"<b>Mô tả:</b> {info['description']}<br><br>"
        f"<b>Thư mục cần có:</b><br>"
        f"<code>{model_path}</code><br><br>"
        f"<b>Các file cần tải:</b><br>"
        f"{', '.join(info['files'][:2])}{'...' if len(info['files']) > 2 else ''}<br><br>"
        f"<b>Hướng dẫn:</b><br>"
        f"1. Truy cập: <a href='{info['hf_url']}'>{info['hf_url']}</a><br>"
        f"2. Click 'Files and versions'<br>"
        f"3. Tải tất cả files về<br>"
        f"4. Giải nén vào thư mục: <code>models/{model_name}/</code><br><br>"
        f"<i>Hoặc chạy script build:</i><br>"
        f"<code>python build-portable/prepare_offline_build.py --download</code>"
    )
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    msg.exec()
