# tab_file.py - Tab xử lý tập tin âm thanh
import sys
import os
import re
import unicodedata

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QProgressBar, QTextEdit, QComboBox, QSlider, 
                             QCheckBox, QFrame, QFormLayout, QMessageBox, QToolButton, 
                             QTabWidget, QStyle, QDialog)
from PyQt6.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from core.config import BASE_DIR, COLORS, MODEL_DOWNLOAD_INFO, DEBUG_LOGGING, ALLOWED_THREADS
from core.config import get_speaker_embedding_models, is_diarization_available
from core.utils import normalize_vietnamese
from core.asr_json import serialize_segments, deserialize_segments, load_asr_json, save_asr_json as _save_asr_json_file
from common import (DragDropLabel, SearchWidget, ClickableTextEdit,
                    SpeakerRenameDialog, SplitSpeakerDialog, SpeakerDiarizationThread,
                    TranscriberThread, show_missing_model_dialog)

DIARIZATION_AVAILABLE = is_diarization_available()
SPEAKER_EMBEDDING_MODELS = get_speaker_embedding_models()
from core.audio_analyzer import (
    AudioQualityAnalyzer, AnalysisResult, QualityMetrics,
    AnalysisThread, check_dnsmos_model_exists, DNSMOSDownloader
)
from quality_result_dialog import QualityResultDialog

class FastJSONLoadThread(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_loading = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, json_path, parent=None):
        super().__init__(parent)
        self.json_path = json_path
        
    def run(self):
        try:
            self.progress_updated.emit(10, "Đang đọc file JSON...")
            data = load_asr_json(self.json_path)

            self.progress_updated.emit(40, "Đang xử lý dữ liệu...")
            segments, speaker_mapping, has_speakers = deserialize_segments(data)

            self.progress_updated.emit(75, "Đang chuẩn bị hiển thị...")

            # Lưu trữ vào self, KO truyền qua signal để tránh PyQt pickling lớn gây lag UI
            self.result_segments = segments
            self.result_speaker_mapping = speaker_mapping
            self.result_has_speakers = has_speakers

            self.finished_loading.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

class BackgroundAudioConvertThread(QThread):
    finished_converting = pyqtSignal(str) # -> Trả về đường dẫn file wav

    def __init__(self, audio_path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path

    def run(self):
        try:
            import os
            import subprocess
            import tempfile
            
            file_ext = os.path.splitext(self.audio_path)[1].lower()
            if file_ext == '.wav':
                self.finished_converting.emit(self.audio_path)
                return
                
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, prefix='asr_playback_')
            temp_path = temp_file.name
            temp_file.close()

            import sys
            if getattr(sys, 'frozen', False):
                ffmpeg_path = os.path.join(os.path.dirname(sys.executable), 'ffmpeg', 'bin', 'ffmpeg.exe')
                if not os.path.exists(ffmpeg_path):
                    ffmpeg_path = 'ffmpeg' # fallback to system ffmpeg
            else:
                from common import BASE_DIR
                ffmpeg_path = os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffmpeg.exe')
                if not os.path.exists(ffmpeg_path):
                    ffmpeg_path = 'ffmpeg'
            
            # ffmpeg -i file -vn -ac 1 -c:a pcm_s16le -y temp
            command = [
                ffmpeg_path,
                '-i', self.audio_path,
                '-vn',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-loglevel', 'quiet',
                '-y',
                temp_path
            ]
            
            # Try FFmpeg directly
            try:
                subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                self.finished_converting.emit(temp_path)
            except Exception as ffmpeg_err:
                print(f"[BackgroundAudioConvertThread] FFmpeg pipe failed: {ffmpeg_err}. Falling back to pydub...")
                from pydub import AudioSegment
                audio = AudioSegment.from_file(self.audio_path)
                audio.export(temp_path, format='wav')
                self.finished_converting.emit(temp_path)
            
        except Exception as e:
            print(f"[BackgroundAudioConvertThread] Lỗi chuyển đổi: {e}")
            self.finished_converting.emit(self.audio_path) # Fallback to original


class FileProcessingTab(QWidget):
    """Tab xử lý tập tin âm thanh"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.selected_file = None
        self.transcriber = None
        self.default_model_path = os.path.join(BASE_DIR, "models", "sherpa-onnx-zipformer-vi-2025-04-20")
        self.segments = []
        self.current_highlight_index = -1
        self._playback_cache = {}  # {original_file_path: temp_wav_path}
        
        # Search state
        self.search_matches = []
        self.current_search_index = -1
        self.last_query = ""
        
        # Debounce timer for render
        self.render_debounce_timer = QTimer(self)
        self.render_debounce_timer.setSingleShot(True)
        self.render_debounce_timer.timeout.connect(self._do_render)
        self.pending_highlight_idx = -1
        self._last_rendered_highlight = -1

        # Media Player Init
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)
        self.player_duration = 0

        # Paragraph state (from SAT)
        self.paragraphs = []
        
        # Speaker name management
        self.speaker_name_mapping = {}
        self.block_speaker_names = {}
        self.custom_speaker_names = set()
        self.merged_speaker_blocks = []
        
        # JSON load flag
        self.loaded_from_json = False
        self._user_clicked_timestamp = 0
        
        # Track if JSON has been saved
        self.json_saved = False
        
        # Progress bar animation
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        self.current_progress_text = "Đang chờ..."
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.update_spinner)
        
        # Initialization flag
        self._initializing = True
        
        # Audio quality analyzer
        self.quality_analyzer = None
        self.analysis_thread = None
        self._init_quality_analyzer()
        
        self.init_ui()
        
        self._initializing = False
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Configuration Group (Collapsible Custom)
        self.config_container = QWidget()
        config_layout = QVBoxLayout(self.config_container)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(0)

        # Header (Triangle + Label)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        self.btn_toggle_config = QToolButton()
        self.btn_toggle_config.setArrowType(Qt.ArrowType.DownArrow)
        self.btn_toggle_config.setStyleSheet(f"""
            QToolButton {{
                border: none;
                background: transparent;
                color: {COLORS['text_primary']};
                max-width: 12px;
                max-height: 12px;
            }}
            QToolButton:hover {{
                background: {COLORS['bg_card']};
                border-radius: 2px;
            }}
        """)
        self.btn_toggle_config.clicked.connect(self.toggle_config)
        
        self.label_config_header = QLabel("Cấu hình")
        self.label_config_header.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        self.label_config_header.mousePressEvent = self.on_header_click
        
        header_layout.addWidget(self.btn_toggle_config)
        header_layout.addWidget(self.label_config_header)
        header_layout.addStretch()
        
        # Nút Thông tin nhỏ
        self.btn_about = QPushButton("ⓘ")
        self.btn_about.setFixedSize(20, 20)
        self.btn_about.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                color: white;
                border-color: {COLORS['accent']};
            }}
        """)
        self.btn_about.setToolTip("Thông tin phần mềm")
        self.btn_about.clicked.connect(self.show_about_dialog)
        header_layout.addWidget(self.btn_about)
        
        config_layout.addWidget(header_widget)
        
        # Content Frame
        self.config_content = QFrame()
        self.config_content.setVisible(True)
        self.config_content.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                background-color: transparent;
                margin-top: 4px;
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                border: none;
            }}
            QComboBox {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']};
            }}
            QCheckBox {{
                color: {COLORS['text_secondary']};
                border: none;
            }}
        """)
        
        form_config = QFormLayout(self.config_content)
        form_config.setSpacing(2)
        form_config.setContentsMargins(8, 4, 8, 4)

        # Model Selection
        self.combo_model = QComboBox()
        self.combo_model.addItem("zipformer-30M-rnnt-6000h (⭐)", "zipformer-30m-rnnt-6000h")
        self.combo_model.addItem("sherpa-onnx-zipformer-vi-2025-04-20", "sherpa-onnx-zipformer-vi-2025-04-20")
        self.combo_model.addItem("ROVER - Voting (chính xác hơn)", "rover-voting")
        self.combo_model.currentIndexChanged.connect(self._reset_analyzer)
        form_config.addRow("Model:", self.combo_model)
        
        # CPU Threads
        from common import ALLOWED_THREADS
        self.slider_threads = QSlider(Qt.Orientation.Horizontal)
        self.slider_threads.setRange(1, ALLOWED_THREADS)
        self.slider_threads.setValue(min(4, ALLOWED_THREADS))
        self.slider_threads.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border']};
                height: 6px;
                background: {COLORS['bg_dark']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        self.label_threads = QLabel("4")
        self.label_threads.setStyleSheet(f"color: {COLORS['text_primary']}; min-width: 20px; padding-bottom: 4px;")
        self.slider_threads.valueChanged.connect(self.on_threads_changed)
        
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(self.slider_threads)
        threads_layout.addWidget(self.label_threads)
        form_config.addRow("Số luồng CPU:", threads_layout)
        
        # Punctuation Confidence Slider
        self.slider_punct_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_punct_conf.setRange(1, 10)
        self.slider_punct_conf.setValue(7)
        self.slider_punct_conf.setEnabled(True)
        self.slider_punct_conf.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border']};
                height: 6px;
                background: {COLORS['bg_dark']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        self.label_punct_conf = QLabel("Cân bằng")
        self.label_punct_conf.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 30px; padding-bottom: 4px;")
        self.slider_punct_conf.valueChanged.connect(self.on_punct_conf_changed)
        
        punct_conf_layout = QHBoxLayout()
        punct_conf_layout.addWidget(self.slider_punct_conf)
        punct_conf_layout.addWidget(self.label_punct_conf)
        form_config.addRow("Mức độ thêm dấu:", punct_conf_layout)
        
        # Casing Confidence Slider
        self.slider_case_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_case_conf.setRange(1, 10)
        self.slider_case_conf.setValue(3)
        self.slider_case_conf.setEnabled(True)
        self.slider_case_conf.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border']};
                height: 6px;
                background: {COLORS['bg_dark']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        self.label_case_conf = QLabel("Ít")
        self.label_case_conf.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 30px; padding-bottom: 4px;")
        self.slider_case_conf.valueChanged.connect(self.on_case_conf_changed)
        
        case_conf_layout = QHBoxLayout()
        case_conf_layout.addWidget(self.slider_case_conf)
        case_conf_layout.addWidget(self.label_case_conf)
        form_config.addRow("Mức độ tự viết hoa:", case_conf_layout)
        

        # Speaker Diarization
        self.check_speaker_diarization = QCheckBox("Phân tách Người nói (Speaker diarization - Chạy lâu)")
        self.check_speaker_diarization.setChecked(False)
        self.check_speaker_diarization.setEnabled(DIARIZATION_AVAILABLE)
        self.check_speaker_diarization.setToolTip("Tự động phân biệt các Người nói khác nhau trong file âm thanh")
        self.check_speaker_diarization.stateChanged.connect(self.on_speaker_diarization_changed)
        form_config.addRow(self.check_speaker_diarization)
        
        # Number of speakers
        self.spin_num_speakers = QComboBox()
        self.spin_num_speakers.addItems(["Không rõ (tự động)", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"])
        self.spin_num_speakers.setCurrentIndex(0)
        self.spin_num_speakers.setEnabled(False)
        self.spin_num_speakers.setToolTip("Số lượng Người nói dự kiến trong file âm thanh")
        
        self.check_show_speaker_labels = QCheckBox("Hiện phân tách Người nói")
        self.check_show_speaker_labels.setChecked(True)
        self.check_show_speaker_labels.setEnabled(False)
        self.check_show_speaker_labels.setToolTip("Hiển thị các dòng phân tách Người nói trong kết quả")
        self.check_show_speaker_labels.stateChanged.connect(self.on_show_speaker_labels_changed)
        
        speaker_settings_layout = QHBoxLayout()
        speaker_settings_layout.addWidget(self.spin_num_speakers)
        speaker_settings_layout.addSpacing(20)
        speaker_settings_layout.addWidget(self.check_show_speaker_labels)
        speaker_settings_layout.addStretch()
        
        form_config.addRow("  └─ Số Người nói:", speaker_settings_layout)
        
        # Diarization Threshold Slider
        self.slider_diarization_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_diarization_threshold.setRange(10, 150)  # 0.10 to 1.50, step 0.01
        self.slider_diarization_threshold.setValue(70)  # Default 0.70
        self.slider_diarization_threshold.setEnabled(False)
        self.slider_diarization_threshold.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border']};
                height: 6px;
                background: {COLORS['bg_dark']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        self.label_diarization_threshold = QLabel("0.70")  # Default display
        self.label_diarization_threshold.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 30px; padding-bottom: 4px;")
        self.slider_diarization_threshold.valueChanged.connect(self.on_diarization_threshold_changed)
        
        diarization_thresh_layout = QHBoxLayout()
        diarization_thresh_layout.addWidget(self.slider_diarization_threshold)
        diarization_thresh_layout.addWidget(self.label_diarization_threshold)
        
        form_config.addRow("  └─ Ngưỡng phân biệt:", diarization_thresh_layout)
        self.label_diarization_threshold_tip = QLabel("Cao (1.00) = Gộp nhiều | Thấp (0.30) = Tách kỹ | ONNX: 0.80-1.20")
        self.label_diarization_threshold_tip.setStyleSheet("font-size: 9px; color: #888; font-style: italic; margin-left: 4px;")
        form_config.addRow("", self.label_diarization_threshold_tip)
        
        # Model embedding extraction + Rerun button
        embedding_layout = QHBoxLayout()
        
        self.combo_speaker_model = QComboBox()
        # ⭐ = Recommended (Altunenes ONNX - Fast, no HF token needed)
        self.combo_speaker_model.addItem("Pyannote Community-1 Altunenes ONNX (Nhanh) ⭐", "community1_onnx")
        self.combo_speaker_model.addItem("Pyannote Community-1 (Pytorch) (Chậm, chính xác)", "community1")
        self.combo_speaker_model.addItem("Nvidia Nemo Titanet small (Dự phòng)", "titanet_small")
        self.combo_speaker_model.setCurrentIndex(0)
        self.combo_speaker_model.setEnabled(False)
        self.combo_speaker_model.setToolTip(
            "⭐ Altunenes ONNX: Khuyến nghị - Nhanh, không cần HF Token\n"
            "   Pyannote Pytorch: Chính xác cao nhất, cần HF Token\n"
            "   Titanet: Dự phòng - Nhẹ, tiếng Anh"
        )
        self.combo_speaker_model.currentIndexChanged.connect(self.on_speaker_model_changed)
        
        self.btn_rerun_diarization = QPushButton("Phân đoạn lại")
        self.btn_rerun_diarization.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                font-size: 11px;
                padding: 4px 8px;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_secondary']};
                border-color: {COLORS['border']};
            }}
        """)
        self.btn_rerun_diarization.setEnabled(False)
        self.btn_rerun_diarization.clicked.connect(self.rerun_speaker_diarization)
        
        embedding_layout.addWidget(self.combo_speaker_model, stretch=1)
        embedding_layout.addWidget(self.btn_rerun_diarization)
        
        form_config.addRow("  └─ Model embedding:", embedding_layout)
        
        # Save RAM Option
        self.check_save_ram = QCheckBox("Tiết kiệm RAM (unload model sau mỗi bước)")
        self.check_save_ram.setChecked(False)
        self.check_save_ram.setToolTip(
            "Khi bật, các model sẽ được giải phóng khỏi bộ nhớ sau mỗi bước xử lý.\n"
            "• Bật: Giảm ~30-50% RAM, nhưng chậm hơn khi xử lý file tiếp theo\n"
            "• Tắt: Giữ model trong RAM, xử lý file tiếp theo nhanh hơn"
        )
        form_config.addRow(self.check_save_ram)
        
        # Auto analyze quality option
        self.chk_auto_analyze = QCheckBox("Tự động phân tích chất lượng khi chọn file")
        self.chk_auto_analyze.setChecked(True)
        self.chk_auto_analyze.setToolTip("Tự động chạy DNSMOS và ASR-Proxy khi thêm file mới")
        self.chk_auto_analyze.stateChanged.connect(self.on_auto_analyze_changed)
        form_config.addRow(self.chk_auto_analyze)
        
        config_layout.addWidget(self.config_content)
        layout.addWidget(self.config_container)

        # 2. File Selection + Action Area
        file_action_layout = QHBoxLayout()
        file_action_layout.setSpacing(8)
        
        self.drop_label = DragDropLabel(self)
        self.drop_label.fileDropped.connect(self.set_file)
        self.drop_label.clicked.connect(self.open_file_dialog)
        file_action_layout.addWidget(self.drop_label, stretch=80)
        
        # Layout cho 3 nút bên phải
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.btn_process = QPushButton("🚀 Xử lý")
        self.btn_process.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
                border: none;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_process.clicked.connect(self.start_transcription)
        self.btn_process.setEnabled(False)
        buttons_layout.addWidget(self.btn_process, stretch=1)
        
        # Nút Lưu kết quả JSON
        self.btn_save_json = QPushButton("💾 Lưu JSON")
        self.btn_save_json.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
                border: none;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_save_json.clicked.connect(self.save_asr_json)
        self.btn_save_json.setEnabled(False)
        self.btn_save_json.setToolTip("Lưu kết quả ASR vào file JSON")
        buttons_layout.addWidget(self.btn_save_json, stretch=1)
        
        # Nút Copy text
        self.btn_copy_text = QPushButton("📋 Copy text")
        self.btn_copy_text.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
                border: 1px solid {COLORS['border']};
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_secondary']};
                border-color: {COLORS['border']};
            }}
        """)
        self.btn_copy_text.clicked.connect(self.copy_text_to_clipboard)
        self.btn_copy_text.setEnabled(False)
        self.btn_copy_text.setToolTip("Sao chép toàn bộ nội dung văn bản")
        buttons_layout.addWidget(self.btn_copy_text, stretch=1)
        
        file_action_layout.addLayout(buttons_layout, stretch=20)
        
        layout.addLayout(file_action_layout)

        # 3. Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Đang chờ tập tin...")
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                color: {COLORS['text_primary']};
                background-color: {COLORS['bg_card']};
                font-size: 12px;
                min-height: 16px;
                max-height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # 4. Output Area with Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['bg_input']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['accent']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {COLORS['border']};
                color: {COLORS['text_primary']};
            }}
        """)
        
        # Tab 1: Nội dung
        self.text_output = ClickableTextEdit()
        self.text_output.setPlaceholderText("Kết quả văn bản sau khi bổ sung dấu sẽ hiển thị ở đây...")
        self.text_output.sentenceClicked.connect(self.seek_to_sentence)
        self.text_output.speakerLabelClicked.connect(self.on_speaker_label_clicked)
        self.text_output.splitSpeakerRequested.connect(self.on_split_speaker_requested)
        self.text_output.mergeSpeakerRequested.connect(self.on_merge_speaker_requested)
        self.tab_widget.addTab(self.text_output, "📝 Nội dung")
        
        # Tab 2: Người nói
        self.text_speaker_raw_output = QTextEdit()
        self.text_speaker_raw_output.setReadOnly(True)
        self.text_speaker_raw_output.setPlaceholderText("Kết quả phân tách Người nói thô sẽ hiển thị ở đây...")
        self.text_speaker_raw_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                font-size: 13px;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                line-height: 1.4;
                font-family: Consolas, monospace;
            }}
        """)
        self.tab_widget.addTab(self.text_speaker_raw_output, "👥 Người nói")
        
        layout.addWidget(self.tab_widget)
        
        # 6. Search Widget in Tab Corner
        self.search_widget = SearchWidget()
        self.search_widget.searchRequested.connect(self.perform_search)
        self.search_widget.nextRequested.connect(lambda: self.navigate_search(1))
        self.search_widget.prevRequested.connect(lambda: self.navigate_search(-1))
        self.search_widget.closed.connect(self.clear_search)
        self.search_widget.set_input_width(250)  # Make search input wider for long words
        self.tab_widget.setCornerWidget(self.search_widget, Qt.Corner.TopRightCorner)
        
        # 5. Player Controls
        self.player_container = QWidget()
        self.player_container.setStyleSheet(f"background-color: {COLORS['bg_card']}; border-radius: 4px;")
        self.player_layout = QHBoxLayout(self.player_container)
        self.player_layout.setContentsMargins(6, 0, 6, 0)
        self.player_layout.setSpacing(4)
        
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                border-radius: 3px;
                padding: 2px;
                min-width: 20px;
                min-height: 20px;
                max-width: 20px;
                max-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.player_layout.addWidget(self.btn_play)
        
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 0)
        self.slider_seek.sliderMoved.connect(self.set_position)
        self.slider_seek.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {COLORS['border']};
                height: 4px;
                background: {COLORS['bg_dark']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }}
        """)
        self.player_layout.addWidget(self.slider_seek)
        
        self.label_time = QLabel("00:00 / 00:00")
        self.label_time.setStyleSheet(f"color: {COLORS['text_primary']}; font-family: monospace; font-size: 11px;")
        self.player_layout.addWidget(self.label_time)
        
        layout.addWidget(self.player_container)
        self.player_container.setVisible(False)

    # Configuration Methods
    def toggle_config(self):
        is_visible = self.config_content.isVisible()
        self.config_content.setVisible(not is_visible)
        if not is_visible:
            self.btn_toggle_config.setArrowType(Qt.ArrowType.DownArrow)
        else:
            self.btn_toggle_config.setArrowType(Qt.ArrowType.RightArrow)

    def on_header_click(self, event):
        self.toggle_config()

    def show_about_dialog(self):
        if self.main_window:
            self.main_window.show_about_dialog()

    def on_threads_changed(self, value):
        self.label_threads.setText(str(value))

    def on_diarization_threshold_changed(self, value):
        threshold = value / 100.0  # 2 decimal places: 50 -> 0.50
        self.label_diarization_threshold.setText(f"{threshold:.2f}")  # Show 0.50, 0.70...
        
        if threshold >= 1.0:
            tip = "Rất cao (Gộp nhiều) - dùng cho ONNX"
        elif threshold >= 0.7:
            tip = "Cao (Gộp nhiều)"
        elif threshold <= 0.4:
            tip = "Thấp (Tách kỹ)"
        else:
            tip = "Trung bình"
        self.label_diarization_threshold.setToolTip(f"Ngưỡng hiện tại: {threshold} - {tip}")

    def on_speaker_model_changed(self, index):
        """Update default threshold when speaker model changes"""
        from core.speaker_diarization import SpeakerDiarizer
        
        model_id = self.combo_speaker_model.currentData()
        default_threshold = SpeakerDiarizer.get_default_threshold(model_id)
        
        # Convert threshold to slider value (multiply by 100 for 2 decimal places)
        slider_value = int(default_threshold * 100)
        self.slider_diarization_threshold.setValue(slider_value)
        
        print(f"[Config] Model changed to {model_id}, threshold set to {default_threshold}")

    def on_speaker_diarization_changed(self, state):
        is_checked = (state == Qt.CheckState.Checked.value or state == 2)
        self.spin_num_speakers.setEnabled(is_checked)
        self.combo_speaker_model.setEnabled(is_checked)
        self.check_show_speaker_labels.setEnabled(is_checked)
        self.slider_diarization_threshold.setEnabled(is_checked)
        self.btn_rerun_diarization.setEnabled(is_checked and bool(self.segments))

    def on_show_speaker_labels_changed(self, state):
        self.render_text_content(immediate=True)
        self._do_render()

    def on_punct_conf_changed(self, value):
        labels = {1: "Rất ít", 3: "Ít", 5: "Vừa", 7: "Nhiều", 10: "Rất nhiều"}
        label = labels.get(value, str(value))
        self.label_punct_conf.setText(label)

    def on_case_conf_changed(self, value):
        labels = {1: "Rất ít", 3: "Ít", 5: "Vừa", 7: "Nhiều", 10: "Rất nhiều"}
        label = labels.get(value, str(value))
        self.label_case_conf.setText(label)

    def _get_playback_path(self, file_path):
        """Convert non-WAV files to temp WAV for accurate QMediaPlayer seeking.
        MP3/M4A seeking is frame-based and inaccurate. WAV seeking is sample-exact."""
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.wav':
            return file_path  # WAV: use directly
            
        if file_path in self._playback_cache:
            cached_path = self._playback_cache[file_path]
            if os.path.exists(cached_path):
                print(f"[_get_playback_path] Using cached WAV for {file_ext}: {cached_path}")
                return cached_path
        
        try:
            from pydub import AudioSegment
            import tempfile
            
            audio = AudioSegment.from_file(file_path)
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, prefix='asr_playback_')
            temp_path = temp_file.name
            temp_file.close()
            
            audio.export(temp_path, format='wav')
            self._playback_cache[file_path] = temp_path
            print(f"[_get_playback_path] Converted {file_ext} -> WAV for accurate seeking: {temp_path}")
            return temp_path
        except Exception as e:
            print(f"[_get_playback_path] Failed to convert, using original: {e}")
            return file_path

    def cleanup_temp_files(self):
        """Dừng nhạc, nhả file lock và xoá toàn bộ file temp cũ"""
        self.player.stop()
        self.player.setSource(QUrl())
        
        retained_cache = {}
        for orig_path, tmp_path in self._playback_cache.items():
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    print(f"[Cleanup] Deleted temporary file: {tmp_path}")
                except Exception as e:
                    print(f"[Cleanup] Failed to delete (keeping in queue): {tmp_path} -> {e}")
                    retained_cache[orig_path] = tmp_path
        self._playback_cache = retained_cache

    def set_file(self, file_path):
        self.cleanup_temp_files()
        
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.wma', '.ogg', '.opus', 
                            '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            QMessageBox.warning(self.window(), "Định dạng không hỗ trợ", 
                f"File '{os.path.basename(file_path)}' không được hỗ trợ.")
            return
        
        self.selected_file = file_path
        self.drop_label.setFileText(os.path.basename(file_path))
        self.btn_process.setEnabled(True)
        self.player_container.setVisible(False)
        self.text_output.clear()
        self.btn_rerun_diarization.setEnabled(False)
        self.btn_copy_text.setEnabled(False)
        self.loaded_from_json = False
        
        # Xóa dữ liệu tên Người nói cũ khi chọn file mới
        self.speaker_name_mapping = {}
        self.block_speaker_names = {}
        self.custom_speaker_names = set()
        
        # Check for existing ASR JSON
        json_path = os.path.splitext(file_path)[0] + '.asr.json'
        if os.path.exists(json_path):
            self.btn_play.setEnabled(False) # Khoá Play
            if hasattr(self.text_output, 'set_clickable'):
                self.text_output.set_clickable(False) # Khoá click tua
            
            # Hiển thị animation loading
            self.current_progress_text = "Vui lòng đợi load thông tin từ JSON..."
            self.progress_bar.setFormat("Vui lòng đợi load thông tin từ JSON...")
            self.start_spinner()
            self.progress_bar.setValue(0)
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents() # Ép UI render progress bar text thay đổi
            
            thread = FastJSONLoadThread(json_path)
            self._json_load_thread = thread
            thread.progress_updated.connect(lambda p, m, t=thread: self._on_json_load_progress(p, m, t))
            thread.finished_loading.connect(lambda t=thread: self._on_json_load_finished(file_path, t))
            thread.error_occurred.connect(lambda err, t=thread: self._on_json_load_error(err, t))
            thread.start()
            
            # Kích hoạt convert WAV ngầm luôn
            self._bg_audio_thread = BackgroundAudioConvertThread(file_path)
            self._bg_audio_thread.finished_converting.connect(lambda p, t=self._bg_audio_thread: self._on_audio_converted(p, file_path, t))
            self._bg_audio_thread.start()
            return

        if self.segments:
             self.btn_rerun_diarization.setEnabled(self.check_speaker_diarization.isChecked())
        
        # Auto analyze audio quality
        if self.chk_auto_analyze.isChecked():
            self.analyze_file_quality()

    def _on_audio_converted(self, temp_path, original_path, thread=None):
        if thread and thread != getattr(self, '_bg_audio_thread', None):
            thread.deleteLater()
            return
            
        if temp_path != original_path:
            self._playback_cache[original_path] = temp_path
            
        url = QUrl.fromLocalFile(os.path.abspath(temp_path))
        self.player.setSource(url)
        self.player_container.setVisible(True)
        self.btn_play.setEnabled(True)
        if hasattr(self.text_output, 'set_clickable'):
            self.text_output.set_clickable(True) # Mở khoá click tua
            
        if getattr(self, '_json_load_thread', None) is None:
            self.stop_spinner()
            self.progress_bar.setFormat("✓ Đã tải dữ liệu và âm thanh hoàn chỉnh")
            self.progress_bar.setValue(100)
            
        thread.deleteLater()
        if getattr(self, '_bg_audio_thread', None) == thread:
            self._bg_audio_thread = None

    def _on_json_load_progress(self, percentage, msg, thread=None):
        if thread and thread != getattr(self, '_json_load_thread', None):
            return
        self.current_progress_text = msg
        # Ép format ngay trên progress bar thay vì chỉ chờ spinner
        self.progress_bar.setFormat(f"⠋ {msg}") 
        self.progress_bar.setValue(percentage)

    def _on_json_load_error(self, err_msg, thread=None):
        if thread and thread != getattr(self, '_json_load_thread', None):
            thread.deleteLater()
            return
        self.stop_spinner()
        p_win = None if self.window().isMinimized() or self.window().isHidden() else self.window()
        QMessageBox.critical(p_win, "Lỗi load JSON", f"Không thể đọc file JSON:\n{err_msg}")
        self.progress_bar.setFormat("Lỗi load JSON")
        
    def _on_json_load_finished(self, file_path, thread=None):
        if thread is None:
            thread = getattr(self, '_json_load_thread', None)
            
        if not thread:
            return
            
        if thread != getattr(self, '_json_load_thread', None):
            thread.deleteLater()
            return

        self.segments = thread.result_segments
        if thread.result_speaker_mapping:
            self.speaker_name_mapping = thread.result_speaker_mapping
        self.has_speaker_diarization = thread.result_has_speakers
        self.current_highlight_index = -1
        self._last_rendered_highlight = -1
        
        # Dọn dẹp an toàn Thread bằng deleteLater()
        thread.deleteLater()
        if getattr(self, '_json_load_thread', None) == thread:
            self._json_load_thread = None
        
        self.loaded_from_json = True
        self.json_saved = True
            
        self.btn_save_json.setEnabled(True)
        self.btn_copy_text.setEnabled(True)
        
        # Bắt đầu incremental render HTML
        self._start_incremental_render()

    def _start_incremental_render(self):
        self.text_output.clear()
        
        chunks = []
        if getattr(self, 'has_speaker_diarization', False):
            chunks = self._build_speaker_view_chunks()
            if chunks:
                chunks[0] = f"<p style='font-size:14px; line-height:1.6; color:{COLORS['text_dark']}; margin:0;'>" + chunks[0]
                chunks[-1] += "</p>"
        else:
            chunks = self._build_normal_view_chunks()
            if chunks:
                chunks[0] = f"<p style='font-size:14px; line-height:1.3; color:{COLORS['text_dark']};'>" + chunks[0]
                chunks[-1] += "</p>"
            
        self._render_chunks = chunks
        self._render_chunk_idx = 0
        if not hasattr(self, '_incremental_timer'):
            self._incremental_timer = QTimer(self)
            self._incremental_timer.timeout.connect(self._render_next_chunk)
        self._incremental_timer.start(10) # 10ms mỗi batch
    
    def _render_next_chunk(self):
        if not hasattr(self, '_render_chunks') or self._render_chunk_idx >= len(self._render_chunks):
            self._incremental_timer.stop()
            if getattr(self, '_bg_audio_thread', None) is None:
                self.stop_spinner()
                self.progress_bar.setFormat("✓ Đã tải từ JSON")
                self.progress_bar.setValue(100)
            else:
                self.progress_bar.setFormat("✓ Đã tải JSON. Đang chuẩn bị âm thanh chất lượng cao...")
            return
            
        cursor = self.text_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Batch size để không quá chậm cũng không quá lag
        batch_size = max(5, len(self._render_chunks) // 20)
        
        html_block = ""
        for _ in range(batch_size):
            if self._render_chunk_idx >= len(self._render_chunks):
                break
            html_block += self._render_chunks[self._render_chunk_idx]
            self._render_chunk_idx += 1
            
        cursor.insertHtml(html_block)
        
        # Cập nhật tiến trình (từ 75% -> 100%)
        prog = 75 + int(25 * (self._render_chunk_idx / len(self._render_chunks)))
        self.progress_bar.setValue(prog)

    def _build_normal_view_chunks(self):
        chunks = []
        para_boundaries = set()
        if self.paragraphs:
            sent_idx = 0
            total_para_sentences = 0
            for para in self.paragraphs:
                if sent_idx > 0:
                    para_boundaries.add(sent_idx)
                num_sentences = len(para.get('sentences', []))
                sent_idx += num_sentences
                total_para_sentences += num_sentences
            
            if total_para_sentences != len(self.segments):
                para_boundaries = set()

        for i, seg in enumerate(self.segments):
            chunk_html = f"<a name='seg_{i}'></a>"
            if i in para_boundaries:
                chunk_html += "<br>"
            
            partials = seg.get('partials', [])
            if partials:
                full_text = seg.get('text', '')
                search_pos = 0
                for chunk_idx, partial in enumerate(partials):
                    chunk_text = partial.get('text', '')
                    if not chunk_text:
                        continue
                    chunk_start_pos = full_text.find(chunk_text, search_pos)
                    if chunk_start_pos == -1:
                        chunk_start_pos = search_pos
                    anchor_id = 1000000 + i * 1000 + chunk_idx
                    chunk_html += self._render_text_with_search_highlight(
                        chunk_text, anchor_id, i, chunk_start_pos
                    ) + " "
                    search_pos = chunk_start_pos + len(chunk_text)
            else:
                text = seg.get('text', '')
                anchor_id = 1000000 + i * 1000
                chunk_html += self._render_text_with_search_highlight(
                    text, anchor_id, i, 0
                ) + " "
                
            chunks.append(chunk_html)
            
        return chunks

    def _build_speaker_view_chunks(self):
        chunks = []
        self._block_render_count = 0
        self.merged_speaker_blocks = []
        
        segments_with_idx = [{**seg, 'index': i} for i, seg in enumerate(self.segments)]
        merged_segments = self._merge_speaker_segments(segments_with_idx, max_gap_sec=2.0)
        
        current_speaker = None
        current_blocks = []
        speaker_block_count = 0
        
        def process_blocks():
            nonlocal current_speaker, current_blocks, speaker_block_count
            if not current_blocks:
                return
            block_render_count = getattr(self, '_block_render_count', 0) + 1
            self._block_render_count = block_render_count
            
            block_info = {
                'speaker': current_speaker,
                'sentences': [],
                'start': current_blocks[0].get('start', 0),
                'end': current_blocks[-1].get('end', 0)
            }
            for block in current_blocks:
                block_info['sentences'].extend(block.get('sentences', []))
            self.merged_speaker_blocks.append(block_info)
            block_idx = len(self.merged_speaker_blocks) - 1
            
            speaker_id = current_blocks[0].get('speaker_id', 0) if current_blocks else 0
            speaker_id_str = str(speaker_id)
            
            if speaker_id_str in self.speaker_name_mapping:
                display_name = self.speaker_name_mapping[speaker_id_str]
            else:
                display_name = current_speaker
            
            html_content = f"<div style='margin: 16px 0; padding: 10px; background-color: #f8f9fa; border-left: 3px solid {COLORS['accent']}; border-radius: 0 4px 4px 0;'>"
            
            if self.check_show_speaker_labels.isChecked():
                anchor_style = f"text-decoration:none; color:{COLORS['accent']}; cursor:pointer;"
                html_content += f"<a href='spk_{speaker_id_str}_{block_idx}' style='{anchor_style}'><div style='font-weight:bold; margin-bottom:8px; font-size:13px;'>{display_name}:</div></a>"
            
            html_content += f"<div style='margin-left:4px; text-align: justify; line-height:1.6; color:{COLORS['text_dark']};'>"
            
            for block in current_blocks:
                for sent in block.get('sentences', []):
                    sent_idx = sent.get('index', 0)
                    text = sent.get('text', '').strip()
                    if not text:
                        continue
                    
                    html_content += f"<a name='seg_{sent_idx}'></a>"
                    
                    seg_data = self.segments[sent_idx] if sent_idx < len(self.segments) else None
                    partials = seg_data.get('partials', []) if seg_data else []
                    
                    if partials:
                        full_text = seg_data.get('text', '') if seg_data else ''
                        search_pos = 0
                        for chunk_idx, partial in enumerate(partials):
                            chunk_text = partial.get('text', '')
                            if not chunk_text:
                                continue
                            chunk_start_pos = full_text.find(chunk_text, search_pos)
                            if chunk_start_pos == -1:
                                chunk_start_pos = search_pos
                            anchor_id = 1000000 + sent_idx * 1000 + chunk_idx
                            html_content += self._render_text_with_search_highlight(
                                chunk_text, anchor_id, sent_idx, chunk_start_pos
                            ) + " "
                            search_pos = chunk_start_pos + len(chunk_text)
                    else:
                        anchor_id = 1000000 + sent_idx * 1000
                        html_content += self._render_text_with_search_highlight(
                            text, anchor_id, sent_idx, 0
                        ) + " "
            
            html_content += "</div></div>"
            chunks.append(html_content)

        for seg in merged_segments:
            speaker = seg.get('speaker', '') or 'Chưa gán'
            if speaker != current_speaker:
                speaker_block_count += 1
                process_blocks()
                current_speaker = speaker
                current_blocks = []
            current_blocks.append(seg)
        
        process_blocks()
        return chunks

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file âm thanh/video", "", "Media Files (*.mp3 *.wav *.m4a *.ogg *.wma *.flac *.aac *.opus *.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv);;All Files (*)")
        if file_path:
            self.set_file(file_path)

    def get_config(self):
        slider_val = self.slider_punct_conf.value()
        # Công thức: slider cao → confidence âm → model mạnh mẽ hơn (thêm nhiều dấu hơn)
        # confidence > 0: cộng vào $KEEP → bảo thủ. confidence < 0: trừ $KEEP → mạnh mẽ
        # slider=2: +0.35, slider=5: ~0, slider=7: -0.37, slider=10: -0.8
        confidence = 0.5 - (slider_val - 1) * (1.3 / 9)
        
        case_val = self.slider_case_conf.value()
        # Slider=1 -> -1.5 (rút hoàn toàn viết hoa)
        # Slider=10 -> 0.5 (khuyến khích viết hoa)
        case_confidence = -1.5 + (case_val - 1) * (2.0 / 9)
        
        # Chỉ bypass khi thanh dấu câu = 1 (không muốn thêm dấu)
        # Nếu case=1 nhưng punct>1 vẫn chạy model (case_confidence âm sẽ suppress viết hoa)
        bypass_restorer = (slider_val == 1)
        
        return {
            "cpu_threads": self.slider_threads.value(),
            "restore_punctuation": True,
            "bypass_restorer": bypass_restorer,
            "punctuation_confidence": confidence,
            "case_confidence": case_confidence,
            "speaker_diarization": self.check_speaker_diarization.isChecked() and DIARIZATION_AVAILABLE,
            "num_speakers": -1 if self.spin_num_speakers.currentIndex() == 0 else int(self.spin_num_speakers.currentText()),
            "speaker_model": self.combo_speaker_model.currentData(),
            "save_ram": self.check_save_ram.isChecked()
        }

    def start_transcription(self):
        if not self.selected_file:
            return
        
        # Xóa dữ liệu tên Người nói cũ khi xử lý ASR mới (có thể sinh ra speaker ID khác)
        self.speaker_name_mapping = {}
        self.block_speaker_names = {}
        self.custom_speaker_names = set()

        # Reset saved flag for new transcription
        self.json_saved = False

        # Kiểm tra xem file có JSON tồn tại trên đĩa không
        json_path = os.path.splitext(self.selected_file)[0] + '.asr.json'
        has_json_file = os.path.exists(json_path)
        
        # Nếu file có JSON, hỏi user muốn làm gì
        self._pending_json_segments = None  # Reset
        if has_json_file:
            msg_box = QMessageBox(self.window())
            msg_box.setWindowTitle("Xử lý lại")
            msg_box.setText("File này đã có dữ liệu ASR.\n\nBạn muốn xử lý như thế nào?")
            
            btn_full = msg_box.addButton("Làm lại từ đầu", QMessageBox.ButtonRole.DestructiveRole)
            btn_text_only = msg_box.addButton("ASR lại text\n(giữ phân tách Người nói)", QMessageBox.ButtonRole.AcceptRole)
            btn_cancel = msg_box.addButton("Hủy", QMessageBox.ButtonRole.RejectRole)
            
            msg_box.setDefaultButton(btn_cancel)
            msg_box.exec()
            
            clicked_btn = msg_box.clickedButton()
            
            if clicked_btn == btn_cancel:
                return
            elif clicked_btn == btn_text_only:
                # Load JSON từ đĩa để lấy segments cũ
                if self._load_asr_json(json_path):
                    self._pending_json_segments = [seg.copy() for seg in self.segments]
                    self._pending_text_only_mode = True
                else:
                    # Nếu load lỗi, xử lý lại từ đầu
                    self._pending_text_only_mode = False
            else:
                # Làm lại từ đầu - xóa dữ liệu cũ
                self._pending_text_only_mode = False

        # Nếu đang phân tích âm thanh, dừng lại
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            self.stop_spinner()

        model_path = self.default_model_path
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))

        model_folder_name = self.combo_model.currentData()

        # ROVER mode: dùng thư mục models làm base, pipeline sẽ tự load 2 model
        from core.asr_engine import ROVER_MODEL_ID, ROVER_MODEL_IDS
        is_rover = (model_folder_name == ROVER_MODEL_ID)

        if is_rover:
            models_dir = os.path.join(application_path, "models")
            # Kiểm tra cả 2 model tồn tại
            for mid in ROVER_MODEL_IDS:
                p = os.path.join(models_dir, mid)
                if not os.path.isdir(p):
                    show_missing_model_dialog(self, mid, p)
                    return
            model_path = models_dir  # Pipeline sẽ dùng thư mục này để tìm cả 2 model
        else:
            model_path = os.path.join(application_path, "models", model_folder_name)
            if not os.path.exists(model_path):
                show_missing_model_dialog(self, model_folder_name, model_path)
                return

        self.toggle_inputs(False)
        self.text_output.clear()
        self.progress_bar.setValue(0)
        self.player_container.setVisible(False)
        
        self.current_progress_text = "Đang khởi tạo..."
        self.start_spinner()

        self.btn_process.setEnabled(False)
        self.btn_save_json.setEnabled(False)
        self.btn_copy_text.setEnabled(False)
        self.drop_label.setEnabled(False)
        self.config_content.setEnabled(False)
        self.segments = []
        self.paragraphs = []
        self.search_matches = []
        self.current_highlight_index = -1
        self.loaded_from_json = False
        self.text_output.clear()
        
        # Dùng get_config() để tránh duplicate logic và dễ bị desync
        config = self.get_config()
        config["rover_mode"] = is_rover

        # Override/thêm các field cần thiết
        config["diarization_threshold"] = self.slider_diarization_threshold.value() / 100.0  # 2 decimal places
        
        # num_speakers cần xử lý riêng vì là text "tự động" hoặc số
        num_speakers_text = self.spin_num_speakers.currentText()
        if "tự động" in num_speakers_text:
            config["num_speakers"] = -1
        else:
            try:
                config["num_speakers"] = int(num_speakers_text)
            except:
                config["num_speakers"] = -1
        
        # Nếu đang ở mode text-only (ASR lại text giữ nguyên speaker), 
        # tắt speaker diarization và lưu segments cũ để map lại
        if getattr(self, '_pending_text_only_mode', False) and self._pending_json_segments:
            config["speaker_diarization"] = False
            config["_text_only_mode"] = True
            config["_original_segments"] = self._pending_json_segments
        else:
            config["_text_only_mode"] = False
        
        # Sử dụng WAV cache nếu có để xử lý nhanh hơn và nhất quán
        audio_file_for_processing = self.selected_file
        if self.selected_file in self._playback_cache:
            cached_wav = self._playback_cache[self.selected_file]
            if os.path.exists(cached_wav):
                audio_file_for_processing = cached_wav
                print(f"[Process] Using cached WAV: {cached_wav}")

        self.transcriber = TranscriberThread(audio_file_for_processing, model_path, config)
        self.transcriber.progress.connect(self.update_progress)
        self.transcriber.finished.connect(self.on_finished)
        self.transcriber.error.connect(self.on_error)
        self.transcriber.start()

    def update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
        spinner = self.spinner_chars[self.spinner_idx]
        self.progress_bar.setFormat(f"{spinner} {self.current_progress_text}")

    def start_spinner(self):
        self.spinner_idx = 0
        if not self.spinner_timer.isActive():
            self.spinner_timer.start(80)

    def stop_spinner(self):
        if self.spinner_timer.isActive():
            self.spinner_timer.stop()

    def update_progress(self, msg):
        self.start_spinner()
        
        if msg.startswith("PHASE:"):
            try:
                parts = msg.split("|", 2)
                if len(parts) >= 3:
                    phase_name = parts[0].replace("PHASE:", "")
                    display_text = parts[1]
                    percentage = int(parts[2])
                    self.progress_bar.setValue(percentage)
                    # Thêm cảnh báo cho phase Diarization
                    if phase_name == "Diarization":
                        self.current_progress_text = f"{display_text} ({percentage}%) - chạy lâu, có thể đứng giao diện một chút, vui lòng chờ"
                    else:
                        self.current_progress_text = f"{display_text} ({percentage}%)"
            except:
                pass
        elif msg.startswith("Progress:"):
            try:
                match = re.search(r'Progress:\s*(\d+)%', msg)
                if match:
                    val = int(match.group(1))
                    self.progress_bar.setValue(val)
                    self.current_progress_text = f"Đang xử lý... ({val}%)"
            except:
                pass
        else:
            self.current_progress_text = f"{msg} 0%"

    def _map_speakers_to_new_segments(self, new_segments, original_segments):
        """Map speaker từ segments cũ sang segments mới dựa trên thớigian"""
        if not original_segments or not new_segments:
            return new_segments
        
        # Tạo danh sách các speaker blocks từ segments cũ
        speaker_blocks = []
        current_speaker = None
        block_start = 0
        block_end = 0
        
        for seg in original_segments:
            speaker = seg.get('speaker', '')
            speaker_id = seg.get('speaker_id', 0)
            seg_start = seg.get('start', seg.get('start_time', 0))
            seg_end = seg.get('end', seg_start + 1.0)
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    speaker_blocks.append({
                        'speaker': current_speaker,
                        'speaker_id': speaker_blocks[-1]['speaker_id'] if speaker_blocks else speaker_id,
                        'start': block_start,
                        'end': block_end
                    })
                block_start = seg_start
                current_speaker = speaker
            block_end = seg_end
        
        # Thêm block cuối
        if current_speaker is not None and original_segments:
            last_seg = original_segments[-1]
            speaker_blocks.append({
                'speaker': current_speaker,
                'speaker_id': last_seg.get('speaker_id', 0),
                'start': block_start,
                'end': block_end
            })
        
        # Map speaker cho segments mới dựa trên thớigian overlap
        for new_seg in new_segments:
            seg_start = new_seg.get('start', 0)
            seg_end = new_seg.get('end', seg_start + 1.0)
            seg_mid = (seg_start + seg_end) / 2  # Dùng midpoint để xác định speaker
            
            # Tìm speaker block chứa midpoint của segment
            best_speaker = speaker_blocks[0]['speaker'] if speaker_blocks else 'Người nói 1'
            best_speaker_id = speaker_blocks[0]['speaker_id'] if speaker_blocks else 0
            
            for block in speaker_blocks:
                if block['start'] <= seg_mid <= block['end']:
                    best_speaker = block['speaker']
                    best_speaker_id = block['speaker_id']
                    break
            
            new_seg['speaker'] = best_speaker
            new_seg['speaker_id'] = best_speaker_id
        
        return new_segments
    
    def on_finished(self, text, result_data):
        try:
            self.segments = result_data.get("segments", [])
            # Sort segments theo thớigian để đảm bảo index khớp với thứ tự hiển thị
            self.segments.sort(key=lambda x: x.get('start', 0))
            self.paragraphs = result_data.get("paragraphs", [])
            self.has_speaker_diarization = result_data.get("has_speaker_diarization", False)
            self.speaker_segments_raw = result_data.get("speaker_segments_raw", [])
            timing = result_data.get("timing", {})
            
            # Nếu đang ở mode text-only, map speaker từ segments cũ sang mới
            if getattr(self, '_pending_text_only_mode', False) and self._pending_json_segments:
                print(f"[Text-Only Mode] Mapping speakers from {len(self._pending_json_segments)} old segments to {len(self.segments)} new segments")
                self.segments = self._map_speakers_to_new_segments(self.segments, self._pending_json_segments)
                self.has_speaker_diarization = True
                # Tạo speaker_segments_raw từ segments đã map
                unique_speakers = {}
                for seg in self.segments:
                    speaker_id = seg.get('speaker_id', 0)
                    speaker_name = seg.get('speaker', f'Người nói {speaker_id + 1}')
                    if speaker_id not in unique_speakers:
                        unique_speakers[speaker_id] = {
                            'speaker': speaker_name,
                            'speaker_id': speaker_id,
                            'start': seg.get('start', 0),
                            'end': seg.get('end', 0)
                        }
                    else:
                        unique_speakers[speaker_id]['end'] = seg.get('end', unique_speakers[speaker_id]['end'])
                self.speaker_segments_raw = list(unique_speakers.values())
                print(f"[Text-Only Mode] Mapped {len(unique_speakers)} unique speakers")
            
            # Cleanup
            self._pending_text_only_mode = False
            self._pending_json_segments = None
            
            # Tạo partials cho mỗi segment (nếu chưa có)
            for seg in self.segments:
                if 'partials' not in seg or not seg['partials']:
                    seg_start = seg.get('start', 0)
                    seg_end = seg.get('end', seg_start + 1.0)
                    seg['partials'] = [{
                        'text': seg.get('text', ''),
                        'timestamp': seg_end
                    }]
                # Đảm bảo có start_time (dùng cho logic highlight)
                if 'start_time' not in seg:
                    seg['start_time'] = seg.get('start', 0)
                # Đảm bảo speaker_id là int (diarization trả về int)
                if 'speaker_id' in seg:
                    try:
                        seg['speaker_id'] = int(seg['speaker_id'])
                    except (ValueError, TypeError):
                        pass  # giữ nguyên nếu không convert được
            
            self._last_rendered_highlight = -1
            self.render_text_content(immediate=True)
            
            if self.has_speaker_diarization and self.speaker_segments_raw:
                self.display_raw_speaker_segments(self.speaker_segments_raw)
            else:
                self.text_speaker_raw_output.setPlainText("Chưa có dữ liệu phân tách Người nói.")
            
            total = timing.get("total", 0.0)
            
            def fmt_time(t):
                if t is None: t = 0.0
                if t < 60: return f"{t:.1f}s"
                m = int(t // 60)
                s = int(t % 60)
                return f"{m}m {s}s"
            
            upload_convert = timing.get("upload_convert", 0.0)
            transcription_detail = timing.get("transcription_detail", 0.0)
            sentence_seg = timing.get("sentence_segmentation", 0.0)
            punctuation = timing.get("punctuation", 0.0)
            alignment = timing.get("alignment", 0.0)
            diarization = timing.get("diarization", 0.0)
            
            time_str = fmt_time(total)
            
            details = f"📊 TỔNG THỜI GIAN: {time_str}\n\n📝 CHI TIẾT CÁC GIAI ĐOẠN:"
            
            if upload_convert > 0.01:
                details += f"\n  • Tải & chuẩn hóa audio: {fmt_time(upload_convert)}"
            if transcription_detail > 0.01:
                details += f"\n  • Nhận dạng giọng nói: {fmt_time(transcription_detail)}"
            if sentence_seg > 0.01:
                details += f"\n  • Phân tách câu: {fmt_time(sentence_seg)}"
            if punctuation > 0.01:
                details += f"\n  • Thêm dấu câu: {fmt_time(punctuation)}"
            if alignment > 0.01:
                details += f"\n  • Căn chỉnh thời gian: {fmt_time(alignment)}"
            if diarization > 0.01 and self.check_speaker_diarization.isChecked():
                details += f"\n  • Phân đoạn Người nói: {fmt_time(diarization)}"
            
            self.stop_spinner()
            self.progress_bar.setFormat(f"✓ Hoàn tất! ({time_str})")
            self.progress_bar.setValue(100)
            
            if self.selected_file and os.path.exists(self.selected_file):
                playback_path = self._get_playback_path(self.selected_file)
                url = QUrl.fromLocalFile(os.path.abspath(playback_path))
                self.player.setSource(url)
                self.player_container.setVisible(True)
                self.btn_play.setEnabled(True)

            self.toggle_inputs(True)
            self.btn_save_json.setEnabled(True)
            self.btn_copy_text.setEnabled(True)
            
            if self.segments and self.selected_file:
                self.btn_rerun_diarization.setEnabled(True)
            
            p_win = None if self.window().isMinimized() or self.window().isHidden() else self.window()
            QMessageBox.information(p_win, "Thành công", f"Đã chuyển đổi xong!\n\n{details}\n\nBạn có thể nghe lại và bấm vào câu để tua.")
        except Exception as e:
            import traceback
            QMessageBox.critical(self.window(), "Lỗi hiển thị", f"Lỗi UI: {e}")

    def save_asr_json(self):
        """Lưu kết quả ASR hiện tại vào file JSON"""
        if not self.selected_file:
            QMessageBox.warning(self.window(), "Lỗi", "Chưa chọn file âm thanh!")
            return
        if not self.segments:
            QMessageBox.warning(self.window(), "Lỗi", "Chưa có dữ liệu để lưu!")
            return
        
        json_path = os.path.splitext(self.selected_file)[0] + '.asr.json'
        
        # Hỏi overwrite nếu file đã tồn tại
        if os.path.exists(json_path):
            reply = QMessageBox.question(self.window(), "Ghi đè",
                f"File JSON đã tồn tại:\n{os.path.basename(json_path)}\n\nBạn muốn ghi đè?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        try:
            model_name = self.combo_model.currentData() if hasattr(self, 'combo_model') else 'unknown'
            duration_sec = self.player_duration / 1000.0 if self.player_duration > 0 else 0

            json_data = serialize_segments(
                self.segments,
                speaker_name_mapping=self.speaker_name_mapping,
                model_name=model_name,
                model_type='file',
                duration_sec=duration_sec
            )
            _save_asr_json_file(json_path, json_data)
            
            # Mark as saved
            self.json_saved = True
            
            QMessageBox.information(self.window(), "Thành công", 
                f"Đã lưu kết quả ASR!\n\n📄 {os.path.basename(json_path)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.window(), "Lỗi", f"Không thể lưu JSON:\n{str(e)}")
    
    def copy_text_to_clipboard(self):
        """Sao chép toàn bộ nội dung văn bản vào clipboard"""
        if not self.segments:
            QMessageBox.warning(self.window(), "Lỗi", "Chưa có dữ liệu để sao chép!")
            return
        
        try:
            from PyQt6.QtWidgets import QApplication
            
            # Tạo text từ segments - giống format hiển thị trên view
            paragraphs = []
            current_speaker = None
            current_texts = []
            
            for seg in self.segments:
                speaker = seg.get('speaker', '')
                text = seg.get('text', '').strip()
                
                if not text:
                    continue
                
                # Kiểm tra speaker name mapping
                speaker_id = seg.get('speaker_id', 0)
                sid_str = str(speaker_id)
                if sid_str in self.speaker_name_mapping:
                    display_name = self.speaker_name_mapping[sid_str]
                else:
                    display_name = speaker
                
                # Nếu đổi speaker, lưu paragraph cũ và bắt đầu paragraph mới
                if display_name != current_speaker and display_name:
                    if current_speaker and current_texts:
                        paragraphs.append(f"{current_speaker}:\n{' '.join(current_texts)}")
                    current_speaker = display_name
                    current_texts = [text]
                else:
                    current_texts.append(text)
            
            # Thêm paragraph cuối cùng
            if current_speaker and current_texts:
                paragraphs.append(f"{current_speaker}:\n{' '.join(current_texts)}")
            
            full_text = '\n'.join(paragraphs)
            
            # Copy vào clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(full_text)
            
            # Hiển thị thông báo
            QMessageBox.information(self.window(), "Thành công", 
                f"Đã sao chép {len(full_text)} ký tự vào clipboard!")
            
        except Exception as e:
            QMessageBox.critical(self.window(), "Lỗi", f"Không thể sao chép:\n{str(e)}")
    
    def _load_asr_json(self, json_path):
        """Load dữ liệu ASR từ file JSON"""
        try:
            data = load_asr_json(json_path)
            segments, speaker_mapping, has_speakers = deserialize_segments(data)

            self.segments = segments
            self.speaker_name_mapping = speaker_mapping
            self.has_speaker_diarization = has_speakers
            self.current_highlight_index = -1
            self._last_rendered_highlight = -1

            # Render
            self.render_text_content(immediate=True)

            # Mark as saved since data was loaded from existing JSON file
            self.json_saved = True

            print(f"[_load_asr_json] Loaded {len(self.segments)} segments from {json_path}")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[_load_asr_json] Error: {e}")
            return False

    def perform_search(self, query):
        if not self.segments:
            return
        
        if not query or not query.strip():
            self.clear_search()
            return
            
        self.last_query = query
        self.search_matches = []
        self.current_search_index = -1
        
        query_norm = normalize_vietnamese(query)
        query_lower = query.lower()
        
        # Build concatenated text with segment boundaries mapping
        # This allows searching across segment boundaries
        concatenated_parts = []
        seg_boundaries = []  # List of (global_start, global_end, seg_idx, seg_start_pos)
        global_pos = 0
        
        for i, seg in enumerate(self.segments):
            text = seg.get('text', '')
            if not text:
                continue
            
            seg_start_global = global_pos
            seg_end_global = global_pos + len(text)
            
            concatenated_parts.append(text)
            seg_boundaries.append({
                'global_start': seg_start_global,
                'global_end': seg_end_global,
                'seg_idx': i,
                'text': text,
                'text_lower': text.lower(),
                'text_norm': normalize_vietnamese(text)
            })
            
            global_pos = seg_end_global + 1  # +1 for space separator between segments
            concatenated_parts.append(' ')  # Add space separator
        
        concatenated_text = ''.join(concatenated_parts)
        concatenated_lower = concatenated_text.lower()
        concatenated_norm = normalize_vietnamese(concatenated_text)
        
        # Helper function to map global position to segment position
        def map_global_to_seg(global_pos):
            """Map global position to (seg_idx, local_pos)"""
            for boundary in seg_boundaries:
                if boundary['global_start'] <= global_pos < boundary['global_end']:
                    return boundary['seg_idx'], global_pos - boundary['global_start']
            # Check if at boundary (last position of a segment)
            for boundary in seg_boundaries:
                if global_pos == boundary['global_end']:
                    return boundary['seg_idx'], global_pos - boundary['global_start']
            return None, None
        
        # Search in concatenated text (case-insensitive)
        start = 0
        while True:
            idx = concatenated_lower.find(query_lower, start)
            if idx == -1:
                break
            
            match_end = idx + len(query)
            
            # Find which segment(s) this match spans
            start_seg_idx, start_local = map_global_to_seg(idx)
            end_seg_idx, end_local = map_global_to_seg(match_end - 1)  # -1 because match_end is exclusive
            
            if start_seg_idx is None:
                start = idx + 1
                continue
            
            if end_seg_idx is None:
                # Match extends beyond last segment, adjust
                end_seg_idx = len(self.segments) - 1
                end_local = len(self.segments[end_seg_idx].get('text', ''))
            
            # Add match for each segment that the query spans
            if start_seg_idx == end_seg_idx:
                # Match within single segment
                seg_text = self.segments[start_seg_idx].get('text', '')
                end_pos = min(start_local + len(query), len(seg_text))
                self.search_matches.append({
                    'seg_idx': start_seg_idx,
                    'start': start_local,
                    'end': end_pos,
                    'text': seg_text[start_local:end_pos],
                    'score': 1.0
                })
            else:
                # Match spans multiple segments
                # Add match to first segment (from start to end of segment)
                seg_text_first = self.segments[start_seg_idx].get('text', '')
                first_seg_match_len = len(seg_text_first) - start_local
                self.search_matches.append({
                    'seg_idx': start_seg_idx,
                    'start': start_local,
                    'end': len(seg_text_first),
                    'text': seg_text_first[start_local:],
                    'score': 1.0,
                    'spans_to_next': True  # Mark that this match continues to next segment
                })
                
                # Add match to second segment (from start of segment)
                # Calculate how much of query goes into second segment
                query_remaining = len(query) - first_seg_match_len - 1  # -1 for space separator
                seg_text_second = self.segments[end_seg_idx].get('text', '')
                second_end = min(int(query_remaining), len(seg_text_second))
                if second_end > 0:
                    self.search_matches.append({
                        'seg_idx': end_seg_idx,
                        'start': 0,
                        'end': second_end,
                        'text': seg_text_second[:second_end],
                        'score': 1.0,
                        'continued_from_prev': True  # Mark that this is continuation
                    })
            
            start = idx + 1
        
        # Also search with normalized text (for Vietnamese accent-insensitive search)
        start = 0
        while True:
            idx = concatenated_norm.find(query_norm, start)
            if idx == -1:
                break
            
            match_end = idx + len(query_norm)
            
            start_seg_idx, start_local = map_global_to_seg(idx)
            end_seg_idx, end_local = map_global_to_seg(match_end - 1)
            
            if start_seg_idx is None:
                start = idx + 1
                continue
            
            if end_seg_idx is None:
                end_seg_idx = len(self.segments) - 1
                end_local = len(self.segments[end_seg_idx].get('text', ''))
            
            # Map normalized position to original position within the segment
            seg_text = self.segments[start_seg_idx].get('text', '')
            orig_start = self._map_norm_to_orig(seg_text, start_local)
            
            # Check for duplicate (already found by exact search)
            is_duplicate = False
            for existing in self.search_matches:
                if existing['seg_idx'] == start_seg_idx and abs(existing['start'] - orig_start) < 2:
                    is_duplicate = True
                    break
            
            if not is_duplicate and orig_start < len(seg_text):
                if start_seg_idx == end_seg_idx:
                    # Match within single segment - use original end
                    end_boundary = seg_boundaries[start_seg_idx]
                    match_len_in_seg = end_boundary['global_end'] - (end_boundary['global_start'] + start_local)
                    if match_len_in_seg > len(query_norm):
                        match_len_in_seg = len(query_norm)
                    
                    orig_end = self._map_norm_to_orig(seg_text, start_local + match_len_in_seg)
                    orig_end = min(orig_end, len(seg_text))
                    
                    self.search_matches.append({
                        'seg_idx': start_seg_idx,
                        'start': orig_start,
                        'end': orig_end,
                        'text': seg_text[orig_start:orig_end],
                        'score': 0.9
                    })
                else:
                    # Match spans multiple segments - add match to first segment
                    orig_end = len(seg_text)
                    self.search_matches.append({
                        'seg_idx': start_seg_idx,
                        'start': orig_start,
                        'end': orig_end,
                        'text': seg_text[orig_start:],
                        'score': 0.9,
                        'spans_to_next': True
                    })
                    
                    # Add match to second segment (normalized)
                    seg_text_second = self.segments[end_seg_idx].get('text', '')
                    if seg_text_second:
                        # Find the boundary for start segment
                        start_boundary = None
                        for boundary in seg_boundaries:
                            if boundary['seg_idx'] == start_seg_idx:
                                start_boundary = boundary
                                break
                        
                        # Calculate remaining length after first segment
                        if start_boundary:
                            first_seg_norm_len = start_boundary['global_end'] - (start_boundary['global_start'] + start_local)
                        else:
                            first_seg_norm_len = len(normalize_vietnamese(seg_text)) - start_local
                        remaining_norm_len = len(query_norm) - first_seg_norm_len - 1  # -1 for space
                        
                        # Map remaining length to original length in second segment
                        second_norm_text = normalize_vietnamese(seg_text_second)
                        actual_remaining = min(int(remaining_norm_len), len(second_norm_text))
                        orig_end_second = self._map_norm_to_orig(seg_text_second, actual_remaining)
                        orig_end_second = min(orig_end_second, len(seg_text_second))
                        
                        if orig_end_second > 0:
                            self.search_matches.append({
                                'seg_idx': end_seg_idx,
                                'start': 0,
                                'end': orig_end_second,
                                'text': seg_text_second[:orig_end_second],
                                'score': 0.9,
                                'continued_from_prev': True
                            })
            
            start = idx + 1
        
        self.search_matches.sort(key=lambda x: (x['seg_idx'], x['start']))
        
        count = len(self.search_matches)
        self.search_widget.label_count.setText(f"0/{count}")
        self.search_widget.set_has_results(count > 0)
        
        if count > 0:
            self.current_search_index = 0
            self.navigate_search(0)
        else:
            self.render_text_content()

    def _map_norm_to_orig(self, original, norm_idx):
        if norm_idx <= 0:
            return 0
        
        base_count = 0
        for i, c in enumerate(original):
            # Xử lý 'đ' giống như trong normalize_vietnamese
            if c.lower() == 'đ':
                base_count += 1
                if base_count > norm_idx:
                    return i
                continue
            
            decomposed = unicodedata.normalize('NFD', c)
            is_base = len(decomposed) > 0 and unicodedata.category(decomposed[0]) != 'Mn'
            if len(decomposed) == 0 or is_base:
                base_count += 1
            if base_count > norm_idx:
                return i
        
        return len(original)

    def navigate_search(self, direction):
        if not self.search_matches:
            return
            
        self.current_search_index = (self.current_search_index + direction) % len(self.search_matches)
        
        self.search_widget.label_count.setText(f"{self.current_search_index + 1}/{len(self.search_matches)}")
        
        self._last_rendered_highlight = -1
        self.render_text_content(immediate=True)
        
        match = self.search_matches[self.current_search_index]
        # Scroll sau khi render đã cập nhật HTML
        QTimer.singleShot(10, lambda: self.scroll_to_segment(match['seg_idx']))

    def clear_search(self):
        self.search_widget.input.clear()
        self.search_widget.label_count.setText("0/0")
        self.search_matches = []
        self.current_search_index = -1
        self._last_rendered_highlight = -1
        self.search_widget.set_has_results(False)
        self.render_text_content(immediate=True)

    def scroll_to_segment(self, seg_idx):
        # Scroll đến anchor seg_{seg_idx} (được thêm trong _do_render)
        self.text_output.scrollToAnchor(f"seg_{seg_idx}")
        QTimer.singleShot(50, lambda: self._ensure_segment_visible(seg_idx))

    def _ensure_segment_visible(self, seg_idx):
        if 0 <= seg_idx < len(self.segments):
            self.text_output.scrollToAnchor(f"seg_{seg_idx}")

    def render_text_content(self, immediate=False):
        if not self.segments:
            return
        
        if immediate:
            self._do_render()
        else:
            self.render_debounce_timer.stop()
            self.render_debounce_timer.start(50)

    def _render_text_with_search_highlight(self, text, anchor_id, seg_idx, chunk_start_pos=0):
        """Render text với search highlight nếu có.
        
        Args:
            text: Text cần render
            anchor_id: ID của anchor
            seg_idx: Index của segment
            chunk_start_pos: Vị trí bắt đầu của chunk này trong segment (để tính offset)
        
        Returns:
            HTML string với highlight nếu cần
        """
        # Escape HTML trước
        display_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Kiểm tra có search matches không
        if not self.search_matches or self.current_search_index < 0:
            # Không có search, chỉ render bình thường với audio highlight
            is_audio_highlight = (self.current_highlight_index == anchor_id)
            if is_audio_highlight:
                return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{display_text}</a>"
            else:
                return f"<a href='s_{anchor_id}' style='color: {COLORS['text_dark']}; text-decoration: none;'>{display_text}</a>"
        
        # Có search matches - tìm các matches trong đoạn text này
        chunk_end_pos = chunk_start_pos + len(text)
        matches_in_chunk = []
        
        for match_idx, match in enumerate(self.search_matches):
            if match['seg_idx'] == seg_idx:
                match_start = match['start']
                match_end = match['end']
                
                # Kiểm tra match có nằm trong chunk này không
                if match_start < chunk_end_pos and match_end > chunk_start_pos:
                    # Điều chỉnh vị trí relative với chunk
                    rel_start = max(0, match_start - chunk_start_pos)
                    rel_end = min(len(text), match_end - chunk_start_pos)
                    
                    # Check if this match is the current one
                    is_current = (match_idx == self.current_search_index)
                    
                    # Also check if this match is part of a multi-segment match
                    # If this match spans to next and current points to it, next is also current
                    # If this match continues from prev and current points to prev, this is also current
                    if not is_current:
                        if match.get('continued_from_prev') and self.current_search_index >= 0:
                            # Check if the previous match is current and spans to this
                            prev_match_idx = match_idx - 1
                            if prev_match_idx == self.current_search_index:
                                prev_match = self.search_matches[prev_match_idx]
                                if prev_match.get('spans_to_next'):
                                    is_current = True
                        elif match.get('spans_to_next') and self.current_search_index >= 0:
                            # Check if the next match is current and continues from this
                            next_match_idx = match_idx + 1
                            if next_match_idx == self.current_search_index:
                                next_match = self.search_matches[next_match_idx]
                                if next_match.get('continued_from_prev'):
                                    is_current = True
                    
                    matches_in_chunk.append({
                        'start': rel_start,
                        'end': rel_end,
                        'is_current': is_current
                    })
        
        if not matches_in_chunk:
            # Không có match trong chunk này
            is_audio_highlight = (self.current_highlight_index == anchor_id)
            if is_audio_highlight:
                return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{display_text}</a>"
            else:
                return f"<a href='s_{anchor_id}' style='color: {COLORS['text_dark']}; text-decoration: none;'>{display_text}</a>"
        
        # Có matches - cần cắt text và render từng phần
        # Sắp xếp matches theo vị trí
        matches_in_chunk.sort(key=lambda x: x['start'])
        
        # Merge các matches chồng lấp
        merged_matches = []
        for match in matches_in_chunk:
            if not merged_matches:
                merged_matches.append(match)
            else:
                last = merged_matches[-1]
                if match['start'] <= last['end']:
                    # Merge
                    last['end'] = max(last['end'], match['end'])
                    last['is_current'] = last['is_current'] or match['is_current']
                else:
                    merged_matches.append(match)
        
        # Render từng phần
        parts = []
        last_end = 0
        
        for match in merged_matches:
            # Phần trước match
            if match['start'] > last_end:
                pre_text = display_text[last_end:match['start']]
                parts.append(f"<span style='color: {COLORS['text_dark']};'>{pre_text}</span>")
            
            # Phần match
            match_text = display_text[match['start']:match['end']]
            if match['is_current']:
                # Match hiện tại - màu cam đậm
                parts.append(f"<span style='background-color: #ff8c00; color: #000000; padding: 1px 2px; border-radius: 2px; font-weight: bold;'>{match_text}</span>")
            else:
                # Các matches khác - màu cam nhạt
                parts.append(f"<span style='background-color: #ffd699; color: #000000; padding: 1px 2px; border-radius: 2px;'>{match_text}</span>")
            
            last_end = match['end']
        
        # Phần còn lại sau match cuối
        if last_end < len(display_text):
            post_text = display_text[last_end:]
            parts.append(f"<span style='color: {COLORS['text_dark']};'>{post_text}</span>")
        
        # Audio highlight cho toàn chunk nếu cần
        is_audio_highlight = (self.current_highlight_index == anchor_id)
        if is_audio_highlight:
            return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{''.join(parts)}</a>"
        else:
            return f"<a href='s_{anchor_id}' style='color: {COLORS['text_dark']}; text-decoration: none;'>{''.join(parts)}</a>"

    def _do_render(self):
        if not self.segments:
            return
        
        if getattr(self, 'has_speaker_diarization', False):
            self._do_render_speaker_view()
            return
        
        self._last_rendered_highlight = self.current_highlight_index

        para_boundaries = set()
        if self.paragraphs:
            sent_idx = 0
            total_para_sentences = 0
            for para in self.paragraphs:
                if sent_idx > 0:
                    para_boundaries.add(sent_idx)
                num_sentences = len(para.get('sentences', []))
                sent_idx += num_sentences
                total_para_sentences += num_sentences
            
            if total_para_sentences != len(self.segments):
                para_boundaries = set()

        html_content = f"<html><body><p style='font-size:14px; line-height:1.3; color:{COLORS['text_dark']};'>"
        
        for i, seg in enumerate(self.segments):
            # Thêm anchor cho segment để scroll khi tìm kiếm
            html_content += f"<a name='seg_{i}'></a>"
            
            if i in para_boundaries:
                html_content += "<br>"
            
            partials = seg.get('partials', [])
            
            if partials:
                # Render theo từng partial chunk (giống Tab Live)
                full_text = seg.get('text', '')
                search_pos = 0  # Vị trí tìm kiếm trong full_text
                
                for chunk_idx, partial in enumerate(partials):
                    chunk_text = partial.get('text', '')
                    if not chunk_text:
                        continue
                    
                    # Tìm vị trí của partial trong full_text
                    chunk_start_pos = full_text.find(chunk_text, search_pos)
                    if chunk_start_pos == -1:
                        chunk_start_pos = search_pos  # Fallback
                    
                    anchor_id = 1000000 + i * 1000 + chunk_idx
                    
                    # Sử dụng hàm mới để render với search highlight
                    html_content += self._render_text_with_search_highlight(
                        chunk_text, anchor_id, i, chunk_start_pos
                    ) + " "
                    
                    # Cập nhật vị trí tìm kiếm tiếp theo
                    search_pos = chunk_start_pos + len(chunk_text)
            else:
                # Fallback: render toàn câu
                text = seg.get('text', '')
                anchor_id = 1000000 + i * 1000
                
                # Sử dụng hàm mới để render với search highlight
                html_content += self._render_text_with_search_highlight(
                    text, anchor_id, i, 0
                ) + " "
            
        html_content += "</p></body></html>"
        
        scrollbar = self.text_output.verticalScrollBar()
        current_scroll = scrollbar.value()
        
        # Disable updates to prevent flickering/jumping to top
        self.text_output.setUpdatesEnabled(False)
        self.text_output.setHtml(html_content)
        
        # Restore scrollbar asynchronously after layout is updated
        from PyQt6.QtCore import QTimer
        self._scroll_attempts = getattr(self, '_scroll_attempts', 0)
        self._scroll_attempts = 0
        
        def restore_scroll():
            self._scroll_attempts += 1
            scrollbar.setValue(current_scroll)
            if scrollbar.maximum() < current_scroll and self._scroll_attempts < 50:
                QTimer.singleShot(10, restore_scroll)
            else:
                self.text_output.setUpdatesEnabled(True)
                
        if current_scroll > 0:
            restore_scroll()
        else:
            self.text_output.setUpdatesEnabled(True)

    def _merge_speaker_segments(self, segments, max_gap_sec=2.0):
        if not segments:
            return []
        
        sorted_segs = sorted(segments, key=lambda x: x.get('start', 0))
        
        merged = []
        current_group = [sorted_segs[0]]
        current_speaker = sorted_segs[0].get('speaker', 'Người nói 1')
        
        for seg in sorted_segs[1:]:
            speaker = seg.get('speaker', '') or 'Chưa gán'
            prev_end = current_group[-1].get('end', 0)
            curr_start = seg.get('start', 0)
            
            gap = curr_start - prev_end
            
            should_merge = False
            if speaker == current_speaker:
                if gap <= max_gap_sec:
                    should_merge = True
            
            if should_merge:
                current_group.append(seg)
            else:
                sentences = []
                for s in current_group:
                    sentences.append({
                        'index': s.get('index', 0),
                        'text': s.get('text', '').strip(),
                        'start': s.get('start', 0),
                        'end': s.get('end', 0)
                    })
                merged.append({
                    'speaker': current_speaker,
                    'speaker_id': current_group[0].get('speaker_id', 0),
                    'sentences': sentences,
                    'start': current_group[0].get('start', 0),
                    'end': current_group[-1].get('end', 0),
                    'first_index': current_group[0].get('index', 0),
                    'indices': [s.get('index', i) for i, s in enumerate(current_group)]
                })
                current_group = [seg]
                current_speaker = speaker
        
        if current_group:
            sentences = []
            for s in current_group:
                sentences.append({
                    'index': s.get('index', 0),
                    'text': s.get('text', '').strip(),
                    'start': s.get('start', 0),
                    'end': s.get('end', 0)
                })
            merged.append({
                'speaker': current_speaker,
                'speaker_id': current_group[0].get('speaker_id', 0),
                'sentences': sentences,
                'start': current_group[0].get('start', 0),
                'end': current_group[-1].get('end', 0),
                'first_index': current_group[0].get('index', 0),
                'indices': [s.get('index', i) for i, s in enumerate(current_group)]
            })
        
        MAX_GAP_FILL = 5.0
        if len(merged) > 1:
            for i in range(len(merged) - 1):
                current_end = merged[i]['end']
                next_start = merged[i + 1]['start']
                gap = next_start - current_end
                
                if gap > 0:
                    extension = min(gap, MAX_GAP_FILL)
                    merged[i]['end'] = current_end + extension
        
        return merged

    def _do_render_speaker_view(self):
        self._block_render_count = 0
        self.merged_speaker_blocks = []
        text_dark_color = COLORS['text_dark']
        html_content = f"<html><body style='font-size:14px; line-height:1.6; color:{text_dark_color};'>"
        
        segments_with_idx = [{**seg, 'index': i} for i, seg in enumerate(self.segments)]
        
        merged_segments = self._merge_speaker_segments(segments_with_idx, max_gap_sec=2.0)
        
        current_speaker = None
        current_blocks = []
        speaker_block_count = 0
        
        for seg in merged_segments:
            speaker = seg.get('speaker', '') or 'Chưa gán'
            
            if speaker != current_speaker:
                speaker_block_count += 1
                if current_blocks:
                    block_render_count = getattr(self, '_block_render_count', 0) + 1
                    self._block_render_count = block_render_count
                    
                    block_info = {
                        'speaker': current_speaker,
                        'sentences': [],
                        'start': current_blocks[0].get('start', 0),
                        'end': current_blocks[-1].get('end', 0)
                    }
                    for block in current_blocks:
                        block_info['sentences'].extend(block.get('sentences', []))
                    self.merged_speaker_blocks.append(block_info)
                    block_idx = len(self.merged_speaker_blocks) - 1
                    
                    speaker_id = current_blocks[0].get('speaker_id', 0) if current_blocks else 0
                    speaker_id_str = str(speaker_id)
                    
                    # Display name resolution
                    if speaker_id_str in self.speaker_name_mapping:
                        display_name = self.speaker_name_mapping[speaker_id_str]
                    else:
                        display_name = current_speaker
                    if DEBUG_LOGGING:
                        print(f"[RENDER DEBUG] Block {block_idx}: speaker_id={speaker_id}({type(speaker_id).__name__}), speaker_id_str='{speaker_id_str}', current_speaker='{current_speaker}', display_name='{display_name}', mapping={self.speaker_name_mapping}")
                    
                    html_content += f"<div style='margin: 16px 0; padding: 10px; background-color: #f8f9fa; border-left: 3px solid {COLORS['accent']}; border-radius: 0 4px 4px 0;'>"
                    
                    if self.check_show_speaker_labels.isChecked():
                        anchor_style = f"text-decoration:none; color:{COLORS['accent']}; cursor:pointer;"
                        html_content += f"<a href='spk_{speaker_id_str}_{block_idx}' style='{anchor_style}'><div style='font-weight:bold; margin-bottom:8px; font-size:13px;'>{display_name}:</div></a>"
                    
                    html_content += f"<div style='margin-left:4px; text-align: justify; line-height:1.6;'>"
                    
                    for block in current_blocks:
                        for sent in block.get('sentences', []):
                            sent_idx = sent.get('index', 0)
                            text = sent.get('text', '').strip()
                            
                            if not text:
                                continue
                            
                            # Thêm anchor cho segment để scroll khi tìm kiếm
                            html_content += f"<a name='seg_{sent_idx}'></a>"
                            
                            # Render theo partial chunks (giống _do_render)
                            seg_data = self.segments[sent_idx] if sent_idx < len(self.segments) else None
                            partials = seg_data.get('partials', []) if seg_data else []
                            
                            if partials:
                                full_text = seg_data.get('text', '') if seg_data else ''
                                search_pos = 0
                                
                                for chunk_idx, partial in enumerate(partials):
                                    chunk_text = partial.get('text', '')
                                    if not chunk_text:
                                        continue
                                    
                                    # Tìm vị trí của partial trong full_text
                                    chunk_start_pos = full_text.find(chunk_text, search_pos)
                                    if chunk_start_pos == -1:
                                        chunk_start_pos = search_pos
                                    
                                    anchor_id = 1000000 + sent_idx * 1000 + chunk_idx
                                    html_content += self._render_text_with_search_highlight(
                                        chunk_text, anchor_id, sent_idx, chunk_start_pos
                                    ) + " "
                                    search_pos = chunk_start_pos + len(chunk_text)
                            else:
                                anchor_id = 1000000 + sent_idx * 1000
                                html_content += self._render_text_with_search_highlight(
                                    text, anchor_id, sent_idx, 0
                                ) + " "
                    
                    html_content += "</div></div>"
                
                current_speaker = speaker
                current_blocks = []
            
            current_blocks.append(seg)
        
        if current_blocks:
            block_render_count = getattr(self, '_block_render_count', 0) + 1
            self._block_render_count = block_render_count
            
            block_info = {
                'speaker': current_speaker,
                'sentences': [],
                'start': current_blocks[0].get('start', 0),
                'end': current_blocks[-1].get('end', 0)
            }
            for block in current_blocks:
                block_info['sentences'].extend(block.get('sentences', []))
            self.merged_speaker_blocks.append(block_info)
            block_idx = len(self.merged_speaker_blocks) - 1
            
            speaker_id = current_blocks[0].get('speaker_id', 0) if current_blocks else 0
            speaker_id_str = str(speaker_id)
            
            # Display name resolution
            if speaker_id_str in self.speaker_name_mapping:
                display_name = self.speaker_name_mapping[speaker_id_str]
            else:
                display_name = current_speaker
            
            html_content += f"<div style='margin: 16px 0; padding: 10px; background-color: #f8f9fa; border-left: 3px solid {COLORS['accent']}; border-radius: 0 4px 4px 0;'>"
            
            if self.check_show_speaker_labels.isChecked():
                anchor_style = f"text-decoration:none; color:{COLORS['accent']}; cursor:pointer;"
                html_content += f"<a href='spk_{speaker_id_str}_{block_idx}' style='{anchor_style}'><div style='font-weight:bold; margin-bottom:8px; font-size:13px;'>{display_name}:</div></a>"
            
            html_content += f"<div style='margin-left:4px; text-align: justify; line-height:1.6;'>"
            
            for block in current_blocks:
                for sent in block.get('sentences', []):
                    sent_idx = sent.get('index', 0)
                    text = sent.get('text', '').strip()
                    
                    if not text:
                        continue
                    
                    # Thêm anchor cho segment để scroll khi tìm kiếm
                    html_content += f"<a name='seg_{sent_idx}'></a>"
                    
                    # Render theo partial chunks (giống _do_render)
                    seg_data = self.segments[sent_idx] if sent_idx < len(self.segments) else None
                    partials = seg_data.get('partials', []) if seg_data else []
                    
                    if partials:
                        full_text = seg_data.get('text', '') if seg_data else ''
                        search_pos = 0
                        
                        for chunk_idx, partial in enumerate(partials):
                            chunk_text = partial.get('text', '')
                            if not chunk_text:
                                continue
                            
                            # Tìm vị trí của partial trong full_text
                            chunk_start_pos = full_text.find(chunk_text, search_pos)
                            if chunk_start_pos == -1:
                                chunk_start_pos = search_pos
                            
                            anchor_id = 1000000 + sent_idx * 1000 + chunk_idx
                            html_content += self._render_text_with_search_highlight(
                                chunk_text, anchor_id, sent_idx, chunk_start_pos
                            ) + " "
                            search_pos = chunk_start_pos + len(chunk_text)
                    else:
                        anchor_id = 1000000 + sent_idx * 1000
                        html_content += self._render_text_with_search_highlight(
                            text, anchor_id, sent_idx, 0
                        ) + " "
            
            html_content += "</div></div>"
        
        html_content += "</body></html>"
        
        scrollbar = self.text_output.verticalScrollBar()
        current_scroll = scrollbar.value()
        
        # Disable updates to prevent flickering/jumping to top
        self.text_output.setUpdatesEnabled(False)
        self.text_output.setHtml(html_content)
        
        # Restore scrollbar asynchronously after layout is updated
        from PyQt6.QtCore import QTimer
        attempts = [0]
        
        def restore_scroll():
            attempts[0] += 1
            scrollbar.setValue(current_scroll)
            if scrollbar.maximum() < current_scroll and attempts[0] < 50:
                QTimer.singleShot(10, restore_scroll)
            else:
                self.text_output.setUpdatesEnabled(True)
                
        if current_scroll > 0:
            restore_scroll()
        else:
            self.text_output.setUpdatesEnabled(True)

    def display_raw_speaker_segments(self, speaker_segments):
        if not speaker_segments:
            self.text_speaker_raw_output.setPlainText("Không có dữ liệu phân tách Người nói.")
            return
        
        lines = []
        lines.append("=" * 70)
        lines.append("KẾT QUẢ PHÂN TÁCH NGƯỜI NÓI (RAW)")
        lines.append("=" * 70)
        
        model_id = self.combo_speaker_model.currentData() if hasattr(self, 'combo_speaker_model') else None
        model_info = SPEAKER_EMBEDDING_MODELS.get(model_id, {}) if model_id else {}
        model_name = model_info.get("name", "Unknown")
        model_size = model_info.get("size", "Unknown")
        
        lines.append(f"Model sử dụng: {model_name}")
        lines.append(f"Kích thước model: {model_size}")
        lines.append(f"Tổng số đoạn: {len(speaker_segments)}")
        lines.append("")
        
        lines.append(f"{'#':<4} {'Người nói':<12} {'Bắt đầu':<12} {'Kết thúc':<12} {'Thớigian':<10}")
        lines.append("-" * 70)
        
        sorted_segments = sorted(speaker_segments, key=lambda x: x.get('start', 0))
        
        for i, seg in enumerate(sorted_segments, 1):
            speaker = seg.get('speaker', 'Unknown')
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = end - start
            
            def fmt_time(t):
                m = int(t // 60)
                s = t % 60
                return f"{m:02d}:{s:05.2f}"
            
            lines.append(f"{i:<4} {speaker:<12} {fmt_time(start):<12} {fmt_time(end):<12} {duration:>6.2f}s")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("Thống kê theo Người nói:")
        lines.append("-" * 70)
        
        speaker_stats = {}
        for seg in sorted_segments:
            speaker = seg.get('speaker', 'Unknown')
            duration = seg.get('end', 0) - seg.get('start', 0)
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'count': 0, 'total_time': 0}
            speaker_stats[speaker]['count'] += 1
            speaker_stats[speaker]['total_time'] += duration
        
        for speaker, stats in sorted(speaker_stats.items()):
            lines.append(f"  {speaker}: {stats['count']} đoạn, tổng {stats['total_time']:.1f}s")
        
        lines.append("=" * 70)
        
        self.text_speaker_raw_output.setPlainText("\n".join(lines))

    def rerun_speaker_diarization(self):
        if not self.selected_file or not self.segments:
            QMessageBox.warning(self.window(), "Thiếu dữ liệu", 
                "Vui lòng xử lý file âm thanh trước khi chạy speaker diarization.")
            return
        
        if not DIARIZATION_AVAILABLE:
            QMessageBox.critical(self.window(), "Lỗi", "Speaker diarization không khả dụng.")
            return
        
        # Xóa dữ liệu tên Người nói cũ khi phân đoạn lại (speaker ID có thể thay đổi)
        self.speaker_name_mapping = {}
        self.block_speaker_names = {}
        self.custom_speaker_names = set()
        
        reply = QMessageBox.question(self.window(), "Xác nhận", 
                                   "Bạn có chắc chắn muốn chạy lại phân đoạn Người nói?\n"
                                   "Quá trình này có thể mất vài phút.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        self.btn_rerun_diarization.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.current_progress_text = "Đang chạy speaker diarization..."
        self.start_spinner()
        self.progress_bar.setValue(0)
        
        num_speakers = -1 if self.spin_num_speakers.currentIndex() == 0 else int(self.spin_num_speakers.currentText())
        speaker_model_id = self.combo_speaker_model.currentData()
        
        # Sử dụng WAV cache nếu có để xử lý nhanh hơn và nhất quán
        audio_file_for_diarization = self.selected_file
        if self.selected_file in self._playback_cache:
            cached_wav = self._playback_cache[self.selected_file]
            if os.path.exists(cached_wav):
                audio_file_for_diarization = cached_wav
                print(f"[Diarization] Using cached WAV: {cached_wav}")
        
        self.diarization_thread = SpeakerDiarizationThread(
            audio_file=audio_file_for_diarization,
            segments=self.segments,
            speaker_model_id=speaker_model_id,
            num_speakers=num_speakers,
            num_threads=self.slider_threads.value(),
            threshold=self.slider_diarization_threshold.value() / 100.0
        )
        self.diarization_thread.progress.connect(self.update_progress)
        self.diarization_thread.finished.connect(self.on_diarization_finished)
        self.diarization_thread.error.connect(self.on_diarization_error)
        self.diarization_thread.start()

    def on_diarization_finished(self, speaker_segments_raw, elapsed_time, merged_segments):
        self.speaker_segments_raw = speaker_segments_raw
        self.has_speaker_diarization = True
        
        if merged_segments:
            self.segments = merged_segments
            # Sort segments theo thớigian để đảm bảo index khớp với thứ tự hiển thị
            self.segments.sort(key=lambda x: x.get('start', 0))
            if self.paragraphs:
                new_paragraphs = []
                seg_idx = 0
                for para in self.paragraphs:
                    para_sents = para.get('sentences', [])
                    new_para_sents = []
                    for _ in para_sents:
                        if seg_idx < len(merged_segments):
                            new_para_sents.append(merged_segments[seg_idx])
                            seg_idx += 1
                    if new_para_sents:
                        new_paragraphs.append({
                            'text': ' '.join(s.get('text', '') for s in new_para_sents),
                            'sentences': new_para_sents
                        })
                self.paragraphs = new_paragraphs
        
        self.stop_spinner()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"Hoàn thành ({elapsed_time:.1f}s)")
        
        self.display_raw_speaker_segments(speaker_segments_raw)
        self.render_text_content(immediate=True)
        
        self.btn_rerun_diarization.setEnabled(True)
        self.btn_process.setEnabled(True)
        
        model_id = self.combo_speaker_model.currentData()
        model_info = SPEAKER_EMBEDDING_MODELS.get(model_id, {})
        model_name = model_info.get("name", model_id)
        model_size = model_info.get("size", "Unknown")
        
        p_win = None if self.window().isMinimized() or self.window().isHidden() else self.window()
        QMessageBox.information(p_win, "Hoàn thành Speaker Diarization", 
            f"Đã chạy xong!\n\n"
            f"Model: {model_name}\n"
            f"Kích thước: {model_size}\n"
            f"Thớ gian xử lý: {elapsed_time:.2f}s\n"
            f"Số đoạn phát hiện: {len(speaker_segments_raw)}")
        
        self.tab_widget.setCurrentIndex(1)

    def on_diarization_error(self, error_msg):
        self.stop_spinner()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Lỗi!")
        
        self.btn_rerun_diarization.setEnabled(True)
        self.btn_process.setEnabled(True)
        
        p_win = None if self.window().isMinimized() or self.window().isHidden() else self.window()
        QMessageBox.critical(p_win, "Lỗi Speaker Diarization", 
            f"Có lỗi xảy ra:\n{error_msg[:500]}")

    def on_error(self, err_msg):
        self.stop_spinner()
        self.progress_bar.setFormat("✗ Lỗi!")
        self.toggle_inputs(True)
        p_win = None if self.window().isMinimized() or self.window().isHidden() else self.window()
        QMessageBox.critical(p_win, "Lỗi xử lý", f"Đã có lỗi xảy ra:\n{err_msg}")

    def toggle_inputs(self, enable):
        self.btn_process.setEnabled(enable and self.selected_file is not None)
        self.drop_label.setEnabled(enable)
        self.config_content.setEnabled(enable)
        if not enable:
            self.drop_label.setStyleSheet(self.drop_label.styleSheet().replace("#e8f4ff", "#d0d0d0"))
        else:
            self.drop_label.setStyleSheet(f"""
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

    def toggle_playback(self):
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.player.play()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def on_player_duration_changed(self, duration):
        self.player_duration = duration
        self.slider_seek.setRange(0, duration)
        self.label_time.setText(f"00:00 / {self.fmt_ms(duration)}")

    def on_player_position_changed(self, position):
        if not self.slider_seek.isSliderDown():
            self.slider_seek.setValue(position)
        
        self.label_time.setText(f"{self.fmt_ms(position)} / {self.fmt_ms(self.player_duration)}")
        
        if not self.segments:
            return
        
        # Skip auto-highlight nếu user vừa click thủ công trong 500ms qua
        import time
        if hasattr(self, '_user_clicked_timestamp') and self._user_clicked_timestamp > 0:
            elapsed = time.time() * 1000 - self._user_clicked_timestamp
            if elapsed < 500:
                return
        
        current_sec = position / 1000.0
        
        # Tìm segment phù hợp nhất với current_time
        # Ưu tiên: 
        # 1. Segment có chứa current_time (start <= current <= end)
        #    - Nếu có overlap (nhiều segment chứa current_time), ưu tiên segment có start_time gần với current_time nhất
        # 2. Nếu không có segment nào chứa current_time, tìm segment có start_time gần nhất trong tương lai
        
        candidates = []  # Các segment chứa current_time
        future_candidates = []  # Các segment trong tương lai
        
        for i, seg in enumerate(self.segments):
            seg_start = seg.get('start', seg.get('start_time', 0))
            seg_end = seg.get('end', seg_start + 1.0)
            
            if seg_start <= current_sec <= seg_end:
                # Segment này chứa current_time
                candidates.append((i, seg_start, seg_end))
            elif seg_start > current_sec:
                # Segment trong tương lai
                future_candidates.append((i, seg_start, seg_end))
        
        best_idx = -1
        
        if candidates:
            # Ưu tiên segment có start_time gần với current_time nhất (segment mới bắt đầu)
            # Nếu current_time nằm trong khoảng 3 giây đầu của segment, ưu tiên segment đó
            best_candidate = None
            min_start_diff = float('inf')
            
            for idx, start, end in candidates:
                # Tính khoảng cách từ start_time đến current_time
                start_diff = current_sec - start
                
                # Ưu tiên segment mới bắt đầu (start_diff nhỏ) nhưng vẫn trong vòng 5 giây đầu
                if start_diff <= 5.0 and start_diff < min_start_diff:
                    min_start_diff = start_diff
                    best_candidate = idx
            
            # Nếu không có segment nào trong vòng 5 giây đầu, chọn segment có end_time xa nhất
            if best_candidate is None:
                max_end = -1
                for idx, start, end in candidates:
                    if end > max_end:
                        max_end = end
                        best_candidate = idx
            
            best_idx = best_candidate
            
        elif future_candidates:
            # Không có segment nào chứa current_time, chọn segment có start_time gần nhất
            min_future_distance = float('inf')
            for idx, start, end in future_candidates:
                distance = start - current_sec
                if distance < min_future_distance:
                    min_future_distance = distance
                    best_idx = idx
        
        # Tính anchor_id từ best_idx
        if best_idx != -1:
            best_anchor = 1000000 + best_idx * 1000
            
            # Phân giải đến mức partial chunk nếu có
            seg = self.segments[best_idx]
            partials = seg.get('partials', [])
            if partials:
                best_chunk_idx = len(partials) - 1
                for chunk_idx, partial in enumerate(partials):
                    if partial.get('timestamp', 0) > current_sec:
                        best_chunk_idx = chunk_idx
                        break
                best_anchor += best_chunk_idx

            if best_anchor != self.current_highlight_index:
                self.highlight_segment(best_anchor)

    def fmt_ms(self, ms):
        s = int(ms / 1000)
        m = int(s / 60)
        s = s % 60
        return f"{m:02}:{s:02}"

    def highlight_segment(self, idx):
        if idx == self.current_highlight_index:
            return
        self.current_highlight_index = idx
        self.pending_highlight_idx = idx
        self.render_debounce_timer.stop()
        self.render_debounce_timer.start(50)

    def set_position(self, position):
        self.player.setPosition(position)

    def seek_to_sentence(self, idx):
        """Click-to-seek: hỗ trợ cả partial anchor (1000000+) và legacy index"""
        timestamp_sec = None
        if DEBUG_LOGGING:
            print(f"[seek_to_sentence] idx={idx}")
        
        if idx >= 1000000:
            # Partial chunk anchor: 1000000 + seg_idx * 1000 + chunk_idx
            adjusted = idx - 1000000
            seg_idx = adjusted // 1000
            chunk_idx = adjusted % 1000
            if DEBUG_LOGGING:
                print(f"[seek_to_sentence] Partial anchor: seg_idx={seg_idx}, chunk_idx={chunk_idx}, total_segments={len(self.segments)}")
            
            if 0 <= seg_idx < len(self.segments):
                seg = self.segments[seg_idx]
                partials = seg.get('partials', [])
                if DEBUG_LOGGING:
                    print(f"[seek_to_sentence] Segment text='{seg.get('text', '')[:40]}', start={seg.get('start', 0):.2f}, partials={len(partials)}")
                if partials and chunk_idx < len(partials):
                    clicked_partial = partials[chunk_idx]
                    if chunk_idx == 0:
                        # Chunk đầu: seek tới start_time của segment
                        timestamp_sec = seg.get('start', seg.get('start_time', partials[0]['timestamp']))
                    else:
                        # Chunk sau: seek tới timestamp của chunk trước (= bắt đầu chunk này)
                        timestamp_sec = partials[chunk_idx - 1]['timestamp']
                    if DEBUG_LOGGING:
                        print(f"[seek_to_sentence] -> chunk '{clicked_partial['text'][:30]}' ts={clicked_partial['timestamp']:.2f}, seek_to={timestamp_sec:.2f}s")
                else:
                    # Fallback: seek tới start của segment
                    timestamp_sec = seg.get('start', seg.get('start_time', 0))
                    if DEBUG_LOGGING:
                        print(f"[seek_to_sentence] -> FALLBACK seek_to={timestamp_sec:.2f}s")
        else:
            # Legacy: segment index trực tiếp
            if 0 <= idx < len(self.segments):
                seg = self.segments[idx]
                timestamp_sec = seg.get('start', seg.get('start_time', 0))
                if DEBUG_LOGGING:
                    print(f"[seek_to_sentence] -> Legacy seek_to={timestamp_sec:.2f}s")
        
        if timestamp_sec is not None:
            import time
            self._user_clicked_timestamp = time.time() * 1000
            
            self.current_highlight_index = idx
            self.player.setPosition(int(timestamp_sec * 1000))
            if DEBUG_LOGGING:
                print(f"[seek_to_sentence] setPosition({int(timestamp_sec * 1000)}ms)")
            self.render_text_content(immediate=True)
    
    def on_speaker_label_clicked(self, speaker_id, block_index):
        """Xử lý khi click vào tên Người nói"""
        if DEBUG_LOGGING:
            print(f"\n[RENAME DEBUG] === on_speaker_label_clicked ===")
            print(f"[TAB_FILE][RENAME] === SPEAKER CLICKED === id={speaker_id}, block={block_index}")
            print(f"[RENAME DEBUG] speaker_id from anchor = '{speaker_id}' (type={type(speaker_id).__name__})")
            print(f"[RENAME DEBUG] block_index = {block_index}")
        
        # Convert to int  
        try:
            speaker_id_int = int(speaker_id)
        except (ValueError, TypeError):
            speaker_id_int = speaker_id
        if DEBUG_LOGGING:
            print(f"[RENAME DEBUG] speaker_id_int = {speaker_id_int} (type={type(speaker_id_int).__name__})")
        
        # Debug: show all unique speaker_ids in segments
        if self.segments:
            unique_ids = set()
            for seg in self.segments:
                sid = seg.get('speaker_id', '???')
                unique_ids.add((sid, type(sid).__name__))
            if DEBUG_LOGGING:
                print(f"[RENAME DEBUG] Unique speaker_ids in segments: {unique_ids}")
        
        # Tìm tên hiện tại
        current_name = None
        speaker_id_str = str(speaker_id)
        if speaker_id_str in self.speaker_name_mapping:
            current_name = self.speaker_name_mapping[speaker_id_str]
        
        if not current_name and self.segments:
            for seg in self.segments:
                if seg.get('speaker_id') == speaker_id_int:
                    current_name = seg.get('speaker', '')
                    break
        
        if not current_name:
            current_name = f"Người nói {speaker_id_int + 1}" if isinstance(speaker_id_int, int) else speaker_id
        
        if DEBUG_LOGGING:
            print(f"[RENAME DEBUG] current_name = '{current_name}'")
        
        # Collect all active speaker names to show in the list
        active_speaker_names = set(self.custom_speaker_names)
        if self.segments:
            for seg in self.segments:
                s_id_str = str(seg.get('speaker_id', 0))
                if s_id_str in self.speaker_name_mapping:
                    active_speaker_names.add(self.speaker_name_mapping[s_id_str])
                else:
                    active_speaker_names.add(seg.get('speaker', 'Người nói 1'))
                    
        dialog = SpeakerRenameDialog(speaker_id_int, current_name, active_speaker_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name, apply_to_all = dialog.get_result()
            if DEBUG_LOGGING:
                print(f"[RENAME DEBUG] new_name='{new_name}', apply_to_all={apply_to_all}")
            
            if new_name:
                if apply_to_all:
                    # === SỬA TẤT CẢ ===
                    self.merged_speaker_blocks = []
                    
                    sid_str = str(speaker_id_int)
                    self.speaker_name_mapping[sid_str] = new_name
                    self.custom_speaker_names.add(new_name)
                    
                    match_count = 0
                    total_count = len(self.segments)
                    for seg in self.segments:
                        seg_sid = seg.get('speaker_id')
                        if seg_sid == speaker_id_int:
                            seg['speaker'] = new_name
                            match_count += 1
                    if DEBUG_LOGGING:
                        print(f"[RENAME DEBUG] Matched {match_count}/{total_count} segments with speaker_id=={speaker_id_int}")
                        print(f"[RENAME DEBUG] speaker_name_mapping = {self.speaker_name_mapping}")
                else:
                    # === SỬA TÊN NÀY: Gán lại block cho speaker khác ===
                    # Tìm block range
                    block_start_idx = None
                    block_end_idx = None
                    
                    if hasattr(self, 'merged_speaker_blocks') and self.merged_speaker_blocks:
                        if 0 <= block_index < len(self.merged_speaker_blocks):
                            block_info = self.merged_speaker_blocks[block_index]
                            sentences = block_info.get('sentences', [])
                            if sentences:
                                indices = [s.get('index', 0) for s in sentences]
                                block_start_idx = min(indices)
                                block_end_idx = max(indices) + 1
                    
                    if block_start_idx is None:
                        current_block_idx = 0
                        prev_speaker = None
                        for i, seg in enumerate(self.segments):
                            seg_speaker = seg.get('speaker', '') or 'Chưa gán'
                            if i == 0 or seg_speaker != prev_speaker:
                                if i > 0:
                                    current_block_idx += 1
                                if current_block_idx == block_index:
                                    block_start_idx = i
                                elif current_block_idx == block_index + 1:
                                    block_end_idx = i
                                    break
                            prev_speaker = seg_speaker
                        if block_end_idx is None:
                            block_end_idx = len(self.segments)
                    
                    self.merged_speaker_blocks = []
                    self.custom_speaker_names.add(new_name)
                    
                    # Tìm speaker_id của target speaker (nếu đã có)
                    target_speaker_id = None
                    
                    # 1. Tìm trong mapping có tên đã sửa
                    for sid_str, mapped_name in self.speaker_name_mapping.items():
                        if mapped_name == new_name:
                            try:
                                target_speaker_id = int(sid_str)
                            except (ValueError, TypeError):
                                target_speaker_id = sid_str
                            break
                    
                    # 2. Tìm trong raw segments if not found
                    if target_speaker_id is None:
                        for seg in self.segments:
                            if seg.get('speaker') == new_name:
                                target_speaker_id = seg.get('speaker_id')
                                break
                    
                    # Nếu không tìm thấy → tạo ID mới = max + 1
                    if target_speaker_id is None:
                        all_ids = [seg.get('speaker_id', 0) for seg in self.segments 
                                   if isinstance(seg.get('speaker_id'), int)]
                        target_speaker_id = (max(all_ids) + 1) if all_ids else 0
                    
                    # Gán lại block cho speaker mới
                    if block_start_idx is not None and block_end_idx is not None:
                        for i in range(block_start_idx, min(block_end_idx, len(self.segments))):
                            self.segments[i]['speaker'] = new_name
                            self.segments[i]['speaker_id'] = target_speaker_id
                
                self.render_text_content(immediate=True)
    
    def _get_speaker_id_from_segment(self, segment):
        """Lấy speaker_id từ segment, ưu tiên trướng speaker_id nếu có"""
        # Ưu tiên dùng speaker_id đã lưu trong segment
        if 'speaker_id' in segment:
            return str(segment['speaker_id'])
        # Nếu không có, trích xuất từ tên
        speaker_name = segment.get('speaker', '')
        match = re.search(r'(\d+)$', str(speaker_name))
        return match.group(1) if match else speaker_name
    
    def _get_speaker_id(self, speaker_name):
        """Trích xuất speaker ID từ tên (e.g., 'Người nói 1' -> '1')"""
        match = re.search(r'(\d+)$', str(speaker_name))
        return match.group(1) if match else speaker_name
    
    def _ensure_segment_split(self, sentence_idx, chunk_idx, direction=None):
        if chunk_idx < 0 or sentence_idx >= len(self.segments):
            return sentence_idx
            
        seg = self.segments[sentence_idx]
        partials = seg.get('partials', [])
        if not partials:
            return sentence_idx
            
        if direction == 'prev':
            split_point = chunk_idx + 1
            if split_point >= len(partials):
                return sentence_idx
        elif direction == 'next':
            split_point = chunk_idx
            if split_point <= 0:
                return sentence_idx
        else:
            split_point = chunk_idx
            if split_point <= 0 or split_point >= len(partials):
                return sentence_idx
            
        left_partials = partials[:split_point]
        right_partials = partials[split_point:]
        
        split_time = right_partials[0].get('timestamp', seg.get('start', 0))
        
        import copy
        left_seg = copy.deepcopy(seg)
        left_seg['partials'] = left_partials
        left_seg['text'] = ' '.join([p.get('text', '').strip() for p in left_partials if p.get('text', '').strip()]).strip()
        left_seg['end'] = split_time
        
        right_seg = copy.deepcopy(seg)
        right_seg['partials'] = right_partials
        right_seg['text'] = ' '.join([p.get('text', '').strip() for p in right_partials if p.get('text', '').strip()]).strip()
        right_seg['start'] = split_time
        if 'start_time' in right_seg:
            right_seg['start_time'] = split_time
            
        self.segments[sentence_idx] = left_seg
        self.segments.insert(sentence_idx + 1, right_seg)
        
        for i in range(sentence_idx, len(self.segments)):
            self.segments[i]['index'] = i
            
        if direction == 'prev':
            return sentence_idx
        return sentence_idx + 1
    def on_split_speaker_requested(self, anchor_id):
        """Xử lý khi yêu cầu tách Người nói"""
        # Convert anchor_id to segment index
        if DEBUG_LOGGING:
            print(f"[TAB_FILE][SPLIT] === SPLIT SPEAKER === anchor_id={anchor_id}")
        if anchor_id >= 1000000:
            adjusted = anchor_id - 1000000
            sentence_idx = adjusted // 1000
            chunk_idx = adjusted % 1000
            sentence_idx = self._ensure_segment_split(sentence_idx, chunk_idx, 'next')
        else:
            sentence_idx = anchor_id
        
        if sentence_idx >= len(self.segments):
            return
        
        # Nếu chưa có diarization, khởi tạo tất cả segment với speaker mặc định
        if not getattr(self, 'has_speaker_diarization', False):
            for seg in self.segments:
                if 'speaker' not in seg:
                    seg['speaker'] = 'Người nói 1'
                    seg['speaker_id'] = 0  # int, consistent with diarization
            self.has_speaker_diarization = True
            self.speaker_name_mapping = {}
            self.block_speaker_names = {}
        
        current_speaker = self.segments[sentence_idx].get('speaker', 'Người nói 1')
        
        # Collect all active speaker names to show in the list
        active_speaker_names = set(self.custom_speaker_names)
        if self.segments:
            for seg in self.segments:
                s_id_str = str(seg.get('speaker_id', 0))
                if s_id_str in self.speaker_name_mapping:
                    active_speaker_names.add(self.speaker_name_mapping[s_id_str])
                else:
                    active_speaker_names.add(seg.get('speaker', 'Người nói 1'))

        dialog = SplitSpeakerDialog(current_speaker, active_speaker_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_speaker_name, split_scope = dialog.get_result()
            
            if new_speaker_name:
                self.custom_speaker_names.add(new_speaker_name)
                
                if split_scope == "to_end":
                    current_block_end = self._find_block_end(sentence_idx)
                    end_idx = current_block_end if current_block_end > 0 else len(self.segments)
                else:
                    end_idx = sentence_idx + 1
                
                # Tìm speaker_id của speaker đã có theo tên, hoặc tạo mới = max + 1
                target_speaker_id = None
                
                # 1. Tìm trong mapping có tên đã sửa
                for sid_str, mapped_name in self.speaker_name_mapping.items():
                    if mapped_name == new_speaker_name:
                        try:
                            target_speaker_id = int(sid_str)
                        except (ValueError, TypeError):
                            target_speaker_id = sid_str
                        break
                
                # 2. Tìm trong raw segments if not found
                if target_speaker_id is None:
                    for seg in self.segments:
                        if seg.get('speaker') == new_speaker_name:
                            target_speaker_id = seg.get('speaker_id')
                            break
                            
                if target_speaker_id is None:
                    all_ids = [seg.get('speaker_id', 0) for seg in self.segments 
                               if isinstance(seg.get('speaker_id'), int)]
                    target_speaker_id = (max(all_ids) + 1) if all_ids else 0
                
                for i in range(sentence_idx, min(end_idx, len(self.segments))):
                    self.segments[i]['speaker'] = new_speaker_name
                    self.segments[i]['speaker_id'] = target_speaker_id
                
                # Đảm bảo cập nhật checkbox hiển thị speaker labels
                if not self.check_show_speaker_labels.isEnabled():
                    self.check_show_speaker_labels.setEnabled(True)
                    self.check_show_speaker_labels.setChecked(True)
                
                # Reset merged_speaker_blocks để render lại từ đầu
                self.merged_speaker_blocks = []
                
                self.render_text_content(immediate=True)
                # Không hiện popup, render trực tiếp
    
    def _find_block_end(self, sentence_idx):
        """Tìm index kết thúc của block chứa sentence_idx"""
        if not self.segments:
            return 0
        
        current_speaker = self.segments[sentence_idx].get('speaker', 'Người nói 1')
        
        # Tìm đến khi gặp speaker khác hoặc hết danh sách
        for i in range(sentence_idx + 1, len(self.segments)):
            if self.segments[i].get('speaker', 'Người nói 1') != current_speaker:
                return i
        
        return len(self.segments)
    
    def _find_block_start(self, sentence_idx):
        """Tìm index bắt đầu của block chứa sentence_idx"""
        if not self.segments:
            return 0
        
        current_speaker = self.segments[sentence_idx].get('speaker', 'Người nói 1')
        
        # Tìm ngược lại đến khi gặp speaker khác hoặc hết danh sách
        for i in range(sentence_idx - 1, -1, -1):
            if self.segments[i].get('speaker', 'Người nói 1') != current_speaker:
                return i + 1
        
        return 0
    
    def on_merge_speaker_requested(self, anchor_id, direction):
        """Xử lý khi yêu cầu gộp Người nói"""
        # Convert anchor_id to segment index
        if anchor_id >= 1000000:
            adjusted = anchor_id - 1000000
            sentence_idx = adjusted // 1000
            chunk_idx = adjusted % 1000
            sentence_idx = self._ensure_segment_split(sentence_idx, chunk_idx, direction)
        else:
            sentence_idx = anchor_id
        
        if sentence_idx >= len(self.segments):
            return
        
        # Nếu chưa có diarization, không cần gộp (chỉ có 1 Người nói mặc định)
        if not getattr(self, 'has_speaker_diarization', False):
            return  # Không hiện popup
        
        current_speaker = self.segments[sentence_idx].get('speaker', 'Người nói 1')
        current_speaker_id = self.segments[sentence_idx].get('speaker_id', 0)
        if DEBUG_LOGGING:
            print(f"[TAB_FILE][MERGE] At sentence_idx={sentence_idx}: current_speaker={current_speaker}, id={current_speaker_id}")
        
        if direction == 'prev':
            if DEBUG_LOGGING:
                print(f"[TAB_FILE][MERGE] Mode: MERGE TO PREVIOUS")
            # Tìm Người nói phía trước
            prev_idx = None
            prev_speaker = None
            for i in range(sentence_idx - 1, -1, -1):
                speaker = self.segments[i].get('speaker', 'Người nói 1')
                if speaker != current_speaker:
                    prev_idx = i
                    prev_speaker = speaker
                    if DEBUG_LOGGING:
                        print(f"[TAB_FILE][MERGE] Found prev speaker at segment[{i}]: '{prev_speaker}'")
                    break
            
            if prev_speaker is None:
                return  # Không hiện popup
            
            # Gộp từ đầu block đến segment hiện tại vào Người nói trước
            block_start = self._find_block_start(sentence_idx)
            # Lấy speaker_id của ngướ i nói trước
            prev_speaker_id = self.segments[prev_idx].get('speaker_id', 0)
            if DEBUG_LOGGING:
                print(f"[TAB_FILE][MERGE] Merging from {block_start} to {sentence_idx} into '{prev_speaker}'")
            for i in range(block_start, sentence_idx + 1):
                old_s = self.segments[i].get('speaker', '')
                old_id = self.segments[i].get('speaker_id', '')
                if DEBUG_LOGGING:
                    print(f"[TAB_FILE][MERGE]   segment[{i}]: '{old_s}'(id={old_id}) -> '{prev_speaker}'(id={prev_speaker_id})")
                self.segments[i]['speaker'] = prev_speaker
                self.segments[i]['speaker_id'] = prev_speaker_id
            self.merged_speaker_blocks = []
            self.render_text_content(immediate=True)
            
        elif direction == 'next':
            # Tìm Người nói phía sau
            next_idx = None
            next_speaker = None
            for i in range(sentence_idx + 1, len(self.segments)):
                speaker = self.segments[i].get('speaker', 'Người nói 1')
                if speaker != current_speaker:
                    next_idx = i
                    next_speaker = speaker
                    break
            
            if next_speaker is None:
                return  # Không hiện popup
            
            # Gộp từ câu hiện tại đến hết block hiện tại vào Người nói sau
            block_end = self._find_block_end(sentence_idx)
            # Lấy speaker_id của ngướ i nói sau
            next_speaker_id = self.segments[next_idx].get('speaker_id', 0)
            for i in range(sentence_idx, block_end):
                self.segments[i]['speaker'] = next_speaker
                self.segments[i]['speaker_id'] = next_speaker_id
            
            # Reset merged_speaker_blocks để render lại từ đầu
            self.merged_speaker_blocks = []
            self.render_text_content(immediate=True)

    # ==================== AUDIO QUALITY METHODS ====================
    
    def _init_quality_analyzer(self):
        """Khởi tạo quality analyzer với offline model"""
        try:
            # Sẽ được lazy load khi cần
            pass
        except Exception as e:
            print(f"[FileTab] Cannot init quality analyzer: {e}")
    
    def _ensure_analyzer(self):
        """Đảm bảo analyzer đã được khởi tạo với model đang chọn"""
        if self.quality_analyzer is None:
            try:
                import sherpa_onnx as so
                
                # Lấy model đang chọn từ combo, không dùng default
                model_folder = self.combo_model.currentData()
                if not model_folder:
                    model_folder = "sherpa-onnx-zipformer-vi-2025-04-20"
                
                model_path = os.path.join(BASE_DIR, "models", model_folder)
                print(f"[FileTab] Using model for analysis: {model_folder}")
                
                # Tìm model files
                def find_file(pattern):
                    if not os.path.exists(model_path):
                        return None
                    files = [f for f in os.listdir(model_path) 
                            if f.startswith(pattern) and f.endswith(".onnx")]
                    float_files = [f for f in files if "int8" not in f]
                    if float_files:
                        return os.path.join(model_path, float_files[0])
                    if files:
                        return os.path.join(model_path, files[0])
                    return None
                
                encoder = find_file("encoder-")
                decoder = find_file("decoder-")
                joiner = find_file("joiner-")
                tokens = os.path.join(model_path, "tokens.txt")
                
                if all([encoder, decoder, joiner]):
                    recognizer = so.OfflineRecognizer.from_transducer(
                        tokens=tokens,
                        encoder=encoder,
                        decoder=decoder,
                        joiner=joiner,
                        num_threads=4,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="modified_beam_search",
                        max_active_paths=8,
                    )
                    self.quality_analyzer = AudioQualityAnalyzer(
                        offline_recognizer=recognizer,
                        online_recognizer=None
                    )
                    print("[FileTab] Quality analyzer initialized")
                else:
                    print(f"[FileTab] Model files not found in {model_path}")
            except Exception as e:
                print(f"[FileTab] Failed to init analyzer: {e}")
        
        return self.quality_analyzer is not None
    
    def _reset_analyzer(self):
        """Reset analyzer khi đổi model để tạo lại với model mới"""
        if self.quality_analyzer is not None:
            print("[FileTab] Resetting analyzer for new model selection")
            self.quality_analyzer = None
            
        # Nếu đang có file và đang bật tự động phân tích thì chạy lại ngay
        if self.selected_file and self.chk_auto_analyze.isChecked():
            self.analyze_file_quality()
    
    def analyze_file_quality(self):
        """Chạy phân tích chất lượng file"""
        if not self.selected_file:
            return
        
        if not self._ensure_analyzer():
            print("[FileTab] Analyzer not available")
            return
        
        # Kiểm tra DNSMOS model
        if not check_dnsmos_model_exists():
            reply = QMessageBox.question(self.window(),
                "Tải model DNSMOS",
                "Cần tải model DNSMOS (~5MB) để phân tích chất lượng.\n\nTải ngay?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._download_dnsmos_and_analyze()
            return
        
        # Chạy phân tích trong thread
        self.current_progress_text = "Đang phân tích chất lượng âm thanh..."
        self.start_spinner()
        self.btn_process.setEnabled(False)  # Không cho xử lý khi đang phân tích
        self.analysis_thread = AnalysisThread(
            self.quality_analyzer,
            file_path=self.selected_file,
            use_offline=True
        )
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.start()
    
    def _download_dnsmos_and_analyze(self):
        """Download DNSMOS rồi phân tích"""
        self.download_thread = DNSMOSDownloader()
        self.download_thread.finished.connect(self.on_dnsmos_download_finished)
        self.download_thread.start()
    
    def on_dnsmos_download_finished(self, success, msg):
        """Callback khi download xong"""
        if success:
            QMessageBox.information(self.window(), "Thành công", "Đã tải model DNSMOS!")
            self.analyze_file_quality()  # Phân tích sau khi tải xong
        else:
            QMessageBox.warning(self.window(), "Lỗi", f"Không thể tải DNSMOS:\n{msg}")
    
    def on_analysis_progress(self, message, percent):
        """Cập nhật progress phân tích"""
        self.progress_bar.setValue(percent)
        self.current_progress_text = f"{message} ({percent}%)"
    
    def on_analysis_finished(self, result):
        """Hiển thị kết quả phân tích"""
        self.stop_spinner()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Sẵn sàng")
        # Enable nút xử lý nếu có file và không đang xử lý chính
        transcriber_running = self.transcriber and self.transcriber.isRunning()
        if self.selected_file and not transcriber_running:
            self.btn_process.setEnabled(True)
        
        if result.error_message:
            QMessageBox.warning(self.window(), "Lỗi phân tích", result.error_message)
            return
        
        # Hiện dialog kết quả
        dialog = QualityResultDialog(result, self)
        dialog.exec()

    def on_auto_analyze_changed(self, state):
        """Xử lý khi checkbox tự động phân tích thay đổi"""
        # Nếu uncheck trong khi đang phân tích, dừng phân tích và enable nút xử lý
        if not state and self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            self.stop_spinner()
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Sẵn sàng")
            # Enable nút xử lý nếu có file
            if self.selected_file:
                self.btn_process.setEnabled(True)
