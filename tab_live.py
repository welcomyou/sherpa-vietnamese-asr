# tab_live.py - Tab xử lý trực tiếp từ microphone
import sys
import os
import json
import re
import wave
import shutil
import tempfile
import unicodedata
import html as html_module
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout, 
                             QMessageBox, QFrame, QToolButton, QLineEdit, QDialog, 
                             QInputDialog, QProgressBar, QFileDialog)
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices

from common import (BASE_DIR, COLORS, MicrophoneRecordThread, ClickableTextEdit, 
                    SpeakerHotkeyDialog, StreamingASRManager, OnlineStreamingASRManager,
                    SpeakerRenameDialog, SplitSpeakerDialog, SearchWidget, normalize_vietnamese,
                    show_missing_model_dialog, MODEL_DOWNLOAD_INFO)
from audio_analyzer import (
    AudioQualityAnalyzer, AnalysisResult, QualityMetrics,
    check_dnsmos_model_exists, DNSMOSDownloader
)
from quality_result_dialog import QualityResultDialog, MicTestDialog


class LiveProcessingTab(QWidget):
    """Tab xử lý trực tiếp từ microphone"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.is_recording = False
        self.recorded_audio = []
        self.transcribed_text = ""
        self.current_partial_text = ""
        self.pending_speaker_preview = None
        self.current_temp_file = None
        self.hotkey_config = self.load_hotkey_config()
        
        # Recording thread
        self.record_thread = None
        
        # ASR thread
        self.asr_thread = None
        
        # Preview thread
        self.preview_thread = None
        
        # Flag để tránh auto-highlight khi user vừa click (fix highlight nhảy sai)
        self._user_clicked_timestamp = 0
        
        # Flag for online streaming model
        self.is_using_online_model = False
        
        # For online model
        self._online_partial_text = ""
        
        # Audio quality analyzer
        self.quality_analyzer = None
        self.mic_analyzer = None
        
        # Speaker name management (giong tab_file)
        self.speaker_name_mapping = {}
        self.block_speaker_names = {}
        self.custom_speaker_names = set()
        self.merged_speaker_blocks = []
        self.has_speaker_diarization = False
        self._pending_mic_to_preview = None  # Lưu mic cần preview sau khi UI sẵn sàng
        
        # Track if audio has been saved
        self.has_recorded_audio = False
        self.wav_saved = False
        
        # Search state (giong tab_file)
        self.search_matches = []
        self.current_search_index = -1
        self.last_query = ""
        
        self.init_ui()
        self.refresh_microphones(try_restore=True)
    
    def showEvent(self, event):
        """Được gọi khi tab được hiển thị"""
        super().showEvent(event)
        # Start preview nếu có mic đang chờ
        if self._pending_mic_to_preview is not None and not self.is_recording:
            device_index = self.combo_microphone.currentData()
            if device_index is not None and device_index >= 0:
                self.start_preview()
            self._pending_mic_to_preview = None
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Configuration Group (Collapsible)
        self.config_container = QWidget()
        config_layout = QVBoxLayout(self.config_container)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(0)

        # Header
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
        
        # Nút Thông tin
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
        self.combo_model.addItem("zipformer-30m-rnnt-streaming-6000h (⭐)", "zipformer-30m-rnnt-streaming-6000h")
        self.combo_model.addItem("zipformer-30m-rnnt-6000h", "zipformer-30m-rnnt-6000h")
        self.combo_model.addItem("sherpa-onnx-zipformer-vi-2025-04-20", "sherpa-onnx-zipformer-vi-2025-04-20")
        self.combo_model.currentIndexChanged.connect(self._reset_mic_analyzer)
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
        self.slider_threads.valueChanged.connect(lambda v: self.label_threads.setText(str(v)))
        
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(self.slider_threads)
        threads_layout.addWidget(self.label_threads)
        form_config.addRow("Số luồng CPU:", threads_layout)
        
        # Hotkey Configuration Button
        self.btn_open_hotkey = QPushButton("⌨️ Cấu hình Hotkey người nói")
        self.btn_open_hotkey.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                color: white;
            }}
        """)
        self.btn_open_hotkey.clicked.connect(self.open_hotkey_dialog)
        form_config.addRow("người nói:", self.btn_open_hotkey)
        
        config_layout.addWidget(self.config_content)
        layout.addWidget(self.config_container)

        # 2. Microphone Selection + Volume Meter
        mic_widget = QWidget()
        mic_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_card']};
                border-radius: 6px;
            }}
        """)
        mic_layout = QHBoxLayout(mic_widget)
        mic_layout.setContentsMargins(8, 8, 8, 8)
        mic_layout.setSpacing(8)
        
        # Microphone combobox
        mic_label = QLabel("🎤 Microphone:")
        mic_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        mic_layout.addWidget(mic_label)
        
        self.combo_microphone = QComboBox()
        self.combo_microphone.setMinimumWidth(200)
        self.combo_microphone.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
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
                color: {COLORS['text_dark']};
                selection-background-color: {COLORS['accent']};
            }}
        """)
        self.combo_microphone.currentIndexChanged.connect(self.on_microphone_changed)
        mic_layout.addWidget(self.combo_microphone)
        
        # Refresh button
        self.btn_refresh_mic = QPushButton("🔄")
        self.btn_refresh_mic.setFixedSize(28, 28)
        self.btn_refresh_mic.setToolTip("Làm mới danh sách microphone")
        self.btn_refresh_mic.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                color: white;
            }}
        """)
        self.btn_refresh_mic.clicked.connect(self.refresh_microphones)
        mic_layout.addWidget(self.btn_refresh_mic)
        
        # Test microphone quality button
        self.btn_test_mic = QPushButton("📊 Đánh giá")
        self.btn_test_mic.setFixedSize(70, 28)
        self.btn_test_mic.setToolTip("Đánh giá chất lượng microphone (5 giây)")
        self.btn_test_mic.setStyleSheet(f"""
            QPushButton {{
                background-color: #17a2b8;
                color: white;
                border: 1px solid #17a2b8;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #138496;
                border-color: #138496;
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
                border-color: #6c757d;
            }}
        """)
        self.btn_test_mic.clicked.connect(self.test_microphone_quality)
        mic_layout.addWidget(self.btn_test_mic)
        
        mic_layout.addSpacing(20)
        
        # Volume meter
        volume_label = QLabel("🔊 Mức âm thanh:")
        volume_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        mic_layout.addWidget(volume_label)
        
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setTextVisible(False)
        self.volume_bar.setFixedHeight(12)
        self.volume_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                background-color: {COLORS['bg_dark']};
                max-width: 150px;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #28a745,
                    stop:0.5 #ffc107,
                    stop:1 #dc3545);
                border-radius: 5px;
            }}
        """)
        mic_layout.addWidget(self.volume_bar)
        
        mic_layout.addStretch()
        layout.addWidget(mic_widget)

        # 3. Control Buttons
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 8, 0, 8)
        control_layout.setSpacing(12)
        
        # Record/Stop button
        self.btn_record = QPushButton("🔴 Ghi âm")
        self.btn_record.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_record.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.btn_record)
        
        # Stop processing button
        self.btn_stop = QPushButton("⏹️ Kết thúc")
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text_dark']};
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #e0a800;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)
        
        # Export button
        self.btn_export = QPushButton("💾 Xuất file WAV")
        self.btn_export.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 140px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.btn_export.clicked.connect(self.export_audio)
        self.btn_export.setEnabled(False)
        control_layout.addWidget(self.btn_export)
        
        control_layout.addStretch()
        layout.addWidget(control_widget)

        # 4. Status label
        self.status_label = QLabel("Sẵn sàng. Chọn microphone và bấm 'Ghi âm' để bắt đầu.")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

        # 5. Output Area
        output_group = QGroupBox("📝 Nội dung nhận dạng")
        output_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        output_layout = QVBoxLayout(output_group)
        
        # Manual Speaker Insert Toolbar
        speaker_toolbar = QHBoxLayout()
        speaker_toolbar.setContentsMargins(0, 0, 0, 4)
        
        lbl_spk = QLabel("Thêm nhanh người nói:")
        lbl_spk.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: normal;")
        
        self.input_manual_speaker = QLineEdit()
        self.input_manual_speaker.setPlaceholderText("Nhập tên người nói...")
        self.input_manual_speaker.setFixedWidth(200)
        self.input_manual_speaker.setStyleSheet(f"background-color: {COLORS['bg_input']}; color: {COLORS['text_dark']}; border: 1px solid {COLORS['border']}; border-radius: 4px; padding: 4px;")
        self.input_manual_speaker.returnPressed.connect(self.on_manual_speaker_insert)
        
        btn_add_spk = QPushButton("Chèn")
        btn_add_spk.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add_spk.setStyleSheet(f"background-color: {COLORS['accent']}; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-weight: bold;")
        btn_add_spk.clicked.connect(self.on_manual_speaker_insert)
        
        # Search Widget (giong tab_file)
        self.search_widget = SearchWidget()
        self.search_widget.searchRequested.connect(self.perform_search)
        self.search_widget.nextRequested.connect(lambda: self.navigate_search(1))
        self.search_widget.prevRequested.connect(lambda: self.navigate_search(-1))
        self.search_widget.closed.connect(self.clear_search)
        self.search_widget.set_input_width(200)  # Make search input wider
        
        speaker_toolbar.addWidget(lbl_spk)
        speaker_toolbar.addWidget(self.input_manual_speaker)
        speaker_toolbar.addWidget(btn_add_spk)
        speaker_toolbar.addStretch()
        speaker_toolbar.addWidget(self.search_widget)
        
        output_layout.addLayout(speaker_toolbar)
        
        self.text_output = ClickableTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Kết quả nhận dạng giọng nói sẽ hiển thị ở đây...")
        self.text_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_dark']};
                font-size: 14px;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self.text_output.speakerLabelClicked.connect(self.on_live_speaker_clicked)
        self.text_output.sentenceClicked.connect(self.on_live_sentence_clicked)
        self.text_output.splitSpeakerRequested.connect(self.on_split_speaker_requested)
        self.text_output.mergeSpeakerRequested.connect(self.on_merge_speaker_requested)
        output_layout.addWidget(self.text_output)
        
        layout.addWidget(output_group)
        
        # 6. Audio Player for Recorded Audio
        self.stream_player_container = QWidget()
        self.stream_player_container.setStyleSheet(f"background-color: {COLORS['bg_card']}; border-radius: 4px;")
        self.stream_player_container.setMinimumHeight(40)
        self.stream_player_layout = QHBoxLayout(self.stream_player_container)
        self.stream_player_layout.setContentsMargins(6, 4, 6, 4)
        self.stream_player_layout.setSpacing(4)
        
        # Play/Pause button
        self.btn_stream_play = QPushButton("▶")
        self.btn_stream_play.setFixedSize(28, 28)
        self.btn_stream_play.setStyleSheet(f"background-color: {COLORS['accent']}; color: white; border: none; border-radius: 4px; font-size: 12px;")
        self.btn_stream_play.clicked.connect(self.on_stream_play_clicked)
        self.stream_player_layout.addWidget(self.btn_stream_play)
        
        # Seek slider
        self.slider_stream_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_stream_seek.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background: {COLORS['border']}; height: 6px; border-radius: 3px; }}
            QSlider::sub-page:horizontal {{ background: {COLORS['accent']}; border-radius: 3px; }}
            QSlider::handle:horizontal {{ background: {COLORS['accent']}; width: 12px; margin: -3px 0; border-radius: 6px; }}
        """)
        self.slider_stream_seek.setRange(0, 0)
        self.slider_stream_seek.sliderReleased.connect(self.on_stream_seek_released)
        self.stream_player_layout.addWidget(self.slider_stream_seek, 1)
        
        # Time label
        self.label_stream_time = QLabel("00:00 / 00:00")
        self.label_stream_time.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; min-width: 90px;")
        self.stream_player_layout.addWidget(self.label_stream_time)
        
        layout.addWidget(self.stream_player_container)
        self.stream_player_container.setVisible(False)
        
        # Media player for streaming audio
        self.stream_player = QMediaPlayer()
        self.stream_audio_output = QAudioOutput()
        self.stream_player.setAudioOutput(self.stream_audio_output)
        self.stream_player.positionChanged.connect(self.on_stream_position_changed)
        self.stream_player.durationChanged.connect(self.on_stream_duration_changed)
        self.stream_player.playbackStateChanged.connect(self.on_stream_state_changed)
        self.stream_duration = 0
    
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

    def refresh_microphones(self, try_restore=True):
        """Refresh danh sách microphone"""
        self.stop_preview()
        self.volume_bar.setValue(0)
        
        self.combo_microphone.clear()
        
        devices = QMediaDevices.audioInputs()
        if not devices:
            self.combo_microphone.addItem("Không tìm thấy microphone", -1)
            self.status_label.setText("⚠️ Không tìm thấy microphone nào!")
            return
        
        current_mic_list = [device.description() for device in devices]
        mic_restored = False
        
        for i, device in enumerate(devices):
            self.combo_microphone.addItem(device.description(), i)
        
        self.combo_microphone.blockSignals(True)
        try:
            if try_restore and self.main_window and hasattr(self.main_window, 'config'):
                config = self.main_window.config
                if 'LiveSettings' in config:
                    live_settings = config['LiveSettings']
                    saved_mic_list_str = live_settings.get('microphone_list', '')
                    saved_mic_list = saved_mic_list_str.split('|||') if saved_mic_list_str else []
                    saved_selected_mic = live_settings.get('selected_microphone', '')
                    
                    if saved_mic_list == current_mic_list and saved_selected_mic:
                        for i in range(self.combo_microphone.count()):
                            if self.combo_microphone.itemText(i) == saved_selected_mic:
                                self.combo_microphone.setCurrentIndex(i)
                                self.status_label.setText(f"✅ Đã khôi phục microphone: {saved_selected_mic}")
                                mic_restored = True
                                break
            
            if not mic_restored:
                self.combo_microphone.setCurrentIndex(0)
        finally:
            self.combo_microphone.blockSignals(False)
        
        # Đánh dấu để start preview khi UI sẵn sàng (trong showEvent)
        if mic_restored and not self.is_recording:
            device_index = self.combo_microphone.currentData()
            if device_index is not None and device_index >= 0:
                self._pending_mic_to_preview = device_index
        elif not mic_restored:
            self.status_label.setText(f"✅ Đã tìm thấy {len(devices)} microphone. Chọn một microphone để bắt đầu.")
    
    def on_microphone_changed(self, index):
        """Khi chọn microphone mới"""
        if index < 0:
            return
        
        device_index = self.combo_microphone.currentData()
        if device_index is not None and device_index >= 0:
            device_name = self.combo_microphone.currentText()
            self.status_label.setText(f"🎤 Đã chọn: {device_name}. Đang kiểm tra âm thanh...")
            # Reset volume bar ngay lập tức khi chuyển microphone
            self.volume_bar.setValue(0)
            self.start_preview()
    
    def start_preview(self):
        """Bắt đầu preview microphone"""
        if self.is_recording:
            return
        
        self.stop_preview()
        
        device_index = self.combo_microphone.currentData()
        if device_index is None or device_index < 0:
            return
        
        self.preview_thread = MicrophoneRecordThread(device_index=device_index)
        self.preview_thread.volume_changed.connect(self.on_volume_changed)
        self.preview_thread.error.connect(self.on_preview_error)
        self.preview_thread.start()
    
    def stop_preview(self):
        """Dừng preview microphone"""
        if self.preview_thread:
            # Ngắt kết nối signal trước khi dừng để tránh xung đột
            try:
                self.preview_thread.volume_changed.disconnect(self.on_volume_changed)
                self.preview_thread.error.disconnect(self.on_preview_error)
            except:
                pass
            self.preview_thread.stop()
            self.preview_thread.wait()
            self.preview_thread = None
    
    def on_preview_error(self, error_msg):
        """Xử lý lỗi preview"""
        if not self.is_recording:
            self.status_label.setText(f"⚠️ Lỗi microphone: {error_msg[:50]}...")
            self.volume_bar.setValue(0)
    
    def on_volume_changed(self, volume):
        """Cập nhật thanh âm lượng"""
        self.volume_bar.setValue(int(volume * 100))
    
    def toggle_recording(self):
        """Bắt đầu hoặc dừng ghi âm"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.pause_recording()

    def on_speaker_proposal(self, name):
        """Handle pending speaker preview from ASR thread"""
        if name:
            # Clear current partial text và partials để không hiển thị chữ xám cũ
            # Speaker change sẽ force end segment ngay lập tức
            self.current_partial_text = ""
            self.current_segment_partials = []  # Reset partials cho segment mới
            self.pending_speaker_preview = name
            print(f"[LiveTab] Speaker proposal received: {name}")
        else:
            self.pending_speaker_preview = None
        self._update_display()

    def _update_display(self):
        """Refreshes the text output"""
        self._update_display_with_timestamps()
    
    def _render_text_with_search_highlight(self, text, anchor_id, seg_idx, chunk_start_pos=0):
        """Render text với search highlight nếu có.
        
        Args:
            text: Text cần render
            anchor_id: ID của anchor
            seg_idx: Index của segment (trong danh sách text segments)
            chunk_start_pos: Vị trí bắt đầu của chunk này trong segment (để tính offset)
        
        Returns:
            HTML string với highlight nếu cần
        """
        # Escape HTML trước
        display_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Kiểm tra có search matches không
        if not self.search_matches or self.current_search_index < 0:
            # Không có search, chỉ render bình thường với audio highlight
            is_audio_highlight = (getattr(self, 'current_highlight_segment', -1) == anchor_id)
            if is_audio_highlight:
                return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{display_text}</a>"
            else:
                return f"<a href='s_{anchor_id}' style='color: black; text-decoration: none;'>{display_text}</a>"
        
        # Có search matches - tìm các matches trong đoạn text này
        chunk_end_pos = chunk_start_pos + len(text)
        matches_in_chunk = []
        
        for match in self.search_matches:
            if match['seg_idx'] == seg_idx:
                match_start = match['start']
                match_end = match['end']
                
                # Kiểm tra match có nằm trong chunk này không
                if match_start < chunk_end_pos and match_end > chunk_start_pos:
                    # Điều chỉnh vị trí relative với chunk
                    rel_start = max(0, match_start - chunk_start_pos)
                    rel_end = min(len(text), match_end - chunk_start_pos)
                    is_current = (self.search_matches.index(match) == self.current_search_index)
                    matches_in_chunk.append({
                        'start': rel_start,
                        'end': rel_end,
                        'is_current': is_current
                    })
        
        if not matches_in_chunk:
            # Không có match trong chunk này
            is_audio_highlight = (getattr(self, 'current_highlight_segment', -1) == anchor_id)
            if is_audio_highlight:
                return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{display_text}</a>"
            else:
                return f"<a href='s_{anchor_id}' style='color: black; text-decoration: none;'>{display_text}</a>"
        
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
                parts.append(f"<span style='color: black;'>{pre_text}</span>")
            
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
            parts.append(f"<span style='color: black;'>{post_text}</span>")
        
        # Audio highlight cho toàn chunk nếu cần
        is_audio_highlight = (getattr(self, 'current_highlight_segment', -1) == anchor_id)
        if is_audio_highlight:
            return f"<a href='s_{anchor_id}' style='color: #222222; text-decoration: none; background-color: {COLORS['highlight']}; padding: 2px 4px; border-radius: 3px; border: 1px solid #daa520;'>{''.join(parts)}</a>"
        else:
            return f"<a href='s_{anchor_id}' style='color: black; text-decoration: none;'>{''.join(parts)}</a>"
    
    def _update_display_with_timestamps(self):
        """Refreshes the text output with clickable partial chunks"""
        if not hasattr(self, 'current_partial_text'):
            self.current_partial_text = ""
        if not hasattr(self, 'pending_speaker_preview'):
            self.pending_speaker_preview = None
        if not hasattr(self, 'clickable_segments'):
            self.clickable_segments = []
        if not hasattr(self, 'current_highlight_segment'):
            self.current_highlight_segment = -1
        
        # FIX: Reset anchor counter khi bắt đầu render mới
        self._anchor_counter = 0
            
        html_parts = []
        text_seg_idx = 0  # Index chỉ tính text segments (cho search)
        
        # Đếm speaker blocks để tạo block_index
        speaker_block_count = 0
        
        for seg in self.clickable_segments:
            if seg.get('type') == 'speaker':
                speaker_name = seg['text'].replace('__SPK_SEP__', '').strip()
                # Kiểm tra xem có tên custom cho block này không
                display_name = speaker_name
                if speaker_block_count in self.block_speaker_names:
                    display_name = self.block_speaker_names[speaker_block_count]
                elif speaker_name in self.speaker_name_mapping:
                    display_name = self.speaker_name_mapping[speaker_name]
                
                # FIX: Dùng display_name làm speaker_id trong link để tránh nhầm lẫn
                # và dùng block_index để xác định chính xác speaker block
                sep_html = f"<div style='display: block; margin-top: 12px; margin-bottom: 4px; font-weight: bold; color: {COLORS['accent']}; border-left: 3px solid {COLORS['accent']}; padding-left: 8px;'><a href='spk_{speaker_block_count}' style='color: {COLORS['accent']}; text-decoration: none;'>{display_name}:</a></div>"
                html_parts.append(sep_html)
                speaker_block_count += 1
            elif seg.get('type') == 'partial':
                text = seg['text']
                anchor_id = seg.get('segment_id', text_seg_idx)
                timestamp = seg.get('start_time', 0)
                
                # Sử dụng hàm mới để render với search highlight
                html_parts.append(self._render_text_with_search_highlight(
                    text, anchor_id, text_seg_idx, 0
                ) + " ")
                text_seg_idx += 1
            else:
                # Segment type='text' - hiển thị theo partial chunks
                segment_id = seg.get('segment_id', text_seg_idx)
                partials = seg.get('partials', [])
                
                # Thêm anchor cho segment để scroll khi tìm kiếm
                html_parts.append(f"<a name='seg_{segment_id}'></a>")
                
                if partials:
                    # Hiển thị từng partial chunk là 1 clickable block
                    full_text = seg.get('text', '')
                    search_pos = 0
                    
                    for chunk_idx, partial in enumerate(partials):
                        chunk_text = partial.get('text', '')
                        if not chunk_text:
                            continue
                        chunk_ts = partial.get('timestamp', 0)
                        
                        # Tìm vị trí của partial trong full_text
                        chunk_start_pos = full_text.find(chunk_text, search_pos)
                        if chunk_start_pos == -1:
                            chunk_start_pos = search_pos
                        
                        # FIX: Dùng sequential anchor_id để tránh collision
                        if not hasattr(self, '_anchor_counter'):
                            self._anchor_counter = 0
                        anchor_id = 1000000 + self._anchor_counter
                        self._anchor_counter += 1
                        partial['_anchor_id'] = anchor_id
                        
                        # Sử dụng hàm mới để render với search highlight
                        html_parts.append(self._render_text_with_search_highlight(
                            chunk_text, anchor_id, text_seg_idx, chunk_start_pos
                        ) + " ")
                        search_pos = chunk_start_pos + len(chunk_text)
                else:
                    # Fallback: hiển thị full text
                    text = seg.get('text', '')
                    anchor_id = 1000000 + segment_id
                    
                    # Sử dụng hàm mới để render với search highlight
                    html_parts.append(self._render_text_with_search_highlight(
                        text, anchor_id, text_seg_idx, 0
                    ) + " ")
                
                text_seg_idx += 1
        
        # Only show current_partial_text if it's not empty and not already finalized
        if self.current_partial_text and self.is_recording:
            # Check if this partial text was already finalized (avoid duplicate display)
            last_finalized = getattr(self, '_last_finalized_text', '')
            if not last_finalized or not self.current_partial_text.startswith(last_finalized):
                html_parts.append(f"<span style='color: gray; font-style: italic;'>{self.current_partial_text}</span>")
            
        if self.pending_speaker_preview:
            sep_html = f"<div style='display: block; margin-top: 8px; margin-bottom: 0px; font-weight: bold; color: {COLORS['accent']}; border-top: 1px dashed {COLORS['border']}; padding-top: 2px;'>{self.pending_speaker_preview}</div>"
            html_parts.append(sep_html)
            
        full_html = "".join(html_parts)
        
        # Lưu scroll position trước khi render để giữ nguyên vị trí khi đang phát lại
        scrollbar = self.text_output.verticalScrollBar()
        current_scroll = scrollbar.value()
        
        self.text_output.setHtml(f"<span>{full_html}</span>")
        
        if self.is_recording:
            scrollbar.setValue(scrollbar.maximum())
        else:
            # Giữ nguyên scroll position khi đang phát lại và highlight
            scrollbar.setValue(current_scroll)
    
    def on_live_sentence_clicked(self, idx):
        """Handle click on a sentence/word group in live streaming view"""
        print(f"[on_live_sentence_clicked] Clicked anchor s_{idx}")
        
        if not hasattr(self, 'clickable_segments'):
            print(f"[on_live_sentence_clicked] No clickable_segments!")
            return
        
        timestamp_sec = None
        
        if idx >= 1000000:
            # FIX: Tìm từ _anchor_id được lưu
            print(f"[on_live_sentence_clicked] Looking for anchor_id={idx}")
            
            for seg in self.clickable_segments:
                if seg.get('type') == 'text':
                    partials = seg.get('partials', [])
                    if partials:
                        for chunk_idx, partial in enumerate(partials):
                            if partial.get('_anchor_id') == idx:
                                # Tìm thấy partial
                                if chunk_idx == 0:
                                    timestamp_sec = seg.get('start_time', partial['timestamp'])
                                else:
                                    timestamp_sec = partials[chunk_idx - 1]['timestamp']
                                
                                chunk_text = partial['text'][:30]
                                print(f"[on_live_sentence_clicked] Found chunk {chunk_idx}: '{chunk_text}...' seeking to={timestamp_sec:.2f}s")
                                break
                    elif seg.get('_anchor_id') == idx:
                        timestamp_sec = seg.get('start_time', 0)
                        print(f"[on_live_sentence_clicked] Found segment at {timestamp_sec:.2f}s")
                        break
                elif seg.get('type') == 'partial' and seg.get('_anchor_id') == idx:
                    timestamp_sec = seg.get('start_time', 0)
                    print(f"[on_live_sentence_clicked] Found partial at {timestamp_sec:.2f}s")
                    break
                
                if timestamp_sec is not None:
                    break
        else:
            for seg in self.clickable_segments:
                if seg.get('segment_id') == idx and seg.get('type') in ('text', 'partial'):
                    timestamp_sec = seg.get('start_time', 0)
                    if timestamp_sec < 0:
                        timestamp_sec = 0
                    seg_type = seg.get('type', 'unknown')
                    print(f"[on_live_sentence_clicked] Found segment {idx} ({seg_type}) at {timestamp_sec:.2f}s")
                    break
        
        if timestamp_sec is not None:
            self.last_clicked_timestamp = timestamp_sec
            self.current_highlight_segment = idx
            self._update_display_with_timestamps()
            
            # Đánh dấu user vừa click để skip auto-highlight trong 500ms (fix nhảy sai)
            import time
            self._user_clicked_timestamp = time.time() * 1000
            
            if self.stream_player.source().isValid():
                position_ms = int(timestamp_sec * 1000)
                self.stream_player.setPosition(position_ms)
                print(f"[on_live_sentence_clicked] Seeked to {position_ms}ms")
                self.status_label.setText(f"⏱️ Đã chọn đoạn tại {timestamp_sec:.1f}s")
            else:
                self.status_label.setText(f"⏱️ Đã chọn đoạn tại {timestamp_sec:.1f}s - Bấm phát để nghe")
        else:
            print(f"[on_live_sentence_clicked] Anchor s_{idx} not found!")
    
    def on_stream_play_clicked(self):
        """Play/Pause button for streaming audio"""
        state = self.stream_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.stream_player.pause()
            self.btn_stream_play.setText("▶")
        else:
            self.stream_player.play()
            self.btn_stream_play.setText("⏸")
    
    def on_stream_seek_released(self):
        """Seek slider released"""
        position = self.slider_stream_seek.value()
        self.stream_player.setPosition(position)
        self.highlight_segment_at_time(position)
    
    def highlight_segment_at_time(self, position_ms):
        """Find and highlight the partial chunk currently being played.
        Highlight chunk tiếp theo (chunk sắp được đọc) thay vì chunk đã đọc xong."""
        if not hasattr(self, 'clickable_segments') or not self.clickable_segments:
            return
        
        # Skip auto-highlight nếu user vừa click thủ công trong 500ms qua
        import time
        if hasattr(self, '_user_clicked_timestamp'):
            elapsed = time.time() * 1000 - self._user_clicked_timestamp
            if elapsed < 500:
                return
        
        current_sec = position_ms / 1000.0
        best_anchor = -1
        best_ts = -1
        min_future_distance = float('inf')
        
        for seg in self.clickable_segments:
            seg_type = seg.get('type', 'text')
            
            if seg_type == 'speaker' or seg_type == 'partial':
                continue
            
            partials = seg.get('partials', [])
            if partials:
                # Tìm chunk tiếp theo (timestamp > current_sec và gần nhất)
                for partial in partials:
                    chunk_ts = partial.get('timestamp', 0)
                    anchor_id = partial.get('_anchor_id', -1)
                    
                    if anchor_id < 0:
                        continue
                    
                    if chunk_ts > current_sec:
                        # Chunk này chưa được đọc (sắp tới)
                        distance = chunk_ts - current_sec
                        if distance < min_future_distance:
                            min_future_distance = distance
                            best_ts = chunk_ts
                            best_anchor = anchor_id
                    elif chunk_ts <= current_sec:
                        # Chunk đã được đọc, chỉ chọn nếu không có chunk nào ở tương lai
                        if min_future_distance == float('inf'):
                            # Đây là chunk cuối cùng của segment
                            distance = current_sec - chunk_ts
                            if distance < 2.0:  # Chỉ chọn nếu vừa mới đọc xong (< 2s)
                                best_ts = chunk_ts
                                best_anchor = anchor_id
            else:
                # Fallback: dùng start_time của segment
                seg_start = seg.get('start_time', 0)
                anchor_id = seg.get('_anchor_id', -1)
                if anchor_id < 0:
                    continue
                    
                if seg_start > current_sec:
                    distance = seg_start - current_sec
                    if distance < min_future_distance:
                        min_future_distance = distance
                        best_ts = seg_start
                        best_anchor = anchor_id
                elif seg_start <= current_sec and min_future_distance == float('inf'):
                    best_ts = seg_start
                    best_anchor = anchor_id
        
        if best_anchor != -1 and best_anchor != getattr(self, 'current_highlight_segment', -1):
            self.current_highlight_segment = best_anchor
            self._update_display_with_timestamps()
            print(f"[highlight] Chunk at {best_ts:.2f}s (current={current_sec:.2f}s) anchor={best_anchor}")
    
    def on_stream_position_changed(self, position):
        """Update slider, time label, and highlight current segment"""
        self.slider_stream_seek.setValue(position)
        self.label_stream_time.setText(f"{self.fmt_ms(position)} / {self.fmt_ms(self.stream_duration)}")
        self.highlight_segment_at_time(position)
    
    def on_stream_duration_changed(self, duration):
        """Duration changed"""
        self.stream_duration = duration
        self.slider_stream_seek.setRange(0, duration)
        self.label_stream_time.setText(f"00:00 / {self.fmt_ms(duration)}")
    
    def on_stream_state_changed(self, state):
        """Playback state changed"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_stream_play.setText("⏸")
        else:
            self.btn_stream_play.setText("▶")
    
    def fmt_ms(self, ms):
        """Format milliseconds to MM:SS"""
        s = int(ms / 1000)
        m = int(s / 60)
        s = s % 60
        return f"{m:02}:{s:02}"
    
    def save_recorded_audio_to_file(self):
        """Save recorded audio chunks to a temporary WAV file"""
        if not hasattr(self, 'recorded_audio') or len(self.recorded_audio) == 0:
            return None
        
        try:
            import wave
            import tempfile
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            all_audio = b''.join(self.recorded_audio)
            
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(all_audio)
            
            print(f"[save_recorded_audio_to_file] Saved to {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"[save_recorded_audio_to_file] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_streaming_audio(self):
        """Export recorded audio to user selected location"""
        self.export_audio()
    
    def start_recording(self):
        """Bắt đầu ghi âm"""
        device_index = self.combo_microphone.currentData()
        if device_index is None or device_index < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn microphone trước khi ghi âm!")
            return
        
        self.stop_preview()
        
        self.transcribed_text = ""
        self.current_partial_text = ""
        self.pending_speaker_preview = None
        self.recorded_audio = []
        self.text_output.clear()
        
        # Reset saved flags for new recording
        self.has_recorded_audio = False
        self.wav_saved = False
        
        self.current_segment_partials = []
        self.clickable_segments = []
        self.segment_counter = 0  # Segment counter tăng dần (0, 1, 2...)
        self._segment_id_counter = 0  # Counter cho split segment
        self._anchor_counter = 0  # Reset anchor counter
        
        # Reset search
        self.clear_search()
        
        self.stream_player.stop()
        self.stream_player.setSource(QUrl())
        self.stream_player_container.setVisible(False)
        
        self.current_highlight_segment = -1
        
        if hasattr(self, 'current_temp_file') and self.current_temp_file:
            try:
                if os.path.exists(self.current_temp_file):
                    os.remove(self.current_temp_file)
                    print(f"[start_recording] Cleaned up old temp file: {self.current_temp_file}")
            except Exception as e:
                print(f"[start_recording] Error cleaning up temp file: {e}")
            self.current_temp_file = None
        
        print("[start_recording] Reset transcribed_text and recorded_audio")
        
        if self.asr_thread:
            self.asr_thread.stop()
            self.asr_thread.wait(2000)
            self.asr_thread = None
        
        selected_model = self.combo_model.currentData()
        model_path = os.path.join(BASE_DIR, "models", selected_model)
        if not os.path.exists(model_path):
            show_missing_model_dialog(self, selected_model, model_path)
            return
        
        config = {
            "cpu_threads": self.slider_threads.value()
        }
        
        is_online_model = "zipformer-30m-rnnt-streaming" in selected_model.lower() or \
                         "streaming" in selected_model.lower()
        
        self.is_using_online_model = is_online_model
        
        try:
            if is_online_model:
                print(f"[start_recording] Using OnlineStreamingASRManager for {selected_model}")
                self.asr_thread = OnlineStreamingASRManager(model_path, config)
            else:
                print(f"[start_recording] Using StreamingASRManager for {selected_model}")
                self.asr_thread = StreamingASRManager(model_path, config)
            
            self.asr_thread.text_ready.connect(self.on_text_ready)
            self.asr_thread.processing_done.connect(self.on_processing_done)
            self.asr_thread.error.connect(self.on_asr_error)
            self.asr_thread.asr_ready.connect(self.on_asr_ready)
            self.asr_thread.speaker_proposal.connect(self.on_speaker_proposal)
            self.asr_thread.start()
            print("[start_recording] Streaming ASR started")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể khởi động ASR: {str(e)}")
            self.is_recording = False
            return
        
        self.is_recording = True
        self.btn_record.setText("⏸️ Dừng")
        self.btn_record.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text_dark']};
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #e0a800;
            }}
        """)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.combo_microphone.setEnabled(False)
        self.btn_refresh_mic.setEnabled(False)
        self.combo_model.setEnabled(False)
        
        self.status_label.setText("🔴 Đang ghi âm... Nói vào microphone.")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")
        
        if not self.recorded_audio:
            self.text_output.clear()
        
        self.record_thread = MicrophoneRecordThread(device_index=device_index)
        self.record_thread.volume_changed.connect(self.on_volume_changed)
        self.record_thread.chunk_ready.connect(self.on_audio_chunk)
        self.record_thread.error.connect(self.on_record_error)
        self.record_thread.start()
    
    def pause_recording(self):
        """Tạm dừng ghi âm"""
        self.is_recording = False
        self.btn_record.setText("🔴 Ghi âm")
        self.btn_record.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        self.status_label.setText("⏸️ Đã tạm dừng ghi âm. Bấm 'Ghi âm' để tiếp tục hoặc 'Kết thúc' để dừng.")
        self.status_label.setStyleSheet(f"color: {COLORS['warning']};")
        
        if self.record_thread:
            self.record_thread.stop()
            self.record_thread.wait()
            self.record_thread = None
        
        self.volume_bar.setValue(0)
        
        self.start_preview()
    
    def stop_processing(self):
        """Kết thúc xử lý và dừng ghi âm hoàn toàn"""
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.stop()
            self.record_thread.wait()
            self.record_thread = None
        
        if self.asr_thread:
            self.asr_thread.stop()
            self.asr_thread.wait()
            self.asr_thread = None
        
        self.btn_record.setText("🔴 Ghi âm")
        self.btn_record.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.combo_microphone.setEnabled(True)
        self.btn_refresh_mic.setEnabled(True)
        self.combo_model.setEnabled(True)
        self.volume_bar.setValue(0)
        
        self.start_preview()
        
        if hasattr(self, 'recorded_audio') and len(self.recorded_audio) > 0:
            temp_path = self.save_recorded_audio_to_file()
            if temp_path:
                self.current_temp_file = temp_path
                self.stream_player.setSource(QUrl.fromLocalFile(temp_path))
                self.stream_player_container.setVisible(True)
                self.stream_player_container.update()
                self.btn_stream_play.setText("▶")
                print(f"[stop_processing] Loaded audio into player: {temp_path}")
        
        self.status_label.setText("✅ Đã kết thúc. Bạn có thể xuất file WAV.")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        
        # Issue #13: Calculate duration correctly based on total bytes, not chunk count
        # Each sample is 2 bytes (16-bit), at 16000 Hz
        total_bytes = sum(len(chunk) for chunk in self.recorded_audio) if self.recorded_audio else 0
        duration_sec = total_bytes / (16000 * 2) if self.recorded_audio else 0
        text_len = len(self.transcribed_text) if hasattr(self, 'transcribed_text') else 0
        
        QMessageBox.information(self, "Hoàn thành", 
            f"Đã ghi âm xong!\n\n"
            f"Thờigian ghi âm: {duration_sec:.1f} giây\n"
            f"Độ dài văn bản: {text_len} ký tự\n\n"
            f"Bạn có thể xuất file WAV để lưu lại.")
    
    def on_audio_chunk(self, chunk):
        """Nhận chunk audio từ recording thread"""
        self.recorded_audio.append(chunk)
        self.has_recorded_audio = True
        
        if self.is_recording and self.asr_thread:
            self.asr_thread.add_audio(chunk)
            if len(self.recorded_audio) <= 5 or len(self.recorded_audio) % 50 == 0:
                print(f"[on_audio_chunk] Chunk #{len(self.recorded_audio)}: {len(chunk)} bytes, is_recording={self.is_recording}")
    
    def on_text_ready(self, text, is_final, timestamp_sec=0.0):
        # Issue #12: Still process final text even if not recording (could be result after pause)
        # But skip partial results when not recording
        if not self.is_recording and not is_final:
            return
        
        if "__SPK_SEP__" in text:
            self.transcribed_text += text
            self.current_partial_text = ""
            if hasattr(self, 'clickable_segments'):
                self.clickable_segments.append({
                    'type': 'speaker',
                    'text': text,
                    'start_time': timestamp_sec
                })
            self._update_display()
            return
        
        if not is_final:
            # Ignore partial text khi có pending speaker change (để không hiển thị chữ xám cũ)
            if self.pending_speaker_preview:
                return
            
            # === XỬ LÝ KHÁC NHAU CHO ONLINE VÀ OFFLINE MODEL ===
            # Offline: partial là diff (text mới thêm)
            # Online: partial là tích lũy (toàn bộ text từ đầu)
            
            if self.is_using_online_model:
                # ONLINE MODEL: partial là tích lũy, cần tính diff
                prev_full_text = ""
                if self.current_segment_partials:
                    prev_full_text = self.current_segment_partials[-1].get('full_text', '')
                
                # Tính diff
                if text.startswith(prev_full_text):
                    new_text = text[len(prev_full_text):].strip()
                else:
                    # ASR correction - text changed completely
                    # Clear all partials and start fresh
                    print(f"[on_text_ready] ONLINE CORRECTION: clearing {len(self.current_segment_partials)} partials")
                    self.current_segment_partials = []
                    new_text = text
                
                # Nếu không có text mới, skip
                if not new_text:
                    return
                    
                # Tạo partial chunk mới (luôn là diff)
                self.current_segment_partials.append({
                    'text': new_text,
                    'full_text': text,
                    'timestamp': timestamp_sec,
                    'is_partial': True
                })
                self.current_partial_text = text
                
            else:
                # OFFLINE MODEL: partial đã là diff
                prev_full_text = ""
                if self.current_segment_partials:
                    prev_full_text = self.current_segment_partials[-1].get('full_text', '')
                
                # Tìm phần text mới (diff)
                new_text = text
                is_correction = False
                if prev_full_text and text.startswith(prev_full_text):
                    # Normal case: text extends previous text
                    new_text = text[len(prev_full_text):].strip()
                elif prev_full_text and not text.startswith(prev_full_text):
                    # ASR correction: text changed (e.g., "quy định" -> "quy hoạch")
                    print(f"[on_text_ready] ASR CORRECTION: '{prev_full_text[:40]}...' -> '{text[:40]}...'")
                    
                    # Find longest common prefix between text and any previous partial
                    best_match_idx = -1
                    best_match_len = 0
                    for i, partial in enumerate(self.current_segment_partials):
                        p_full = partial.get('full_text', '')
                        # Find common prefix length
                        common_len = 0
                        for j in range(min(len(p_full), len(text))):
                            if p_full[j] == text[j]:
                                common_len += 1
                            else:
                                break
                        if common_len > best_match_len:
                            best_match_len = common_len
                            best_match_idx = i
                    
                    if best_match_idx >= 0 and best_match_len > 10:  # At least 10 chars match
                        # Keep partials up to best match, remove rest
                        removed_count = len(self.current_segment_partials) - best_match_idx - 1
                        for _ in range(removed_count):
                            removed = self.current_segment_partials.pop()
                            print(f"[on_text_ready] Removed partial: '{removed.get('text', '')[:30]}...'")
                        
                        # FIX: Truncate the kept partial to only include text up to common prefix
                        # This prevents duplicate display when ASR corrects text
                        kept_partial = self.current_segment_partials[-1]
                        kept_full_text = kept_partial.get('full_text', '')
                        kept_text = kept_partial.get('text', '')
                        
                        # Calculate how much of the kept partial belongs to common prefix
                        prev_cumulative_len = 0
                        if best_match_idx > 0:
                            prev_cumulative_len = len(self.current_segment_partials[-2].get('full_text', ''))
                        
                        chars_in_common = best_match_len - prev_cumulative_len
                        
                        if chars_in_common <= 0:
                            # Common prefix doesn't reach this partial, remove it too
                            removed = self.current_segment_partials.pop()
                            print(f"[on_text_ready] Removed kept partial (no common): '{removed.get('text', '')[:30]}...'")
                            # Recalculate new_text from start
                            new_text = text[best_match_len:].strip()
                        elif chars_in_common < len(kept_full_text):
                            # Need to truncate: find word boundary
                            # Get the portion of kept_text that belongs to common prefix
                            # kept_text starts at position (prev_cumulative_len - len(prev_partial_before_kept))
                            # Actually, kept_text is a diff, so we need to map position in full_text to kept_text
                            
                            # Simple approach: truncate kept_partial to end at common prefix
                            # Find where to cut in kept_text (position in full_text -> position in this partial's text)
                            full_text_before_kept = kept_full_text[:len(kept_full_text) - len(kept_text)]
                            chars_from_kept_in_common = best_match_len - len(full_text_before_kept)
                            
                            if chars_from_kept_in_common > 0 and chars_from_kept_in_common < len(kept_text):
                                # Find last space before cut point to avoid cutting mid-word
                                cut_point = chars_from_kept_in_common
                                while cut_point > 0 and kept_text[cut_point-1] != ' ':
                                    cut_point -= 1
                                
                                if cut_point > 0:
                                    truncated_text = kept_text[:cut_point].rstrip()
                                    kept_partial['text'] = truncated_text
                                    kept_partial['full_text'] = kept_full_text[:len(full_text_before_kept) + cut_point].rstrip()
                                    print(f"[on_text_ready] Truncated kept partial to: '{truncated_text[:40]}...'")
                                else:
                                    # Can't find good cut point, remove this partial
                                    removed = self.current_segment_partials.pop()
                                    print(f"[on_text_ready] Removed kept partial (no word boundary): '{removed.get('text', '')[:30]}...'")
                            
                            # Calculate diff from the (possibly truncated) kept partial
                            new_full_text = self.current_segment_partials[-1].get('full_text', '') if self.current_segment_partials else ''
                            if text.startswith(new_full_text):
                                new_text = text[len(new_full_text):].strip()
                            else:
                                new_text = text[best_match_len:].strip()
                        else:
                            # Kept partial is entirely within common prefix, no truncation needed
                            if text.startswith(kept_full_text):
                                new_text = text[len(kept_full_text):].strip()
                            else:
                                new_text = text[best_match_len:].strip()
                        
                        print(f"[on_text_ready] Kept {len(self.current_segment_partials)} partials, new_text='{new_text[:40]}...'")
                    else:
                        # No good match, clear all and start fresh
                        print(f"[on_text_ready] No good prefix match, clearing {len(self.current_segment_partials)} partials")
                        self.current_segment_partials = []
                        new_text = text
                        is_correction = True
                
                # Nếu text không đổi, không lưu
                if not new_text or new_text == prev_full_text:
                    return
                    
                # Group với partial trước nếu gần nhau (<1s và ngắn) và không phải correction
                should_merge = False
                if not is_correction and self.current_segment_partials:
                    last_partial = self.current_segment_partials[-1]
                    time_diff = timestamp_sec - last_partial['timestamp']
                    last_word_count = len(last_partial.get('text', '').split())
                    current_word_count = len(new_text.split())
                    
                    if time_diff < 1.0 and (last_word_count + current_word_count) < 8:
                        should_merge = True
                        last_partial['text'] += ' ' + new_text
                        last_partial['full_text'] = text
                        last_partial['timestamp_end'] = timestamp_sec
                
                if not should_merge:
                    self.current_segment_partials.append({
                        'text': new_text,
                        'full_text': text,
                        'timestamp': timestamp_sec,
                        'is_partial': True
                    })
                
                self.current_partial_text = text
            
            # Kết thúc xử lý partial cho cả 2 loại model
            self._update_display_with_timestamps()
            return
        
        # === FINAL: Cả 2 loại model xử lý giống nhau ===
        # Final: lưu segment với các partial chunks
        if hasattr(self, 'clickable_segments'):
            # Xóa partial previews của segment này nếu có
            self.clickable_segments = [
                s for s in self.clickable_segments 
                if not (s.get('type') == 'partial' and s.get('segment_id') == self.segment_counter)
            ]
            
            # For OFFLINE model: Use final text as ground truth, use partials only for timestamps
            # Create synthetic partials from final text based on partial timestamps
            # For ONLINE model: Keep existing partials as-is
            if not self.is_using_online_model and self.current_segment_partials:
                synthetic_partials = []
                final_words = text.split()
                num_partials = len(self.current_segment_partials)
                
                # Distribute words evenly across partials
                words_per_partial = len(final_words) // num_partials if num_partials > 0 else len(final_words)
                remainder = len(final_words) % num_partials if num_partials > 0 else 0
                
                word_idx = 0
                for i, orig_partial in enumerate(self.current_segment_partials):
                    # Calculate word count for this partial (distribute remainder to first ones)
                    chunk_word_count = words_per_partial + (1 if i < remainder else 0)
                    if word_idx + chunk_word_count > len(final_words):
                        chunk_word_count = len(final_words) - word_idx
                    
                    if chunk_word_count <= 0:
                        break
                    
                    chunk_words = final_words[word_idx:word_idx + chunk_word_count]
                    chunk_text = ' '.join(chunk_words)
                    
                    synthetic_partials.append({
                        'text': chunk_text,
                        'full_text': ' '.join(final_words[:word_idx + chunk_word_count]),
                        'timestamp': orig_partial['timestamp'],
                        'is_partial': True
                    })
                    
                    word_idx += chunk_word_count
                
                # If there are remaining words, add to last partial
                if word_idx < len(final_words):
                    remaining_words = final_words[word_idx:]
                    if synthetic_partials:
                        synthetic_partials[-1]['text'] += ' ' + ' '.join(remaining_words)
                        synthetic_partials[-1]['full_text'] = text
                    else:
                        synthetic_partials.append({
                            'text': text,
                            'full_text': text,
                            'timestamp': timestamp_sec,
                            'is_partial': True
                        })
                
                self.current_segment_partials = synthetic_partials
                print(f"[on_text_ready] Created {len(synthetic_partials)} synthetic partials from final text (offline model)")
            elif not self.current_segment_partials:
                # No partials at all (both online/offline) - create single partial with full text
                self.current_segment_partials.append({
                    'text': text,
                    'full_text': text,
                    'timestamp': timestamp_sec,
                    'is_partial': True
                })
                print(f"[on_text_ready] No partials, created single partial with full text")
            
            # Thêm segment với partial chunks
            if self.current_segment_partials:
                seg_start_ts = self.current_segment_partials[0]['timestamp']
            else:
                seg_start_ts = timestamp_sec
            
            self.clickable_segments.append({
                'type': 'text',
                'text': text,
                'start_time': seg_start_ts,
                'partials': self.current_segment_partials.copy(),  # list các partial chunks
                'segment_id': self.segment_counter
            })
            self.segment_counter += 1
        
        # Debug: Log partial chunks before clearing
        print(f"[on_text_ready] Final segment {self.segment_counter}: {len(self.current_segment_partials)} partials")
        for i, p in enumerate(self.current_segment_partials[:5]):  # Log first 5
            print(f"  Partial[{i}]: '{p.get('text', '')[:40]}...' ts={p.get('timestamp', 0):.2f}")
        
        self.current_segment_partials = []
        
        # Cập nhật transcribed_text
        if self.transcribed_text:
            self.transcribed_text += " " + text
        else:
            self.transcribed_text = text
        
        # IMPORTANT: Clear current_partial_text and reset to empty
        # to avoid displaying it on top of finalized segments
        self.current_partial_text = ""
        self._last_finalized_text = text  # Track last finalized text for comparison
        
        self._update_display_with_timestamps()
    
    def on_asr_error(self, msg):
        """Xử lý lỗi ASR"""
        print(f"[ASR Error] {msg}")
        QMessageBox.critical(self, "Lỗi ASR", f"Có lỗi xảy ra:\n{msg[:200]}")
    
    def on_processing_done(self):
        """Called when ASR has finished processing"""
        print("[on_processing_done] Processing complete")
        self.status_label.setText("⏸️ Đã tạm dừng. Bấm 'Ghi âm' để tiếp tục hoặc 'Kết thúc' để dừng.")
        
        if not self.is_recording:
            self.start_preview()
    
    def on_asr_ready(self):
        """Called when ASR thread is ready"""
        print(f"[on_asr_ready] ASR thread is ready, is_recording={self.is_recording}")
        if self.is_recording and self.asr_thread:
            print("[on_asr_ready] Starting ASR recording...")
            self.asr_thread.start_recording()
        else:
            print(f"[on_asr_ready] Not starting: is_recording={self.is_recording}, asr_thread={self.asr_thread}")
    
    def on_record_error(self, error_msg):
        """Xử lý lỗi recording"""
        QMessageBox.critical(self, "Lỗi ghi âm", f"Có lỗi xảy ra khi ghi âm:\n{error_msg}")
        self.pause_recording()
        
    def load_hotkey_config(self):
        config_path = os.path.join(BASE_DIR, "speaker_hotkeys.json")
        default_config = {str(i): "" for i in range(1, 10)}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
        except Exception as e:
            print(f"Error loading hotkeys: {e}")
        return default_config

    def save_hotkey_config(self):
        config_path = os.path.join(BASE_DIR, "speaker_hotkeys.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.hotkey_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể lưu cấu hình hotkey: {e}")

    def open_hotkey_dialog(self):
        dialog = SpeakerHotkeyDialog(self.hotkey_config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.hotkey_config = dialog.get_config()
            self.save_hotkey_config()
            
    def on_manual_speaker_insert(self):
        name = self.input_manual_speaker.text().strip()
        if name:
            self.insert_speaker_separator(name)
            self.input_manual_speaker.clear()
            

    def insert_speaker_separator(self, name):
        """Insert a speaker separator line into the transcript via ASR thread"""
        if not name:
            return
            
        if self.asr_thread:
            self.asr_thread.insert_speaker(name)
        else:
            separator_text = f" __SPK_SEP__{name}__SPK_SEP__"
            if self.transcribed_text:
                self.transcribed_text += separator_text
            else:
                self.transcribed_text = separator_text
            
            self._update_display()


    
    def _export_json(self, json_path):
        """Export clickable_segments to JSON file for Tab File compatibility"""
        if not hasattr(self, 'clickable_segments') or not self.clickable_segments:
            print("[_export_json] No clickable_segments to export")
            return False
        
        try:
            # Calculate duration
            total_bytes = sum(len(chunk) for chunk in self.recorded_audio) if self.recorded_audio else 0
            duration_sec = total_bytes / (16000 * 2) if total_bytes > 0 else 0
            
            # Get model info
            model_name = self.combo_model.currentData() if hasattr(self, 'combo_model') else 'unknown'
            model_type = 'online' if getattr(self, 'is_using_online_model', False) else 'offline'
            
            # Build JSON segments from clickable_segments
            json_segments = []
            
            # FIX: Speaker cùng tên có cùng speaker_id
            # Xây dựng mapping tên -> speaker_id
            speaker_name_to_id = {}
            next_speaker_id = 0
            
            # Pass 1: Xác định các tên speaker duy nhất theo thứ tự xuất hiện đầu tiên
            for seg in self.clickable_segments:
                if seg.get('type') == 'speaker':
                    speaker_name = seg['text'].replace('__SPK_SEP__', '').strip()
                    display_name = speaker_name
                    blk_idx = sum(1 for s in self.clickable_segments[:self.clickable_segments.index(seg)] if s.get('type') == 'speaker')
                    if blk_idx in self.block_speaker_names:
                        display_name = self.block_speaker_names[blk_idx]
                    elif speaker_name in self.speaker_name_mapping:
                        display_name = self.speaker_name_mapping[speaker_name]
                    
                    if display_name not in speaker_name_to_id:
                        speaker_name_to_id[display_name] = next_speaker_id
                        next_speaker_id += 1
            
            # Pass 2: Export với speaker_id theo tên
            segment_id_counter = 0
            current_speaker = None
            current_text_parts = []
            current_partials = []
            current_start_time = 0
            
            def flush_current_text():
                """Ghi lại text segment hiện tại nếu có"""
                nonlocal current_text_parts, current_partials, current_start_time, segment_id_counter
                if current_text_parts:
                    json_segments.append({
                        'type': 'text',
                        'text': ' '.join(current_text_parts),
                        'start_time': current_start_time,
                        'segment_id': segment_id_counter,
                        'partials': current_partials
                    })
                    segment_id_counter += 1
                    current_text_parts = []
                    current_partials = []
            
            for seg in self.clickable_segments:
                if seg.get('type') == 'speaker':
                    speaker_name = seg['text'].replace('__SPK_SEP__', '').strip()
                    display_name = speaker_name
                    blk_idx = sum(1 for s in self.clickable_segments[:self.clickable_segments.index(seg)] if s.get('type') == 'speaker')
                    if blk_idx in self.block_speaker_names:
                        display_name = self.block_speaker_names[blk_idx]
                    elif speaker_name in self.speaker_name_mapping:
                        display_name = self.speaker_name_mapping[speaker_name]
                    
                    speaker_id = speaker_name_to_id.get(display_name, 0)
                    
                    # Nếu đổi speaker, flush text cũ
                    if current_speaker != display_name:
                        flush_current_text()
                        current_speaker = display_name
                        json_segments.append({
                            'type': 'speaker',
                            'speaker': display_name,
                            'speaker_id': speaker_id,
                            'start_time': seg.get('start_time', 0)
                        })
                    
                elif seg.get('type') == 'text':
                    if not current_text_parts:
                        current_start_time = seg.get('start_time', 0)
                    
                    current_text_parts.append(seg.get('text', ''))
                    
                    for p in seg.get('partials', []):
                        current_partials.append({
                            'text': p.get('text', ''),
                            'timestamp': p.get('timestamp', 0)
                        })
            
            # Flush phần cuối
            flush_current_text()
            
            # Build final JSON
            json_data = {
                'version': 1,
                'model': model_name,
                'model_type': model_type,
                'created_at': datetime.now().isoformat(),
                'duration_sec': round(duration_sec, 2),
                'speaker_names': dict(self.speaker_name_mapping) if self.speaker_name_mapping else {},
                'segments': json_segments
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"[_export_json] Exported {len(json_segments)} segments to {json_path}")
            return True
            
        except Exception as e:
            print(f"[_export_json] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_audio(self):
        """Xuất file audio đã ghi âm"""
        if not self.recorded_audio:
            QMessageBox.warning(self, "Không có dữ liệu", "Chưa có dữ liệu ghi âm nào!")
            return
        
        default_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Xuất file audio", 
            default_name,
            "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)")
        
        if not file_path:
            return
        
        try:
            import wave
            import shutil
            
            audio_data = b''.join(self.recorded_audio)
            total_bytes = len(audio_data)
            duration_sec = total_bytes / (16000 * 2)
            
            print(f"[Export] Total: {total_bytes} bytes, ~{duration_sec:.1f}s")
            
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_export_{os.getpid()}.wav")
            
            with wave.open(temp_wav, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            print(f"[Export] Saved temp WAV: {temp_wav}")
            
            if file_path.endswith('.mp3'):
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_wav(temp_wav)
                    audio.export(file_path, format='mp3')
                except ImportError:
                    file_path = file_path[:-4] + '.wav'
                    shutil.copy(temp_wav, file_path)
            else:
                shutil.copy(temp_wav, file_path)
            
            if os.path.exists(temp_wav):
                os.unlink(temp_wav)
            
            # Mark as saved
            self.wav_saved = True
            
            # Export JSON alongside audio
            json_path = os.path.splitext(file_path)[0] + '.asr.json'
            json_exported = self._export_json(json_path)
            
            if json_exported:
                QMessageBox.information(self, "Thành công", 
                    f"Đã xuất file thành công!\n\n"
                    f"🎵 Audio: {file_path}\n"
                    f"📄 JSON: {json_path}")
            else:
                QMessageBox.information(self, "Thành công", 
                    f"Đã xuất file thành công!\n\nĐường dẫn: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất file", f"Không thể xuất file:\n{str(e)}")

    def keyPressEvent(self, event):
        """Xử lý phím tắt số 1-9 để chèn ngưởi nói nhanh"""
        from PyQt6.QtCore import Qt
        
        key = event.key()
        modifiers = event.modifiers()
        
        # Ctrl+F để mở tìm kiếm
        if key == Qt.Key.Key_F and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.search_widget.input.setFocus()
            self.search_widget.input.selectAll()
            return
        
        # Escape để đóng tìm kiếm
        if key == Qt.Key.Key_Escape:
            self.clear_search()
            return
        
        # Kiểm tra xem focus có đang ở ô nhập liệu (QLineEdit) không
        focus_widget = self.focusWidget()
        is_focus_in_input = isinstance(focus_widget, QLineEdit)
        
        # Xử lý phím số 1-9 để chèn speaker qua hotkey
        # (không xử lý khi đang focus vào ô nhập liệu để cho phép nhập số)
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            if is_focus_in_input:
                # Đang focus vào ô nhập liệu -> để số được nhập vào ô
                super().keyPressEvent(event)
                return
            
            # Xử lý hotkey số 1-9
            number = str(key - Qt.Key.Key_0)  # Key_1 - Key_0 = 1
            speaker_name = self.hotkey_config.get(number, "")
            if speaker_name:
                print(f"[Hotkey] Pressed {number} -> Insert speaker: {speaker_name}")
                self.insert_speaker_separator(speaker_name)
            return
        
        # Các phím khác xử lý bình thường
        super().keyPressEvent(event)

    # ==================== AUDIO QUALITY METHODS ====================
    
    def test_microphone_quality(self):
        """Test chất lượng microphone với dialog mới"""
        device_index = self.combo_microphone.currentData()
        if device_index is None or device_index < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn microphone trước!")
            return
        
        # Kiểm tra DNSMOS model
        if not check_dnsmos_model_exists():
            reply = QMessageBox.question(
                self,
                "Tải model DNSMOS",
                "Cần tải model DNSMOS (~5MB) để phân tích.\n\nTải ngay?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._download_dnsmos_for_mic()
            return
        
        # Đảm bảo analyzer đã sẵn sàng
        if not self._ensure_mic_analyzer():
            QMessageBox.warning(self, "Lỗi", "Không thể khởi tạo analyzer!")
            return
        
        # Mở dialog ghi âm - truyền device_name thay vì index
        device_name = self.combo_microphone.currentText()
        dialog = MicTestDialog(self.mic_analyzer, device_name, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            audio_bytes = dialog.get_audio()
            if audio_bytes:
                # Chuyển bytes -> numpy array
                import numpy as np
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Phân tích async để không block UI
                from audio_analyzer import AnalysisThread
                self.mic_analysis_thread = AnalysisThread(
                    self.mic_analyzer, audio=audio
                )
                self.mic_analysis_thread.finished.connect(self._on_mic_analysis_done)
                self.mic_analysis_thread.start()
    
    def _on_mic_analysis_done(self, result):
        """Callback khi phân tích mic xong"""
        if result.error_message:
            QMessageBox.warning(self, "Lỗi phân tích", result.error_message)
        else:
            result_dialog = QualityResultDialog(result, self)
            result_dialog.exec()
    
    def _download_dnsmos_for_mic(self):
        """Download DNSMOS model cho mic test"""
        self.download_thread = DNSMOSDownloader()
        self.download_thread.finished.connect(self._on_dnsmos_downloaded_for_mic)
        self.download_thread.start()
    
    def _on_dnsmos_downloaded_for_mic(self, success, msg):
        """Callback khi download xong"""
        if success:
            QMessageBox.information(self, "Thành công", "Đã tải model DNSMOS!\n\nVui lòng thử lại.")
        else:
            QMessageBox.warning(self, "Lỗi", f"Không thể tải DNSMOS:\n{msg}")
    
    
    def _reset_mic_analyzer(self):
        """Reset mic analyzer khi đổi model"""
        if self.mic_analyzer is not None:
            print("[LiveTab] Resetting mic analyzer for new model selection")
            self.mic_analyzer = None
            
    def _ensure_mic_analyzer(self):
        """Đảm bảo analyzer cho mic test đã được khởi tạo"""
        if self.mic_analyzer is None:
            try:
                import sherpa_onnx as so
                
                # Dùng streaming recognizer từ asr_thread nếu model trùng VÀ không đang recording
                # (OnlineRecognizer không thread-safe, không thể dùng chung khi đang ghi âm)
                online_recognizer = None
                if (not self.is_recording and 
                    self.asr_thread is not None and 
                    hasattr(self.asr_thread, 'worker')):
                    # Chỉ dùng nếu model đang chọn trùng với model của asr_thread
                    current_model = self.combo_model.currentData()
                    if hasattr(self.asr_thread, 'model_path') and current_model in self.asr_thread.model_path:
                        if hasattr(self.asr_thread.worker, 'recognizer'):
                            online_recognizer = self.asr_thread.worker.recognizer
                            print("[LiveTab] Reusing ASR recognizer for mic test (not recording)")
                
                # Nếu chưa có recognizer, tạo mới từ model đang chọn
                if online_recognizer is None:
                    print("[LiveTab] Creating temporary recognizer for mic test")
                    
                    # Lấy model đang chọn
                    model_folder = self.combo_model.currentData() if hasattr(self, 'combo_model') else None
                    if not model_folder:
                        # Thử tìm streaming model
                        model_folder = "zipformer-30m-rnnt-streaming-6000h"
                    
                    model_path = os.path.join(BASE_DIR, "models", model_folder)
                    
                    def find_file(pattern):
                        if not os.path.exists(model_path):
                            return None
                        files = [f for f in os.listdir(model_path) 
                                if f.startswith(pattern) and f.endswith(".onnx")]
                        if files:
                            return os.path.join(model_path, files[0])
                        return None
                    
                    encoder = find_file("encoder-")
                    decoder = find_file("decoder-")
                    joiner = find_file("joiner-")
                    tokens = os.path.join(model_path, "tokens.txt")
                    
                    if all([encoder, decoder, joiner]):
                        # Kiểm tra có phải streaming model không
                        is_streaming = 'streaming' in model_folder
                        
                        if is_streaming:
                            online_recognizer = so.OnlineRecognizer.from_transducer(
                                tokens=tokens,
                                encoder=encoder,
                                decoder=decoder,
                                joiner=joiner,
                                num_threads=4,
                                sample_rate=16000,
                                feature_dim=80,
                                decoding_method="greedy_search",
                                max_active_paths=4,
                            )
                            print(f"[LiveTab] Created streaming recognizer from {model_folder}")
                        else:
                            # Dùng offline recognizer cho analysis
                            offline_recognizer = so.OfflineRecognizer.from_transducer(
                                tokens=tokens,
                                encoder=encoder,
                                decoder=decoder,
                                joiner=joiner,
                                num_threads=4,
                                sample_rate=16000,
                                feature_dim=80,
                                decoding_method="greedy_search",
                            )
                            # Tạo analyzer với offline recognizer
                            self.mic_analyzer = AudioQualityAnalyzer(
                                offline_recognizer=offline_recognizer,
                                online_recognizer=None
                            )
                            print(f"[LiveTab] Created offline recognizer from {model_folder}")
                            return self.mic_analyzer is not None
                
                # Chỉ tạo analyzer nếu có ít nhất một recognizer
                if online_recognizer:
                    self.mic_analyzer = AudioQualityAnalyzer(
                        offline_recognizer=None,
                        online_recognizer=online_recognizer
                    )
                else:
                    print("[LiveTab] No recognizer available for mic test")
                    return False
            except Exception as e:
                print(f"[LiveTab] Failed to init mic analyzer: {e}")
        
        return self.mic_analyzer is not None

    # ==================== SPEAKER MANAGEMENT METHODS ====================
    # Các hàm quản lý ngườii nói (đổi tên, tách, gộp) - tương tự tab_file
    
    def on_live_speaker_clicked(self, speaker_id, block_index):
        """Xử lý khi click vào tên ngườii nói - nâng cấp để hỗ trợ đổi tên/tách/gộp"""
        print(f"\n{'='*60}")
        print(f"[TAB_LIVE][RENAME] === SPEAKER CLICKED === id={speaker_id}, block={block_index}")
        print(f"[TAB_LIVE][RENAME] Input: speaker_id='{speaker_id}', block_index={block_index}")
        
        # FIX: Với format link mới 'spk_{block_index}', speaker_id chính là block_index
        try:
            actual_block_index = int(speaker_id)
        except (ValueError, TypeError):
            actual_block_index = block_index
        
        # Tìm tên speaker tại block_index
        found_speaker_name = None
        speaker_count = 0
        for seg in self.clickable_segments:
            if seg.get('type') == 'speaker':
                if speaker_count == actual_block_index:
                    found_speaker_name = seg.get('text', '').replace('__SPK_SEP__', '').strip()
                    # Kiểm tra nếu có tên custom trong block_speaker_names
                    if actual_block_index in self.block_speaker_names:
                        found_speaker_name = self.block_speaker_names[actual_block_index]
                    break
                speaker_count += 1
        
        # Chuyển đổi clickable_segments sang format segments tương thích
        segments = self._convert_clickable_to_segments()
        if not segments:
            return
            
        # Lấy tên hiện tại của ngườii nói
        current_name = found_speaker_name
        
        if not current_name:
            current_name = f"Ngườii nói {actual_block_index + 1}"
        
        # Cập nhật block_index để dùng khi rename
        block_index = actual_block_index
        
        # Collect all active speaker names
        active_speaker_names = set(self.custom_speaker_names)
        if segments:
            for seg in segments:
                s_id = str(seg.get('speaker_id', 0))
                if s_id in self.speaker_name_mapping:
                    active_speaker_names.add(self.speaker_name_mapping[s_id])
                else:
                    active_speaker_names.add(seg.get('speaker', 'Người nói 1'))
        
        dialog = SpeakerRenameDialog(speaker_id, current_name, active_speaker_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name, apply_to_all = dialog.get_result()
            
            if new_name:
                if apply_to_all:
                    # FIX: Áp dụng cho tất cả đoạn có cùng TÊN HIỆN TẠI (current_name)
                    self.custom_speaker_names.add(new_name)
                    
                    # Cập nhật tất cả các speaker có cùng tên trong clickable_segments
                    speaker_count = 0
                    for seg in self.clickable_segments:
                        if seg.get('type') == 'speaker':
                            seg_text = seg.get('text', '').replace('__SPK_SEP__', '').strip()
                            # Kiểm tra cả tên gốc và tên custom
                            display_name = seg_text
                            if speaker_count in self.block_speaker_names:
                                display_name = self.block_speaker_names[speaker_count]
                            
                            if display_name == current_name:
                                seg['text'] = f"__SPK_SEP__{new_name}__SPK_SEP__"
                                # Xóa custom name nếu có để dùng tên mới từ segment
                                if speaker_count in self.block_speaker_names:
                                    del self.block_speaker_names[speaker_count]
                            speaker_count += 1
                else:
                    # Chỉ đổi tên cho block này
                    self.custom_speaker_names.add(new_name)
                    self.block_speaker_names[block_index] = new_name
                    
                    # Tìm và cập nhật segment tương ứng trong clickable_segments
                    speaker_count = 0
                    for seg in self.clickable_segments:
                        if seg.get('type') == 'speaker':
                            if speaker_count == block_index:
                                seg['text'] = f"__SPK_SEP__{new_name}__SPK_SEP__"
                                break
                            speaker_count += 1
                
                self._update_display()
    
    def on_split_speaker_requested(self, sentence_idx):
        """Xử lý khi yêu cầu tách ngườii nói"""
        print(f"\n{'='*60}")
        print(f"[TAB_LIVE][SPLIT] === SPLIT SPEAKER === sentence_idx={sentence_idx}")
        print(f"[TAB_LIVE][SPLIT] Input sentence_idx={sentence_idx}")
        original_anchor_id = sentence_idx
        print(f"[Live] Split speaker requested with anchor_id {original_anchor_id}")
        
        if not hasattr(self, 'clickable_segments') or not self.clickable_segments:
            print("[Live] No clickable_segments available")
            QMessageBox.warning(self, "Lỗi", "Chưa có nội dung nào để tách!")
            return
        
        # Chuyển đổi anchor_id thành index thực trong danh sách text segments
        # Anchor ID trong tab_live: 1000000 + segment_id * 1000 + chunk_idx
        real_idx = self._anchor_id_to_segment_index(original_anchor_id)
        if real_idx is None:
            print(f"[Live] Cannot find segment for anchor_id {original_anchor_id}")
            return
        
        print(f"[Live] Mapped anchor_id {original_anchor_id} to real_idx {real_idx}")
        
        segments = self._convert_clickable_to_segments()
        print(f"[Live] Converted {len(segments)} segments from clickable_segments")
        
        if real_idx >= len(segments):
            print(f"[Live] Invalid real_idx {real_idx}, only {len(segments)} segments")
            return
        
        sentence_idx = real_idx
        
        # Khởi tạo speaker nếu chưa có
        if not getattr(self, 'has_speaker_diarization', False):
            self.has_speaker_diarization = True
            print("[Live] Initialized speaker diarization")
        
        current_speaker = segments[sentence_idx].get('speaker', 'Ngườii nói 1')
        print(f"[Live] Current speaker at idx {sentence_idx}: {current_speaker}")
        
        dialog = SplitSpeakerDialog(current_speaker, self.custom_speaker_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_speaker_name, split_scope = dialog.get_result()
            print(f"[Live] Dialog accepted: new_name='{new_speaker_name}', scope='{split_scope}'")
            
            if new_speaker_name:
                self.custom_speaker_names.add(new_speaker_name)
                
                # Tìm vị trí chi tiết: segment nào, chunk nào
                location = self._find_segment_and_chunk(original_anchor_id)
                
                if location is None:
                    print(f"[Live] ERROR: Cannot find location for anchor {original_anchor_id}")
                    QMessageBox.warning(self, "Lỗi", "Không tìm thấy vị trí để tách ngườii nói!")
                    return
                
                clickable_idx, chunk_idx, seg_type = location
                print(f"[Live] Found at clickable_idx={clickable_idx}, chunk_idx={chunk_idx}, type={seg_type}")
                
                # Tạo speaker segment mới
                # Tính start_time từ chunk tại vị trí tách
                split_start_time = 0
                if seg_type == 'text' and clickable_idx < len(self.clickable_segments):
                    split_seg = self.clickable_segments[clickable_idx]
                    partials = split_seg.get('partials', [])
                    if partials and chunk_idx > 0 and chunk_idx < len(partials):
                        split_start_time = partials[chunk_idx - 1].get('timestamp', 0)
                    elif chunk_idx == 0:
                        split_start_time = split_seg.get('start_time', 0)
                    elif partials:
                        split_start_time = partials[0].get('timestamp', 0)
                
                new_speaker_seg = {
                    'type': 'speaker',
                    'text': f"__SPK_SEP__{new_speaker_name}__SPK_SEP__",
                    'segment_id': -1,
                    'start_time': split_start_time
                }
                
                if split_scope == "to_end":
                    # Từ chunk này đến hết: cần tách segment thành 2 phần
                    if seg_type == 'text' and chunk_idx > 0:
                        # Tách segment thành 2 phần tại chunk_idx
                        self._split_text_segment_at_chunk(clickable_idx, chunk_idx)
                        # Chèn speaker mới vào giữa 2 phần
                        self.clickable_segments.insert(clickable_idx + 1, new_speaker_seg)
                        print(f"[Live] Split segment at chunk {chunk_idx}, inserted speaker at {clickable_idx + 1}")
                    else:
                        # Chèn speaker vào trước segment này
                        self.clickable_segments.insert(clickable_idx, new_speaker_seg)
                        print(f"[Live] Inserted speaker at clickable_idx {clickable_idx}")
                else:
                    # Chỉ chunk này: tách segment thành 3 phần (trước, chunk này, sau)
                    if seg_type == 'text':
                        self._split_text_segment_for_single_chunk(clickable_idx, chunk_idx, new_speaker_name, current_speaker)
                        print(f"[Live] Split segment for single chunk {chunk_idx}")
                    else:
                        # Partial đơn lẻ: chèn speaker trước và sau
                        self.clickable_segments.insert(clickable_idx, new_speaker_seg)
                        old_speaker_seg = {
                            'type': 'speaker',
                            'text': f"__SPK_SEP__{current_speaker}__SPK_SEP__",
                            'segment_id': -1
                        }
                        self.clickable_segments.insert(clickable_idx + 2, old_speaker_seg)
                
                self._update_display()
                print("[Live] Display updated")
    
    def on_merge_speaker_requested(self, sentence_idx, direction):
        """Xử lý khi yêu cầu gộp ngườii nói - GỘP CẢ TEXT SEGMENTS"""
        print(f"\n{'='*60}")
        print(f'[TAB_LIVE][MERGE] === MERGE REQUESTED sentence_idx={sentence_idx} direction={direction}')
        print(f"[TAB_LIVE][MERGE] Input sentence_idx={sentence_idx}, direction='{direction}'")
        print(f"[Live] Merge speaker requested: anchor_id={sentence_idx}, direction={direction}")
        
        if not getattr(self, 'has_speaker_diarization', False):
            print("[Live] No speaker diarization, skip merge")
            return
        
        # Chuyển đổi anchor_id thành index thực
        real_idx = self._anchor_id_to_segment_index(sentence_idx)
        if real_idx is None:
            print(f"[Live] Cannot find segment for anchor_id {sentence_idx}")
            return
        
        print(f"[Live] Mapped anchor_id {sentence_idx} to real_idx {real_idx}")
        
        # Tìm block của current speaker trong clickable_segments
        current_block = self._find_speaker_block_by_text_idx(real_idx)
        if current_block is None:
            print(f"[Live] Cannot find block for text_idx {real_idx}")
            return
        
        current_speaker = current_block['speaker']
        print(f"[Live] Current speaker: {current_speaker}, block range: {current_block['start']}-{current_block['end']}")
        
        if direction == 'prev':
            # Tìm block trước
            prev_block = self._find_prev_speaker_block(current_block['start'])
            if prev_block:
                print(f"[Live] Prev speaker: {prev_block['speaker']}, range: {prev_block['start']}-{prev_block['end']}")
                # Gộp current block vào prev block
                self._merge_blocks(prev_block, current_block)
                self._update_display()
                print(f"[Live] Merged into prev speaker: {prev_block['speaker']}")
            else:
                print("[Live] No prev block found")
                
        elif direction == 'next':
            # Tìm block sau
            next_block = self._find_next_speaker_block(current_block['end'])
            if next_block:
                print(f"[Live] Next speaker: {next_block['speaker']}, range: {next_block['start']}-{next_block['end']}")
                # Gộp current block vào next block
                self._merge_blocks(next_block, current_block)
                self._update_display()
                print(f"[Live] Merged into next speaker: {next_block['speaker']}")
            else:
                print("[Live] No next block found")
    
    def _update_speaker_for_segments_range(self, start_idx, end_idx, new_speaker):
        """Cập nhật speaker cho các segments trong khoảng [start_idx, end_idx) trong clickable_segments"""
        # Trong clickable_segments: speaker, text, text, speaker, text, ...
        # Cần tìm speaker block chứa start_idx và cập nhật nó
        
        text_seg_count = 0
        current_speaker_idx = -1
        speaker_idx = -1
        
        # Tìm speaker block của start_idx
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') == 'speaker':
                speaker_idx = i
            elif seg.get('type') == 'text':
                if text_seg_count == start_idx:
                    current_speaker_idx = speaker_idx
                    break
                text_seg_count += 1
        
        # Cập nhật speaker block
        if current_speaker_idx >= 0:
            self.clickable_segments[current_speaker_idx]['text'] = f"__SPK_SEP__{new_speaker}__SPK_SEP__"
            print(f"[Live] Updated speaker block at {current_speaker_idx} to: {new_speaker}")
        
        # Cập nhật speaker trong các text segments
        text_seg_count = 0
        updated_count = 0
        for seg in self.clickable_segments:
            if seg.get('type') == 'text':
                if start_idx <= text_seg_count < end_idx:
                    seg['speaker'] = new_speaker
                    updated_count += 1
                elif text_seg_count >= end_idx:
                    break
                text_seg_count += 1
        
        print(f"[Live] Updated {updated_count} text segments to speaker: {new_speaker}")
    
    def _remove_speaker_separator_between(self, speaker1, speaker2):
        """Xóa speaker separator giữa hai ngườii nói khi gộp"""
        # Tìm speaker1 trong danh sách, sau đó xóa speaker2 ngay sau nó
        found_speaker1 = False
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') == 'speaker':
                seg_text = seg.get('text', '').replace('__SPK_SEP__', '').strip()
                if found_speaker1 and seg_text == speaker2:
                    # Xóa speaker separator của speaker2 (ngay sau speaker1)
                    del self.clickable_segments[i]
                    print(f"[Live] Removed separator for {speaker2} at index {i}")
                    return True
                elif seg_text == speaker1:
                    found_speaker1 = True
        return False
    
    def _find_speaker_block_by_text_idx(self, text_idx):
        """Tìm speaker block chứa text segment index"""
        text_count = 0
        current_block = None
        
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') == 'speaker':
                if current_block:
                    current_block['end'] = i  # end exclusive
                speaker_name = seg.get('text', '').replace('__SPK_SEP__', '').strip()
                current_block = {
                    'speaker': speaker_name,
                    'start': i,
                    'end': None,
                    'text_segments': []
                }
            elif seg.get('type') == 'text':
                if text_count == text_idx and current_block:
                    # Tìm thấy, ghi nhận tất cả text segments trong block này
                    result = current_block.copy()
                    # Tìm tất cả text segments trong block
                    for j in range(current_block['start'] + 1, len(self.clickable_segments)):
                        if self.clickable_segments[j].get('type') == 'speaker':
                            break
                        if self.clickable_segments[j].get('type') == 'text':
                            result['text_segments'].append(j)
                    result['end'] = result['text_segments'][-1] + 1 if result['text_segments'] else current_block['start'] + 1
                    return result
                text_count += 1
        
        return None
    
    def _find_prev_speaker_block(self, before_idx):
        """Tìm speaker block trước index"""
        # Tìm speaker gần nhất trước before_idx
        prev_speaker_idx = None
        for i in range(before_idx - 1, -1, -1):
            if self.clickable_segments[i].get('type') == 'speaker':
                prev_speaker_idx = i
                break
        
        if prev_speaker_idx is None:
            return None
        
        speaker_name = self.clickable_segments[prev_speaker_idx].get('text', '').replace('__SPK_SEP__', '').strip()
        text_segments = []
        
        # Tìm tất cả text segments trong block này
        for j in range(prev_speaker_idx + 1, len(self.clickable_segments)):
            if self.clickable_segments[j].get('type') == 'speaker':
                break
            if self.clickable_segments[j].get('type') == 'text':
                text_segments.append(j)
        
        return {
            'speaker': speaker_name,
            'start': prev_speaker_idx,
            'end': text_segments[-1] + 1 if text_segments else prev_speaker_idx + 1,
            'text_segments': text_segments
        }
    
    def _find_next_speaker_block(self, after_idx):
        """Tìm speaker block sau index"""
        # Tìm speaker gần nhất sau after_idx
        next_speaker_idx = None
        for i in range(after_idx, len(self.clickable_segments)):
            if self.clickable_segments[i].get('type') == 'speaker':
                next_speaker_idx = i
                break
        
        if next_speaker_idx is None:
            return None
        
        speaker_name = self.clickable_segments[next_speaker_idx].get('text', '').replace('__SPK_SEP__', '').strip()
        text_segments = []
        
        # Tìm tất cả text segments trong block này
        for j in range(next_speaker_idx + 1, len(self.clickable_segments)):
            if self.clickable_segments[j].get('type') == 'speaker':
                break
            if self.clickable_segments[j].get('type') == 'text':
                text_segments.append(j)
        
        return {
            'speaker': speaker_name,
            'start': next_speaker_idx,
            'end': text_segments[-1] + 1 if text_segments else next_speaker_idx + 1,
            'text_segments': text_segments
        }
    
    def _merge_blocks(self, target_block, source_block):
        """Gộp source_block vào target_block"""
        print(f"[Live] Merging {source_block['speaker']} into {target_block['speaker']}")
        
        # Xác định block nào ở trước, block nào ở sau
        if source_block['start'] < target_block['start']:
            first_block, second_block = source_block, target_block
        else:
            first_block, second_block = target_block, source_block
        
        print(f"[Live] First block: {first_block['speaker']} range {first_block['start']}-{first_block['end']}")
        print(f"[Live] Second block: {second_block['speaker']} range {second_block['start']}-{second_block['end']}")
        
        # Gộp text và partials theo đúng thứ tự thờigian
        merged_texts = []
        merged_partials = []
        
        # Từ block đầu tiên
        for idx in first_block['text_segments']:
            seg = self.clickable_segments[idx]
            merged_texts.append(seg.get('text', ''))
            for p in seg.get('partials', []):
                merged_partials.append(p.copy())
        
        # Từ block thứ hai
        for idx in second_block['text_segments']:
            seg = self.clickable_segments[idx]
            merged_texts.append(seg.get('text', ''))
            for p in seg.get('partials', []):
                merged_partials.append(p.copy())
        
        # Tạo segment mới đã gộp
        merged_seg = {
            'type': 'text',
            'text': ' '.join(merged_texts),
            'start_time': self.clickable_segments[first_block['text_segments'][0]].get('start_time', 0) if first_block['text_segments'] else 0,
            'segment_id': self._get_next_segment_id(),
            'partials': merged_partials
        }
        
        # Xóa tất cả text segments và speaker của cả 2 block
        # Xóa từ cuối đến đầu để index không bị thay đổi
        all_indices = (first_block['text_segments'] + [first_block['start']] + 
                      second_block['text_segments'] + [second_block['start']])
        
        # Sắp xếp giảm dần để xóa an toàn
        for idx in sorted(all_indices, reverse=True):
            if 0 <= idx < len(self.clickable_segments):
                del self.clickable_segments[idx]
        
        # Chèn segment mới và speaker (dùng tên của target_block)
        insert_pos = min(first_block['start'], second_block['start'])
        self.clickable_segments.insert(insert_pos, {'type': 'speaker', 'text': f"__SPK_SEP__{target_block['speaker']}__SPK_SEP__", 'segment_id': -1})
        self.clickable_segments.insert(insert_pos + 1, merged_seg)
        
        print(f"[Live] Created merged segment with {len(merged_partials)} partials at position {insert_pos}")
    
    def _convert_clickable_to_segments(self):
        """Chuyển đổi clickable_segments sang format segments tương thích với tab_file"""
        segments = []
        current_speaker = 'Ngườii nói 1'
        seg_idx = 0
        
        for seg in self.clickable_segments:
            if seg.get('type') == 'speaker':
                current_speaker = seg['text'].replace('__SPK_SEP__', '').strip()
            elif seg.get('type') == 'text':
                text = seg.get('text', '')
                if text:
                    segments.append({
                        'speaker': current_speaker,
                        'speaker_id': current_speaker,
                        'text': text,
                        'index': seg_idx
                    })
                    seg_idx += 1
            elif seg.get('type') == 'partial':
                text = seg.get('text', '')
                if text:
                    segments.append({
                        'speaker': current_speaker,
                        'speaker_id': current_speaker,
                        'text': text,
                        'index': seg_idx
                    })
                    seg_idx += 1
        
        return segments
    
    def _get_speaker_id_from_segment(self, segment):
        """Lấy speaker_id từ segment, ưu tiên trường speaker_id nếu có"""
        if 'speaker_id' in segment:
            return str(segment['speaker_id'])
        speaker_name = segment.get('speaker', '')
        match = re.search(r'(\d+)$', str(speaker_name))
        return match.group(1) if match else speaker_name
    
    def _find_block_end(self, sentence_idx):
        """Tìm index kết thúc của block chứa sentence_idx"""
        segments = self._convert_clickable_to_segments()
        if not segments or sentence_idx >= len(segments):
            return len(segments) if segments else 0
        
        current_speaker = segments[sentence_idx].get('speaker', 'Ngườii nói 1')
        
        for i in range(sentence_idx + 1, len(segments)):
            if segments[i].get('speaker', 'Ngườii nói 1') != current_speaker:
                return i
        
        return len(segments)
    
    def _anchor_id_to_segment_index(self, anchor_id):
        """Chuyển đổi anchor ID sang index trong danh sách text segments (partials)"""
        # FIX: Dùng _anchor_id được lưu trong partial để tìm
        text_seg_count = 0
        for seg in self.clickable_segments:
            if seg.get('type') == 'text':
                partials = seg.get('partials', [])
                if partials:
                    for chunk_idx, partial in enumerate(partials):
                        if partial.get('_anchor_id') == anchor_id:
                            return text_seg_count
                elif seg.get('_anchor_id') == anchor_id:
                    return text_seg_count
                text_seg_count += 1
            elif seg.get('type') == 'partial':
                if seg.get('_anchor_id') == anchor_id:
                    return text_seg_count
                text_seg_count += 1
        return None
    
    def _anchor_id_to_clickable_index(self, anchor_id):
        """Chuyển đổi anchor ID sang index TRỰC TIẾP trong clickable_segments"""
        # Trả về index của partial trong clickable_segments để chèn speaker đúng vị trí
        
        # FIX: Trích xuất segment_id và chunk_idx từ anchor_id
        if anchor_id < 1000000:
            return None
        
        adjusted = anchor_id - 1000000
        target_seg_id = adjusted // 1000
        target_chunk_idx = adjusted % 1000
        
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') == 'text':
                seg_id = seg.get('segment_id', 0)
                partials = seg.get('partials', [])
                
                if seg_id == target_seg_id:
                    # Tìm đúng segment
                    if partials and target_chunk_idx < len(partials):
                        return i
                    elif not partials and target_chunk_idx == 0:
                        return i
            elif seg.get('type') == 'partial':
                seg_id = seg.get('segment_id', 0)
                if seg_id == target_seg_id and target_chunk_idx == 0:
                    return i
        
        return None
    
    def _find_segment_and_chunk(self, anchor_id):
        """Tìm vị trí segment và chunk dựa vào anchor_id"""
        # FIX: Dùng _anchor_id được lưu trong partial để tìm
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') == 'text':
                partials = seg.get('partials', [])
                if partials:
                    for chunk_idx, partial in enumerate(partials):
                        if partial.get('_anchor_id') == anchor_id:
                            return (i, chunk_idx, 'text')
                elif seg.get('_anchor_id') == anchor_id:
                    return (i, 0, 'text')
            elif seg.get('type') == 'partial':
                if seg.get('_anchor_id') == anchor_id:
                    return (i, 0, 'partial')
        return None
    
    def _get_next_segment_id(self):
        """Lấy segment_id tiếp theo (max hiện tại + 1)"""
        max_id = 0
        for seg in self.clickable_segments:
            if seg.get('type') == 'text':
                seg_id = seg.get('segment_id', 0)
                if isinstance(seg_id, int) and seg_id > max_id:
                    max_id = seg_id
        return max_id + 1
    
    def _split_text_segment_at_chunk(self, clickable_idx, chunk_idx):
        """Tách segment text thành 2 phần tại chunk_idx"""
        seg = self.clickable_segments[clickable_idx]
        partials = seg.get('partials', [])
        
        if not partials or chunk_idx >= len(partials):
            return
        
        # Tạo segment mới chứa các chunk từ chunk_idx trở đi
        new_partials = partials[chunk_idx:]
        
        # FIX: Lấy max segment_id hiện tại + 1 để đảm bảo không trùng
        next_id = self._get_next_segment_id()
        
        new_seg = {
            'type': 'text',
            'text': ' '.join([p.get('text', '') for p in new_partials]),
            'partials': new_partials,
            'segment_id': next_id,  # ID mới duy nhất
            'start_time': new_partials[0].get('timestamp', 0) if new_partials else 0
        }
        
        # Cập nhật segment cũ chỉ chứa các chunk trước chunk_idx
        old_partials = partials[:chunk_idx]
        seg['partials'] = old_partials
        seg['text'] = ' '.join([p.get('text', '') for p in old_partials])
        
        # Chèn segment mới sau segment cũ
        self.clickable_segments.insert(clickable_idx + 1, new_seg)
    
    def _split_text_segment_for_single_chunk(self, clickable_idx, chunk_idx, new_speaker, old_speaker):
        """Tách segment thành 3 phần: trước chunk, chunk này, sau chunk"""
        seg = self.clickable_segments[clickable_idx]
        partials = seg.get('partials', [])
        
        if not partials or chunk_idx >= len(partials):
            return
        
        # FIX: Tính ID mới cho các segment sau khi tách
        # Xóa segment cũ trước để tính đúng max
        del self.clickable_segments[clickable_idx]
        
        # Lấy các ID mới liên tiếp (có thể cần tới 3 ID)
        next_id = self._get_next_segment_id()
        
        new_segments = []
        
        # Phần 1: Các chunk trước chunk_idx
        if chunk_idx > 0:
            pre_partials = partials[:chunk_idx]
            pre_seg = {
                'type': 'text',
                'text': ' '.join([p.get('text', '') for p in pre_partials]),
                'partials': pre_partials,
                'segment_id': next_id,
                'start_time': pre_partials[0].get('timestamp', 0) if pre_partials else 0
            }
            new_segments.append(pre_seg)
            next_id += 1
        
        # Phần 2: Chunk này (thuộc người nói mới)
        target_partial = partials[chunk_idx]
        mid_seg = {
            'type': 'text',
            'text': target_partial.get('text', ''),
            'partials': [target_partial],
            'segment_id': next_id,
            'start_time': target_partial.get('timestamp', 0)
        }
        
        if chunk_idx > 0:
            new_segments.append({'type': 'speaker', 'text': f"__SPK_SEP__{new_speaker}__SPK_SEP__", 'segment_id': -1})
        
        new_segments.append(mid_seg)
        next_id += 1
        
        # Phần 3: Các chunk sau chunk_idx (thuộc ngườii nói cũ)
        if chunk_idx < len(partials) - 1:
            post_partials = partials[chunk_idx + 1:]
            post_seg = {
                'type': 'text',
                'text': ' '.join([p.get('text', '') for p in post_partials]),
                'partials': post_partials,
                'segment_id': next_id,
                'start_time': post_partials[0].get('timestamp', 0) if post_partials else 0
            }
            new_segments.append({'type': 'speaker', 'text': f"__SPK_SEP__{old_speaker}__SPK_SEP__", 'segment_id': -1})
            new_segments.append(post_seg)
        
        # Chèn tất cả segment mới vào vị trí cũ
        for j, new_seg in enumerate(new_segments):
            self.clickable_segments.insert(clickable_idx + j, new_seg)
    
    def _find_clickable_index_by_sentence_idx(self, sentence_idx):
        """Tìm index trong clickable_segments tương ứng với sentence_idx"""
        # sentence_idx là index trong danh sách text segments (không tính speaker)
        # Cần tìm vị trí trong clickable_segments (có thể có speaker xen kẽ)
        
        text_seg_count = 0
        for i, seg in enumerate(self.clickable_segments):
            if seg.get('type') in ('text', 'partial'):
                if text_seg_count == sentence_idx:
                    return i
                text_seg_count += 1
        
        # Nếu không tìm thấy, trả về cuối danh sách
        return len(self.clickable_segments)
    # ==================== SEARCH METHODS ====================
    
    def _map_norm_to_orig(self, original, norm_idx):
        """Ánh xạ vị trí từ normalized text (không dấu) sang original text.
        
        Args:
            original: Text gốc có dấu
            norm_idx: Vị trí trong normalized text
        
        Returns:
            Vị trí tương ứng trong original text
        """
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
    
    def perform_search(self, query):
        """Thực hiện tìm kiếm trong clickable_segments - tìm trong toàn bộ text"""
        if not hasattr(self, 'clickable_segments') or not self.clickable_segments:
            return
        
        if not query or not query.strip():
            self.clear_search()
            return
            
        self.last_query = query
        self.search_matches = []
        self.current_search_index = -1
        
        query_norm = normalize_vietnamese(query)
        query_lower = query.lower()
        
        # Tìm kiếm trong tất cả các text segments
        text_seg_idx = 0  # Index chỉ tính text segments (không tính speaker)
        for seg in self.clickable_segments:
            if seg.get('type') != 'text':
                continue
            
            # Lấy text tổng hợp của segment
            full_text = seg.get('text', '')
            if not full_text:
                text_seg_idx += 1
                continue
            
            # Tìm kiếm chính xác trong toàn bộ text
            text_lower = full_text.lower()
            start = 0
            while True:
                idx = text_lower.find(query_lower, start)
                if idx == -1:
                    break
                self.search_matches.append({
                    'seg_idx': text_seg_idx,
                    'clickable_idx': self.clickable_segments.index(seg),
                    'start': idx,
                    'end': idx + len(query),
                    'text': full_text[idx:idx + len(query)],
                    'score': 1.0
                })
                start = idx + 1
            
            # Tìm kiếm không dấu (fuzzy search) trong toàn bộ text
            text_norm = normalize_vietnamese(full_text)
            start = 0
            while True:
                idx = text_norm.find(query_norm, start)
                if idx == -1:
                    break
                
                # Ánh xạ vị trí từ normalized sang original
                orig_start = self._map_norm_to_orig(full_text, idx)
                orig_end = self._map_norm_to_orig(full_text, idx + len(query_norm))
                
                # Kiểm tra trùng lặp
                is_duplicate = False
                for existing in self.search_matches:
                    if existing['seg_idx'] == text_seg_idx and \
                       abs(existing['start'] - orig_start) < 2:
                        is_duplicate = True
                        break
                
                if not is_duplicate and orig_start < len(full_text):
                    end_pos = min(orig_end, len(full_text))
                    self.search_matches.append({
                        'seg_idx': text_seg_idx,
                        'clickable_idx': self.clickable_segments.index(seg),
                        'start': orig_start,
                        'end': end_pos,
                        'text': full_text[orig_start:end_pos],
                        'score': 0.9
                    })
                start = idx + 1
            
            text_seg_idx += 1
        
        # Sắp xếp kết quả theo thứ tự xuất hiện
        self.search_matches.sort(key=lambda x: (x['seg_idx'], x['start']))
        
        count = len(self.search_matches)
        self.search_widget.label_count.setText(f"0/{count}")
        
        if count > 0:
            self.current_search_index = 0
            self.navigate_search(0)
        else:
            self._update_display_with_timestamps()
    
    def navigate_search(self, direction):
        """Di chuyển đến kết quả tìm kiếm tiếp theo/trước đó"""
        if not self.search_matches:
            return
            
        self.current_search_index = (self.current_search_index + direction) % len(self.search_matches)
        
        self.search_widget.label_count.setText(f"{self.current_search_index + 1}/{len(self.search_matches)}")
        
        # Cập nhật hiển thị với highlight
        self._update_display_with_timestamps()
        
        # Scroll đến segment có kết quả
        match = self.search_matches[self.current_search_index]
        QTimer.singleShot(10, lambda: self.scroll_to_segment(match['seg_idx']))
    
    def clear_search(self):
        """Xóa kết quả tìm kiếm"""
        if hasattr(self, 'search_widget'):
            self.search_widget.input.clear()
            self.search_widget.label_count.setText("0/0")
        self.search_matches = []
        self.current_search_index = -1
        self._update_display_with_timestamps()
    
    def scroll_to_segment(self, seg_idx):
        """Scroll đến segment có index tương ứng"""
        # Tìm anchor tương ứng với text segment
        text_count = 0
        for seg in self.clickable_segments:
            if seg.get('type') == 'text':
                if text_count == seg_idx:
                    # Tìm anchor của segment này (dùng segment_id)
                    seg_id = seg.get('segment_id', 0)
                    anchor_name = f"seg_{seg_id}"
                    self.text_output.scrollToAnchor(anchor_name)
                    return
                text_count += 1
    
    def rename_live_speaker(self, old_name, new_name):
        """Rename all occurrences of old_name to new_name in transcribed text"""
        if not self.transcribed_text:
            return
            
        old_token = f"__SPK_SEP__{old_name}__SPK_SEP__"
        new_token = f"__SPK_SEP__{new_name}__SPK_SEP__"
        
        self.transcribed_text = self.transcribed_text.replace(old_token, new_token)
        
        if self.pending_speaker_preview == old_name:
            self.pending_speaker_preview = new_name
        
        print(f"[Live] Renamed speaker {old_name} -> {new_name}")
