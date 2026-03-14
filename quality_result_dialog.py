# -*- coding: utf-8 -*-
"""
quality_result_dialog.py - Dialog hiển thị kết quả phân tích chất lượng âm thanh
Theme: Đơn giản, phông đen, chữ trắng (giống giao diện Tab)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTextEdit, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core.audio_analyzer import (
    AudioQualityAnalyzer, AnalysisResult, QualityMetrics,
    check_dnsmos_model_exists, download_dnsmos_model_sync,
    DNSMOSDownloader
)


# Color scheme
BG_DARK = '#2b2b2b'
TEXT_PRIMARY = '#ffffff'
TEXT_SECONDARY = '#cccccc'
TEXT_MUTED = '#888888'

# Màu cho các mức chất lượng
COLOR_GOOD = '#4ade80'      # Xanh lá
COLOR_OK = '#fbbf24'        # Vàng
COLOR_MEDIUM = '#fb923c'    # Cam
COLOR_BAD = '#f87171'       # Đỏ


def get_dnsmos_color(score: float) -> str:
    if score >= 4.0:
        return COLOR_GOOD
    elif score >= 3.0:
        return COLOR_OK
    elif score >= 2.0:
        return COLOR_MEDIUM
    else:
        return COLOR_BAD


def get_dnsmos_label(score: float) -> str:
    if score >= 4.0:
        return "Tốt"
    elif score >= 3.0:
        return "Khá"
    elif score >= 2.0:
        return "Trung bình"
    else:
        return "Kém"


def get_confidence_color(confidence: float) -> str:
    if confidence >= 0.85:
        return COLOR_GOOD
    elif confidence >= 0.75:
        return COLOR_OK
    elif confidence >= 0.60:
        return COLOR_MEDIUM
    else:
        return COLOR_BAD


def get_confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "Xuất sắc"
    elif confidence >= 0.75:
        return "Tốt"
    elif confidence >= 0.60:
        return "Trung bình"
    else:
        return "Kém"


class QualityResultDialog(QDialog):
    """Dialog hiển thị kết quả phân tích chất lượng âm thanh"""
    
    def __init__(self, result: AnalysisResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Kết quả đánh giá chất lượng âm thanh")
        self.setMinimumWidth(420)
        self.setMinimumHeight(350)
        self.setup_ui()
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {BG_DARK}; }}
            QLabel {{ color: {TEXT_PRIMARY}; background-color: transparent; }}
            QPushButton {{
                background-color: #3a3a3a;
                color: {TEXT_PRIMARY};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: #555555; }}
            QTextEdit {{
                background-color: #363636;
                color: {TEXT_PRIMARY};
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }}
        """)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header
        self._setup_header(layout)
        
        # Metrics
        self._setup_metrics(layout)
        
        # Sample text
        self._setup_sample_text(layout)
        
        # Suggestions
        self._setup_suggestions(layout)
        
        # Buttons
        self._setup_buttons(layout)
    
    def _setup_header(self, layout):
        hbox = QHBoxLayout()
        
        if self.result.is_ready:
            icon = "✓"
            status = "Sẵn sàng cho nhận dạng"
            color = COLOR_GOOD
        else:
            icon = "⚠"
            status = "Cần cải thiện chất lượng"
            color = COLOR_BAD
        
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(f"font-size: 18px; color: {color};")
        hbox.addWidget(lbl_icon)
        
        vbox = QVBoxLayout()
        lbl_status = QLabel(status)
        lbl_status.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        vbox.addWidget(lbl_status)
        
        if self.result.metrics.duration_analyzed > 0:
            lbl_info = QLabel(f"Đã phân tích: {self.result.metrics.duration_analyzed:.1f}s, {self.result.metrics.num_segments} đoạn")
            lbl_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
            vbox.addWidget(lbl_info)
        
        hbox.addLayout(vbox)
        hbox.addStretch()
        layout.addLayout(hbox)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: #444444;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)
    
    def _setup_metrics(self, layout):
        # Title
        lbl_title = QLabel("Chất lượng âm thanh (DNSMOS)")
        lbl_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
        layout.addWidget(lbl_title)
        
        # SIG
        sig_color = get_dnsmos_color(self.result.metrics.dnsmos_sig)
        sig_label = get_dnsmos_label(self.result.metrics.dnsmos_sig)
        hbox_sig = QHBoxLayout()
        hbox_sig.addWidget(QLabel("SIG - Chất lượng giọng nói:"))
        hbox_sig.addWidget(QLabel(f"{self.result.metrics.dnsmos_sig:.2f}/5"))
        lbl_sig_status = QLabel(f"({sig_label})")
        lbl_sig_status.setStyleSheet(f"color: {sig_color}; font-weight: bold;")
        hbox_sig.addWidget(lbl_sig_status)
        hbox_sig.addStretch()
        layout.addLayout(hbox_sig)
        
        # BAK
        bak_color = get_dnsmos_color(self.result.metrics.dnsmos_bak)
        bak_label = get_dnsmos_label(self.result.metrics.dnsmos_bak)
        hbox_bak = QHBoxLayout()
        hbox_bak.addWidget(QLabel("BAK - Chất lượng nhiễu, vang nền:"))
        hbox_bak.addWidget(QLabel(f"{self.result.metrics.dnsmos_bak:.2f}/5"))
        lbl_bak_status = QLabel(f"({bak_label})")
        lbl_bak_status.setStyleSheet(f"color: {bak_color}; font-weight: bold;")
        hbox_bak.addWidget(lbl_bak_status)
        hbox_bak.addStretch()
        layout.addLayout(hbox_bak)
        
        # OVRL
        ovrl_color = get_dnsmos_color(self.result.metrics.dnsmos_ovrl)
        ovrl_label = get_dnsmos_label(self.result.metrics.dnsmos_ovrl)
        hbox_ovrl = QHBoxLayout()
        hbox_ovrl.addWidget(QLabel("OVRL - Tổng thể:"))
        hbox_ovrl.addWidget(QLabel(f"{self.result.metrics.dnsmos_ovrl:.2f}/5"))
        lbl_ovrl_status = QLabel(f"({ovrl_label})")
        lbl_ovrl_status.setStyleSheet(f"color: {ovrl_color}; font-weight: bold;")
        hbox_ovrl.addWidget(lbl_ovrl_status)
        hbox_ovrl.addStretch()
        layout.addLayout(hbox_ovrl)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: #444444;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)
        
        # ASR Confidence
        if self.result.metrics.asr_confidence > 0:
            conf_color = get_confidence_color(self.result.metrics.asr_confidence)
            conf_label = get_confidence_label(self.result.metrics.asr_confidence)
            hbox_asr = QHBoxLayout()
            hbox_asr.addWidget(QLabel("ASRProxy - Độ tự tin nhận dạng:"))
            hbox_asr.addWidget(QLabel(f"{self.result.metrics.asr_confidence:.1%}"))
            lbl_asr_status = QLabel(f"({conf_label})")
            lbl_asr_status.setStyleSheet(f"color: {conf_color}; font-weight: bold;")
            hbox_asr.addWidget(lbl_asr_status)
            hbox_asr.addStretch()
            layout.addLayout(hbox_asr)
    
    def _setup_sample_text(self, layout):
        if self.result.metrics.sample_text:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"background-color: #444444;")
            sep.setFixedHeight(1)
            layout.addWidget(sep)
            
            lbl_title = QLabel("Chữ nhận dạng được:")
            lbl_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
            layout.addWidget(lbl_title)
            
            text_edit = QTextEdit()
            text_edit.setPlainText(self.result.metrics.sample_text)
            text_edit.setReadOnly(True)
            text_edit.setMaximumHeight(60)
            layout.addWidget(text_edit)
    
    def _setup_suggestions(self, layout):
        if self.result.suggestions:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"background-color: #444444;")
            sep.setFixedHeight(1)
            layout.addWidget(sep)
            
            lbl_title = QLabel("Gợi ý cải thiện:")
            lbl_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
            layout.addWidget(lbl_title)
            
            for suggestion in self.result.suggestions:
                clean = suggestion
                if clean.startswith(('🟡', '🔴', '🟢')):
                    clean = clean[2:].strip()
                lbl = QLabel(f"• {clean}")
                lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
                lbl.setWordWrap(True)
                layout.addWidget(lbl)
    
    def _setup_buttons(self, layout):
        hbox = QHBoxLayout()
        hbox.addStretch()
        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(self.accept)
        hbox.addWidget(btn_close)
        layout.addLayout(hbox)


class MicTestDialog(QDialog):
    """Dialog test microphone"""
    
    recording_finished = pyqtSignal(bytes)
    
    def __init__(self, analyzer: AudioQualityAnalyzer, device_name: str, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.device_name = device_name
        self.recorded_audio = None
        self.setWindowTitle("Đánh giá Microphone")
        self.setMinimumWidth(380)
        self.setMinimumHeight(250)
        self.setup_ui()
        self.apply_dark_theme()
    
    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {BG_DARK}; }}
            QLabel {{ color: {TEXT_PRIMARY}; background-color: transparent; }}
            QPushButton {{
                background-color: #3a3a3a;
                color: {TEXT_PRIMARY};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: #555555; }}
            QPushButton:disabled {{ background-color: #444444; color: #888888; }}
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                height: 18px;
                background-color: #3a3a3a;
                color: {TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{ background-color: {COLOR_OK}; border-radius: 3px; }}
        """)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        lbl_instruction = QLabel(
            "Vui lòng đứng tại vị trí phát biểu và nói nội dung bất kỳ\n"
            "(ví dụ: 'Một hai ba bốn năm' hoặc đoạn văn ngắn)\n\n"
            "Thời gian ghi âm: 8-10 giây"
        )
        lbl_instruction.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; line-height: 1.5;")
        lbl_instruction.setWordWrap(True)
        layout.addWidget(lbl_instruction)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Sẵn sàng")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self.lbl_status)
        
        hbox = QHBoxLayout()
        hbox.addStretch()
        self.btn_start = QPushButton("Bắt đầu ghi âm")
        self.btn_start.clicked.connect(self.start_recording)
        hbox.addWidget(self.btn_start)
        self.btn_cancel = QPushButton("Hủy")
        self.btn_cancel.clicked.connect(self.reject)
        hbox.addWidget(self.btn_cancel)
        hbox.addStretch()
        layout.addLayout(hbox)
        
        self.recording_thread = None
    
    def start_recording(self):
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Đang ghi âm...")
        self.lbl_status.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        
        self.recording_thread = RecordingThread(self.device_name, duration=10)  # 10s cho DNSMOS chính xác
        self.recording_thread.progress.connect(self.update_progress)
        self.recording_thread.finished.connect(self.on_recording_finished)
        self.recording_thread.error.connect(self.on_recording_error)
        self.recording_thread.start()
    
    def update_progress(self, percent):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(f"Đang ghi âm... {percent}%")
    
    def on_recording_finished(self, audio_data):
        self.recorded_audio = audio_data
        self.accept()
    
    def on_recording_error(self, error_msg):
        self.lbl_status.setText(f"Lỗi: {error_msg}")
        self.lbl_status.setStyleSheet(f"color: {COLOR_BAD}; font-size: 12px;")
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Lỗi", f"Không thể ghi âm:\n{error_msg}")
    
    def get_audio(self):
        return self.recorded_audio


class RecordingThread(QThread):
    """Thread ghi âm không block UI - tìm device theo tên"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bytes)
    error = pyqtSignal(str)
    
    def __init__(self, device_name: str, duration: int = 8):
        super().__init__()
        self.device_name = device_name
        self.duration = duration
    
    def _find_device_index(self):
        """Tìm sounddevice index từ device name"""
        import sounddevice as sd
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and self.device_name in d['name']:
                return i
        # Fallback: default device
        return sd.default.device[0] if sd.default.device[0] is not None else 0
    
    def run(self):
        try:
            import sounddevice as sd
            import numpy as np
            
            # Tìm đúng device index
            device_index = self._find_device_index()
            print(f"[RecordingThread] Using device index {device_index} for '{self.device_name}'")
            
            sample_rate = 16000
            channels = 1
            recorded = []
            
            def callback(indata, frames, time_info, status):
                recorded.append(indata.copy())
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype='float32',
                device=device_index,
                callback=callback
            ):
                for i in range(self.duration * 10):
                    self.msleep(100)
                    progress = int((i + 1) / (self.duration * 10) * 100)
                    self.progress.emit(progress)
            
            if recorded:
                audio = np.concatenate(recorded, axis=0)
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                self.finished.emit(audio_bytes)
            else:
                self.error.emit("Không ghi được dữ liệu")
        except Exception as e:
            self.error.emit(str(e))


# Test
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    result = AnalysisResult(
        metrics=QualityMetrics(
            dnsmos_sig=3.65,
            dnsmos_bak=2.17,
            dnsmos_ovrl=2.41,
            asr_confidence=0.745,
            sample_text="Một hai ba bốn năm sáu bảy tám",
            duration_analyzed=5.5,
            num_segments=3
        ),
        suggestions=[
            "Có nhiễu nền: Cố gắng giảm âm thanh xung quanh",
            "ASR có thể sai sót: Kiểm tra kết quả sau khi nhận dạng"
        ],
        is_ready=True
    )
    
    dialog = QualityResultDialog(result)
    dialog.exec()
