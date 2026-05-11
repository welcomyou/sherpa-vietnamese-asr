# app.py - Main entry point
import sys
import os

# ─── Splash screen NGAY TỪ ĐẦU (Win32 ctypes, zero deps) ───
# Dùng Win32 API qua ctypes:
# 1. ctypes là stdlib → load <10ms, không cần đợi numpy/ORT/sherpa (~1-2s)
# 2. Không xung đột DLL với onnxruntime (khác Qt6)
# 3. WS_EX_TOPMOST → luôn nổi trên các cửa sổ khác
# Module splash_win32 ở root (KHÔNG nằm trong core/ vì core/__init__.py
# sẽ trigger import nặng).
_splash_ok = False
if __name__ == "__main__" and sys.platform == "win32":
    try:
        import splash_win32
        _splash_ok = splash_win32.show()
    except Exception as _e:
        print(f"[Splash] Import failed: {_e}")

# ─── Setup logging sớm nhất có thể (không import Qt6/ORT) ───
from core.log_config import setup_logging
setup_logging("desktop")

# ─── Import numpy + ORT + sherpa TRƯỚC Qt6 (Windows DLL load order) ───
# Qt6 load một số DLL làm ORT initialization routine fail nếu ORT load SAU Qt6.
# Thứ tự đúng: numpy → onnxruntime → sherpa_onnx → PyQt6
import numpy

try:
    import onnxruntime as _ort
    print(f"[Init] onnxruntime {_ort.__version__}")
except ImportError as e:
    print(f"[Init] onnxruntime not available: {e}")

try:
    import sherpa_onnx
    print(f"[Init] sherpa_onnx loaded (version: {sherpa_onnx.__version__ if hasattr(sherpa_onnx, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"[Init] sherpa_onnx not available: {e}")

# Pump message queue để splash repaint sau khi ORT load xong (~1-2s)
if _splash_ok:
    try:
        splash_win32.pump()
    except Exception:
        pass

if __name__ == "__main__":
    os.environ["QT_MEDIA_BACKEND"] = "windows"
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    _app = QApplication(sys.argv)

import configparser

# === Cleanup temporary files from previous crashed sessions ===
def cleanup_temp_files():
    """Xóa các file tạm asr_* trong thư mục temp của hệ thống."""
    import tempfile

    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('asr_'):
            try:
                filepath = os.path.join(temp_dir, filename)
                # Chỉ xóa regular files, bỏ qua symlinks và directories
                if os.path.isfile(filepath) and not os.path.islink(filepath):
                    os.unlink(filepath)
                    print(f"[Cleanup] Deleted: {filepath}")
            except OSError:
                pass

# Use centralized CPU detection from core
from core.config import ALLOWED_THREADS, DEFAULT_THREADS
print(f"[Init] CPU: {DEFAULT_THREADS} physical cores, {ALLOWED_THREADS} logical threads")

# Set OpenMP/MKL threads to match allowed cores to prevent over-subscription
os.environ["OMP_NUM_THREADS"] = str(ALLOWED_THREADS)
os.environ["MKL_NUM_THREADS"] = str(ALLOWED_THREADS)

if "QT_MEDIA_BACKEND" not in os.environ:
    os.environ["QT_MEDIA_BACKEND"] = "windows"

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent

from core.config import BASE_DIR, CONFIG_FILE, COLORS, ALLOWED_THREADS, apply_theme
from tab_file import FileProcessingTab
from tab_live import LiveProcessingTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sherpa-vietnamese-asr")
        self.resize(950, 750)
        self.center_on_screen()
        
        # Flag to prevent saving config during initialization
        self._applying_config = False
        
        # Load config
        self.config = self.load_config()

        # Áp dụng theme TRƯỚC init_ui — stylesheet đọc COLORS tại lúc tạo widget
        theme = self.config['Appearance'].get('theme', 'dark') if 'Appearance' in self.config else 'dark'
        apply_theme(theme)

        self.init_ui()
        self.apply_config()

    def center_on_screen(self):
        """Đặt cửa sổ ra giữa màn hình và đảm bảo không bị mất title bar"""
        try:
            screen = self.screen() or QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                window_geom = self.frameGeometry()
                window_geom.moveCenter(screen_geom.center())
                
                # Đảm bảo title bar không bị che khuất
                new_top_left = window_geom.topLeft()
                new_top_left.setY(max(screen_geom.top(), new_top_left.y()))
                self.move(new_top_left)
        except Exception as e:
            print(f"[Init] Could not center window: {e}")

    def keyPressEvent(self, event):
        """Handle global hotkeys"""
        if hasattr(self, 'main_tabs') and self.main_tabs.currentWidget() == self.tab_live:
            if hasattr(self.tab_live, 'is_recording') and self.tab_live.is_recording:
                key = event.text()
                # Check for number keys 1-9
                if key in [str(i) for i in range(1, 10)]:
                    name = self.tab_live.hotkey_config.get(key, "")
                    if name:
                        self.tab_live.insert_speaker_separator(name)
                        event.accept()
                        return
                        
        super().keyPressEvent(event)

    def load_config(self):
        """Load configuration from config.ini with separate sections for File and Live tabs"""
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE, encoding='utf-8')
        else:
            # Create default config with separate sections
            config['FileSettings'] = {
                'model': 'zipformer-30m-rnnt-6000h',
                'cpu_threads': '4',
                'punctuation_confidence_slider': '6',
                'case_confidence_slider': '6',
                'speaker_diarization': 'True',
                'show_speaker_labels': 'True',
                'num_speakers': '0',
                'speaker_model': 'community1_pure_ort',
                'save_ram': 'True',
                'auto_analyze_quality': 'True',
                'rms_normalize': 'False',
                'bypass_vad': 'False',
            }
            config['LiveSettings'] = {
                'model': 'zipformer-30m-rnnt-6000h',
                'cpu_threads': '4',
                'microphone_list': '',
                'selected_microphone': '',
            }
            config['Appearance'] = {
                'theme': 'dark',
            }
            self.save_config(config)

        # Migrate old config format (single Settings section) to new format
        if 'Settings' in config and 'FileSettings' not in config:
            old = config['Settings']
            config['FileSettings'] = {
                'model': old.get('model', 'a-little-better-model'),
                'cpu_threads': old.get('cpu_threads', '4'),
                'punctuation_confidence_slider': old.get('punctuation_confidence_slider', '7'),
                'case_confidence_slider': old.get('case_confidence_slider', '3'),
                'speaker_diarization': old.get('speaker_diarization', 'True'),
                'show_speaker_labels': old.get('show_speaker_labels', 'True'),
                'num_speakers': old.get('num_speakers', '0'),
                'speaker_model': old.get('speaker_model', 'community1_pure_ort'),
                'save_ram': old.get('save_ram', 'True'),
                'auto_analyze_quality': 'True',
                'rms_normalize': 'False',
                'bypass_vad': 'False',
            }
            config['LiveSettings'] = {
                'model': old.get('model', 'a-little-better-model'),
                'cpu_threads': old.get('cpu_threads', '4'),
                'microphone_list': '',
                'selected_microphone': '',
            }
            del config['Settings']
            self.save_config(config)
        
        # Ensure both sections exist
        if 'FileSettings' not in config:
            config['FileSettings'] = {
                'model': 'zipformer-30m-rnnt-6000h',
                'cpu_threads': '4',
                'punctuation_confidence_slider': '6',
                'case_confidence_slider': '6',
                'speaker_diarization': 'True',
                'show_speaker_labels': 'True',
                'num_speakers': '0',
                'speaker_model': 'community1_pure_ort',
                'save_ram': 'True',
                'bypass_vad': 'False',
            }
        if 'LiveSettings' not in config:
            config['LiveSettings'] = {
                'model': 'zipformer-30m-rnnt-6000h',
                'cpu_threads': '4',
                'microphone_list': '',
                'selected_microphone': '',
            }
        if 'Appearance' not in config:
            config['Appearance'] = {'theme': 'dark'}

        return config

    def save_config(self, config=None):
        """Save configuration to config.ini"""
        if config is None:
            config = self.config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            config.write(f)

    def apply_config(self):
        """Apply loaded configuration to UI - separate configs for File and Live tabs"""
        self._applying_config = True
        
        print(f"[Config] Applying config from {CONFIG_FILE}")
        _SENSITIVE_KEYS = {'hf_token', 'api_key', 'secret', 'password', 'token'}
        _safe = lambda section: {k: ('***' if k.lower() in _SENSITIVE_KEYS else v) for k, v in dict(self.config[section]).items()}
        print(f"[Config] FileSettings: {_safe('FileSettings')}")
        print(f"[Config] LiveSettings: {_safe('LiveSettings')}")
        
        try:
            # Apply File tab settings
            file_settings = self.config['FileSettings']
            file_tab = self.tab_file
            
            # Block signals to prevent saving during initialization
            widgets_to_block = [
                file_tab.combo_model, file_tab.slider_threads, file_tab.slider_punct_conf,
                file_tab.slider_case_conf, file_tab.check_speaker_diarization, file_tab.check_show_speaker_labels,
                file_tab.spin_num_speakers, file_tab.combo_speaker_model, file_tab.check_save_ram,
                file_tab.check_rms_normalize, file_tab.check_bypass_vad,
            ]
            for w in widgets_to_block:
                w.blockSignals(True)
            
            try:
                model = file_settings.get('model', 'a-little-better-model')
                index = file_tab.combo_model.findData(model)
                if index >= 0:
                    file_tab.combo_model.setCurrentIndex(index)
                
                threads = file_settings.getint('cpu_threads', DEFAULT_THREADS)
                if threads > ALLOWED_THREADS:
                    print(f"[Config] cpu_threads={threads} > max={ALLOWED_THREADS} (physical cores), clamping")
                    threads = ALLOWED_THREADS
                file_tab.slider_threads.setValue(threads)
                file_tab.label_threads.setText(str(threads))
                
                punct_conf = file_settings.getint('punctuation_confidence_slider', 6)
                file_tab.slider_punct_conf.setValue(punct_conf)
                file_tab.on_punct_conf_changed(punct_conf)
                
                case_conf = file_settings.getint('case_confidence_slider', 6)
                file_tab.slider_case_conf.setValue(case_conf)
                file_tab.on_case_conf_changed(case_conf)
                
                speaker_diarization = file_settings.getboolean('speaker_diarization', True)
                file_tab.check_speaker_diarization.setChecked(speaker_diarization)
                file_tab.on_speaker_diarization_changed(file_tab.check_speaker_diarization.checkState().value)

                overlap_separation = file_settings.getboolean('overlap_separation', False)
                file_tab.check_overlap_separation.setChecked(overlap_separation)

                show_speaker_labels = file_settings.getboolean('show_speaker_labels', True)
                file_tab.check_show_speaker_labels.setChecked(show_speaker_labels)
                
                num_speakers = file_settings.getint('num_speakers', 0)
                if num_speakers == 0:
                    file_tab.spin_num_speakers.setCurrentIndex(0)
                elif 2 <= num_speakers <= 5:
                    file_tab.spin_num_speakers.setCurrentIndex(num_speakers - 1)
                
                speaker_model = file_settings.get('speaker_model', 'community1_pure_ort')
                # Migration: remap old model IDs to pure_ort
                if speaker_model in ('titanet_small', 'community1_onnx', 'community1', 'campp_pure_ort'):
                    speaker_model = 'community1_pure_ort'
                index = file_tab.combo_speaker_model.findData(speaker_model)
                if index >= 0:
                    file_tab.combo_speaker_model.setCurrentIndex(index)
                file_tab.on_speaker_model_changed(file_tab.combo_speaker_model.currentIndex())
                
                save_ram = file_settings.getboolean('save_ram', True)
                file_tab.check_save_ram.setChecked(save_ram)

                rms_normalize = file_settings.getboolean('rms_normalize', True)
                file_tab.check_rms_normalize.setChecked(rms_normalize)

                bypass_vad = file_settings.getboolean('bypass_vad', False)
                file_tab.check_bypass_vad.setChecked(bypass_vad)

                # Auto analyze quality setting
            finally:
                # Unblock signals
                for w in widgets_to_block:
                    w.blockSignals(False)
            
            # Apply Live tab settings
            live_settings = self.config['LiveSettings']
            live_tab = self.tab_live
            
            live_tab.combo_model.blockSignals(True)
            live_tab.slider_threads.blockSignals(True)
            try:
                model = live_settings.get('model', 'a-little-better-model')
                index = live_tab.combo_model.findData(model)
                if index >= 0:
                    live_tab.combo_model.setCurrentIndex(index)
                
                threads = live_settings.getint('cpu_threads', DEFAULT_THREADS)
                if threads > ALLOWED_THREADS:
                    print(f"[Config] Live cpu_threads={threads} > max={ALLOWED_THREADS}, clamping")
                    threads = ALLOWED_THREADS
                live_tab.slider_threads.setValue(threads)
                live_tab.label_threads.setText(str(threads))
            finally:
                live_tab.combo_model.blockSignals(False)
                live_tab.slider_threads.blockSignals(False)
            
            # Refresh microphones after config is applied to restore selected microphone
            live_tab.refresh_microphones(try_restore=True)
        finally:
            self._applying_config = False

    def save_file_config(self):
        """Save File tab UI state to config"""
        if self._applying_config:
            print("[Config] Skipping save_file_config - applying config")
            return
        
        # Disconnect signals temporarily to prevent recursive saves
        self._disconnect_file_signals()
        
        try:
            file_tab = self.tab_file
            print(f"[Config] Saving file config: model={file_tab.combo_model.currentData()}, threads={file_tab.slider_threads.value()}")
            
            # Get number of speakers (0 = auto)
            num_speakers = 0
            if file_tab.spin_num_speakers.currentIndex() > 0:
                try:
                    num_speakers = int(file_tab.spin_num_speakers.currentText())
                except:
                    num_speakers = 0
            
            # Get current data with fallback
            model = file_tab.combo_model.currentData()
            if model is None:
                model = file_tab.combo_model.currentText() or 'zipformer-30m-rnnt-6000h'
            
            speaker_model = file_tab.combo_speaker_model.currentData()
            if speaker_model is None:
                speaker_model = 'community1_pure_ort'
            
            # Update config while preserving keys not in UI (e.g., hf_token)
            self.config['FileSettings']['model'] = model
            self.config['FileSettings']['cpu_threads'] = str(file_tab.slider_threads.value())
            self.config['FileSettings']['punctuation_confidence_slider'] = str(file_tab.slider_punct_conf.value())
            self.config['FileSettings']['case_confidence_slider'] = str(file_tab.slider_case_conf.value())
            self.config['FileSettings']['speaker_diarization'] = str(file_tab.check_speaker_diarization.isChecked())
            self.config['FileSettings']['overlap_separation'] = str(file_tab.check_overlap_separation.isChecked())
            self.config['FileSettings']['show_speaker_labels'] = str(file_tab.check_show_speaker_labels.isChecked())
            self.config['FileSettings']['num_speakers'] = str(num_speakers)
            self.config['FileSettings']['speaker_model'] = speaker_model
            self.config['FileSettings']['save_ram'] = str(file_tab.check_save_ram.isChecked())
            self.config['FileSettings']['rms_normalize'] = str(file_tab.check_rms_normalize.isChecked())
            self.config['FileSettings']['bypass_vad'] = str(file_tab.check_bypass_vad.isChecked())

            self.save_config()
        finally:
            # Reconnect signals
            self._connect_file_signals()
    
    def save_live_config(self):
        """Save Live tab UI state to config"""
        if self._applying_config:
            return
        
        # Disconnect signals temporarily
        self._disconnect_live_signals()
        
        try:
            live_tab = self.tab_live
            
            # Get current data with fallback
            model = live_tab.combo_model.currentData()
            if model is None:
                model = live_tab.combo_model.currentText() or 'zipformer-30m-rnnt-6000h'
            
            # Get current microphone list and selected microphone
            current_mic_list = []
            selected_mic = ""
            for i in range(live_tab.combo_microphone.count()):
                mic_name = live_tab.combo_microphone.itemText(i)
                if mic_name and not mic_name.startswith("Không tìm thấy"):
                    current_mic_list.append(mic_name)
            if live_tab.combo_microphone.currentIndex() >= 0:
                selected_mic = live_tab.combo_microphone.currentText()
            
            # Update config while preserving keys not in UI
            self.config['LiveSettings']['model'] = model
            self.config['LiveSettings']['cpu_threads'] = str(live_tab.slider_threads.value())
            self.config['LiveSettings']['microphone_list'] = '|||'.join(current_mic_list)
            self.config['LiveSettings']['selected_microphone'] = selected_mic
            self.save_config()
        finally:
            # Reconnect signals
            self._connect_live_signals()
    
    def closeEvent(self, event):
        """Save config when closing the application with confirmation"""
        print("[Config] Closing app, saving config...")
        
        # Check if there are unsaved data
        live_tab = self.tab_live
        file_tab = self.tab_file
        
        has_unsaved_live = live_tab.has_recorded_audio and not live_tab.wav_saved
        has_unsaved_file = file_tab.segments and not file_tab.json_saved
        
        # Build confirmation message
        messages = ["Bạn có chắc muốn tắt ứng dụng?"]
        
        if has_unsaved_live:
            messages.append("⚠️ Chưa lưu file WAV của dịch trực tiếp!")
        
        if has_unsaved_file:
            messages.append("⚠️ Chưa lưu file JSON của dịch tập tin!")
        
        # Show confirmation dialog if there are unsaved data
        if has_unsaved_live or has_unsaved_file:
            from PyQt6.QtWidgets import QMessageBox
            
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Xác nhận tắt ứng dụng")
            msg_box.setText("<br>".join(messages))
            msg_box.setIcon(QMessageBox.Icon.Warning)
            
            btn_yes = msg_box.addButton("Tắt ứng dụng", QMessageBox.ButtonRole.YesRole)
            btn_no = msg_box.addButton("Hủy", QMessageBox.ButtonRole.NoRole)
            msg_box.setDefaultButton(btn_no)
            
            msg_box.exec()
            
            if msg_box.clickedButton() != btn_yes:
                event.ignore()
                return
        
        self.save_file_config()
        self.save_live_config()
        print(f"[Config] Config saved to {CONFIG_FILE}")
        
        # Clean up temporary playback files from file tab
        if hasattr(file_tab, 'cleanup_temp_files'):
            file_tab.cleanup_temp_files()
            
        event.accept()

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {COLORS['bg_dark']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: bold;
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
        
        # Tab 1: Xử lý tập tin
        self.tab_file = FileProcessingTab(main_window=self)
        self.main_tabs.addTab(self.tab_file, "📁 Xử lý tập tin")
        
        # Tab 2: Xử lý trực tiếp
        self.tab_live = LiveProcessingTab(main_window=self)
        self.main_tabs.addTab(self.tab_live, "🎤 Xử lý trực tiếp")

        # Theme toggle (top-right corner, ngang hàng với tab bar)
        self.btn_theme = QPushButton()
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.setToolTip("Đổi giao diện Sáng / Tối (cần khởi động lại)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self._refresh_theme_button()
        # Wrapper để có margin phải, button không dính sát cạnh
        corner = QWidget()
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 4, 8, 4)
        corner_layout.addWidget(self.btn_theme)
        self.main_tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        main_layout.addWidget(self.main_tabs)

        # Connect signals to save config when settings change
        self.connect_config_signals()
    
    def _refresh_theme_button(self):
        """Render nút theme theo theme đang dùng (gọi sau apply_theme).

        Nút hiển thị icon + label của theme HIỆN TẠI để user biết đang ở đâu.
        Click sẽ chuyển sang theme còn lại + nhắc restart.
        """
        from core.config import CURRENT_THEME
        is_dark = CURRENT_THEME == 'dark'
        # Hiện theme hiện tại — icon + tên ngắn
        self.btn_theme.setText("🌙 Tối" if is_dark else "☀️ Sáng")
        # Pill button — nền card, chữ primary, viền border
        self.btn_theme.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 14px;
                padding: 4px 14px;
                font-size: 12px;
                font-weight: 600;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                color: white;
                border-color: {COLORS['accent']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_hover']};
            }}
        """)

    def toggle_theme(self):
        """Chuyển theme Light ↔ Dark, lưu config, nhắc restart."""
        from core.config import CURRENT_THEME
        new_theme = 'light' if CURRENT_THEME == 'dark' else 'dark'

        if 'Appearance' not in self.config:
            self.config['Appearance'] = {}
        self.config['Appearance']['theme'] = new_theme
        self.save_config()

        # Feedback ngay: đổi label nút sang theme mới để user thấy click có hiệu lực
        self.btn_theme.setText("🌙 Tối" if new_theme == 'dark' else "☀️ Sáng")

        label = "Sáng" if new_theme == 'light' else "Tối"
        QMessageBox.information(
            self, "Đổi giao diện",
            f"Đã chuyển sang giao diện {label}.\n\n"
            f"Vui lòng khởi động lại ứng dụng để áp dụng theme mới."
        )

    def connect_config_signals(self):
        """Connect UI signals to auto-save config when settings change"""
        self._connect_file_signals()
        self._connect_live_signals()
    
    def _connect_file_signals(self):
        """Connect File tab signals"""
        file_tab = self.tab_file
        file_tab.combo_model.currentIndexChanged.connect(self.save_file_config)
        file_tab.slider_threads.valueChanged.connect(self.save_file_config)
        file_tab.slider_punct_conf.valueChanged.connect(self.save_file_config)
        file_tab.slider_case_conf.valueChanged.connect(self.save_file_config)
        file_tab.check_speaker_diarization.stateChanged.connect(self.save_file_config)
        file_tab.check_overlap_separation.stateChanged.connect(self.save_file_config)
        file_tab.check_show_speaker_labels.stateChanged.connect(self.save_file_config)
        file_tab.spin_num_speakers.currentIndexChanged.connect(self.save_file_config)
        file_tab.combo_speaker_model.currentIndexChanged.connect(self.save_file_config)
        file_tab.check_save_ram.stateChanged.connect(self.save_file_config)
        file_tab.check_rms_normalize.stateChanged.connect(self.save_file_config)
        file_tab.check_bypass_vad.stateChanged.connect(self.save_file_config)

        # threshold slider removed from UI — no signal to connect
    
    def _disconnect_file_signals(self):
        """Disconnect File tab signals temporarily"""
        file_tab = self.tab_file
        try:
            file_tab.combo_model.currentIndexChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.slider_threads.valueChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.slider_punct_conf.valueChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.slider_case_conf.valueChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_speaker_diarization.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_overlap_separation.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_show_speaker_labels.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.spin_num_speakers.currentIndexChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.combo_speaker_model.currentIndexChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_save_ram.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_rms_normalize.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            file_tab.check_bypass_vad.stateChanged.disconnect(self.save_file_config)
        except:
            pass
        try:
            pass  # threshold slider removed from UI
        except:
            pass
    
    def _connect_live_signals(self):
        """Connect Live tab signals"""
        live_tab = self.tab_live
        live_tab.combo_model.currentIndexChanged.connect(self.save_live_config)
        live_tab.slider_threads.valueChanged.connect(self.save_live_config)
        live_tab.combo_microphone.currentIndexChanged.connect(self.save_live_config)
    
    def _disconnect_live_signals(self):
        """Disconnect Live tab signals temporarily"""
        live_tab = self.tab_live
        try:
            live_tab.combo_model.currentIndexChanged.disconnect(self.save_live_config)
        except:
            pass
        try:
            live_tab.slider_threads.valueChanged.disconnect(self.save_live_config)
        except:
            pass
        try:
            live_tab.combo_microphone.currentIndexChanged.disconnect(self.save_live_config)
        except:
            pass

    def show_about_dialog(self):
        """Hiển thị dialog thông tin phần mềm"""
        from PyQt6.QtWidgets import QDialog
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Thông tin")
        dialog.setFixedSize(520, 480)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                line-height: 1.6;
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Tiêu đề
        title = QLabel("<h2 style='color: #007bff; margin-bottom: 10px;'>sherpa-vietnamese-asr</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Nội dung chính
        from core.version import get_version
        app_version = get_version()

        content_text = f"""
<p style='color: #cccccc; margin: 8px 0;'><b>Thiết kế:</b> Nguyễn Hồng Quân<br>
nhquan.thanhuy@tphcm.gov.vn — 098.558.3555<br>
Chuyên viên Phòng Chuyển đổi số - Cơ yếu,<br>
Văn phòng Thành ủy Thành phố Hồ Chí Minh.</p>

<p style='color: #cccccc; margin: 8px 0;'><b>Lập trình:</b> Claude và những người bạn</p>

<p style='color: #cccccc; margin: 8px 0;'><b>Phiên bản:</b> {app_version}</p>

<p style='color: #ffd700; margin: 15px 0; font-weight: bold; text-align: center;'>
PHẦN MỀM SỬ DỤNG TRONG MÔI TRƯỜNG GIÁO DỤC, HÀNH CHÍNH CÔNG, TỔ CHỨC ĐẢNG, ĐOÀN THỂ.<br>
KHÔNG SỬ DỤNG CHO MỤC ĐÍCH THƯƠNG MẠI.
</p>

<p style='color: #28a745; margin: 10px 0; font-weight: bold;'>CHỨC NĂNG:</p>
<p style='color: #cccccc; margin: 5px 0;'>
• Chuyển ghi âm thành văn bản tiếng Việt (offline)<br>
• 3 model ASR: Zipformer 30M, 68M, ROVER<br>
• Phân tách người nói: Pyannote Community-1, Senko CAM++<br>
• NaturalTurn: nhận diện lượt nói tự nhiên<br>
• Tự động thêm dấu câu, viết hoa<br>
• Nhận dạng real-time qua microphone<br>
• Hỗ trợ hotwords (từ khóa tùy chỉnh)<br>
• Đánh giá chất lượng âm thanh (DNSMOS)
</p>

<p style='color: #28a745; margin: 10px 0; font-weight: bold;'>CÔNG NGHỆ:</p>
<ul style='color: #cccccc; margin: 5px 0; padding-left: 20px;'>
<li><b>ASR:</b> Sherpa-ONNX, Zipformer RNN-T (30M + 68M)</li>
<li><b>Diarization:</b> Pyannote Community-1 + Senko CAM++ (Pure ONNX Runtime)</li>
<li><b>Dấu câu:</b> ViBERT-capu (ONNX)</li>
<li><b>VAD:</b> Pyannote Segmentation (ONNX)</li>
<li><b>Resampling:</b> SoXR HQ</li>
<li><b>Giao diện:</b> PyQt6</li>
</ul>
"""
        content = QLabel(content_text)
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(content)
        
        layout.addStretch()
        
        # Nút đóng
        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)
        
        dialog.exec()


if __name__ == "__main__":
    # Dọn dẹp file tạm từ lần chạy trước (nếu bị crash)
    cleanup_temp_files()

    # _app đã tạo ở đầu file (trước imports nặng)
    app = _app

    # Load MainWindow
    window = MainWindow()
    window.show()

    # Đóng Win32 splash sau khi MainWindow hiển thị
    if _splash_ok:
        try:
            splash_win32.destroy()
        except Exception:
            pass

    # Raise MainWindow lên trên cùng một lần sau khi splash đóng
    window.raise_()
    window.activateWindow()

    ret = app.exec()
    os._exit(ret)
