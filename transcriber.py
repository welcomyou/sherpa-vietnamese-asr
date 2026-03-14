# transcriber.py - Thin QThread wrapper around core.asr_engine.TranscriberPipeline
# Core logic nằm trong core/asr_engine.py

import logging
from PyQt6.QtCore import QThread, pyqtSignal

from core.asr_engine import TranscriberPipeline

logger = logging.getLogger(__name__)


class TranscriberThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, dict)
    error = pyqtSignal(str)

    def __init__(self, file_path, model_path, config):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.config = config
        self.is_running = True

    def run(self):
        try:
            pipeline = TranscriberPipeline(
                file_path=self.file_path,
                model_path=self.model_path,
                config=self.config,
                progress_callback=self.progress.emit,
                cancel_check=lambda: not self.is_running
            )

            result_data = pipeline.run()

            if result_data is None:
                # Cancelled
                return

            self.finished.emit(result_data["text"], result_data)
            print("[Transcriber] Finished signal emitted successfully")

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

    def stop(self):
        self.is_running = False
