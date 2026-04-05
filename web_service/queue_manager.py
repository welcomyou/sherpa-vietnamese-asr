"""
Hàng đợi xử lý ASR - FIFO, 1 file tại 1 thời điểm.
"""

import os
import json
import time
import threading
import logging
import subprocess
from typing import Optional, Callable

from web_service.config import server_config, UPLOAD_DIR
from web_service.database import db
from web_service.session_manager import ws_manager

logger = logging.getLogger("asr.queue")


def convert_to_wav(input_path: str, progress_callback: Optional[Callable[[int], None]] = None) -> str:
    """Convert audio/video sang WAV bằng ffmpeg (giữ nguyên sample rate gốc).
    Việc resample sang 16kHz mono sẽ do librosa xử lý trong load_audio()
    để đảm bảo chất lượng resampling cao (soxr_vhq), giống pipeline desktop.

    Args:
        progress_callback: Optional callback(percent: int) nhận % tiến độ 0-99
    """
    if input_path.lower().endswith(".wav"):
        return input_path

    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    if os.path.exists(output_path):
        return output_path

    # Tim ffmpeg
    from core.asr_engine import setup_ffmpeg_path
    setup_ffmpeg_path()

    try:
        # Lấy duration bằng ffprobe để tính % tiến độ
        total_duration = 0
        if progress_callback:
            try:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    input_path,
                ]
                probe_result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, timeout=30
                )
                if probe_result.returncode == 0 and probe_result.stdout.strip():
                    total_duration = float(probe_result.stdout.strip())
            except Exception:
                pass

        if progress_callback and total_duration > 0:
            # Dùng Popen + -progress pipe:1 để parse tiến độ realtime
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-acodec", "pcm_s16le",
                "-progress", "pipe:1",
                output_path,
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            last_pct = -1
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        current_us = int(line.split("=")[1])
                        pct = min(99, int(current_us / (total_duration * 1_000_000) * 100))
                        if pct > last_pct:
                            last_pct = pct
                            progress_callback(pct)
                    except (ValueError, ZeroDivisionError):
                        pass

            proc.wait(timeout=300)
            if proc.returncode != 0:
                stderr_text = proc.stderr.read() if proc.stderr else ""
                logger.error(f"ffmpeg error: {stderr_text[:500]}")
                raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
        else:
            # Fallback: blocking call không có progress (file WAV hoặc không cần)
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-acodec", "pcm_s16le",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr.decode(errors='replace')[:500]}")
                raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

        logger.info(f"Converted to WAV: {output_path}")
        return output_path
    except FileNotFoundError:
        logger.error("ffmpeg not found!")
        raise RuntimeError("ffmpeg not found. Please ensure ffmpeg is in PATH or bundled.")


class QueueManager:
    """Quản lý hàng đợi xử lý ASR. Nghiêm ngặt 1 file tại 1 thời điểm."""

    def __init__(self):
        self.current_file_id: Optional[int] = None
        self.current_session_id: Optional[str] = None
        self._cancelled = False
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._paused = False
        self._event_loop = None  # Reference toi asyncio event loop chinh
        self._last_progress_time = 0
        self._last_progress_msg = ''

    @property
    def is_processing(self) -> bool:
        return self.current_file_id is not None

    def cancel_check(self) -> bool:
        """Duoc truyen vao TranscriberPipeline.cancel_check"""
        return self._cancelled

    def progress_callback(self, msg: str):
        """
        Duoc truyen vao TranscriberPipeline.progress_callback
        Format: "PHASE:Name|Message|Percent" hoac plain text
        """
        if not self.current_file_id:
            return

        percent = 0
        message = msg
        phase_name = ""

        if msg.startswith("PHASE:"):
            try:
                parts = msg[6:].split("|")
                if len(parts) >= 3:
                    phase_name = parts[0]
                    message = parts[1]
                    percent = int(parts[2])
            except (ValueError, IndexError):
                pass

        # Throttle: chi gui update moi 1 giay hoac khi phase thay doi
        now = time.monotonic()
        phase_changed = message != self._last_progress_msg
        if not phase_changed and (now - self._last_progress_time) < 1.0:
            return
        self._last_progress_time = now
        self._last_progress_msg = message

        # Cap nhat DB
        try:
            db.update_queue_progress(self.current_file_id, percent, message)
        except Exception as e:
            logger.debug(f"DB progress update failed: {e}")

        # Gui WebSocket (thread-safe)
        if self.current_session_id:
            self._send_ws(self.current_session_id, {
                "type": "progress",
                "file_id": self.current_file_id,
                "percent": percent,
                "message": message,
                "phase": phase_name,
            })

    def broadcast_queue_positions(self):
        """Gửi queue_position cập nhật cho TẤT CẢ sessions có file đang waiting."""
        waiting_items = db.get_all_waiting_queue_items()
        total = len(waiting_items)
        for idx, item in enumerate(waiting_items):
            self._send_ws(item["session_id"], {
                "type": "queue_position",
                "file_id": item["file_id"],
                "position": idx + 1,
                "total": total,
            })

    def add_to_queue(self, file_id: int, session_id: str, config: dict) -> dict:
        """Them file vao queue. Tra ve thong tin vi tri."""
        # Kiểm tra session đã có file trong queue chưa
        if db.has_session_in_queue(session_id):
            return {"error": "Bạn đã có 1 file đang chờ xử lý. Vui lòng đợi."}

        db.add_to_queue(file_id, session_id, config)

        # Tu dong trigger worker
        self.process_next()

        # Nếu file được pick lên xử lý ngay (queue trống) → position=0
        # Tránh flicker "đang ở vị trí #1" rồi biến mất ngay
        if self.current_file_id == file_id:
            return {"success": True, "position": 0, "total": 0}

        position = db.get_queue_position(file_id)
        total = db.get_queue_total_waiting()

        # Broadcast vị trí mới cho tất cả sessions đang chờ
        self.broadcast_queue_positions()

        return {"success": True, "position": position, "total": total}

    def process_next(self):
        """Lấy file tiếp theo trong queue và xử lý"""
        if self._paused:
            return

        with self._lock:
            if self.current_file_id is not None:
                return  # Đang xử lý file khác

            item = db.get_next_queue_item()
            if item is None:
                return  # Queue trong

            self.current_file_id = item["file_id"]
            self.current_session_id = item["session_id"]
            self._cancelled = False

        # Chay trong thread rieng
        self._worker_thread = threading.Thread(
            target=self._process_item, args=(item,), daemon=True
        )
        self._worker_thread.start()

    def _process_item(self, item: dict):
        """Xử lý 1 item trong queue (ASR hoặc Summarization)."""
        file_id = item["file_id"]
        session_id = item["session_id"]
        config = json.loads(item["config_json"]) if item["config_json"] else {}

        job_type = config.get("job_type", "asr")

        if job_type == "summarize":
            self._process_summarize(item, config)
            return

        # --- ASR job (code gốc) ---
        logger.info(f"Processing file_id={file_id} session={session_id}")

        db.set_queue_processing(file_id)

        # Cập nhật meeting status
        meeting = db.get_meeting_by_file_id(file_id)
        if meeting:
            db.update_meeting(meeting["id"], status="processing")

        # Thông báo bắt đầu
        self._send_ws(session_id, {"type": "processing_started", "file_id": file_id})

        try:
            # 1. Đường dẫn file
            stored_filename = item["stored_filename"]
            file_path = os.path.join(UPLOAD_DIR, stored_filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # 2. Convert sang WAV (với progress bar realtime từ ffmpeg)
            self.progress_callback("PHASE:Convert|Đang chuyển định dạng audio...|0")

            def _convert_progress(pct):
                self.progress_callback(f"PHASE:Convert|Đang chuyển định dạng audio ({pct}%)|{pct}")

            wav_path = convert_to_wav(file_path, progress_callback=_convert_progress)

            if self._cancelled:
                raise InterruptedError("Cancelled by user")

            # 3. Gọi core pipeline
            from core.asr_engine import TranscriberPipeline
            from core.config import BASE_DIR

            model_name = config.get("model", server_config.get("default_asr_model"))

            # ROVER mode detection
            from core.asr_engine import ROVER_MODEL_ID
            is_rover = (model_name == ROVER_MODEL_ID)
            if is_rover:
                model_path = os.path.join(BASE_DIR, "models")
            else:
                model_path = os.path.join(BASE_DIR, "models", model_name)

            punct_slider = max(1, min(10, int(config.get("punctuation_confidence",
                                          server_config.get("default_punctuation_confidence")))))
            case_slider = max(1, min(10, int(config.get("case_confidence",
                                         server_config.get("default_case_confidence")))))

            # Convert slider (1-10) to float confidence
            # confidence > 0: cong vao $KEEP → bao thu. confidence < 0: tru $KEEP → manh me
            # slider=2: +0.35, slider=5: ~0, slider=7: -0.37, slider=10: -0.8
            punct_confidence = 0.5 - (punct_slider - 1) * (1.3 / 9)
            # slider=1 → -1.5 (tat viet hoa), slider=10 → 0.5 (nhieu viet hoa)
            case_confidence_val = -1.5 + (case_slider - 1) * (2.0 / 9)
            # Chi bypass khi slider dau cau = 1 (khong muon them dau)
            # Neu case=1 nhung punct>1 van chay model (case_confidence am se suppress viet hoa)
            bypass_restorer = (punct_slider <= 1)

            speaker_model_id = config.get("speaker_model",
                                            server_config.get("default_speaker_model"))

            # Kiem tra model speaker co san khong, neu khong thi fallback
            from core.speaker_diarization import SpeakerDiarizer, get_available_models
            available_speaker_models = get_available_models()
            if speaker_model_id not in available_speaker_models:
                if available_speaker_models:
                    # Fallback: uu tien community1_onnx > titanet_small > bat ky
                    for preferred in ["community1_onnx", "titanet_small", "community1"]:
                        if preferred in available_speaker_models:
                            logger.warning(f"Speaker model '{speaker_model_id}' not available, "
                                         f"falling back to '{preferred}'")
                            speaker_model_id = preferred
                            break
                    else:
                        speaker_model_id = list(available_speaker_models.keys())[0]
                        logger.warning(f"Falling back to '{speaker_model_id}'")
                else:
                    logger.warning("No speaker models available, disabling diarization")
                    config["speaker_diarization"] = False

            # Lấy threshold từ request, nếu không có thì dùng default của model
            raw_threshold = config.get("diarization_threshold")
            if raw_threshold is not None:
                diarization_threshold = int(raw_threshold) / 100.0
            else:
                # Tu dong dung default threshold phu hop voi tung model
                diarization_threshold = SpeakerDiarizer.get_default_threshold(speaker_model_id)

            # num_speakers: 0 hoac -1 deu la auto-detect
            raw_num_speakers = int(config.get("num_speakers", 0))
            num_speakers = -1 if raw_num_speakers <= 0 else raw_num_speakers

            pipeline_config = {
                "cpu_threads": server_config.cpu_threads,
                "restore_punctuation": True,  # Luon True giong desktop, bypass_restorer xu ly skip
                "bypass_restorer": bypass_restorer,
                "punctuation_confidence": punct_confidence,
                "case_confidence": case_confidence_val,
                "speaker_diarization": config.get("speaker_diarization", True),
                "speaker_model": speaker_model_id,
                "num_speakers": num_speakers,
                "diarization_threshold": diarization_threshold,
                "save_ram": False,  # Server giữ model trong RAM để phục vụ request tiếp nhanh hơn
                "rover_mode": is_rover,

                "preprocess_rms_normalize": config.get("rms_normalize", False),
            }

            logger.info(f"Pipeline config: {pipeline_config}")

            pipeline = TranscriberPipeline(
                file_path=wav_path,
                model_path=model_path,
                config=pipeline_config,
                progress_callback=self.progress_callback,
                cancel_check=self.cancel_check,
            )

            result = pipeline.run()

            if self._cancelled:
                raise InterruptedError("Cancelled by user")

            # 4. Đánh giá chất lượng: tận dụng DNSMOS + ASR confidence từ pipeline
            quality_info = result.get("quality_info")
            asr_conf = result.get("asr_confidence")
            if quality_info and asr_conf is not None:
                quality_info["asr_confidence"] = round(float(asr_conf), 4)
            elif asr_conf is not None:
                quality_info = {"asr_confidence": round(float(asr_conf), 4)}

            if self._cancelled:
                raise InterruptedError("Cancelled by user")

            # 5. Serialize kết quả
            from core.asr_json import serialize_segments

            segments = result.get("segments", [])
            speaker_names = result.get("speaker_names", {})

            asr_json = serialize_segments(
                segments=segments,
                speaker_name_mapping=speaker_names,
                model_name=model_name,
                model_type="file",
                duration_sec=result.get("duration_sec", 0),
                timing=result.get("timing"),
            )

            # Thêm quality_info vào kết quả
            if quality_info:
                asr_json["quality_info"] = quality_info

            # 6. Lưu kết quả vào DB
            db.update_file(
                file_id,
                status="completed",
                asr_result_json=json.dumps(asr_json, ensure_ascii=False),
                speaker_names_json=json.dumps(speaker_names, ensure_ascii=False),
                model_used=model_name,
                duration_sec=result.get("duration_sec", 0),
                completed_at=__import__("datetime").datetime.utcnow().isoformat(),
            )
            db.set_queue_completed(file_id)

            # Cập nhật meeting khi hoàn thành
            meeting = db.get_meeting_by_file_id(file_id)
            if meeting:
                db.update_meeting(meeting["id"],
                    status="completed",
                    asr_result_json=json.dumps(asr_json, ensure_ascii=False))

            logger.info(f"Completed file_id={file_id}")

            # Thông báo hoàn thành
            self._send_ws(session_id, {
                "type": "asr_complete",
                "file_id": file_id,
                "result": asr_json,
            })

        except InterruptedError:
            db.set_queue_cancelled(file_id)
            meeting = db.get_meeting_by_file_id(file_id)
            if meeting:
                db.update_meeting(meeting["id"], status="error",
                    error_message="Cancelled by user")
            logger.info(f"Cancelled file_id={file_id}")
            self._send_ws(session_id, {"type": "asr_cancelled", "file_id": file_id})

        except Exception as e:
            error_msg = str(e)
            db.set_queue_error(file_id, error_msg)
            meeting = db.get_meeting_by_file_id(file_id)
            if meeting:
                db.update_meeting(meeting["id"], status="error",
                    error_message=error_msg)
            logger.error(f"Error processing file_id={file_id}: {error_msg}", exc_info=True)
            self._send_ws(session_id, {
                "type": "asr_error",
                "file_id": file_id,
                "error": error_msg,
            })

        finally:
            with self._lock:
                self.current_file_id = None
                self.current_session_id = None
                self._cancelled = False

            # Broadcast vị trí queue mới cho tất cả sessions đang chờ
            self.broadcast_queue_positions()

            # Xử lý file tiếp theo
            self.process_next()

    def _process_summarize(self, item: dict, config: dict):
        """Xử lý summarization job (chạy trong worker thread)."""
        file_id = item["file_id"]
        session_id = item["session_id"]

        logger.info(f"Summarization job: file_id={file_id} session={session_id}")

        db.set_queue_processing(file_id)

        # Thông báo bắt đầu
        self._send_ws(session_id, {
            "type": "summary_started", "file_id": file_id,
        })

        try:
            from web_service.summarizer import run_summarization
            run_summarization(
                file_id=file_id,
                session_id=session_id,
                send_ws=self._send_ws,
                progress_cb=self.progress_callback,
            )
            db.set_queue_completed(file_id)

        except InterruptedError:
            db.set_queue_cancelled(file_id)
            logger.info(f"Summarization cancelled: file_id={file_id}")
            self._send_ws(session_id, {
                "type": "summary_error", "file_id": file_id,
                "error": "Đã hủy bởi người dùng",
            })

        except Exception as e:
            error_msg = str(e)
            db.set_queue_error(file_id, error_msg)
            logger.error(f"Summarization error: file_id={file_id}: {error_msg}", exc_info=True)
            self._send_ws(session_id, {
                "type": "summary_error", "file_id": file_id,
                "error": error_msg,
            })

        finally:
            with self._lock:
                self.current_file_id = None
                self.current_session_id = None
                self._cancelled = False

            self.broadcast_queue_positions()
            self.process_next()

    def add_summarize_to_queue(self, file_id: int, session_id: str) -> dict:
        """Thêm summarization job vào queue chung (cùng priority với ASR)."""
        if db.has_session_in_queue(session_id):
            return {"error": "Bạn đã có 1 tác vụ đang chờ xử lý. Vui lòng đợi."}

        config = {"job_type": "summarize"}
        db.add_to_queue(file_id, session_id, config)

        self.process_next()

        if self.current_file_id == file_id:
            return {"success": True, "position": 0, "total": 0}

        position = db.get_queue_position(file_id)
        total = db.get_queue_total_waiting()
        self.broadcast_queue_positions()

        return {"success": True, "position": position, "total": total}

    def cancel(self, file_id: int) -> bool:
        """Huy xu ly file. Tra ve True neu thanh cong."""
        with self._lock:
            if self.current_file_id == file_id:
                self._cancelled = True
                logger.info(f"Cancelling processing file_id={file_id}")
                return True

        # Neu file dang waiting trong queue -> xoa
        pos = db.get_queue_position(file_id)
        if pos > 0:
            db.remove_from_queue(file_id)
            logger.info(f"Removed from queue file_id={file_id}")
            # Broadcast vị trí mới cho các sessions còn lại
            self.broadcast_queue_positions()
            return True

        return False

    def cancel_for_session(self, session_id: str):
        """Huy tat ca items cua session"""
        processing_ids = db.cancel_session_queue(session_id)
        for fid in processing_ids:
            if self.current_file_id == fid:
                self._cancelled = True

    def pause(self):
        self._paused = True
        logger.info("Queue paused")

    def resume(self):
        self._paused = False
        logger.info("Queue resumed")
        self.process_next()

    def _send_ws(self, session_id: str, data: dict):
        """Helper gui WebSocket tu worker thread (thread-safe)."""
        try:
            import asyncio
            loop = self._event_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send_to_session(session_id, data), loop
                )
        except Exception as e:
            logger.debug(f"WebSocket send failed: {e}")

    def _send_ws_broadcast(self, data: dict):
        """Helper broadcast WebSocket tu worker thread (thread-safe)."""
        try:
            import asyncio
            loop = self._event_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(data), loop
                )
        except Exception as e:
            logger.debug(f"WebSocket broadcast failed: {e}")


# Singleton
queue_manager = QueueManager()
