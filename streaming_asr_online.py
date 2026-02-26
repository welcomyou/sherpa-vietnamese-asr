"""Real-time ASR using OnlineRecognizer (Streaming Transducer) for Zipformer models"""
import os
import queue
import numpy as np
import collections
from PyQt6.QtCore import QThread, pyqtSignal

# Import torch at module level for VAD (Issue B6 fix)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# BASE_DIR should be defined to locate resources if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VADTrigger:
    """
    VAD Trigger using Ring Buffer.
    - buffers audio continuously.
    - detects 'Speech Start' to trigger stream.
    - detects 'Silence' to end stream.
    """
    def __init__(self, sample_rate=16000, trigger_level=0.3):
        self.sample_rate = sample_rate
        self.threshold = trigger_level
        
        # VAD Parameters
        self.window_size = 512
        self.min_silence_duration = 0.5
        self.min_speech_duration = 0.3
        self.speech_pad_ms = 300
        
        # Ring buffer for context - tăng lên 1.2s để giữ âm đầu sau endpoint (fix mất chữ)
        self.context_duration = 1.2
        self.maxlen = int(self.context_duration * sample_rate / self.window_size)
        self.ring_buffer = collections.deque(maxlen=self.maxlen)
        
        # Buffer for VAD processing
        self.vad_buffer = np.array([], dtype=np.float32)
        
        # State
        self.triggered = False
        self.voiceless_count = 0 
        self.speech_chunks = 0
        
        # Load VAD
        self.vad_model = None
        self.vad_available = False
        try:
            import torch
            self.vad_model, _ = torch.hub.load(
                'snakers4/silero-vad', 
                'silero_vad',
                force_reload=False, 
                onnx=False, 
                verbose=False
            )
            self.vad_available = True
            print(f"[VADTrigger] Loaded Silero VAD, threshold={self.threshold}")
        except Exception as e:
            print(f"[VADTrigger] Failed to load VAD: {e}")
    
    def process(self, audio_chunk):
        """
        Process a chunk (bytes or float32 array).
        Returns: (is_speech, prob) or (None, None) if insufficient data
        """
        if not self.vad_available:
            return True, 1.0
            
        if isinstance(audio_chunk, bytes):
            audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk

        self.ring_buffer.append(audio_float)
        self.vad_buffer = np.concatenate([self.vad_buffer, audio_float])
        
        if len(self.vad_buffer) < 512:
            return None, None
        
        # torch đã import ở module level (B6 fix)
        max_prob = 0.0
        processed_any = False
        
        while len(self.vad_buffer) >= 512:
            chunk = self.vad_buffer[:512]
            self.vad_buffer = self.vad_buffer[512:]
            
            with torch.no_grad():
                 prob = self.vad_model(torch.from_numpy(chunk), self.sample_rate).item()
                 max_prob = max(max_prob, prob)
            processed_any = True
        
        if processed_any:
            is_speech = max_prob > self.threshold
            return is_speech, max_prob
        else:
            return None, None

    def get_context(self):
        """Get concatenated audio from ring buffer"""
        if not self.ring_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.ring_buffer))
    
    def clear_buffer(self):
        self.ring_buffer.clear()
        self.vad_buffer = np.array([], dtype=np.float32)

    def reset(self):
        """Reset VAD state completely - only clear buffer, keep model states"""
        self.clear_buffer()
        # Không reset states của VAD model để tránh mất thởi gian warm-up
        # self.vad_model.reset_states() - Đã bỏ để giữ nguyên internal states


class OnlineStreamingASRThread(QThread):
    """
    Streaming ASR Thread using OnlineRecognizer (for Zipformer Streaming models).
    Uses endpoint detection from the model itself.
    """
    text_ready = pyqtSignal(str, bool, float)  # text, is_final, timestamp_sec
    processing_done = pyqtSignal()
    error = pyqtSignal(str)
    asr_ready = pyqtSignal()
    speaker_proposal = pyqtSignal(str)
    
    def __init__(self, model_path, config, audio_queue):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.audio_queue = audio_queue
        
        self.is_running = False
        self.is_recording_active = False
        
        # NO VAD: Feed audio directly to model
        
        # Online recognizer
        self.recognizer = None
        self.stream = None
        
        # Tracking
        self.last_text = ""
        self.current_speaker = "người nói 1"
        
        # Sample counter for timestamp tracking
        self.total_samples = 0
        self.segment_start_samples = 0
        
        # Speech duration tracking (Issue #3: max_speech_duration check)
        self.current_speech_duration = 0.0
        self.max_speech_duration = 12.0  # Max 12 seconds per segment
        
        # Pending actions
        self.pending_speaker = None
        
    def stop(self):
        self.is_running = False
        self.is_recording_active = False

    def start_recording(self):
        """External trigger to enable processing"""
        self.is_recording_active = True
        print("[OnlineStreamingASR] Started listening (NO VAD)...")

    def insert_speaker(self, name):
        """Request to insert a speaker separator"""
        self.pending_speaker = name
        self.speaker_proposal.emit(name)

    def run(self):
        """NO VAD VERSION: Feed audio directly to model, let model decide endpoint"""
        try:
            import sherpa_onnx
            import glob
            import time
            import torch
            
            self.is_running = True
            
            # Limit PyTorch threads
            cpu_threads = self.config.get("cpu_threads", 4)
            try:
                torch.set_num_threads(cpu_threads)
                torch.set_num_interop_threads(1)
                print(f"[OnlineStreamingASR] Set PyTorch threads: {cpu_threads}")
            except RuntimeError as e:
                print(f"[OnlineStreamingASR] PyTorch threads already set: {e}")
            
            # Load Model - Look for chunk-64 files
            print(f"[OnlineStreamingASR] Loading ONLINE model (NO VAD) from {self.model_path}")
            
            encoder_files = glob.glob(os.path.join(self.model_path, "encoder*-chunk-64*.onnx"))
            decoder_files = glob.glob(os.path.join(self.model_path, "decoder*-chunk-64*.onnx"))
            joiner_files = glob.glob(os.path.join(self.model_path, "joiner*-chunk-64*.onnx"))
            tokens = os.path.join(self.model_path, "tokens.txt")
            
            if not encoder_files:
                encoder_files = glob.glob(os.path.join(self.model_path, "encoder*.onnx"))
            if not decoder_files:
                decoder_files = glob.glob(os.path.join(self.model_path, "decoder*.onnx"))
            if not joiner_files:
                joiner_files = glob.glob(os.path.join(self.model_path, "joiner*.onnx"))
            
            if not (encoder_files and decoder_files and joiner_files and os.path.exists(tokens)):
                self.error.emit(f"Model files missing in {self.model_path}")
                return

            print(f"[OnlineStreamingASR] Using encoder: {os.path.basename(encoder_files[0])}")
            print(f"[OnlineStreamingASR] Using decoder: {os.path.basename(decoder_files[0])}")
            print(f"[OnlineStreamingASR] Using joiner: {os.path.basename(joiner_files[0])}")
            
            # Create OnlineRecognizer - NO VAD, model handles everything
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder_files[0],
                decoder=decoder_files[0],
                joiner=joiner_files[0],
                num_threads=self.config.get("cpu_threads", 4),
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=3.0,
                rule2_min_trailing_silence=2.0,  # Giảm xuống 2.0 vì không có VAD delay
                rule3_min_utterance_length=20.0,
                decoding_method="modified_beam_search",
                max_active_paths=8,
                hotwords_file="",
                hotwords_score=1.5,
            )
            
            print("[OnlineStreamingASR] Online model loaded successfully (NO VAD).")
            self.asr_ready.emit()
            
            # Wait for start_recording trigger
            while self.is_running and not self.is_recording_active:
                self.msleep(50)
            
            # Create stream immediately - no VAD, feed all audio
            self.stream = self.recognizer.create_stream()
            self.segment_start_samples = 0
            print("[OnlineStreamingASR] Stream created, feeding all audio directly...")
            
            while self.is_running:
                try:
                    chunk_bytes = self.audio_queue.get(timeout=0.1)
                    
                    if not self.is_recording_active:
                        continue
                    
                    # Convert to float32
                    audio_float = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Count samples for timestamp tracking
                    self.total_samples += len(audio_float)
                    self.current_speech_duration += (len(audio_float) / 16000.0)
                    
                    # Handle pending speaker insertion
                    if self.pending_speaker:
                        speaker_name = self.pending_speaker
                        self.pending_speaker = None
                        print(f"[OnlineStreamingASR] Queuing speaker: {speaker_name}")
                        self.queued_speaker = speaker_name
                    
                    # ALWAYS feed audio to stream (NO VAD)
                    self.stream.accept_waveform(16000, audio_float)
                    
                    # Decode when ready
                    if self.recognizer.is_ready(self.stream):
                        self.recognizer.decode_stream(self.stream)
                        
                        # Get current result
                        current_text = self.recognizer.get_result(self.stream).strip().lower()
                        
                        # Emit partial if changed and not empty
                        if current_text and current_text != self.last_text:
                            # Timestamp chính xác: thờigian audio đã xử lý
                            timestamp_sec = self.total_samples / 16000.0
                            segment_start_sec = self.segment_start_samples / 16000.0
                            if timestamp_sec < segment_start_sec:
                                timestamp_sec = segment_start_sec
                            
                            print(f"[OnlineStreamingASR] Partial: '{current_text[:40]}...' at ts={timestamp_sec:.2f}s")
                            self.text_ready.emit(current_text, False, timestamp_sec)
                            self.last_text = current_text
                        
                        # Check endpoint or max duration
                        is_endpoint = self.recognizer.is_endpoint(self.stream)
                        is_max_duration = self.current_speech_duration > self.max_speech_duration
                        
                        # Force endpoint if speaker change requested
                        if hasattr(self, 'queued_speaker') and self.queued_speaker:
                            print("[OnlineStreamingASR] Force endpoint due to speaker change")
                            is_endpoint = True
                        
                        if is_endpoint or is_max_duration:
                            if is_max_duration and not is_endpoint:
                                print(f"[OnlineStreamingASR] Max duration reached ({self.current_speech_duration:.1f}s)")
                            else:
                                print("[OnlineStreamingASR] Endpoint detected -> Finalizing segment")
                            
                            # Get final result
                            final_text = self.recognizer.get_result(self.stream).strip().lower()
                            
                            if final_text:
                                print(f"[OnlineStreamingASR] Final: {final_text}")
                                timestamp_sec = self.segment_start_samples / 16000.0
                                self.text_ready.emit(final_text, True, timestamp_sec)
                            
                            # Insert queued speaker if any
                            if hasattr(self, 'queued_speaker') and self.queued_speaker:
                                print(f"[OnlineStreamingASR] Inserting speaker: {self.queued_speaker}")
                                token = f" __SPK_SEP__{self.queued_speaker}__SPK_SEP__"
                                self.text_ready.emit(token, True, 0.0)
                                self.speaker_proposal.emit("")
                                self.queued_speaker = None
                            
                            # Reset and create new stream immediately
                            # IMPORTANT: Clear excess queued chunks to avoid processing stale audio
                            dropped = 0
                            while self.audio_queue.qsize() > 5:  # Keep small buffer for continuity
                                try:
                                    self.audio_queue.get_nowait()
                                    dropped += 1
                                except queue.Empty:
                                    break
                            if dropped > 0:
                                print(f"[OnlineStreamingASR] Dropped {dropped} queued chunks after endpoint")
                            
                            self.recognizer.reset(self.stream)
                            self.last_text = ""
                            self.current_speech_duration = 0.0
                            self.segment_start_samples = self.total_samples
                            print("[OnlineStreamingASR] Stream reset, continuing...")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[OnlineStreamingASR] Loop Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.processing_done.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            print(f"[OnlineStreamingASR] Fatal Error: {e}")


class OnlineStreamingASRManager(QThread):
    """
    Manager class to handle Online ASR thread and Queue.
    """
    text_ready = pyqtSignal(str, bool, float)  # text, is_final, timestamp_sec
    processing_done = pyqtSignal()
    error = pyqtSignal(str)
    asr_ready = pyqtSignal()
    speaker_proposal = pyqtSignal(str)
    
    def __init__(self, model_path, config):
        super().__init__()
        self.model_path = model_path
        self.config = config
        # maxsize=200 (~10s audio) để tránh RAM tràn khi ASR lag (Issue B3)
        self.audio_queue = queue.Queue(maxsize=200)
        self.worker = None
        
    def start(self):
        """Start the worker thread"""
        self.worker = OnlineStreamingASRThread(self.model_path, self.config, self.audio_queue)
        self.worker.text_ready.connect(self.text_ready)
        self.worker.processing_done.connect(self.processing_done)
        self.worker.error.connect(self.error)
        self.worker.asr_ready.connect(self.asr_ready)
        self.worker.speaker_proposal.connect(self.speaker_proposal)
        self.worker.start()
        
    def stop(self):
        """Stop the worker thread"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        
        # Clear any remaining audio in queue to prevent old chunks being processed
        # when starting a new recording (Issue: duplicate data)
        cleared = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            print(f"[OnlineStreamingASRManager] Cleared {cleared} old chunks from queue")
            
    def add_audio(self, chunk):
        """Add audio to queue (non-blocking to prevent UI freeze)"""
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest chunk to prevent UI freeze when ASR lags
            print("[OnlineStreamingASR] WARNING: Queue full, dropping oldest chunk")
            try:
                self.audio_queue.get_nowait()  # Remove oldest
                self.audio_queue.put_nowait(chunk)  # Add new
            except queue.Empty:
                pass  # Queue became empty between check
        
    def start_recording(self):
        """Start actual recording/processing state"""
        if self.worker:
            self.worker.start_recording()

    def insert_speaker(self, name):
        """Insert speaker separator safely"""
        if self.worker:
            self.worker.insert_speaker(name)
