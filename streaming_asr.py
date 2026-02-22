"""Real-time ASR using OfflineRecognizer with VAD Trigger and Continuous Streaming"""
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
        self.min_speech_duration = 0.3  # Reduced for faster response
        self.speech_pad_ms = 300 # Amount of audio to prepend (context)
        
        # Ring buffer for context - giảm xuống 0.3s để tránh hallucination (Issue A1)
        # Context quá dài (>0.5s) có thể đưa nhiễu/im lặng vào đầu stream
        self.context_duration = 0.3
        self.maxlen = int(self.context_duration * sample_rate / self.window_size)
        self.ring_buffer = collections.deque(maxlen=self.maxlen)
        
        # Buffer for VAD processing (to ensure 512 sample chunks)
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
            # Use snakers4/silero-vad
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
            return True, 1.0 # Fallback: always speech if no VAD
            
        # Convert to float32 if needed
        if isinstance(audio_chunk, bytes):
            audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk

        # Add to ring buffer for context
        self.ring_buffer.append(audio_float)
        
        # Add to VAD buffer
        self.vad_buffer = np.concatenate([self.vad_buffer, audio_float])
        
        # Only process if we have >= 512 samples (Silero requirement)
        if len(self.vad_buffer) < 512:
            return None, None
            
        # torch imported at module level for performance (Issue #7)
        max_prob = 0.0
        processed_any = False
        
        # Process in 512-sample chunks
        while len(self.vad_buffer) >= 512:
            chunk = self.vad_buffer[:512]
            self.vad_buffer = self.vad_buffer[512:]
            
            # torch đã import ở module level (B6 fix)
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
        """Get concatenated audio from ring buffer (for prepending)"""
        if not self.ring_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.ring_buffer))
    
    def clear_buffer(self):
        self.ring_buffer.clear()
        self.vad_buffer = np.array([], dtype=np.float32)

    def reset(self):
        """Reset VAD state completely"""
        self.clear_buffer()
        if hasattr(self.vad_model, 'reset_states'):
            try:
                self.vad_model.reset_states()
            except:
                pass


class StreamingASRThread(QThread):
    """
    Continuous Streaming ASR Thread.
    Lifecycle:
    - IDLE: Checking VAD. If Speech Start -> enter RECORDING, create stream.
    - RECORDING: Feed audio to stream. Check VAD for Silence. 
      - If Silence > 0.5s -> Commit Final, Reset Stream, enter IDLE.
    """
    text_ready = pyqtSignal(str, bool, float) # text, is_final, timestamp_sec
    processing_done = pyqtSignal()
    error = pyqtSignal(str)
    asr_ready = pyqtSignal()
    speaker_proposal = pyqtSignal(str) # New signal for UI preview
    
    def __init__(self, model_path, config, audio_queue):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.audio_queue = audio_queue
        
        self.is_running = False
        self.is_recording_active = False # Controls external Pause/Resume
        
        self.state = 'IDLE' # 'IDLE' or 'RECORDING'
        self.vad = None # Initialized in run()
        
        # Stream management
        self.recognizer = None
        self.stream = None
        
        # Logic params
        self.silence_frames = 0
        self.speech_frames = 0
        self.max_silence_chunks = int(0.6 / 0.05) # ~12 chunks = 0.6s
        self.max_speech_duration = 15.0 # seconds (Reduced to 15s to avoid lag)
        self.current_speech_duration = 0.0
        
        # For partial results
        self.last_partial_text = ""
        self.stream_partial_text = ""  # Track partial text of current stream
        
        # Decode Throttle
        self.last_decode_time = 0
        self.decode_interval = 1.0 # Decode every 1.0 second
        
        # Sample counter for timestamp tracking
        self.total_samples = 0
        self.segment_start_samples = 0
        
        # Pending actions
        self.pending_speaker = None
        self.pending_speaker_mutex = False # Basic thread safety via variable assignment logic

    def stop(self):
        self.is_running = False
        self.is_recording_active = False

    def start_recording(self):
        """External trigger to enable processing"""
        self.is_recording_active = True
        if self.vad:
            self.vad.reset()
        self.state = 'IDLE'
        print("[StreamingASR] Started listening...")

    def insert_speaker(self, name):
        """Request to insert a speaker separator"""
        self.pending_speaker = name
        self.speaker_proposal.emit(name)

    def run(self):
        try:
            import sherpa_onnx
            import glob
            import time
            import torch
            
            self.is_running = True
            
            # Limit PyTorch threads for VAD
            cpu_threads = self.config.get("cpu_threads", 4)
            try:
                torch.set_num_threads(cpu_threads)
                torch.set_num_interop_threads(1)
                print(f"[StreamingASR] Set PyTorch threads: {cpu_threads}")
            except RuntimeError as e:
                # Already set in previous session, ignore
                print(f"[StreamingASR] PyTorch threads already set: {e}")
            
            # 0. Initialize VAD (in thread)
            print("[StreamingASR] Initializing VAD...")
            self.vad = VADTrigger()
            
            # 1. Load Model
            print(f"[StreamingASR] Loading model from {self.model_path}")
            encoder_files = glob.glob(os.path.join(self.model_path, "encoder*.onnx"))
            decoder_files = glob.glob(os.path.join(self.model_path, "decoder*.onnx"))
            joiner_files = glob.glob(os.path.join(self.model_path, "joiner*.onnx"))
            tokens = os.path.join(self.model_path, "tokens.txt")
            
            if not (encoder_files and decoder_files and joiner_files and os.path.exists(tokens)):
                self.error.emit(f"Model files missing in {self.model_path}")
                return

            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder_files[0],
                decoder=decoder_files[0],
                joiner=joiner_files[0],
                num_threads=1,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search"
            )
            
            print("[StreamingASR] Model loaded.")
            self.asr_ready.emit()
            
            # Wait for start_recording trigger
            while self.is_running and not self.is_recording_active:
                self.msleep(50)
            
            while self.is_running:
                try:
                    # Get chunk (approx 50ms)
                    chunk_bytes = self.audio_queue.get(timeout=0.1)
                    
                    if not self.is_recording_active:
                        continue
                        
                    # Pre-convert to float32
                    audio_float = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Count samples for timestamp tracking
                    self.total_samples += len(audio_float)
                    
                    # Run VAD
                    is_speech, prob = self.vad.process(audio_float)
                    
                    # === HANDLE PENDING SPEAKER (IDLE CASE) ===
                    # If we are IDLE, we can insert immediately.
                    # If we are RECORDING, we wait until the end of the current utterance (should_end logic).
                    if self.pending_speaker and self.state == 'IDLE':
                        speaker_name = self.pending_speaker
                        self.pending_speaker = None
                        print(f"[StreamingASR] Inserting speaker (IDLE): {speaker_name}")
                        token = f" __SPK_SEP__{speaker_name}__SPK_SEP__"
                        self.text_ready.emit(token, True, 0.0)  # Speaker token has no timestamp
                        self.speaker_proposal.emit("") # Clear proposal since it's committed
                    
                    # === STATE MACHINE ===
                    
                    # Always feed audio to ASR if Recording, regardless of VAD decision for this specific micro-chunk
                    if self.state == 'RECORDING':
                        # Feed current chunk
                        self.stream.accept_waveform(16000, audio_float)
                        self.current_speech_duration += (len(audio_float) / 16000.0)
                    
                    # If VAD has no decision (insufficient data), skip state logic
                    if is_speech is None:
                        continue
                        
                    if self.state == 'IDLE':
                        if is_speech:
                            print("[StreamingASR] Speech detected -> Starting Stream")
                            self.state = 'RECORDING'
                            self.silence_frames = 0
                            self.current_speech_duration = 0.0
                            self.last_partial_text = ""
                            self.stream_partial_text = ""
                            self.last_decode_time = time.time()
                            
                            # Record start sample for this segment
                            self.segment_start_samples = self.total_samples
                            
                            # Create new stream
                            if self.stream is not None:
                                pass # Should be None in IDLE usually
                            self.stream = self.recognizer.create_stream()
                            
                            # Feed context (ring buffer) to capture start of sentence
                            context = self.vad.get_context()
                            self.stream.accept_waveform(16000, context)
                            
                    elif self.state == 'RECORDING':
                        
                        # VAD Logic for End detection (only process if is_speech is not None)
                        if is_speech is not None:
                            if is_speech:
                                self.silence_frames = 0
                            else:
                                self.silence_frames += 1
                        
                        # Check End conditions
                        should_end = False
                        
                        # 1. Silence timeout
                        if self.silence_frames > self.max_silence_chunks:
                            print(f"[StreamingASR] Silence timeout ({self.silence_frames} chunks)")
                            should_end = True
                            
                        # 2. Max duration
                        if self.current_speech_duration > self.max_speech_duration:
                            print("[StreamingASR] Max duration reached")
                            should_end = True
                        
                        # 3. Force end if pending speaker (cut immediately when speaker is requested)
                        force_end_for_speaker = False
                        if self.pending_speaker:
                            print(f"[StreamingASR] Force end segment due to pending speaker: {self.pending_speaker}")
                            should_end = True
                            force_end_for_speaker = True
                            
                        if should_end:
                            # === SPEECH END ===
                            # Determine end reason for different handling
                            if force_end_for_speaker:
                                end_reason = 'speaker_change'
                            elif self.current_speech_duration > self.max_speech_duration:
                                end_reason = 'max_duration'
                            else:
                                end_reason = 'silence'
                            
                            # 1. Feed tail silence (300ms)
                            tail_padding = np.zeros(int(0.3 * 16000), dtype=np.float32)
                            self.stream.accept_waveform(16000, tail_padding)
                            
                            # 2. Final Decode
                            self.recognizer.decode_stream(self.stream)
                            text = self.stream.result.text.strip().lower()
                            
                            if text:
                                print(f"[StreamingASR] Final: {text}")
                                # Use segment start time as base, but word-level timestamps will be calculated in app.py
                                timestamp_sec = self.segment_start_samples / 16000.0
                                self.text_ready.emit(text, True, timestamp_sec)
                            
                            # Emit Pending Speaker if queued during this utterance
                            if self.pending_speaker:
                                speaker_name = self.pending_speaker
                                self.pending_speaker = None
                                print(f"[StreamingASR] Inserting speaker (Post-Speech): {speaker_name}")
                                token = f" __SPK_SEP__{speaker_name}__SPK_SEP__"
                                self.text_ready.emit(token, True, 0.0)  # Fix Issue #5: add timestamp argument
                                self.speaker_proposal.emit("") # Clear proposal
                            
                            # 3. Reset - Only drop chunks on silence timeout, not max_duration or speaker_change
                            # When max_duration or speaker_change is reached, keep queue for next segment to avoid audio gaps
                            if end_reason == 'silence':
                                dropped = 0
                                while self.audio_queue.qsize() > 5:  # Keep a small buffer for continuity
                                    try:
                                        self.audio_queue.get_nowait()
                                        dropped += 1
                                    except queue.Empty:
                                        break
                                if dropped > 0:
                                    print(f"[StreamingASR] Dropped {dropped} queued chunks after silence")
                            else:
                                print(f"[StreamingASR] {end_reason}: keeping queue for continuity")
                            
                            self.stream = None
                            self.state = 'IDLE'
                            self.vad.reset() # Reset VAD state to avoid noise trigger
                            
                        else:
                            # === CONTINUOUS DECODE (Partial) ===
                            # Backpressure check: If queue is large (>10 chunks ~ 0.5s), skip partial decode to catch up
                            is_lagging = self.audio_queue.qsize() > 10
                            
                            # Throttle decoding to save CPU
                            current_time = time.time()
                            if not is_lagging and (current_time - self.last_decode_time > self.decode_interval):
                                self.recognizer.decode_stream(self.stream)
                                text = self.stream.result.text.strip().lower()
                                
                                if text and text != self.last_partial_text:
                                    # Calculate timestamp based on segment start time + elapsed duration (Issue #2)
                                    # This is more accurate than total_samples - decode_interval
                                    segment_start_sec = self.segment_start_samples / 16000.0
                                    elapsed_since_start = (self.total_samples - self.segment_start_samples) / 16000.0
                                    # Partial timestamp is segment start + portion of elapsed time
                                    timestamp_sec = segment_start_sec + max(0, elapsed_since_start - self.decode_interval)
                                    if timestamp_sec < segment_start_sec:
                                        timestamp_sec = segment_start_sec
                                    self.text_ready.emit(text, False, timestamp_sec)
                                    self.last_partial_text = text
                                
                                self.last_decode_time = current_time
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[StreamingASR] Loop Error: {e}")
                    import traceback
                    traceback.print_exc()

            self.processing_done.emit()

        except Exception as e:
            self.error.emit(str(e))
            print(f"[StreamingASR] Fatal Error: {e}")


class StreamingASRManager(QThread):
    """
    Manager class to handle ASR thread and Queue.
    """
    text_ready = pyqtSignal(str, bool, float) # text, is_final, timestamp_sec
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
        self.worker = StreamingASRThread(self.model_path, self.config, self.audio_queue)
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
            print(f"[StreamingASRManager] Cleared {cleared} old chunks from queue")
            
    def add_audio(self, chunk):
        """Add audio to queue (non-blocking to prevent UI freeze)"""
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest chunk to prevent UI freeze when ASR lags
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
