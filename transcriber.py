import os
import re
import sys
import time
import unicodedata
from PyQt6.QtCore import QThread, pyqtSignal
from punctuation_restorer_improved import ImprovedPunctuationRestorer

# sherpa_onnx is imported lazily to ensure DLL paths are set up first
_sherpa_onnx = None

def get_sherpa_onnx():
    """Lazy import sherpa_onnx to ensure DLL paths are set up first."""
    global _sherpa_onnx
    if _sherpa_onnx is None:
        import sherpa_onnx as so
        _sherpa_onnx = so
    return _sherpa_onnx

# Import SAT pipeline
try:
    from sat_segmenter import SATPunctuationPipeline, ChunkedSATProcessor, SATSegmenter
    SAT_AVAILABLE = True
except ImportError:
    SAT_AVAILABLE = False

try:
    from speaker_diarization import SpeakerDiarizer
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    DIARIZATION_AVAILABLE = False
    print(f"[Transcriber] Speaker diarization not available: {e}")

# Global var for tracking diarization progress
_diarization_last_progress = 0
_diarization_progress_emitter = None  # Callback để emit progress đến GUI

def diarization_progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    """
    Progress callback for speaker diarization.
    Returns 0 to continue, non-zero to stop.
    """
    global _diarization_last_progress, _diarization_progress_emitter
    if num_total_chunks == 0:
        return 0
    
    progress = int(num_processed_chunk / num_total_chunks * 100)
    
    # Only update every 5% to minimize impact on processing speed
    # while still keeping UI responsive.
    if progress >= _diarization_last_progress + 5 or num_processed_chunk == num_total_chunks:
        _diarization_last_progress = progress
        print(f"[Diarization] Progress: {progress}% ({num_processed_chunk}/{num_total_chunks})")
        
        # Emit progress đến GUI nếu có callback
        # Map diarization progress (0-100) to phase progress (10-85)
        # Diarization chiếm phần lớn thởi gian nên cần chiếm phần lớn progress bar
        if _diarization_progress_emitter:
            phase_progress = 10 + int(progress * 0.75)  # 10% -> 85%
            _diarization_progress_emitter(phase_progress)
        
        # Yield CPU time to allow GUI to update.
        # Sherpa-onnx C++ code holds GIL during processing, blocking the GUI.
        # This 1ms sleep gives the GUI thread a chance to process events.
        QThread.msleep(1)
    
    return 0

class TranscriberThread(QThread):
    progress = pyqtSignal(str) # Emits log messages
    finished = pyqtSignal(str, dict) # Emits final text and timing info (dict)
    error = pyqtSignal(str)    # Emits error message

    def __init__(self, file_path, model_path, config):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.config = config
        self.is_running = True
    
    def _setup_ffmpeg_path(self):
        """Thiết lập đường dẫn ffmpeg cho pydub"""
        from pydub.utils import which
        
        # Kiểm tra ffmpeg đã có trong PATH chưa
        if which("ffmpeg"):
            return
        
        # Các vị trí phổ biến trên Windows
        possible_paths = [
            os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        
        for ffmpeg_path in possible_paths:
            if os.path.exists(ffmpeg_path):
                # Thiết lập cho pydub
                import pydub
                pydub.AudioSegment.converter = ffmpeg_path
                pydub.AudioSegment.ffmpeg = ffmpeg_path
                pydub.AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
                return

    def run(self):
        try:
            start_time = time.time()
            self.progress.emit("PHASE:Init|Đang khởi tạo mô hình|0")
            
            # Các biến đo thờigian chi tiết
            timing_details = {
                "upload_convert": 0.0,      # Upload và convert file âm thanh
                "transcription": 0.0,        # Chuyển đổi thành văn bản
                "sentence_segmentation": 0.0,  # Cắt câu
                "punctuation": 0.0,          # Thêm dấu
                "alignment": 0.0,            # Căn chỉnh thờigian
                "diarization": 0.0,          # Phân đoạn Người nói
            }
            phase_start_time = start_time
            
            # Extract config (some are unused but kept for compatibility or future use)
            cpu_threads = self.config.get("cpu_threads", 4)
            restore_punctuation = self.config.get("restore_punctuation", False)
            use_sat_pipeline = self.config.get("use_sat_pipeline", False) and SAT_AVAILABLE
            sat_threshold = self.config.get("sat_threshold", 0.3)  # Ngưỡng tách câu SAT (thấp = tách nhiều)
            sat_paragraph_threshold = self.config.get("sat_paragraph_threshold", 0.3)  # Ngưỡng tách đoạn
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Không tìm thấy thư mục mô hình tại: {self.model_path}\nVui lòng tải mô hình và đặt vào thư mục 'models'.")

            # Define model files
            # Dynamic lookup for model files
            def find_file(pattern):
                try:
                    files = [f for f in os.listdir(self.model_path) if f.startswith(pattern) and f.endswith(".onnx")]
                    # Prefer int8 if available
                    int8_files = [f for f in files if "int8" in f]
                    if int8_files:
                        return os.path.join(self.model_path, int8_files[0])
                    if files:
                        return os.path.join(self.model_path, files[0])
                except Exception:
                    pass
                return None

            encoder = find_file("encoder-")
            decoder = find_file("decoder-")
            joiner = find_file("joiner-")
            tokens = os.path.join(self.model_path, "tokens.txt")
            bpe_model = os.path.join(self.model_path, "bpe.model")
            
            if not all([encoder, decoder, joiner]) or not os.path.exists(tokens):
                 raise FileNotFoundError(f"Thiếu file model (encoder/decoder/joiner/tokens) trong: {self.model_path}")

            self.progress.emit("PHASE:Init|Đang khởi tạo mô hình|30")
            
            # Initialize Sherpa-ONNX recognizer
            # Check if we need to pass dictionary or kwargs
            # from_transducer args: tokens, encoder, decoder, joiner, num_threads, ...
            
            # Prepare kwargs
            kwargs = {
                "tokens": tokens,
                "encoder": encoder,
                "decoder": decoder,
                "joiner": joiner,
                "num_threads": cpu_threads,
                "sample_rate": 16000,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
            }
            pass_bpe = False
            if os.path.exists(bpe_model):
                # We need to verify if from_transducer accepts bpe_model
                # Usually purely defined by presence of file isn't enough, but for these models it likely applies.
                # However, sherpa-onnx python api might differ.
                # The tokens.txt serves as the symbol table.
                pass
            
            recognizer = get_sherpa_onnx().OfflineRecognizer.from_transducer(**kwargs)
            
            self.progress.emit("PHASE:Init|Đang khởi tạo mô hình|60")
            
            self.progress.emit("PHASE:LoadAudio|Đang đọc file audio|0")
            load_audio_start = time.time()
            
            # Load and resample audio using librosa
            import librosa
            import numpy as np
            
            # Load audio at 16kHz mono
            self.progress.emit("PHASE:LoadAudio|Đang chuẩn hóa audio|30")
            
            # Xử lý các định dạng đặc biệt (m4a, ogg, wma) bằng cách chuyển đổi sang wav nếu cần
            file_to_load = self.file_path
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            # Các định dạng cần chuyển đổi
            needs_conversion = ['.m4a', '.ogg', '.wma', '.opus']
            
            if file_ext in needs_conversion:
                try:
                    # Thử load trực tiếp với librosa trước
                    audio, sample_rate = librosa.load(self.file_path, sr=16000, mono=True)
                except Exception as e:
                    # Nếu không được, chuyển đổi sang wav tạm thờii
                    self.progress.emit(f"PHASE:LoadAudio|Đang chuyển đổi {file_ext} sang wav|35")
                    try:
                        from pydub import AudioSegment
                        
                        # Tìm ffmpeg trong các vị trí phổ biến trên Windows
                        self._setup_ffmpeg_path()
                        
                        temp_wav = self.file_path + '.temp.wav'
                        
                        # Xác định format cho pydub
                        format_map = {'.m4a': 'm4a', '.ogg': 'ogg', '.wma': 'wma', '.opus': 'opus'}
                        audio_format = format_map.get(file_ext, file_ext[1:])
                        
                        audio_segment = AudioSegment.from_file(self.file_path, format=audio_format)
                        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                        audio_segment.export(temp_wav, format='wav')
                        file_to_load = temp_wav
                        audio, sample_rate = librosa.load(file_to_load, sr=16000, mono=True)
                        # Xóa file tạm
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    except ImportError:
                        raise ImportError(f"Không thể đọc file {file_ext}. Vui lòng cài đặt pydub: pip install pydub")
                    except Exception as e2:
                        raise Exception(f"Không thể đọc file {file_ext}. Đảm bảo đã cài ffmpeg: {e2}")
            else:
                audio, sample_rate = librosa.load(file_to_load, sr=16000, mono=True)
            
            # Kết thúc đo thờigian load audio
            timing_details["upload_convert"] = time.time() - load_audio_start
            self.progress.emit(f"PHASE:LoadAudio|Đã tải audio ({timing_details['upload_convert']:.1f}s)|100")
            
            # Process in segments to prevent RAM explosion (Sherpa-ONNX infinite context issue)
            import math
            segment_duration = 30 # seconds
            segment_samples = 16000 * segment_duration
            total_samples = len(audio)
            
            # Find silent positions for smart splitting
            def find_silent_regions(audio_data, sample_rate=16000, threshold=0.01, min_silence_duration=0.3):
                """Find silent regions in audio. Returns list of (start_sample, end_sample).
                
                Sử dụng vectorized numpy để tăng tốc độ xử lý.
                """
                frame_length = int(sample_rate * 0.01)  # 10ms frames
                num_frames = len(audio_data) // frame_length
                
                if num_frames == 0:
                    return []
                
                # Vectorized: reshape và tính RMS energy cho tất cả frames
                # Trim audio để chia hết cho frame_length
                trimmed = audio_data[:num_frames * frame_length]
                frames = trimmed.reshape(num_frames, frame_length)
                energies = np.sqrt(np.mean(frames ** 2, axis=1))
                
                # Find silent frames (vectorized)
                is_silent = energies < threshold
                
                # Group consecutive silent frames using diff
                silent_regions = []
                min_frames = int(min_silence_duration / 0.01)
                
                # Tìm các vùng silent bằng cách tìm transitions
                silent_starts = np.where(np.diff(is_silent.astype(int)) == 1)[0] + 1
                silent_ends = np.where(np.diff(is_silent.astype(int)) == -1)[0] + 1
                
                # Xử lý edge cases
                if is_silent[0]:
                    silent_starts = np.insert(silent_starts, 0, 0)
                if is_silent[-1]:
                    silent_ends = np.append(silent_ends, len(is_silent))
                
                # Tạo silent regions
                for start, end in zip(silent_starts, silent_ends):
                    if end - start >= min_frames:
                        start_sample = int(start) * frame_length
                        end_sample = int(end) * frame_length
                        silent_regions.append((start_sample, min(end_sample, len(audio_data))))
                
                return silent_regions
            
            # Pre-compute silent regions
            self.progress.emit("PHASE:LoadAudio|Đang phân tích khoảng lặng|60")
            silent_regions = find_silent_regions(audio)
            
            def find_best_split_point(target_sample, search_window=2*16000):
                """Find the best split point near target_sample, preferring silent regions."""
                search_start = max(0, target_sample - search_window)
                search_end = min(total_samples, target_sample + search_window)
                
                # Find silent regions within search window
                best_point = target_sample  # default to target
                best_distance = float('inf')  # Initialize để tìm min đúng
                
                for silent_start, silent_end in silent_regions:
                    # Check if this silent region is within search window
                    if silent_end >= search_start and silent_start <= search_end:
                        # Use the middle of the silent region as split point
                        mid_silent = (silent_start + silent_end) // 2
                        distance = abs(mid_silent - target_sample)
                        if distance < best_distance:
                            best_distance = distance
                            best_point = mid_silent
                
                return best_point
            
            # Build segment boundaries using smart splitting
            segment_boundaries = [0]  # Start at 0
            current_pos = 0
            
            while current_pos + segment_samples < total_samples:
                target = current_pos + segment_samples
                best_split = find_best_split_point(target)
                # Ensure we make progress (at least 20 seconds)
                if best_split <= current_pos + 20 * 16000:
                    best_split = target  # fallback to exact split
                segment_boundaries.append(best_split)
                current_pos = best_split
            
            segment_boundaries.append(total_samples)  # End at total length
            num_segments = len(segment_boundaries) - 1
            
            full_text_parts = []
            all_words = [] # List of {"text": str, "start": float, "end": float}

            self.progress.emit(f"PHASE:Transcription|Đang chuyển thành văn bản|0")
            
            for i in range(num_segments):
                start_sample = segment_boundaries[i]
                end_sample = segment_boundaries[i + 1]
                
                # Check cancellation
                if not self.is_running:
                    return

                # Create a NEW stream for each segment to reset context/memory
                s = recognizer.create_stream()
                
                chunk = audio[start_sample:end_sample]
                s.accept_waveform(16000, chunk)
                recognizer.decode_stream(s)
                
                result = s.result
                segment_text = result.text.strip()
                
                if segment_text:
                    # Zipformer output is uppercase, normalize it
                    segment_text = segment_text.lower()
                    full_text_parts.append(segment_text)
                    
                    # Process timestamps if available
                    if hasattr(result, 'timestamps') and hasattr(result, 'tokens'):
                        ts = result.timestamps
                        toks = result.tokens
                        
                        # Calculate time offset for this chunk (in seconds)
                        time_offset = start_sample / 16000
                        
                        for j, (t_val, tok) in enumerate(zip(ts, toks)):
                            start_abs = t_val + time_offset
                            
                            # Estimate end time using next token or arbitrary duration (e.g. 0.3s)
                            if j < len(ts) - 1:
                                end_abs = ts[j+1] + time_offset
                            else:
                                end_abs = start_abs + 0.3 # default duration for last token
                            
                            # Normalize token
                            tok_display = tok.lower()
                            
                            all_words.append({
                                "text": tok_display, 
                                "start": start_abs, 
                                "end": end_abs
                            })
                
                # Update progress for transcription phase (0% -> 100%)
                percent = int((i + 1) / num_segments * 100)
                self.progress.emit(f"PHASE:Transcription|Đang chuyển thành văn bản|{percent}")
                
                # Explicit delete to help GC
                del s
            
            full_text = " ".join(full_text_parts)
            
            # Post-processing for Zipformer output (often lower cased and generic)
            # Zipformer output for Vietnamese might need basic formatting
            if full_text:
                full_text = full_text.capitalize()
            
            # --- MERGE BPE TOKENS INTO WORDS ---
            # Sherpa-ONNX trả về BPE tokens (e.g., " kính", "thưa", " đồng")
            # Tokens bắt đầu bằng SPACE -> word mới
            # Cần gộp thành words hoàn chỉnh để alignment chính xác
            def merge_tokens_to_words(tokens_list):
                """Merge BPE tokens into complete words.
                
                Sherpa-ONNX convention: SPACE prefix indicates start of new word.
                Example: [' kính', 'thưa', ' đồng'] -> ['kính', 'thưa', 'đồng']
                """
                if not tokens_list:
                    return []
                
                merged = []
                current_word = None
                
                for tok_info in tokens_list:
                    tok = tok_info["text"]
                    # Token bắt đầu bằng SPACE -> bắt đầu word mới
                    if tok.startswith(" "):
                        # Lưu word cũ nếu có
                        if current_word is not None:
                            merged.append(current_word)
                        # Tạo word mới (bỏ space prefix)
                        current_word = {
                            "text": tok.lstrip(" "),
                            "start": tok_info["start"],
                            "end": tok_info["end"]
                        }
                    else:
                        # Nối vào word hiện tại
                        if current_word is not None:
                            current_word["text"] += tok
                            current_word["end"] = tok_info["end"]
                        else:
                            # Edge case: token đầu tiên không có space
                            current_word = {
                                "text": tok,
                                "start": tok_info["start"],
                                "end": tok_info["end"]
                            }
                
                # Đừng quên word cuối cùng
                if current_word is not None:
                    merged.append(current_word)
                
                return merged
            
            # Thực hiện merge
            if all_words:
                original_count = len(all_words)
                all_words = merge_tokens_to_words(all_words)
                merged_count = len(all_words)
                print(f"[Transcriber] Merged {original_count} BPE tokens into {merged_count} words")
            
            transcription_end_time = time.time()
            timing_details["transcription"] = transcription_end_time - start_time - timing_details["upload_convert"]
            transcribe_duration = transcription_end_time - start_time
            
            restore_duration = 0.0
            
            final_segments = []
            paragraphs = []  # Lưu paragraph boundaries từ SAT

            if restore_punctuation and full_text:
                try:
                    if use_sat_pipeline and SAT_AVAILABLE:
                        # --- CHUNKED SAT PIPELINE: Tách câu theo chunk → Gán dấu từng câu ---
                        save_ram = self.config.get("save_ram", False)
                        punct_confidence = self.config.get("punctuation_confidence", 0.3)
                        
                        self.progress.emit("PHASE:SAT|Đang khởi tạo SAT model|0")
                        
                        # Initialize SAT segmenter
                        sat_segmenter = SATSegmenter(
                            threshold=sat_threshold,
                            paragraph_threshold=sat_paragraph_threshold
                        )
                        sat_segmenter.initialize()
                        
                        # Create chunked processor
                        chunked_processor = ChunkedSATProcessor(
                            chunk_size=500,  # 500 words per chunk
                            segmenter=sat_segmenter
                        )
                        
                        # Progress callback for chunked SAT
                        def sat_progress_callback(current_chunk, total_chunks, sentences_so_far):
                            percent = int((current_chunk / total_chunks) * 100)
                            self.progress.emit(f"PHASE:SAT|Đang tách câu (chunk {current_chunk}/{total_chunks})|{percent}")
                        
                        # Early exit check
                        def should_stop():
                            return not self.is_running
                        
                        sat_start = time.time()
                        
                        # Process text in chunks
                        sentences = chunked_processor.process_chunked(
                            full_text,
                            threshold=sat_threshold,
                            progress_callback=sat_progress_callback,
                            should_stop=should_stop
                        )
                        
                        # Check if cancelled
                        if not self.is_running:
                            return
                        
                        timing_details["sentence_segmentation"] = time.time() - sat_start
                        
                        # Unload SAT model if save_ram is enabled
                        if save_ram:
                            sat_segmenter.unload()
                            import gc
                            gc.collect()
                            self.progress.emit("PHASE:Punctuation|Đã giải phóng SAT model|0")
                        
                        # Punctuation restoration per sentence
                        self.progress.emit("PHASE:Punctuation|Đang khởi tạo Punctuation model|0")
                        punct_restorer = ImprovedPunctuationRestorer(device="cpu", confidence=punct_confidence)
                        
                        punct_start = time.time()
                        punctuated_sentences = []
                        total_sentences = len(sentences)
                        
                        for i, sentence in enumerate(sentences):
                            # Check for early exit
                            if not self.is_running:
                                return
                            
                            punctuated = punct_restorer.restore(sentence)
                            punctuated_sentences.append(punctuated)
                            
                            # Progress per sentence (every 5 sentences or at end)
                            if (i + 1) % 5 == 0 or i + 1 == total_sentences:
                                percent = int((i + 1) / total_sentences * 100)
                                self.progress.emit(f"PHASE:Punctuation|Đang thêm dấu câu ({i+1}/{total_sentences})|{percent}")
                        
                        # Update sentences with punctuated versions
                        sentences = punctuated_sentences
                        
                        # Unload punctuation model if save_ram is enabled
                        if save_ram:
                            punct_restorer.unload()
                            import gc
                            gc.collect()
                        
                        timing_details["punctuation"] = time.time() - punct_start
                        
                        # Reconstruct full text
                        full_text = '. '.join(sentences)
                        if full_text and not full_text.endswith('.'):
                            full_text += '.'
                        
                        # Empty paragraphs for chunked mode (no paragraph info)
                        paragraphs = []
                        
                        self.progress.emit("PHASE:Align|Đang căn chỉnh thời gian|0")
                        
                    else:
                        # --- ORIGINAL PIPELINE: Gán dấu toàn bộ → Tách câu bằng regex ---
                        self.progress.emit("PHASE:Punctuation|Đang thêm dấu câu|0")
                        
                        punct_confidence = self.config.get("punctuation_confidence", 0.3)
                        punct_start = time.time()
                        restorer = ImprovedPunctuationRestorer(device="cpu", confidence=punct_confidence)
                        restored_text_raw = restorer.restore(full_text)
                        full_text = restored_text_raw
                        timing_details["punctuation"] = time.time() - punct_start
                        
                        self.progress.emit("PHASE:Punctuation|Đang thêm dấu câu|100")
                        
                        self.progress.emit("PHASE:Align|Đang căn chỉnh thời gian|0")
                        
                        # Split by sentence endings (. ? !)
                        sentences = re.split(r'(?<=[.?!])\s+', full_text)
                        
                        # Không có paragraph info với original pipeline
                        paragraphs = []
                    
                    # --- ALIGNMENT LOGIC (dùng chung cho cả 2 pipeline) ---
                    # Sử dụng thuật toán DTW-style alignment để mapping từ ASR tokens sang câu sau punctuation
                    align_start = time.time()
                    current_word_idx = 0
                    total_sentences = len(sentences)
                    last_align_progress = 0
                    
                    def normalize_word(word):
                        """Normalize word for comparison: lowercase, remove punctuation, no spaces.
                        
                        Also applies Unicode NFD normalization for consistent Vietnamese comparison.
                        """
                        word = word.lower().strip()
                        # Loại bỏ dấu câu phổ biến (giữ lại chữ cái Unicode)
                        word = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE)
                        # Xóa khoảng trắng
                        word = word.replace(' ', '')
                        return word
                    
                    def find_word_sequence_match(asr_words, target_words, start_idx, max_look_ahead=50):
                        """
                        Tìm vị trí bắt đầu và kết thúc của target_words trong asr_words.
                        Sử dụng fuzzy matching để xử lý các trường hợp ASR sai hoặc thêm/bớt từ.
                        
                        Returns: (start_idx, end_idx) hoặc (None, None) nếu không tìm thấy
                        """
                        if not target_words:
                            return None, None
                        
                        first_target = normalize_word(target_words[0])
                        if not first_target:
                            return None, None
                        
                        end_search = min(start_idx + max_look_ahead, len(asr_words))
                        best_match = None
                        best_score = 0
                        
                        for i in range(start_idx, end_search):
                            asr_word = normalize_word(asr_words[i]['text'])
                            
                            # Kiểm tra từ đầu tiên có khớp không
                            if asr_word == first_target or (len(asr_word) > 2 and len(first_target) > 2 and 
                                                            (asr_word in first_target or first_target in asr_word)):
                                # Thử match toàn bộ sequence
                                matched_count = 1
                                last_matched_idx = i
                                asr_offset = 0  # Offset để tracking skip
                                
                                for j in range(1, len(target_words)):
                                    target_word = normalize_word(target_words[j])
                                    if not target_word:
                                        matched_count += 1
                                        continue
                                    
                                    asr_idx = i + j + asr_offset
                                    if asr_idx >= len(asr_words):
                                        break
                                    
                                    asr_target_word = normalize_word(asr_words[asr_idx]['text'])
                                    if asr_target_word == target_word or (len(asr_target_word) > 2 and len(target_word) > 2 and
                                                                          (asr_target_word in target_word or target_word in asr_target_word)):
                                        matched_count += 1
                                        last_matched_idx = asr_idx
                                    else:
                                        # Cho phép skip 1 từ trong ASR (có thể là filler word hoặc lỗi)
                                        if asr_idx + 1 < len(asr_words):
                                            asr_next = normalize_word(asr_words[asr_idx + 1]['text'])
                                            if asr_next == target_word or (len(asr_next) > 2 and len(target_word) > 2 and
                                                                           (asr_next in target_word or target_word in asr_next)):
                                                matched_count += 1
                                                last_matched_idx = asr_idx + 1
                                                asr_offset += 1  # Tăng offset sau khi skip
                                                continue  # Tiếp tục match từ tiếp theo
                                        # Không skip được → thoát vòng lặp nội
                                        break
                                
                                # Tính score
                                score = matched_count / len(target_words)
                                if score > best_score:
                                    best_score = score
                                    best_match = (i, last_matched_idx)
                                
                                # Nếu match hoàn hảo, return ngay
                                if score >= 0.95:
                                    break
                        
                        # Chấp nhận match nếu >= 70% từ khớp
                        if best_score >= 0.7:
                            return best_match
                        return None, None
                    
                    for sent_idx, sent in enumerate(sentences):
                        if not sent.strip(): 
                            progress = int((sent_idx + 1) / total_sentences * 100)
                            if progress >= last_align_progress + 10:
                                self.progress.emit(f"PHASE:Align|Đang căn chỉnh thờigian|{progress}")
                                last_align_progress = progress
                            continue
                        
                        sent_words = [w for w in sent.split() if w.strip()]
                        
                        if not sent_words: 
                            progress = int((sent_idx + 1) / total_sentences * 100)
                            if progress >= last_align_progress + 10:
                                self.progress.emit(f"PHASE:Align|Đang căn chỉnh thờigian|{progress}")
                                last_align_progress = progress
                            continue
                        
                        # Tách từ và normalize (bỏ dấu câu để so sánh)
                        sent_words_clean = [normalize_word(w) for w in sent_words]
                        sent_words_clean = [w for w in sent_words_clean if w]
                        
                        start_t = -1
                        end_t = -1
                        
                        # Sử dụng fuzzy sequence matching
                        match_start, match_end = find_word_sequence_match(
                            all_words, sent_words_clean, current_word_idx
                        )
                        
                        if match_start is not None:
                            start_t = all_words[match_start]['start']
                            end_t = all_words[match_end]['end']
                            current_word_idx = match_end + 1
                        else:
                            # Fallback: tìm từ đầu tiên bằng cách đơn giản
                            first_word = normalize_word(sent_words_clean[0]) if sent_words_clean else ""
                            temp_idx = current_word_idx
                            while temp_idx < len(all_words):
                                asr_word = normalize_word(all_words[temp_idx]['text'])
                                if first_word in asr_word or asr_word in first_word:
                                    start_t = all_words[temp_idx]['start']
                                    # Estimate end dựa trên số từ và duration trung bình
                                    # Sử dụng median duration từ các words đã match trước đó nếu có
                                    if current_word_idx > 0 and all_words:
                                        # Tính trung bình duration của các words đã qua
                                        past_durations = [
                                            all_words[k]['end'] - all_words[k]['start'] 
                                            for k in range(min(current_word_idx, len(all_words)))
                                            if all_words[k]['end'] > all_words[k]['start']
                                        ]
                                        avg_word_duration = sum(past_durations) / len(past_durations) if past_durations else 0.3
                                    else:
                                        avg_word_duration = 0.3
                                    
                                    estimated_duration = len(sent_words_clean) * avg_word_duration
                                    end_t = min(start_t + estimated_duration, 
                                               all_words[-1]['end'] if all_words else start_t + 1.0)
                                    # NHẢY ĐÚNG SỐ TỪ để tránh overlap với câu sau
                                    current_word_idx = min(temp_idx + len(sent_words_clean), len(all_words))
                                    break
                                temp_idx += 1
                            
                            if start_t == -1:
                                # Last resort fallback - đảm bảo không quay lại vị trí cũ
                                fallback_idx = min(current_word_idx, len(all_words)-1) if all_words else 0
                                start_t = all_words[fallback_idx]['start'] if all_words else 0.0
                                end_t = start_t + 1.0
                                # Vẫn nhảy current_word_idx để tránh câu sau bị lặp
                                current_word_idx = min(current_word_idx + len(sent_words_clean), len(all_words))

                        final_segments.append({
                            "text": sent,
                            "start": start_t,
                            "end": end_t
                        })
                        
                        # Emit progress mỗi 10%
                        progress = int((sent_idx + 1) / total_sentences * 100)
                        if progress >= last_align_progress + 10:
                            self.progress.emit(f"PHASE:Align|Đang căn chỉnh thờigian|{progress}")
                            last_align_progress = progress
                    
                    # Kết thúc đo thờigian alignment
                    timing_details["alignment"] = time.time() - align_start

                except Exception as e:
                     self.progress.emit(f"Lỗi khi thêm dấu câu: {e}")
                     import traceback
                     traceback.print_exc()
                     
                     restore_end_time = time.time()
                     restore_duration = restore_end_time - transcription_end_time
                     timing_details["punctuation"] = restore_duration
            
            else:
                 # No punctuation restoration - still need alignment progress
                 self.progress.emit("PHASE:Align|Đang căn chỉnh thời gian|0")
                 # If no restore, we just return the full text.
                 # But we can still return rough segments based on pauses?
                 # ideally we just make one big segment or many small word segments.
                 # Let's check if the text has any punctuation from the model (Zipformer usually doesn't).
                 # If not, let's just make one big segment for now OR segment by 30s chunks?
                 # User wants to click sentences. Without punctuation, "sentences" are ill-defined.
                 # We can just return 'all_words' as segments? Too granular.
                 # Let's chunk by 5-10 words or pauses > 0.5s.
                 
                 current_seg_words = []
                 current_start = -1
                 
                 for i, w in enumerate(all_words):
                     if current_start == -1: current_start = w['start']
                     current_seg_words.append(w['text'])
                     
                     # Check for pause
                     is_pause = False
                     if i < len(all_words) - 1:
                         if all_words[i+1]['start'] - w['end'] > 0.8:
                             is_pause = True
                     
                     if is_pause or len(current_seg_words) > 15:
                         final_segments.append({
                             "text": "".join(current_seg_words).strip(),
                             "start": current_start,
                             "end": w['end']
                         })
                         current_seg_words = []
                         current_start = -1
                 
                 if current_seg_words:
                      final_segments.append({
                             "text": "".join(current_seg_words).strip(),
                             "start": current_start,
                             "end": all_words[-1]['end']
                         })
                 
                 self.progress.emit("PHASE:Align|Đang căn chỉnh thời gian|100")

            # --- SPEAKER DIARIZATION ---
            speaker_segments = []
            speaker_segments_raw = []
            diarization_start = None
            if self.config.get("speaker_diarization", False) and DIARIZATION_AVAILABLE:
                try:
                    diarization_start = time.time()
                    self.progress.emit("PHASE:Diarization|Đang phân tách Người nói|0")
                    
                    num_speakers = self.config.get("num_speakers", 2)
                    speaker_model_id = self.config.get("speaker_model", "titanet_small")
                    diarizer = SpeakerDiarizer(
                        embedding_model_id=speaker_model_id,
                        num_clusters=num_speakers,
                        num_threads=self.config.get("cpu_threads", 4),
                        threshold=self.config.get("diarization_threshold", 0.6)
                    )
                    
                    # Get raw speaker segments first
                    self.progress.emit("PHASE:Diarization|Đang phân tách Người nói|10")
                    diarizer.initialize()
                    
                    # Reset progress tracker
                    global _diarization_last_progress, _diarization_progress_emitter
                    _diarization_last_progress = 0
                    
                    # Thiết lập callback để emit progress đến GUI
                    def emit_diarization_progress(progress_val):
                        self.progress.emit(f"PHASE:Diarization|Đang phân tách Người nói|{progress_val}")
                    
                    _diarization_progress_emitter = emit_diarization_progress
                    
                    # Process with progress callback to prevent UI lag
                    # Truyền audio đã load để tránh load lại lần 2 (tối ưu RAM và I/O)
                    print("[Transcriber] Starting speaker diarization (process #1)...")
                    self.progress.emit("PHASE:Diarization|Đang phân tách Người nói|20")
                    raw_segments = diarizer.process(
                        self.file_path, 
                        progress_callback=diarization_progress_callback,
                        audio_data=audio,
                        audio_sample_rate=16000
                    )
                    print(f"[Transcriber] Speaker diarization done: {len(raw_segments)} segments")
                    
                    # Convert to dict format for raw storage
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
                    
                    # Xóa callback sau khi hoàn thành
                    _diarization_progress_emitter = None
                    
                    self.progress.emit("PHASE:Diarization|Đang ghép nối với văn bản|85")
                    
                    # Merge with transcription (pass pre-computed segments to avoid running twice)
                    self.progress.emit("PHASE:Diarization|Đang ghép nối với văn bản|87")
                    
                    # Estimate merge progress - gọi callback để cập nhật progress
                    # Merge chiếm 85% -> 95% (10% của progress bar)
                    def merge_progress_wrapper(current, total):
                        if total > 0:
                            merge_pct = int(current / total * 8)  # 87% -> 95%
                            overall = 87 + merge_pct
                            self.progress.emit(f"PHASE:Diarization|Đang ghép nối với văn bản|{overall}")
                    
                    speaker_segments = diarizer.process_with_transcription(
                        self.file_path,
                        final_segments,
                        speaker_segments=raw_segments  # Pass segments from previous process() call
                    )
                    
                    # Small yield after processing to update UI
                    QThread.msleep(5)
                    
                    if speaker_segments:
                        final_segments = speaker_segments
                        self.progress.emit("PHASE:Diarization|Hoàn tất phân tách|98")
                        QThread.msleep(10)
                        self.progress.emit("PHASE:Diarization|Hoàn tất phân tách|100")
                        print(f"[Transcriber] Speaker diarization completed: {len(final_segments)} segments")
                    
                    # Tính thờigian diarization
                    if diarization_start:
                        timing_details["diarization"] = time.time() - diarization_start
                    
                    # Unload diarizer model if save_ram is enabled
                    save_ram = self.config.get("save_ram", False)
                    if save_ram:
                        diarizer.unload()
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    # Xóa callback nếu có lỗi
                    _diarization_progress_emitter = None
                    print(f"[Transcriber] Speaker diarization failed: {e}")
                    import traceback
                    traceback.print_exc()

            total_duration = time.time() - start_time
            self.progress.emit("PHASE:Complete|Hoàn tất|100")
            
            # Tổng hợp timing info chi tiết
            timing_info = {
                "transcription": transcribe_duration,
                "restoration": restore_duration,
                "total": total_duration,
                # Chi tiết từng giai đoạn
                "upload_convert": timing_details["upload_convert"],
                "transcription_detail": timing_details["transcription"],
                "sentence_segmentation": timing_details["sentence_segmentation"],
                "punctuation": timing_details["punctuation"],
                "alignment": timing_details["alignment"],
                "diarization": timing_details["diarization"],
            }
            
            # Pack result - include paragraph info if available
            result_data = {
                "text": full_text,
                "segments": final_segments,
                "timing": timing_info,
                "paragraphs": paragraphs,
                "has_speaker_diarization": len(speaker_segments) > 0,
                "speaker_segments_raw": speaker_segments_raw
            }
            
            # DEBUG: Print final results to console
            print("\n" + "="*70)
            print("[Transcriber] TRANSCRIPTION COMPLETED")
            print("="*70)
            print(f"[Transcriber] Full text length: {len(full_text)} chars")
            print(f"[Transcriber] Total segments: {len(final_segments)}")
            print(f"[Transcriber] Has speaker diarization: {len(speaker_segments) > 0}")
            if speaker_segments_raw:
                print(f"[Transcriber] Speaker segments: {len(speaker_segments_raw)}")
                for i, seg in enumerate(speaker_segments_raw[:10], 1):  # Print first 10
                    print(f"  [{i}] {seg.get('speaker', 'Unknown')}: {seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s")
                if len(speaker_segments_raw) > 10:
                    print(f"  ... and {len(speaker_segments_raw) - 10} more segments")
            print(f"[Transcriber] Total time: {total_duration:.2f}s")
            print("="*70)
            print("[Transcriber] Emitting finished signal...")
            
            self.finished.emit(full_text, result_data)
            print("[Transcriber] Finished signal emitted successfully")
            
        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

    def stop(self):
        self.is_running = False

