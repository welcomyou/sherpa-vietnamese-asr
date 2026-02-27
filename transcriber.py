import logging
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from PyQt6.QtCore import QThread, pyqtSignal

# Setup logger
logger = logging.getLogger(__name__)
from punctuation_restorer_improved import ImprovedPunctuationRestorer

# =============================================================================
# OVERLAP CHUNKING CONFIGURATION
# =============================================================================
OVERLAP_SEC = 3.0  # Overlap duration in seconds (3s = sweet spot)
OVERLAP_SAMPLES = int(OVERLAP_SEC * 16000)  # 48000 samples @ 16kHz
MAX_OVERLAP_WORDS = 100  # Nới lỏng thành 100 từ, trành cắt xén rác vì tiếng Việt phát âm đơn âm tiết nhanh
FUZZY_MATCH_THRESHOLD = 0.8  # Ngưỡng Levenshtein similarity
MIN_MATCH_RATIO = 0.5  # Ngưỡng tối thiểu để chấp nhận overlap match


def normalize_word_for_overlap(word):
    """Chuẩn hóa từ để so sánh overlap: lowercase, bỏ dấu câu, Unicode NFC."""
    word = word.lower().strip()
    word = unicodedata.normalize('NFC', word)
    word = re.sub(r'[^\w]', '', word, flags=re.UNICODE)
    return word


def words_match(w1, w2, threshold=FUZZY_MATCH_THRESHOLD):
    """
    So sánh 2 từ đã normalize.
    Returns True nếu:
    - Giống hệt, HOẶC
    - Một từ chứa từ kia (substring), HOẶC
    - Levenshtein similarity >= threshold
    """
    if w1 == w2:
        return True
    if not w1 or not w2:
        return False
    # Substring match (cho trường hợp BPE cắt khác nhau)
    if len(w1) > 2 and len(w2) > 2:
        if w1 in w2 or w2 in w1:
            return True
    # Levenshtein similarity
    ratio = SequenceMatcher(None, w1, w2).ratio()
    return ratio >= threshold


def find_overlap_alignment(tail_words, head_words):
    """
    Tìm alignment tối ưu giữa tail (cuối chunk trước) và head (đầu chunk sau)
    sử dụng thuật toán LCS (Longest Common Subsequence) kết hợp fuzzy matching.
    
    Trả về: (best_cut_index_in_head, action, num_words_to_drop)
        - best_cut_index_in_head: index đầu tiên trong head_words MÀ KHÔNG thuộc overlap
        - action: "cut_head", "drop_tail", "drop_head"
        - num_words_to_drop: Số từ cần pop khỏi merged_words ở chunk trước
    
    Nguyên tắc: 
    Nếu có match đáng kể, nối tiếp.
    Nếu không có match đáng kể giữa 2 vùng overlap, tiến hành drop bề phía có CONFIDENCE (prob) thấp hơn, tránh lặp lại cùng đoạn âm thanh mà lại bị lệch text.
    """
    if not tail_words or not head_words:
        return 0, "none", 0
    
    # Ghi nhận số lượng từ thực sự (nếu drop tail/head sẽ drop toàn bộ số này)
    original_tail_len = len(tail_words)
    original_head_len = len(head_words)
    
    # Giới hạn số từ để tối ưu hiệu năng (nâng MAX_OVERLAP_WORDS = 100)
    tail_words_truncated = tail_words[-MAX_OVERLAP_WORDS:]
    head_words_truncated = head_words[:MAX_OVERLAP_WORDS]
    
    tail_normalized = [normalize_word_for_overlap(w["text"]) for w in tail_words_truncated]
    head_normalized = [normalize_word_for_overlap(w["text"]) for w in head_words_truncated]
    
    best_score = 0
    best_cut_index = 0  # Mặc định: không cắt gì ở head
    best_pop_count = 0  # Mặc định: không pop từ nào ở tail
    
    # Thử tất cả offset có thể: tail bắt đầu khớp với head tại vị trí nào?
    min_offset = -len(tail_normalized) + 1
    max_offset = len(head_normalized)
    
    for offset in range(min_offset, max_offset):
        score = 0
        matched_tail_indices = []
        matched_head_indices = []
        
        for i, tail_w in enumerate(tail_normalized):
            head_idx = i + offset
            if 0 <= head_idx < len(head_normalized):
                if words_match(tail_w, head_normalized[head_idx]):
                    score += 1
                    matched_tail_indices.append(i)
                    matched_head_indices.append(head_idx)
        
        # Tính tỉ lệ match dựa trên cửa sổ overlap thực tế
        overlap_window = min(len(head_normalized), len(tail_normalized) + offset) - max(0, offset)
        match_ratio = score / max(1, overlap_window)
        # Bỏ overall_ratio gò bó, thay bằng số lượng từ overlap tối thiểu tương đối
        
        if score > best_score and match_ratio >= MIN_MATCH_RATIO:
            # Điều kiện thêm: nếu match ít, cần phải nằm sát mép (ví dụ đầu mốc)
            best_score = score
            best_cut_index = matched_head_indices[-1] + 1
            # Tính số từ cần pop khỏi original_tail_words
            # Số từ không match nằm ở cuối: len(tail_truncated) - 1 - match_idx
            # Cộng thêm phần bị cắt bên ngoài (original - truncated) nếu có
            truncated_diff = original_tail_len - len(tail_normalized)
            best_pop_count = (len(tail_normalized) - 1 - matched_tail_indices[-1])
    
    # GUARD: Kiểm tra xem có sự khác biệt (divergence) giữa 2 chuỗi không
    # Nếu hai bên không khớp hoàn hảo trong vùng overlap nhỏ nhất, coi là phân kỳ.
    min_len = min(len(tail_normalized), len(head_normalized))
    
    # Chỉ coi là phân kỳ nếu điểm khớp nhỏ hơn số từ của chuỗi ngắn nhất VÀ có cắt xén rác
    is_diverged = (best_score < min_len) and (best_pop_count > 0)
    
    if best_score == 0 or is_diverged:
        # Tách riêng phần bị lệch (divergent) để so sánh xác suất
        # Nếu score == 0, mâu thuẫn là toàn bộ overlap.
        # Nếu score > 0, mâu thuẫn là đoạn đuôi dư ra sau khớp của Tail và đoạn nằm sau khớp của Head.
        if best_score == 0:
            div_tail = tail_words
            div_head = head_words
        else:
            div_tail = tail_words[-best_pop_count:] if best_pop_count > 0 else []
            div_head = head_words[best_cut_index:] if best_cut_index < len(head_words) else []
            
        tail_prob = sum(w.get("prob", 1.0) for w in div_tail) / max(1, len(div_tail))
        head_prob = sum(w.get("prob", 1.0) for w in div_head) / max(1, len(div_head))
        
        # Log chi tiết các từ và xác suất để debug
        tail_words_str = " ".join([f"{w['text']}({w.get('prob', 1.0):.2f})" for w in div_tail])
        head_words_str = " ".join([f"{w['text']}({w.get('prob', 1.0):.2f})" for w in div_head])
        
        reason = "không có match (score=0)" if best_score == 0 else "đọ phần bị lệch (divergent words)"
        debug_msg = f"[OVERLAP RESOLVE DEBUG] So sánh xác suất do {reason}:\n"
        debug_msg += f"  - TAIL (phần lệch cuối chunk trước): AvgProb={tail_prob:.3f} | Words: {tail_words_str}\n"
        debug_msg += f"  - HEAD (phần lệch đầu chunk sau): AvgProb={head_prob:.3f} | Words: {head_words_str}"
        print(debug_msg)
        logger.info(debug_msg)
        
        # Nếu tail tự tin vượt trội hơn đoạn đầu chuỗi mới
        if tail_prob > head_prob:
            decision_msg = f"[OVERLAP RESOLVE] Quyết định: DROP HEAD (Xóa phần đầu của chunk sau). Tail ({tail_prob:.3f}) > Head ({head_prob:.3f})"
            print(decision_msg)
            logger.info(decision_msg)
            # Xóa phần overlap của head, giữ nguyên chunk trước
            return len(head_words), "drop_head", 0 
        else:
            decision_msg = f"[OVERLAP RESOLVE] Quyết định: DROP TAIL (Xóa tail của chunk trước). Head ({head_prob:.3f}) >= Tail ({tail_prob:.3f})"
            print(decision_msg)
            logger.info(decision_msg)
            return 0, "drop_tail", original_tail_len # Giữ toàn bộ head, xoá TOÀN BỘ tail
            
    # Debug khi lặp chữ an toàn (Perfect match)
    print(f"[OVERLAP RESOLVE] PERFECT MATCH (score={best_score}). Pop {best_pop_count} words from Tail. Cut {best_cut_index} words from Head.")
    return best_cut_index, "cut_head", best_pop_count


def merge_chunks_with_overlap(chunk_results, overlap_duration_sec=OVERLAP_SEC):
    """
    Merge danh sách chunk results, loại bỏ text trùng lặp ở vùng overlap.
    
    Args:
        chunk_results: list of dict, mỗi dict chứa:
            - "words": list of {"text", "start", "end", "local_start", "local_end"}
            - "audio_start_abs": float (giây tuyệt đối trong file gốc)
            - "audio_end_abs": float
            - "overlap_sec": float (thờigian overlap ở đầu chunk)
        overlap_duration_sec: float, thờigian overlap (giây)
    
    Returns:
        merged_words: list of {"text", "start", "end"} - danh sách words toàn bộ audio
        merged_text: str - full text đã merge
    """
    if not chunk_results:
        return [], ""
    
    merged_words = []
    
    for chunk_idx, chunk in enumerate(chunk_results):
        chunk_words = chunk["words"]
        
        if chunk_idx == 0:
            # Chunk đầu tiên: lấy toàn bộ words
            merged_words.extend(chunk_words)
        else:
            # Chunk từ thứ 2 trở đi: cần xử lý overlap
            prev_chunk = chunk_results[chunk_idx - 1]
            prev_words = prev_chunk["words"]
            
            # Lấy tail words từ chunk trước (nằm trong vùng overlap)
            prev_audio_duration = prev_chunk["audio_end_abs"] - prev_chunk["audio_start_abs"]
            overlap_start_local = prev_audio_duration - overlap_duration_sec
            tail_words = [w for w in prev_words if w.get("local_start", 0) >= max(0, overlap_start_local)]
            
            # Lấy head words từ chunk hiện tại (nằm trong vùng overlap)
            head_words = [w for w in chunk_words if w.get("local_start", 0) < overlap_duration_sec]
            
            # Tìm điểm cắt tối ưu
            cut_index, action, pop_count = find_overlap_alignment(tail_words, head_words)
            
            # Xử lý xoá text hụt (trường hợp tự tin bên chunk mới hơn so với rác bên chunk cũ HOẶC đuôi rác khi có match)
            if pop_count > 0:
                print(f"   -> Popping {pop_count} words from merged_words tail")
                del merged_words[-pop_count:]
            
            # Chỉ lấy words SAU điểm cắt (bỏ phần đã trùng)
            remaining_words = chunk_words[cut_index:] if cut_index < len(chunk_words) else []
            
            merged_words.extend(remaining_words)
    
    merged_text = " ".join([w["text"] for w in merged_words])
    return merged_words, merged_text

# sherpa_onnx is imported lazily to ensure DLL paths are set up first
_sherpa_onnx = None

def get_sherpa_onnx():
    """Lazy import sherpa_onnx to ensure DLL paths are set up first."""
    global _sherpa_onnx
    if _sherpa_onnx is None:
        import sherpa_onnx as so
        _sherpa_onnx = so
    return _sherpa_onnx


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


def split_long_segments(segments: list, max_duration: float = 12.0, preserve_raw_words: bool = False) -> list:
    """
    Chia nhỏ các segment dài thành nhiều segment ngắn hơn dựa trên số từ.
    
    Quy tắc chia:
    - >12s: chia làm 2 phần
    - >24s: chia làm 3 phần  
    - >36s: chia làm 4 phần
    - Cứ thêm 12s thì thêm 1 phần
    
    Chia theo số từ (không chia theo thờigian), đảm bảo mỗi phần có số từ đều nhau.
    Timestamp được tính lại tuyến tính dựa trên số từ.
    
    Args:
        segments: List các segment dict với 'text', 'start', 'end'
        max_duration: Thờigian tối đa cho mỗi segment (mặc định 12s)
        
    Returns:
        List các segment đã được chia nhỏ
    """
    if not segments:
        return segments
    
    result = []
    
    for seg in segments:
        duration = seg.get('end', 0) - seg.get('start', 0)
        text = seg.get('text', '').strip()
        
        if duration <= max_duration or not text:
            result.append(seg)
            continue
            
        def process_sub_text(sub_text, sub_start, sub_end, sub_raw_words):
            sub_duration = sub_end - sub_start
            if sub_duration <= max_duration or not sub_text:
                part_seg = {'text': sub_text, 'start': round(sub_start, 3), 'end': round(sub_end, 3)}
                if preserve_raw_words and sub_raw_words:
                    part_seg['raw_words'] = sub_raw_words
                for k, v in seg.items():
                    if k not in ['text', 'start', 'end', 'raw_words']:
                        part_seg[k] = v
                result.append(part_seg)
                return

            # Cần chia đôi/ba nếu vẫn quá dài (chia theo số từ bù trừ tuyến tính - logic cũ)
            num_parts = int(sub_duration / max_duration) + 1
            if sub_duration % max_duration == 0:
                num_parts = int(sub_duration / max_duration)
            num_parts = max(2, num_parts)
            
            words = sub_text.split()
            total_words = len(words)
            if total_words < num_parts:
                part_seg = {'text': sub_text, 'start': round(sub_start, 3), 'end': round(sub_end, 3)}
                if preserve_raw_words and sub_raw_words:
                    part_seg['raw_words'] = sub_raw_words
                for k, v in seg.items():
                    if k not in ['text', 'start', 'end', 'raw_words']:
                        part_seg[k] = v
                result.append(part_seg)
                return
                
            words_per_part = total_words // num_parts
            remainder = total_words % num_parts
            total_raw = len(sub_raw_words)
            time_per_word = (sub_end - sub_start) / total_words if total_words > 0 else 0
            
            word_idx = 0
            raw_idx = 0
            
            for part_idx in range(num_parts):
                current_part_words = words_per_part + (1 if part_idx < remainder else 0)
                if current_part_words == 0:
                    continue
                    
                part_words = words[word_idx:word_idx + current_part_words]
                part_text = ' '.join(part_words)
                
                if sub_raw_words:
                    raw_per_part = total_raw // num_parts
                    raw_remainder = total_raw % num_parts
                    current_raw_words = raw_per_part + (1 if part_idx < raw_remainder else 0)
                    
                    if current_raw_words > 0 and raw_idx < total_raw:
                        part_start = sub_raw_words[raw_idx]['start']
                        last_raw_idx = min(raw_idx + current_raw_words - 1, total_raw - 1)
                        part_end = sub_raw_words[last_raw_idx]['end']
                        part_raw_words = sub_raw_words[raw_idx:last_raw_idx + 1]
                        raw_idx += current_raw_words
                    else:
                        part_start = sub_start + word_idx * time_per_word
                        part_end = sub_start + (word_idx + current_part_words) * time_per_word
                        part_raw_words = []
                else:
                    part_start = sub_start + word_idx * time_per_word
                    part_end = sub_start + (word_idx + current_part_words) * time_per_word
                    part_raw_words = []
                    
                if part_end > sub_end: part_end = sub_end
                if part_start < sub_start: part_start = sub_start
                if part_idx > 0 and part_start < result[-1]['end']:
                    part_start = result[-1]['end']
                    if part_end < part_start: part_end = part_start + 0.1
                    
                part_seg = {'text': part_text, 'start': round(part_start, 3), 'end': round(part_end, 3)}
                if preserve_raw_words and part_raw_words:
                    part_seg['raw_words'] = part_raw_words
                for k, v in seg.items():
                    if k not in ['text', 'start', 'end', 'raw_words']:
                        part_seg[k] = v
                result.append(part_seg)
                word_idx += current_part_words

        # Ưu tiên chia theo dấu phẩy trước
        if ',' in text:
            # Tách chuỗi theo dấu phẩy, giữ lại dấu phẩy
            parts = re.split(r'(?<=,)\s+', text)
            if len(parts) > 1:
                # Nếu chia được theo phẩy, tính timestamp cho từng đoạn
                all_seg_words = text.split()
                total_seg_words = len(all_seg_words)
                raw_words = seg.get('raw_words', [])
                
                time_per_word = duration / total_seg_words if total_seg_words > 0 else 0
                word_offset = 0
                raw_offset = 0
                
                for part in parts:
                    part = part.strip()
                    if not part: continue
                    part_word_count = len(part.split())
                    
                    if raw_words:
                        # Map raw words roughly
                        part_raw_words = raw_words[raw_offset:raw_offset + part_word_count]
                        if part_raw_words:
                            p_start = part_raw_words[0]['start']
                            p_end = part_raw_words[-1]['end']
                        else:
                            p_start = seg.get('start', 0) + word_offset * time_per_word
                            p_end = seg.get('start', 0) + (word_offset + part_word_count) * time_per_word
                        raw_offset += part_word_count
                    else:
                        p_start = seg.get('start', 0) + word_offset * time_per_word
                        p_end = seg.get('start', 0) + (word_offset + part_word_count) * time_per_word
                        part_raw_words = []
                        
                    word_offset += part_word_count
                    
                    # Gọi đệ quy/xử lý mảng (dùng hàm con)
                    process_sub_text(part, p_start, p_end, part_raw_words)
                continue
                
        # Nếu không có dấu phẩy hoặc chia qua dấu phẩy xong, xử lý trực tiếp
        process_sub_text(text, seg.get('start', 0), seg.get('end', 0), seg.get('raw_words', []))
        
    return result


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
            
            cpu_threads = self.config.get("cpu_threads", 4)
            restore_punctuation = self.config.get("restore_punctuation", False)
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Không tìm thấy thư mục mô hình tại: {self.model_path}\nVui lòng tải mô hình và đặt vào thư mục 'models'.")

            # Define model files
            # Dynamic lookup for model files
            def find_file(pattern):
                try:
                    files = [f for f in os.listdir(self.model_path) if f.startswith(pattern) and f.endswith(".onnx")]
                    # Prefer float (non-int8) if available for better accuracy
                    float_files = [f for f in files if "int8" not in f]
                    if float_files:
                        return os.path.join(self.model_path, float_files[0])
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
                "decoding_method": "modified_beam_search",
                "max_active_paths": 8,
            }
            
            # Thêm hotwords nếu có
            from common import get_hotwords_config, BASE_DIR
            hotwords_config = get_hotwords_config(self.model_path, BASE_DIR)
            if hotwords_config:
                kwargs.update(hotwords_config)
                print(f"[Hotwords] Enabled with score {hotwords_config.get('hotwords_score', 1.5)}")
            
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
            
            # Các định dạng cần chuyển đổi (bao gồm cả video)
            needs_conversion = ['.m4a', '.ogg', '.wma', '.opus', 
                                '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
            
            if file_ext in needs_conversion:
                try:
                    # Thử load trực tiếp với librosa trước
                    audio, sample_rate = librosa.load(self.file_path, sr=16000, mono=True, res_type="soxr_vhq")
                except Exception as e:
                    # Nếu không được, chuyển đổi sang wav tạm thờii
                    self.progress.emit(f"PHASE:LoadAudio|Đang chuyển đổi {file_ext} sang wav|35")
                    try:
                        from pydub import AudioSegment
                        
                        # Tìm ffmpeg trong các vị trí phổ biến trên Windows
                        self._setup_ffmpeg_path()
                        
                        temp_wav = self.file_path + '.temp.wav'
                        
                        # Xác định format cho pydub (một số định dạng video pydub tự detect được, không cần ép format)
                        format_map = {'.m4a': 'm4a', '.ogg': 'ogg', '.wma': 'wma', '.opus': 'opus'}
                        if file_ext in format_map:
                            audio_format = format_map[file_ext]
                            audio_segment = AudioSegment.from_file(self.file_path, format=audio_format)
                        else:
                            # Để pydub (ffmpeg) tự phát hiện codec/format đối với file video
                            audio_segment = AudioSegment.from_file(self.file_path)
                            
                        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                        audio_segment.export(temp_wav, format='wav')
                        file_to_load = temp_wav
                        audio, sample_rate = librosa.load(file_to_load, sr=16000, mono=True, res_type="soxr_vhq")
                        # Xóa file tạm
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    except ImportError:
                        raise ImportError(f"Không thể đọc file {file_ext}. Vui lòng cài đặt pydub: pip install pydub")
                    except Exception as e2:
                        raise Exception(f"Không thể đọc file {file_ext}. Đảm bảo đã cài ffmpeg: {e2}")
            else:
                audio, sample_rate = librosa.load(file_to_load, sr=16000, mono=True, res_type="soxr_vhq")
            
            # Peak normalization: đảm bảo audio ở mức tối ưu cho log-mel feature extraction.
            # Chỉ thực sự ảnh hưởng khi volume rất thấp (ghi xa mic, gain nhỏ).
            # Audio bình thường gần như không bị thay đổi.
            peak = np.max(np.abs(audio))
            if peak > 0 and peak < 0.5:
                audio = audio / peak * 0.95
                print(f"[Transcriber] Peak normalization: {peak:.4f} → 0.95 (audio volume quá thấp)")
            
            # Kết thúc đo thờigian load audio
            timing_details["upload_convert"] = time.time() - load_audio_start
            self.progress.emit(f"PHASE:LoadAudio|Đã tải audio ({timing_details['upload_convert']:.1f}s)|100")
            
            # Process in segments to prevent RAM explosion (Sherpa-ONNX infinite context issue)
            import math
            segment_duration = 30 # seconds
            segment_samples = 16000 * segment_duration
            total_samples = len(audio)
            
            # Sử dụng overlap configuration đã định nghĩa ở đầu file
            overlap_sec = OVERLAP_SEC
            overlap_samples = OVERLAP_SAMPLES
            
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
            
            # Lưu kết quả mỗi chunk để merge overlap sau
            chunk_results = []

            self.progress.emit(f"PHASE:Transcription|Đang chuyển thành văn bản|0")
            
            for i in range(num_segments):
                # Tính vùng audio CÓ OVERLAP
                logical_start = segment_boundaries[i]
                logical_end = segment_boundaries[i + 1]
                
                if i == 0:
                    # Chunk đầu: không có overlap phía trước
                    actual_start = logical_start
                    overlap_at_start = 0
                else:
                    # Chunk từ thứ 2: thêm overlap phía trước
                    actual_start = max(0, logical_start - overlap_samples)
                    overlap_at_start = logical_start - actual_start
                
                actual_end = logical_end
                
                # Check cancellation
                if not self.is_running:
                    return

                # Create a NEW stream for each segment to reset context/memory
                s = recognizer.create_stream()
                
                chunk = audio[actual_start:actual_end]
                s.accept_waveform(16000, chunk)
                recognizer.decode_stream(s)
                
                result = s.result
                segment_text = result.text.strip()
                
                if segment_text:
                    # Zipformer output is uppercase, normalize it
                    segment_text = segment_text.lower()
                    
                    # Process timestamps if available
                    chunk_words = []
                    if hasattr(result, 'timestamps') and hasattr(result, 'tokens'):
                        ts = result.timestamps
                        toks = result.tokens
                        
                        # Calculate time offset for this chunk (in seconds)
                        time_offset = actual_start / 16000
                        
                        # Grab log probabilities if available
                        ys_log_probs = getattr(result, 'ys_log_probs', None)
                        
                        import math
                        for j, (t_val, tok) in enumerate(zip(ts, toks)):
                            local_start = t_val
                            if j < len(ts) - 1:
                                local_end = ts[j + 1]
                            else:
                                local_end = local_start + 0.3  # default duration for last token
                            
                            if ys_log_probs is not None and j < len(ys_log_probs):
                                prob = math.exp(ys_log_probs[j])
                            else:
                                prob = 1.0
                            
                            # Timestamp tuyệt đối trong file gốc
                            abs_start = local_start + time_offset
                            abs_end = local_end + time_offset
                            
                            # Normalize token
                            tok_display = tok.lower()
                            
                            chunk_words.append({
                                "text": tok_display,
                                "start": abs_start,
                                "end": abs_end,
                                "local_start": local_start,
                                "local_end": local_end,
                                "prob": prob
                            })
                    
                    # === XỬ LÝ MERGE BPE NGAY TẠI ĐÂY CHO TỪNG CHUNK ===
                    merged_chunk_words = []
                    current_word = None
                    for tok_info in chunk_words:
                        tok = tok_info["text"]
                        # Xử lý cả 2 loại space phổ biến U+0020 và U+2581
                        if tok.startswith(" ") or tok.startswith("\u2581"):
                            if current_word is not None:
                                current_word["prob"] = sum(current_word["probs"]) / len(current_word["probs"])
                                merged_chunk_words.append(current_word)
                            current_word = {
                                "text": tok.lstrip(" ").lstrip("\u2581"),
                                "start": tok_info["start"],
                                "end": tok_info["end"],
                                "local_start": tok_info["local_start"],
                                "local_end": tok_info["local_end"],
                                "probs": [tok_info.get("prob", 1.0)]
                            }
                        else:
                            if current_word is not None:
                                current_word["text"] += tok
                                current_word["end"] = tok_info["end"]
                                current_word["local_end"] = tok_info["local_end"]
                                current_word["probs"].append(tok_info.get("prob", 1.0))
                            else:
                                current_word = {
                                    "text": tok,
                                    "start": tok_info["start"],
                                    "end": tok_info["end"],
                                    "local_start": tok_info["local_start"],
                                    "local_end": tok_info["local_end"],
                                    "probs": [tok_info.get("prob", 1.0)]
                                }
                    if current_word is not None:
                        current_word["prob"] = sum(current_word["probs"]) / len(current_word["probs"])
                        merged_chunk_words.append(current_word)
                    
                    # Update lại danh sách words đã merge BPE
                    chunk_words = merged_chunk_words
                    
                    # Cập nhật segment_text từ chunk_words đã merge
                    segment_text = " ".join(w["text"] for w in chunk_words)
                    
                    chunk_results.append({
                        "text": segment_text,
                        "words": chunk_words,
                        "audio_start_abs": actual_start / 16000.0,
                        "audio_end_abs": actual_end / 16000.0,
                        "overlap_sec": overlap_at_start / 16000.0,
                    })
                else:
                    # Nếu chunk không có text, vẫn lưu để giữ đúng thứ tự
                    chunk_results.append({
                        "text": "",
                        "words": [],
                        "audio_start_abs": actual_start / 16000.0,
                        "audio_end_abs": actual_end / 16000.0,
                        "overlap_sec": overlap_at_start / 16000.0,
                    })
                
                # Update progress for transcription phase (0% -> 100%)
                percent = int((i + 1) / num_segments * 100)
                self.progress.emit(f"PHASE:Transcription|Đang chuyển thành văn bản|{percent}")
                
                # Explicit delete to help GC
                del s
            
            # Merge chunks với overlap handling
            all_words, full_text = merge_chunks_with_overlap(chunk_results, overlap_sec)
            
            # Post-processing for Zipformer output (often lower cased and generic)
            # Zipformer output for Vietnamese might need basic formatting
            if full_text:
                full_text = full_text.capitalize()
            
            print(f"[Transcriber] Merged chunks into {len(all_words)} words")
            
            transcription_end_time = time.time()
            timing_details["transcription"] = transcription_end_time - start_time - timing_details["upload_convert"]
            transcribe_duration = transcription_end_time - start_time
            
            restore_duration = 0.0
            
            final_segments = []
            paragraphs = []  # Lưu paragraph boundaries từ SAT

            if restore_punctuation and full_text:
                punct_start = time.time()
                try:
                    if self.config.get("bypass_restorer", False):
                        # Bỏ qua hoàn toàn model GecBERT nếu User kéo cả 2 mốc về Rất Ít (1)
                        self.progress.emit("PHASE:Punctuation|Bỏ qua model (Mức độ Rất Ít)|100")
                    else:
                        self.progress.emit("PHASE:Punctuation|Đang thêm dấu câu (Sliding Window)|0")
                        
                        punct_confidence = self.config.get("punctuation_confidence", 0.3)
                        case_confidence = self.config.get("case_confidence", -1.0)
                        
                        # Sliding Window cần model dứt khoát hơn (confidence thấp hơn)
                        # UI Map:
                        # Mức Trượt = 1  (Ít dấu) -> confidence UI trả về = 0.8
                        # Mức Trượt = 5  (Vừa)    -> confidence UI trả về = ~0.53
                        # Mức Trượt = 10 (Nhiều)  -> confidence UI trả về = 0.2
                        # GecBERT model: tự tin chèn dấu cao nhất ở mốc ÂM
                        # Ta sẽ shift giá trị để được mức âm tương đương:
                        window_confidence = punct_confidence - 0.8  
                        restorer = ImprovedPunctuationRestorer(device="cpu", confidence=window_confidence, case_confidence=case_confidence)
                        
                        def punct_progress_cb(current, total):
                            if not self.is_running:
                                raise Exception("Cancelled by user")
                            percent = int((current / max(1, total)) * 100)
                            self.progress.emit(f"PHASE:Punctuation|Đang thêm dấu câu ({current}/{total})|{percent}")

                        restored_text_raw = restorer.restore(full_text, progress_callback=punct_progress_cb)
                        full_text = restored_text_raw
                        
                        # Giải phóng RAM nếu được yêu cầu
                        if self.config.get("save_ram", False):
                            restorer.unload()
                            import gc
                            gc.collect()
                            
                    timing_details["punctuation"] = time.time() - punct_start
                    
                    self.progress.emit("PHASE:Align|Đang căn chỉnh thời gian|0")
                    
                    # Tách câu bằng regex (tách theo . ? !) để tạo thành các segment riêng biệt
                    # Giữ nguyên dấu câu bằng positive lookbehind
                    sentences = re.split(r'(?<=[.?!])\s+', full_text)
                    
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
                            seg_words = all_words[match_start:match_end + 1]
                            current_word_idx = match_end + 1
                        else:
                            # Fallback: không tìm thấy nguyên chuỗi khớp, tìm theo từ đầu tiên
                            first_word = normalize_word(sent_words_clean[0]) if sent_words_clean else ""
                            temp_idx = current_word_idx
                            seg_words = []
                            found_first = False
                            while temp_idx < len(all_words):
                                asr_word = normalize_word(all_words[temp_idx]['text'])
                                if first_word and (first_word in asr_word or asr_word in first_word):
                                    found_first = True
                                    break
                                temp_idx += 1
                                
                            if found_first:
                                # Nhảy đúng số từ của câu để lấy đoạn raw_words
                                end_idx = min(temp_idx + len(sent_words_clean) - 1, len(all_words) - 1)
                                start_t = all_words[temp_idx]['start']
                                end_t = all_words[end_idx]['end']
                                seg_words = all_words[temp_idx:end_idx + 1]
                                current_word_idx = end_idx + 1
                            else:
                                # Last resort fallback: Nếu không khớp chữ nào, gán đại đoạn tiếp theo có số lượng từ tương ứng
                                fallback_idx = min(current_word_idx, len(all_words)-1) if all_words else 0
                                end_idx = min(fallback_idx + len(sent_words_clean) - 1, len(all_words)-1) if all_words else 0
                                
                                start_t = all_words[fallback_idx]['start'] if all_words else 0.0
                                end_t = all_words[end_idx]['end'] if all_words else 0.0
                                
                                seg_words = all_words[fallback_idx:end_idx + 1] if all_words else []
                                current_word_idx = end_idx + 1

                        final_segments.append({
                            "text": sent,
                            "start": start_t,
                            "end": end_t,
                            "raw_words": seg_words
                        })
                        
                        # Emit progress mỗi 10%
                        progress = int((sent_idx + 1) / total_sentences * 100)
                        if progress >= last_align_progress + 10:
                            self.progress.emit(f"PHASE:Align|Đang căn chỉnh thờigian|{progress}")
                            last_align_progress = progress
                    
                    # Kết thúc đo thờigian alignment
                    timing_details["alignment"] = time.time() - align_start
                    
                    # Fix lỗi thời gian kết thúc vượt quá thời gian bắt đầu của câu kế tiếp
                    if final_segments:
                        for i in range(len(final_segments) - 1):
                            next_start = final_segments[i+1]['start']
                            # Nếu kết thúc vượt quá bắt đầu câu sau -> ép bằng
                            if final_segments[i]['end'] > next_start:
                                final_segments[i]['end'] = next_start
                            # Ép luôn các từ bên trong (nếu có)
                            if 'raw_words' in final_segments[i]:
                                for w in final_segments[i]['raw_words']:
                                    if w['end'] > next_start:
                                        w['end'] = next_start
                                    if w['start'] > next_start:
                                        w['start'] = next_start
                    # Chia nhỏ các segment dài (>12s) theo số từ
                    do_diarization = self.config.get("speaker_diarization", False) and DIARIZATION_AVAILABLE
                    if final_segments:
                        original_count = len(final_segments)
                        final_segments = split_long_segments(final_segments, max_duration=12.0, preserve_raw_words=do_diarization)
                        new_count = len(final_segments)
                        if new_count > original_count:
                            logger.info(f"Đã chia nhỏ {original_count} segment thành {new_count} segment (các segment >12s)")
                        
                        # Xóa raw_words sau khi split xong nếu không làm diarization để tránh làm bộ nhớ và UI cồng kềnh
                        if not do_diarization:
                            for seg in final_segments:
                                if 'raw_words' in seg:
                                    del seg['raw_words']
                except Exception as e:
                    if 'restorer' in locals() and restorer:
                        restorer.unload()
                    if str(e) == "Cancelled by user":
                        return
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

