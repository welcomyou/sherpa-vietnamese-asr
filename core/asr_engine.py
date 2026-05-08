# core/asr_engine_myort.py - ASR pipeline using pure ONNX Runtime (no sherpa-onnx dependency)
# Copy of core/asr_engine.py with sherpa_onnx replaced by onnxruntime
# KHÔNG import PyQt6 - pure Python, dùng callback thay signal

import logging
import math
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from core.config import DEBUG_LOGGING, get_hotwords_config, BASE_DIR
from core.hotword_context import build_context_graph

logger = logging.getLogger(__name__)

# Safe print cho Windows console (tránh UnicodeEncodeError cp1252)
_original_print = print
def print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        _original_print(text.encode('ascii', errors='replace').decode(), **kwargs)

# =============================================================================
# OVERLAP CHUNKING CONFIGURATION
# =============================================================================
OVERLAP_SEC = 3.0
OVERLAP_SAMPLES = int(OVERLAP_SEC * 16000)
MAX_OVERLAP_WORDS = 100
FUZZY_MATCH_THRESHOLD = 0.8
MIN_MATCH_RATIO = 0.5


# =============================================================================
# OVERLAP RESOLUTION FUNCTIONS
# =============================================================================

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
    if len(w1) > 2 and len(w2) > 2:
        if w1 in w2 or w2 in w1:
            return True
    return SequenceMatcher(None, w1, w2).ratio() >= threshold


def find_overlap_alignment(tail_words, head_words):
    """
    Tìm alignment tối ưu giữa tail (cuối chunk trước) và head (đầu chunk sau)
    sử dụng thuật toán sliding offset kết hợp fuzzy matching.

    Trả về: (best_cut_index_in_head, action, num_words_to_drop)
        - best_cut_index_in_head: index đầu tiên trong head_words MÀ KHÔNG thuộc overlap
        - action: "cut_head", "drop_tail", "drop_head", "none"
        - num_words_to_drop: Số từ cần pop khỏi merged_words ở chunk trước

    Nguyên tắc:
    Nếu có match đáng kể, nối tiếp.
    Nếu không có match đáng kể giữa 2 vùng overlap, tiến hành drop bên phía có
    CONFIDENCE (prob) thấp hơn, tránh lặp lại cùng đoạn âm thanh mà lại bị lệch text.
    """
    if not tail_words or not head_words:
        return 0, "none", 0

    # Ghi nhận số lượng từ thực sự (nếu drop tail/head sẽ drop toàn bộ số này)
    original_tail_len = len(tail_words)
    original_head_len = len(head_words)

    # Giới hạn số từ để tối ưu hiệu năng (MAX_OVERLAP_WORDS = 100)
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

        if score > best_score and match_ratio >= MIN_MATCH_RATIO:
            best_score = score
            best_cut_index = matched_head_indices[-1] + 1
            # Tính số từ cần pop khỏi original_tail_words
            truncated_diff = original_tail_len - len(tail_normalized)
            best_pop_count = (len(tail_normalized) - 1 - matched_tail_indices[-1])

    # GUARD: Kiểm tra xem có sự khác biệt (divergence) giữa 2 chuỗi không
    min_len = min(len(tail_normalized), len(head_normalized))

    # Chỉ coi là phân kỳ nếu điểm khớp nhỏ hơn số từ của chuỗi ngắn nhất VÀ có cắt xén rác
    is_diverged = (best_score < min_len) and (best_pop_count > 0)

    if best_score == 0 or is_diverged:
        # Tách riêng phần bị lệch (divergent) để so sánh xác suất
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
        if DEBUG_LOGGING:
            debug_msg = f"[OVERLAP RESOLVE DEBUG] So sánh xác suất do {reason}:\n"
            debug_msg += f"  - TAIL (phần lệch cuối chunk trước): AvgProb={tail_prob:.3f} | Words: {tail_words_str}\n"
            debug_msg += f"  - HEAD (phần lệch đầu chunk sau): AvgProb={head_prob:.3f} | Words: {head_words_str}"
            print(debug_msg)
            logger.info(debug_msg)

        # Nếu tail tự tin vượt trội hơn đoạn đầu chuỗi mới
        if tail_prob > head_prob:
            if DEBUG_LOGGING:
                decision_msg = f"[OVERLAP RESOLVE] Quyết định: DROP HEAD (Xóa phần đầu của chunk sau). Tail ({tail_prob:.3f}) > Head ({head_prob:.3f})"
                print(decision_msg)
                logger.info(decision_msg)
            # Xóa phần overlap của head, giữ nguyên chunk trước
            return len(head_words), "drop_head", 0
        else:
            if DEBUG_LOGGING:
                decision_msg = f"[OVERLAP RESOLVE] Quyết định: DROP TAIL (Xóa tail của chunk trước). Head ({head_prob:.3f}) >= Tail ({tail_prob:.3f})"
                print(decision_msg)
                logger.info(decision_msg)
            return 0, "drop_tail", original_tail_len  # Giữ toàn bộ head, xoá TOÀN BỘ tail

    # Debug khi lặp chữ an toàn (Perfect match)
    if DEBUG_LOGGING:
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
            - "overlap_sec": float (thời gian overlap ở đầu chunk)
        overlap_duration_sec: float, thời gian overlap (giây)

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
                if DEBUG_LOGGING:
                    print(f"   -> Popping {pop_count} words from merged_words tail")
                del merged_words[-pop_count:]

            # Chỉ lấy words SAU điểm cắt (bỏ phần đã trùng)
            remaining_words = chunk_words[cut_index:] if cut_index < len(chunk_words) else []

            merged_words.extend(remaining_words)

    merged_text = " ".join([w["text"] for w in merged_words])
    return merged_words, merged_text


def remove_repeated_ngrams(words, max_ngram=1, max_gap_sec=0.3):
    """
    ⚠️ KHÔNG DÙNG HÀM NÀY - đã bị disable vì gây mất từ hợp lệ.

    Hàm này xóa nhầm các từ lặp có chủ đích trong tiếng Việt, ví dụ:
    - "một một bảy" → mất "một" (đọc số 117)
    - "chiều chiều ra đứng bên bờ" → mất "chiều" (thơ/ca dao)

    Phát hiện và xóa n-gram lặp liên tiếp trong danh sách words.

    Ví dụ: "thành phố thành phố hồ chí minh" → "thành phố hồ chí minh"

    Điều kiện xóa:
    - Cùng chuỗi từ (normalized) lặp liên tiếp
    - Khoảng cách thời gian giữa 2 lần xuất hiện <= max_gap_sec
    - Giữ bản có probability trung bình cao hơn
    """
    if len(words) < 2:
        return words

    i = 0
    result = list(words)  # copy

    for ngram_size in range(max_ngram, 0, -1):  # Ưu tiên n-gram dài trước
        i = 0
        while i + 2 * ngram_size <= len(result):
            # Lấy 2 cụm liên tiếp có cùng kích thước
            gram1 = result[i:i + ngram_size]
            gram2 = result[i + ngram_size:i + 2 * ngram_size]

            text1 = " ".join(normalize_word_for_overlap(w["text"]) for w in gram1)
            text2 = " ".join(normalize_word_for_overlap(w["text"]) for w in gram2)

            if text1 == text2 and text1:
                # Kiểm tra gap thời gian
                gap = gram2[0].get("start", 0) - gram1[-1].get("end", 0)
                if gap <= max_gap_sec:
                    # So sánh prob trung bình, giữ bản tốt hơn
                    prob1 = sum(w.get("prob", 1.0) for w in gram1) / ngram_size
                    prob2 = sum(w.get("prob", 1.0) for w in gram2) / ngram_size

                    if prob1 >= prob2:
                        # Xóa gram2 (bản lặp)
                        del result[i + ngram_size:i + 2 * ngram_size]
                    else:
                        # Xóa gram1 (bản yếu hơn)
                        del result[i:i + ngram_size]

                    if DEBUG_LOGGING:
                        print(f"[DEDUP] Removed repeated {ngram_size}-gram: '{text1}' "
                              f"(gap={gap:.3f}s, prob1={prob1:.3f}, prob2={prob2:.3f})")
                    continue  # Không tăng i, kiểm tra lại vị trí này
            i += 1

    return result


# =============================================================================
# SEGMENT SPLITTING
# =============================================================================

def split_long_segments(segments, max_duration=12.0, preserve_raw_words=False):
    """
    Chia nhỏ các segment dài thành nhiều segment ngắn hơn dựa trên số từ.

    Quy tắc chia:
    - >12s: chia làm 2 phần
    - >24s: chia làm 3 phần
    - >36s: chia làm 4 phần
    - Cứ thêm 12s thì thêm 1 phần

    Chia theo số từ (không chia theo thời gian), đảm bảo mỗi phần có số từ đều nhau.
    Timestamp được tính lại tuyến tính dựa trên số từ.
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
            parts = re.split(r'(?<=,)\s+', text)
            if len(parts) > 1:
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
                    process_sub_text(part, p_start, p_end, part_raw_words)
                continue

        process_sub_text(text, seg.get('start', 0), seg.get('end', 0), seg.get('raw_words', []))

    return result


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def _find_ffmpeg():
    """Find ffmpeg executable path."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    possible = [
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    return None


def _load_audio_ffmpeg_pipe(file_path, sample_rate=16000):
    """Load audio via ffmpeg pipe — decode + resample + mono in 1 pass.
    No intermediate copies in Python memory. Peak RAM = output array only.

    Returns: float32 numpy array (mono, target sample rate)
    """
    import subprocess

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found")

    cmd = [
        ffmpeg, "-i", file_path,
        "-vn",                                       # no video
        "-af", "aresample=resampler=soxr:precision=20",  # SoXR HQ resampler
        "-ac", "1",                                  # mono
        "-ar", str(sample_rate),                     # target sample rate
        "-f", "f32le",                               # raw float32 little-endian
        "-acodec", "pcm_f32le",
        "-loglevel", "error",
        "pipe:1"                                     # output to stdout
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=creationflags)
    raw_bytes, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {stderr.decode('utf-8', errors='replace')[:200]}")

    audio = np.frombuffer(raw_bytes, dtype=np.float32)
    return audio


def load_audio(file_path, sample_rate=16000, progress_callback=None, **kwargs):
    """
    Load và chuẩn hóa audio file thành float32 array tại sample rate mong muốn.

    Strategy:
      1. WAV/FLAC 16kHz mono: soundfile direct read (fastest, zero resample)
      2. WAV/FLAC other rate: ffmpeg pipe (decode + resample in 1 pass)
      3. MP3/M4A/video: ffmpeg pipe (decode + resample + mono in 1 pass)

    RAM: output array only (~660MB for 3h audio). No intermediate copies.

    Args:
        file_path: Đường dẫn file audio/video
        sample_rate: Sample rate đích (mặc định 16000)
        progress_callback: callable(message) - callback báo tiến trình

    Returns:
        numpy.ndarray - audio float32 array, mono, normalized
    """
    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    file_ext = os.path.splitext(file_path)[1].lower()

    # Fast path: WAV/FLAC already at target rate + mono → soundfile direct
    if file_ext in ('.wav', '.flac'):
        import soundfile as sf
        info = sf.info(file_path)
        if info.samplerate == sample_rate and info.channels == 1:
            audio, _ = sf.read(file_path, dtype='float32')
        elif info.samplerate == sample_rate and info.channels > 1:
            audio, _ = sf.read(file_path, dtype='float32')
            audio = audio.mean(axis=1)
        else:
            # WAV/FLAC needs resample → ffmpeg pipe
            emit("Đang resample audio (ffmpeg)")
            audio = _load_audio_ffmpeg_pipe(file_path, sample_rate)
    else:
        # MP3/M4A/video/etc → ffmpeg pipe: decode + resample + mono in 1 pass
        emit("Đang đọc audio (ffmpeg)")
        audio = _load_audio_ffmpeg_pipe(file_path, sample_rate)

    print(f"[Audio] Loaded: {len(audio)/sample_rate:.1f}s, {len(audio)*4/1024/1024:.0f}MB")

    # Peak normalization (low volume boost)
    peak = np.max(np.abs(audio))
    if peak > 0 and peak < 0.5:
        audio = audio / peak * 0.95
        print(f"[Audio] Peak normalization: {peak:.4f} -> 0.95 (low volume)")

    return audio


def find_silent_regions(audio_data, sample_rate=16000, threshold=0.01, min_silence_duration=0.3):
    """
    Tìm các vùng im lặng trong audio. Returns list of (start_sample, end_sample).
    Sử dụng vectorized numpy.
    """
    frame_length = int(sample_rate * 0.01)
    num_frames = len(audio_data) // frame_length

    if num_frames == 0:
        return []

    trimmed = audio_data[:num_frames * frame_length]
    frames = trimmed.reshape(num_frames, frame_length)
    energies = np.sqrt(np.mean(frames ** 2, axis=1))

    is_silent = energies < threshold
    min_frames = int(min_silence_duration / 0.01)

    silent_starts = np.where(np.diff(is_silent.astype(int)) == 1)[0] + 1
    silent_ends = np.where(np.diff(is_silent.astype(int)) == -1)[0] + 1

    if is_silent[0]:
        silent_starts = np.insert(silent_starts, 0, 0)
    if is_silent[-1]:
        silent_ends = np.append(silent_ends, len(is_silent))

    silent_regions = []
    for start, end in zip(silent_starts, silent_ends):
        if end - start >= min_frames:
            start_sample = int(start) * frame_length
            end_sample = int(end) * frame_length
            silent_regions.append((start_sample, min(end_sample, len(audio_data))))

    return silent_regions


def find_best_split_point(target_sample, total_samples, silent_regions, search_window=2*16000):
    """Tìm điểm chia tốt nhất gần target_sample, ưu tiên vùng im lặng."""
    search_start = max(0, target_sample - search_window)
    search_end = min(total_samples, target_sample + search_window)

    best_point = target_sample
    best_distance = float('inf')

    for silent_start, silent_end in silent_regions:
        if silent_end >= search_start and silent_start <= search_end:
            mid_silent = (silent_start + silent_end) // 2
            distance = abs(mid_silent - target_sample)
            if distance < best_distance:
                best_distance = distance
                best_point = mid_silent

    return best_point


# =============================================================================
# VAD-BASED SEGMENTATION (Silero VAD ONNX) - delegated to core/vad_utils.py
# =============================================================================

from core.vad_utils import get_vad_segments, unload_vad_model


def chunk_long_segment(seg_start, seg_end, max_sec=30, overlap_sec=3.0, sample_rate=16000):
    """
    Chunk một VAD segment dài thành các đoạn nhỏ hơn với overlap.
    Segment <= max_sec: giữ nguyên, không overlap.
    Segment > max_sec: chia đều, mỗi cặp liền kề overlap overlap_sec giây.

    Returns:
        list of (actual_start, actual_end, overlap_at_start) — đơn vị sample
    """
    duration = (seg_end - seg_start) / sample_rate

    if duration <= max_sec:
        return [(seg_start, seg_end, 0)]

    n = math.ceil(duration / max_sec)
    # chunk_len = (duration + (n-1)*overlap_sec) / n
    chunk_len_samples = int(((duration + (n - 1) * overlap_sec) / n) * sample_rate)
    step_samples = chunk_len_samples - int(overlap_sec * sample_rate)

    chunks = []
    for i in range(n):
        c_start = seg_start + i * step_samples
        c_end = min(c_start + chunk_len_samples, seg_end)
        overlap_at_start = 0 if i == 0 else int(overlap_sec * sample_rate)

        # Đảm bảo chunk cuối bao phủ đến hết segment
        if i == n - 1:
            c_end = seg_end

        chunks.append((c_start, c_end, overlap_at_start))

    return chunks


def concat_vad_speech(audio, vad_segments):
    """
    Nối tất cả VAD speech segments thành 1 audio liên tục, bỏ silence.

    Args:
        audio: numpy array audio đã preprocess
        vad_segments: list of (start_sample, end_sample)

    Returns:
        concat_audio: numpy array chỉ chứa speech
        offset_map: list of (concat_start_sample, original_start_sample, length_samples)
                     dùng để map timestamp từ concat space → original space
    """
    parts = []
    offset_map = []
    concat_pos = 0
    for seg_start, seg_end in vad_segments:
        seg_len = seg_end - seg_start
        offset_map.append((concat_pos, seg_start, seg_len))
        parts.append(audio[seg_start:seg_end])
        concat_pos += seg_len

    if not parts:
        return audio.copy(), [(0, 0, len(audio))]

    concat_audio = np.concatenate(parts)
    return concat_audio, offset_map


def map_concat_time_to_original(concat_time, offset_map, sample_rate=16000):
    """
    Map timestamp từ concat audio space → original audio space.

    Args:
        concat_time: thời gian (giây) trong concat audio
        offset_map: list of (concat_start_sample, original_start_sample, length_samples)
        sample_rate: 16000

    Returns:
        original_time: thời gian (giây) trong audio gốc
    """
    concat_sample = int(concat_time * sample_rate)

    for concat_start, orig_start, length in offset_map:
        if concat_start <= concat_sample < concat_start + length:
            offset_in_seg = concat_sample - concat_start
            return (orig_start + offset_in_seg) / sample_rate

    # Fallback: nếu nằm ngoài (do làm tròn), dùng segment gần nhất
    if offset_map:
        # Trước segment đầu
        if concat_sample < offset_map[0][0]:
            return offset_map[0][1] / sample_rate
        # Sau segment cuối
        last = offset_map[-1]
        return (last[1] + last[2]) / sample_rate

    return concat_time




# =============================================================================
# ORT LAZY IMPORT + FBANK
# =============================================================================

_ort_module = None
_knf_opts = None  # NOTE: not thread-safe — use threading.Lock in production

def get_ort():
    """Lazy import onnxruntime."""
    global _ort_module
    if _ort_module is None:
        import onnxruntime as ort_mod
        ort_mod.set_default_logger_severity(3)
        _ort_module = ort_mod
    return _ort_module


def compute_fbank_ort(audio, sr=16000):
    """Compute fbank using kaldi-native-fbank (same C++ backend as sherpa-onnx)."""
    global _knf_opts
    import kaldi_native_fbank as knf
    if _knf_opts is None:
        _knf_opts = knf.FbankOptions()
        _knf_opts.frame_opts.dither = 0.0
        _knf_opts.frame_opts.snip_edges = False
        _knf_opts.frame_opts.samp_freq = sr
        _knf_opts.frame_opts.frame_length_ms = 25.0
        _knf_opts.frame_opts.frame_shift_ms = 10.0
        _knf_opts.frame_opts.window_type = "povey"
        _knf_opts.mel_opts.num_bins = 80
        _knf_opts.mel_opts.low_freq = 20.0
        _knf_opts.mel_opts.high_freq = 7600.0
        _knf_opts.energy_floor = 1.0
    fbank = knf.OnlineFbank(_knf_opts)
    fbank.accept_waveform(sr, audio)
    fbank.input_finished()
    n = fbank.num_frames_ready
    features = np.empty((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fbank.get_frame(i)
    return features


def _log_add(a, b):
    """LogAdd: log(exp(a) + exp(b)) — numerically stable."""
    if a < b: a, b = b, a
    diff = b - a
    return a if diff < -36.0 else a + np.log1p(np.exp(diff))


# =============================================================================
# MODEL CACHE - Tái sử dụng model giữa các request (server mode)
# Queue xử lý tuần tự 1 file → không cần thread lock
# =============================================================================

_recognizer_cache = {}  # {(model_path, threads, paths): recognizer}
_punct_restorer = None  # Singleton - model giống nhau, chỉ đổi confidence per-call
_diarizer_cache = None
_diarizer_cache_key = None


def clear_model_cache(which="all"):
    """Giải phóng model cache. which: 'all', 'recognizer', 'restorer', 'diarizer'"""
    global _recognizer_cache, _punct_restorer
    global _diarizer_cache, _diarizer_cache_key

    if which in ("all", "recognizer"):
        _recognizer_cache.clear()
        logger.info("[ModelCache] Cleared ASR recognizer cache")

    if which in ("all", "restorer"):
        if _punct_restorer is not None:
            try:
                _punct_restorer.unload()
            except Exception:
                pass
            _punct_restorer = None
            logger.info("[ModelCache] Cleared PunctuationRestorer cache")

    if which in ("all", "diarizer"):
        if _diarizer_cache is not None:
            try:
                _diarizer_cache.unload()
            except Exception:
                pass
            _diarizer_cache = None
            _diarizer_cache_key = None
            logger.info("[ModelCache] Cleared SpeakerDiarizer cache")

    import gc
    gc.collect()


def _get_cached_restorer(device, confidence, case_confidence, prefer_int8=False):
    """Lấy PunctuationRestorer từ cache, cập nhật confidence per-request.
    Model BERT load 1 lần duy nhất. Confidence chỉ là threshold cộng vào
    logit sau softmax — đổi thoải mái mà không cần reload model.
    """
    global _punct_restorer

    if _punct_restorer is not None:
        # Reuse model, chỉ cập nhật confidence (không reload ~200-400MB model)
        _punct_restorer.gec_model.confidence = confidence
        _punct_restorer.gec_model.case_confidence = case_confidence
        logger.info(f"[ModelCache] Reusing PunctuationRestorer "
                    f"(conf={confidence:.3f}, case={case_confidence:.3f})")
        return _punct_restorer

    from core.punctuation_restorer_improved import ImprovedPunctuationRestorer
    _punct_restorer = ImprovedPunctuationRestorer(
        device=device, confidence=confidence, case_confidence=case_confidence,
        prefer_int8=prefer_int8,
    )
    logger.info(f"[ModelCache] Loaded new PunctuationRestorer "
                f"(conf={confidence:.3f}, case={case_confidence:.3f})")
    return _punct_restorer


def _get_cached_diarizer(embedding_model_id, num_clusters, num_threads,
                          threshold, auth_token=None):
    """Lấy SpeakerDiarizer từ cache hoặc tạo mới + initialize.

    Chiến lược cache:
    - Pyannote/Altunenes: cache theo (model_id, threads, threshold), đổi num_speakers
      per-call mà không reload model (~50-200MB ONNX)
    - Sherpa: cache theo (model_id, threads, threshold, num_clusters) vì clustering
      config baked vào OfflineSpeakerDiarization object, không đổi được
    """
    global _diarizer_cache, _diarizer_cache_key

    if _diarizer_cache is not None:
        old_model, old_threads, old_threshold = _diarizer_cache_key[:3]
        base_match = (old_model == embedding_model_id and
                      old_threads == num_threads and
                      old_threshold == round(threshold, 4))

        if base_match:
            is_sherpa = (_diarizer_cache._pyannote_backend is None and
                         _diarizer_cache.sd is not None)

            if is_sherpa:
                # Sherpa: chỉ reuse nếu num_clusters cũng khớp
                old_nc = _diarizer_cache_key[3] if len(_diarizer_cache_key) > 3 else None
                if old_nc == num_clusters:
                    # NaturalTurn luôn bật
                    logger.info(f"[ModelCache] Reusing SpeakerDiarizer sherpa "
                                f"({embedding_model_id})")
                    return _diarizer_cache
                # num_clusters đổi → phải recreate sherpa (clustering baked in)
            else:
                # Pyannote/Altunenes: update num_speakers per-call, không reload model
                _update_diarizer_speakers(_diarizer_cache, num_clusters)
                logger.info(f"[ModelCache] Reusing SpeakerDiarizer pyannote "
                            f"({embedding_model_id}, speakers={num_clusters})")
                return _diarizer_cache

    # Cần tạo mới — unload cũ nếu có
    if _diarizer_cache is not None:
        try:
            _diarizer_cache.unload()
        except Exception:
            pass

    from core.speaker_diarization import SpeakerDiarizer
    diarizer = SpeakerDiarizer(
        embedding_model_id=embedding_model_id,
        num_clusters=num_clusters,
        num_threads=num_threads,
        threshold=threshold,
        auth_token=auth_token,
    )
    diarizer.initialize()
    _diarizer_cache = diarizer
    _diarizer_cache_key = (embedding_model_id, num_threads, round(threshold, 4), num_clusters)
    logger.info(f"[ModelCache] Loaded new SpeakerDiarizer ({embedding_model_id})")
    return diarizer


def _update_diarizer_speakers(diarizer, num_clusters):
    """Cập nhật num_speakers trên cached pyannote/altunenes backend.
    Backend hỗ trợ per-call num_speakers — chỉ cần update attributes."""
    diarizer.num_clusters = num_clusters
    backend = diarizer._pyannote_backend
    if backend is None:
        return
    if num_clusters > 0:
        backend.num_speakers = -1  # Dùng min/max range thay vì ép cứng
        backend.min_speakers = max(2, num_clusters - 1)
        backend.max_speakers = num_clusters + 1
    else:
        backend.num_speakers = -1
        backend.min_speakers = None
        backend.max_speakers = None


# =============================================================================
# ROVER: DUAL-MODEL ALIGNMENT & MERGE
# =============================================================================

ROVER_MODEL_IDS = ["zipformer-30m-rnnt-6000h", "sherpa-onnx-zipformer-vi-2025-04-20"]
ROVER_MODEL_ID = "rover-voting"  # ID dùng trong UI/config


def create_recognizer(model_path, cpu_threads=4, max_active_paths=8):
    """Tạo hoặc lấy từ cache ORT sessions (encoder, decoder, joiner + tokens).
    Returns dict with ORT sessions + id2token mapping."""
    cache_key = (os.path.normpath(model_path), cpu_threads, max_active_paths)
    if cache_key in _recognizer_cache:
        logger.info(f"[ModelCache] Reusing ORT recognizer: {os.path.basename(model_path)}")
        return _recognizer_cache[cache_key]

    def find_file(pattern):
        files = [f for f in os.listdir(model_path) if f.startswith(pattern) and f.endswith(".onnx")]
        float_files = [f for f in files if "int8" not in f]
        if float_files:
            return os.path.join(model_path, float_files[0])
        if files:
            return os.path.join(model_path, files[0])
        return None

    encoder = find_file("encoder-")
    decoder = find_file("decoder-")
    joiner = find_file("joiner-")
    tokens_path = os.path.join(model_path, "tokens.txt")

    if not all([encoder, decoder, joiner]) or not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Thiếu file model trong: {model_path}")

    ort = get_ort()
    prov = ['CPUExecutionProvider']

    # Tính Z = threads thực (có HT bonus nếu có)
    from core.config import compute_ort_threads
    Z = compute_ort_threads(cpu_threads)

    # Encoder: full Z threads cho matmul lớn
    opts_enc = ort.SessionOptions()
    opts_enc.intra_op_num_threads = Z
    opts_enc.inter_op_num_threads = 1
    opts_enc.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts_enc.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts_enc.log_severity_level = 3
    opts_enc.enable_cpu_mem_arena = False  # Tránh arena leak — save_ram sẽ unload session
    # Cache optimized graph: lần 2+ load nhanh hơn, giảm disk I/O
    opts_enc.optimized_model_filepath = encoder + ".opt"

    # Decoder/Joiner: 2 threads (benchmark sweet spot)
    # dec=1 chậm hơn, dec>=3 gây thread scheduling overhead cho model nhỏ
    dec_joi_threads = min(2, Z)
    opts_small = ort.SessionOptions()
    opts_small.intra_op_num_threads = dec_joi_threads
    opts_small.inter_op_num_threads = 1
    opts_small.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts_small.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts_small.enable_cpu_mem_arena = False  # Tiết kiệm RAM cho decoder/joiner (model nhỏ)
    opts_small.log_severity_level = 3

    enc_sess = ort.InferenceSession(encoder, opts_enc, providers=prov)
    dec_sess = ort.InferenceSession(decoder, opts_small, providers=prov)
    joi_sess = ort.InferenceSession(joiner, opts_small, providers=prov)

    # Load tokens
    id2token = {}
    with open(tokens_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2token[int(parts[-1])] = parts[0]

    # Get vocab size from joiner
    V = joi_sess.get_outputs()[0].shape[-1]
    if not V:
        V = len(id2token)

    # Hotword context graph (Aho-Corasick)
    context_graph = None
    try:
        hw_config = get_hotwords_config(model_path)
        hw_file = hw_config.get("hotwords_file", "")
        bpe_model = os.path.join(model_path, "bpe.model")
        if hw_file and os.path.exists(bpe_model):
            hw_score = hw_config.get("hotwords_score", 1.5)
            context_graph = build_context_graph(hw_file, bpe_model, default_score=hw_score)
    except Exception as e:
        print(f"[Hotwords] Failed to build context graph: {e}")

    recognizer = {
        'enc_sess': enc_sess, 'dec_sess': dec_sess, 'joi_sess': joi_sess,
        'id2token': id2token, 'vocab_size': V,
        'max_active_paths': max_active_paths,
        'model_path': model_path, 'dec_cache': {},
        'context_graph': context_graph,
    }

    logger.info(f"[ModelCache] Cached new ORT recognizer: {os.path.basename(model_path)} "
                f"(enc={os.path.basename(encoder)}, V={V}, "
                f"threads: enc={Z} dec/joi={dec_joi_threads} [cpu_threads={cpu_threads}, Z={Z}])")

    _recognizer_cache[cache_key] = recognizer
    return recognizer


def _ort_beam_search(recognizer, features, beam_size=8):
    """ORT modified beam search matching sherpa-onnx C++ exactly.
    Returns (token_ids, timestamps_frames, ys_log_probs, T, emit_logits).
    emit_logits: list of raw joiner logits [V] per emitted token (for entropy, 1-pass).

    Hotword boosting (nếu có context_graph):
    - Giống sherpa-onnx: Aho-Corasick automaton inject vào beam search
    - Mỗi hypothesis giữ 1 context_state
    - Non-blank token → forward_one_step → score_delta cộng vào log-prob
    - Cuối decode → finalize: trừ partial score chưa hoàn thành
    """
    BLANK_ID = 0
    UNK_ID = 2     # BPE <unk> — skip hotword matching (giống sherpa-onnx C++)
    CONTEXT_SIZE = 2

    enc_sess = recognizer['enc_sess']
    dec_sess = recognizer['dec_sess']
    joi_sess = recognizer['joi_sess']
    V = recognizer['vocab_size']
    dec_cache = recognizer['dec_cache']
    ctx_graph = recognizer.get('context_graph')  # None nếu không có hotwords

    x = features[np.newaxis, :, :].astype(np.float32)
    x_lens = np.array([features.shape[0]], dtype=np.int64)
    enc_out, enc_lens = enc_sess.run(None, {"x": x, "x_lens": x_lens})
    T = int(enc_lens[0])
    enc_out = enc_out[0, :T, :]

    init_ys = [-1] * (CONTEXT_SIZE - 1) + [BLANK_ID]
    dec_input = [max(0, y) for y in init_ys]
    ctx_key = tuple(dec_input)
    if ctx_key not in dec_cache:
        dec_cache[ctx_key] = dec_sess.run(
            None, {"y": np.array([dec_input], dtype=np.int64)})[0][0]

    # hyps_dict: key=tuple(ys) → (ys, log_prob, frames, ys_probs, emit_logits, context_state)
    init_ctx_state = ctx_graph.root if ctx_graph else None
    hyps_dict = {tuple(init_ys): (list(init_ys), 0.0, [], [], [], init_ctx_state)}

    # Pre-allocate
    D = enc_out.shape[1]
    D_dec = dec_cache[ctx_key].shape[0]
    enc_buf = np.empty((beam_size, D), dtype=np.float32)
    dec_buf = np.empty((beam_size, D_dec), dtype=np.float32)

    for t in range(T):
        prev = list(hyps_dict.values())
        B = len(prev)

        # Decoder: cache lookup
        missing_ctxs, missing_idx = [], []
        for i, (ys, lp, fr, yp, el, cs) in enumerate(prev):
            ctx = tuple(max(0, y) for y in ys[-CONTEXT_SIZE:])
            cached = dec_cache.get(ctx)
            if cached is not None:
                dec_buf[i] = cached
            else:
                missing_ctxs.append(ctx)
                missing_idx.append(i)

        if missing_ctxs:
            batch = np.array([list(c) for c in missing_ctxs], dtype=np.int64)
            results = dec_sess.run(None, {"y": batch})[0]
            for j, idx in enumerate(missing_idx):
                dec_buf[idx] = results[j]
                dec_cache[missing_ctxs[j]] = results[j].copy()

        # Joiner
        enc_buf[:B] = enc_out[t]
        logits = joi_sess.run(
            None, {"encoder_out": enc_buf[:B], "decoder_out": dec_buf[:B]})[0]

        # Log-softmax + score accumulation
        mx = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - mx
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        for i in range(B):
            log_probs[i, :] += prev[i][1]

        # Global top-k
        flat = log_probs.reshape(-1)
        k = min(beam_size, len(flat))
        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

        # Build new hypotheses with log-sum-exp dedup
        new_hyps = {}
        for idx in top_idx:
            hi = int(idx // V)
            token = int(idx % V)
            score = float(flat[idx])
            p_ys, p_lp, p_fr, p_yp, p_el, p_cs = prev[hi]

            if token == BLANK_ID:
                new_ys = list(p_ys)
                new_fr, new_yp, new_el = list(p_fr), list(p_yp), list(p_el)
                new_cs = p_cs
            else:
                tok_lp = float(log_probs[hi, token]) - p_lp
                new_ys = p_ys + [token]
                new_fr = p_fr + [t]
                new_yp = p_yp + [tok_lp]
                new_el = p_el + [logits[hi].copy()]

                # Hotword boosting (skip blank + unk, giống sherpa-onnx C++)
                new_cs = p_cs
                if ctx_graph is not None and p_cs is not None and token != UNK_ID:
                    hw_delta, new_cs = ctx_graph.forward_one_step(p_cs, token)
                    score += hw_delta

            key = tuple(new_ys)
            if key in new_hyps:
                old = new_hyps[key]
                new_hyps[key] = (old[0], _log_add(old[1], score), old[2], old[3], old[4], old[5])
            else:
                new_hyps[key] = (new_ys, score, new_fr, new_yp, new_el, new_cs)

        hyps_dict = new_hyps

    # Finalize: trừ partial hotword score chưa hoàn thành
    if ctx_graph is not None:
        for key in hyps_dict:
            ys, lp, fr, yp, el, cs = hyps_dict[key]
            if cs is not None:
                lp += ctx_graph.finalize(cs)
            hyps_dict[key] = (ys, lp, fr, yp, el, cs)

    # Length-normalized final selection
    best = max(hyps_dict.values(), key=lambda h: h[1] / max(len(h[0]), 1))
    token_ids = [tok for tok in best[0][CONTEXT_SIZE:] if tok > 0]
    return token_ids, best[2], best[3], T, best[4]


    # (greedy search & decode_chunk_greedy removed — ROVER Model B uses beam=4 via decode_chunk)


def _compute_token_entropy(raw_logits, V):
    """Compute entropy metrics from a single token's joiner logits.
    Shared function — dùng chung cho beam search và greedy search.
    Returns: dict {tsallis_norm, margin, entropy_norm, top1_prob}"""
    max_entropy = math.log(V) if V > 1 else 1.0
    alpha_ts = 1.0 / 3.0
    tsallis_max_val = (1.0 / (alpha_ts - 1.0)) * (1.0 - V ** (1.0 - alpha_ts)) if V > 1 else 1.0

    logits_s = raw_logits - np.max(raw_logits)
    probs = np.exp(logits_s)
    probs /= np.sum(probs)
    entropy = -float(np.sum(probs * np.log(probs + 1e-30)))
    tsallis = float((1.0 / (alpha_ts - 1.0)) * (1.0 - np.sum(probs ** alpha_ts)))
    tsallis_norm = tsallis / tsallis_max_val if tsallis_max_val > 0 else 0.0
    sorted_p = np.sort(probs)[::-1]
    top1 = float(sorted_p[0])
    top2 = float(sorted_p[1]) if len(sorted_p) > 1 else 1e-10
    return {
        "tsallis_norm": round(float(tsallis_norm), 4),
        "margin": round(top1 - top2, 4),
        "entropy_norm": round(entropy / max_entropy, 4),
        "top1_prob": top1,
    }


_ENTROPY_FALLBACK = {"tsallis_norm": 0, "margin": 1, "entropy_norm": 0, "top1_prob": 1.0}


def _finalize_word_entropy(w):
    """Aggregate BPE-level entropy → word-level. Called during BPE merge.
    Per-token confidence: tính confidence per-token rồi average,
    tránh cross-penalty giữa margin_min token A × tsallis_max token B."""
    w["prob"] = sum(w["probs"]) / len(w["probs"])
    del w["probs"]
    ents = w.pop("_ents", [])
    if ents:
        # Giữ tsallis_max/margin_min cho suspect_detect (backward compat)
        w["tsallis_max"] = round(float(max(e["tsallis_norm"] for e in ents)), 4)
        w["margin_min"] = round(float(min(e["margin"] for e in ents)), 4)
        w["entropy_norm"] = round(float(np.mean([e["entropy_norm"] for e in ents])), 4)
        # Per-token confidence: tính trên từng token rồi average
        token_confs = [e["margin"] * (1.0 - e["tsallis_norm"]) for e in ents]
        w["_conf"] = round(float(sum(token_confs) / len(token_confs)), 4)
    else:
        w["tsallis_max"] = None
        w["margin_min"] = None
        w["entropy_norm"] = None
        w["_conf"] = None


def decode_chunk(recognizer, audio_chunk, time_offset=0.0, precomputed_features=None):
    """
    Decode một audio chunk using ORT beam search, trả về merged words list.
    Entropy (tsallis, margin) tính trong cùng 1 pass beam search — không cần decode lần 2.
    precomputed_features: nếu có, dùng thay vì tính fbank lại (ROVER shared fbank).
    """
    id2token = recognizer['id2token']
    beam_size = recognizer.get('max_active_paths', 8)

    # Compute fbank (hoặc dùng precomputed)
    features = precomputed_features if precomputed_features is not None else compute_fbank_ort(audio_chunk, 16000)
    if features.shape[0] == 0:
        return []

    # Beam search decode (returns emit_logits for 1-pass entropy)
    token_ids, frames, ys_log_probs_list, T, emit_logits = _ort_beam_search(
        recognizer, features, beam_size)

    if not token_ids:
        return []

    V = recognizer['vocab_size']

    # Convert token IDs to BPE token strings
    toks = [id2token.get(tid, '') for tid in token_ids]

    # Compute timestamps (frame index → seconds)
    chunk_dur = len(audio_chunk) / 16000.0
    ts = [f / T * chunk_dur for f in frames] if T > 0 else []

    if not ts:
        return []

    if len(ts) >= 2:
        avg_bpe_dur = (ts[-1] - ts[0]) / (len(ts) - 1)
    else:
        avg_bpe_dur = 0.08

    # Compute entropy metrics from emit_logits (1-pass, no 2nd decode)
    token_entropy = []
    for j in range(len(token_ids)):
        if j < len(emit_logits):
            token_entropy.append(_compute_token_entropy(emit_logits[j], V))
        else:
            token_entropy.append(_ENTROPY_FALLBACK)

    chunk_words = []
    for j, (t_val, tok) in enumerate(zip(ts, toks)):
        local_start = t_val
        local_end = ts[j + 1] if j < len(ts) - 1 else local_start + avg_bpe_dur
        prob = math.exp(ys_log_probs_list[j]) if j < len(ys_log_probs_list) else 1.0

        chunk_words.append({
            "text": tok.lower(), "start": local_start + time_offset,
            "end": local_end + time_offset, "local_start": local_start,
            "local_end": local_end, "prob": prob,
            "_ent": token_entropy[j] if j < len(token_entropy) else None,
        })

    # Lưu raw BPE tokens/timestamps
    _raw_bpe_tokens = list(toks)
    _raw_bpe_timestamps_local = list(ts)

    # Merge BPE tokens thành words (+ aggregate entropy per word)
    merged = []
    current_word = None
    for tok_info in chunk_words:
        tok = tok_info["text"]
        ent = tok_info.get("_ent")
        if tok.startswith(" ") or tok.startswith("\u2581"):
            if current_word is not None:
                _finalize_word_entropy(current_word)
                merged.append(current_word)
            current_word = {
                "text": tok.lstrip(" ").lstrip("\u2581"),
                "start": tok_info["start"], "end": tok_info["end"],
                "local_start": tok_info["local_start"], "local_end": tok_info["local_end"],
                "last_bpe_start": tok_info["start"],
                "probs": [tok_info.get("prob", 1.0)],
                "_ents": [ent] if ent else [],
            }
        else:
            if current_word is not None:
                current_word["text"] += tok
                current_word["end"] = tok_info["end"]
                current_word["local_end"] = tok_info["local_end"]
                current_word["last_bpe_start"] = tok_info["start"]
                current_word["probs"].append(tok_info.get("prob", 1.0))
                if ent:
                    current_word["_ents"].append(ent)
            else:
                current_word = {
                    "text": tok, "start": tok_info["start"], "end": tok_info["end"],
                    "local_start": tok_info["local_start"], "local_end": tok_info["local_end"],
                    "last_bpe_start": tok_info["start"],
                    "probs": [tok_info.get("prob", 1.0)],
                    "_ents": [ent] if ent else [],
                }
    if current_word is not None:
        _finalize_word_entropy(current_word)
        merged.append(current_word)

    # Attach raw BPE data vào word đầu tiên (để entropy module lấy được)
    if merged:
        merged[0]["_chunk_bpe_tokens"] = _raw_bpe_tokens
        merged[0]["_chunk_bpe_timestamps_local"] = _raw_bpe_timestamps_local

    # Tính lại word.end dựa trên last_bpe_start + avg_bpe_duration
    for wi in range(len(merged)):
        w = merged[wi]
        estimated_end = w["last_bpe_start"] + avg_bpe_dur
        if wi < len(merged) - 1:
            estimated_end = min(estimated_end, merged[wi + 1]["start"])
        w["end"] = estimated_end
        w["local_end"] = estimated_end - time_offset
        del w["last_bpe_start"]

    return merged


def _zscore(prob, mean, std):
    """Chuyển prob thô thành z-score (bao nhiêu std trên/dưới mean)."""
    if std < 0.01:
        return 0.0
    return (prob - mean) / std


def _word_confidence(w):
    """Confidence score cho 1 từ, kết hợp margin và tsallis. Range [0, 1].
    margin cao + tsallis thấp = model rất chắc chắn."""
    margin = w.get("margin_min")
    tsallis = w.get("tsallis_max")
    if margin is not None and tsallis is not None:
        return margin * (1.0 - tsallis)
    return w.get("prob", 0.5)


def _block_confidence(words):
    """Average confidence cho một block từ."""
    if not words:
        return 0.0
    scores = [_word_confidence(w) for w in words]
    return sum(scores) / len(scores)


# ── Hotword bonus cho ROVER ──
_hotword_phrases_cache = None


def _get_hotword_phrases():
    """Lấy list hotword phrases (lowercase) từ hotword.txt. Cache lần đầu."""
    global _hotword_phrases_cache
    if _hotword_phrases_cache is not None:
        return _hotword_phrases_cache

    from core.hotword_context import parse_hotwords_file
    phrases = parse_hotwords_file(
        os.path.join(BASE_DIR, "hotword.txt"))
    # Normalize: lowercase, sorted dài → ngắn (match dài trước)
    _hotword_phrases_cache = sorted(
        [p.lower() for p, _ in phrases],
        key=len, reverse=True)
    return _hotword_phrases_cache


def _count_hotword_matches(words, context_before=None, context_after=None):
    """Đếm bao nhiêu từ trong block nằm trong hotword phrase.

    Scan text block (+ context trước/sau) để match hotword.
    Context giải quyết case: "ban" ở equal block + "tổ chức" ở replace block
    → ghép lại "ban tổ chức" → match hotword "BAN TỔ CHỨC".

    Chỉ đếm match trên từ trong block chính (không đếm context).

    Args:
        words: block words chính (replace block)
        context_before: list words trước block (từ equal block trước)
        context_after: list words sau block (từ equal block sau)

    Returns: số từ trong block được hotword support / tổng từ (ratio 0.0 - 1.0)
    """
    if not words:
        return 0.0
    phrases = _get_hotword_phrases()
    if not phrases:
        return 0.0

    # Ghép context + block + context
    ctx_before = list(context_before or [])
    ctx_after = list(context_after or [])
    all_words = ctx_before + list(words) + ctx_after

    text = ' '.join(normalize_word_for_overlap(w["text"]) for w in all_words)
    matched_chars = set()

    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            for c in range(idx, idx + len(phrase)):
                matched_chars.add(c)
            start = idx + 1

    if not matched_chars:
        return 0.0

    # Chỉ đếm match trên từ TRONG BLOCK (skip context)
    n_matched_words = 0
    pos = 0
    block_start_idx = len(ctx_before)  # index đầu tiên của block trong all_words
    block_end_idx = block_start_idx + len(words)

    for wi, w in enumerate(all_words):
        w_text = normalize_word_for_overlap(w["text"])
        w_start = text.find(w_text, pos)
        if w_start >= 0:
            w_end = w_start + len(w_text)
            # Chỉ đếm nếu từ nằm trong block chính
            if block_start_idx <= wi < block_end_idx:
                if any(c in matched_chars for c in range(w_start, w_end)):
                    n_matched_words += 1
            pos = w_end

    return n_matched_words / len(words)


# Hotword ROVER bonus: khi 1 block có hotword mà block kia không có,
# cộng bonus vào confidence của block có hotword.
# bonus = hw_ratio × HOTWORD_ROVER_BONUS
# hw_ratio = % từ trong block match hotword
# Giá trị 0.5: đủ mạnh để ưu tiên hotword khi confidence gần nhau,
# nhưng không override khi confidence chênh quá lớn (>0.5)
HOTWORD_ROVER_BONUS = 0.5


def rover_merge_words(words_a, words_b):
    """
    ROVER v3: Confidence-based word selection giữa Model A và B.

    Quy tắc:
    1. equal → giữ A (cả 2 đồng ý)
    2. replace → so sánh block-level confidence trực tiếp, chọn block tốt hơn
    3. insert (B only) → bổ sung nếu B confidence > 0.20
    4. delete (A only) → giữ A

    Returns: (merged_words, disagree_indices)
    """
    if not words_a:
        return list(words_b) if words_b else [], set()
    if not words_b:
        return list(words_a), set()

    texts_a = [normalize_word_for_overlap(w["text"]) for w in words_a]
    texts_b = [normalize_word_for_overlap(w["text"]) for w in words_b]

    matcher = SequenceMatcher(None, texts_a, texts_b, autojunk=False)

    result = []
    n_replace_b = 0
    n_replace_total = 0
    n_sup = 0

    # Pre-compute opcodes để truy cập context trước/sau mỗi replace block
    opcodes = matcher.get_opcodes()
    CONTEXT_WORDS = 3  # số từ context trước/sau block để match hotword

    for oi, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == 'equal':
            result.extend(words_a[i1:i2])

        elif tag == 'replace':
            block_a = words_a[i1:i2]
            block_b = words_b[j1:j2]

            conf_a = _block_confidence(block_a)
            conf_b = _block_confidence(block_b)

            # Lấy context trước/sau từ equal blocks lân cận
            ctx_before_a = ctx_before_b = None
            ctx_after_a = ctx_after_b = None
            if oi > 0:
                pt, pi1, pi2, pj1, pj2 = opcodes[oi - 1]
                if pt == 'equal':
                    ctx_before_a = words_a[max(pi1, pi2 - CONTEXT_WORDS):pi2]
                    ctx_before_b = words_b[max(pj1, pj2 - CONTEXT_WORDS):pj2]
            if oi < len(opcodes) - 1:
                nt, ni1, ni2, nj1, nj2 = opcodes[oi + 1]
                if nt == 'equal':
                    ctx_after_a = words_a[ni1:min(ni2, ni1 + CONTEXT_WORDS)]
                    ctx_after_b = words_b[nj1:min(nj2, nj1 + CONTEXT_WORDS)]

            # Hotword tiebreaker: match hotword với context rộng
            # VD: equal="ban" + replace_B="tổ chức ký" → context "ban tổ chức ký" → match "BAN TỔ CHỨC"
            hw_a = _count_hotword_matches(block_a, ctx_before_a, ctx_after_a)
            hw_b = _count_hotword_matches(block_b, ctx_before_b, ctx_after_b)

            if hw_a > 0 and hw_b == 0:
                conf_a += hw_a * HOTWORD_ROVER_BONUS
            elif hw_b > 0 and hw_a == 0:
                conf_b += hw_b * HOTWORD_ROVER_BONUS

            n_replace_total += 1

            if conf_b > conf_a:
                chosen = block_b
                n_replace_b += 1
            else:
                chosen = block_a

            for w in chosen:
                w["_disagree"] = True
            result.extend(chosen)

        elif tag == 'delete':
            result.extend(words_a[i1:i2])

        elif tag == 'insert':
            for k in range(j1, j2):
                wb = words_b[k]
                if _word_confidence(wb) > 0.20:
                    wb["_source"] = "B_supplement"
                    wb["_disagree"] = True
                    result.append(wb)
                    n_sup += 1

    # Sort theo timestamp (supplements từ insert có thể nằm sai vị trí)
    result.sort(key=lambda w: w["start"])

    # Dedup: nếu B_supplement trùng timestamp với từ đã có → bỏ
    if n_sup > 0:
        deduped = []
        for w in result:
            if w.get("_source") == "B_supplement":
                overlap = False
                w_norm = normalize_word_for_overlap(w["text"])
                for existing in deduped:
                    if (existing.get("_source") != "B_supplement" and
                        abs(existing["start"] - w["start"]) < 0.15 and
                        normalize_word_for_overlap(existing["text"]) == w_norm):
                        overlap = True
                        break
                if not overlap:
                    deduped.append(w)
                else:
                    n_sup -= 1
                    if DEBUG_LOGGING:
                        print(f"[ROVER] Dropped duplicate B-supplement '{w['text']}' "
                              f"at t={w['start']:.2f}")
            else:
                deduped.append(w)
        result = deduped

    # Build disagree indices từ _disagree flag
    final_disagree = set()
    for i, w in enumerate(result):
        if w.get("_disagree"):
            final_disagree.add(i)

    # Cleanup _source tag, GIỮ _disagree flag trên word dict
    # để rebuild index chính xác sau merge_chunks_with_overlap
    for w in result:
        w.pop("_source", None)

    if n_replace_total > 0 or n_sup > 0:
        print(f"[ROVER] Replace: {n_replace_b}/{n_replace_total}→B | Sup: {n_sup}")

    return result, final_disagree


# =============================================================================
# FILLER REMOVAL: Xóa từ vô nghĩa đứng riêng lẻ (à, ờ, á, a, ...)
# =============================================================================

FILLER_WORDS = {"à", "ờ", "ừ", "ơ", "uh", "um"}


def remove_filler_words(words):
    """
    Xóa các từ vô nghĩa (filler) đứng riêng lẻ sau ASR.
    Chỉ xóa khi từ nằm đơn lẻ (không phải phần của cụm có nghĩa).
    """
    if not words:
        return words

    result = []
    removed = 0
    for w in words:
        if w["text"].lower() in FILLER_WORDS:
            removed += 1
            if DEBUG_LOGGING:
                print(f"[Filler] Removed '{w['text']}' at t={w['start']:.2f}")
        else:
            result.append(w)

    if removed > 0:
        print(f"[Filler] Removed {removed} filler words ({len(words)} → {len(result)})")

    return result


# =============================================================================
# GAP RECOVER: Phát hiện từ bị miss qua phân tích gap giữa từ ASR,
# re-ASR bằng resample slowdown để recover từ bị thiếu.
# Thay thế Peak Surgery cũ — chính xác hơn nhờ phân tích từng gap.
# =============================================================================



def count_energy_peaks(audio_segment, sr=16000, threshold_factor=1.0):
    """Đếm syllable peaks dựa trên energy envelope. Trả về peak times (seconds)."""
    from scipy.signal import find_peaks as _find_peaks

    frame_len = int(sr * 0.010)
    hop_len = int(sr * 0.005)
    num_frames = max(1, (len(audio_segment) - frame_len) // hop_len + 1)

    energy = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_len
        frame = audio_segment[start:start + frame_len]
        if len(frame) > 0:
            energy[i] = np.sqrt(np.mean(frame ** 2))

    kernel = np.hanning(7)
    kernel /= kernel.sum()
    energy_smooth = np.convolve(energy, kernel, mode='same')

    non_silence = energy_smooth[energy_smooth > np.max(energy_smooth) * 0.05]
    if len(non_silence) == 0:
        return []

    threshold = np.mean(non_silence) * threshold_factor
    min_dist = int(90 / (hop_len / sr * 1000))

    peaks, _ = _find_peaks(energy_smooth, distance=min_dist, height=threshold,
                           prominence=threshold * 0.3)
    return (peaks * hop_len / sr).tolist()



def _compute_gap_features(audio_segment, sr=16000):
    """Tính erange và sbr cho một đoạn audio trong gap."""
    if len(audio_segment) < 50:
        return 0.0, 0.0

    # Energy range (dao động sóng âm)
    frame_len = int(sr * 0.010)
    hop_len = int(sr * 0.005)
    num_frames = max(1, (len(audio_segment) - frame_len) // hop_len + 1)
    frame_energies = np.array([
        np.sqrt(np.mean(audio_segment[i * hop_len:i * hop_len + frame_len] ** 2))
        for i in range(num_frames)
    ])
    erange = float(np.max(frame_energies) - np.min(frame_energies))

    # Speech band ratio (tỷ lệ năng lượng dải giọng nói 300-3000Hz)
    n_fft = min(512, len(audio_segment))
    fft_mag = np.abs(np.fft.rfft(audio_segment[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    speech_band = (freqs >= 300) & (freqs <= 3000)
    total_energy = np.sum(fft_mag ** 2) + 1e-10
    sbr = float(np.sum(fft_mag[speech_band] ** 2) / total_energy)

    return erange, sbr


def compute_disagree_indices(words_main, words_other_text):
    """
    So sánh text ASR chính vs model thứ 2, trả về set indices trong words_main
    mà 2 model khác nhau (disagree).

    Args:
        words_main: list of word dicts (ASR chính)
        words_other_text: list of strings (text từ model thứ 2)

    Returns:
        set of indices trong words_main có disagree
    """
    main_text = [normalize_word_for_overlap(w["text"]) for w in words_main]
    other_text = [normalize_word_for_overlap(w) for w in words_other_text]

    matcher = SequenceMatcher(None, main_text, other_text)
    disagree = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        # replace, delete: từ phía main khác other
        for k in range(i1, i2):
            disagree.add(k)
        # insert: chỉ other có → flag neighbor trong main
        if tag == 'insert':
            if i1 > 0:
                disagree.add(i1 - 1)
            if i1 < len(main_text):
                disagree.add(i1)

    return disagree


def suspect_detect(all_words, audio, disagree_indices=None):
    """
    Phát hiện từ nghi ngờ ASR lỗi (sai từ, thiếu từ, dư từ).
    Kết hợp tín hiệu:
      1. Disagree: 2 model Zipformer khác nhau → gần chắc chắn khó
      2. Tsallis entropy (alpha=1/3, max aggregation): đo sự phân vân của model,
         nhạy hơn Shannon entropy với đáp án cạnh tranh yếu (NVIDIA 2025)
      3. Gap acoustic: phát hiện từ bị thiếu giữa 2 từ

    Strategy hiện tại: "disagree OR (tsallis_max > 0.04 AND margin_min < 0.6)"
    Benchmark 2026-03-28 (beam search 1-pass entropy):
      test_vimd (sạch): F1=0.332, %to=2.9%
      11_min (nhiễu):   F1=0.433, %to=12.2%

    TODO: Tích hợp ChunkFormer CTC dominance (khanhld/chunkformer-ctc-large-vie)
    khi chấp nhận được dung lượng (+558MB RAM, +587MB disk).
    Tốc độ ChunkFormer nhanh (~48s/22min audio, ~16s/11min audio).
    Strategy nâng cấp: "disagree OR (tsallis_max > 0.06 AND cf_dom_min < 10)"
    → Yêu cầu CẢ Zipformer phân vân (tsallis) VÀ ChunkFormer CTC phân vân (dominance)
    → Recall 63-81%, precision 43-45% (+4% recall, precision giữ nguyên)
    Cách implement:
      - Load ChunkFormer: ChunkFormerModel.from_pretrained('khanhld/chunkformer-ctc-large-vie')
      - Monkey-patch model.model.ctc.log_softmax để capture CTC logits [T, 6992]
      - model.endless_decode(file_path, chunk_size=64, left/right_context=128)
      - dominance = top1_prob / top2_prob mỗi CTC frame (lọc blank token!=0)
      - Map sang Zip words bằng timestamp: fps = T_frames / audio_duration
      - Mỗi word: cf_dom_min = min(dominance) trong non-blank frames
    Xem d:/tmp/asr-vn-research/test_3signals.py cho reference implementation.

    Args:
        all_words: list of word dicts (có 'text','start','end','prob',
                   và 'tsallis_max'/'entropy_norm' nếu đã tính entropy trước)
        audio: numpy array audio (cho gap detection)
        disagree_indices: set of int — indices trong all_words mà 2 model khác nhau
                          (None nếu không có model thứ 2)

    Returns:
        all_words đã được gắn '_suspect_level' = "warning" vào từ nghi ngờ.
    """
    sr = 16000
    if len(all_words) < 2:
        return all_words

    n = len(all_words)
    suspect_flags = [False] * n
    gap_suspect_indices = set()

    # ── Kiểm tra có entropy data không ──
    has_tsallis = any(w.get("tsallis_max") is not None for w in all_words)
    has_entropy = any(w.get("entropy_norm") is not None for w in all_words)
    has_disagree = disagree_indices is not None and len(disagree_indices) > 0

    # ── WORD-LEVEL SCORING ──
    # Tsallis entropy (alpha=1/3, max aggregation) + margin_min (p1-p2):
    # Tsallis đo model phân vân, margin_min đo khoảng cách top1-top2 nhỏ nhất.
    # AND kết hợp: chỉ flag khi CẢ 2 tín hiệu đồng ý (tsallis cao VÀ margin thấp).
    # Benchmark 2026-03-28: F1 tăng gấp 2 so với ngưỡng cũ (0.08/0.5).
    #   test_vimd (sạch): F1=0.332, %to=2.9%
    #   11_min (nhiễu):   F1=0.433, %to=12.2%
    TSALLIS_TH = 0.04       # tsallis_max > 0.04 (cũ: 0.08)
    MARGIN_TH = 0.6         # margin_min < 0.6 (cũ: 0.5)
    ENTROPY_TH = 0.10       # fallback Shannon nếu chưa có tsallis

    has_margin = any(w.get("margin_min") is not None for w in all_words)

    for i in range(n):
        # Tín hiệu 1: Disagree (2 model ra từ khác nhau)
        if has_disagree and i in disagree_indices:
            suspect_flags[i] = True
            continue

        # Tín hiệu 2: Tsallis entropy AND margin_min (cả 2 phải thỏa)
        if has_tsallis:
            ts = all_words[i].get("tsallis_max")
            mg = all_words[i].get("margin_min")
            if ts is not None and ts > TSALLIS_TH:
                if has_margin and mg is not None:
                    # AND: tsallis cao VÀ margin thấp → model thực sự phân vân
                    if mg < MARGIN_TH:
                        suspect_flags[i] = True
                else:
                    # Không có margin data → dùng tsallis alone (ngưỡng chặt hơn)
                    if ts > 0.12:
                        suspect_flags[i] = True
        elif has_entropy:
            ent = all_words[i].get("entropy_norm")
            if ent is not None and ent > ENTROPY_TH:
                suspect_flags[i] = True

    # ── GAP-LEVEL DETECTION ──
    GAP_MIN_MS = 200
    GAP_VAD_TH = 0.90
    GAP_ERANGE_TH = 0.04
    GAP_LONG_MS = 500
    GAP_PEAKS_TH = 3

    for i in range(n - 1):
        wc, wn = all_words[i], all_words[i + 1]
        gap_start = wc["end"]
        gap_end = wn["start"]
        gap_ms = (gap_end - gap_start) * 1000

        if gap_ms < GAP_MIN_MS:
            continue

        gs = int(gap_start * sr)
        ge = int(gap_end * sr)
        if gs >= ge or gs < 0 or ge > len(audio):
            continue

        gap_audio = audio[gs:ge]
        if len(gap_audio) < 80:
            continue

        peaks = count_energy_peaks(gap_audio, sr)
        gap_erange, _ = _compute_gap_features(gap_audio, sr)

        from core.vad_utils import get_cached_vad_probs
        vad_probs = get_cached_vad_probs()
        vad_max_val = 0.0
        if vad_probs is not None:
            w0 = max(0, min(gs // 512, len(vad_probs) - 1))
            w1 = max(w0 + 1, min(ge // 512, len(vad_probs)))
            gv = vad_probs[w0:w1]
            if len(gv) > 0:
                vad_max_val = float(np.max(gv))

        if (vad_max_val >= GAP_VAD_TH
                and (gap_ms >= GAP_LONG_MS or len(peaks) >= GAP_PEAKS_TH)
                and gap_erange >= GAP_ERANGE_TH):
            gap_suspect_indices.add(i)
            wc["gap_after_ms"] = int(gap_ms)
            wn["gap_before_ms"] = int(gap_ms)

    # ── GÁN KẾT QUẢ ──
    n_disagree = n_entropy = n_gap = 0
    for i in range(n):
        if suspect_flags[i]:
            all_words[i]["_suspect_level"] = "warning"
            if has_disagree and i in disagree_indices:
                n_disagree += 1
            else:
                n_entropy += 1
        elif i in gap_suspect_indices or (i > 0 and i - 1 in gap_suspect_indices):
            all_words[i]["_suspect_level"] = "warning"
            n_gap += 1

    n_total = n_disagree + n_entropy + n_gap
    ent_label = "tsallis+margin" if (has_tsallis and has_margin) else ("tsallis" if has_tsallis else "entropy")
    if n_total > 0:
        print(f"[SuspectDetect] {n_total}/{n} từ nghi ngờ "
              f"({n_total * 100 / n:.0f}%): "
              f"{n_disagree} disagree + {n_entropy} {ent_label} + {n_gap} gap")

    return all_words







# =============================================================================
# CORE TRANSCRIPTION PIPELINE (No Qt dependency)
# =============================================================================

class TranscriberPipeline:
    """
    Core ASR pipeline - không phụ thuộc PyQt6.
    Dùng callback thay cho pyqtSignal.

    Usage:
        pipeline = TranscriberPipeline(file_path, model_path, config,
                                        progress_callback=print)
        result = pipeline.run()
    """

    def __init__(self, file_path, model_path, config,
                 progress_callback=None, cancel_check=None):
        """
        Args:
            file_path: Đường dẫn file audio
            model_path: Đường dẫn thư mục model
            config: dict cấu hình (cpu_threads, restore_punctuation, ...)
            progress_callback: callable(str) - nhận thông báo tiến trình
            cancel_check: callable() -> bool - trả True nếu cần hủy
        """
        self.file_path = file_path
        self.model_path = model_path
        self.config = config
        self.progress_callback = progress_callback or (lambda msg: None)
        self.cancel_check = cancel_check or (lambda: False)

    _last_phase = None
    _phase_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.asr_phase')

    def _emit(self, msg):
        self.progress_callback(msg)
        # Ghi phase ra file cho monitor (chỉ khi phase ĐỔI, tránh disk I/O thừa)
        if msg.startswith("PHASE:"):
            phase_name = msg.split("|")[0]
            if phase_name != self._last_phase:
                self._last_phase = phase_name
                try:
                    with open(self._phase_file, 'w', encoding='utf-8') as f:
                        f.write(msg)
                except OSError:
                    pass

    @staticmethod
    def _cleanup_phase_file():
        """Xóa file .asr_phase (khi bắt đầu hoặc kết thúc pipeline)."""
        try:
            pf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.asr_phase')
            if os.path.exists(pf):
                os.remove(pf)
        except OSError:
            pass

    def _is_cancelled(self):
        return self.cancel_check()

    def run(self):
        """
        Chạy toàn bộ pipeline ASR.

        Returns:
            dict với keys: text, segments, timing, paragraphs,
                          has_speaker_diarization, speaker_segments_raw
        """
        self._cleanup_phase_file()  # Xóa stale phase từ lần trước
        self._last_phase = None
        try:
            return self._run_pipeline()
        finally:
            self._cleanup_phase_file()  # Xóa khi xong hoặc crash
            if self.config.get("save_ram", False):
                try:
                    clear_model_cache("all")
                    unload_vad_model()
                except Exception:
                    pass

    def _run_pipeline(self):
        """Internal pipeline implementation."""
        start_time = time.time()
        self._emit("PHASE:Init|Đang khởi tạo mô hình|0")

        timing_details = {
            "upload_convert": 0.0,
            "transcription": 0.0,
            "sentence_segmentation": 0.0,
            "punctuation": 0.0,
            "alignment": 0.0,
            "diarization": 0.0,
        }

        cpu_threads = self.config.get("cpu_threads", 4)
        restore_punctuation = self.config.get("restore_punctuation", False)

        # Detect ROVER mode
        is_rover = self.config.get("rover_mode", False)
        rover_recognizer = None  # Model phụ cho ROVER

        if is_rover:
            # ROVER: load cả 2 model
            # model_path có thể là thư mục models/ (từ UI) hoặc models/xxx (fallback)
            # Thử trực tiếp trước, nếu không có thì lùi 1 cấp
            if os.path.isdir(os.path.join(self.model_path, ROVER_MODEL_IDS[0])):
                models_dir = self.model_path
            else:
                models_dir = os.path.dirname(self.model_path)
            missing = []
            for mid in ROVER_MODEL_IDS:
                p = os.path.join(models_dir, mid)
                if not os.path.isdir(p):
                    missing.append(mid)
            if missing:
                raise FileNotFoundError(
                    f"ROVER cần cả 2 model. Thiếu: {', '.join(missing)}\n"
                    f"Vui lòng tải về thư mục 'models'."
                )

            self._emit("PHASE:Init|Đang khởi tạo 2 mô hình (ROVER)|10")
            primary_path = os.path.join(models_dir, ROVER_MODEL_IDS[0])
            secondary_path = os.path.join(models_dir, ROVER_MODEL_IDS[1])

            recognizer = create_recognizer(primary_path, cpu_threads, max_active_paths=8)
            self._emit("PHASE:Init|Đang khởi tạo mô hình phụ (ROVER)|40")
            rover_recognizer = create_recognizer(secondary_path, cpu_threads, max_active_paths=8)
            self._emit("PHASE:Init|Đã khởi tạo 2 mô hình (ROVER)|60")
            print(f"[ROVER] Loaded 2 models: {ROVER_MODEL_IDS[0]} + {ROVER_MODEL_IDS[1]}")
        else:
            # Single model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục mô hình tại: {self.model_path}\n"
                    f"Vui lòng tải mô hình và đặt vào thư mục 'models'."
                )

            self._emit("PHASE:Init|Đang khởi tạo mô hình|30")
            recognizer = create_recognizer(self.model_path, cpu_threads)
            # Clear decoder cache từ file trước (tránh tích lũy unbounded trong server mode)
            recognizer['dec_cache'].clear()
            self._emit("PHASE:Init|Đang khởi tạo mô hình|60")

        # Load audio
        self._emit("PHASE:LoadAudio|Đang đọc file audio|0")
        load_audio_start = time.time()

        self._emit("PHASE:LoadAudio|Đang chuẩn hóa audio|30")
        audio = load_audio(self.file_path, progress_callback=lambda msg: self._emit(f"PHASE:LoadAudio|{msg}|35"),
                           res_type=self.config.get("resample_quality", "soxr_hq"))

        timing_details["upload_convert"] = time.time() - load_audio_start
        # Cộng thời gian ffmpeg pre-convert (nếu có, từ BackgroundAudioConvertThread)
        timing_details["upload_convert"] += self.config.get("_pre_convert_time", 0.0)
        self._emit(f"PHASE:LoadAudio|Đã tải audio ({timing_details['upload_convert']:.1f}s)|100")

        # Segment audio
        total_samples = len(audio)

        # --- VAD-based segmentation → concat speech → silence-based chunking ---
        # Pipeline: VAD → preprocess → concat speech (bỏ silence) → chunk 30s → ASR → map timestamp về gốc
        vad_segments = None
        offset_map = None    # map timestamp concat → original
        concat_audio = None  # audio chỉ chứa speech (dùng cho ASR)

        try:
            if self.config.get("bypass_vad", False):
                raise RuntimeError("VAD_BYPASSED_BY_USER")

            self._emit("PHASE:VAD|Đang phát hiện vùng có tiếng nói|0")
            vad_segments = get_vad_segments(audio, progress_callback=self._emit)
            self._emit(f"PHASE:VAD|Phát hiện {len(vad_segments)} đoạn nói|100")

            total_speech_sec = sum(e - s for s, e in vad_segments) / 16000.0
            total_audio_sec = total_samples / 16000.0
            print(f"[VAD] {len(vad_segments)} segments "
                  f"({total_speech_sec:.1f}s speech / {total_audio_sec:.1f}s audio)")

            # --- Audio preprocessing (RMS normalize trên full audio, cần global VAD context) ---
            if not self.config.get("skip_preprocessing", False):
                try:
                    from core.audio_preprocessing import preprocess_audio
                    preprocess_start = time.time()
                    audio = preprocess_audio(
                        audio, vad_segments,
                        sample_rate=16000,
                        enable_rms_normalize=self.config.get("preprocess_rms_normalize", False),
                        progress_callback=self._emit,
                    )
                    preprocess_time = time.time() - preprocess_start
                    timing_details["preprocessing"] = preprocess_time
                    print(f"[Preprocess] Done in {preprocess_time:.2f}s")
                except Exception as e:
                    print(f"[Preprocess] Error (skipping): {e}")

            # --- Merge VAD segments có gap nhỏ (< 5s) ---
            # Tránh cắt speech ở ranh giới chuyển người nói (background noise, pause ngắn)
            MAX_VAD_GAP = 5 * 16000  # 5s in samples
            if len(vad_segments) > 1:
                merged_vad = [vad_segments[0]]
                for seg_s, seg_e in vad_segments[1:]:
                    prev_s, prev_e = merged_vad[-1]
                    if seg_s - prev_e <= MAX_VAD_GAP:
                        merged_vad[-1] = (prev_s, seg_e)  # merge
                    else:
                        merged_vad.append((seg_s, seg_e))
                if len(merged_vad) < len(vad_segments):
                    print(f"[VAD] Merged {len(vad_segments)} -> {len(merged_vad)} segments (gap < 5s)")
                vad_segments = merged_vad

            # --- Concat speech: nối VAD segments, bỏ silence ---
            self._emit("PHASE:VAD|Đang nối các đoạn nói|95")
            concat_audio, offset_map = concat_vad_speech(audio, vad_segments)
            silence_removed = (total_samples - len(concat_audio)) / 16000.0
            print(f"[Concat] {len(vad_segments)} segments -> {len(concat_audio)/16000.0:.1f}s speech "
                  f"(bỏ {silence_removed:.1f}s silence)")

            # --- Silence-based chunking trên concat audio ---
            concat_total = len(concat_audio)
            concat_silent_regions = find_silent_regions(concat_audio)

            segment_samples = 16000 * 30
            segment_boundaries = [0]
            current_pos = 0
            while current_pos + segment_samples < concat_total:
                target = current_pos + segment_samples
                best_split = find_best_split_point(target, concat_total, concat_silent_regions)
                if best_split <= current_pos + 20 * 16000:
                    best_split = target
                segment_boundaries.append(best_split)
                current_pos = best_split
            segment_boundaries.append(concat_total)

            chunk_plan = []  # [(start, end, overlap_at_start)] trên concat_audio
            for i in range(len(segment_boundaries) - 1):
                logical_start = segment_boundaries[i]
                logical_end = segment_boundaries[i + 1]
                if i == 0:
                    chunk_plan.append((logical_start, logical_end, 0))
                else:
                    actual_start = max(0, logical_start - OVERLAP_SAMPLES)
                    chunk_plan.append((actual_start, logical_end, logical_start - actual_start))

            print(f"[Chunk] {len(chunk_plan)} chunks trên concat audio")

            # Giải phóng VAD model trước khi transcribe
            if self.config.get("save_ram", False):
                unload_vad_model()
                import gc
                gc.collect()

        except Exception as e:
            # Fallback: silence-based chunking trên original audio nếu VAD lỗi hoặc user bypass
            if str(e) == "VAD_BYPASSED_BY_USER":
                print("[VAD] Bypassed by user — silence-based chunking trên toàn audio")
                self._emit("PHASE:LoadAudio|Bỏ qua VAD, đang phân tích khoảng lặng|60")
            else:
                print(f"[VAD] Error: {e}, fallback to silence-based chunking")
                self._emit("PHASE:LoadAudio|Đang phân tích khoảng lặng (fallback)|60")

            concat_audio = audio  # fallback: dùng nguyên audio gốc
            offset_map = [(0, 0, total_samples)]  # identity map

            silent_regions = find_silent_regions(audio)
            segment_samples = 16000 * 30
            segment_boundaries = [0]
            current_pos = 0
            while current_pos + segment_samples < total_samples:
                target = current_pos + segment_samples
                best_split = find_best_split_point(target, total_samples, silent_regions)
                if best_split <= current_pos + 20 * 16000:
                    best_split = target
                segment_boundaries.append(best_split)
                current_pos = best_split
            segment_boundaries.append(total_samples)

            chunk_plan = []
            for i in range(len(segment_boundaries) - 1):
                logical_start = segment_boundaries[i]
                logical_end = segment_boundaries[i + 1]
                if i == 0:
                    chunk_plan.append((logical_start, logical_end, 0))
                else:
                    actual_start = max(0, logical_start - OVERLAP_SAMPLES)
                    chunk_plan.append((actual_start, logical_end, logical_start - actual_start))

        num_chunks = len(chunk_plan)

        # Entropy đã tính trong decode_chunk (1-pass beam search) — không cần setup riêng

        # Transcribe chunks
        chunk_results = []
        phase_label = "Đang chuyển thành văn bản (ROVER)" if is_rover else "Đang chuyển thành văn bản"
        self._emit(f"PHASE:Transcription|{phase_label}|0")

        rover_raw_a = []
        rover_raw_b = []

        # ── Helper: process 1 chunk ──
        def _process_single_chunk(rec, i, actual_start, actual_end, overlap_at_start,
                                  use_wpe=False, shared_fbank=None):
            """Decode 1 chunk, map timestamps. Thread-safe (no shared mutable state)."""
            chunk_audio = concat_audio[actual_start:actual_end]
            concat_time_offset = actual_start / 16000.0

            if use_wpe:
                try:
                    from core.audio_preprocessing import apply_wpe_dereverberation, adaptive_peak_limit
                    chunk_audio = apply_wpe_dereverberation(chunk_audio)
                    chunk_audio = adaptive_peak_limit(chunk_audio)
                except Exception as e:
                    logger.warning(f"[WPE] Chunk {i} error (skipping): {e}")

            chunk_words = decode_chunk(rec, chunk_audio, concat_time_offset,
                                       precomputed_features=shared_fbank)
            for w in chunk_words:
                w["start"] = map_concat_time_to_original(w["start"], offset_map)
                w["end"] = map_concat_time_to_original(w["end"], offset_map)

            return {
                "text": " ".join(w["text"] for w in chunk_words) if chunk_words else "",
                "words": chunk_words,
                "audio_start_abs": concat_time_offset,
                "audio_end_abs": actual_end / 16000.0,
                "overlap_sec": overlap_at_start / 16000.0,
                "vad_group": 0,
            }

        use_wpe = self.config.get("preprocess_wpe", False)

        # ── Parallel 2-worker ASR (>= 4 chunks, >= 4 physical cores) ──
        # Benchmark (6C/12T, 1h audio):
        #   Non-ROVER: 129.9s → 77.1s (1.68x), CPU 75→83%
        #   ROVER:     288.2s → 169.0s (1.70x), CPU 77→84%
        #
        # Thread allocation strategy based on total available threads:
        #   HT (logical > physical): each worker gets PHYSICAL_CORES threads
        #     6C/12T: enc=6 each, total 12 = all HT threads, beam search fills idle HT
        #     4C/8T:  enc=4 each, total 8  = all HT threads
        #   No HT (logical == physical, >= 6 cores): each worker gets threads//2
        #     8C/8T:  enc=4 each, total 8  = all cores, beam search interleaves
        #   < 4 cores: sequential (not enough cores to benefit)
        from core.config import LOGICAL_THREADS, PHYSICAL_CORES as _PHYS
        has_ht = (LOGICAL_THREADS > _PHYS)
        use_parallel = (num_chunks >= 4 and _PHYS >= 4)

        if use_parallel:
            import threading

            if has_ht:
                # HT: each worker gets full physical cores
                # Total 2*P = L → HT threads absorb overlap
                worker_threads = _PHYS
            else:
                # No HT: split cores evenly between 2 workers
                # 8C: 4+4, 6C: 3+3, 4C: 2+2
                worker_threads = max(2, _PHYS // 2)

            # Resolve model paths for worker creation
            if is_rover:
                # ROVER: primary_path and secondary_path already resolved at init
                if os.path.isdir(os.path.join(self.model_path, ROVER_MODEL_IDS[0])):
                    _models_dir = self.model_path
                else:
                    _models_dir = os.path.dirname(self.model_path)
                _primary_path = os.path.join(_models_dir, ROVER_MODEL_IDS[0])
                _secondary_path = os.path.join(_models_dir, ROVER_MODEL_IDS[1])
            else:
                _primary_path = self.model_path

            # Recreate recognizer(s) with worker_threads if different from original
            if worker_threads != cpu_threads:
                recognizer = create_recognizer(_primary_path, worker_threads,
                                               max_active_paths=8 if is_rover else 8)
                recognizer['dec_cache'].clear()
                if is_rover and rover_recognizer is not None:
                    rover_recognizer = create_recognizer(
                        _secondary_path, worker_threads, max_active_paths=8)
                    rover_recognizer['dec_cache'].clear()

            # Create 2nd worker's recognizer(s) with separate decoder caches
            recognizer_2 = create_recognizer(_primary_path, worker_threads,
                                              max_active_paths=8 if is_rover else 8)
            recognizer_2 = dict(recognizer_2)
            recognizer_2['dec_cache'] = {}

            rover_recognizer_2 = None
            if is_rover and rover_recognizer is not None:
                rover_recognizer_2 = create_recognizer(
                    _secondary_path, worker_threads, max_active_paths=8)
                rover_recognizer_2 = dict(rover_recognizer_2)
                rover_recognizer_2['dec_cache'] = {}

            chunk_results = [None] * num_chunks
            # ROVER: per-chunk raw words for merge later
            rover_raw_a_arr = [None] * num_chunks if is_rover else None
            rover_raw_b_arr = [None] * num_chunks if is_rover else None

            _completed = [0]
            _lock = threading.Lock()
            _errors = {}

            def _worker(rec, rec_rover, chunk_indices, worker_id):
                try:
                    for idx in chunk_indices:
                        if self._is_cancelled():
                            return
                        actual_start, actual_end, overlap_at_start = chunk_plan[idx]

                        if is_rover and rec_rover is not None:
                            # ROVER: fbank once, decode both models
                            chunk_audio = concat_audio[actual_start:actual_end]
                            concat_time_offset = actual_start / 16000.0

                            if use_wpe:
                                try:
                                    from core.audio_preprocessing import apply_wpe_dereverberation, adaptive_peak_limit
                                    chunk_audio = apply_wpe_dereverberation(chunk_audio)
                                    chunk_audio = adaptive_peak_limit(chunk_audio)
                                except Exception as e:
                                    logger.warning(f"[WPE] Chunk {idx} error (skipping): {e}")

                            shared_fbank = compute_fbank_ort(chunk_audio, 16000)
                            words_a = decode_chunk(rec, chunk_audio, concat_time_offset,
                                                    precomputed_features=shared_fbank)
                            words_b = decode_chunk(rec_rover, chunk_audio, concat_time_offset,
                                                    precomputed_features=shared_fbank)

                            for w in words_a:
                                w["start"] = map_concat_time_to_original(w["start"], offset_map)
                                w["end"] = map_concat_time_to_original(w["end"], offset_map)
                            for w in words_b:
                                w["start"] = map_concat_time_to_original(w["start"], offset_map)
                                w["end"] = map_concat_time_to_original(w["end"], offset_map)

                            rover_raw_a_arr[idx] = words_a
                            rover_raw_b_arr[idx] = words_b

                            chunk_results[idx] = {
                                "text": " ".join(w["text"] for w in words_a) if words_a else "",
                                "words": words_a,
                                "audio_start_abs": concat_time_offset,
                                "audio_end_abs": actual_end / 16000.0,
                                "overlap_sec": overlap_at_start / 16000.0,
                                "vad_group": 0,
                            }
                        else:
                            # Non-ROVER: single model
                            chunk_results[idx] = _process_single_chunk(
                                rec, idx, actual_start, actual_end, overlap_at_start,
                                use_wpe=use_wpe)

                        with _lock:
                            _completed[0] += 1
                            pct = int(_completed[0] / num_chunks * 100)
                            self._emit(f"PHASE:Transcription|{phase_label}|{pct}")
                except Exception as e:
                    _errors[worker_id] = e
                    logger.error(f"[ASR Worker {worker_id}] Error: {e}")

            # Split: even chunks → worker 1, odd chunks → worker 2
            indices_w1 = list(range(0, num_chunks, 2))
            indices_w2 = list(range(1, num_chunks, 2))

            w1 = threading.Thread(target=_worker,
                                  args=(recognizer, rover_recognizer, indices_w1, 1),
                                  name="asr-w1")
            w2 = threading.Thread(target=_worker,
                                  args=(recognizer_2, rover_recognizer_2, indices_w2, 2),
                                  name="asr-w2")
            w1.start()
            w2.start()
            w1.join()
            w2.join()

            if _errors:
                logger.error(f"[ASR Parallel] Worker errors: {_errors}")
                for i in range(num_chunks):
                    if chunk_results[i] is None:
                        actual_start, actual_end, overlap_at_start = chunk_plan[i]
                        chunk_results[i] = _process_single_chunk(
                            recognizer, i, actual_start, actual_end, overlap_at_start,
                            use_wpe=use_wpe)

            # Collect ROVER raw words (ordered by chunk index)
            if is_rover and rover_raw_a_arr is not None:
                rover_raw_a = [w for w in rover_raw_a_arr if w is not None]
                rover_raw_b = [w for w in rover_raw_b_arr if w is not None]

            del recognizer_2
            if rover_recognizer_2 is not None:
                del rover_recognizer_2

        else:
            # ── Sequential fallback: < 4 chunks or < 4 physical cores ──
            for i, (actual_start, actual_end, overlap_at_start) in enumerate(chunk_plan):
                if self._is_cancelled():
                    return None

                if is_rover:
                    chunk_audio = concat_audio[actual_start:actual_end]
                    concat_time_offset = actual_start / 16000.0

                    if use_wpe:
                        try:
                            from core.audio_preprocessing import apply_wpe_dereverberation, adaptive_peak_limit
                            chunk_audio = apply_wpe_dereverberation(chunk_audio)
                            chunk_audio = adaptive_peak_limit(chunk_audio)
                        except Exception as e:
                            logger.warning(f"[WPE] Chunk {i} error (skipping): {e}")

                    shared_fbank = compute_fbank_ort(chunk_audio, 16000)
                    chunk_words = decode_chunk(recognizer, chunk_audio, concat_time_offset,
                                               precomputed_features=shared_fbank)
                    chunk_words_b = decode_chunk(rover_recognizer, chunk_audio, concat_time_offset,
                                                 precomputed_features=shared_fbank)

                    for w in chunk_words:
                        w["start"] = map_concat_time_to_original(w["start"], offset_map)
                        w["end"] = map_concat_time_to_original(w["end"], offset_map)
                    for w in chunk_words_b:
                        w["start"] = map_concat_time_to_original(w["start"], offset_map)
                        w["end"] = map_concat_time_to_original(w["end"], offset_map)

                    rover_raw_a.append(chunk_words)
                    rover_raw_b.append(chunk_words_b)
                    chunk_words_for_result = chunk_words
                else:
                    cr = _process_single_chunk(
                        recognizer, i, actual_start, actual_end, overlap_at_start,
                        use_wpe=use_wpe)
                    chunk_words_for_result = cr["words"]

                chunk_results.append({
                    "text": " ".join(w["text"] for w in chunk_words_for_result) if chunk_words_for_result else "",
                    "words": chunk_words_for_result,
                    "audio_start_abs": actual_start / 16000.0,
                    "audio_end_abs": actual_end / 16000.0,
                    "overlap_sec": overlap_at_start / 16000.0,
                    "vad_group": 0,
                })

                percent = int((i + 1) / num_chunks * 100)
                self._emit(f"PHASE:Transcription|{phase_label}|{percent}")

        # ROVER: giữ A, bổ sung phần dư từ B, track disagree
        rover_disagree = set()
        if is_rover and rover_raw_a:
            self._emit("PHASE:Transcription|Đang merge ROVER (A + B supplement)|99")

            word_offset = 0  # track global index cho disagree_indices
            for ci in range(len(rover_raw_a)):
                words_a = rover_raw_a[ci]
                words_b = rover_raw_b[ci]

                merged, chunk_disagree = rover_merge_words(words_a, words_b)
                # Shift disagree indices to global
                for idx in chunk_disagree:
                    rover_disagree.add(word_offset + idx)

                chunk_results[ci]["words"] = merged
                chunk_results[ci]["text"] = " ".join(w["text"] for w in merged) if merged else ""
                word_offset += len(merged)

        # Merge chunks overlap (tất cả cùng 1 group vì đã concat)
        all_words = []
        if len(chunk_results) == 1:
            all_words.extend(chunk_results[0]["words"])
        elif len(chunk_results) > 1:
            merged_w, _ = merge_chunks_with_overlap(chunk_results, OVERLAP_SEC)
            all_words.extend(merged_w)

        # ── DNSMOS: tính ngay trên concat_audio (đã VAD = chỉ speech, không cần VAD lại) ──
        # concat_audio chỉ chứa tiếng nói → lấy 3 mẫu 9s (đầu/giữa/cuối) → 3 DNSMOS calls
        self._emit("PHASE:QualityAnalysis|Đang phân tích chất lượng âm thanh|0")
        _dnsmos_result = None
        try:
            from core.audio_analyzer import AudioQualityAnalyzer
            _dnsmos_analyzer = AudioQualityAnalyzer()
            DNSMOS_LEN = 144160  # 9.01s @ 16kHz — đúng input size DNSMOS model
            concat_len = len(concat_audio)
            all_dnsmos = []
            if concat_len >= 8000:  # >= 0.5s speech
                positions = [0.15, 0.50, 0.85]
                for pos in positions:
                    center = int(concat_len * pos)
                    start = max(0, center - DNSMOS_LEN // 2)
                    end = min(concat_len, start + DNSMOS_LEN)
                    if end - start >= 8000:
                        d = _dnsmos_analyzer.compute_dnsmos(concat_audio[start:end])
                        if d:
                            all_dnsmos.append(d)
            if all_dnsmos:
                _dnsmos_result = {
                    "dnsmos_sig": round(float(np.mean([s["SIG"] for s in all_dnsmos])), 2),
                    "dnsmos_bak": round(float(np.mean([s["BAK"] for s in all_dnsmos])), 2),
                    "dnsmos_ovrl": round(float(np.mean([s["OVRL"] for s in all_dnsmos])), 2),
                }
            del _dnsmos_analyzer
            self._emit("PHASE:QualityAnalysis|Phân tích chất lượng hoàn tất|100")
        except Exception as e:
            print(f"[DNSMOS] Error (non-critical): {e}")

        # Giải phóng intermediate lists + concat_audio — đã merge vào all_words, ASR xong
        del chunk_results, rover_raw_a, rover_raw_b, concat_audio

        full_text = " ".join(w["text"] for w in all_words)

        if full_text:
            full_text = full_text.capitalize()

        print(f"[Transcriber] Merged chunks into {len(all_words)} words")

        # Entropy: đã tính per-chunk trong transcription loop → không cần tính lại
        # (cả single-model và ROVER đều tính entropy trong loop trên)

        # ── Disagree: rebuild từ _disagree flag trên word dict (stable sau mọi merge/cut) ──
        if is_rover:
            disagree_indices = set()
            for i, w in enumerate(all_words):
                if w.get("_disagree"):
                    disagree_indices.add(i)
                    w.pop("_disagree", None)  # cleanup sau khi rebuild
            disagree_indices = disagree_indices if disagree_indices else None
        else:
            disagree_indices = None

        if disagree_indices:
            print(f"[Disagree] ROVER: {len(disagree_indices)}/{len(all_words)} từ "
                  f"({len(disagree_indices)*100/max(1,len(all_words)):.1f}%)")

        # Suspect Detect: disagree OR entropy (tsallis > 0.04 AND margin < 0.6)
        all_words = suspect_detect(all_words, audio, disagree_indices=disagree_indices)

        import gc; gc.collect()

        # Xóa từ vô nghĩa (filler words)
        all_words = remove_filler_words(all_words)
        full_text = " ".join(w["text"] for w in all_words)
        if full_text:
            full_text = full_text.capitalize()

        transcription_end_time = time.time()
        timing_details["transcription"] = transcription_end_time - start_time - timing_details["upload_convert"]

        # Giải phóng ASR recognizer nếu tiết kiệm RAM
        # (recognizer không còn cần sau transcription)
        if self.config.get("save_ram", False):
            clear_model_cache("recognizer")

        restore_duration = 0.0
        final_segments = []
        paragraphs = []


        # ══════════════════════════════════════════════════════
        # Speaker diarization — chạy TRƯỚC punctuation
        # Flow: diarization gán speaker cho all_words → split by speaker →
        #       punctuation per speaker → sentence split
        # ══════════════════════════════════════════════════════
        speaker_segments = []
        speaker_segments_raw = []
        diarization_start = None
        _diar_speaker_groups = None

        if self.config.get("speaker_diarization", False) and all_words:
            try:
                from core.speaker_diarization import SpeakerDiarizer
                DIARIZATION_AVAILABLE = True
            except ImportError:
                DIARIZATION_AVAILABLE = False

            if DIARIZATION_AVAILABLE:
                try:
                    diarization_start = time.time()
                    self._emit("PHASE:Diarization|Đang phân tách Người nói|0")

                    num_speakers = self.config.get("num_speakers", 2)
                    speaker_model_id = self.config.get("speaker_model", "titanet_small")
                    hf_token = self.config.get("hf_token") or os.environ.get('HF_TOKEN', None)

                    self._emit("PHASE:Diarization|Đang tải model phân tách|5")

                    # --- CAM++ Senko pipeline (best for all audio lengths) ---
                    if speaker_model_id in ("senko_campp", "senko_campp_optimized"):
                        if speaker_model_id == "senko_campp_optimized":
                            from core.speaker_diarization_senko_campp_optimized import SenkoCamppDiarizerOptimized
                            diarizer_3d = SenkoCamppDiarizerOptimized(
                                num_speakers=num_speakers,
                                num_threads=self.config.get("cpu_threads", 4))
                        else:
                            from core.speaker_diarization_senko_campp import SenkoCamppDiarizer
                            diarizer_3d = SenkoCamppDiarizer(
                                num_speakers=num_speakers,
                                num_threads=self.config.get("cpu_threads", 4))
                        diarizer_3d.initialize()

                        def campp_progress(pct):
                            self._emit(f"PHASE:Diarization|Đang phân tách (3D-Speaker CAM++)|{int(pct)}")

                        self._emit("PHASE:Diarization|Đang phân tách (3D-Speaker CAM++)|10")
                        raw_dict_segs = diarizer_3d.process(
                            audio_file=None, audio_data=audio, audio_sample_rate=16000,
                            progress_callback=campp_progress)

                        from core.speaker_diarization import Segment, SpeakerDiarizer
                        raw_segments = [Segment(s['start'], s['end'], s['speaker'])
                                        for s in raw_dict_segs]
                        speaker_segments_raw = [
                            {"speaker": f"Người nói {s['speaker']+1}", "speaker_id": s['speaker'],
                             "start": s['start'], "end": s['end'],
                             "duration": s['end']-s['start']}
                            for s in raw_dict_segs
                        ]

                        merger = SpeakerDiarizer()
                        raw_segments = merger._post_process_diarization_segments(raw_segments)
                        speaker_segments_raw = [
                            {"speaker": f"Người nói {seg.speaker+1}", "speaker_id": seg.speaker,
                             "start": seg.start, "end": seg.end, "duration": seg.duration}
                            for seg in raw_segments
                        ]

                        self._emit("PHASE:Diarization|Đang ghép nối với văn bản|85")
                        one_seg = [{"text": full_text, "start": all_words[0]["start"],
                                    "end": all_words[-1]["end"], "raw_words": list(all_words)}]
                        diar_results = merger.process_with_transcription(
                            self.file_path, one_seg, speaker_segments=raw_segments)

                    else:
                        # --- Pyannote Community-1 pipeline (ResNet34) ---
                        logger.info("[DEBUG] Getting cached diarizer")
                        diarizer = _get_cached_diarizer(
                            embedding_model_id=speaker_model_id,
                            num_clusters=num_speakers,
                            num_threads=self.config.get("cpu_threads", 4),
                            threshold=self.config.get("diarization_threshold", 0.6),
                            auth_token=hf_token,
                        )

                        if self._is_cancelled():
                            raise InterruptedError("Cancelled by user")

                        self._emit("PHASE:Diarization|Đang phân tách Người nói|10")

                        _last_progress = [0]

                        def diarization_progress_callback(num_processed, num_total):
                            if num_total == 0:
                                return 0
                            progress = int(num_processed / num_total * 100)
                            if progress >= _last_progress[0] + 5 or num_processed == num_total:
                                _last_progress[0] = progress
                                phase_progress = 10 + int(progress * 0.75)
                                self._emit(f"PHASE:Diarization|Đang phân tách Người nói|{phase_progress}")
                                time.sleep(0.001)
                            return 1 if self._is_cancelled() else 0

                        print("[Transcriber] Starting speaker diarization...")
                        raw_segments = diarizer.process(
                            self.file_path,
                            progress_callback=diarization_progress_callback,
                            audio_data=audio,
                            audio_sample_rate=16000,
                            asr_words=all_words if all_words else None,
                        )
                        print(f"[Transcriber] Speaker diarization done: {len(raw_segments)} segments")

                        if self._is_cancelled():
                            raise InterruptedError("Cancelled by user")

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

                        self._emit("PHASE:Diarization|Đang ghép nối với văn bản|85")

                        # Tạo 1 segment chứa TẤT CẢ raw_words → Fix 1,2,3 chạy trên toàn bộ
                        one_seg = [{
                            "text": full_text,
                            "start": all_words[0]["start"],
                            "end": all_words[-1]["end"],
                            "raw_words": list(all_words),
                        }]

                        diar_results = diarizer.process_with_transcription(
                            self.file_path,
                            one_seg,
                            speaker_segments=raw_segments
                        )

                    if diar_results:
                        _diar_speaker_groups = []
                        for dseg in diar_results:
                            _diar_speaker_groups.append({
                                "speaker": dseg.get("speaker", "Người nói 1"),
                                "speaker_id": dseg.get("speaker_id", 0),
                                "raw_words": dseg.get("raw_words", []),
                                "start": dseg.get("start", 0),
                                "end": dseg.get("end", 0),
                            })
                        print(f"[Transcriber] Speaker diarization: {len(_diar_speaker_groups)} speaker turns")

                    self._emit("PHASE:Diarization|Hoàn tất phân tách|100")

                    if diarization_start:
                        timing_details["diarization"] = time.time() - diarization_start

                    # ═══════════════════════════════════════════════════════
                    # Overlap speaker separation (opt-in via config flag)
                    # Chạy Conv-TasNet trên các vùng overlap do pyannote detect,
                    # match streams ↔ speakers bằng CAM++ embedding, ASR từng stream
                    # với context ghép thêm → 2 parallel segments cho mỗi region.
                    # ═══════════════════════════════════════════════════════
                    self._overlap_segments = []
                    if self.config.get("overlap_separation", False) and raw_segments:
                        try:
                            # Lấy overlap_regions từ diarizer vừa chạy (Senko hoặc pure_ort)
                            _active_diar = locals().get('diarizer_3d') or locals().get('diarizer')
                            _overlap_regions = []
                            if _active_diar is not None and hasattr(_active_diar, 'overlap_regions'):
                                _overlap_regions = list(_active_diar.overlap_regions)

                            if _overlap_regions:
                                self._emit(f"PHASE:OverlapSep|Đang tách giọng overlap ({len(_overlap_regions)} vùng)|0")
                                overlap_start_t = time.time()
                                from core.overlap_separator import OverlapSeparator
                                sep = OverlapSeparator(num_threads=self.config.get("cpu_threads", 4))

                                def _ov_progress(pct):
                                    self._emit(f"PHASE:OverlapSep|Đang tách giọng overlap|{int(pct)}")

                                _ov_results = sep.process(audio, raw_segments,
                                                          _overlap_regions,
                                                          progress_callback=_ov_progress)

                                # ASR từng per-speaker audio + filter words vào window
                                ov_segments = []
                                for ri, reg in enumerate(_ov_results):
                                    region_start = reg['start']; region_end = reg['end']
                                    self._emit(f"PHASE:OverlapSep|ASR overlap {ri+1}/{len(_ov_results)}|"
                                               f"{int(50 + (ri+1)/max(1,len(_ov_results))*40)}")
                                    for spk, spk_audio in reg['audio_per_speaker'].items():
                                        real_s = reg['real_start_per_speaker'][spk]
                                        real_e = reg['real_end_per_speaker'][spk]
                                        try:
                                            words_concat = decode_chunk(recognizer, spk_audio.astype(np.float32), time_offset=0.0)
                                        except Exception as _asr_err:
                                            logger.warning(f"[OverlapSep] ASR fail region {ri} spk {spk}: {_asr_err}")
                                            continue
                                        # shift global: real time = concat_time + (region_start - real_s)
                                        shift = region_start - real_s
                                        kept_words = []
                                        for w in words_concat:
                                            ws = float(w.get('start', 0)); we = float(w.get('end', ws))
                                            mid = (ws + we) / 2.0
                                            if real_s <= mid <= real_e:
                                                nw = dict(w)
                                                nw['start'] = ws + shift
                                                nw['end'] = we + shift
                                                kept_words.append(nw)
                                        if not kept_words:
                                            continue
                                        text = ' '.join(
                                            (w.get('word') or w.get('text') or '').strip()
                                            for w in kept_words
                                            if (w.get('word') or w.get('text'))).strip()
                                        if not text:
                                            continue
                                        ov_segments.append({
                                            'speaker': f"Người nói {spk + 1}",
                                            'speaker_id': int(spk),
                                            'start': region_start,
                                            'end': region_end,
                                            'text': text,
                                            'raw_words': kept_words,
                                            'overlap': True,
                                        })
                                self._overlap_segments = ov_segments
                                timing_details["overlap_separation"] = time.time() - overlap_start_t
                                self._emit(f"PHASE:OverlapSep|Tách giọng overlap xong ({len(ov_segments)} dòng)|100")
                                logger.info(f"[OverlapSep] {len(_overlap_regions)} regions → {len(ov_segments)} parallel segments ({timing_details['overlap_separation']:.1f}s)")
                            else:
                                logger.info("[OverlapSep] No overlap regions detected, skipping.")
                        except Exception as _ov_err:
                            import traceback
                            logger.error(f"[OverlapSep] Overlap separation FAILED: {_ov_err}")
                            logger.error(traceback.format_exc())
                            self._overlap_segments = []

                    if self.config.get("save_ram", False):
                        clear_model_cache("diarizer")

                except InterruptedError:
                    raise
                except Exception as e:
                    import traceback
                    logger.error(f"Speaker diarization FAILED: {e}")
                    logger.error(traceback.format_exc())
                    self._emit(f"PHASE:Diarization|Lỗi phân tách người nói: {str(e)[:80]}|0")

        # Giải phóng audio — diarization xong, không cần nữa
        del audio
        gc.collect()

        # ── Diarization-first: punctuation trên full_text + speaker boundary hints ──
        # Chạy punctuation 1 lần trên toàn bộ text (full context),
        # inject pause_hints=1.0 tại ranh giới chuyển speaker (nudge thêm dấu chấm).
        # Sau đó sentence split + map speaker labels từ diarization.
        if _diar_speaker_groups is not None and _diar_speaker_groups and restore_punctuation:
            punct_start = time.time()
            try:
                self._emit("PHASE:Punctuation|Đang thêm dấu câu|0")

                punct_confidence = self.config.get("punctuation_confidence", 0.3)
                case_confidence = self.config.get("case_confidence", -1.0)
                bypass = self.config.get("bypass_restorer", False)

                # Build pause_hints từ word timestamps + inject tại speaker boundaries
                # pause_hints[i] = gap (giây) sau word i
                # Tại word cuối mỗi speaker turn: đảm bảo >= 1.0 (nudge dấu chấm)
                pause_hints = None
                if all_words and len(all_words) >= 2:
                    pause_hints = []
                    # Tập hợp word indices cuối mỗi speaker turn
                    speaker_boundary_times = set()
                    for turn in _diar_speaker_groups:
                        rw = turn.get("raw_words", [])
                        if rw:
                            speaker_boundary_times.add(rw[-1].get("end", 0))

                    for i in range(len(all_words)):
                        if i < len(all_words) - 1:
                            gap = all_words[i + 1].get('start', 0) - all_words[i].get('end', 0)
                            gap = max(0.0, gap)
                        else:
                            gap = 1.0  # word cuối → kết thúc

                        # Inject tại speaker boundary: đảm bảo gap >= 1.0
                        w_end = all_words[i].get("end", 0)
                        if w_end in speaker_boundary_times:
                            gap = max(gap, 1.0)

                        pause_hints.append(gap)

                # Chạy punctuation 1 lần trên full_text (full context, GecBERT tự chunk)
                if not bypass and full_text.strip():
                    restorer = _get_cached_restorer("cpu", punct_confidence, case_confidence,
                                                        prefer_int8=self.config.get("save_ram", False))

                    # GecBERT runs 3 iterations, each calls progress 0-100
                    # Wrap to show overall progress across all iterations
                    _punct_iter = [0]  # mutable counter
                    _punct_n_iters = 3  # GecBERT default iterations

                    def punct_progress_cb(current, total):
                        if self._is_cancelled():
                            raise Exception("Cancelled by user")
                        # Detect iteration boundary: current==total → iteration done
                        iter_pct = int((current / max(1, total)) * 100)
                        if current >= total:
                            _punct_iter[0] += 1
                        overall = int((_punct_iter[0] * 100 + iter_pct) / _punct_n_iters)
                        overall = min(overall, 99)  # cap at 99, 100 sent after restore
                        self._emit(f"PHASE:Punctuation|Đang thêm dấu câu (bước {_punct_iter[0]+1}/{_punct_n_iters})|{overall}")

                    full_text = restorer.restore(
                        full_text, progress_callback=punct_progress_cb,
                        pause_hints=pause_hints)

                self._emit("PHASE:Align|Đang căn chỉnh thời gian|0")

                # Sentence split trên full_text đã punct
                sentences = re.split(r'(?<=[.?!])\s+', full_text)

                # Build word-index → speaker mapping từ diarization
                # Mỗi word trong all_words được gán speaker_id từ _diar_speaker_groups
                word_speaker = [0] * len(all_words)
                word_speaker_name = ["Người nói 1"] * len(all_words)
                global_idx = 0
                for turn in _diar_speaker_groups:
                    rw = turn.get("raw_words", [])
                    spk_id = turn.get("speaker_id", 0)
                    spk_name = turn.get("speaker", "Người nói 1")
                    for _ in rw:
                        if global_idx < len(all_words):
                            word_speaker[global_idx] = spk_id
                            word_speaker_name[global_idx] = spk_name
                        global_idx += 1

                # Alignment: map sentences → all_words (giống flow gốc)
                current_word_idx = 0
                total_sentences = len(sentences)
                last_align_progress = 0

                def normalize_word(word):
                    word = word.lower().strip()
                    word = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE)
                    word = word.replace(' ', '')
                    return word

                for sent_idx, sent in enumerate(sentences):
                    if not sent.strip():
                        continue

                    sent_words = [w for w in sent.split() if w.strip()]
                    if not sent_words:
                        continue

                    sent_words_clean = [normalize_word(w) for w in sent_words]
                    sent_words_clean = [w for w in sent_words_clean if w]

                    # Tìm match trong all_words
                    match_len = len(sent_words_clean)
                    best_start = current_word_idx
                    # Simple forward match
                    if best_start < len(all_words):
                        first_target = sent_words_clean[0] if sent_words_clean else ""
                        for si in range(current_word_idx, min(current_word_idx + 50, len(all_words))):
                            if normalize_word(all_words[si].get("text", "")) == first_target:
                                best_start = si
                                break

                    end_idx = min(best_start + match_len, len(all_words))
                    if end_idx <= best_start:
                        end_idx = min(best_start + 1, len(all_words))

                    seg_words = all_words[best_start:end_idx]
                    if seg_words:
                        # Split sentence tại speaker boundaries nếu sentence span nhiều speakers
                        # Group consecutive words by speaker
                        sub_groups = []
                        cur_spk = word_speaker[best_start] if best_start < len(word_speaker) else 0
                        cur_start = 0  # index trong sent_words
                        for wi_off in range(end_idx - best_start):
                            wi = best_start + wi_off
                            w_spk = word_speaker[wi] if wi < len(word_speaker) else cur_spk
                            if w_spk != cur_spk:
                                sub_groups.append((cur_spk, cur_start, wi_off))
                                cur_spk = w_spk
                                cur_start = wi_off
                        sub_groups.append((cur_spk, cur_start, end_idx - best_start))

                        if len(sub_groups) == 1:
                            # Chỉ 1 speaker → gán nguyên sentence
                            spk_id = sub_groups[0][0]
                            spk_name = word_speaker_name[best_start] if best_start < len(word_speaker_name) else "Người nói 1"
                            final_segments.append({
                                "text": sent,
                                "start": seg_words[0].get("start", 0),
                                "end": seg_words[-1].get("end", 0),
                                "speaker": spk_name,
                                "speaker_id": spk_id,
                                "raw_words": seg_words,
                            })
                        else:
                            # Nhiều speakers → split text theo tỷ lệ words
                            for spk_id, grp_start, grp_end in sub_groups:
                                grp_words = seg_words[grp_start:grp_end]
                                if not grp_words:
                                    continue
                                # Tách text theo tỷ lệ
                                total_w = len(seg_words)
                                t_start = int(grp_start / total_w * len(sent_words))
                                t_end = int(grp_end / total_w * len(sent_words))
                                if grp_end == total_w:
                                    t_end = len(sent_words)  # đảm bảo lấy hết
                                grp_text = " ".join(sent_words[t_start:t_end])
                                if not grp_text.strip():
                                    continue
                                spk_name = word_speaker_name[best_start + grp_start] if (best_start + grp_start) < len(word_speaker_name) else "Người nói 1"
                                final_segments.append({
                                    "text": grp_text,
                                    "start": grp_words[0].get("start", 0),
                                    "end": grp_words[-1].get("end", 0),
                                    "speaker": spk_name,
                                    "speaker_id": spk_id,
                                    "raw_words": grp_words,
                                })

                    current_word_idx = end_idx

                    progress = int((sent_idx + 1) / total_sentences * 100)
                    if progress >= last_align_progress + 10:
                        self._emit(f"PHASE:Align|Đang căn chỉnh thời gian|{progress}")
                        last_align_progress = progress
                        time.sleep(0)  # Yield GIL → UI thread xử lý events

                restore_duration = time.time() - punct_start
                timing_details["punctuation"] = restore_duration

                # Fix overlapping timestamps
                if final_segments:
                    for i in range(len(final_segments) - 1):
                        next_start = final_segments[i+1]['start']
                        if final_segments[i]['end'] > next_start:
                            final_segments[i]['end'] = next_start

                # Split long segments
                if final_segments:
                    original_count = len(final_segments)
                    final_segments = split_long_segments(final_segments, max_duration=12.0, preserve_raw_words=True)

                print(f"[Transcriber] Diar-first pipeline: {len(final_segments)} segments, "
                      f"{len(set(s.get('speaker_id',0) for s in final_segments))} speakers")

            except Exception as e:
                if str(e) == "Cancelled by user":
                    return None
                print(f"[Transcriber] Diar-first punctuation failed: {e}")
                import traceback
                traceback.print_exc()
                _diar_speaker_groups = None  # fallback to original flow

        # Punctuation restoration (original flow — khi không có diarization)
        if not final_segments and restore_punctuation and full_text:
            punct_start = time.time()
            try:
                if self.config.get("bypass_restorer", False):
                    self._emit("PHASE:Punctuation|Bỏ qua model (Mức độ Rất Ít)|100")
                else:
                    self._emit("PHASE:Punctuation|Đang thêm dấu câu (Sliding Window)|0")

                    punct_confidence = self.config.get("punctuation_confidence", 0.3)
                    case_confidence = self.config.get("case_confidence", -1.0)
                    logger.info(f"[DEBUG] Getting PunctuationRestorer (confidence={punct_confidence:.3f})")
                    restorer = _get_cached_restorer("cpu", punct_confidence, case_confidence,
                                                        prefer_int8=self.config.get("save_ram", False))
                    logger.info("[DEBUG] PunctuationRestorer ready")

                    _punct_iter2 = [0]
                    _punct_n_iters2 = 3

                    def punct_progress_cb(current, total):
                        if self._is_cancelled():
                            raise Exception("Cancelled by user")
                        iter_pct = int((current / max(1, total)) * 100)
                        if current >= total:
                            _punct_iter2[0] += 1
                        overall = int((_punct_iter2[0] * 100 + iter_pct) / _punct_n_iters2)
                        overall = min(overall, 99)
                        self._emit(f"PHASE:Punctuation|Đang thêm dấu câu (bước {_punct_iter2[0]+1}/{_punct_n_iters2})|{overall}")

                    # Tính pause_hints từ word timestamps để gợi ý model thêm dấu
                    pause_hints = None
                    if all_words and len(all_words) >= 2:
                        pause_hints = []
                        for i in range(len(all_words)):
                            if i < len(all_words) - 1:
                                gap = all_words[i + 1].get('start', 0) - all_words[i].get('end', 0)
                                pause_hints.append(max(0.0, gap))
                            else:
                                pause_hints.append(1.0)  # Từ cuối → gợi ý kết thúc câu
                        # Đảm bảo pause_hints khớp số từ trong full_text
                        num_words = len(full_text.split())
                        if len(pause_hints) != num_words:
                            logger.warning(f"[Pause hints] Mismatch: {len(pause_hints)} hints vs {num_words} words, disabling")
                            pause_hints = None

                    if pause_hints:
                        long_pauses = sum(1 for g in pause_hints if g > 0.5)
                        mid_pauses = sum(1 for g in pause_hints if 0.2 < g <= 0.5)
                        print(f"[Pause hints] {len(pause_hints)} words, {long_pauses} long pauses (>0.5s -> period), {mid_pauses} mid pauses (0.2-0.5s -> comma)")
                    else:
                        print("[Pause hints] No pause data")

                    logger.info(f"[DEBUG] Punctuation input ({len(full_text)} chars): {full_text[:150]}...")
                    restored_text_raw = restorer.restore(full_text, progress_callback=punct_progress_cb, pause_hints=pause_hints)
                    logger.info(f"[DEBUG] Punctuation output ({len(restored_text_raw)} chars): {restored_text_raw[:150]}...")
                    has_punct = any(c in restored_text_raw for c in '.,!?')
                    logger.info(f"[DEBUG] Punctuation result: has_punctuation={has_punct}, changed={full_text != restored_text_raw}")
                    full_text = restored_text_raw

                    if self.config.get("save_ram", False):
                        clear_model_cache("restorer")
                    logger.info("[DEBUG] Punctuation restore complete")

                timing_details["punctuation"] = time.time() - punct_start

                logger.info("[DEBUG] Starting alignment phase")
                self._emit("PHASE:Align|Đang căn chỉnh thời gian|0")

                sentences = re.split(r'(?<=[.?!])\s+', full_text)

                # Alignment
                align_start = time.time()
                current_word_idx = 0
                total_sentences = len(sentences)
                last_align_progress = 0

                def normalize_word(word):
                    word = word.lower().strip()
                    word = re.sub(r'[^\w\s]', '', word, flags=re.UNICODE)
                    word = word.replace(' ', '')
                    return word

                def find_word_sequence_match(asr_words, target_words, start_idx, max_look_ahead=50):
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

                        if asr_word == first_target or (len(asr_word) > 2 and len(first_target) > 2 and
                                                        (asr_word in first_target or first_target in asr_word)):
                            matched_count = 1
                            last_matched_idx = i
                            asr_offset = 0

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
                                    if asr_idx + 1 < len(asr_words):
                                        asr_next = normalize_word(asr_words[asr_idx + 1]['text'])
                                        if asr_next == target_word or (len(asr_next) > 2 and len(target_word) > 2 and
                                                                       (asr_next in target_word or target_word in asr_next)):
                                            matched_count += 1
                                            last_matched_idx = asr_idx + 1
                                            asr_offset += 1
                                            continue
                                    break

                            score = matched_count / len(target_words)
                            if score > best_score:
                                best_score = score
                                best_match = (i, last_matched_idx)

                            if score >= 0.95:
                                break

                    if best_score >= 0.7:
                        return best_match
                    return None, None

                for sent_idx, sent in enumerate(sentences):
                    if not sent.strip():
                        progress = int((sent_idx + 1) / total_sentences * 100)
                        if progress >= last_align_progress + 10:
                            self._emit(f"PHASE:Align|Đang căn chỉnh thời gian|{progress}")
                            last_align_progress = progress
                        continue

                    sent_words = [w for w in sent.split() if w.strip()]
                    if not sent_words:
                        progress = int((sent_idx + 1) / total_sentences * 100)
                        if progress >= last_align_progress + 10:
                            self._emit(f"PHASE:Align|Đang căn chỉnh thời gian|{progress}")
                            last_align_progress = progress
                        continue

                    sent_words_clean = [normalize_word(w) for w in sent_words]
                    sent_words_clean = [w for w in sent_words_clean if w]

                    start_t = -1
                    end_t = -1

                    match_start, match_end = find_word_sequence_match(
                        all_words, sent_words_clean, current_word_idx
                    )

                    if match_start is not None:
                        start_t = all_words[match_start]['start']
                        end_t = all_words[match_end]['end']
                        seg_words = all_words[match_start:match_end + 1]
                        current_word_idx = match_end + 1
                    else:
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
                            end_idx = min(temp_idx + len(sent_words_clean) - 1, len(all_words) - 1)
                            start_t = all_words[temp_idx]['start']
                            end_t = all_words[end_idx]['end']
                            seg_words = all_words[temp_idx:end_idx + 1]
                            current_word_idx = end_idx + 1
                        else:
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

                    progress = int((sent_idx + 1) / total_sentences * 100)
                    if progress >= last_align_progress + 10:
                        self._emit(f"PHASE:Align|Đang căn chỉnh thời gian|{progress}")
                        last_align_progress = progress
                        time.sleep(0)  # Yield GIL → UI thread xử lý events

                timing_details["alignment"] = time.time() - align_start

                # Fix overlapping timestamps
                if final_segments:
                    for i in range(len(final_segments) - 1):
                        next_start = final_segments[i+1]['start']
                        if final_segments[i]['end'] > next_start:
                            final_segments[i]['end'] = next_start
                        if 'raw_words' in final_segments[i]:
                            for w in final_segments[i]['raw_words']:
                                if w['end'] > next_start:
                                    w['end'] = next_start
                                if w['start'] > next_start:
                                    w['start'] = next_start

                # Split long segments
                if final_segments:
                    original_count = len(final_segments)
                    final_segments = split_long_segments(final_segments, max_duration=12.0, preserve_raw_words=True)
                    new_count = len(final_segments)
                    if new_count > original_count:
                        logger.info(f"Đã chia nhỏ {original_count} segment thành {new_count} segment (các segment >12s)")

                logger.info(f"[DEBUG] Alignment done: {len(final_segments)} segments")

            except Exception as e:
                # Không unload restorer ở đây - cache quản lý lifecycle
                if str(e) == "Cancelled by user":
                    return None
                self._emit(f"Lỗi khi thêm dấu câu: {e}")
                import traceback
                traceback.print_exc()
                restore_end_time = time.time()
                restore_duration = restore_end_time - transcription_end_time
                timing_details["punctuation"] = restore_duration

                # CRITICAL: Khi punctuation lỗi, final_segments vẫn rỗng.
                # Fallback: tách segment theo khoảng lặng (giống no-punctuation mode)
                if not final_segments and all_words:
                    logger.warning("[FALLBACK] Punctuation failed, falling back to pause-based segmentation")
                    self._emit("PHASE:Align|Fallback: tách theo khoảng lặng|0")
                    current_seg_words = []
                    current_start = -1
                    seg_start_idx = 0
                    for i, w in enumerate(all_words):
                        if current_start == -1:
                            current_start = w['start']
                            seg_start_idx = i
                        current_seg_words.append(w['text'])
                        is_pause = False
                        if i < len(all_words) - 1:
                            if all_words[i+1]['start'] - w['end'] > 0.8:
                                is_pause = True
                        if is_pause or len(current_seg_words) > 15:
                            final_segments.append({
                                "text": " ".join(current_seg_words).strip(),
                                "start": current_start,
                                "end": w['end'],
                                "raw_words": all_words[seg_start_idx:i+1]
                            })
                            current_seg_words = []
                            current_start = -1
                    if current_seg_words:
                        final_segments.append({
                            "text": " ".join(current_seg_words).strip(),
                            "start": current_start,
                            "end": all_words[-1]['end'],
                            "raw_words": all_words[seg_start_idx:]
                        })
                    self._emit("PHASE:Align|Fallback hoàn tất|100")
                    logger.info(f"[FALLBACK] Created {len(final_segments)} segments from pause-based segmentation")
        elif not final_segments:
            # No punctuation - segment by pauses
            self._emit("PHASE:Align|Đang căn chỉnh thời gian|0")

            current_seg_words = []
            current_start = -1
            seg_start_idx = 0

            for i, w in enumerate(all_words):
                if current_start == -1:
                    current_start = w['start']
                    seg_start_idx = i
                current_seg_words.append(w['text'])

                is_pause = False
                if i < len(all_words) - 1:
                    if all_words[i+1]['start'] - w['end'] > 0.8:
                        is_pause = True

                if is_pause or len(current_seg_words) > 15:
                    final_segments.append({
                        "text": " ".join(current_seg_words).strip(),
                        "start": current_start,
                        "end": w['end'],
                        "raw_words": all_words[seg_start_idx:i+1]
                    })
                    current_seg_words = []
                    current_start = -1

            if current_seg_words:
                final_segments.append({
                    "text": " ".join(current_seg_words).strip(),
                    "start": current_start,
                    "end": all_words[-1]['end'],
                    "raw_words": all_words[seg_start_idx:]
                })

            self._emit("PHASE:Align|Đang căn chỉnh thời gian|100")

        total_duration = time.time() - start_time
        self._emit("PHASE:Complete|Hoàn tất|100")

        transcribe_duration = transcription_end_time - start_time

        timing_info = {
            "transcription": transcribe_duration,
            "restoration": restore_duration,
            "total": total_duration,
            "upload_convert": timing_details["upload_convert"],
            "transcription_detail": timing_details["transcription"],
            "sentence_segmentation": timing_details["sentence_segmentation"],
            "punctuation": timing_details["punctuation"],
            "alignment": timing_details["alignment"],
            "diarization": timing_details["diarization"],
        }

        duration_sec = total_samples / 16000.0 if total_samples > 0 else 0.0

        # ASR confidence trung bình từ pipeline (tận dụng prob đã tính, không cần chạy lại)
        word_probs = [w.get("prob", 1.0) for w in all_words if w.get("prob") is not None]
        asr_confidence = float(np.mean(word_probs)) if word_probs else None

        result_data = {
            "text": full_text,
            "segments": final_segments,
            "timing": timing_info,
            "paragraphs": paragraphs,
            "has_speaker_diarization": len(speaker_segments) > 0 or (_diar_speaker_groups is not None and len(_diar_speaker_groups) > 0),
            "speaker_segments_raw": speaker_segments_raw,
            "duration_sec": duration_sec,
            "speaker_names": {},
            "asr_confidence": asr_confidence,
            "quality_info": _dnsmos_result,
            # Overlap separation results (parallel segments for vùng 2-speaker overlap).
            # Empty list nếu feature không bật hoặc không có overlap. Additive —
            # downstream cũ bỏ qua field này không ảnh hưởng.
            "overlap_segments": getattr(self, "_overlap_segments", []) or [],
        }

        logger.info(f"TRANSCRIPTION COMPLETED: {len(full_text)} chars, "
                    f"{len(final_segments)} segments, "
                    f"speakers={len(speaker_segments) > 0}, "
                    f"time={total_duration:.2f}s")

        return result_data
