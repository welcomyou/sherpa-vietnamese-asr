# core/asr_engine.py - ASR pipeline: overlap handling, segment splitting, audio loading, transcription
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

logger = logging.getLogger(__name__)

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

    # Post-merge dedup: xóa n-gram lặp liên tiếp (hallucination hoặc overlap miss)
    merged_words = remove_repeated_ngrams(merged_words)

    merged_text = " ".join([w["text"] for w in merged_words])
    return merged_words, merged_text


def remove_repeated_ngrams(words, max_ngram=5, max_gap_sec=0.3):
    """
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

def setup_ffmpeg_path():
    """Thiết lập đường dẫn ffmpeg cho pydub"""
    from pydub.utils import which

    if which("ffmpeg"):
        return

    possible_paths = [
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]

    for ffmpeg_path in possible_paths:
        if os.path.exists(ffmpeg_path):
            import pydub
            pydub.AudioSegment.converter = ffmpeg_path
            pydub.AudioSegment.ffmpeg = ffmpeg_path
            pydub.AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
            return


def load_audio(file_path, sample_rate=16000, progress_callback=None):
    """
    Load và chuẩn hóa audio file thành float32 array tại sample rate mong muốn.

    Args:
        file_path: Đường dẫn file audio/video
        sample_rate: Sample rate đích (mặc định 16000)
        progress_callback: callable(message) - callback báo tiến trình

    Returns:
        numpy.ndarray - audio float32 array, mono, normalized
    """
    import librosa

    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    file_ext = os.path.splitext(file_path)[1].lower()

    needs_conversion = ['.m4a', '.ogg', '.wma', '.opus',
                        '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']

    file_to_load = file_path

    if file_ext in needs_conversion:
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=True, res_type="soxr_vhq")
        except Exception:
            emit(f"Đang chuyển đổi {file_ext} sang wav")
            try:
                from pydub import AudioSegment
                setup_ffmpeg_path()

                temp_wav = file_path + '.temp.wav'
                format_map = {'.m4a': 'm4a', '.ogg': 'ogg', '.wma': 'wma', '.opus': 'opus'}
                if file_ext in format_map:
                    audio_segment = AudioSegment.from_file(file_path, format=format_map[file_ext])
                else:
                    audio_segment = AudioSegment.from_file(file_path)

                audio_segment = audio_segment.set_channels(1)
                audio_segment.export(temp_wav, format='wav')
                file_to_load = temp_wav
                audio, sr = librosa.load(file_to_load, sr=sample_rate, mono=True, res_type="soxr_vhq")
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except ImportError:
                raise ImportError(f"Không thể đọc file {file_ext}. Vui lòng cài đặt pydub: pip install pydub")
            except Exception as e2:
                raise Exception(f"Không thể đọc file {file_ext}. Đảm bảo đã cài ffmpeg: {e2}")
    else:
        audio, sr = librosa.load(file_to_load, sr=sample_rate, mono=True, res_type="soxr_vhq")

    # Peak normalization
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




# =============================================================================
# SHERPA ONNX LAZY IMPORT
# =============================================================================

_sherpa_onnx = None

def get_sherpa_onnx():
    """Lazy import sherpa_onnx to ensure DLL paths are set up first."""
    global _sherpa_onnx
    if _sherpa_onnx is None:
        import sherpa_onnx as so
        _sherpa_onnx = so
    return _sherpa_onnx


# =============================================================================
# ROVER: DUAL-MODEL ALIGNMENT & MERGE
# =============================================================================

ROVER_MODEL_IDS = ["zipformer-30m-rnnt-6000h", "sherpa-onnx-zipformer-vi-2025-04-20"]
ROVER_MODEL_ID = "rover-voting"  # ID dùng trong UI/config


def create_recognizer(model_path, cpu_threads=4, max_active_paths=8):
    """Tạo OfflineRecognizer từ model_path."""
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
    tokens = os.path.join(model_path, "tokens.txt")

    if not all([encoder, decoder, joiner]) or not os.path.exists(tokens):
        raise FileNotFoundError(f"Thiếu file model trong: {model_path}")

    kwargs = {
        "tokens": tokens, "encoder": encoder, "decoder": decoder, "joiner": joiner,
        "num_threads": cpu_threads, "sample_rate": 16000, "feature_dim": 80,
        "decoding_method": "modified_beam_search", "max_active_paths": max_active_paths,
    }
    hotwords_config = get_hotwords_config(model_path, BASE_DIR)
    if hotwords_config:
        kwargs.update(hotwords_config)
        print(f"[Hotwords] {os.path.basename(model_path)}: score {hotwords_config.get('hotwords_score', 1.5)}")

    return get_sherpa_onnx().OfflineRecognizer.from_transducer(**kwargs)


def decode_chunk(recognizer, audio_chunk, time_offset=0.0):
    """
    Decode một audio chunk, trả về merged words list.
    Tách riêng để dùng chung cho single-model và ROVER.
    """
    s = recognizer.create_stream()
    s.accept_waveform(16000, audio_chunk)
    recognizer.decode_stream(s)

    result = s.result
    text = result.text.strip()
    if not text:
        del s
        return []

    text = text.lower()

    if not (hasattr(result, 'timestamps') and hasattr(result, 'tokens')):
        del s
        return []

    ts = result.timestamps
    toks = result.tokens
    ys_log_probs = getattr(result, 'ys_log_probs', None)

    if len(ts) >= 2:
        avg_bpe_dur = (ts[-1] - ts[0]) / (len(ts) - 1)
    else:
        avg_bpe_dur = 0.08

    chunk_words = []
    for j, (t_val, tok) in enumerate(zip(ts, toks)):
        local_start = t_val
        local_end = ts[j + 1] if j < len(ts) - 1 else local_start + avg_bpe_dur
        prob = math.exp(ys_log_probs[j]) if ys_log_probs is not None and j < len(ys_log_probs) else 1.0

        chunk_words.append({
            "text": tok.lower(), "start": local_start + time_offset,
            "end": local_end + time_offset, "local_start": local_start,
            "local_end": local_end, "prob": prob,
        })

    # Merge BPE tokens thành words
    merged = []
    current_word = None
    for tok_info in chunk_words:
        tok = tok_info["text"]
        if tok.startswith(" ") or tok.startswith("\u2581"):
            if current_word is not None:
                current_word["prob"] = sum(current_word["probs"]) / len(current_word["probs"])
                merged.append(current_word)
            current_word = {
                "text": tok.lstrip(" ").lstrip("\u2581"),
                "start": tok_info["start"], "end": tok_info["end"],
                "local_start": tok_info["local_start"], "local_end": tok_info["local_end"],
                "last_bpe_start": tok_info["start"],
                "probs": [tok_info.get("prob", 1.0)],
            }
        else:
            if current_word is not None:
                current_word["text"] += tok
                current_word["end"] = tok_info["end"]
                current_word["local_end"] = tok_info["local_end"]
                current_word["last_bpe_start"] = tok_info["start"]
                current_word["probs"].append(tok_info.get("prob", 1.0))
            else:
                current_word = {
                    "text": tok, "start": tok_info["start"], "end": tok_info["end"],
                    "local_start": tok_info["local_start"], "local_end": tok_info["local_end"],
                    "last_bpe_start": tok_info["start"],
                    "probs": [tok_info.get("prob", 1.0)],
                }
    if current_word is not None:
        current_word["prob"] = sum(current_word["probs"]) / len(current_word["probs"])
        merged.append(current_word)

    # Tính lại word.end dựa trên last_bpe_start + avg_bpe_duration
    for wi in range(len(merged)):
        w = merged[wi]
        estimated_end = w["last_bpe_start"] + avg_bpe_dur
        if wi < len(merged) - 1:
            estimated_end = min(estimated_end, merged[wi + 1]["start"])
        w["end"] = estimated_end
        w["local_end"] = estimated_end - time_offset
        del w["last_bpe_start"]

    del s
    return merged


def rover_merge_words(words_a, words_b):
    """
    ROVER: Align và merge kết quả từ 2 model dựa trên timestamp.

    Quy tắc:
    1. Cả 2 model cùng từ → giữ (đồng thuận), lấy prob cao hơn
    2. Khác từ → chọn từ có prob cao hơn
    3. Chỉ 1 model có từ, prob >= 0.7 → giữ (model kia bỏ sót)
    4. Chỉ 1 model có từ, prob < 0.5 → bỏ (hallucination)
    """
    if not words_a:
        return words_b
    if not words_b:
        return words_a

    # Align bằng timestamp: match từng từ trong A với từ gần nhất trong B
    TIME_TOLERANCE = 0.2  # seconds

    used_b = set()
    pairs = []  # (word_a, word_b, time)

    for wa in words_a:
        best_match = None
        best_dist = TIME_TOLERANCE
        for bi, wb in enumerate(words_b):
            if bi in used_b:
                continue
            dist = abs(wa["start"] - wb["start"])
            if dist < best_dist:
                best_dist = dist
                best_match = bi
        if best_match is not None:
            used_b.add(best_match)
            pairs.append((wa, words_b[best_match]))
        else:
            pairs.append((wa, None))

    # Từ trong B không match với A nào
    for bi, wb in enumerate(words_b):
        if bi not in used_b:
            pairs.append((None, wb))

    # Sort by timestamp
    def pair_time(p):
        wa, wb = p
        if wa and wb:
            return min(wa["start"], wb["start"])
        return (wa or wb)["start"]
    pairs.sort(key=pair_time)

    # Merge
    result = []
    for wa, wb in pairs:
        if wa and wb:
            na = normalize_word_for_overlap(wa["text"])
            nb = normalize_word_for_overlap(wb["text"])
            if na == nb:
                # Đồng thuận — giữ, lấy prob cao hơn
                chosen = wa if wa.get("prob", 0) >= wb.get("prob", 0) else wb
                result.append(chosen)
            else:
                # Khác nhau — chọn prob cao hơn
                chosen = wa if wa.get("prob", 0) >= wb.get("prob", 0) else wb
                if DEBUG_LOGGING:
                    loser = wb if chosen is wa else wa
                    print(f"[ROVER] '{chosen['text']}'({chosen.get('prob', 0):.3f}) "
                          f"beats '{loser['text']}'({loser.get('prob', 0):.3f}) "
                          f"at t={chosen['start']:.2f}")
                result.append(chosen)
        elif wa:
            # Chỉ model A có — cần prob >= 0.6 để giữ
            prob = wa.get("prob", 0)
            if prob >= 0.6:
                result.append(wa)
            else:
                if DEBUG_LOGGING:
                    print(f"[ROVER] Dropped A-only '{wa['text']}'({prob:.3f}) at t={wa['start']:.2f}")
        else:
            # Chỉ model B có — cần prob >= 0.6 để giữ
            prob = wb.get("prob", 0)
            if prob >= 0.6:
                result.append(wb)
            else:
                if DEBUG_LOGGING:
                    print(f"[ROVER] Dropped B-only '{wb['text']}'({prob:.3f}) at t={wb['start']:.2f}")

    return result


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

    def _emit(self, msg):
        self.progress_callback(msg)

    def _is_cancelled(self):
        return self.cancel_check()

    def run(self):
        """
        Chạy toàn bộ pipeline ASR.

        Returns:
            dict với keys: text, segments, timing, paragraphs,
                          has_speaker_diarization, speaker_segments_raw
        """
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

            recognizer = create_recognizer(primary_path, cpu_threads, max_active_paths=12)
            self._emit("PHASE:Init|Đang khởi tạo mô hình phụ (ROVER)|40")
            rover_recognizer = create_recognizer(secondary_path, cpu_threads, max_active_paths=12)
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
            self._emit("PHASE:Init|Đang khởi tạo mô hình|60")

        # Load audio
        self._emit("PHASE:LoadAudio|Đang đọc file audio|0")
        load_audio_start = time.time()

        self._emit("PHASE:LoadAudio|Đang chuẩn hóa audio|30")
        audio = load_audio(self.file_path, progress_callback=lambda msg: self._emit(f"PHASE:LoadAudio|{msg}|35"))

        timing_details["upload_convert"] = time.time() - load_audio_start
        self._emit(f"PHASE:LoadAudio|Đã tải audio ({timing_details['upload_convert']:.1f}s)|100")

        # Segment audio
        total_samples = len(audio)

        # --- VAD-based segmentation ---
        try:
            self._emit("PHASE:VAD|Đang phát hiện vùng có tiếng nói|0")
            vad_segments = get_vad_segments(audio, progress_callback=self._emit)
            self._emit(f"PHASE:VAD|Phát hiện {len(vad_segments)} đoạn nói|100")

            # Build chunk plan từ VAD segments
            chunk_plan = []  # [(actual_start, actual_end, overlap_at_start, vad_group_idx)]
            for vad_idx, (seg_start, seg_end) in enumerate(vad_segments):
                sub_chunks = chunk_long_segment(seg_start, seg_end, max_sec=30, overlap_sec=OVERLAP_SEC)
                for (c_start, c_end, c_overlap) in sub_chunks:
                    chunk_plan.append((c_start, c_end, c_overlap, vad_idx))

            total_speech_sec = sum(e - s for s, e in vad_segments) / 16000.0
            total_audio_sec = total_samples / 16000.0
            print(f"[VAD] {len(vad_segments)} VAD segments -> {len(chunk_plan)} chunks "
                  f"({total_speech_sec:.1f}s speech / {total_audio_sec:.1f}s audio)")

            # Giải phóng VAD model trước khi transcribe
            if self.config.get("save_ram", False):
                unload_vad_model()
                import gc
                gc.collect()

        except Exception as e:
            # Fallback: dùng silence-based chunking nếu VAD lỗi
            print(f"[VAD] Error: {e}, fallback to silence-based chunking")
            self._emit("PHASE:LoadAudio|Đang phân tích khoảng lặng (fallback)|60")
            silent_regions = find_silent_regions(audio)

            segment_duration = 30
            segment_samples = 16000 * segment_duration
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
                    chunk_plan.append((logical_start, logical_end, 0, 0))
                else:
                    actual_start = max(0, logical_start - OVERLAP_SAMPLES)
                    chunk_plan.append((actual_start, logical_end, logical_start - actual_start, 0))

        num_chunks = len(chunk_plan)

        # Transcribe each chunk
        chunk_results = []
        phase_label = "Đang chuyển thành văn bản (ROVER)" if is_rover else "Đang chuyển thành văn bản"
        self._emit(f"PHASE:Transcription|{phase_label}|0")

        for i, (actual_start, actual_end, overlap_at_start, vad_idx) in enumerate(chunk_plan):
            if self._is_cancelled():
                return None

            chunk_audio = audio[actual_start:actual_end]
            time_offset = actual_start / 16000.0

            # Decode với model chính
            chunk_words = decode_chunk(recognizer, chunk_audio, time_offset)

            # ROVER: decode với model phụ và merge
            if is_rover and rover_recognizer is not None:
                chunk_words_b = decode_chunk(rover_recognizer, chunk_audio, time_offset)
                chunk_words = rover_merge_words(chunk_words, chunk_words_b)
                # Dedup ngay sau ROVER merge để xóa duplicate do timestamp lệch giữa 2 model
                chunk_words = remove_repeated_ngrams(chunk_words, max_gap_sec=0.8)

            if chunk_words:
                segment_text = " ".join(w["text"] for w in chunk_words)
                chunk_results.append({
                    "text": segment_text,
                    "words": chunk_words,
                    "audio_start_abs": time_offset,
                    "audio_end_abs": actual_end / 16000.0,
                    "overlap_sec": overlap_at_start / 16000.0,
                    "vad_group": vad_idx,
                })
            else:
                chunk_results.append({
                    "text": "",
                    "words": [],
                    "audio_start_abs": time_offset,
                    "audio_end_abs": actual_end / 16000.0,
                    "overlap_sec": overlap_at_start / 16000.0,
                    "vad_group": vad_idx,
                })

            percent = int((i + 1) / num_chunks * 100)
            self._emit(f"PHASE:Transcription|{phase_label}|{percent}")

        # Merge chunks (group by vad_group, chỉ merge overlap trong cùng VAD segment)
        all_words = []
        from itertools import groupby
        for vad_idx, group_iter in groupby(chunk_results, key=lambda cr: cr.get("vad_group", 0)):
            group_chunks = list(group_iter)
            if len(group_chunks) == 1:
                # VAD segment ngắn, không chunk → lấy trực tiếp
                all_words.extend(group_chunks[0]["words"])
            else:
                # VAD segment dài bị chunk → merge overlap
                merged_w, _ = merge_chunks_with_overlap(group_chunks, OVERLAP_SEC)
                all_words.extend(merged_w)

        full_text = " ".join(w["text"] for w in all_words)

        if full_text:
            full_text = full_text.capitalize()

        print(f"[Transcriber] Merged chunks into {len(all_words)} words")

        transcription_end_time = time.time()
        timing_details["transcription"] = transcription_end_time - start_time - timing_details["upload_convert"]

        restore_duration = 0.0
        final_segments = []
        paragraphs = []

        # Punctuation restoration
        if restore_punctuation and full_text:
            punct_start = time.time()
            try:
                if self.config.get("bypass_restorer", False):
                    self._emit("PHASE:Punctuation|Bỏ qua model (Mức độ Rất Ít)|100")
                else:
                    self._emit("PHASE:Punctuation|Đang thêm dấu câu (Sliding Window)|0")

                    from core.punctuation_restorer_improved import ImprovedPunctuationRestorer

                    punct_confidence = self.config.get("punctuation_confidence", 0.3)
                    case_confidence = self.config.get("case_confidence", -1.0)
                    logger.info(f"[DEBUG] Creating ImprovedPunctuationRestorer (confidence={punct_confidence:.3f})")
                    restorer = ImprovedPunctuationRestorer(
                        device="cpu", confidence=punct_confidence, case_confidence=case_confidence
                    )
                    logger.info("[DEBUG] ImprovedPunctuationRestorer created (model+quantization done)")

                    def punct_progress_cb(current, total):
                        if self._is_cancelled():
                            raise Exception("Cancelled by user")
                        percent = int((current / max(1, total)) * 100)
                        self._emit(f"PHASE:Punctuation|Đang thêm dấu câu ({current}/{total})|{percent}")

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
                        restorer.unload()
                        import gc
                        gc.collect()
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
                if 'restorer' in locals() and restorer:
                    restorer.unload()
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
        else:
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
                        "text": "".join(current_seg_words).strip(),
                        "start": current_start,
                        "end": w['end'],
                        "raw_words": all_words[seg_start_idx:i+1]
                    })
                    current_seg_words = []
                    current_start = -1

            if current_seg_words:
                final_segments.append({
                    "text": "".join(current_seg_words).strip(),
                    "start": current_start,
                    "end": all_words[-1]['end'],
                    "raw_words": all_words[seg_start_idx:]
                })

            self._emit("PHASE:Align|Đang căn chỉnh thời gian|100")

        # Speaker diarization
        speaker_segments = []
        speaker_segments_raw = []
        diarization_start = None

        if self.config.get("speaker_diarization", False):
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
                    diarizer = SpeakerDiarizer(
                        embedding_model_id=speaker_model_id,
                        num_clusters=num_speakers,
                        num_threads=self.config.get("cpu_threads", 4),
                        threshold=self.config.get("diarization_threshold", 0.6),
                        auth_token=hf_token
                    )

                    self._emit("PHASE:Diarization|Đang tải model phân tách|5")
                    logger.info("[DEBUG] diarizer.initialize() start")
                    diarizer.initialize()
                    logger.info("[DEBUG] diarizer.initialize() done")

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
                    self._emit("PHASE:Diarization|Đang phân tách Người nói|20")
                    # Pass pre-loaded audio to avoid torchaudio hang on Windows daemon threads
                    raw_segments = diarizer.process(
                        self.file_path,
                        progress_callback=diarization_progress_callback,
                        audio_data=audio,
                        audio_sample_rate=16000,
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

                    if self._is_cancelled():
                        raise InterruptedError("Cancelled by user")

                    self._emit("PHASE:Diarization|Đang ghép nối với văn bản|85")

                    speaker_segments = diarizer.process_with_transcription(
                        self.file_path,
                        final_segments,
                        speaker_segments=raw_segments
                    )

                    if speaker_segments:
                        final_segments = speaker_segments
                        self._emit("PHASE:Diarization|Hoàn tất phân tách|100")
                        print(f"[Transcriber] Speaker diarization completed: {len(final_segments)} segments")

                    if diarization_start:
                        timing_details["diarization"] = time.time() - diarization_start

                    if self.config.get("save_ram", False):
                        diarizer.unload()
                        import gc
                        gc.collect()

                except InterruptedError:
                    raise  # Propagate cancellation to caller
                except Exception as e:
                    print(f"[Transcriber] Speaker diarization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._emit(f"PHASE:Diarization|Lỗi phân tách người nói: {str(e)[:80]}|0")

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

        result_data = {
            "text": full_text,
            "segments": final_segments,
            "timing": timing_info,
            "paragraphs": paragraphs,
            "has_speaker_diarization": len(speaker_segments) > 0,
            "speaker_segments_raw": speaker_segments_raw,
            "duration_sec": duration_sec,
            "speaker_names": {},
        }

        logger.info(f"TRANSCRIPTION COMPLETED: {len(full_text)} chars, "
                    f"{len(final_segments)} segments, "
                    f"speakers={len(speaker_segments) > 0}, "
                    f"time={total_duration:.2f}s")

        return result_data
