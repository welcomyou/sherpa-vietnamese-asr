# core/vad_utils.py - Shared VAD (Voice Activity Detection) module
# Dùng chung cho cả asr_engine.py và audio_analyzer.py

import os
import numpy as np

from core.config import BASE_DIR

# =============================================================================
# VAD SESSION MANAGEMENT (Silero VAD ONNX)
# =============================================================================

_vad_session = None


def _get_vad_session():
    """Lazy load Silero VAD ONNX session."""
    global _vad_session
    if _vad_session is None:
        import onnxruntime as ort
        vad_model_path = os.path.join(BASE_DIR, "models", "silero-vad", "silero_vad_16k_op15.onnx")
        if not os.path.exists(vad_model_path):
            vad_model_path = os.path.join(BASE_DIR, "models", "silero-vad", "silero_vad.onnx")
        if not os.path.exists(vad_model_path):
            raise FileNotFoundError(
                f"Không tìm thấy Silero VAD model trong: "
                f"{os.path.join(BASE_DIR, 'models', 'silero-vad')}"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _vad_session = ort.InferenceSession(
            vad_model_path, providers=['CPUExecutionProvider'], sess_options=opts
        )
        print(f"[VAD] Loaded Silero VAD ONNX: {os.path.basename(vad_model_path)}")
    return _vad_session


def unload_vad_model():
    """Giải phóng VAD model khỏi RAM."""
    global _vad_session
    if _vad_session is not None:
        del _vad_session
        _vad_session = None
        print("[VAD] Model unloaded")


# =============================================================================
# VAD INFERENCE
# =============================================================================

def _run_vad_inference(audio, sample_rate=16000, threshold=0.5,
                       min_silence_ms=300, min_speech_ms=250):
    """
    Chạy Silero VAD ONNX inference trên audio.

    Args:
        audio: numpy array float32, mono
        sample_rate: 16000
        threshold: ngưỡng speech probability
        min_silence_ms: khoảng lặng tối thiểu để tách segment (ms)
        min_speech_ms: speech tối thiểu để giữ segment (ms)

    Returns:
        list of (start_window, end_window) — index theo window
    """
    session = _get_vad_session()
    window_size = 512
    context_size = 64
    total_samples = len(audio)
    num_windows = total_samples // window_size

    if num_windows == 0:
        return []

    # Reset LSTM state
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr_input = np.array(sample_rate, dtype=np.int64)
    context = np.zeros(context_size, dtype=np.float32)

    # Chạy VAD trên từng window
    probabilities = []
    for i in range(num_windows):
        chunk = audio[i * window_size: (i + 1) * window_size]
        input_data = np.concatenate([context, chunk]).reshape(1, -1).astype(np.float32)
        out, state = session.run(None, {
            'input': input_data, 'state': state, 'sr': sr_input
        })
        probabilities.append(float(out[0][0]))
        context = chunk[-context_size:]

    if not probabilities:
        return []

    # Chuyển probabilities → speech segments (window index)
    min_silence_windows = int(min_silence_ms * sample_rate / 1000 / window_size)
    min_speech_windows = int(min_speech_ms * sample_rate / 1000 / window_size)

    speech_segments = []
    is_speech = False
    speech_start = 0
    silence_count = 0

    for i, prob in enumerate(probabilities):
        if prob >= threshold:
            if not is_speech:
                speech_start = i
                is_speech = True
            silence_count = 0
        else:
            if is_speech:
                silence_count += 1
                if silence_count >= min_silence_windows:
                    speech_end = i - silence_count + 1
                    if speech_end - speech_start >= min_speech_windows:
                        speech_segments.append((speech_start, speech_end))
                    is_speech = False
                    silence_count = 0

    # Segment cuối cùng
    if is_speech:
        speech_end = len(probabilities)
        if speech_end - speech_start >= min_speech_windows:
            speech_segments.append((speech_start, speech_end))

    return speech_segments


# =============================================================================
# MAIN API: get_vad_segments
# =============================================================================

def get_vad_segments(audio, sample_rate=16000,
                     threshold=0.3,
                     min_silence_ms=300,
                     min_speech_ms=250,
                     padding_ms=300,
                     merge_gap_ms=500,
                     auto_boost=True,
                     fallback_full=True,
                     progress_callback=None):
    """
    Dùng Silero VAD (ONNX) để phát hiện các đoạn có tiếng nói trong audio.

    Retry logic:
    1. Chạy với threshold mặc định (0.3) — giữ tốt đầu/đuôi câu
    2. Nếu không tìm thấy speech, retry với threshold=0.2
    3. Nếu vẫn không và auto_boost=True, boost amplitude rồi retry

    Args:
        audio: numpy array float32, mono, 16kHz
        sample_rate: 16000
        threshold: ngưỡng speech probability (default 0.3)
        min_silence_ms: khoảng lặng tối thiểu để tách segment (ms)
        min_speech_ms: speech tối thiểu để giữ segment (ms)
        padding_ms: padding thêm trước/sau mỗi segment (ms)
        merge_gap_ms: merge segments có gap nhỏ hơn giá trị này (ms)
        auto_boost: tự động boost amplitude nếu audio quá nhỏ
        fallback_full: True = trả về toàn bộ audio nếu không tìm thấy speech
                       False = trả về list rỗng
        progress_callback: callable(str) nhận thông báo tiến trình

    Returns:
        list of (start_sample, end_sample) — các đoạn speech trong audio gốc
    """
    window_size = 512
    total_samples = len(audio)

    if total_samples < window_size:
        return [(0, total_samples)] if fallback_full else []

    # --- Lần 1: threshold mặc định ---
    speech_segments = _run_vad_inference(
        audio, sample_rate, threshold, min_silence_ms, min_speech_ms
    )

    # --- Lần 2: retry với threshold thấp hơn ---
    if not speech_segments:
        print("[VAD] No speech found, retrying with threshold=0.2...")
        speech_segments = _run_vad_inference(
            audio, sample_rate, threshold=0.2,
            min_silence_ms=200, min_speech_ms=150
        )

    # --- Lần 3: boost amplitude rồi retry ---
    if not speech_segments and auto_boost:
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6 and max_amp < 0.5:
            print(f"[VAD] Audio amplitude low (max={max_amp:.4f}), boosting and retrying...")
            boosted = (audio / max_amp * 0.95).astype(np.float32)
            speech_segments = _run_vad_inference(
                boosted, sample_rate, threshold=0.2,
                min_silence_ms=200, min_speech_ms=150
            )
            if speech_segments:
                print(f"[VAD] Found {len(speech_segments)} segments after boosting")

    # --- Fallback ---
    if not speech_segments:
        if fallback_full:
            print("[VAD] Still no speech found, using entire audio as fallback")
            return [(0, total_samples)]
        else:
            return []

    # Chuyển từ window index → sample index, thêm padding
    padding_samples = int(padding_ms * sample_rate / 1000)
    result = []
    for seg_start_w, seg_end_w in speech_segments:
        start_sample = max(0, seg_start_w * window_size - padding_samples)
        end_sample = min(total_samples, seg_end_w * window_size + padding_samples)
        result.append((start_sample, end_sample))

    # Merge segments quá gần nhau
    if merge_gap_ms > 0 and len(result) > 1:
        merge_gap = int(merge_gap_ms * sample_rate / 1000)
        merged = [result[0]]
        for start, end in result[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        result = merged

    print(f"[VAD] Found {len(result)} speech segments "
          f"({sum(e - s for s, e in result) / sample_rate:.1f}s speech "
          f"/ {total_samples / sample_rate:.1f}s audio)")

    return result
