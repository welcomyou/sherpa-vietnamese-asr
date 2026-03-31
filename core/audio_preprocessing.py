"""Audio preprocessing cho far-field ASR.

Preprocessing pipeline (chạy trên full audio, trước chunking):
1. Per-segment RMS normalization (adaptive) — sửa volume mismatch
2. Adaptive peak limiter — đảm bảo output trong [-1, 1]

WPE dereverberation chạy per-chunk (≤30s) trong vòng lặp transcription
tại asr_engine.py để tránh OOM với file dài.

References:
- Per-segment normalization: Google AGC (Prabhavalkar et al., ICASSP 2015)
  https://research.google.com/pubs/archive/43289.pdf
- NARA-WPE: Drude et al., "NARA-WPE: A Python package for weighted
  prediction error dereverberation in reverberant environments"
  https://github.com/fgnt/nara_wpe
  Improved WPE (IEEE TASLP 2024):
  https://dl.acm.org/doi/10.1109/TASLP.2024.3440003
- NOTSOFAR-1 Challenge (2024): đội NAIST dùng nara-WPE + Zipformer-T
  https://arxiv.org/html/2501.17304v2
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


# =========================================================================
# 1. PER-SEGMENT RMS NORMALIZATION (adaptive, không hardcode)
# =========================================================================
# Nguyên lý: thay vì normalize toàn file theo peak (dễ bị lệch do
# tiếng động lớn), tính RMS từng speech segment, lấy median RMS
# làm target, scale từng segment về target.
#
# Tham khảo: Google Adaptive AGC (Prabhavalkar et al. 2015) —
# gain tự tính từ phân bố thống kê của signal.
# =========================================================================

def compute_segment_rms(audio_segment):
    """Tính RMS (Root Mean Square) của 1 audio segment."""
    if len(audio_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_segment ** 2)))


def per_segment_rms_normalize(audio, vad_segments, sample_rate=16000,
                               min_segment_ms=100, max_gain_db=20.0,
                               crossfade_ms=5):
    """Per-segment RMS normalization — adaptive, không hardcode target.

    Thuật toán:
    1. Tính RMS từng speech segment (từ VAD)
    2. Target RMS = median(tất cả segment RMS) — tự adaptive
    3. Scale từng segment: gain = target / segment_rms
    4. Giới hạn gain tối đa (tránh khuếch đại noise quá mức)
    5. Crossfade tại biên segment (tránh click/pop)

    Args:
        audio: numpy float32, mono, 16kHz
        vad_segments: list of (start_sample, end_sample) từ VAD
        sample_rate: 16000
        min_segment_ms: bỏ qua segment quá ngắn
        max_gain_db: giới hạn gain tối đa (dB)
        crossfade_ms: thời gian crossfade tại biên (ms)

    Returns:
        numpy float32 — audio đã normalize
    """
    if len(vad_segments) == 0:
        return audio

    min_samples = int(min_segment_ms * sample_rate / 1000)
    max_gain_linear = 10 ** (max_gain_db / 20.0)
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    # Bước 1: Tính RMS từng segment
    segment_rms_list = []
    for seg_start, seg_end in vad_segments:
        if seg_end - seg_start < min_samples:
            continue
        rms = compute_segment_rms(audio[seg_start:seg_end])
        if rms > 1e-8:  # bỏ segment gần như silent
            segment_rms_list.append((seg_start, seg_end, rms))

    if len(segment_rms_list) == 0:
        return audio

    # Bước 2: Target = median RMS (adaptive, không hardcode)
    all_rms = np.array([r for _, _, r in segment_rms_list])
    target_rms = float(np.median(all_rms))

    if target_rms < 1e-8:
        return audio

    logger.info(f"[Preprocess] RMS normalize: {len(segment_rms_list)} segments, "
                f"target_rms={target_rms:.6f} "
                f"(min={np.min(all_rms):.6f}, max={np.max(all_rms):.6f}, "
                f"ratio={np.max(all_rms)/np.min(all_rms):.1f}x)")

    # Bước 3: Tạo gain map cho toàn bộ audio
    # Khởi tạo gain = 1.0 (không thay đổi) cho vùng non-speech
    gain_map = np.ones(len(audio), dtype=np.float32)

    for seg_start, seg_end, rms in segment_rms_list:
        gain = target_rms / rms
        gain = min(gain, max_gain_linear)  # Giới hạn gain tối đa
        gain = max(gain, 1.0 / max_gain_linear)  # Giới hạn attenuation
        gain_map[seg_start:seg_end] = gain

    # Bước 4: Smooth gain tại biên (crossfade) — tránh click/pop
    if crossfade_samples > 0:
        for seg_start, seg_end, _ in segment_rms_list:
            # Fade in tại đầu segment
            fade_len = min(crossfade_samples, (seg_end - seg_start) // 4)
            if fade_len > 0 and seg_start > 0:
                prev_gain = gain_map[max(0, seg_start - 1)]
                seg_gain = gain_map[seg_start]
                fade = np.linspace(prev_gain, seg_gain, fade_len, dtype=np.float32)
                gain_map[seg_start:seg_start + fade_len] = fade

            # Fade out tại cuối segment
            if fade_len > 0 and seg_end < len(audio):
                seg_gain = gain_map[seg_end - 1]
                next_gain = gain_map[min(len(audio) - 1, seg_end)]
                fade = np.linspace(seg_gain, next_gain, fade_len, dtype=np.float32)
                gain_map[seg_end - fade_len:seg_end] = fade

    # Bước 5: Apply gain
    result = audio * gain_map

    # Log thống kê
    gains_applied = [target_rms / r for _, _, r in segment_rms_list]
    gains_db = [20 * np.log10(g) for g in gains_applied]
    boosted = sum(1 for g in gains_db if g > 1.0)
    attenuated = sum(1 for g in gains_db if g < -1.0)
    logger.info(f"[Preprocess] RMS normalize applied: "
                f"{boosted} segments boosted, {attenuated} attenuated, "
                f"gain range: {min(gains_db):+.1f}dB to {max(gains_db):+.1f}dB")

    return result


# =========================================================================
# 2. NARA-WPE DEREVERBERATION
# =========================================================================
# Weighted Prediction Error (WPE) — loại bỏ late reverberation.
# Dùng linear prediction trong frequency domain.
#
# - REVERB challenge: giảm ~22% relative WER single-channel
# - NOTSOFAR-1 2024: đội NAIST dùng WPE + Zipformer-T
# - Không tạo artifact như deep-learning SE
#
# Reference: Nakatani et al., "Speech Dereverberation Based on
# Variance-Normalized Delayed Linear Prediction"
# =========================================================================

def apply_wpe_dereverberation(audio, sample_rate=16000,
                               fft_size=512, hop_size=128,
                               taps=10, delay=3, iterations=3):
    """Áp dụng WPE dereverberation cho 1 chunk audio (≤30s).

    Hàm này được gọi per-chunk trong vòng lặp transcription,
    KHÔNG chạy trên toàn bộ file để tránh OOM.

    Parameters tuned cho 16kHz speech (theo NARA-WPE defaults + NOTSOFAR-1):
    - fft_size=512 (32ms window — chuẩn cho 16kHz speech)
    - hop_size=128 (8ms shift — 75% overlap)
    - taps=10 (số filter taps — 10 phù hợp cho phòng họp thông thường)
    - delay=3 (prediction delay — giữ early reflections, chỉ bỏ late reverb)
    - iterations=3 (số vòng lặp EM — đủ hội tụ)

    Args:
        audio: numpy float32, mono, 16kHz (chunk ≤30s)
        sample_rate: 16000
        fft_size, hop_size, taps, delay, iterations: WPE parameters

    Returns:
        numpy float32 — audio đã dereverberation
    """
    try:
        from nara_wpe.wpe import wpe_v6 as wpe
        from nara_wpe.utils import stft, istft
    except ImportError:
        logger.warning("[Preprocess] nara-wpe not installed, skipping dereverberation. "
                       "Install: pip install nara-wpe")
        return audio

    if len(audio) < fft_size * 2:
        return audio

    logger.info(f"[Preprocess] WPE dereverberation: fft={fft_size}, "
                f"taps={taps}, delay={delay}, iter={iterations}")

    # STFT → (1, F, T)
    stft_signal = stft(audio, size=fft_size, shift=hop_size)
    stft_signal = stft_signal.T[np.newaxis, ...]

    # WPE
    enhanced_stft = wpe(
        stft_signal,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode='full',
    )

    # iSTFT
    result = istft(enhanced_stft[0].T, size=fft_size, shift=hop_size)

    # Đảm bảo cùng length với input
    if len(result) > len(audio):
        result = result[:len(audio)]
    elif len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))

    return result.astype(np.float32)


# =========================================================================
# 3. ADAPTIVE PEAK LIMITER
# =========================================================================
# Đảm bảo output trong [-1, 1] sau tất cả preprocessing.
# Soft clipping thay vì hard clip — bảo toàn dynamics.
# =========================================================================

def adaptive_peak_limit(audio, target_peak=0.95):
    """Soft peak limiting — đảm bảo audio trong [-1, 1].

    Nếu peak > target_peak, scale toàn bộ về target_peak.
    Đây là linear scaling (không phải compression), không tạo artifact.

    Args:
        audio: numpy float32
        target_peak: mức peak tối đa mong muốn (default 0.95)

    Returns:
        numpy float32
    """
    peak = np.max(np.abs(audio))
    if peak > target_peak:
        audio = audio * (target_peak / peak)
        logger.info(f"[Preprocess] Peak limited: {peak:.4f} -> {target_peak}")
    return audio


# =========================================================================
# MAIN PREPROCESSING PIPELINE
# =========================================================================

def preprocess_audio(audio, vad_segments, sample_rate=16000,
                     enable_rms_normalize=True,
                     progress_callback=None):
    """Pipeline preprocessing chính (chạy trên full audio).

    Thứ tự xử lý:
    1. Per-segment RMS normalization (dùng VAD segments — cần global context)
    2. Adaptive peak limiter

    WPE dereverberation được xử lý per-chunk trong asr_engine.py.

    Args:
        audio: numpy float32, mono, 16kHz
        vad_segments: list of (start_sample, end_sample) từ VAD
        sample_rate: 16000
        enable_rms_normalize: bật/tắt RMS normalization
        progress_callback: callable(str)

    Returns:
        numpy float32 — audio đã preprocessing
    """
    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    original_rms = np.sqrt(np.mean(audio ** 2))
    result = audio.copy()

    # --- Bước 1: Per-segment RMS normalization ---
    if enable_rms_normalize and len(vad_segments) > 0:
        emit("PHASE:Preprocess|Đang chuẩn hóa âm lượng từng đoạn|50")
        result = per_segment_rms_normalize(result, vad_segments, sample_rate)

    # --- Bước 2: Adaptive peak limiter ---
    result = adaptive_peak_limit(result)

    # Log tổng hợp
    final_rms = np.sqrt(np.mean(result ** 2))
    rms_change_db = 20 * np.log10(final_rms / original_rms) if original_rms > 1e-8 else 0
    logger.info(f"[Preprocess] Done: RMS {original_rms:.6f} -> {final_rms:.6f} "
                f"({rms_change_db:+.1f} dB)")
    emit("PHASE:Preprocess|Preprocessing hoàn tất|100")

    return result
