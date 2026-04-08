"""
Đánh giá chất lượng âm thanh cho web service.
Lightweight - không phụ thuộc PyQt6.
Tính: DNSMOS (SIG, BAK, OVRL) và ASR Confidence.

Đồng bộ logic với desktop (core/audio_analyzer.py):
- VAD segment trước khi tính DNSMOS (chỉ tính trên speech, không tính trên silence/noise)
- Stratified sampling cho file dài (3 sample ở 15%, 50%, 85%)
- Peak normalization cho audio nhỏ
- Resampling bằng soxr_vhq
- Model file discovery ưu tiên float over int8 (giống TranscriberPipeline)
"""

import os
import glob
import logging
import math
import numpy as np

logger = logging.getLogger("asr.quality")

# DNSMOS constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DNSMOS_DIR = os.path.join(BASE_DIR, "models", "dnsmos")
DNSMOS_MODEL_NAME = "sig_bak_ovr.onnx"
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
DNSMOS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
SAMPLE_RATE = 16000

# Cache ONNX sessions
_dnsmos_session = None
_asr_recognizer_cache = {}  # {model_path: recognizer}


def _load_dnsmos_session():
    """Load DNSMOS ONNX model (cached)."""
    global _dnsmos_session
    if _dnsmos_session is not None:
        return _dnsmos_session

    model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
    if not os.path.exists(model_path):
        # Auto-download
        try:
            import urllib.request, hashlib
            os.makedirs(DNSMOS_DIR, exist_ok=True)
            tmp_path = model_path + ".tmp"
            logger.info("Downloading DNSMOS model...")
            urllib.request.urlretrieve(DNSMOS_URL, tmp_path)
            sha256 = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            if sha256.hexdigest() != DNSMOS_SHA256:
                os.remove(tmp_path)
                logger.error("DNSMOS SHA-256 mismatch — file corrupted or tampered")
                return None
            os.rename(tmp_path, model_path)
            logger.info("DNSMOS model downloaded.")
        except Exception as e:
            logger.error(f"Failed to download DNSMOS: {e}")
            return None

    try:
        import onnxruntime as ort
        _dnsmos_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        return _dnsmos_session
    except Exception as e:
        logger.error(f"Failed to load DNSMOS model: {e}")
        return None


def _compute_dnsmos_single(audio: np.ndarray, session) -> dict | None:
    """Compute DNSMOS on a single 9.01s chunk."""
    try:
        target_len = 144160  # 9.01s @ 16kHz

        if len(audio) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(audio)] = audio[:target_len].astype(np.float32)
        else:
            padded = audio[:target_len].astype(np.float32)

        input_data = padded.reshape(1, -1)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})

        output_arr = outputs[0]
        scores = output_arr[0] if len(output_arr.shape) == 2 else output_arr

        # Polynomial mapping (Microsoft DNSMOS standard)
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

        return {
            "SIG": float(np.clip(p_sig(scores[0]), 1.0, 5.0)),
            "BAK": float(np.clip(p_bak(scores[1]), 1.0, 5.0)),
            "OVRL": float(np.clip(p_ovr(scores[2]), 1.0, 5.0)),
        }
    except Exception as e:
        logger.error(f"DNSMOS compute error: {e}")
        return None


def compute_dnsmos(audio: np.ndarray) -> dict | None:
    """Compute DNSMOS scores averaged over sliding windows."""
    session = _load_dnsmos_session()
    if session is None:
        return None

    target_len = 144160
    if len(audio) <= target_len:
        return _compute_dnsmos_single(audio, session)

    all_scores = []
    step = target_len // 2  # 50% overlap
    for start in range(0, len(audio) - target_len + 1, step):
        chunk = audio[start:start + target_len]
        score = _compute_dnsmos_single(chunk, session)
        if score:
            all_scores.append(score)

    if not all_scores:
        return None

    return {
        "SIG": float(np.mean([s["SIG"] for s in all_scores])),
        "BAK": float(np.mean([s["BAK"] for s in all_scores])),
        "OVRL": float(np.mean([s["OVRL"] for s in all_scores])),
    }


# =============================================================================
# MODEL FILE DISCOVERY (đồng bộ với TranscriberPipeline)
# =============================================================================

def _find_model_file(model_path: str, prefix: str):
    """Tìm model file, ưu tiên float over int8 (giống TranscriberPipeline)."""
    files = [f for f in os.listdir(model_path)
             if f.startswith(prefix) and f.endswith(".onnx")]
    float_files = [f for f in files if "int8" not in f]
    if float_files:
        return os.path.join(model_path, float_files[0])
    if files:
        return os.path.join(model_path, files[0])
    return None


def _get_asr_recognizer(model_path: str):
    """Lấy hoặc tạo cached ASR recognizer (tránh load model mỗi lần)."""
    global _asr_recognizer_cache
    if model_path in _asr_recognizer_cache:
        return _asr_recognizer_cache[model_path]

    try:
        import sherpa_onnx as so

        tokens = os.path.join(model_path, "tokens.txt")
        encoder = _find_model_file(model_path, "encoder-")
        decoder = _find_model_file(model_path, "decoder-")
        joiner = _find_model_file(model_path, "joiner-")

        if not all([encoder, decoder, joiner]) or not os.path.exists(tokens):
            logger.warning(f"Cannot find ASR model files in {model_path}")
            return None

        recognizer = so.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=2,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=8,
        )
        _asr_recognizer_cache[model_path] = recognizer
        logger.info(f"Cached ASR recognizer for quality analysis: {os.path.basename(model_path)}")
        return recognizer
    except Exception as e:
        logger.error(f"Failed to create ASR recognizer: {e}")
        return None


# =============================================================================
# STRATIFIED SAMPLING + VAD (đồng bộ với desktop audio_analyzer.py)
# =============================================================================

def _stratified_sample(audio: np.ndarray, segment_sec: int = 10):
    """
    Lấy mẫu stratified: 3 đoạn ở vị trí 15%, 50%, 85%.
    Đảm bảo đại diện đầu, giữa, cuối file.
    """
    if len(audio) < SAMPLE_RATE * 2:  # File quá ngắn < 2s
        return [audio]

    samples = []
    positions = [0.15, 0.50, 0.85]
    segment_samples = segment_sec * SAMPLE_RATE

    for pos in positions:
        center = int(len(audio) * pos)
        start = max(0, center - segment_samples // 2)
        end = min(len(audio), start + segment_samples)

        if end - start > SAMPLE_RATE:  # Ít nhất 1s
            samples.append(audio[start:end])

    return samples if samples else [audio]


def _vad_segment(audio: np.ndarray):
    """
    Dùng Silero VAD (shared từ vad_utils) để cắt các đoạn speech.
    Giống logic desktop audio_analyzer.py.
    """
    try:
        from core.vad_utils import get_vad_segments

        vad_segments = get_vad_segments(
            audio, sample_rate=SAMPLE_RATE,
            padding_ms=600,  # VAD_PAD_SEC = 0.6 giống desktop
            merge_gap_ms=500,
            auto_boost=True,
            fallback_full=False,
        )

        if not vad_segments:
            return []

        return [audio[start:end] for start, end in vad_segments]

    except Exception as e:
        logger.error(f"VAD error: {e}")
        return [audio]


def compute_asr_confidence(audio: np.ndarray, model_path: str) -> tuple[float | None, str]:
    """
    Compute ASR confidence dùng cached recognizer.
    Returns: (confidence 0-1 hoặc None, text)
    """
    recognizer = _get_asr_recognizer(model_path)
    if recognizer is None:
        return None, ""

    try:
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
        recognizer.decode_stream(stream)

        result = stream.result
        text = result.text.strip()

        if hasattr(result, 'ys_log_probs') and result.ys_log_probs:
            log_probs = result.ys_log_probs
            mean_log_prob = float(np.mean(log_probs))
            confidence = float(np.exp(mean_log_prob))
            return min(max(confidence, 0.0), 1.0), text

        return None, text
    except Exception as e:
        logger.error(f"ASR confidence error: {e}")
        return None, ""


def analyze_audio_quality(wav_path: str, model_path: str,
                          progress_callback=None,
                          asr_confidence: float = None) -> dict | None:
    """
    Analyze audio quality: DNSMOS + ASR confidence.
    - DNSMOS: stratified sampling (3 sample) + VAD + DNSMOS model
    - ASR confidence: tận dụng từ pipeline (không chạy ASR lại)

    Args:
        asr_confidence: Confidence đã tính từ pipeline (0-1). Nếu None thì bỏ qua.
    """
    try:
        import librosa
        import soundfile as sf

        if progress_callback:
            progress_callback("PHASE:Quality|Đang đánh giá chất lượng âm thanh...|92")

        # Load audio nhẹ bằng soundfile (không resample — DNSMOS chỉ cần 16kHz)
        audio, sr = sf.read(wav_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # stereo -> mono
        # Resample nếu không phải 16kHz (WAV từ ffmpeg thường đã 16kHz)
        if sr != SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, SAMPLE_RATE, quality='HQ')

        if len(audio) < SAMPLE_RATE * 0.5:
            logger.warning("Audio too short for quality analysis")
            return None

        # Stratified sampling (3 đoạn 10s ở 15%, 50%, 85%)
        samples = _stratified_sample(audio)
        del audio  # Giải phóng — chỉ cần samples

        all_dnsmos_scores = []

        for sample in samples:
            segments = _vad_segment(sample)
            if not segments:
                # Fallback: tính DNSMOS trên raw sample
                if len(sample) >= SAMPLE_RATE * 0.5:
                    dnsmos = compute_dnsmos(sample)
                    if dnsmos:
                        all_dnsmos_scores.append(dnsmos)
                continue

            for seg in segments:
                if len(seg) >= SAMPLE_RATE * 0.5:
                    dnsmos = compute_dnsmos(seg)
                    if dnsmos:
                        all_dnsmos_scores.append(dnsmos)

        del samples  # Giải phóng

        result = {}

        # DNSMOS trung bình
        if all_dnsmos_scores:
            result["dnsmos_sig"] = round(float(np.mean([s["SIG"] for s in all_dnsmos_scores])), 2)
            result["dnsmos_bak"] = round(float(np.mean([s["BAK"] for s in all_dnsmos_scores])), 2)
            result["dnsmos_ovrl"] = round(float(np.mean([s["OVRL"] for s in all_dnsmos_scores])), 2)

        # ASR Confidence — tận dụng từ pipeline, không chạy ASR lại
        if asr_confidence is not None:
            result["asr_confidence"] = round(float(asr_confidence), 4)

        if not result:
            return None

        # Add labels (giống desktop)
        if "dnsmos_ovrl" in result:
            ovrl = result["dnsmos_ovrl"]
            if ovrl >= 4.0:
                result["dnsmos_label"] = "Tốt"
            elif ovrl >= 3.0:
                result["dnsmos_label"] = "Khá"
            elif ovrl >= 2.0:
                result["dnsmos_label"] = "Trung bình"
            else:
                result["dnsmos_label"] = "Kém"

        if "asr_confidence" in result:
            conf = result["asr_confidence"]
            if conf >= 0.85:
                result["confidence_label"] = "Xuất sắc"
            elif conf >= 0.75:
                result["confidence_label"] = "Tốt"
            elif conf >= 0.60:
                result["confidence_label"] = "Trung bình"
            else:
                result["confidence_label"] = "Kém"

        result["num_segments"] = total_segments
        result["sample_text"] = sample_texts[0] if sample_texts else ""

        logger.info(f"Quality analysis: {result}")
        return result

    except Exception as e:
        logger.error(f"Quality analysis error: {e}")
        return None
