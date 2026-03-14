# core/config.py - Cấu hình, hằng số, model registry, hotwords
# KHÔNG import PyQt6 - pure Python

import os
import sys
import json
import multiprocessing

import psutil

# === Debug Logging Configuration ===
DEBUG_LOGGING = False

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "config.ini")

# === Color Scheme (framework-independent) ===
COLORS = {
    'bg_dark': '#2b2b2b',
    'bg_card': '#3a3a3a',
    'bg_input': '#f5f5f5',
    'text_primary': '#ffffff',
    'text_secondary': '#cccccc',
    'text_dark': '#222222',
    'accent': '#007bff',
    'accent_hover': '#0056b3',
    'border': '#555555',
    'border_light': '#aaaaaa',
    'highlight': '#ffd700',
    'success': '#28a745',
    'warning': '#ffc107',
    'search_match': '#00ced1',
    'search_current': '#ff4500',
}


# === CPU Detection ===
def get_allowed_cpu_count():
    """
    Detects the number of physical CPU cores (not hyperthreads).
    For CPU-bound workloads (ONNX, PyTorch, MKL), physical cores give best performance.
    """
    physical = psutil.cpu_count(logical=False)

    try:
        p = psutil.Process(os.getpid())
        affinity = p.cpu_affinity()
        if affinity and physical:
            return min(len(affinity), physical)
        if affinity:
            return len(affinity) // 2 or 1
    except Exception as e:
        print(f"[Init] Could not get CPU affinity: {e}")

    if physical:
        return physical

    try:
        logical = os.cpu_count() or multiprocessing.cpu_count()
        return max(1, logical // 2)
    except:
        return 4

ALLOWED_THREADS = get_allowed_cpu_count()


# === Model Download Information ===
MODEL_DOWNLOAD_INFO = {
    "sherpa-onnx-zipformer-vi-2025-04-20": {
        "name": "Sherpa-ONNX Zipformer Vietnamese",
        "hf_url": "https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20",
        "description": "Model ASR chính cho tiếng Việt",
        "files": ["encoder-epoch-12-avg-8.onnx", "decoder-epoch-12-avg-8.onnx", "joiner-epoch-12-avg-8.onnx"]
    },
    "zipformer-30m-rnnt-6000h": {
        "name": "Zipformer-30M-RNNT-6000h",
        "hf_url": "https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h",
        "description": "Model ASR nhẹ, nhanh",
        "files": ["encoder-epoch-20-avg-10.onnx", "decoder-epoch-20-avg-10.onnx", "joiner-epoch-20-avg-10.onnx"]
    },
    "zipformer-30m-rnnt-streaming-6000h": {
        "name": "Zipformer-30M-RNNT-Streaming-6000h",
        "hf_url": "https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h",
        "description": "Model ASR streaming cho thu âm trực tiếp (chunk 64)",
        "files": ["encoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx", "decoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx", "joiner-epoch-31-avg-11-chunk-64-left-128.fp16.onnx"]
    },
    "sat-12l-sm": {
        "name": "SAT (Segment Any Text)",
        "hf_url": "https://huggingface.co/segment-any-text/sat-12l-sm",
        "description": "Model tách câu/tách đoạn",
        "files": ["model_optimized.onnx"]
    },
    "vibert-capu": {
        "name": "ViBERT-capu",
        "hf_url": "https://huggingface.co/dragonSwing/vibert-capu",
        "description": "Model thêm dấu câu tiếng Việt",
        "files": ["pytorch_model.bin"]
    }
}

# === Speaker Embedding Models ===
# Được import lazy từ speaker_diarization.py khi cần
_SPEAKER_EMBEDDING_MODELS = None
_DIARIZATION_AVAILABLE = None

def get_speaker_embedding_models():
    """Lazy load speaker embedding models registry."""
    global _SPEAKER_EMBEDDING_MODELS, _DIARIZATION_AVAILABLE
    if _SPEAKER_EMBEDDING_MODELS is None:
        try:
            from core.speaker_diarization import SPEAKER_EMBEDDING_MODELS as _models
            _SPEAKER_EMBEDDING_MODELS = _models
            _DIARIZATION_AVAILABLE = True
        except ImportError:
            _SPEAKER_EMBEDDING_MODELS = {}
            _DIARIZATION_AVAILABLE = False
    return _SPEAKER_EMBEDDING_MODELS

def is_diarization_available():
    """Check if speaker diarization module is available."""
    global _DIARIZATION_AVAILABLE
    if _DIARIZATION_AVAILABLE is None:
        get_speaker_embedding_models()
    return _DIARIZATION_AVAILABLE


# === Hotwords / BPE Vocab Helpers ===
def ensure_bpe_vocab(model_path):
    """
    Tự động sinh file bpe.vocab từ bpe.model nếu chưa tồn tại.
    Sherpa-ONNX cần file vocab dạng text để xử lý hotwords với BPE.

    Args:
        model_path: Đường dẫn đến thư mục chứa model

    Returns:
        Đường dẫn đến file bpe.vocab hoặc chuỗi rỗng nếu không thể tạo
    """
    bpe_model = os.path.join(model_path, "bpe.model")
    bpe_vocab = os.path.join(model_path, "bpe.vocab")

    if os.path.exists(bpe_model) and not os.path.exists(bpe_vocab):
        try:
            print(f"[Hotwords] Generating bpe.vocab from {bpe_model}...")
            try:
                import sentencepiece as sp
            except ImportError:
                print("[Hotwords] ERROR: Module 'sentencepiece' not found!")
                print("[Hotwords] Please install it: pip install sentencepiece")
                print("[Hotwords] OR manually create bpe.vocab using:")
                print(f"[Hotwords]   python -c \"import sentencepiece as sp; s = sp.SentencePieceProcessor(model_file='{bpe_model}'); open('{bpe_vocab}', 'w', encoding='utf-8').write('\\n'.join([f'{{s.IdToPiece(i)}}\\t{{s.GetScore(i)}}' for i in range(s.GetPieceSize())]))\"")
                return ""

            processor = sp.SentencePieceProcessor(model_file=bpe_model)

            with open(bpe_vocab, 'w', encoding='utf-8') as f:
                for i in range(processor.GetPieceSize()):
                    piece = processor.IdToPiece(i)
                    score = processor.GetScore(i)
                    f.write(f"{piece}\t{score}\n")

            print(f"[Hotwords] Generated bpe.vocab with {processor.GetPieceSize()} entries")
            return bpe_vocab
        except Exception as e:
            print(f"[Hotwords] Error generating bpe.vocab: {e}")
            return ""

    if os.path.exists(bpe_vocab):
        return bpe_vocab
    return ""


def prepare_hotwords_file(hotwords_path, base_dir):
    """
    Chuẩn bị file hotwords cho Sherpa-ONNX.
    Đọc file hotwords.txt, bỏ qua dòng trống và comment.

    Args:
        hotwords_path: Đường dẫn đến file hotwords
        base_dir: Thư mục gốc của ứng dụng

    Returns:
        Đường dẫn đến file hotwords đã xử lý, hoặc chuỗi rỗng
    """
    if not hotwords_path:
        hotwords_path = os.path.join(base_dir, "hotwords.txt")

    if not os.path.exists(hotwords_path):
        return ""

    try:
        with open(hotwords_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                valid_lines.append(line)

        if not valid_lines:
            return ""

        # Ghi lại file đã clean
        cleaned_path = hotwords_path + ".cleaned"
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines))

        print(f"[Hotwords] Prepared {len(valid_lines)} hotwords from {hotwords_path}")
        return cleaned_path
    except Exception as e:
        print(f"[Hotwords] Error preparing hotwords: {e}")
        return ""


def get_hotwords_config(model_path, base_dir=None):
    """
    Trả về dict config hotwords cho Sherpa-ONNX recognizer.

    Args:
        model_path: Đường dẫn đến thư mục model
        base_dir: Thư mục gốc (mặc định dùng BASE_DIR)

    Returns:
        Dict config hoặc {} nếu không có hotwords
    """
    if base_dir is None:
        base_dir = BASE_DIR

    hotwords_file = prepare_hotwords_file("", base_dir)
    if not hotwords_file:
        return {}

    bpe_vocab = ensure_bpe_vocab(model_path)

    config = {
        "hotwords_file": hotwords_file,
        "hotwords_score": 1.5,
    }

    if bpe_vocab:
        config["modeling_unit"] = "bpe"
        config["bpe_vocab"] = bpe_vocab

    return config
