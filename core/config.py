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
# Source of Truth — mọi UI surface (Desktop, Web, Admin) phải dùng dict này.
# Web CSS vars (:root) trong static/css/style.css phải mirror chính xác.
# KHÔNG hardcode hex — luôn dùng COLORS['key'] hoặc var(--key).
#
# COLORS là dict mutable, được cập nhật in-place bởi apply_theme() trước
# khi UI khởi tạo. Mọi import `from core.config import COLORS` vẫn tham
# chiếu cùng object → đổi theme = đổi giá trị trong dict (không reassign).
COLORS_DARK = {
    'bg_dark': '#2b2b2b',
    'bg_card': '#3a3a3a',
    'bg_elevated': '#464646',
    'bg_input': '#464646',
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
    'danger': '#dc3545',
    'search_match': '#00ced1',
    'search_current': '#ff4500',
    'low_confidence': '#DFC8B0',
}

# Light theme — mirror cấu trúc Dark nhưng đảo chiều surface (card nổi lên
# trên page xám nhạt), text-primary đậm để đạt WCAG AA (≥4.5:1), semantic
# color (success/warning/danger) đậm hơn so với Dark để giữ tương phản
# khi dùng làm text trên nền trắng. Vẫn đủ saturated khi dùng làm button bg.
COLORS_LIGHT = {
    # Surfaces (card nổi trên page xám nhạt, giống Linear/Slack)
    'bg_dark': '#eef1f5',          # page bg, cool light gray
    'bg_card': '#ffffff',          # cards/panels nổi lên
    'bg_elevated': '#f5f7fa',      # input/elevated subtle
    'bg_input': '#ffffff',         # input fields trắng, dựa vào border
    # Text (AAA primary, AA secondary)
    'text_primary': '#1f2937',     # slate-800, 12.6:1 trên trắng
    'text_secondary': '#5b6776',   # slate-600, 5.9:1 trên trắng
    'text_dark': '#1f2937',        # text trên nền vàng (highlight/warning)
    # Brand
    'accent': '#0b66d8',           # blue, 5.4:1 trên trắng (text + bg ok)
    'accent_hover': '#094a9c',     # darker hover, 7:1
    # Borders
    'border': '#d1d5db',           # gray-300, đủ visible trên page xám nhạt
    'border_light': '#9ca3af',     # gray-400, separator mạnh
    # Semantic — đậm hơn Dark để text đọc được trên trắng (>=4.5:1)
    'highlight': '#fcd34d',        # amber-300 bg với text_dark = 9.4:1
    'success': '#15803d',          # green-700, text 5.7:1 / bg+white 5.0:1
    'warning': '#d97706',          # amber-600, text 4.5:1 / bg+text_dark 5.1:1
    'danger': '#b91c1c',           # red-700, text 6.7:1 / bg+white 6.4:1
    # Search highlights (giữ saturated, đảm bảo text_dark đọc được)
    'search_match': '#a5f3fc',     # cyan-200 bg + text_dark = 13:1
    'search_current': '#fb923c',   # orange-400 bg + text_dark = 6:1, nổi bật hơn match
    # Low confidence text — màu ấm/ochre nhưng vẫn AA trên trắng
    'low_confidence': '#92400e',   # amber-800, 6.5:1 trên trắng
}

# COLORS là alias trỏ đến theme đang dùng. Mặc định Dark; apply_theme()
# cập nhật in-place để các module đã import vẫn thấy giá trị mới.
COLORS = dict(COLORS_DARK)

# Theme hiện tại — chỉ đọc, set qua apply_theme()
CURRENT_THEME = 'dark'


def apply_theme(theme_name):
    """
    Áp dụng theme bằng cách cập nhật COLORS in-place.

    Phải gọi TRƯỚC khi tạo bất kỳ widget nào, vì PyQt6 stylesheet được
    đọc tại thời điểm setStyleSheet() — đổi theme runtime không cập nhật
    widget đã render.

    Args:
        theme_name: 'light' hoặc 'dark' (case-insensitive, fallback 'dark')
    """
    global CURRENT_THEME
    theme_name = (theme_name or 'dark').lower().strip()
    if theme_name not in ('light', 'dark'):
        theme_name = 'dark'
    source = COLORS_LIGHT if theme_name == 'light' else COLORS_DARK
    COLORS.clear()
    COLORS.update(source)
    CURRENT_THEME = theme_name
    return theme_name


# === CPU Detection ===
def _detect_cpu_topology():
    """
    Phát hiện CPU topology: physical cores, logical threads, VM/vCPU.

    Trả về: (physical_cores, logical_threads, is_vm)

    Quy tắc:
    - Máy thật có HT: physical < logical (vd: 6 cores, 12 threads)
    - VM/vCPU: physical == logical (mỗi vCPU = 1 core = 1 thread, không HT)
    - Nếu không detect được physical → giả sử logical // 2
    """
    logical = None
    physical = None
    is_vm = False

    try:
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
    except Exception:
        pass

    if not logical:
        try:
            logical = os.cpu_count() or multiprocessing.cpu_count()
        except:
            logical = 4

    # VM detection
    try:
        if sys.platform == 'win32':
            import subprocess
            r = subprocess.run(['wmic', 'computersystem', 'get', 'model'],
                               capture_output=True, text=True, timeout=5)
            model = r.stdout.lower()
            is_vm = any(x in model for x in ['virtual', 'vmware', 'hyper-v', 'kvm', 'qemu', 'xen'])
        elif sys.platform == 'linux':
            try:
                with open('/sys/class/dmi/id/product_name', 'r') as f:
                    model = f.read().lower()
                is_vm = any(x in model for x in ['virtual', 'vmware', 'kvm', 'qemu', 'xen'])
            except:
                pass
    except:
        pass

    # vCPU: physical == logical hoặc physical is None trên VM
    if is_vm and (physical is None or physical == logical):
        physical = logical  # mỗi vCPU = 1 core

    if physical is None:
        physical = max(1, logical // 2)  # fallback: giả sử HT 2x

    has_ht = logical > physical

    try:
        print(f"[CPU] {physical} physical cores, {logical} logical threads"
              f"{', HT' if has_ht else ''}{', VM' if is_vm else ''}")
    except (ValueError, OSError):
        pass  # stdout co the la DEVNULL khi chay headless

    return physical, logical, is_vm


PHYSICAL_CORES, LOGICAL_THREADS, IS_VM = _detect_cpu_topology()

# Slider max = physical cores (user chọn bao nhiêu core muốn dùng)
ALLOWED_THREADS = PHYSICAL_CORES

# Default = physical cores
DEFAULT_THREADS = PHYSICAL_CORES


def compute_ort_threads(cpu_threads, full_ht=False):
    """
    Tính Z = số threads thực cho ORT intra_op.

    cpu_threads: giá trị user chọn trên UI (1 → PHYSICAL_CORES)
    full_ht: True = dùng ~90% HT (cho diarization embedding — bottleneck duy nhất cần nhiều thread)
             False = dùng đúng physical cores (tối ưu cho hầu hết ONNX models)

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Benchmark 6C/12T (Intel), 10 min audio, mỗi component riêng biệt: │
    │                                                                     │
    │ ASR Encoder:    Z=6 → 0.84s (BEST)   Z=9 → 1.42s   Z=12 → 2.24s │
    │ Punctuation:    Z=6 → 5.12s (BEST)   Z=9 → 7.98s   Z=12 → 7.35s │
    │ Diar Segment:   Z=6 → 17.6s          Z=9 → 17.6s(B) Z=12 → 21.0s│
    │ Diar Embedding: Z=6 → 128s           Z=9 → 54.3s   Z=11 → 52.2s │
    │                                                       (BEST)       │
    │                                                                     │
    │ → Hầu hết models: Z = PHYSICAL_CORES là tối ưu                     │
    │ → CHỈ Embedding model: Z = PHYSICAL + 80-90% HT bonus nhanh hơn    │
    │ → Z = LOGICAL_THREADS (full) lại CHẬM hơn do over-subscription     │
    └─────────────────────────────────────────────────────────────────────┘

    Returns Z (>= 1)
    """
    if LOGICAL_THREADS <= PHYSICAL_CORES:
        # Không HT (VM, hoặc CPU không HT)
        return max(1, cpu_threads)

    if full_ht:
        # Diar embedding: ~90% logical threads (sweet spot, Z=11 trên 6C/12T)
        # full_ht chỉ tối ưu cho ResNet embedding model duy nhất
        ht_ratio = LOGICAL_THREADS / PHYSICAL_CORES
        Z = cpu_threads + int(cpu_threads * (ht_ratio - 1) * 0.85)
        return max(1, min(Z, LOGICAL_THREADS))
    else:
        # Hầu hết models: đúng physical cores là nhanh nhất
        # (ASR encoder, punctuation, segmentation đều chậm hơn khi thêm HT)
        return max(1, cpu_threads)


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
        hotwords_path = os.path.join(base_dir, "hotword.txt")

    if not os.path.exists(hotwords_path):
        return ""

    try:
        with open(hotwords_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Tách trọng số nếu có (format: "TỪ KHÓA :2.5")
                score_part = ""
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    try:
                        float(parts[1].strip())
                        line = parts[0].strip()
                        score_part = " :" + parts[1].strip()
                    except ValueError:
                        pass  # Không phải score, giữ nguyên

                # Uppercase toàn bộ (BPE vocab là uppercase)
                line = line.upper()
                valid_lines.append(line + score_part)

        if not valid_lines:
            return ""

        # Ghi file đã clean vào thư mục temp (unique filename)
        import tempfile
        tmp_fd, cleaned_path = tempfile.mkstemp(suffix='.txt', prefix='asr_hotword_')
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
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
