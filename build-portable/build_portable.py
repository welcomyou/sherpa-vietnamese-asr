#!/usr/bin/env python3
"""
ASR-VN Build Script - Build portable EXE from virtualenv
Usage: python build-portable/build_portable.py

Prerequisites:
    1. python build-portable/setup_build_env.py  # Setup .venv with all dependencies
    2. python build-portable/build_portable.py   # Build portable distribution
"""
import io
import os
import sys
import subprocess
import shutil
import zipfile
import stat
from pathlib import Path

# Get project root (parent of build-portable directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration
VENV_DIR = PROJECT_ROOT / ".envtietkiem"
DIST_DIR = PROJECT_ROOT / "dist" / "sherpa-vietnamese-asr"
BUILD_DIR = PROJECT_ROOT / "build"
PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-embed-amd64.zip"

# Source files to copy (relative to project root)
SOURCE_FILES = [
    "app.py", "transcriber.py", "common.py",
    "tab_file.py", "tab_live.py",
    "quality_result_dialog.py",
    "streaming_asr.py", "streaming_asr_online.py",
    "config.ini", "hotword.txt", "verb-form-vocab.txt",
    "speaker_hotkeys.json", "ffmpeg.exe", "ffprobe.exe",
]

# Module directories to copy
MODULE_DIRS = ["core"]

# Data directories to copy
DATA_DIRS = ["models", "vocabulary"]


def get_venv_path():
    """Get site-packages path in venv"""
    if sys.platform == 'win32':
        return VENV_DIR / "Lib" / "site-packages"
    return VENV_DIR / "lib" / f"python{VENV_DIR.name}" / "site-packages"


def check_venv():
    """Check if virtual environment exists"""
    if not VENV_DIR.exists():
        print("[ERROR] Virtual environment not found!")
        print()
        print("Please run setup first:")
        print("  python build-portable/setup_build_env.py")
        return False
    return True


def download_python_embedded():
    """Download Python embedded if not exists"""
    output = PROJECT_ROOT / "python-embedded.zip"
    if output.exists():
        print(f"[OK] {output.name} already exists")
        return output
    
    print(f"[DL] Downloading Python 3.12.0 embedded...")
    print(f"     From: {PYTHON_EMBED_URL}")
    
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        urllib.request.urlretrieve(PYTHON_EMBED_URL, output)
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"[OK] Downloaded: {output.name} ({size_mb:.1f} MB)")
        return output
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        sys.exit(1)


def setup_python(zip_file):
    """Extract and configure Python embedded"""
    python_dir = DIST_DIR / "python"
    python_dir.mkdir(parents=True, exist_ok=True)
    
    print("[EX] Extracting Python...")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(python_dir)
    
    # Configure python312._pth (enable site-packages)
    pth_content = "python312.zip\n.\nLib/site-packages\nimport site\n"
    (python_dir / "python312._pth").write_bytes(pth_content.encode('utf-8'))
    print("[OK] Configured python312._pth")
    
    # Create sitecustomize.py for DLL loading
    sitecustomize = '''import sys
import os
import io
# Fix Windows console encoding (cp1252 -> utf-8) cho tieng Viet
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
python_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(python_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
site_packages = os.path.join(python_dir, 'Lib', 'site-packages')
_dll_dirs = [
    os.path.join(site_packages, 'PyQt6', 'Qt6', 'bin'),
    os.path.join(site_packages, 'numpy', '_core'),
    os.path.join(site_packages, 'onnxruntime', 'capi'),
    os.path.join(site_packages, 'sherpa_onnx', 'lib'),
    python_dir,
    site_packages,
]
if hasattr(os, 'add_dll_directory'):
    for subpath in _dll_dirs:
        if os.path.exists(subpath):
            try:
                os.add_dll_directory(subpath)
            except:
                pass
if sys.platform == 'win32':
    paths_to_add = [p for p in _dll_dirs if os.path.exists(p)]
    os.environ['PATH'] = os.pathsep.join(paths_to_add + [os.environ.get('PATH', '')])
'''
    (python_dir / "sitecustomize.py").write_text(sitecustomize, encoding='utf-8')
    print("[OK] Created sitecustomize.py")


def copy_venv_packages():
    """Copy packages from virtualenv"""
    print("[PKG] Copying packages from venv...")
    
    venv_site = get_venv_path()
    dst_site = DIST_DIR / "python" / "Lib" / "site-packages"
    dst_site.mkdir(parents=True, exist_ok=True)
    
    if not venv_site.exists():
        print(f"[ERROR] Venv site-packages not found: {venv_site}")
        return False
    
    # Packages to exclude (desktop doesn't need web service packages or unused deps)
    EXCLUDE_PACKAGES = {
        'moonshine_voice',

        # === PyTorch + ecosystem (~1,340 MB) — app uses ONNX Runtime only ===
        'torch', 'torchaudio', 'torio', 'torchmetrics', 'torchgen',
        'torchcodec', 'torch_audiomentations', 'torch_pitch_shift',
        'functorch', 'ml_dtypes',
        # torch transitive deps
        'sympy', 'networkx', 'jinja2', 'markupsafe',

        # === PyTorch Lightning (~15 MB) ===
        'lightning', 'lightning_fabric', 'lightning_utilities',
        'pytorch_lightning', 'pytorch_metric_learning',

        # === Pyannote (~12 MB) — pure ORT diarization replaces it ===
        'pyannote', 'pyannote_audio', 'pyannote_core', 'pyannote_pipeline',
        'pyannote_database', 'pyannote_metrics', 'pyannote_onnx_extended',
        'pyannoteai', 'pyannoteai_sdk',
        'asteroid_filterbanks', 'julius', 'einops',

        # === numba/llvmlite (~133 MB) — librosa only uses load/resample via soxr ===
        'numba', 'llvmlite',

        # === Large unused packages ===
        'pandas',           # 67 MB — not imported in app
        'pymupdf', 'fitz',  # 51 MB — PDF processing, not used
        'onnx',             # 45 MB — model builder, only onnxruntime needed
        'ctranslate2',      # 61 MB — not used
        'av',               # 79 MB (with .libs) — video processing

        # === Web framework (not needed for desktop) ===
        'fastapi', 'uvicorn', 'starlette', 'httptools', 'websockets',
        'python_multipart', 'multipart',
        'python_jose', 'jose', 'passlib', 'bcrypt',
        'h11', 'httpcore', 'httpx', 'anyio', 'sniffio',
        'watchfiles', 'uvloop',
        'pydantic', 'pydantic_core',  # FastAPI dep, desktop doesn't need
        'ecdsa', 'rsa', 'pyasn1',    # JWT/crypto deps

        # === Crypto / SSL (desktop doesn't use SSL or JWT) ===
        'cryptography', 'cffi', 'pycparser',

        # === Database / ORM (desktop uses no DB) ===
        'sqlalchemy', 'alembic', 'aiosqlite', 'greenlet', 'mako',

        # === gRPC (~17 MB) ===
        'grpc', 'grpcio', 'grpcio_tools',
        'google', 'googleapis_common_protos', 'protobuf',

        # === Async HTTP (not needed) ===
        'aiohttp', 'aiofiles', 'aiohappyeyeballs', 'aiosignal',
        'frozenlist', 'multidict', 'yarl', 'propcache',

        # === Monitoring / Logging / Terminal UI ===
        'opentelemetry', 'opentelemetry_api', 'opentelemetry_sdk',
        'opentelemetry_proto', 'opentelemetry_exporter_otlp',
        'opentelemetry_exporter_otlp_proto_http',
        'opentelemetry_exporter_otlp_proto_grpc',
        'opentelemetry_exporter_otlp_proto_common',
        'opentelemetry_semantic_conventions',
        'pygments', 'rich', 'colorlog', 'colorama',
        'coloredlogs', 'humanfriendly',

        # === Text processing not used ===
        'jiwer', 'langid', 'mosestokenizer', 'wtpsplit', 'textgrid',
        'markdown_it', 'mdurl',

        # === Math / Utils not used ===
        'optuna', 'primePy', 'mpmath', 'skops', 'prettytable',

        # === Audio/ML extras not used ===
        'adapters', 'rapidfuzz', 'chunkformer',
        'wespeakerruntime', 'silero_vad', 'faster_whisper',
        'diarize', 'nara_wpe', 'kaldiio',

        # === transformers + deps (~60 MB) — replaced by minimal stub ===
        'transformers',
        'huggingface_hub', 'safetensors',
        'regex', 'requests', 'urllib3', 'certifi', 'charset_normalizer', 'idna',
        'fsspec', 'tqdm',

        # === Misc not needed at runtime ===
        'bottleneck',       # optional pandas dep
        'pooch',            # librosa dataset downloader
        'flatbuffers',      # onnx dep
        'wcwidth',          # terminal width
        'docopt',           # CLI parser
        'openfile',         # file opener
        'toolwrapper',      # CLI wrapper

        # === Build / packaging tools ===
        'pip', 'setuptools', 'wheel', 'pkg_resources', '_distutils_hack',
        'distutils_precedence',
        'pyinstaller', 'pyinstaller_hooks_contrib', 'altgraph', 'pefile',
    }

    def should_exclude(name):
        """Check if package or its dist-info / .libs should be excluded"""
        lower = name.lower()
        # Strip .libs suffix (e.g. "av.libs" -> "av")
        base = lower.replace('.libs', '')
        for pkg in EXCLUDE_PACKAGES:
            pkg_under = pkg.replace('-', '_')
            if lower == pkg or base == pkg or lower == pkg_under or base == pkg_under:
                return True
            # Match dist-info: e.g. "torch-2.8.0.dist-info"
            if lower.startswith(pkg + '-') or lower.startswith(pkg_under + '-'):
                return True
        return False

    # Copy all packages
    copied = 0
    for item in venv_site.iterdir():
        if item.name.startswith('~'):
            continue
        if should_exclude(item.name):
            print(f"  [SKIP] {item.name}")
            continue
        dst = dst_site / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        
        try:
            if item.is_dir():
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(item, dst)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Skip {item.name}: {e}")
    
    # Clean .pth files that cause issues
    for pth in dst_site.glob("*pywin32*.pth"):
        pth.unlink()
    for pth in dst_site.glob("*distutils-precedence*.pth"):
        pth.unlink()
    
    _create_stubs(dst_site)

    print(f"[OK] Copied {copied} packages")
    return True


def _create_stubs(dst_site):
    """Create stub packages for excluded dependencies (numba, pooch, transformers)."""
    numba_stub = dst_site / "numba"
    numba_stub.mkdir(exist_ok=True)
    (numba_stub / "__init__.py").write_text(
        '"""Stub: numba excluded from portable build."""\n'
        '__version__ = "0.0.0"\n'
        '_nop = lambda *a, **kw: (lambda f: f)\n'
        'jit = generated_jit = vectorize = guvectorize = stencil = _nop\n',
        encoding='utf-8')

    pooch_stub = dst_site / "pooch"
    pooch_stub.mkdir(exist_ok=True)
    (pooch_stub / "__init__.py").write_text('''\
"""Stub: pooch excluded from portable build (only used for downloading example audio)."""
import os as _os, tempfile as _tmp

def os_cache(name):
    return _os.path.join(_tmp.gettempdir(), name)

class _Pooch:
    def __init__(self):
        self.registry = {}
        self.urls = {}
        self.path = ""
        self.base_url = ""
    def fetch(self, *a, **kw): return ""
    def load_registry(self, *a, **kw): pass
    def load_registry_from_doi(self, *a, **kw): pass

def create(*a, **kw):
    p = _Pooch()
    for k, v in kw.items():
        if hasattr(p, k):
            setattr(p, k, v if v is not None else getattr(p, k))
    return p

def retrieve(*a, **kw): return ""
''', encoding='utf-8')

    # Create transformers stub (uses tokenizers lib directly, saves ~60 MB)
    tf_stub = dst_site / "transformers"
    tf_stub.mkdir(exist_ok=True)
    (tf_stub / "__init__.py").write_text(
        '"""Minimal transformers stub — only AutoTokenizer for local BERT models."""\n'
        'from transformers._auto_tokenizer import AutoTokenizer\n'
        '__version__ = "stub"\n',
        encoding='utf-8')
    (tf_stub / "_auto_tokenizer.py").write_text('''\
"""Minimal AutoTokenizer using tokenizers library (no HuggingFace download)."""
import os
import numpy as np

class _BatchEncoding:
    def __init__(self, input_ids, attention_mask, token_type_ids, word_ids_list):
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask,
                     "token_type_ids": token_type_ids}
        self._word_ids = word_ids_list
    def __getitem__(self, key):
        return self.data[key]
    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]

class _BertTokenizerCompat:
    def __init__(self, vocab_file, do_lower_case=False, model_max_length=1024, **kw):
        from tokenizers import BertWordPieceTokenizer
        self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=do_lower_case)
        self.model_max_length = model_max_length
        self.vocab = dict(self.tokenizer.get_vocab())
        self.encoder = self.vocab
        self._pad_id = self.vocab.get("[PAD]", 0)
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens[1:-1]
    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, **kw):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if not add_special_tokens: ids = ids[1:-1]
        if max_length and truncation: ids = ids[:max_length]
        return ids
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str): tokens = [tokens]
        return [self.vocab.get(t, 0) for t in tokens]
    def add_tokens(self, tokens):
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = max(self.vocab.values()) + 1 if self.vocab else 0
                self.tokenizer.add_special_tokens([t])
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    def __call__(self, texts, return_tensors=None, padding=False,
                 is_split_into_words=False, truncation=False,
                 add_special_tokens=True, max_length=None, **kw):
        all_ids, all_word_ids = [], []
        for text in texts:
            if is_split_into_words:
                ids, wids = [], []
                for wi, word in enumerate(text):
                    enc = self.tokenizer.encode(word, add_special_tokens=False)
                    ids.extend(enc.ids); wids.extend([wi] * len(enc.ids))
            else:
                enc = self.tokenizer.encode(text)
                ids = enc.ids if add_special_tokens else enc.ids[1:-1]
                wids = list(range(len(ids)))
            if truncation:
                lim = max_length or self.model_max_length
                ids, wids = ids[:lim], wids[:lim]
            all_ids.append(ids); all_word_ids.append(wids)
        if padding:
            ml = max(len(x) for x in all_ids)
            attn = []
            for i in range(len(all_ids)):
                pl = ml - len(all_ids[i])
                attn.append([1]*len(all_ids[i]) + [0]*pl)
                all_word_ids[i] += [None]*pl
                all_ids[i] += [self._pad_id]*pl
        else:
            attn = [[1]*len(x) for x in all_ids]
        if return_tensors == "np":
            ids_arr = np.array(all_ids, dtype=np.int64)
            attn_arr = np.array(attn, dtype=np.int64)
            type_ids = np.zeros_like(ids_arr)
        else:
            ids_arr, attn_arr = all_ids, attn
            type_ids = [[0]*len(x) for x in all_ids]
        return _BatchEncoding(ids_arr, attn_arr, type_ids, all_word_ids)

class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, do_basic_tokenize=False, do_lower_case=False,
                       model_max_length=512, **kw):
        if os.path.isdir(path):
            vocab_file = os.path.join(path, "vocab.txt")
            if os.path.exists(vocab_file):
                return _BertTokenizerCompat(vocab_file, do_lower_case=do_lower_case,
                                            model_max_length=model_max_length)
        raise FileNotFoundError(f"Cannot load tokenizer from {path}")
''', encoding='utf-8')

    print("[OK] Created stubs (numba, pooch, transformers)")


def copy_pyd_files():
    """Copy .pyd C extension files from venv (only for packages already copied)"""
    print("[EXT] Copying C extensions (.pyd)...")

    venv_site = get_venv_path()
    dst_site = DIST_DIR / "python" / "Lib" / "site-packages"

    copied = 0
    skipped = 0
    for pyd in venv_site.rglob("*.pyd"):
        rel_path = pyd.relative_to(venv_site)
        # Skip .pyd from excluded packages: check if top-level dir exists in dst
        top_dir = rel_path.parts[0] if len(rel_path.parts) > 1 else None
        if top_dir and not (dst_site / top_dir).exists():
            skipped += 1
            continue
        dst = dst_site / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or pyd.stat().st_size != dst.stat().st_size:
            shutil.copy2(pyd, dst)
            copied += 1

    if skipped:
        print(f"  [SKIP] {skipped} .pyd from excluded packages")
    print(f"[OK] Copied {copied} C extensions")


def copy_dlls():
    """Copy DLLs from venv packages"""
    print("[DLL] Copying DLLs...")

    venv_site = get_venv_path()
    dst_python = DIST_DIR / "python"
    dst_site = dst_python / "Lib" / "site-packages"

    # Copy .libs folders (skip those belonging to excluded packages)
    for libs_dir in venv_site.rglob("*.libs"):
        # e.g. av.libs -> check "av" against exclude list
        pkg_name = libs_dir.name.replace('.libs', '')
        # Skip if the .libs parent package was excluded (not in dst)
        if not (dst_site / pkg_name).exists():
            print(f"  [SKIP] {libs_dir.name} (excluded package)")
            continue
        dst_libs = dst_site / libs_dir.relative_to(venv_site)
        if dst_libs.exists():
            shutil.rmtree(dst_libs)
        shutil.copytree(libs_dir, dst_libs)
    
    # Copy numpy DLLs
    numpy_libs = venv_site / "numpy.libs"
    if numpy_libs.exists():
        for dll in numpy_libs.glob("*.dll"):
            shutil.copy2(dll, dst_python)
    
    # Copy sherpa-onnx DLLs (skip onnxruntime.dll — dùng bản từ pip onnxruntime)
    sherpa_lib = venv_site / "sherpa_onnx" / "lib"
    if sherpa_lib.exists():
        for dll in sherpa_lib.glob("*.dll"):
            if dll.name.lower() == "onnxruntime.dll":
                continue  # Tránh ghi đè ORT 1.24 bằng ORT 1.23 của sherpa
            shutil.copy2(dll, dst_python)

    # Copy onnxruntime.dll từ pip package (phiên bản mới nhất, đồng bộ với .pyd)
    ort_capi = venv_site / "onnxruntime" / "capi"
    if ort_capi.exists():
        ort_dll = ort_capi / "onnxruntime.dll"
        if ort_dll.exists():
            shutil.copy2(ort_dll, dst_python)
            print(f"  [OK] onnxruntime.dll from pip ({ort_dll.stat().st_size/1024/1024:.1f} MB)")
        ort_shared = ort_capi / "onnxruntime_providers_shared.dll"
        if ort_shared.exists():
            shutil.copy2(ort_shared, dst_python)

    # Đồng bộ: cập nhật sherpa_onnx/lib/onnxruntime.dll = pip version (tránh conflict)
    dst_sherpa_ort = dst_site / "sherpa_onnx" / "lib" / "onnxruntime.dll"
    dst_ort_capi = dst_site / "onnxruntime" / "capi" / "onnxruntime.dll"
    if dst_ort_capi.exists() and dst_sherpa_ort.exists():
        shutil.copy2(dst_ort_capi, dst_sherpa_ort)
        print("  [OK] Synced sherpa_onnx ORT = pip ORT (avoid version mismatch)")

    print("[OK] DLLs copied")


def copy_vcredist_dlls():
    """Copy VC++ Redistributable DLLs from PyQt6 to python/ for portable deployment.
    
    These DLLs are required for PyQt6 to run on machines without VC++ Redist installed.
    PyQt6 includes these DLLs in its Qt6/bin folder (Microsoft Universal C Runtime).
    """
    print("[VCREDIST] Copying VC++ Redistributable DLLs...")
    
    venv_site = get_venv_path()
    dst_python = DIST_DIR / "python"
    
    # PyQt6 includes VC++ runtime DLLs in Qt6/bin
    pyqt6_bin = venv_site / "PyQt6" / "Qt6" / "bin"
    
    VCREDIST_DLLS = [
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_2.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'concrt140.dll',
        'vcomp140.dll',    # OpenMP runtime - required by llama_cpp ggml-cpu.dll
    ]

    # Uu tien System32 (phien ban moi nhat, tuong thich ORT 1.24+)
    # Fallback sang PyQt6/Qt6/bin neu System32 khong co
    system32 = Path(os.environ.get('SYSTEMROOT', r'C:\Windows')) / 'System32'
    sources = [system32, pyqt6_bin] if system32.exists() else [pyqt6_bin]

    copied = 0
    for dll_name in VCREDIST_DLLS:
        for src_dir in sources:
            src = src_dir / dll_name
            if src.exists():
                shutil.copy2(src, dst_python)
                # Cung copy vao PyQt6/Qt6/bin (de PyQt6 cung dung ban moi)
                qt6_bin_dst = dst_python / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
                if qt6_bin_dst.exists():
                    shutil.copy2(src, qt6_bin_dst)
                copied += 1
                break
        else:
            print(f"  [WARN] {dll_name} not found")

    if copied > 0:
        print(f"[OK] Copied {copied} VC++ Redistributable DLLs (from System32)")
    else:
        print("[WARN] No VC++ Redistributable DLLs copied")

    # NOTE: Desktop app chay tren Windows 10/11 (co system ICU san)
    # Khong can ship ICU DLLs. Webapp build (build_portable_online.py) tu xu ly ICU shim.


def cleanup_pyqt6():
    """Remove unused Qt6 modules to save ~400 MB (WebEngine, QML, Quick3D, etc.)"""
    print("[QT6] Cleaning up unused Qt6 modules...")

    qt6_dir = DIST_DIR / "python" / "Lib" / "site-packages" / "PyQt6" / "Qt6"
    if not qt6_dir.exists():
        print("  [SKIP] PyQt6/Qt6 not found")
        return

    saved = 0

    # App only uses: QtWidgets, QtCore, QtGui, QtMultimedia
    # Keep these DLLs + their dependencies
    KEEP_DLLS = {
        # Core Qt modules used by app
        'Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll', 'Qt6Multimedia.dll',
        'Qt6Network.dll',     # needed by Multimedia
        'Qt6Svg.dll',         # SVG icon support
        'Qt6OpenGL.dll',      # GPU rendering
        'Qt6OpenGLWidgets.dll',
        'Qt6MultimediaWidgets.dll',
        # MSVC runtime (required)
        'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
        'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll',
        # DirectX / OpenGL fallback
        'd3dcompiler_47.dll', 'opengl32sw.dll',
        # ICU: desktop khong can (system ICU co san tren Win 10/11)
        # Webapp build tu them ICU shim rieng
    }

    bin_dir = qt6_dir / "bin"
    if bin_dir.exists():
        for dll in bin_dir.glob("*.dll"):
            if dll.name not in KEEP_DLLS:
                size = dll.stat().st_size
                dll.unlink()
                saved += size

    # Remove WebEngine resources entirely
    resources_dir = qt6_dir / "resources"
    if resources_dir.exists():
        for f in resources_dir.rglob('*'):
            if f.is_file():
                saved += f.stat().st_size
        shutil.rmtree(resources_dir)

    # Remove translations (52 MB) — keep only Vietnamese + English
    translations_dir = qt6_dir / "translations"
    if translations_dir.exists():
        for f in list(translations_dir.iterdir()):
            name = f.name.lower()
            if name.startswith('qtbase_vi') or name.startswith('qtbase_en') \
                    or name == 'qt_vi.qm' or name == 'qt_en.qm':
                continue
            if f.is_dir():
                for sub in f.rglob('*'):
                    if sub.is_file():
                        saved += sub.stat().st_size
                shutil.rmtree(f)
            else:
                saved += f.stat().st_size
                f.unlink()

    # Remove QML directory (18 MB) — app uses QtWidgets, not QML
    qml_dir = qt6_dir / "qml"
    if qml_dir.exists():
        for f in qml_dir.rglob('*'):
            if f.is_file():
                saved += f.stat().st_size
        shutil.rmtree(qml_dir)

    # Remove unused plugins
    plugins_dir = qt6_dir / "plugins"
    KEEP_PLUGINS = {'platforms', 'styles', 'imageformats', 'multimedia', 'iconengines'}
    if plugins_dir.exists():
        for d in list(plugins_dir.iterdir()):
            if d.is_dir() and d.name not in KEEP_PLUGINS:
                for f in d.rglob('*'):
                    if f.is_file():
                        saved += f.stat().st_size
                shutil.rmtree(d)

    # Remove PyQt6-WebEngine .pyd bindings
    pyqt6_dir = DIST_DIR / "python" / "Lib" / "site-packages" / "PyQt6"
    for pattern in ['QtWebEngine*', 'QtQml*', 'QtQuick*', 'QtDesigner*',
                    'QtPdf*', 'QtRemoteObjects*', 'QtBluetooth*',
                    'QtNfc*', 'QtSensors*', 'QtSerialPort*',
                    'QtPositioning*', 'QtTest*', 'QtHelp*',
                    'QtSpatialAudio*', 'QtTextToSpeech*']:
        for f in pyqt6_dir.glob(pattern):
            if f.is_file():
                saved += f.stat().st_size
                f.unlink()
            elif f.is_dir():
                for sub in f.rglob('*'):
                    if sub.is_file():
                        saved += sub.stat().st_size
                shutil.rmtree(f)

    print(f"[OK] Cleaned PyQt6: saved {saved / 1024**2:.0f} MB")


def copy_source_files():
    """Copy application source code"""
    print("[SRC] Copying source files...")

    for f in SOURCE_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, DIST_DIR)
        else:
            print(f"  [WARN] Not found: {f}")

    # Copy module directories (core/)
    for mod in MODULE_DIRS:
        src = PROJECT_ROOT / mod
        dst = DIST_DIR / mod
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            print(f"  [OK] {mod}/ copied")
        else:
            print(f"  [WARN] Module not found: {mod}/")

    print("[OK] Source files copied")


def copy_data():
    """Copy data directories (models, vocabulary)"""
    print("[DATA] Copying data directories...")

    # Model dirs/files to EXCLUDE entirely (unused at runtime)
    EXCLUDE_MODELS = {
        'moonshine-base-vi',
        # Unused ASR models
        'gipformer-65M-rnnt',
        'myfinetune', 'myfinetune2', 'myfinetune3',
        'sherpa-onnx-zipformer-vi-30M',
        'sherpa-onnx-zipformer-vi-2025-04-20_tmp',
        # NOTE: zipformer-30m-rnnt-streaming-6000h is KEPT (used by tab_live.py)
        # CEM models (not integrated in runtime)
        'cem-retrain',
        # Legacy diarization (replaced by pure ORT)
        'speaker_embedding',           # nemo titanet — not needed for pure ORT
        'speaker_diarization',         # sherpa-onnx segmentation — not needed for pure ORT
        # Unused diarization variants
        'diarizen',                    # 278MB wavlm-large — not used in runtime
        'ecapa-wespeaker',             # not referenced in code
        'campp-wespeaker',             # campp_pure_ort not in model registry
        # Audio enhancement not used
        'gtcrn',
        # Summary/LLM models (chưa hoàn thiện)
        '.cache',
    }
    # Skip LLM model files by extension
    EXCLUDE_MODEL_EXTENSIONS = {'.gguf'}

    # CEM root files to exclude (glob patterns checked separately)
    CEM_PATTERNS = ['cem_*.pt', 'cem_*.onnx', 'cem_*.json']

    for dir_name in DATA_DIRS:
        src = PROJECT_ROOT / dir_name
        if src.exists():
            dst = DIST_DIR / dir_name
            if dst.exists():
                shutil.rmtree(dst)

            if dir_name == 'models':
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    # Skip excluded dirs
                    if item.name in EXCLUDE_MODELS:
                        print(f"  [SKIP] {item.name}")
                        continue
                    # Skip CEM root files
                    if any(item.match(p) for p in CEM_PATTERNS):
                        print(f"  [SKIP] {item.name}")
                        continue
                    # Skip LLM model files
                    if item.suffix.lower() in EXCLUDE_MODEL_EXTENSIONS:
                        print(f"  [SKIP] {item.name}")
                        continue
                    dst_item = dst / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst_item, ignore=shutil.ignore_patterns(
                            '.git', '.cache', '.gitattributes', '__pycache__'))
                    else:
                        shutil.copy2(item, dst_item)
            else:
                shutil.copytree(src, dst)
            # Count files
            count = sum(1 for _ in dst.rglob('*') if _.is_file())
            print(f"  [OK] {dir_name}: {count} files")
        else:
            print(f"  [WARN] Not found: {dir_name}")

    # Clean vibert-capu: Desktop giữ int8, xóa fp32 (tiết kiệm 328MB)
    vibert_dst = DIST_DIR / "models" / "vibert-capu"
    if vibert_dst.exists():
        VIBERT_KEEP = {'vibert-capu.int8.onnx', 'vocab.txt', 'config.json'}
        removed = 0
        for f in list(vibert_dst.rglob('*')):
            if f.is_file() and f.name not in VIBERT_KEEP:
                size_mb = f.stat().st_size / 1024 / 1024
                f.unlink()
                removed += size_mb
        # Remove empty subdirs
        for d in sorted(vibert_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] vibert-capu: removed {removed:.0f} MB (kept int8.onnx, vocab.txt, config.json)")

    # Clean pyannote PyTorch dir: keep only plda/ data (needed by pure ORT)
    pyannote_dst = DIST_DIR / "models" / "pyannote" / "speaker-diarization-community-1"
    if pyannote_dst.exists():
        removed = 0
        for f in list(pyannote_dst.rglob('*')):
            if f.is_file() and f.suffix == '.bin':
                size_mb = f.stat().st_size / 1024 / 1024
                f.unlink()
                removed += size_mb
        # Remove empty dirs
        for d in sorted(pyannote_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] pyannote: removed {removed:.0f} MB pytorch_model.bin (kept plda/)")

    # Global cleanup: remove non-ONNX artifacts from ALL model dirs
    models_dst = DIST_DIR / "models"
    if models_dst.exists():
        # Extensions that are NOT needed at runtime
        JUNK_EXTS = {'.bin', '.pt', '.pth', '.pkl', '.safetensors', '.filepart',
                     '.md', '.gitattributes'}
        removed_total = 0
        for f in list(models_dst.rglob('*')):
            if f.is_file() and f.suffix.lower() in JUNK_EXTS:
                size_mb = f.stat().st_size / 1024 / 1024
                removed_total += size_mb
                print(f"  [DEL] {f.relative_to(models_dst)} ({size_mb:.1f} MB)")
                f.unlink()

        # Desktop build: Remove fp32 ONNX when int8 exists (desktop prefers int8)
        for model_dir in models_dst.iterdir():
            if not model_dir.is_dir():
                continue
            int8_files = list(model_dir.glob('*int8*.onnx'))
            for int8f in int8_files:
                fp32_name = int8f.name.replace('.int8', '')
                fp32_path = model_dir / fp32_name
                if fp32_path.exists():
                    size_mb = fp32_path.stat().st_size / 1024 / 1024
                    removed_total += size_mb
                    print(f"  [DEL] {fp32_path.relative_to(models_dst)} ({size_mb:.1f} MB) — int8 exists")
                    fp32_path.unlink()

        # Remove pyannote-onnx fallback embedding models (pure ORT uses embedding_encoder.onnx)
        for name in ['embedding_model.onnx', 'embedding_model_split.onnx']:
            p = models_dst / 'pyannote-onnx' / name
            if p.exists() and (models_dst / 'pyannote-onnx' / 'embedding_encoder.onnx').exists():
                size_mb = p.stat().st_size / 1024 / 1024
                removed_total += size_mb
                print(f"  [DEL] pyannote-onnx/{name} ({size_mb:.1f} MB) — encoder exists")
                p.unlink()

        # Remove empty dirs
        for d in sorted(models_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed_total > 0:
            print(f"  [TRIM] Cleaned {removed_total:.0f} MB non-ONNX artifacts from models/")

    print("[OK] Data copied")


def create_launcher():
    """Create launcher batch file"""
    print("[LNCH] Creating launcher...")
    
    bat_content = '''@echo off
chcp 65001 >nul
setlocal

set "BASE_DIR=%~dp0"
set "PYTHON_EXE=%BASE_DIR%python\\python.exe"
set "APP_SCRIPT=%BASE_DIR%app.py"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Không tìm thấy Python embedded
    pause
    exit /b 1
)

set "PYTHONHOME=%BASE_DIR%python"
set "PYTHONDONTWRITEBYTECODE=1"
set "QT6_BIN=%BASE_DIR%python\\Lib\\site-packages\\PyQt6\\Qt6\\bin"
set "PATH=%QT6_BIN%;%BASE_DIR%python;%BASE_DIR%python\\Lib\\site-packages;%PATH%"

"%PYTHON_EXE%" "%APP_SCRIPT%" %*

if %errorlevel% neq 0 (
    echo.
    echo [Loi] Chuong trinh ket thuc voi ma loi %errorlevel%
    pause
)
exit /b %errorlevel%
'''
    
    launcher_path = DIST_DIR / "sherpa-vietnamese-asr.bat"
    launcher_path.write_text(bat_content, encoding='utf-8')
    
    # Create README
    readme = '''ASR-VN Portable
===============

Run: Double-click sherpa-vietnamese-asr.bat

Requirements:
- Windows 10/11 64-bit
- No Python installation required

Folder structure:
- python/           : Python embedded runtime
- models/           : AI models
- vocabulary/       : Vocabulary data
- *.py              : Source code
- sherpa-vietnamese-asr.bat : Launcher
'''
    (DIST_DIR / "README.txt").write_text(readme, encoding='utf-8')
    
    print(f"[OK] Launcher created: sherpa-vietnamese-asr.bat")


def trim_portable():
    """Remove unnecessary files from build to reduce size"""
    site = DIST_DIR / "python" / "Lib" / "site-packages"
    if not site.exists():
        return

    removed = 0
    # Remove __pycache__
    for d in list(site.rglob("__pycache__")):
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    removed += f.stat().st_size
            shutil.rmtree(d, ignore_errors=True)

    # Remove .dist-info (pip metadata, not needed at runtime)
    for d in site.glob("*.dist-info"):
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    removed += f.stat().st_size
            shutil.rmtree(d, ignore_errors=True)

    # Remove test dirs
    for d in list(site.rglob("tests")):
        if d.is_dir() and d.parent.parent == site:
            for f in d.rglob("*"):
                if f.is_file():
                    removed += f.stat().st_size
            shutil.rmtree(d, ignore_errors=True)

    # Remove .pyi stubs (EXCEPT packages using lazy_loader: librosa, scipy, sklearn)
    keep_pyi = {'librosa', 'scipy', 'sklearn'}
    for f in site.rglob("*.pyi"):
        if f.is_file():
            # Check if this .pyi is in a package that needs it
            rel = f.relative_to(site)
            pkg = rel.parts[0] if rel.parts else ""
            if pkg not in keep_pyi:
                removed += f.stat().st_size
                f.unlink()

    if removed > 0:
        print(f"[OK] Trimmed {removed / 1024 / 1024:.0f} MB (pycache, dist-info, tests, stubs)")


def clean_build():
    """Clean temporary build files"""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def calculate_size():
    """Calculate total distribution size"""
    total = sum(f.stat().st_size for f in DIST_DIR.rglob('*') if f.is_file())
    return total / 1024**3  # GB


def main():
    print("="*60)
    print("ASR-VN Portable Build")
    print("="*60)
    print()
    
    # Check prerequisites
    if not check_venv():
        return 1
    
    # Clean old build
    print("[CLEAN] Cleaning old build...")
    if DIST_DIR.exists():
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        import stat
        shutil.rmtree(DIST_DIR, onerror=on_rm_error)
    DIST_DIR.mkdir(parents=True)
    
    try:
        # Download and setup Python
        zip_file = download_python_embedded()
        setup_python(zip_file)
        
        # Copy from venv
        if not copy_venv_packages():
            return 1
        copy_pyd_files()
        copy_dlls()
        copy_vcredist_dlls()  # Copy VC++ Redist DLLs for PyQt6 on target machines

        # Trim unused Qt6 modules (WebEngine, QML, Quick3D, etc.)
        cleanup_pyqt6()

        # Copy source and data
        copy_source_files()
        copy_data()

        # Create launcher
        create_launcher()

        # Trim unnecessary files
        trim_portable()

        # Cleanup
        clean_build()

        # Xóa .opt files — máy target sẽ tự tạo lần đầu chạy (phụ thuộc ORT version + CPU)
        for opt_file in DIST_DIR.rglob("*.opt"):
            opt_file.unlink()
            print(f"  [DEL] {opt_file.relative_to(DIST_DIR)} (ORT cache, auto-generated on target)")

        # Report
        print()
        print("="*60)
        print("BUILD SUCCESS!")
        print("="*60)
        print(f"Location: {DIST_DIR.absolute()}")
        print(f"Run:      {DIST_DIR / 'sherpa-vietnamese-asr.bat'}")
        print(f"Size:     {calculate_size():.2f} GB")
        print()
        print("You can zip the folder and distribute it.")
        print("No Python installation required on target machine.")
        
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print(f"BUILD FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Fix stdout encoding khi chay truc tiep (cp1252 -> utf-8)
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    sys.exit(main())
