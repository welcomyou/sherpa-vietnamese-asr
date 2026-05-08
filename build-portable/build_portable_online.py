#!/usr/bin/env python3
"""
Build script cho ban sherpa-vietnamese-asr-service (web service + admin GUI).
Dua tren build_portable.py nhung them web_service/ va loai tru streaming model.

Usage: python build-portable/build_portable_online.py
"""
import io
import os
import sys
import shutil
import stat
from pathlib import Path

# Fix stdout encoding on Windows (cp1252 -> utf-8) de print Unicode khong bi crash
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Reuse logic tu build_portable.py
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Import cac ham tu build_portable.py
sys.path.insert(0, str(SCRIPT_DIR))
from build_portable import (
    VENV_DIR, BUILD_DIR, PYTHON_EMBED_URL,
    check_venv, download_python_embedded, setup_python,
    copy_pyd_files, copy_dlls, copy_vcredist_dlls, cleanup_pyqt6, _create_stubs,
    get_venv_path, clean_build, calculate_size,
)

# Override output directory
DIST_DIR_ONLINE = PROJECT_ROOT / "dist" / "sherpa-vietnamese-asr-service"

# Source files cho ban online (KHONG co tab_live, streaming)
ONLINE_SOURCE_FILES = [
    "server_launcher.py",
    "server_gui.py",
    "service_installer.py",
    "config.ini",
    "hotword.txt",           # hotwords cho Sherpa-ONNX
    "verb-form-vocab.txt",
    "speaker_hotkeys.json",
    "ffmpeg.exe",
    "ffprobe.exe",
    "requirements-online.txt",
]

# Models KHONG copy
EXCLUDE_MODELS_ONLINE = {
    "moonshine-base-vi",
    "zipformer-30m-rnnt-streaming-6000h",
    # Unused ASR models
    'gipformer-65M-rnnt', 'myfinetune', 'myfinetune2', 'myfinetune3',
    'sherpa-onnx-zipformer-vi-30M',
    'sherpa-onnx-zipformer-vi-2025-04-20_tmp',
    # CEM (not integrated in runtime)
    'cem-retrain',
    # Legacy diarization (pure ORT replaces)
    'speaker_embedding', 'speaker_diarization',
    # Unused diarization variants
    'diarizen',                    # 278MB wavlm-large — not used in runtime
    'ecapa-wespeaker',             # not referenced in code
    'campp-wespeaker',             # campp_pure_ort not in model registry
    # Not used
    'gtcrn',
    # Summary/LLM models (chưa hoàn thiện)
    '.cache',
}

# GGUF model files (match by extension, skip tất cả)
EXCLUDE_MODEL_EXTENSIONS = {'.gguf'}

# Packages khong can cho web service (giam dung luong)
EXCLUDE_PACKAGES_SERVICES = {
    'moonshine_voice',

    # === PyTorch + ecosystem — app uses ONNX Runtime only ===
    'torch', 'torchaudio', 'torio', 'torchmetrics', 'torchgen',
    'torchcodec', 'torch_audiomentations', 'torch_pitch_shift',
    'functorch', 'ml_dtypes',
    'sympy', 'networkx',
    # jinja2 + markupsafe: GIU LAI — llama_cpp requires jinja2 for chat templates

    # === PyTorch Lightning ===
    'lightning', 'lightning_fabric', 'lightning_utilities',
    'pytorch_lightning', 'pytorch_metric_learning',

    # === Pyannote — pure ORT diarization replaces it ===
    'pyannote', 'pyannote_audio', 'pyannote_core', 'pyannote_pipeline',
    'pyannote_database', 'pyannote_metrics', 'pyannote_onnx_extended',
    'pyannoteai', 'pyannoteai_sdk',
    'asteroid', 'asteroid_filterbanks', 'julius', 'einops',

    # numba/llvmlite: GIỮ LẠI — umap-learn (Senko diarization) cần numba.njit thật
    # 'numba', 'llvmlite',

    # === Large unused packages ===
    'pandas', 'pymupdf', 'fitz', 'onnx', 'ctranslate2', 'av',

    # === Desktop-only packages ===
    'sounddevice', '_sounddevice_data',
    'matplotlib', 'mpl_toolkits', 'kiwisolver', 'contourpy', 'cycler',
    'pyparsing',

    # Auth: auth.py dung hashlib, khong dung passlib
    'passlib', 'bcrypt',
    # Database: web service dung sqlite3 truc tiep, khong can ORM
    'sqlalchemy', 'aiosqlite', 'greenlet', 'alembic', 'mako',
    # HTTP client: web service khong dung aiohttp/httpx
    'aiohttp', 'aiofiles', 'aiohappyeyeballs', 'aiosignal',
    'frozenlist', 'multidict', 'yarl', 'propcache',
    'httpcore', 'httpx',

    # === gRPC ===
    'grpc', 'grpcio', 'grpcio_tools',
    'google', 'googleapis_common_protos', 'protobuf',

    # === Monitoring / Logging ===
    'opentelemetry', 'opentelemetry_api', 'opentelemetry_sdk',
    'opentelemetry_proto', 'opentelemetry_exporter_otlp',
    'opentelemetry_exporter_otlp_proto_http',
    'opentelemetry_exporter_otlp_proto_grpc',
    'opentelemetry_exporter_otlp_proto_common',
    'opentelemetry_semantic_conventions',
    'pygments', 'rich', 'colorlog', 'colorama',

    # === transformers + deps — replaced by minimal stub ===
    'transformers',
    'huggingface_hub', 'safetensors',
    'regex', 'fsspec',  # tqdm: GIỮ LẠI — umap-learn cần tqdm.auto

    # === Text processing / Math / Misc not used ===
    'jiwer', 'langid', 'mosestokenizer', 'wtpsplit', 'textgrid',
    'markdown_it', 'mdurl',
    'optuna', 'primePy', 'mpmath', 'skops', 'prettytable',
    'adapters', 'rapidfuzz', 'chunkformer',
    'wespeakerruntime', 'silero_vad', 'faster_whisper',
    'diarize', 'nara_wpe', 'kaldiio',
    'bottleneck', 'pooch', 'flatbuffers', 'wcwidth', 'docopt',
    'openfile', 'toolwrapper',

    # === Build / packaging tools ===
    'pip', 'setuptools', 'wheel', 'pkg_resources', '_distutils_hack',
    'distutils_precedence',
    'pyinstaller', 'pyinstaller_hooks_contrib', 'altgraph', 'pefile',
}


def should_exclude_services(name):
    """Check if package or its dist-info / .libs should be excluded for service build"""
    lower = name.lower()
    base = lower.replace('.libs', '')
    for pkg in EXCLUDE_PACKAGES_SERVICES:
        pkg_under = pkg.replace('-', '_')
        if lower == pkg or base == pkg or lower == pkg_under or base == pkg_under:
            return True
        if lower.startswith(pkg + '-') or lower.startswith(pkg_under + '-'):
            return True
    return False


def copy_venv_packages_services():
    """Copy packages from virtualenv, excluding desktop-only packages"""
    print("[PKG] Copying packages from venv (services)...")

    venv_site = get_venv_path()
    dst_site = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages"
    dst_site.mkdir(parents=True, exist_ok=True)

    if not venv_site.exists():
        print(f"[ERROR] Venv site-packages not found: {venv_site}")
        return False

    copied = 0
    skipped = 0
    for item in venv_site.iterdir():
        if item.name.startswith('~'):
            continue
        if should_exclude_services(item.name):
            print(f"  [SKIP] {item.name}")
            skipped += 1
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

    # Clean CUDA DLLs from llama_cpp (server is CPU only, saves ~300MB)
    llama_lib = dst_site / "llama_cpp" / "lib"
    if llama_lib.exists():
        for cuda_dll in list(llama_lib.glob("cuda*.dll")) + list(llama_lib.glob("cublas*.dll")):
            cuda_dll.unlink()
            print(f"  [CLEAN] Removed CUDA DLL: {cuda_dll.name}")

    # Create stubs (same as desktop build)
    from build_portable import _create_stubs
    _create_stubs(dst_site)

    print(f"[OK] Copied {copied} packages, skipped {skipped}")
    return True


def copy_source_files_online():
    """Copy source files cho ban online"""
    print("[SRC] Copying online source files...")

    for f in ONLINE_SOURCE_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, DIST_DIR_ONLINE)
        else:
            print(f"  [WARN] Not found: {f}")

    # Portable build: bind 0.0.0.0 thay vì IP cố định của máy dev
    config_dst = DIST_DIR_ONLINE / "config.ini"
    if config_dst.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(config_dst, encoding="utf-8")
        if cfg.has_section("ServerSettings"):
            cfg.set("ServerSettings", "host", "0.0.0.0")
            with open(config_dst, "w", encoding="utf-8") as f:
                cfg.write(f)
            print("  [OK] config.ini: host set to 0.0.0.0")

    # Copy core/ module
    core_src = PROJECT_ROOT / "core"
    core_dst = DIST_DIR_ONLINE / "core"
    if core_dst.exists():
        shutil.rmtree(core_dst)
    shutil.copytree(core_src, core_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("  [OK] core/ copied")

    # Copy web_service/ module
    ws_src = PROJECT_ROOT / "web_service"
    ws_dst = DIST_DIR_ONLINE / "web_service"
    if ws_dst.exists():
        shutil.rmtree(ws_dst)
    shutil.copytree(ws_src, ws_dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "data", "certs"  # certs tu generate khi chay
    ))
    # Tao thu muc data rong va certs rong
    (ws_dst / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (ws_dst / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (ws_dst / "certs").mkdir(parents=True, exist_ok=True)
    print("  [OK] web_service/ copied")

    # Copy vocabulary/
    vocab_src = PROJECT_ROOT / "vocabulary"
    if vocab_src.exists():
        vocab_dst = DIST_DIR_ONLINE / "vocabulary"
        if vocab_dst.exists():
            shutil.rmtree(vocab_dst)
        shutil.copytree(vocab_src, vocab_dst)
        print("  [OK] vocabulary/ copied")

    print("[OK] Online source files copied")


def copy_server_extras():
    """Copy ICU shim (cho Windows Server 2016), nssm.exe, bat files"""
    print("[EXTRA] Copying server extras...")

    # ICU shim DLLs (Qt6Core.dll can ICU, Windows Server 2016 khong co)
    icu_shim_dir = SCRIPT_DIR / "icu-shim"
    qt6_bin = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
    qt6_bin.mkdir(parents=True, exist_ok=True)
    icu_copied = 0
    for icu_name in ['icuuc.dll', 'icuuc73.dll', 'icuin73.dll', 'icudt73.dll']:
        src = icu_shim_dir / icu_name
        if src.exists():
            shutil.copy2(src, qt6_bin)
            icu_copied += 1
    if icu_copied >= 4:
        print("  [OK] ICU shim + ICU 73 (Windows Server compatible)")
    else:
        print(f"  [WARN] ICU shim khong day du ({icu_copied}/4) trong {icu_shim_dir}")

    # nssm.exe (cai Windows Service)
    nssm_src = SCRIPT_DIR / "nssm.exe"
    if nssm_src.exists():
        shutil.copy2(nssm_src, DIST_DIR_ONLINE)
        print("  [OK] nssm.exe")
    else:
        print("  [WARN] nssm.exe khong tim thay")

    # Bat files
    for bat_name in ['start-gui.bat', 'start-server.bat', 'install-service.bat']:
        bat_src = SCRIPT_DIR / "server-bats" / bat_name
        if bat_src.exists():
            shutil.copy2(bat_src, DIST_DIR_ONLINE)
            print(f"  [OK] {bat_name}")

    print("[OK] Server extras copied")


def copy_models_online():
    """Copy models, loai tru unused models"""
    print("[MODEL] Copying models...")

    models_src = PROJECT_ROOT / "models"
    models_dst = DIST_DIR_ONLINE / "models"

    if not models_src.exists():
        print("  [WARN] models/ not found")
        return

    # CEM root file patterns to skip
    CEM_PATTERNS = ['cem_*.pt', 'cem_*.onnx', 'cem_*.json']

    models_dst.mkdir(parents=True, exist_ok=True)

    for item in models_src.iterdir():
        if item.name in EXCLUDE_MODELS_ONLINE:
            print(f"  [SKIP] {item.name}")
            continue
        if any(item.match(p) for p in CEM_PATTERNS):
            print(f"  [SKIP] {item.name}")
            continue
        if item.suffix.lower() in EXCLUDE_MODEL_EXTENSIONS:
            print(f"  [SKIP] {item.name}")
            continue

        dst_item = models_dst / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, ignore=shutil.ignore_patterns(
                            '.git', '.cache', '.gitattributes', '__pycache__'))
        else:
            shutil.copy2(item, dst_item)
        print(f"  [OK] {item.name}")

    # Clean vibert-capu: keep only onnx + vocab + config
    vibert_dst = models_dst / "vibert-capu"
    if vibert_dst.exists():
        VIBERT_KEEP = {'vibert-capu.onnx', 'vocab.txt', 'config.json'}
        removed = 0
        for f in list(vibert_dst.rglob('*')):
            if f.is_file() and f.name not in VIBERT_KEEP:
                removed += f.stat().st_size / 1024 / 1024
                f.unlink()
        for d in sorted(vibert_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] vibert-capu: removed {removed:.0f} MB")

    # Clean pyannote: keep only plda/ (remove pytorch_model.bin)
    pyannote_dst = models_dst / "pyannote" / "speaker-diarization-community-1"
    if pyannote_dst.exists():
        removed = 0
        for f in list(pyannote_dst.rglob('*.bin')):
            removed += f.stat().st_size / 1024 / 1024
            f.unlink()
        for d in sorted(pyannote_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] pyannote: removed {removed:.0f} MB pytorch_model.bin")

    # Global cleanup: remove non-ONNX artifacts from ALL model dirs
    JUNK_EXTS = {'.bin', '.pt', '.pth', '.pkl', '.safetensors', '.filepart',
                 '.md', '.gitattributes'}
    removed_total = 0
    for f in list(models_dst.rglob('*')):
        if f.is_file() and f.suffix.lower() in JUNK_EXTS:
            size_mb = f.stat().st_size / 1024 / 1024
            removed_total += size_mb
            print(f"  [DEL] {f.relative_to(models_dst)} ({size_mb:.1f} MB)")
            f.unlink()
    # Remove int8 duplicates when fp32 exists
    for model_dir in models_dst.iterdir():
        if not model_dir.is_dir():
            continue
        for int8f in list(model_dir.glob('*int8*.onnx')):
            fp32_name = int8f.name.replace('.int8', '')
            if (model_dir / fp32_name).exists():
                size_mb = int8f.stat().st_size / 1024 / 1024
                removed_total += size_mb
                print(f"  [DEL] {int8f.relative_to(models_dst)} ({size_mb:.1f} MB) — fp32 exists")
                int8f.unlink()
    for name in ['embedding_model.onnx', 'embedding_model_split.onnx']:
        p = models_dst / 'pyannote-onnx' / name
        if p.exists() and (models_dst / 'pyannote-onnx' / 'embedding_encoder.onnx').exists():
            size_mb = p.stat().st_size / 1024 / 1024
            removed_total += size_mb
            print(f"  [DEL] pyannote-onnx/{name} ({size_mb:.1f} MB)")
            p.unlink()
    for d in sorted(models_dst.rglob('*'), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if removed_total > 0:
        print(f"  [TRIM] Cleaned {removed_total:.0f} MB non-ONNX artifacts")

    count = sum(1 for _ in models_dst.rglob("*") if _.is_file())
    print(f"[OK] Models copied: {count} files")


def create_launcher_online():
    """Tao launcher batch cho ban services"""
    print("[LNCH] Creating service launcher...")

    # Server launcher
    bat_content = (
        '@echo off\n'
        'chcp 65001 >nul\n'
        'setlocal\n'
        '\n'
        'set "BASE_DIR=%~dp0"\n'
        'set "PYTHON_EXE=%BASE_DIR%python\\python.exe"\n'
        '\n'
        'if not exist "%PYTHON_EXE%" (\n'
        '    echo ERROR: Khong tim thay Python embedded\n'
        '    pause\n'
        '    exit /b 1\n'
        ')\n'
        '\n'
        'set "PYTHONHOME=%BASE_DIR%python"\n'
        'set "PYTHONDONTWRITEBYTECODE=1"\n'
        'set "QT6_BIN=%BASE_DIR%python\\Lib\\site-packages\\PyQt6\\Qt6\\bin"\n'
        'set "PATH=%QT6_BIN%;%BASE_DIR%python;%BASE_DIR%python\\Lib\\site-packages;%BASE_DIR%;%PATH%"\n'
        '\n'
        'echo ===================================\n'
        'echo  Sherpa Vietnamese ASR - Service\n'
        'echo ===================================\n'
        'echo.\n'
        '\n'
        'rem Doc port tu config.ini\n'
        'set "PORT=8443"\n'
        'for /f "tokens=2 delims== " %%a in (\'findstr /i "^port" "%BASE_DIR%config.ini" 2^>nul\') do set "PORT=%%a"\n'
        '\n'
        'if "%1"=="--no-gui" goto :headless\n'
        '\n'
        'rem Thu khoi dong GUI (can Windows 10 1809+ / Server 2019+)\n'
        '"%PYTHON_EXE%" -c "from PyQt6.QtWidgets import QApplication" >nul 2>&1\n'
        'if %errorlevel% equ 0 (\n'
        '    echo Khoi dong Admin GUI...\n'
        '    "%PYTHON_EXE%" "%BASE_DIR%server_gui.py" %*\n'
        '    goto :done\n'
        ')\n'
        '\n'
        'echo [Thong bao] Giao dien GUI can Windows 10 1809+ / Server 2019+\n'
        'echo Tu dong chuyen sang che do headless...\n'
        'echo.\n'
        '\n'
        ':headless\n'
        'echo Server dang chay. Truy cap:\n'
        'echo.\n'
        'echo   https://localhost:%PORT%\n'
        'echo   https://[IP-may-nay]:%PORT%\n'
        'echo.\n'
        'echo Dang nhap admin de quan tri he thong qua web.\n'
        'echo Nhan Ctrl+C de dung server.\n'
        'echo -------------------------------------------------------\n'
        'echo.\n'
        '"%PYTHON_EXE%" "%BASE_DIR%server_launcher.py" --no-gui\n'
        '\n'
        ':done\n'
        'if %errorlevel% neq 0 (\n'
        '    echo.\n'
        '    echo [Loi] Chuong trinh ket thuc voi ma loi %errorlevel%\n'
        '    pause\n'
        ')\n'
        '\n'
        'exit /b %errorlevel%\n'
    )

    (DIST_DIR_ONLINE / "sherpa-vietnamese-asr-service.bat").write_text(bat_content, encoding="utf-8")

    # start-gui.bat
    (DIST_DIR_ONLINE / "start-gui.bat").write_text(
        (SCRIPT_DIR / ".." / "dist" / "sherpa-vietnamese-asr-service" / "start-gui.bat").read_text(encoding="utf-8")
        if (DIST_DIR_ONLINE / ".." / ".." / "dist" / "sherpa-vietnamese-asr-service" / "start-gui.bat").exists()
        else "", encoding="utf-8"
    )

    # start-server.bat
    (DIST_DIR_ONLINE / "start-server.bat").write_text(
        (SCRIPT_DIR / ".." / "dist" / "sherpa-vietnamese-asr-service" / "start-server.bat").read_text(encoding="utf-8")
        if (DIST_DIR_ONLINE / ".." / ".." / "dist" / "sherpa-vietnamese-asr-service" / "start-server.bat").exists()
        else "", encoding="utf-8"
    )

    # install-service.bat
    (DIST_DIR_ONLINE / "install-service.bat").write_text(
        (SCRIPT_DIR / ".." / "dist" / "sherpa-vietnamese-asr-service" / "install-service.bat").read_text(encoding="utf-8")
        if (DIST_DIR_ONLINE / ".." / ".." / "dist" / "sherpa-vietnamese-asr-service" / "install-service.bat").exists()
        else "", encoding="utf-8"
    )

    # README
    readme = '''Sherpa Vietnamese ASR - Service
================================

Cac file khoi dong:
  start-server.bat      -> Chay server headless (moi Windows)
  start-gui.bat         -> Mo GUI quan tri (can Windows 10 1809+ / Server 2019+)
  install-service.bat   -> Cai dat thanh Windows Service (can quyen Admin)

Yeu cau:
- Windows 10+ / Windows Server 2016+ (64-bit)
- Khong can cai Python

Sau khi chay:
- Truy cap https://IP:8443 tren browser
- Admin mac dinh: admin / admin (doi ngay sau khi dang nhap)
- Dang nhap admin -> nut "Quan tri" hien ra -> quan ly phien, queue, user qua web
- Browser se canh bao "Not Secure" vi self-signed cert, bam "Advanced" -> "Proceed"
'''
    (DIST_DIR_ONLINE / "README.txt").write_text(readme, encoding="utf-8")

    print("[OK] Services launcher created")


def main():
    global DIST_DIR_ONLINE

    print("=" * 60)
    print("Sherpa Vietnamese ASR - Service Portable Build")
    print("=" * 60)
    print()

    if not check_venv():
        return 1

    # Clean old build
    print("[CLEAN] Cleaning old service build...")
    if DIST_DIR_ONLINE.exists():
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(DIST_DIR_ONLINE, onerror=on_rm_error)

    # Also clean old-named directory if exists
    old_dir = PROJECT_ROOT / "dist" / "sherpa-asr-vn-online"
    if old_dir.exists():
        print("[CLEAN] Removing old sherpa-asr-vn-online...")
        shutil.rmtree(old_dir, onerror=lambda f, p, e: (os.chmod(p, stat.S_IWRITE), f(p)))

    DIST_DIR_ONLINE.mkdir(parents=True)

    try:
        # Tam thoi override DIST_DIR trong build_portable module
        import build_portable
        original_dist = build_portable.DIST_DIR
        build_portable.DIST_DIR = DIST_DIR_ONLINE

        # Download and setup Python
        zip_file = download_python_embedded()
        setup_python(zip_file)

        # Copy from venv (with services-specific exclusions)
        if not copy_venv_packages_services():
            return 1
        copy_pyd_files()
        copy_dlls()
        copy_vcredist_dlls()  # Copy VC++ Redist DLLs for PyQt6 on target machines

        # Trim unused Qt6 modules
        cleanup_pyqt6()

        # Restore
        build_portable.DIST_DIR = original_dist

        # Copy online-specific files
        copy_source_files_online()
        copy_server_extras()  # ICU shim, nssm.exe, bat files (SAU cleanup_pyqt6)
        copy_models_online()

        # Write VERSION file (auto from git)
        print("[VER] Writing VERSION file...")
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from core.version import get_version
            version = get_version()
            (DIST_DIR_ONLINE / "VERSION").write_text(version, encoding='utf-8')
            print(f"[OK] VERSION = {version}")
        except Exception as e:
            print(f"[WARN] Cannot determine version: {e}")
            (DIST_DIR_ONLINE / "VERSION").write_text("unknown", encoding='utf-8')

        create_launcher_online()

        # Trim (reuse from desktop build)
        original_dist2 = build_portable.DIST_DIR
        build_portable.DIST_DIR = DIST_DIR_ONLINE
        build_portable.trim_portable()
        build_portable.DIST_DIR = original_dist2

        clean_build()

        # Xóa .opt files — máy target sẽ tự tạo lần đầu chạy (phụ thuộc ORT version + CPU)
        for opt_file in DIST_DIR_ONLINE.rglob("*.opt"):
            opt_file.unlink()
            print(f"  [DEL] {opt_file.relative_to(DIST_DIR_ONLINE)} (ORT cache, auto-generated on target)")

        # Post-build validation
        print()
        print("[CHECK] Validating build...")
        critical = [
            "python/python.exe",
            "python/sitecustomize.py",
            "python/python312._pth",
            "server_launcher.py",
            "server_gui.py",
            "config.ini",
            "ffmpeg.exe",
            "ffprobe.exe",
            "core/__init__.py",
            "core/config.py",
            "core/asr_engine.py",
            "core/speaker_diarization.py",
            "core/speaker_diarization_pure_ort.py",
            "core/speaker_diarization_senko_campp.py",
            "core/speaker_diarization_senko_campp_optimized.py",
            "web_service/__init__.py",
            "web_service/server.py",
            "web_service/config.py",
            "web_service/database.py",
            "web_service/auth.py",
            "web_service/session_manager.py",
            "web_service/queue_manager.py",
            "web_service/ssl_utils.py",
            "web_service/audio_quality.py",
            "web_service/static/index.html",
            "web_service/static/js/app.js",
            "web_service/static/css/style.css",
            "web_service/static/manifest.json",
            "sherpa-vietnamese-asr-service.bat",
        ]
        # Check required packages
        pkg_dir = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages"
        required_pkgs = [
            "fastapi", "uvicorn", "starlette", "sherpa_onnx",
            "numpy", "onnxruntime", "cryptography",
            "jose",
        ]
        missing = []
        for f in critical:
            if not (DIST_DIR_ONLINE / f).exists():
                missing.append(f"  [MISS] {f}")
        for pkg in required_pkgs:
            if not (pkg_dir / pkg).exists():
                missing.append(f"  [MISS] package: {pkg}")

        if missing:
            print("[WARN] Missing files/packages:")
            for m in missing:
                print(m)
        else:
            print("[OK] All critical files and packages present")

        # Report
        total = sum(f.stat().st_size for f in DIST_DIR_ONLINE.rglob("*") if f.is_file())
        size_gb = total / 1024**3

        print()
        print("=" * 60)
        print("BUILD SUCCESS!")
        print("=" * 60)
        print(f"Location: {DIST_DIR_ONLINE.absolute()}")
        print(f"Run:      {DIST_DIR_ONLINE / 'sherpa-vietnamese-asr-service.bat'}")
        print(f"Size:     {size_gb:.2f} GB")

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"BUILD FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
