#!/usr/bin/env python3
"""
Build script cho ban sherpa-vietnamese-asr-service (web service + admin GUI).
Dua tren build_portable.py nhung them web_service/ va loai tru streaming model.

Usage: python build-portable/build_portable_online.py
"""
import os
import sys
import shutil
import stat
from pathlib import Path

# Reuse logic tu build_portable.py
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Import cac ham tu build_portable.py
sys.path.insert(0, str(SCRIPT_DIR))
from build_portable import (
    VENV_DIR, BUILD_DIR, PYTHON_EMBED_URL,
    check_venv, download_python_embedded, setup_python,
    copy_pyd_files, copy_dlls,
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
    "zipformer-30m-rnnt-streaming-6000h",  # Khong can streaming
}

# Packages khong can cho web service (giam dung luong)
EXCLUDE_PACKAGES_SERVICES = {
    'moonshine_voice',
    # Desktop-only packages
    'sounddevice', '_sounddevice_data',
    'pyinstaller', 'pyinstaller_hooks_contrib', 'altgraph', 'pefile',
    'matplotlib', 'mpl_toolkits', 'kiwisolver', 'contourpy', 'cycler',
    'pyparsing',
    # Not used by web service
    'ctranslate2', 'pandas', 'llvmlite', 'numba',
    'av', 'networkx', 'sympy', 'mpmath',
    'adapters', 'torio', 'torchcodec', 'rapidfuzz',
    'alembic',
    # Build / packaging tools
    'pip', 'setuptools', 'wheel', 'pkg_resources', '_distutils_hack',
    'distutils_precedence',
}


def should_exclude_services(name):
    """Check if package or its dist-info should be excluded for service build"""
    lower = name.lower()
    for pkg in EXCLUDE_PACKAGES_SERVICES:
        if lower == pkg or lower.startswith(pkg + '-') or lower.startswith(pkg.replace('-', '_') + '-'):
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


def copy_models_online():
    """Copy models, loai tru streaming model"""
    print("[MODEL] Copying models (excluding streaming)...")

    models_src = PROJECT_ROOT / "models"
    models_dst = DIST_DIR_ONLINE / "models"

    if not models_src.exists():
        print("  [WARN] models/ not found")
        return

    models_dst.mkdir(parents=True, exist_ok=True)

    for item in models_src.iterdir():
        if item.name in EXCLUDE_MODELS_ONLINE:
            print(f"  [SKIP] {item.name}")
            continue

        dst_item = models_dst / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)
        print(f"  [OK] {item.name}")

    count = sum(1 for _ in models_dst.rglob("*") if _.is_file())
    print(f"[OK] Models copied: {count} files")


def create_launcher_online():
    """Tao launcher batch cho ban services"""
    print("[LNCH] Creating service launcher...")

    # Server launcher
    bat_content = '''@echo off
chcp 65001 >nul
setlocal

set "BASE_DIR=%~dp0"
set "PYTHON_EXE=%BASE_DIR%python\\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Khong tim thay Python embedded
    pause
    exit /b 1
)

set "PYTHONHOME=%BASE_DIR%python"
set "PYTHONDONTWRITEBYTECODE=1"
set "PATH=%BASE_DIR%python;%BASE_DIR%python\\Lib\\site-packages;%BASE_DIR%;%PATH%"

echo ===================================
echo  Sherpa Vietnamese ASR - Service
echo ===================================
echo.

if "%1"=="--no-gui" (
    echo Starting server (no GUI)...
    "%PYTHON_EXE%" "%BASE_DIR%server_launcher.py" --no-gui %*
) else (
    echo Starting admin GUI...
    "%PYTHON_EXE%" "%BASE_DIR%server_gui.py" %*
)

exit /b %errorlevel%
'''

    (DIST_DIR_ONLINE / "sherpa-vietnamese-asr-service.bat").write_text(bat_content, encoding="utf-8")

    # README
    readme = '''Sherpa Vietnamese ASR - Service
================================

Chay: Double-click sherpa-vietnamese-asr-service.bat

Cac che do chay:
1. sherpa-vietnamese-asr-service.bat         -> Mo GUI quan tri (PyQt6)
2. sherpa-vietnamese-asr-service.bat --no-gui -> Chay server khong co GUI

Yeu cau:
- Windows Server 2022 / Windows 10/11 64-bit
- Khong can cai Python

Thu muc:
- python/           : Python embedded runtime
- models/           : AI models (khong co streaming model)
- core/             : Core ASR pipeline (dung chung voi desktop)
- web_service/      : FastAPI web service + frontend
- server_gui.py     : Admin GUI (PyQt6)
- server_launcher.py: Server entry point

Sau khi chay:
- Truy cap https://IP:8443 tren browser
- Admin mac dinh: admin / admin (doi ngay sau khi dang nhap)
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

        # Restore
        build_portable.DIST_DIR = original_dist

        # Copy online-specific files
        copy_source_files_online()
        copy_models_online()
        create_launcher_online()

        clean_build()

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
            "web_service/__init__.py",
            "web_service/server.py",
            "web_service/config.py",
            "web_service/database.py",
            "web_service/auth.py",
            "web_service/session_manager.py",
            "web_service/queue_manager.py",
            "web_service/ssl_utils.py",
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
            "torch", "numpy", "cryptography",
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
