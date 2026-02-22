#!/usr/bin/env python3
"""
ASR-VN Build Script - Build portable EXE from virtualenv
Usage: python build-portable/build_portable.py

Prerequisites:
    1. python build-portable/setup_build_env.py  # Setup .venv with all dependencies
    2. python build-portable/build_portable.py   # Build portable distribution
"""
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
DIST_DIR = PROJECT_ROOT / "dist" / "Lightweight_ASR"
BUILD_DIR = PROJECT_ROOT / "build"
PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-embed-amd64.zip"

# Source files to copy (relative to project root)
SOURCE_FILES = [
    "app.py", "transcriber.py", "speaker_diarization.py", "sat_segmenter.py",
    "punctuation_restorer_improved.py", "gec_model.py", "modeling_seq2labels.py",
    "configuration_seq2labels.py", "vocabulary.py", "utils.py",
    "common.py", "tab_file.py", "tab_live.py", "audio_analyzer.py",
    "quality_result_dialog.py", "streaming_asr.py", "streaming_asr_online.py",
    "config.ini", "verb-form-vocab.txt", "speaker_hotkeys.json", "ffmpeg.exe", "ffprobe.exe"
]

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
python_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(python_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
site_packages = os.path.join(python_dir, 'Lib', 'site-packages')
if hasattr(os, 'add_dll_directory'):
    for subpath in [
        os.path.join(site_packages, 'numpy', '_core'),
        os.path.join(site_packages, 'torch', 'lib'),
        os.path.join(site_packages, 'sherpa_onnx', 'lib'),
        python_dir,
    ]:
        if os.path.exists(subpath):
            try:
                os.add_dll_directory(subpath)
            except:
                pass
if sys.platform == 'win32':
    paths_to_add = []
    for subpath in [
        os.path.join(site_packages, 'torch', 'lib'),
        os.path.join(site_packages, 'numpy', '_core'),
        os.path.join(site_packages, 'sherpa_onnx', 'lib'),
        python_dir,
        site_packages,
    ]:
        if os.path.exists(subpath):
            paths_to_add.append(subpath)
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
    
    # Packages to exclude
    EXCLUDE_PACKAGES = {'moonshine_voice'}
    
    # Copy all packages
    copied = 0
    for item in venv_site.iterdir():
        if item.name.startswith('~'):
            continue
        if item.name in EXCLUDE_PACKAGES:
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
    
    print(f"[OK] Copied {copied} packages")
    return True


def copy_pyd_files():
    """Copy .pyd C extension files from venv"""
    print("[EXT] Copying C extensions (.pyd)...")
    
    venv_site = get_venv_path()
    dst_site = DIST_DIR / "python" / "Lib" / "site-packages"
    
    copied = 0
    for pyd in venv_site.rglob("*.pyd"):
        # Preserve directory structure
        rel_path = pyd.relative_to(venv_site)
        dst = dst_site / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or pyd.stat().st_size != dst.stat().st_size:
            shutil.copy2(pyd, dst)
            copied += 1
    
    print(f"[OK] Copied {copied} C extensions")


def copy_dlls():
    """Copy DLLs from venv packages"""
    print("[DLL] Copying DLLs...")
    
    venv_site = get_venv_path()
    dst_python = DIST_DIR / "python"
    dst_site = dst_python / "Lib" / "site-packages"
    
    # Copy .libs folders
    for libs_dir in venv_site.rglob("*.libs"):
        dst_libs = dst_site / libs_dir.relative_to(venv_site)
        if dst_libs.exists():
            shutil.rmtree(dst_libs)
        shutil.copytree(libs_dir, dst_libs)
    
    # Copy torch DLLs
    torch_lib = venv_site / "torch" / "lib"
    if torch_lib.exists():
        for dll in torch_lib.glob("*.dll"):
            shutil.copy2(dll, dst_python)
    
    # Copy numpy DLLs
    numpy_libs = venv_site / "numpy.libs"
    if numpy_libs.exists():
        for dll in numpy_libs.glob("*.dll"):
            shutil.copy2(dll, dst_python)
    
    # Copy sherpa-onnx DLLs
    sherpa_lib = venv_site / "sherpa_onnx" / "lib"
    if sherpa_lib.exists():
        for dll in sherpa_lib.glob("*.dll"):
            shutil.copy2(dll, dst_python)
    
    print("[OK] DLLs copied")


def copy_source_files():
    """Copy application source code"""
    print("[SRC] Copying source files...")
    
    for f in SOURCE_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, DIST_DIR)
        else:
            print(f"  [WARN] Not found: {f}")
    
    print("[OK] Source files copied")


def copy_data():
    """Copy data directories (models, vocabulary)"""
    print("[DATA] Copying data directories...")
    
    # Models to exclude
    EXCLUDE_MODELS = {'moonshine-base-vi'}
    
    for dir_name in DATA_DIRS:
        src = PROJECT_ROOT / dir_name
        if src.exists():
            dst = DIST_DIR / dir_name
            if dst.exists():
                shutil.rmtree(dst)
            
            if dir_name == 'models':
                # Copy models but exclude unused ones
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    if item.name in EXCLUDE_MODELS:
                        print(f"  [SKIP] model: {item.name}")
                        continue
                    dst_item = dst / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst_item)
                    else:
                        shutil.copy2(item, dst_item)
            else:
                shutil.copytree(src, dst)
            # Count files
            count = sum(1 for _ in dst.rglob('*') if _.is_file())
            print(f"  [OK] {dir_name}: {count} files")
        else:
            print(f"  [WARN] Not found: {dir_name}")
    
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
set "PATH=%BASE_DIR%python;%BASE_DIR%python\\Lib\\site-packages;%PATH%"

"%PYTHON_EXE%" "%APP_SCRIPT%" %*
exit /b %errorlevel%
'''
    
    launcher_path = DIST_DIR / "Lightweight_ASR.bat"
    launcher_path.write_text(bat_content, encoding='utf-8')
    
    # Create README
    readme = '''ASR-VN Portable
===============

Run: Double-click Lightweight_ASR.bat

Requirements:
- Windows 10/11 64-bit
- No Python installation required

Folder structure:
- python/           : Python embedded runtime
- models/           : AI models
- vocabulary/       : Vocabulary data
- *.py              : Source code
- Lightweight_ASR.bat : Launcher
'''
    (DIST_DIR / "README.txt").write_text(readme, encoding='utf-8')
    
    print(f"[OK] Launcher created: Lightweight_ASR.bat")


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
        
        # Copy source and data
        copy_source_files()
        copy_data()
        
        # Create launcher
        create_launcher()
        
        # Cleanup
        clean_build()
        
        # Report
        print()
        print("="*60)
        print("BUILD SUCCESS!")
        print("="*60)
        print(f"Location: {DIST_DIR.absolute()}")
        print(f"Run:      {DIST_DIR / 'Lightweight_ASR.bat'}")
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
    sys.exit(main())
