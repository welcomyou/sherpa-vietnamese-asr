#!/usr/bin/env python3
"""
Script dọn dẹp các thư viện không dùng trong envtietkiem
GIẢM DUNG LƯỢNG build portable

Usage: python build-portable/cleanup_unused_packages.py
"""

import subprocess
import sys
from pathlib import Path

# Thư viện CHẮC CHẮN KHÔNG DÙNG (có thể xóa an toàn)
UNUSED_PACKAGES = [
    # ASR/AI models không dùng
    "moonshine_voice",           # Đã exclude, dùng sherpa-onnx
    "faster_whisper",            # Không dùng Whisper
    "chunkformer",               # Chỉ dùng trong test
    "silero_vad",                # Không dùng VAD này
    "wespeakerruntime",          # Không dùng WeSpeaker runtime
    
    # Deep Learning frameworks thừa
    "pytorch_lightning",         # Không dùng Lightning
    "pytorch_metric_learning",   # Không dùng metric learning
    "lightning",                 # PyTorch Lightning core
    "lightning_fabric",          # Lightning fabric
    "lightning_utilities",       # Lightning utilities
    "torchmetrics",              # Metrics (có thể là dependency)
    
    # Utilities không cần thiết
    "optuna",                    # Hyperparameter tuning
    "tensorboardX",              # Logging
    "opentelemetry",             # Tracing/telemetry
    "opentelemetry_api",
    "opentelemetry_exporter_otlp_proto_common",
    "opentelemetry_exporter_otlp_proto_grpc",
    "opentelemetry_exporter_otlp_proto_http",
    "opentelemetry_exporter_otlp",
    "opentelemetry_proto",
    "opentelemetry_sdk",
    "opentelemetry_semantic_conventions",
    
    # Data processing không dùng
    "pandas",                    # Có thể không cần, dùng numpy đủ
    "pyarrow",                   # Arrow format (nếu có)
    
    # Text processing không dùng
    "jiwer",                     # WER/CER metrics (chỉ để đánh giá)
    "langid",                    # Language detection
    "mosestokenizer",            # Moses tokenizer
    "wtpsplit",                  # Text splitting
    "textgrid",                  # TextGrid format
    
    # CLI/UI không cần
    "colorama",                  # Terminal colors (Windows)
    "coloredlogs",               # Colored logging
    "colorlog",                  # Color logging
    "humanfriendly",             # Friendly CLI
    "prettytable",               # CLI tables
    "pygments",                  # Syntax highlighting
    "markdown_it",               # Markdown parser
    "mdurl",                     # URL parser cho markdown
    "rich",                      # Rich terminal output
    
    # Database ORM (chỉ là dependency)
    "sqlalchemy",                # ORM
    "alembic",                   # DB migration
    "greenlet",                  # Coroutine (SQLAlchemy dep)
    
    # Math/Science không dùng trực tiếp
    "primePy",                   # Prime numbers
    "mpmath",                    # Arbitrary precision math
    "networkx",                  # Graph processing (có thể là sklearn dep)
    "skops",                     # sklearn ops
    
    # Video không dùng
    "torchcodec",                # Video decoding
    "torchgen",                  # Codegen
    
    # Image processing không dùng
    "PIL",                       # Pillow - image processing
    "pillow",
    "contourpy",                 # Matplotlib dep (nếu không cần plot)
    "kiwisolver",                # Matplotlib dep
    "fontTools",                 # Font tools (matplotlib dep)
    "cycler",                    # Matplotlib dep
    "pyparsing",                 # Parsing (matplotlib dep)
    "matplotlib",                # Plotting (XEM XÉT - có thể cần)
    "mpl_toolkits",              # Matplotlib toolkits
    
    # Audio augmentation không dùng
    "torch_audiomentations",     # Audio augmentation
    "torch_pitch_shift",         # Pitch shifting
    "pyrubberband",              # Time-stretching (có thể dùng trong test)
    
    # HTTP/async không cần (dùng requests đủ)
    "grpc",                      # gRPC
    "googleapis_common_protos",
    "grpcio",
    "aiohttp",                   # Async HTTP
    "aiohappyeyeballs",
    "aiosignal",
    "frozenlist",
    "multidict",
    "yarl",
    "async_timeout",             # Nếu có
    
    # Development tools không cần
    "pip",                       # Package manager
    "setuptools",                # Build tools
    "pyreadline3",               # Readline (Windows)
]

# Thư viện CÓ THỂ XÓA (tùy chọn, cần test kỹ)
OPTIONAL_PACKAGES = [
    "scipy",                     # DÙNG trong sklearn AHC - KHÔNG XÓA
    "numba",                     # JIT compiler (librosa speedup) - NÊN GIỮ
    "joblib",                    # Parallel processing (sklearn dep)
    "threadpoolctl",             # Thread control (sklearn dep)
    "audioread",                 # Audio reading (librosa backend)
    "pooch",                     # Data download (librosa dep)
    "soxr",                      # High quality resampling (librosa)
    "lazy_loader",               # Lazy loading
    "platformdirs",              # Platform directories
    "packaging",                 # Version handling
    "filelock",                  # File locking - DÙNG trong vocabulary.py
]

def get_venv_python():
    """Get Python path in venv"""
    if sys.platform == 'win32':
        return Path(".envtietkiem/Scripts/python.exe")
    return Path(".envtietkiem/bin/python")

def get_venv_pip():
    """Get pip path in venv"""
    if sys.platform == 'win32':
        return Path(".envtietkiem/Scripts/pip.exe")
    return Path(".envtietkiem/bin/pip")

def check_package_exists(package):
    """Check if package is installed"""
    try:
        result = subprocess.run(
            [str(get_venv_python()), "-c", f"import {package.replace('-', '_')}"],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except:
        return False

def get_package_size(package):
    """Get approximate package size"""
    try:
        result = subprocess.run(
            [str(get_venv_python()), "-c", 
             f"import {package.replace('-', '_')}; import os; p = {package.replace('-', '_')}.__path__[0]; print(sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(p) for f in files))"],
            capture_output=True,
            text=True,
            check=False
        )
        return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    except:
        return 0

def uninstall_package(package):
    """Uninstall a package"""
    try:
        result = subprocess.run(
            [str(get_venv_pip()), "uninstall", "-y", package],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"    Error uninstalling {package}: {e}")
        return False

def main():
    print("="*70)
    print("CLEANUP UNUSED PACKAGES FROM .envtietkiem")
    print("="*70)
    print()
    
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"[ERROR] Virtual environment not found at: {venv_python}")
        print("Please run: python build-portable/setup_build_env.py")
        return 1
    
    print(f"Found venv: {venv_python.parent.parent}")
    print()
    
    # Check which packages exist
    print("[1/3] Checking installed packages...")
    existing_packages = []
    for pkg in UNUSED_PACKAGES:
        if check_package_exists(pkg):
            size = get_package_size(pkg)
            existing_packages.append((pkg, size))
            print(f"  [OK] {pkg}: {size/1024/1024:.1f} MB")
    
    if not existing_packages:
        print("  No unused packages found!")
        return 0
    
    total_size = sum(size for _, size in existing_packages)
    print(f"\n  Found {len(existing_packages)} packages to remove")
    print(f"  Estimated space to free: {total_size/1024/1024:.1f} MB")
    print()
    
    # Ask for confirmation
    print("[2/3] Ready to uninstall")
    print("WARNING: This will permanently remove packages from .envtietkiem")
    print()
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Cancelled.")
        return 0
    
    # Uninstall
    print()
    print("[3/3] Uninstalling packages...")
    removed = []
    failed = []
    
    for pkg, size in existing_packages:
        print(f"  Removing {pkg}...", end=" ")
        if uninstall_package(pkg):
            print(f"[OK] ({size/1024/1024:.1f} MB)")
            removed.append((pkg, size))
        else:
            print("✗ FAILED")
            failed.append(pkg)
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Removed: {len(removed)}/{len(existing_packages)} packages")
    print(f"Space freed: {sum(s for _, s in removed)/1024/1024:.1f} MB")
    
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    print()
    print("Next step:")
    print("  python build-portable/build_portable.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
