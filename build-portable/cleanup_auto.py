#!/usr/bin/env python3
"""
Auto cleanup - Không cần confirm
Usage: python build-portable/cleanup_auto.py
"""

import subprocess
import sys
from pathlib import Path

# Thư viện CHẮC CHẮN KHÔNG DÙNG
UNUSED_PACKAGES = [
    # ASR/AI models không dùng
    "moonshine_voice", "faster_whisper", "chunkformer", "silero_vad", "wespeakerruntime",
    # Deep Learning frameworks thừa
    "lightning", "lightning_fabric", "lightning_utilities", "torchmetrics",
    "pytorch_lightning", "pytorch_metric_learning",
    # Utilities không cần thiết
    "optuna", "tensorboardX", "opentelemetry", "opentelemetry_api",
    "opentelemetry_exporter_otlp_proto_common", "opentelemetry_exporter_otlp_proto_grpc",
    "opentelemetry_exporter_otlp_proto_http", "opentelemetry_exporter_otlp",
    "opentelemetry_proto", "opentelemetry_sdk", "opentelemetry_semantic_conventions",
    # Data processing
    "pandas", "jiwer", "langid", "mosestokenizer", "wtpsplit", "textgrid",
    # CLI/UI
    "colorama", "coloredlogs", "colorlog", "humanfriendly", "prettytable", 
    "pygments", "markdown_it", "mdurl", "rich",
    # Database
    "sqlalchemy", "alembic", "greenlet",
    # Math/Science
    "primePy", "mpmath", "networkx", "skops",
    # Video/Codegen
    "torchgen",
    # Image
    "pillow", "contourpy", "kiwisolver", "fontTools", "cycler", "pyparsing",
    "matplotlib", "mpl_toolkits",
    # Audio augmentation
    "torch_audiomentations", "torch_pitch_shift", "pyrubberband",
    # HTTP/async
    "grpc", "googleapis_common_protos", "grpcio",
    "aiohttp", "aiohappyeyeballs", "aiosignal", "frozenlist", "multidict", "yarl",
    # Dev tools
    "pip", "setuptools", "pyreadline3",
]

def get_venv_pip():
    if sys.platform == 'win32':
        return Path('.envtietkiem/Scripts/pip.exe')
    return Path('.envtietkiem/bin/pip')

def main():
    print("="*70)
    print("AUTO CLEANUP - XOA PACKAGES KHONG DUNG")
    print("="*70)
    print()
    
    pip = get_venv_pip()
    if not pip.exists():
        print("[ERROR] Khong tim thay venv!")
        return 1
    
    print(f"Found pip: {pip}")
    print(f"Packages to remove: {len(UNUSED_PACKAGES)}")
    print()
    
    removed = 0
    failed = []
    
    for pkg in UNUSED_PACKAGES:
        print(f"Removing {pkg}...", end=" ")
        try:
            result = subprocess.run(
                [str(pip), "uninstall", "-y", pkg],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("[OK]")
                removed += 1
            else:
                if "not installed" in result.stderr.lower() or "not found" in result.stderr.lower():
                    print("[NOT INSTALLED]")
                else:
                    print(f"[FAIL: {result.stderr[:50]}]")
                    failed.append(pkg)
        except Exception as e:
            print(f"[ERROR: {e}]")
            failed.append(pkg)
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Removed: {removed}/{len(UNUSED_PACKAGES)}")
    if failed:
        print(f"Failed: {len(failed)} - {', '.join(failed[:5])}")
    print()
    print("Next: python build-portable/test_imports.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
