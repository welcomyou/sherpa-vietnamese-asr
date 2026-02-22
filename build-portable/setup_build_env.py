#!/usr/bin/env python3
"""
Setup isolated build environment using virtualenv
Usage: python build-portable/setup_build_env.py
"""
import subprocess
import sys
import os
from pathlib import Path

# Get project root (parent of build-portable directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
VENV_DIR = PROJECT_ROOT / ".envtietkiem"


def create_venv():
    """Create virtual environment"""
    print("[1/4] Creating virtual environment...")
    if VENV_DIR.exists():
        print("      Removing old venv...")
        import shutil
        shutil.rmtree(VENV_DIR)
    
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    print("      ✓ Virtual environment created")


def get_venv_python():
    """Get Python executable path in venv"""
    if sys.platform == 'win32':
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_packages():
    """Install required packages"""
    print("[2/4] Installing packages...")
    venv_python = get_venv_python()
    
    # Core packages
    packages = [
        "numpy==2.3.5",  # Pin version for compatibility
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "soundfile",
        "PyQt6",
        "psutil",
        "sentencepiece",
        "onnxruntime",
        "requests",
        "tqdm",
        "huggingface_hub",
        "scipy",
        "numba",
        "wtpsplit",
        "sentence-transformers",
        "rapidfuzz",
    ]
    
    # Upgrade pip first
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], 
                   capture_output=True)
    
    # Install packages
    for pkg in packages:
        print(f"      Installing {pkg}...")
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"      ⚠ Failed: {pkg}")
        else:
            print(f"      ✓ {pkg}")
    
    print("      ✓ All packages installed")


def install_sherpa():
    """Install sherpa-onnx from local wheel or download"""
    print("[3/4] Installing sherpa-onnx...")
    venv_python = get_venv_python()
    
    # Find sherpa wheel in project root
    wheels = list(PROJECT_ROOT.glob("sherpa*.whl"))
    if wheels:
        wheel = wheels[0]
        print(f"      Installing from {wheel.name}...")
        subprocess.run([str(venv_python), "-m", "pip", "install", str(wheel)], check=True)
        print("      ✓ sherpa-onnx installed")
    else:
        print("      ⚠ sherpa wheel not found, downloading...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "sherpa-onnx"])


def test_imports():
    """Test if all imports work"""
    print("[4/4] Testing imports...")
    venv_python = get_venv_python()
    
    test_code = """
import numpy
import torch
import transformers
import librosa
import soundfile
import PyQt6
import psutil
import sherpa_onnx
import wtpsplit
print("All imports OK!")
print(f"numpy: {numpy.__version__}")
print(f"torch: {torch.__version__}")
print(f"sherpa_onnx: {hasattr(sherpa_onnx, 'OfflineRecognizer')}")
"""
    result = subprocess.run([str(venv_python), "-c", test_code], 
                           capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Errors:", result.stderr)


def main():
    print("="*60)
    print("SETUP BUILD ENVIRONMENT")
    print("="*60)
    print()
    
    try:
        create_venv()
        install_packages()
        install_sherpa()
        test_imports()
        
        print()
        print("="*60)
        print("✓ Setup complete!")
        print("="*60)
        print(f"Python: {get_venv_python()}")
        print()
        print("Next step:")
        print("  python build-portable/build_portable.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
