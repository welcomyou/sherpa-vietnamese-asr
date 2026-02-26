#!/usr/bin/env python3
"""
Script chuẩn bị cho việc build .exe offline
Tự động kiểm tra và tải các model cần thiết về thư mục models/

Usage: python build-portable/prepare_offline_build.py
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Get project root (parent of build-portable directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Model configurations
MODELS_CONFIG = {
    # ASR Models
    "zipformer-30m-rnnt-6000h": {
        "type": "huggingface",
        "repo_id": "hynt/Zipformer-30M-RNNT-6000h",
        "local_dir": "zipformer-30m-rnnt-6000h",
        "description": "Zipformer 30M RNNT (lightweight)",
        "check_file": "encoder-epoch-20-avg-10.onnx"
    },
    "zipformer-30m-rnnt-streaming-6000h": {
        "type": "huggingface",
        "repo_id": "hynt/Zipformer-30M-RNNT-Streaming-6000h",
        "local_dir": "zipformer-30m-rnnt-streaming-6000h",
        "description": "Zipformer 30M Streaming (chunk 64)",
        "check_file": "encoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx"
    },
    "sherpa-onnx-zipformer-vi-2025-04-20": {
        "type": "huggingface",
        "repo_id": "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20",
        "local_dir": "sherpa-onnx-zipformer-vi-2025-04-20",
        "description": "Sherpa-ONNX Zipformer Vietnamese (main)",
        "check_file": "encoder-epoch-12-avg-8.onnx"
    },
    
    # NLP Models
    "vibert-capu": {
        "type": "huggingface",
        "repo_id": "dragonSwing/vibert-capu",
        "local_dir": "vibert-capu",
        "description": "ViBERT-capu (Punctuation Restoration)",
        "check_file": "pytorch_model.bin"
    },
    
    # Speaker Diarization Models
    "nemo_en_titanet_small": {
        "type": "huggingface_file",
        "repo_id": "csukuangfj/speaker-embedding-models",
        "filename": "nemo_en_titanet_small.onnx",
        "local_dir": "speaker_embedding",
        "description": "Nemo TitaNet Small (speaker embedding)",
        "check_file": "nemo_en_titanet_small.onnx"
    },
    "eres2netv2_zh": {
        "type": "huggingface_file",
        "repo_id": "csukuangfj/speaker-embedding-models",
        "filename": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        "local_dir": "speaker_embedding",
        "description": "3D Speaker ERes2NetV2 (ZH+EN)",
        "check_file": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
    },
    "sherpa-onnx-pyannote-segmentation-3-0": {
        "type": "github_tar",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "local_dir": "speaker_diarization",
        "description": "Pyannote Segmentation",
        "check_file": "model.onnx",
        "tar_inner_path": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    },
    
    # Audio Quality Model
    "dnsmos": {
        "type": "direct_download",
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "local_dir": "dnsmos",
        "description": "DNSMOS (Audio Quality Assessment)",
        "check_file": "sig_bak_ovr.onnx"
    }
}


def check_model_exists(model_id: str) -> bool:
    """Kiểm tra model đã tồn tại chưa"""
    config = MODELS_CONFIG[model_id]
    models_dir = PROJECT_ROOT / "models"
    check_path = models_dir / config["local_dir"] / config["check_file"]
    return check_path.exists()


def download_huggingface_model(model_id: str) -> bool:
    """Tải model từ HuggingFace (snapshot download)"""
    config = MODELS_CONFIG[model_id]
    local_path = PROJECT_ROOT / "models" / config["local_dir"]
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   Repo: {config['repo_id']}")
        print(f"   Local: {local_path}")
        print(f"   {config['description']}")
        
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=str(local_path),
        )
        print(f"✅ Đã tải xong {model_id}")
        return True
        
    except ImportError:
        print("❌ Lỗi: Chưa cài huggingface_hub")
        print("   Cài đặt: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi tải {model_id}: {e}")
        return False


def download_huggingface_file(model_id: str) -> bool:
    """Tải file đơn lẻ từ HuggingFace"""
    config = MODELS_CONFIG[model_id]
    local_dir = PROJECT_ROOT / "models" / config["local_dir"]
    
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   Repo: {config['repo_id']}")
        print(f"   File: {config['filename']}")
        print(f"   {config['description']}")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            local_dir=str(local_dir),
        )
        print(f"✅ Đã tải xong {model_id}")
        return True
        
    except ImportError:
        print("❌ Lỗi: Chưa cài huggingface_hub")
        print("   Cài đặt: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi tải {model_id}: {e}")
        return False


def download_github_tar(model_id: str) -> bool:
    """Tải và giải nén tar.bz2 từ GitHub releases"""
    config = MODELS_CONFIG[model_id]
    models_dir = PROJECT_ROOT / "models"
    local_dir = models_dir / config["local_dir"]
    tar_path = models_dir / "temp_download.tar.bz2"
    
    try:
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   URL: {config['url']}")
        print(f"   {config['description']}")
        
        # Create temp directory for download
        temp_dir = models_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tar file
        print("   Đang download...")
        urllib.request.urlretrieve(config['url'], tar_path)
        
        # Extract
        print("   Đang giải nén...")
        with tarfile.open(tar_path, 'r:bz2') as tar:
            tar.extractall(path=temp_dir)
        
        # Move model file to correct location
        extracted_model = temp_dir / config['tar_inner_path']
        target_path = local_dir / "model.onnx"
        
        if extracted_model.exists():
            import shutil
            shutil.move(str(extracted_model), str(target_path))
            print(f"✅ Đã tải xong {model_id}")
        else:
            print(f"❌ Không tìm thấy file trong archive: {config['tar_inner_path']}")
            return False
        
        # Cleanup
        tar_path.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tải {model_id}: {e}")
        tar_path.unlink(missing_ok=True)
        return False


def download_direct(model_id: str) -> bool:
    """Tải file trực tiếp từ URL"""
    config = MODELS_CONFIG[model_id]
    local_dir = PROJECT_ROOT / "models" / config["local_dir"]
    local_path = local_dir / config["check_file"]
    
    try:
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   URL: {config['url']}")
        print(f"   {config['description']}")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            if block_num % 10 == 0:  # Update every 10 blocks
                print(f"   Progress: {percent}%", end='\r')
        
        urllib.request.urlretrieve(config['url'], local_path, reporthook=progress_hook)
        print(f"\n✅ Đã tải xong {model_id}")
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tải {model_id}: {e}")
        return False


def download_model(model_id: str) -> bool:
    """Tải model theo loại"""
    config = MODELS_CONFIG[model_id]
    
    if config["type"] == "huggingface":
        return download_huggingface_model(model_id)
    elif config["type"] == "huggingface_file":
        return download_huggingface_file(model_id)
    elif config["type"] == "github_tar":
        return download_github_tar(model_id)
    elif config["type"] == "direct_download":
        return download_direct(model_id)
    else:
        print(f"❌ Không xác định được loại model: {config['type']}")
        return False


def main():
    """Main function - kiểm tra và tải tất cả models"""
    print("=" * 70)
    print("KIỂM TRA VÀ TẢI MODELS CHO BUILD .EXE OFFLINE")
    print("=" * 70)
    
    # Kiểm tra xem có thiếu model nào không
    missing_models = []
    for model_id in MODELS_CONFIG:
        if check_model_exists(model_id):
            config = MODELS_CONFIG[model_id]
            print(f"✅ {model_id}: Đã có ({config['description']})")
        else:
            missing_models.append(model_id)
            print(f"❌ {model_id}: Chưa có - Sẽ tải về")
    
    if not missing_models:
        print("\n" + "=" * 70)
        print("✅ Tất cả model đã sẵn sàng!")
        print("=" * 70)
        return True
    
    print(f"\n{'=' * 70}")
    print(f"CẦN TẢI {len(missing_models)} MODELS")
    print(f"{'=' * 70}\n")
    
    # Tải các model còn thiếu
    results = {}
    for model_id in missing_models:
        results[model_id] = download_model(model_id)
    
    # Tổng kết
    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    
    success_count = sum(1 for v in results.values() if v)
    failed_count = len(results) - success_count
    
    for model_id, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {model_id}")
    
    print(f"\n   Thành công: {success_count}/{len(results)}")
    if failed_count > 0:
        print(f"   Thất bại: {failed_count}")
    print("=" * 70)
    
    return failed_count == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
