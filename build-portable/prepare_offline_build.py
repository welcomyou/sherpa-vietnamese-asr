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
    "sherpa-onnx-pyannote-segmentation-3-0": {
        "type": "github_tar",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "local_dir": "speaker_diarization",
        "description": "Pyannote Segmentation",
        "check_file": "model.onnx",
        "tar_inner_path": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    },
    
    # Pyannote Community-1 Pipeline (Offline - no HF Token needed at runtime)
    "pyannote_community1_pipeline": {
        "type": "huggingface",
        "repo_id": "pyannote/speaker-diarization-community-1",
        "local_dir": "pyannote/speaker-diarization-community-1",
        "description": "Pyannote Community-1 Pipeline (Full Offline)",
        "check_file": "config.yaml"
    },
    "pyannote_segmentation_3.0": {
        "type": "huggingface",
        "repo_id": "pyannote/segmentation-3.0",
        "local_dir": "pyannote/segmentation-3.0",
        "description": "Pyannote Segmentation 3.0 (dependency)",
        "check_file": "pytorch_model.bin"
    },
    "pyannote_wespeaker": {
        "type": "huggingface",
        "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "local_dir": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "description": "WeSpeaker embedding (dependency)",
        "check_file": "pytorch_model.bin"
    },
    
    # Altunenes ONNX Models (Pure ONNX - no pyannote.audio needed)
    "altunenes_segmentation": {
        "type": "huggingface_file",
        "repo_id": "altunenes/speaker-diarization-community-1-onnx",
        "filename": "segmentation-community-1.onnx",
        "local_dir": "pyannote-onnx",
        "description": "Altunenes Segmentation ONNX",
        "check_file": "segmentation-community-1.onnx"
    },
    "altunenes_embedding": {
        "type": "huggingface_file",
        "repo_id": "altunenes/speaker-diarization-community-1-onnx",
        "filename": "embedding_model.onnx",
        "local_dir": "pyannote-onnx",
        "description": "Altunenes Embedding ONNX",
        "check_file": "embedding_model.onnx"
    },
    
    # 3DSpeaker CAM++ (192-dim, dùng bởi speaker_diarization_3dspeaker_campp.py)
    "campp_3dspeaker": {
        "type": "huggingface_file",
        "repo_id": "3D-Speaker/3D-Speaker",
        "filename": "campplus_cn_en_common_200k.onnx",
        "local_dir": "campp-3dspeaker",
        "description": "CAM++ 192-dim Speaker Embedding (3D-Speaker, 27MB)",
        "check_file": "campplus_cn_en_common_200k.onnx"
    },

    # Silero VAD (dùng bởi 3dspeaker_campp diarization)
    "silero_vad": {
        "type": "huggingface_file",
        "repo_id": "snakers4/silero-vad",
        "filename": "files/silero_vad_16k_op15.onnx",
        "local_dir": "silero-vad",
        "description": "Silero VAD 16kHz ONNX (1.3MB)",
        "check_file": "silero_vad_16k_op15.onnx"
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

        # P2 Supply chain: verify SHA-256 nếu config có pin hash
        if config.get("sha256"):
            import hashlib
            h = hashlib.sha256()
            with open(tar_path, "rb") as _f:
                for _chunk in iter(lambda: _f.read(8192), b""):
                    h.update(_chunk)
            got = h.hexdigest()
            if got != config["sha256"]:
                tar_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {model_id}!\n"
                    f"  Expected: {config['sha256']}\n  Got:      {got}\n"
                    "Có thể bị poisoning upstream. KHÔNG sử dụng file này."
                )
            print(f"   ✓ SHA-256 verified: {got[:16]}...")

        # Extract (with path traversal protection)
        print("   Đang giải nén...")
        with tarfile.open(tar_path, 'r:bz2') as tar:
            for member in tar.getmembers():
                member_path = (temp_dir / member.name).resolve()
                if not str(member_path).startswith(str(temp_dir.resolve())):
                    raise ValueError(f"Path traversal detected in archive: {member.name}")
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
        
        tmp_path = str(local_path) + ".tmp"
        urllib.request.urlretrieve(config['url'], tmp_path, reporthook=progress_hook)

        # P2 Supply chain: verify SHA-256 nếu config có pin hash
        if config.get("sha256"):
            import hashlib
            h = hashlib.sha256()
            with open(tmp_path, "rb") as _f:
                for _chunk in iter(lambda: _f.read(8192), b""):
                    h.update(_chunk)
            got = h.hexdigest()
            if got != config["sha256"]:
                import os; os.remove(tmp_path)
                raise ValueError(
                    f"SHA-256 mismatch for {model_id}!\n"
                    f"  Expected: {config['sha256']}\n  Got:      {got}\n"
                    "Có thể bị poisoning upstream. KHÔNG sử dụng file này."
                )
            print(f"\n   ✓ SHA-256 verified: {got[:16]}...")
        else:
            print(f"\n[WARN] {model_id}: chưa có SHA-256 pin — khuyến nghị thêm vào MODELS_CONFIG")

        import os; os.replace(tmp_path, str(local_path))
        print(f"✅ Đã tải xong {model_id}")
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
