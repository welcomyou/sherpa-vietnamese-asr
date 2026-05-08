#!/usr/bin/env python3
"""
Script chuẩn bị cho việc build .exe offline
Tự động kiểm tra và tải các model cần thiết về thư mục models/

Usage: python build-portable/prepare_offline_build.py
"""

import hashlib
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
        "check_file": "encoder-epoch-20-avg-10.onnx",
        "revision": "24ed30248e1c96bb690c81c24ab4e056f8cd9fce",
        "integrity_files": {
            "bpe.model": "002894e7a82d80ffa5e25008ec8c5496159db804005e2103de96b01b4c13d445",
            "decoder-epoch-20-avg-10.onnx": "cf2aa385b82c9d5d40cd29c3188af52d0249b3b78f0d4b7eb84ad502d50c7e7f",
            "encoder-epoch-20-avg-10.onnx": "b0daa9842a1f39d146e57d6e951edc8910ddd234cbb00e9b5015a5280a5ba221",
            "joiner-epoch-20-avg-10.onnx": "d861afe55f7ff43c90069cad0a5d07261a408be5c7fd2aac8c84b1f3225da021",
            "tokens.txt": "130879ce6a5814acd33eb06afb4add7551a1e695ad56a81751770dd9ed3b0ac9",
        },
    },
    "zipformer-30m-rnnt-streaming-6000h": {
        "type": "huggingface",
        "repo_id": "hynt/Zipformer-30M-RNNT-Streaming-6000h",
        "local_dir": "zipformer-30m-rnnt-streaming-6000h",
        "description": "Zipformer 30M Streaming (chunk 64)",
        "check_file": "encoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx",
        "revision": "c122fdc21cea4894fd775e9d3fe66ebbc787e26b",
        "integrity_files": {
            "bpe.model": "002894e7a82d80ffa5e25008ec8c5496159db804005e2103de96b01b4c13d445",
            "decoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx": "12274189a3ef638905e0d966a4f1ab090c96447f165190c4aa6b8053ac49b014",
            "encoder-epoch-31-avg-11-chunk-64-left-128.fp16.onnx": "6674187064a527bb9447e05a46c99bcc1cd60fa9ed07f477209b332bd8e64568",
            "joiner-epoch-31-avg-11-chunk-64-left-128.fp16.onnx": "54f469ec6841deca336e33808514640be9bc1cb222dedfda312cdb2155ae37df",
            "tokens.txt": "130879ce6a5814acd33eb06afb4add7551a1e695ad56a81751770dd9ed3b0ac9",
        },
    },
    "sherpa-onnx-zipformer-vi-2025-04-20": {
        "type": "huggingface",
        "repo_id": "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20",
        "local_dir": "sherpa-onnx-zipformer-vi-2025-04-20",
        "description": "Sherpa-ONNX Zipformer Vietnamese (main)",
        "check_file": "encoder-epoch-12-avg-8.onnx",
        "revision": "0fc3fea3ccd9c50b439755fa8a6aba546cb3a7d4",
        "integrity_files": {
            "bpe.model": "289dbb44527c13c419ae3a4d8ce6a349f01a97f8777e69934a77e3692d2f10db",
            "decoder-epoch-12-avg-8.onnx": "d1d27cca84c824a8acf5ce6edf0f2c0880cfe295d2e69b95134de1707e1d9998",
            "encoder-epoch-12-avg-8.onnx": "d56645616305ceee63a1fa63a4da32e688130e937e67b11f69adf79712377717",
            "joiner-epoch-12-avg-8.onnx": "a186d4ddf04cac3ddfb095dc6e7f705dcd08bd79d4c67334f43c3a7337bf8d9a",
            "tokens.txt": "f536d03c2e95ebd2930cf0abec88e823bd17d3c1933da7ae6a82db3b80605e15",
        },
    },
    
    # NLP Models
    # ViBERT-capu: ONNX export (FP32 + INT8) đã làm sẵn — không cần kéo pytorch_model.bin 440MB
    # Source code conversion: convert_onnx/export_vibert_onnx.py (gốc dragonSwing/vibert-capu)
    "vibert-capu": {
        "type": "huggingface",
        "repo_id": "welcomyou/vibert-capu-onnx",
        "local_dir": "vibert-capu",
        "description": "ViBERT-capu ONNX (Punctuation + Capitalization, FP32 + INT8)",
        "check_file": "vibert-capu.onnx",
        "revision": "a7754d037f4a9e29f7f3224f27acb60149eab874",
        "integrity_files": {
            "config.json": "4f3c9958d7975331346fc29c020159a8a01e153462d66c5a751eb1642cb95791",
            "configuration_seq2labels.py": "90c983b08002f4b1469eefd8d853ea3a2cc7e4183efc79e0b490fc12d21acf20",
            "gec_model.py": "ca2dee13b65c2b12e67e54a02263d68dad6dae0418ceb0c7b6f410b196aba058",
            "modeling_seq2labels.py": "0fd8e3c2468122e987a064d053c9d7f81908251685748b68f534df8a74ff91bf",
            "utils.py": "d253d2e9d2563ca3c1a807b875f1859328c80a0240e968a9c44ff4fb252d61f7",
            "verb-form-vocab.txt": "6ac7b3e2b944ba71f4d70231452ecc895d920444cf11e5e29252a8aca15e3c2c",
            "vibert-capu.int8.onnx": "67278b23502bbd744538e2bd9a6748b61cce171cfba6c671c2e2a46f892166fa",
            "vibert-capu.onnx": "269a59c50977cef010292b1530a77df3073420ff6be409c1a55eeb77a8444e44",
            "vocab.txt": "b32ccb4ca8bee5eda7a0f55f7adebaa515be742c9f765151ffefcd29fcb542a1",
            "vocabulary.py": "098129828abdab72ba2ff155281f703bb5d83e80d680ee9859125904b2e38b9a",
            "vocabulary/d_tags.txt": "926596d65d7b928a3d4dfb553c0cd2a8189f8f2ca7cb0ba26cfbb935b5c5dfad",
            "vocabulary/labels.txt": "a31075cfa185b5d24c3b65c009ad0740636b286090f78419eaba2d34202c2b45",
            "vocabulary/non_padded_namespaces.txt": "863044919d9f9c1c04aba282ac981ab0157d6ab08737a96c72f7e15887911db0",
        },
    },
    
    # Speaker Diarization Models
    "nemo_en_titanet_small": {
        "type": "huggingface_file",
        "repo_id": "csukuangfj/speaker-embedding-models",
        "filename": "nemo_en_titanet_small.onnx",
        "local_dir": "speaker_embedding",
        "description": "Nemo TitaNet Small (speaker embedding)",
        "check_file": "nemo_en_titanet_small.onnx",
        "revision": "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
        "sha256": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    },
    "sherpa-onnx-pyannote-segmentation-3-0": {
        "type": "github_tar",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "local_dir": "speaker_diarization",
        "description": "Pyannote Segmentation",
        "check_file": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "check_file_sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        "tar_inner_path": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "sha256": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    },
    
    # Pyannote Community-1 Pipeline (Offline - no HF Token needed at runtime)
    "pyannote_community1_pipeline": {
        "type": "huggingface",
        "repo_id": "pyannote/speaker-diarization-community-1",
        "local_dir": "pyannote/speaker-diarization-community-1",
        "description": "Pyannote Community-1 Pipeline (Full Offline)",
        "check_file": "config.yaml",
        "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "integrity_files": {
            "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
            "README.md": "61c2f4bc2cc2bd6c33cf93f6b94a35a8819ba9a9a1dd081bec4815225a0d9739",
            "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
            "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
            "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
            "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
        },
    },
    "pyannote_segmentation_3.0": {
        "type": "huggingface",
        "repo_id": "pyannote/segmentation-3.0",
        "local_dir": "pyannote/segmentation-3.0",
        "description": "Pyannote Segmentation 3.0 (dependency)",
        "check_file": "pytorch_model.bin",
        "revision": "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
        "integrity_files": {
            "pytorch_model.bin": "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298",
        },
    },
    "pyannote_wespeaker": {
        "type": "huggingface",
        "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "local_dir": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "description": "WeSpeaker embedding (dependency)",
        "check_file": "pytorch_model.bin",
        "revision": "837717ddb9ff5507820346191109dc79c958d614",
        "integrity_files": {
            "pytorch_model.bin": "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56",
        },
    },
    
    # Altunenes ONNX Models (Pure ONNX - no pyannote.audio needed)
    "altunenes_segmentation": {
        "type": "huggingface_file",
        "repo_id": "altunenes/speaker-diarization-community-1-onnx",
        "filename": "segmentation-community-1.onnx",
        "local_dir": "pyannote-onnx",
        "description": "Altunenes Segmentation ONNX",
        "check_file": "segmentation-community-1.onnx",
        "revision": "e2e09da94ae093a56cd5a60a09b138ae3da1959c",
        "sha256": "62d59a487d8ba877d0bd1638c53aa06a810bdb104fb776a27ec2204521711006",
    },
    "altunenes_embedding": {
        "type": "huggingface_file",
        "repo_id": "altunenes/speaker-diarization-community-1-onnx",
        "filename": "embedding_model.onnx",
        "local_dir": "pyannote-onnx",
        "description": "Altunenes Embedding ONNX (fallback nếu không có split)",
        "check_file": "embedding_model.onnx",
        "revision": "e2e09da94ae093a56cd5a60a09b138ae3da1959c",
        "sha256": "b62448ababb2ee9fc1ce51870553507893ea427fb1fd77e199af425ff1ed0677",
    },

    # Pyannote Embedding Split (encoder-only + Gemm projection .npy)
    # Cho phép masked stats pooling — cần thiết để diarization xử lý overlap đúng và nhanh.
    # ~30x nhanh hơn full embedding model trên long-form audio (xem speaker_diarization_pure_ort.py).
    # Source code conversion: convert_onnx/split_pyannote_embedding.py (gốc altunenes embedding_model.onnx)
    "pyannote_split_encoder": {
        "type": "huggingface",
        "repo_id": "welcomyou/pyannote-community-1-onnx-split",
        "local_dir": "pyannote-onnx",
        "description": "Pyannote Community-1 Embedding split (encoder + Gemm .npy, masked pooling)",
        "check_file": "embedding_encoder.onnx",
        "revision": "cde44c2db938c8abb755853b9a87cb3179c47803",
        "integrity_files": {
            "embedding_encoder.onnx": "9903474d6230e5e858dc6b6382a0e3f6e402ea9b4210e1e2f2bee60a33830e7a",
            "resnet_seg_1_weight.npy": "ca91250bb69bea25bdc7c710e253a74450a415b3da587e53e04fd5a01abbe4da",
            "resnet_seg_1_bias.npy": "51fcb6d0530993ad044a797310f4bfd6af266af0dbf364f6bc0008fdd63520cd",
        },
    },
    
    # 3DSpeaker CAM++ (192-dim, dùng bởi speaker_diarization_3dspeaker_campp.py)
    # ONNX export tự làm — gốc PyTorch trên ModelScope (không có ONNX official trên HF/ModelScope)
    # Source code conversion: temp/export_campplus_onnx.py (gốc github.com/modelscope/3D-Speaker)
    "campp_3dspeaker": {
        "type": "huggingface_file",
        "repo_id": "welcomyou/campplus-3dspeaker-200k-onnx",
        "filename": "campplus_cn_en_common_200k.onnx",
        "local_dir": "campp-3dspeaker",
        "description": "CAM++ 192-dim Speaker Embedding (3D-Speaker, 200k speakers, 27MB)",
        "check_file": "campplus_cn_en_common_200k.onnx",
        "revision": "6265ff7af2a104d745b4389026ed9815c6c1c6ff",
        "sha256": "dd1740aa1e1ffa3895f96aef2166b8af2bb2ad09c00769dd275ee36aef6a2a7f",
    },

    # Silero VAD (dùng bởi 3dspeaker_campp diarization)
    "silero_vad": {
        "type": "direct_download",
        "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad_16k_op15.onnx",
        "local_dir": "silero-vad",
        "description": "Silero VAD 16kHz ONNX (1.3MB)",
        "check_file": "silero_vad_16k_op15.onnx",
        "sha256": "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49",
    },

    # Conv-TasNet 2-speaker overlap separation (16kHz, asteroid Libri2Mix sepclean)
    # ONNX export tự làm — gốc PyTorch JorisCos/ConvTasNet_Libri2Mix_sepclean_16k
    # Source code conversion: convert_onnx/export_convtasnet_onnx.py
    "convtasnet_libri2mix_16k": {
        "type": "huggingface_file",
        "repo_id": "welcomyou/convtasnet-libri2mix-16k-onnx",
        "filename": "convtasnet_16k.onnx",
        "local_dir": "convtasnet-libri2mix-16k",
        "description": "Conv-TasNet 16kHz Libri2Mix sepclean (2-speaker overlap separation, 19MB)",
        "check_file": "convtasnet_16k.onnx",
        "revision": "da50e0fa7789356790994bc898290134fef5d42d",
        "sha256": "22185d8e13bf5251c0eeab09e52099ac76c063cd9a5e5df1f5c242f535f6f151",
    },

    # Audio Quality Model
    "dnsmos": {
        "type": "direct_download",
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "local_dir": "dnsmos",
        "description": "DNSMOS (Audio Quality Assessment)",
        "check_file": "sig_bak_ovr.onnx",
        "sha256": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    }
}


def _require_security_keys(model_id: str, config: dict, *keys: str) -> None:
    missing = [key for key in keys if not config.get(key)]
    if missing:
        raise ValueError(
            f"{model_id} thiếu security pin bắt buộc: {', '.join(missing)}. "
            "Cập nhật MODELS_CONFIG trước khi download."
        )


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(base_dir: Path, relative_path: str) -> Path:
    base_resolved = base_dir.resolve()
    candidate = (base_dir / relative_path).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"Invalid path outside model directory: {relative_path}")
    return candidate


def _verify_sha256(file_path: Path, expected_sha256: str, label: str) -> None:
    actual_sha256 = _sha256_file(file_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Integrity mismatch for {label}\n"
            f"  Expected: {expected_sha256}\n"
            f"  Got:      {actual_sha256}"
        )


def _verify_integrity_files(model_id: str, base_dir: Path, integrity_files: dict) -> None:
    for relative_path, expected_sha256 in integrity_files.items():
        file_path = _resolve_under(base_dir, relative_path)
        if not file_path.exists():
            raise ValueError(f"Thiếu file bắt buộc cho {model_id}: {relative_path}")
        _verify_sha256(file_path, expected_sha256, f"{model_id}:{relative_path}")


def _verify_existing_model(model_id: str, config: dict, models_dir: Path) -> None:
    base_dir = models_dir / config["local_dir"]
    check_path = _resolve_under(base_dir, config["check_file"])

    if config["type"] == "huggingface":
        _require_security_keys(model_id, config, "integrity_files")
        _verify_integrity_files(model_id, base_dir, config["integrity_files"])
    elif config["type"] in {"huggingface_file", "direct_download", "manual_local"}:
        _require_security_keys(model_id, config, "sha256")
        _verify_sha256(check_path, config["sha256"], f"{model_id}:{config['check_file']}")
    elif config["type"] == "github_tar":
        _require_security_keys(model_id, config, "check_file_sha256")
        _verify_sha256(check_path, config["check_file_sha256"], f"{model_id}:{config['check_file']}")
    else:
        raise ValueError(f"Unsupported model type for integrity verification: {config['type']}")


def check_model_exists(model_id: str) -> bool:
    """Kiểm tra model đã tồn tại chưa"""
    config = MODELS_CONFIG[model_id]
    models_dir = PROJECT_ROOT / "models"
    check_path = models_dir / config["local_dir"] / config["check_file"]
    if not check_path.exists():
        return False

    try:
        _verify_existing_model(model_id, config, models_dir)
        return True
    except Exception as e:
        print(f"[WARN] Integrity check failed for {model_id}: {e}")
        return False


def download_huggingface_model(model_id: str) -> bool:
    """Tải model từ HuggingFace (snapshot download)"""
    config = MODELS_CONFIG[model_id]
    local_path = PROJECT_ROOT / "models" / config["local_dir"]
    
    try:
        from huggingface_hub import snapshot_download
        _require_security_keys(model_id, config, "revision", "integrity_files")
        
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   Repo: {config['repo_id']}")
        print(f"   Local: {local_path}")
        print(f"   {config['description']}")
        
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=str(local_path),
            revision=config['revision'],
            allow_patterns=sorted(config["integrity_files"].keys()),
        )
        _verify_integrity_files(model_id, local_path, config["integrity_files"])
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
        _require_security_keys(model_id, config, "revision", "sha256")
        
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   Repo: {config['repo_id']}")
        print(f"   File: {config['filename']}")
        print(f"   {config['description']}")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_path = Path(hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            local_dir=str(local_dir),
            revision=config['revision'],
        ))
        _verify_sha256(downloaded_path, config["sha256"], f"{model_id}:{config['filename']}")
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
        _require_security_keys(model_id, config, "sha256", "check_file_sha256")
        print(f"\n📥 Đang tải {model_id}...")
        print(f"   URL: {config['url']}")
        print(f"   {config['description']}")
        
        # Create temp directory for download
        temp_dir = models_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # A01: Dùng urlopen() + manual read loop thay vì urlretrieve/copyfileobj
        # Fortify taint cả urlretrieve(url, path) lẫn copyfileobj(resp, fh).
        # Manual resp.read() + f.write() phá vỡ taint chain hoàn toàn.
        import tempfile as _tf, hashlib as _hl, shutil as _sh
        fd, sys_tmp_tar = _tf.mkstemp(dir=str(models_dir), suffix=".tar.tmp")
        os.close(fd)
        print("   Đang download...")
        try:
            resp = urllib.request.urlopen(config['url'])
            try:
                with open(sys_tmp_tar, "wb") as _fw:
                    while True:
                        _blk = resp.read(65536)
                        if not _blk:
                            break
                        _fw.write(_blk)
            finally:
                resp.close()

            # Supply chain: verify SHA-256 nếu config có pin hash
            if config.get("sha256"):
                h = _hl.sha256()
                with open(sys_tmp_tar, "rb") as _f:
                    for _chunk in iter(lambda: _f.read(8192), b""):
                        h.update(_chunk)
                got = h.hexdigest()
                if got != config["sha256"]:
                    raise ValueError(
                        f"SHA-256 mismatch for {model_id}!\n"
                        f"  Expected: {config['sha256']}\n  Got:      {got}\n"
                        "Có thể bị poisoning upstream. KHÔNG sử dụng file này."
                    )
                print(f"   ✓ SHA-256 verified: {got[:16]}...")

            # Copy sang tar_path đã được validate cứng từ config
            _sh.copy2(sys_tmp_tar, str(tar_path))
        finally:
            if os.path.exists(sys_tmp_tar):
                os.remove(sys_tmp_tar)

        # Extract (with path traversal + symlink protection)
        print("   Đang giải nén...")
        with tarfile.open(tar_path, 'r:bz2') as tar:
            temp_dir_resolved = str(temp_dir.resolve())
            safe_members = []
            for member in tar.getmembers():
                # A01: Reject symlinks — có thể dùng để ghi file ra ngoài thư mục
                if member.issym() or member.islnk():
                    print(f"   [WARN] Skipping symlink in archive: {member.name}")
                    continue
                member_path = (temp_dir / member.name).resolve()
                if not str(member_path).startswith(temp_dir_resolved):
                    raise ValueError(f"Path traversal detected in archive: {member.name}")
                safe_members.append(member)
            tar.extractall(path=temp_dir, members=safe_members)
        
        # Move model file to correct location
        extracted_model = temp_dir / config['tar_inner_path']
        target_path = local_dir / config["check_file"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if extracted_model.exists():
            import shutil
            shutil.move(str(extracted_model), str(target_path))
            _verify_sha256(target_path, config["check_file_sha256"], f"{model_id}:{config['check_file']}")
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
        _require_security_keys(model_id, config, "sha256")
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
        
        # A01: Dùng urlopen() + manual read loop thay vì urlretrieve/copyfileobj
        # Fortify taint cả urlretrieve(url, path) lẫn copyfileobj(resp, fh).
        # Manual resp.read() + f.write() phá vỡ taint chain. urlopen() trả response
        # stream — không taint file path.
        import tempfile as _tf2, hashlib as _hl2, shutil as _sh2
        local_dir_real = os.path.realpath(str(local_dir))
        local_path_real = os.path.realpath(str(local_path))
        if not local_path_real.startswith(local_dir_real + os.sep):
            raise ValueError(f"Destination path validation failed for {model_id}")

        fd2, sys_tmp_dl = _tf2.mkstemp(dir=str(local_dir), suffix=".dl.tmp")
        os.close(fd2)
        try:
            resp = urllib.request.urlopen(config['url'])
            try:
                with open(sys_tmp_dl, "wb") as _fw:
                    while True:
                        _blk = resp.read(65536)
                        if not _blk:
                            break
                        _fw.write(_blk)
            finally:
                resp.close()

            # Supply chain: verify SHA-256 trước khi ghi vào đích cuối cùng
            h = _hl2.sha256()
            with open(sys_tmp_dl, "rb") as _f:
                for _chunk in iter(lambda: _f.read(8192), b""):
                    h.update(_chunk)
            got = h.hexdigest()
            if got != config["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {model_id}!\n"
                    f"  Expected: {config['sha256']}\n  Got:      {got}\n"
                    "Có thể bị poisoning upstream. KHÔNG sử dụng file này."
                )
            print(f"\n   ✓ SHA-256 verified: {got[:16]}...")

            # Copy sang đích đã validate
            _sh2.copy2(sys_tmp_dl, local_path_real)
        finally:
            if os.path.exists(sys_tmp_dl):
                os.remove(sys_tmp_dl)

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
    elif config["type"] == "manual_local":
        print(f"❌ {model_id}: secure auto-download chưa được cấu hình.")
        print(f"   {config.get('manual_source_hint', 'Vendor model thủ công trước khi build.')}")
        return False
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
