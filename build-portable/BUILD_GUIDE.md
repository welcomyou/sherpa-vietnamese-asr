# Hướng dẫn Build ASR-VN Portable

## Yêu cầu

- Windows 10/11 64-bit
- Python 3.12 (để chạy build scripts)
- Internet (để tải dependencies và models lần đầu)
- HuggingFace Token (để tải pyannote models - chỉ cần 1 lần)

## Các bước build

### Bước 1: Setup môi trường build

```bash
python build-portable/setup_build_env.py
```

Script này sẽ:
- Tạo virtual environment `.envtietkiem`
- Cài đặt tất cả dependencies cần thiết:
  - `pyannote.audio` - Community-1 PyTorch backend
  - `kaldi_native_fbank` - Feature extraction cho ONNX
  - `pydub` - Audio format conversion
  - `scikit-learn` - AHC clustering cho ONNX
  - `filelock` - Thread-safe vocabulary
  - Và các thư viện khác...

### Bước 2: Tải models (Offline preparation)

```bash
# Nếu có HuggingFace Token (khuyến nghị)
set HF_TOKEN=your_token_here
python build-portable/prepare_offline_build.py

# Hoặc không cần token (chỉ tải models không cần auth)
python build-portable/prepare_offline_build.py
```

Script này sẽ tải về:
- ASR Models: zipformer-30m-rnnt-6000h, zipformer-30m-rnnt-streaming-6000h
- Punctuation: vibert-capu
- Speaker Diarization:
  - nemo_en_titanet_small.onnx (sherpa-onnx backend)
  - sherpa-onnx-pyannote-segmentation-3-0
  - **pyannote/speaker-diarization-community-1** (full pipeline)
  - **pyannote/segmentation-3.0** (dependency)
  - **pyannote/wespeaker-voxceleb-resnet34-LM** (dependency)
  - **altunenes/speaker-diarization-community-1-onnx** (ONNX models)
- Audio Quality: dnsmos

### Bước 3 (Tùy chọn): Dọn dẹp thư viện không dùng

Giảm dung lượng build bằng cách xóa các thư viện không cần thiết (~300-400 MB):

```bash
# Cách 1: Script tự động (khuyến nghị)
python build-portable/cleanup_unused_packages.py

# Cách 2: Manual (nếu script lỗi)
.envtietkiem\Scripts\pip uninstall -y moonshine_voice faster_whisper lightning pytorch_lightning
.envtietkiem\Scripts\pip uninstall -y tensorboardX opentelemetry-api opentelemetry-sdk
.envtietkiem\Scripts\pip uninstall -y colorama coloredlogs rich matplotlib pillow pandas
.envtietkiem\Scripts\pip uninstall -y grpcio sqlalchemy alembic optuna

# Kiểm tra imports vẫn hoạt động
python build-portable/test_imports.py
```

Xem chi tiết: `build-portable/PACKAGES_ANALYSIS.md`

### Bước 4: Build portable

```bash
build-portable/build.bat
```

Hoặc chạy trực tiếp:
```bash
python build-portable/build_portable.py
```

Output sẽ nằm tại: `dist/sherpa-vietnamese-asr/`

## Cấu trúc sau build

```
dist/sherpa-vietnamese-asr/
├── python/                          # Python embedded
│   ├── python.exe
│   ├── Lib/site-packages/           # All dependencies
│   └── ...
├── models/                          # AI Models
│   ├── zipformer-30m-rnnt-6000h/
│   ├── zipformer-30m-rnnt-streaming-6000h/
│   ├── sherpa-onnx-zipformer-vi-2025-04-20/
│   ├── vibert-capu/
│   ├── speaker_embedding/
│   │   └── nemo_en_titanet_small.onnx
│   ├── speaker_diarization/
│   │   └── sherpa-onnx-pyannote-segmentation-3-0/
│   ├── pyannote/                    # NEW: Community-1 Pipeline
│   │   ├── speaker-diarization-community-1/
│   │   ├── segmentation-3.0/
│   │   └── wespeaker-voxceleb-resnet34-LM/
│   ├── pyannote-onnx/               # NEW: Altunenes ONNX
│   │   ├── segmentation-community-1.onnx
│   │   └── embedding_model.onnx
│   └── dnsmos/
├── vocabulary/                      # Vocabulary data
├── app.py                           # Main application
├── speaker_diarization.py           # Main diarization module
├── speaker_diarization_pyannote.py      # NEW: Community-1 PyTorch
├── speaker_diarization_onnx_altunenes.py # NEW: Community-1 ONNX
├── ...
└── sherpa-vietnamese-asr.bat        # Launcher
```

## Speaker Diarization Models

| Model | Backend | File | Offline | Chất lượng |
|-------|---------|------|---------|------------|
| titanet_small | sherpa-onnx | `speaker_embedding/nemo_en_titanet_small.onnx` | ✅ | Tốt |
| community1 | pyannote.audio | `pyannote/speaker-diarization-community-1/` | ✅ | Xuất sắc |
| community1_onnx | ONNX Runtime | `pyannote-onnx/*.onnx` | ✅ | Rất tốt |

## Lưu ý

1. **HF Token**: Chỉ cần khi chạy `prepare_offline_build.py` lần đầu để tải pyannote models. Sau đó chạy hoàn toàn offline.

2. **Dung lượng**: Build hoàn chỉnh ~2-3GB tùy models.

3. **Không cần cài đặt Python** trên máy đích.

4. **Windows Defender**: Có thể cảnh báo với file `.bat`, chọn "More info" -> "Run anyway".
