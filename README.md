# Sherpa Vietnamese ASR

Ứng dụng nhận dạng giọng nói tiếng Việt **offline**, chạy trên **CPU**. Không cần GPU, không gửi dữ liệu ra internet.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)
![Version](https://img.shields.io/badge/Version-2.5.4-orange)
![Runtime](https://img.shields.io/badge/Runtime-ONNX%20only-success)

## Ba chế độ sử dụng

| | Desktop App | Web Service | Offline PWA |
|--|-------------|-------------|-------------|
| **Giao diện** | PyQt6 GUI | FastAPI + Web UI | Browser PWA installable |
| **Xử lý ASR** | Local Python + ONNX Runtime | Server-side queue | Browser-side WASM/WebGPU |
| **Đa người dùng** | Không | Có (anonymous + login) | Không, chạy trên thiết bị |
| **Launcher** | `sherpa-vietnamese-asr.bat` | `sherpa-vietnamese-asr-service.bat` | Tự bật cùng web service nếu `[OfflinePWA] enabled=true` |

## Tính năng

### Chung (Desktop, Web Service & Offline PWA)

- **Chuyển giọng nói thành văn bản** — MP3, WAV, M4A, FLAC, AAC, OGG, MP4, MKV, AVI, MOV, WEBM...
- **3 model ASR** — Zipformer 30M (nhanh), Zipformer 68M (chính xác), ROVER (bỏ phiếu 2 model)
- **Phân tách người nói** (Speaker Diarization) — Pure ONNX Runtime, không cần PyTorch:
  - Pyannote Community-1 (ResNet34-LM + PLDA + VBx, masked stats pool tối ưu)
  - Senko CAM++ 192-dim (spectral clustering / UMAP+HDBSCAN, nhanh hơn)
  - Senko CAM++ Optimized (batch inference, 2.5x nhanh hơn bản thường)
- **Tách giọng overlap 2-người nói** (opt-in) — Conv-TasNet ONNX + Hungarian match qua CAM++ embedding (xem `core/overlap_separator.py`)
- **NaturalTurn** — thuật toán nhận diện lượt nói tự nhiên, gộp backchannel vào người nói chính (Cychosz et al., Scientific Reports 2025)
- **Tự động thêm dấu câu, viết hoa** — ViBERT-capu (ONNX FP32/INT8) + pause hints
- **Hỗ trợ hotwords** — tên riêng, thuật ngữ chuyên ngành (Aho-Corasick)
- **Đánh giá chất lượng** — DNSMOS + ASR confidence
- **Decode/resampling thống nhất** — FFmpeg pipe + SoXR 16 kHz mono cho ASR và diarization, giảm lệch kết quả giữa file audio/video

### Desktop App

- **Thu âm trực tiếp** — real-time streaming, phím 1-9 đánh dấu người nói
- **Click câu để tua** — phát lại đồng bộ với văn bản, dual-highlight 2 speaker khi play đoạn overlap
- **Đổi tên / gộp / tách người nói** — chỉnh sửa kết quả diarization trực tiếp trên UI
- **Theme Dark / Light** — chuyển đổi giao diện (apply_theme in-place, không cần restart)

### Web Service

- **Tóm tắt cuộc họp** — Gemma 4 E2B qua llama-cpp-python (GGUF, chạy CPU)
- **PWA online** — cài trên mobile/desktop như app native
- **Offline PWA host** — chạy song song trên port riêng, mặc định `https://IP:8444`
- **Admin GUI** — quản lý server, session, queue, user
- **Windows Service** — chạy headless hoặc cài service

### Offline PWA

- **Chạy ASR trong trình duyệt** — Sherpa-ONNX WASM, ONNX Runtime Web, WebGPU cho dấu câu FP32 khi trình duyệt hỗ trợ
- **Model bootstrap đầy đủ** — tải model pack từ server nội bộ, cache bằng OPFS/Cache Storage để dùng sau khi mất mạng
- **Pipeline gần tương đương desktop** — VAD, ROVER 2 model, hotwords, punctuation, CAM++ diarization, Pyannote Community-1 VBx
- **Quản lý file cục bộ** — thư viện nguồn, panel kết quả, editor transcript, export/import `.asr.json`
- **Benchmark/calibration thiết bị** — profile hiệu năng theo thiết bị, re-calibration thủ công và metadata nghi ngờ ASR trong báo cáo

## Công nghệ

| Thành phần | Công nghệ |
|-----------|-----------|
| ASR | Sherpa-ONNX, Zipformer RNN-T (30M + 68M) |
| Diarization | Pyannote Community-1 + Senko CAM++ (Pure ONNX Runtime) |
| Overlap separation | Conv-TasNet 16k Libri2Mix (ONNX) |
| NaturalTurn | Backchannel detection (Cychosz et al. 2025) |
| Dấu câu | ViBERT-capu (ONNX FP32 / INT8) |
| VAD | Silero VAD + Pyannote Segmentation (ONNX) |
| Summarizer | Gemma 4 E2B (GGUF, llama-cpp-python) |
| Resampling | SoXR (HQ/VHQ) |
| Desktop GUI | PyQt6 (Dark/Light theme) |
| Web backend | FastAPI, WebSocket, SQLite |
| Offline PWA runtime | WASM, ONNX Runtime Web, WebGPU, OPFS, Cache Storage |
| Inference | ONNX Runtime (CPU) — **không cần PyTorch / asteroid / pyannote.audio ở runtime** |

## Yêu cầu

| | Tối thiểu | Khuyến nghị |
|--|-----------|-------------|
| OS | Windows 10 (64-bit) | Windows 10/11 |
| RAM | 8 GB | 16 GB+ |
| CPU | Intel i3 / Ryzen 3 | Intel i7 / Ryzen 7+ |
| Storage | 2 GB | 5 GB |

> Không cần GPU. Bản portable không cần cài Python.

## Cài đặt (development)

```bash
git clone https://github.com/welcomyou/sherpa-vietnamese-asr.git
cd sherpa-vietnamese-asr
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # Desktop
pip install -r requirements-online.txt   # Web service (thêm)
```

## Chạy

```bash
python app.py                            # Desktop app
python server_gui.py                     # Web service (GUI admin)
python server_launcher.py --no-gui       # Web service (headless)
python -m uvicorn offline_pwa.server:app --host 0.0.0.0 --port 8444  # Offline PWA standalone dev
```

Web service mặc định chạy HTTPS tại `https://IP:8443`. Offline PWA mặc định chạy kèm tại `https://IP:8444`. Admin mặc định: `admin` / `admin`.

## Build Portable

Build bản portable không cần cài Python trên máy đích:

```bash
# 1. Tải models (lần đầu) — auto-download từ HuggingFace với SHA256 pin
python build-portable/prepare_offline_build.py

# 2. Build
python build-portable/build_portable.py         # Desktop (~1.3 GB)
python build-portable/build_portable_online.py   # Web service + Offline PWA
```

Output trong `dist/sherpa-vietnamese-asr-<version>/` hoặc `dist/sherpa-vietnamese-asr-service-<version>/`. Bản service validate đủ asset runtime/model manifest của Offline PWA trước khi build xong.

## Models trên HuggingFace

Toàn bộ ONNX model export bởi project này được publish trên collection [welcomyou/sherpa-vietnamese-asr](https://huggingface.co/collections/welcomyou/sherpa-vietnamese-asr-69fd2a837f318846d84c15d4):

| Repo | License | Source |
|---|---|---|
| [welcomyou/vibert-capu-onnx](https://huggingface.co/welcomyou/vibert-capu-onnx) | CC-BY-SA-4.0 | dragonSwing/vibert-capu (FP32 + INT8) |
| [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | Apache-2.0 | 3D-Speaker CAM++ 200k speakers |
| [welcomyou/pyannote-community-1-onnx-split](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split) | CC-BY-4.0 | pyannote Community-1 (masked-pool split) |
| [welcomyou/convtasnet-libri2mix-16k-onnx](https://huggingface.co/welcomyou/convtasnet-libri2mix-16k-onnx) | CC-BY-SA-4.0 | JorisCos Conv-TasNet (asteroid) |

Script export reproducible trong [`convert_onnx/`](convert_onnx/). Build script `prepare_offline_build.py` verify SHA256 mọi file tải về để chống supply-chain attack.

## Cấu trúc

```
├── app.py                        # Desktop entry point
├── tab_file.py                   # Tab xử lý file
├── tab_live.py                   # Tab thu âm trực tiếp
├── streaming_asr.py              # Streaming ASR (offline VAD)
├── streaming_asr_online.py       # Streaming ASR (online, no VAD)
├── transcriber.py                # Transcriber thread wrapper
├── common.py                     # Shared UI components
├── server_gui.py                 # Web service admin GUI (PyQt6)
├── server_launcher.py            # Web service entry point
├── service_installer.py          # Windows Service installer
├── core/                         # Core modules (dùng chung desktop & web)
│   ├── asr_engine.py             # ASR pipeline (chunk, overlap, ROVER)
│   ├── asr_json.py               # Đọc/ghi .asr.json
│   ├── config.py                 # Config, CPU detection, model registry, theme
│   ├── vad_utils.py              # Silero VAD (ONNX)
│   ├── speaker_diarization.py    # Diarization dispatcher + NaturalTurn
│   ├── speaker_diarization_pure_ort.py   # Pure ORT diarization (ResNet34 + masked stats pool)
│   ├── speaker_diarization_senko_campp.py      # Senko CAM++ diarization
│   ├── speaker_diarization_senko_campp_optimized.py  # Senko CAM++ optimized
│   ├── overlap_separator.py      # 2-speaker overlap separation (Conv-TasNet ONNX)
│   ├── punctuation_restorer_improved.py  # Dấu câu (ViBERT + pause)
│   ├── gec_model.py              # GEC ONNX inference
│   ├── audio_analyzer.py         # DNSMOS quality analysis
│   ├── audio_decode.py           # FFmpeg/SoXR decode helpers dùng chung
│   ├── audio_preprocessing.py    # RMS normalize, preprocessing
│   ├── hotword_context.py        # Aho-Corasick hotword boosting
│   └── utils.py                  # Shared helpers
├── convert_onnx/                 # ONNX export scripts (reproducible)
│   ├── export_vibert_onnx.py     # ViBERT FP32 + INT8
│   ├── export_campplus_onnx.py   # CAM++ 200k from PyTorch (3D-Speaker)
│   ├── export_convtasnet_onnx.py # Conv-TasNet from asteroid
│   └── split_pyannote_embedding.py # Pyannote encoder + Gemm split
├── web_service/                  # FastAPI web service
│   ├── server.py                 # API endpoints + WebSocket
│   ├── database.py               # SQLite management
│   ├── auth.py                   # JWT authentication
│   ├── session_manager.py        # Session + anonymous timeout
│   ├── queue_manager.py          # Job queue
│   ├── config.py                 # Server config
│   ├── ssl_utils.py              # SSL cert generation
│   ├── audio_quality.py          # Audio quality (DNSMOS)
│   ├── summarizer.py             # Meeting summarizer
│   └── static/                   # Frontend (HTML/JS/CSS/PWA)
├── offline_pwa/                  # Browser-side full offline PWA
│   ├── server.py                 # Same-origin model/static host
│   ├── config.py                 # [OfflinePWA] config
│   ├── model_manifest.json       # Model pack manifest
│   └── static/                   # PWA shell, workers, WASM/vendor assets
├── shared_ui/                    # CSS/JS dùng chung web service và Offline PWA
├── build-portable/               # Build scripts
├── models/                       # AI models (tải riêng)
├── vocabulary/                   # Vocabulary data
├── config.ini                    # Runtime config cục bộ, không commit
└── hotword.txt                   # Hotword list
```

## Ghi nhận

### Thư viện & Models

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) — ASR Engine
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) — Vietnamese ASR Model (30M)
- [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) — Vietnamese ASR Model (68M)
- [ViBERT-capu](https://huggingface.co/dragonSwing/vibert-capu) — Punctuation Restoration (base PyTorch; ONNX bản convert: [welcomyou/vibert-capu-onnx](https://huggingface.co/welcomyou/vibert-capu-onnx))
- [Senko](https://github.com/narcotic-sh/senko) — Speaker Diarization Pipeline (CAM++ + spectral/UMAP+HDBSCAN clustering)
- [Pyannote](https://github.com/pyannote/pyannote-audio) — Speaker Diarization Pipeline (Community-1)
- [altunenes/speaker-diarization-community-1-onnx](https://huggingface.co/altunenes/speaker-diarization-community-1-onnx) — ONNX models cho Pyannote Community-1 (segmentation + embedding)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker) — CAM++ 192-dim Speaker Embedding (ONNX bản convert: [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx))
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) — ResNet34-LM Speaker Embedding
- [Asteroid](https://github.com/asteroid-team/asteroid) — [JorisCos/ConvTasNet_Libri2Mix_sepclean_16k](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k) cho overlap separation (ONNX bản convert: [welcomyou/convtasnet-libri2mix-16k-onnx](https://huggingface.co/welcomyou/convtasnet-libri2mix-16k-onnx))
- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge) — Audio Quality Assessment
- [ONNX Runtime Web](https://onnxruntime.ai/) — Browser inference cho Offline PWA
- [FFmpeg.wasm](https://ffmpegwasm.netlify.app/) và [mpg123-decoder](https://github.com/eshaz/mpg123-decoder) — Browser audio decode cho Offline PWA

### Papers

- Cychosz et al., "Natural conversational turn-taking" — *Scientific Reports* 2025 ([doi](https://www.nature.com/articles/s41598-025-24381-1)) — NaturalTurn backchannel detection
- Bredin et al., "Pyannote.audio 2.1" — *arXiv* 2023 ([2310.00032](https://arxiv.org/abs/2310.00032)) — Speaker diarization pipeline
- Chen et al., "3D-Speaker" — *arXiv* 2024 ([2403.19971](https://arxiv.org/abs/2403.19971)) — CAM++ spectral clustering
- Prabhavalkar et al., "Automatic gain control and multi-style training for robust ASR" — *ICASSP* 2015 ([pdf](https://research.google.com/pubs/archive/43289.pdf)) — Per-segment RMS normalization

## License

[MIT License](LICENSE)

> **Lưu ý**: PyQt6 sử dụng GPL v3. Để dùng thương mại closed-source, cần commercial license hoặc thay bằng PySide6 (LGPL).
