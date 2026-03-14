# Sherpa Vietnamese ASR

Ứng dụng nhận dạng giọng nói tiếng Việt **offline**, chạy trên **CPU**. Gồm 2 bản phân phối:

- **Desktop App** (`sherpa-vietnamese-asr`): GUI PyQt6, xử lý file và thu âm trực tiếp
- **Web Service** (`sherpa-vietnamese-asr-service`): FastAPI server đa người dùng, admin GUI, PWA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Web Service](#web-service)
- [Build Portable](#build-portable)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Tính năng chính

### Desktop App

**Xử lý File (Offline)**
- Nhận dạng giọng nói: MP3, M4A, WAV, OGG, FLAC, AAC, WMA, MP4, MKV, AVI, MOV...
- Phân tách người nói (Speaker Diarization)
- Thêm dấu câu tự động (BERT)
- Phân tích chất lượng âm thanh (DNSMOS)
- Phát lại đồng bộ: click câu để nghe đoạn tương ứng
- Tìm kiếm có dấu/không dấu

**Thu âm Trực tiếp (Live Streaming)**
- Nhận dạng real-time (Dual-stream: Draft + Commit)
- Chọn microphone, đánh giá chất lượng
- Phân tách người nói bằng phím 1-9
- VAD tích hợp, loại bỏ khoảng lặng
- Xuất file WAV + JSON

### Web Service

- Server HTTPS đa người dùng (FastAPI + Uvicorn)
- Upload file âm thanh, xử lý ASR qua hàng đợi
- Quản lý phiên, xác thực JWT, phân quyền admin/user
- Admin GUI (PyQt6) quản lý server, session, queue, user
- PWA: cài đặt trên mobile/desktop như app native
- Chạy headless (--no-gui) hoặc cài Windows Service

### Kỹ thuật nổi bật

- Chạy hoàn toàn offline, không gửi dữ liệu ra ngoài
- Tối ưu CPU, tự động điều chỉnh số luồng
- Xử lý file lớn: chia nhỏ tránh tràn RAM
- Model ASR: Zipformer (nhiều phiên bản)
- Thêm dấu câu thông minh: kết hợp BERT + pause hints (khoảng lặng giữa các từ) để xác định vị trí dấu câu chính xác hơn
- PWA: Web UI hỗ trợ cài đặt offline, Service Worker cache-first

---

## Yêu cầu hệ thống

| Component | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| **OS** | Windows 10 (64-bit) | Windows 10/11 (64-bit) |
| **Python** | 3.10 | 3.10 - 3.12 |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 3 GB (cho models) | 5 GB |
| **CPU** | Intel i3 / Ryzen 3 | Intel i7 / Ryzen 7+ |

> Không cần GPU, chạy hoàn toàn trên CPU.

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/welcomyou/sherpa-vietnamese-asr.git
cd sherpa-vietnamese-asr
```

### 2. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
# Desktop app
pip install -r requirements.txt

# Web service (thêm)
pip install -r requirements-online.txt
```

### 4. Tải Models

```bash
python build-portable/prepare_offline_build.py
```

Script tự kiểm tra, tải, giải nén models còn thiếu (~2GB, 10-30 phút).

<details>
<summary>Tải thủ công</summary>

#### ASR Models (chọn ít nhất 1)

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **sherpa-onnx-zipformer-vi-2025-04-20** | [HuggingFace](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) | `models/sherpa-onnx-zipformer-vi-2025-04-20/` | ~800 MB |
| **zipformer-30m-rnnt-6000h** | [HuggingFace](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | `models/zipformer-30m-rnnt-6000h/` | ~200 MB |
| **zipformer-30m-rnnt-streaming-6000h** | [HuggingFace](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h) | `models/zipformer-30m-rnnt-streaming-6000h/` | ~200 MB |

> - `sherpa-onnx-zipformer-vi-2025-04-20`: Độ chính xác cao nhất
> - `zipformer-30m-rnnt-6000h`: Nhẹ, nhanh, máy cấu hình thấp
> - `zipformer-30m-rnnt-streaming-6000h`: Thu âm trực tiếp

#### NLP Models

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **vibert-capu** | [HuggingFace](https://huggingface.co/dragonSwing/vibert-capu) | `models/vibert-capu/` | ~437 MB |

#### Speaker Diarization Models

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **nemo_en_titanet_small** | [HuggingFace](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/nemo_en_titanet_small.onnx) | `models/speaker_embedding/` | ~38 MB |
| **sherpa-onnx-pyannote-segmentation-3-0** | [GitHub](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2) | `models/speaker_diarization/` | ~50 MB |

#### Audio Quality Model

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **DNSMOS** | [GitHub](https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx) | `models/dnsmos/` | ~1 MB |

</details>

### 5. Chạy ứng dụng

```bash
# Desktop app
python app.py

# Web service (GUI admin)
python server_gui.py

# Web service (headless)
python server_launcher.py --no-gui
```

---

## Hướng dẫn sử dụng

### Tab "Xử lý tập tin" (Offline)

1. **Chọn file**: Kéo thả hoặc click chọn (MP3, WAV, M4A, MP4, MKV...)
2. **Cấu hình**: Model ASR, CPU threads, bật/tắt phân tách người nói
3. **Xử lý**: Click "Bắt đầu xử lý", đợi hoàn tất
4. **Sau xử lý**: Play, click câu để tua, chuột phải để tách/gộp người nói, tìm kiếm, copy, lưu JSON

### Tab "Thu âm trực tiếp" (Live)

1. Chọn microphone, test chất lượng (tùy chọn)
2. Click "Bắt đầu ghi âm", nói vào mic
3. Nhấn phím **1-9** đánh dấu người nói
4. "Dừng" tạm dừng, "Kết thúc" hoàn tất, "Xuất file WAV" lưu kèm JSON

---

## Web Service

### Truy cập

Sau khi khởi động server, truy cập `https://IP:8443` trên trình duyệt.

- Admin mặc định: `admin` / `admin` (đổi ngay sau khi đăng nhập)
- Self-signed cert: trình duyệt cảnh báo "Not Secure", bấm Advanced > Proceed

### Admin GUI

Chạy `server_gui.py` để mở giao diện quản trị:
- **Status**: Start/Stop server, cấu hình port, CPU threads
- **Sessions**: Quản lý phiên kết nối
- **Queue**: Hàng đợi xử lý file
- **Users**: Quản lý tài khoản, quota lưu trữ
- **Config**: Upload limit, timeout, SSL cert, JWT

### API Docs

Xem tại `https://IP:8443/api/docs` (Swagger UI tự động từ FastAPI).

---

## Build Portable

Build bản portable không cần cài Python trên máy đích.

### Chuẩn bị

```bash
# Tạo môi trường build
python build-portable/setup_build_env.py

# Tải models (nếu chưa có)
python build-portable/prepare_offline_build.py
```

### Build Desktop App

```bash
python build-portable/build_portable.py
```

Output: `dist/sherpa-vietnamese-asr/` + launcher `sherpa-vietnamese-asr.bat`

Loại trừ packages web (fastapi, uvicorn, starlette...) để giảm dung lượng.

### Build Web Service

```bash
python build-portable/build_portable_online.py
```

Output: `dist/sherpa-vietnamese-asr-service/` + launcher `sherpa-vietnamese-asr-service.bat`

Loại trừ packages desktop (sounddevice, matplotlib, pyinstaller...) và streaming model.

### Sử dụng bản portable

Copy toàn bộ folder `dist/sherpa-vietnamese-asr/` hoặc `dist/sherpa-vietnamese-asr-service/` sang máy đích, double-click file `.bat` để chạy. Không cần cài Python.

---

## Cấu trúc dự án

```
sherpa-vietnamese-asr/
├── app.py                          # Desktop GUI entry point
├── tab_file.py                     # Tab xử lý file
├── tab_live.py                     # Tab thu âm trực tiếp
├── transcriber.py                  # ASR offline wrapper
├── streaming_asr.py                # ASR streaming logic
├── streaming_asr_online.py         # ASR streaming cho web service
├── common.py                       # Shared utilities
├── quality_result_dialog.py        # Dialog kết quả chất lượng âm thanh
├── server_launcher.py              # Web service entry point
├── server_gui.py                   # Web service admin GUI
├── service_installer.py            # Windows Service installer
├── core/                           # Core modules (dùng chung desktop & web)
│   ├── config.py                   # Config, model registry, hotwords
│   ├── asr_engine.py               # ASR processing engine
│   ├── asr_json.py                 # Đọc/ghi .asr.json
│   ├── speaker_diarization.py      # Speaker diarization dispatcher
│   ├── speaker_diarization_pyannote.py   # PyAnnote backend
│   ├── speaker_diarization_onnx_altunenes.py  # ONNX backend
│   ├── punctuation_restorer_improved.py  # Thêm dấu câu (BERT + VAD)
│   ├── gec_model.py                # Grammar error correction
│   ├── gec_utils.py                # GEC utilities
│   ├── audio_analyzer.py           # Phân tích chất lượng âm thanh (DNSMOS)
│   ├── utils.py                    # Shared helpers
│   └── vocabulary.py               # BPE vocabulary handling
├── web_service/                    # FastAPI web service
│   ├── server.py                   # API endpoints
│   ├── database.py                 # SQLite management
│   ├── auth.py                     # JWT authentication
│   ├── session_manager.py          # Session handling
│   ├── queue_manager.py            # Job queue
│   ├── ssl_utils.py                # SSL cert generation
│   ├── config.py                   # Server config
│   ├── audio_quality.py            # Audio quality (DNSMOS)
│   └── static/                     # Frontend (HTML/JS/CSS/PWA)
│       ├── index.html              # SPA chính
│       ├── manifest.json           # PWA manifest
│       ├── sw.js                   # Service Worker (offline)
│       ├── js/                     # Modules: app, upload, player, search...
│       └── icons/                  # PWA icons
├── build-portable/                 # Build scripts
│   ├── build_portable.py           # Build desktop app
│   ├── build_portable_online.py    # Build web service
│   ├── setup_build_env.py          # Setup build environment
│   └── prepare_offline_build.py    # Download models
├── models/                         # AI models
├── vocabulary/                     # Vocabulary data
├── config.ini                      # Runtime config
├── hotword.txt                     # Hotword list (tên riêng, thuật ngữ)
├── requirements.txt                # Desktop dependencies
└── requirements-online.txt         # Web service dependencies
```

---

## Troubleshooting

### "Không tìm thấy model"

```bash
python build-portable/prepare_offline_build.py
```

### "DLL load failed" hoặc lỗi sherpa_onnx

Cài [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), hoặc tạo lại môi trường ảo.

### "No module named 'xxx'"

```bash
pip install -r requirements.txt
pip install -r requirements-online.txt  # nếu dùng web service
```

### "FFmpeg not found"

Tải FFmpeg từ https://ffmpeg.org/download.html, đặt `ffmpeg.exe` cùng thư mục `app.py`.

### Chạy chậm/quá tải CPU

Giảm CPU Threads trong cấu hình, chọn model nhẹ hơn (zipformer-30m).

---

## License

[MIT License](LICENSE)

### Third-party Licenses

| Thư viện | License |
|----------|---------|
| sherpa-onnx | Apache-2.0 |
| PyQt6 | GPL-3.0 / Commercial |
| transformers | Apache-2.0 |
| torch | BSD-3-Clause |
| FastAPI | MIT |
| librosa | ISC |
| numpy | BSD-3-Clause |
| onnxruntime | MIT |
| speechbrain | Apache-2.0 |

> **Lưu ý**: PyQt6 sử dụng GPL v3. Để dùng thương mại closed-source, mua commercial license hoặc thay bằng PySide6 (LGPL).

---

## Ghi nhận

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - ASR Engine
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) - Vietnamese ASR Model
- [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) - Vietnamese ASR Model (main)
- [ViBERT-capu](https://huggingface.co/dragonSwing/vibert-capu) - Punctuation Restoration
- [DNSMOS](https://github.com/microsoft/DNS-Challenge) - Audio Quality Assessment

---

**Liên hệ**: Tạo [Issue](https://github.com/welcomyou/sherpa-vietnamese-asr/issues) trên GitHub nếu gặp vấn đề.
