# Sherpa Vietnamese ASR

Ứng dụng nhận dạng giọng nói tiếng Việt **offline**, chạy trên **CPU**. Không cần GPU, không gửi dữ liệu ra internet.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)
![Version](https://img.shields.io/badge/Version-2.0-orange)

## Hai bản phân phối

| | Desktop App | Web Service |
|--|-------------|-------------|
| **Giao diện** | PyQt6 GUI | FastAPI + Web UI (PWA) |
| **Đa người dùng** | Không | Có (anonymous + login) |
| **File** | `sherpa-vietnamese-asr.bat` | `sherpa-vietnamese-asr-service.bat` |

## Tính năng

- **Chuyển giọng nói thành văn bản** — hỗ trợ MP3, WAV, M4A, MP4, MKV, AVI...
- **Phân tách người nói** (Speaker Diarization) — Pure ONNX Runtime, không cần PyTorch
- **Tự động thêm dấu câu, viết hoa** — ViBERT-capu + pause hints
- **ROVER** — bỏ phiếu 2 model ASR để tăng độ chính xác
- **Thu âm trực tiếp** (Desktop) — real-time streaming, phím 1-9 đánh dấu người nói
- **Click câu để tua** — phát lại đồng bộ với văn bản
- **Hỗ trợ hotwords** — tên riêng, thuật ngữ chuyên ngành
- **Đánh giá chất lượng** — DNSMOS + ASR confidence
- **PWA** (Web) — cài trên mobile/desktop như app native

## Công nghệ

| Thành phần | Công nghệ |
|-----------|-----------|
| ASR | Sherpa-ONNX, Zipformer RNN-T |
| Diarization | Pyannote Community-1 (Pure ONNX Runtime) |
| Dấu câu | ViBERT-capu (ONNX) |
| VAD | Silero VAD (ONNX) |
| Desktop GUI | PyQt6 |
| Web backend | FastAPI, WebSocket, SQLite |
| Inference | ONNX Runtime (CPU) |

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
pip install -r requirements-online.txt   # Web service
```

## Chạy

```bash
python app.py                            # Desktop app
python server_gui.py                     # Web service (GUI admin)
python server_launcher.py --no-gui       # Web service (headless)
```

## Build Portable

```bash
python build-portable/build_portable.py         # Desktop (~1.4 GB)
python build-portable/build_portable_online.py   # Web service (~1.4 GB)
```

Output trong `dist/`. Copy folder sang máy đích, double-click `.bat` để chạy.

## Cấu trúc

```
├── app.py                     # Desktop entry point
├── tab_file.py                # Tab xử lý file
├── tab_live.py                # Tab thu âm trực tiếp
├── core/                      # Core modules (dùng chung)
│   ├── asr_engine.py          # ASR pipeline (chunk, overlap, ROVER, postprocess)
│   ├── speaker_diarization_pure_ort.py  # Diarization (Pure ORT)
│   ├── punctuation_restorer_improved.py # Dấu câu (ViBERT + pause)
│   ├── vad_utils.py           # Silero VAD
│   └── audio_analyzer.py      # DNSMOS quality
├── web_service/               # FastAPI web service
│   ├── server.py              # API endpoints + WebSocket
│   ├── queue_manager.py       # Job queue
│   ├── session_manager.py     # Session + anonymous timeout
│   └── static/                # Frontend (PWA)
├── build-portable/            # Build scripts
├── models/                    # AI models (tải riêng)
└── config.ini                 # Runtime config
```

## Ghi nhận

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) — ASR Engine
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) — Vietnamese ASR Model
- [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) — Vietnamese ASR Model
- [ViBERT-capu](https://huggingface.co/dragonSwing/vibert-capu) — Punctuation Restoration
- [Pyannote](https://github.com/pyannote/pyannote-audio) — Speaker Diarization
- [DNSMOS](https://github.com/microsoft/DNS-Challenge) — Audio Quality

## License

[MIT License](LICENSE)

> **Lưu ý**: PyQt6 sử dụng GPL v3. Để dùng thương mại closed-source, cần commercial license hoặc thay bằng PySide6 (LGPL).
