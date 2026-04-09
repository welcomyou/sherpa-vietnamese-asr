# CLAUDE.md — Sherpa Vietnamese ASR

## Project Overview

Ứng dụng nhận dạng giọng nói tiếng Việt offline, chạy CPU. Hai bản phân phối:
- **Desktop App** (`app.py`) — PyQt6 GUI, 1 user
- **Web Service** (`server_launcher.py`) — FastAPI + PWA, multi-user

## Key Commands

```bash
# Build portable (output: dist/)
python build-portable/build_portable.py           # Desktop (~1.2 GB)
python build-portable/build_portable_online.py     # Web service (~1.7 GB)

# Run dev
python app.py                                      # Desktop
python server_launcher.py --no-gui                 # Web headless

# Versioning (SemVer from git tags)
git tag v2.2.0 -m "Description"                    # Release → build ra "2.2.0"
# Không tag → build ra "2.1.2+3.abc1234" (dev build, tự động)

# Sync to GitHub repo
# Copy code từ d:\App\asr-vn → D:\App\sherpa-vietnamese-asr rồi push
```

## Architecture

### Core modules (dùng chung desktop & web)
- `core/asr_engine.py` — ASR pipeline chính (chunk, overlap, ROVER voting)
- `core/speaker_diarization.py` — Dispatcher + NaturalTurn + model registry
- `core/speaker_diarization_pure_ort.py` — Pyannote Community-1 (ResNet34+PLDA+VBx)
- `core/speaker_diarization_senko_campp.py` — Senko CAM++ 192-dim (spectral/UMAP+HDBSCAN)
- `core/speaker_diarization_senko_campp_optimized.py` — Senko optimized (batch, 2.5x faster)
- `core/punctuation_restorer_improved.py` — ViBERT-capu (ONNX) dấu câu
- `core/version.py` — Auto-version từ git tags → VERSION file

### Entry points
- `app.py` — Desktop (PyQt6). Config trong `config.ini` [FileSettings] + [LiveSettings]
- `server_launcher.py` — Web service. Config trong `config.ini` [ServerSettings]
- `server_gui.py` — Web admin GUI (PyQt6 wrapper cho server)
- `resource_monitor.py` — Monitor CPU/RAM/Disk của desktop app

## Build System

### Versioning
- `core/version.py` đọc `git describe --tags` → ghi `VERSION` file vào dist/
- About dialog (desktop + web) đọc version tự động, không hardcode
- Portable build không có `.git` → đọc từ `VERSION` file

### Build lưu ý quan trọng
- Venv: `.envtietkiem/` (không phải `.venv`)
- Senko diarization cần: `numba`, `llvmlite`, `tqdm`, `pynndescent`, `umap-learn`, `hdbscan` — KHÔNG được exclude
- Giữ `.dist-info` cho: `pynndescent`, `umap_learn`, `hdbscan`, `numba`, `llvmlite`, `scikit_learn` (dùng `importlib.metadata`)
- Strip `.opt` files (ORT cache, máy target tự tạo)
- Models: `campp-3dspeaker/` + `pyannote-onnx/` cần cho Senko

### Models trong build
- `zipformer-30m-rnnt-6000h` — ASR 30M (nhanh)
- `sherpa-onnx-zipformer-vi-2025-04-20` — ASR 68M (chính xác, default web)
- `pyannote-onnx/` — segmentation + embedding ONNX (diarization)
- `campp-3dspeaker/` — CAM++ 192-dim (Senko diarization)
- `silero-vad/` — Silero VAD
- `vibert-capu/` — Punctuation (desktop: int8, web: fp32)
- `dnsmos/` — Audio quality

## GitHub Sync Workflow

"Đồng bộ qua repo github" = copy code từ `d:\App\asr-vn` → `D:\App\sherpa-vietnamese-asr`:
1. Copy top-level .py files, README.md, LICENSE, resource_monitor.py
2. `rm -f core/*.py` rồi copy lại (full sync, xóa file cũ)
3. Copy `web_service/*.py` + `web_service/static/*`
4. Copy `build-portable/*.py`
5. `git add -A && git commit && git tag && git push --tags`

**Không** copy: `.envtietkiem/`, `dist/`, `temp/`, `models/`, `*.exe`, `*.dll`, `.claude/`

## Conventions

- Test/benchmark/experiment files → `temp/` (không commit)
- Desktop: i5-12400/8GB, ưu tiên không tràn RAM, save_ram=True mặc định
- Web: 20vCPU/32GB, có summarizer Gemma 4 E2B
- Emoji OK trong PyQt6 UI
- Commit message tiếng Việt hoặc English đều OK
