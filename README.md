# Sherpa Vietnamese ASR

Ứng dụng nhận dạng giọng nói tiếng Việt **offline**, chạy trên **CPU**, giao diện trực quan, hỗ trợ xử lý file âm thanh và thu âm trực tiếp.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Tính năng chính

### 🎙️ Xử lý File Âm thanh (Offline)
- **📝 Nhận dạng giọng nói**: Hỗ trợ nhiều định dạng (MP3, M4A, WAV, OGG, FLAC, AAC, WMA)
- **👥 Phân tách người nói (Speaker Diarization)**: Tự động nhận diện hoặc gán thủ công các người nói khác nhau
- **✏️ Thêm dấu câu tự động**: Sử dụng BERT để phục hồi dấu câu tiếng Việt
- **📄 Tách đoạn thông minh (SAT)**: Tách văn bản thành các câu có nghĩa dựa trên ngữ nghĩa
- **📊 Phân tích chất lượng âm thanh**: Đánh giá SIG/BAK/OVRL theo chuẩn DNSMOS và đánh giá độ tự tin nhận dạng của model ASR
- **▶️ Phát lại đồng bộ**: Click vào câu để nghe đoạn âm thanh tương ứng
- **🔍 Tìm kiếm nội dung**: Tìm kiếm có dấu/không dấu

### 🎤 Thu âm Trực tiếp (Live Streaming)
- **⚡ Nhận dạng real-time**: Kiến trúc Dual-stream (Draft + Commit)
- **🎚️ Hỗ trợ thu âm từ microphone**: Chọn và đánh giá chất lượng microphone (DNSMOS và độ tự tin của model)
- **👥 Phân tách người nói (Speaker Diarization)**: Tự động nhận diện hoặc gán thủ công các người nói khác nhau
- **🎯 VAD tích hợp**: Tự động phát hiện giọng nói, loại bỏ khoảng lặng
- **▶️ Phát lại đồng bộ**: Click vào câu để nghe đoạn âm thanh tương ứng
- **🔍 Tìm kiếm nội dung**: Tìm kiếm có dấu/không dấu

### ⚙️ Tối ưu & Hiệu suất
- **🔒 Chạy hoàn toàn offline**: Không gửi dữ liệu lên server
- **💻 Tối ưu CPU**: Tự động điều chỉnh số luồng theo cấu hình máy
- **📦 Xử lý file lớn**: Chia nhỏ file để tránh tràn RAM
- **🚀 Model ASR**: Zipformer

---

## 💻 Yêu cầu hệ thống

| Component | Yêu cầu tối thiểu | Khuyến nghị |
|-----------|-------------------|-------------|
| **OS** | Windows 10 (64-bit) | Windows 10/11 (64-bit) |
| **Python** | 3.10 | 3.10 - 3.12 |
| **RAM** | 8 GB | 16 GB trở lên |
| **Storage** | 3 GB (cho models) | 5 GB |
| **CPU** | Intel i3 / AMD Ryzen 3 | Intel i7 / AMD Ryzen 7 trở lên |

> **Lưu ý**: Không cần GPU, ứng dụng chạy hoàn toàn trên CPU.

---

## 🚀 Cài đặt

### Bước 1: Clone repository

```bash
# Clone repo về máy
git clone https://github.com/welcomyou/sherpa-vietnamese-asr.git

# Di chuyển vào thư mục project
cd sherpa-vietnamese-asr
```

### Bước 2: Tạo virtual environment

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies

```bash
# Cài đặt tất cả thư viện cần thiết
pip install -r requirements.txt
```

> **Lưu ý**: Quá trình này có thể mất 5-10 phút tùy kết nối mạng.

### Bước 4: Tải Models

> **⚠️ QUAN TRỌNG**: Đây là bước **BẮT BUỘC**. Ứng dụng không thể chạy nếu thiếu models.

### Cách 1: Tải tự động (Khuyến nghị)

Chạy script tự động tải tất cả models:

```bash
python build-portable/prepare_offline_build.py
```

Script sẽ:
- ✅ Kiểm tra models nào đã có, chưa có
- ✅ Tự động tải những models còn thiếu
- ✅ Giải nén và đặt vào đúng thư mục

**Thờigian tải**: 10-30 phút tùy tốc độ mạng (tổng dung lượng ~2GB)

### Cách 2: Tải thủ công

Nếu bạn muốn tải thủ công hoặc script tự động bị lỗi:

#### 1. ASR Models (Bắt buộc - chọn ít nhất 1)

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **sherpa-onnx-zipformer-vi-2025-04-20** | [HuggingFace](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) | `models/sherpa-onnx-zipformer-vi-2025-04-20/` | ~800 MB |
| **zipformer-30m-rnnt-6000h** | [HuggingFace](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | `models/zipformer-30m-rnnt-6000h/` | ~200 MB |
| **zipformer-30m-rnnt-streaming-6000h** | [HuggingFace](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h) | `models/zipformer-30m-rnnt-streaming-6000h/` | ~200 MB |

> **Lưu ý**: 
> - `sherpa-onnx-zipformer-vi-2025-04-20`: Model chính, độ chính xác cao nhất
> - `zipformer-30m-rnnt-6000h`: Model nhẹ, nhanh, cho máy cấu hình thấp
> - `zipformer-30m-rnnt-streaming-6000h`: Dùng cho thu âm trực tiếp

#### 2. NLP Models

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **sat-12l-sm** | [HuggingFace](https://huggingface.co/segment-any-text/sat-12l-sm) | `models/sat-12l-sm/` | ~530 MB |
| **vibert-capu** | [HuggingFace](https://huggingface.co/dragonSwing/vibert-capu) | `models/vibert-capu/` | ~437 MB |

#### 3. Speaker Diarization Models

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **nemo_en_titanet_small** | [HuggingFace](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/nemo_en_titanet_small.onnx) | `models/speaker_embedding/` | ~38 MB |
| **eres2netv2_zh** | [HuggingFace](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx) | `models/speaker_embedding/` | ~68 MB |
| **sherpa-onnx-pyannote-segmentation-3-0** | [GitHub](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2) | `models/speaker_diarization/` | ~50 MB |

#### 4. Audio Quality Model

| Model | Link | Thư mục | Dung lượng |
|-------|------|---------|------------|
| **DNSMOS** | [GitHub](https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx) | `models/dnsmos/` | ~1 MB |

> **Lưu ý**: DNSMOS sẽ tự động tải khi bạn sử dụng tính năng phân tích chất lượng lần đầu.

### Cấu trúc thư mục models sau khi tải

```
models/
├── sherpa-onnx-zipformer-vi-2025-04-20/    # ASR chính
│   ├── encoder-epoch-12-avg-8.onnx
│   ├── decoder-epoch-12-avg-8.onnx
│   ├── joiner-epoch-12-avg-8.onnx
│   ├── tokens.txt
│   └── bpe.model
├── zipformer-30m-rnnt-6000h/               # ASR nhẹ
│   ├── encoder-epoch-20-avg-10.int8.onnx
│   ├── decoder-epoch-20-avg-10.onnx
│   ├── joiner-epoch-20-avg-10.onnx
│   └── tokens.txt
├── zipformer-30m-rnnt-streaming-6000h/     # ASR streaming
│   ├── encoder-epoch-20-avg-10-chunk-64-left-128.int8.onnx
│   ├── decoder-epoch-20-avg-10.onnx
│   ├── joiner-epoch-20-avg-10.onnx
│   └── tokens.txt
├── sat-12l-sm/                             # Tách câu
│   ├── model_optimized.onnx
│   └── config.json
├── vibert-capu/                            # Thêm dấu câu
│   ├── pytorch_model.bin
│   └── config.json
├── speaker_embedding/                      # Speaker diarization
│   ├── nemo_en_titanet_small.onnx
│   └── 3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx
├── speaker_diarization/                    # Segmentation
│   └── model.onnx
└── dnsmos/                                 # Audio quality
    └── sig_bak_ovr.onnx
```

### Bước 5: Chạy ứng dụng

```bash
# Đảm bảo đang ở trong thư mục project và virtual environment đã được kích hoạt
python app.py
```

Giao diện ứng dụng sẽ mở ra với 2 tab chính:
- **"Xử lý tập tin"**: Xử lý file âm thanh có sẵn
- **"Thu âm trực tiếp"**: Thu âm và nhận dạng real-time

## 📖 Hướng dẫn sử dụng

### Tab "Xử lý tập tin" (Offline)

1. **Chọn file âm thanh**:
   - Kéo thả file vào vùng "Kéo thả file âm thanh vào đây"
   - Hoặc click để chọn file
   - Hỗ trợ: MP3, M4A, WAV, OGG, FLAC, AAC, WMA

2. **Cấu hình**:
   - **Model**: Chọn model ASR (khuyến nghị: sherpa-onnx-zipformer-vi-2025-04-20)
   - **CPU Threads**: Số luồng xử lý (mặc định: 4)
   - **SAT Threshold**: Ngưỡng tách câu (1-10, cao = tách nhiều câu)
   - **Phân tách người nói**: Bật/tắt speaker diarization (độ chính xác tạm nhưng chạy lâu)

3. **Xử lý**:
   - Click "Bắt đầu xử lý"
   - Đợi quá trình hoàn tất (thời gian tùy độ dài file)
   - Kết quả hiển thị dạng hội thoại có phân biệt người nói

4. **Tính năng sau xử lý**:
   - **Play**: Nghe lại file gốc
   - **Click vào câu**: Tua đến đoạn âm thanh tương ứng
   - **Click chuột phải - phân tách người nói**: Phân tách, gộp người nói
   - **Search**: Tìm kiếm nội dung (có dấu/không dấu)
   - **Copy**: Sao chép văn bản
   - **Save**: Lưu kết quả ra file JSON để sau này mở file âm thanh lại không cần chạy ASR lại.

### Tab "Thu âm trực tiếp" (Live)

1. **Chọn microphone** từ danh sách

2. **Test microphone** (tùy chọn):
   - Click "Đánh giá chất lượng"
   - Ghi âm 8 giây test
   - Xem kết quả đánh giá SIG/BAK/OVRL

3. **Bắt đầu ghi âm**:
   - Click "Bắt đầu ghi âm"
   - Nói vào microphone
   - Văn bản hiển thị real-time

4. **Đánh dấu người nói**:
   - Nhấn phím số **1-9** để đánh dấu người nói
   - Click chuột phải - phân tách, gộp người nói
   - Ví dụ: Nhấn "1" khi người A nói, nhấn "2" khi người B nói

5. **Dừng ghi âm**:
   - Click "Dừng ghi âm"
   - Kết quả lưu tự động
---

## 📁 Cấu trúc dự án

```
sherpa-vietnamese-asr/
├── app.py                      # Entry point
├── tab_file.py                 # Tab xử lý file
├── tab_live.py                 # Tab thu âm trực tiếp
├── transcriber.py              # ASR offline logic
├── streaming_asr.py            # ASR streaming logic
├── streaming_asr_online.py     # Online streaming manager
├── speaker_diarization.py      # Phân tách người nói
├── audio_analyzer.py           # Phân tích chất lượng âm thanh
├── punctuation_restorer_improved.py  # Thêm dấu câu
├── sat_segmenter.py            # Tách đoạn văn bản
├── quality_result_dialog.py    # Dialog hiển thị kết quả
├── common.py                   # Utilities & shared components
├── build-portable/             # Scripts build & tải models
│   └── prepare_offline_build.py
├── models/                     # AI models (tự động tải)
├── vocabulary/                 # Vocabulary cho NLP
├── requirements.txt            # Python dependencies
├── README.md                   # File này
└── LICENSE                     # MIT License
```

---

## 🔧 Troubleshooting

### Lỗi: "Không tìm thấy model"

**Nguyên nhân**: Chưa tải models về hoặc đặt sai vị trí

**Giải pháp**:
```bash
# Chạy lại script tải model
python build-portable/prepare_offline_build.py
```

### Lỗi: "DLL load failed" hoặc lỗi sherpa_onnx

**Nguyên nhân**: Thiếu Visual C++ Redistributable hoặc conflict thư viện

**Giải pháp**:
1. Cài đặt [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Gỡ cài đặt và cài lại trong môi trường ảo mới

### Lỗi: "No module named 'xxx'"

**Nguyên nhân**: Chưa cài đủ dependencies

**Giải pháp**:
```bash
pip install -r requirements.txt
```

### Lỗi: "Model not found at: models/..."

**Nguyên nhân**: Chưa tải model hoặc tải thiếu file

**Giải pháp**: 
- Kiểm tra đã có đủ file trong thư mục `models/` chưa
- Tải lại model bị thiếu theo link trong bảng ở trên

### Lỗi: "FFmpeg not found"

**Nguyên nhân**: Thiếu FFmpeg để xử lý MP3/M4A

**Giải pháp**:
1. Tải FFmpeg từ https://ffmpeg.org/download.html
2. Giải nén và thêm vào PATH, HOẶC
3. Để `ffmpeg.exe` cùng thư mục với `app.py`

### Ứng dụng chạy chậm/quá tải CPU

**Giải pháp**:
- Giảm "CPU Threads" trong cấu hình (mặc định: 4)
- Đóng các ứng dụng khác đang chạy
- Chọn model nhẹ hơn (zipformer-30m thay vì sherpa-onnx-zipformer-vi)

---

## 📝 Changelog

### v1.0.0
- Phiên bản đầu tiên với đầy đủ tính năng ASR offline và live streaming
- Hỗ trợ speaker diarization và punctuation restoration
- Tích hợp audio quality analyzer (DNSMOS)
- Giao diện PyQt6 với dark theme

---

## 📄 License

Dự án này sử dụng [MIT License](LICENSE).

### Third-party Licenses

| Thư viện | License |
|----------|---------|
| sherpa-onnx | Apache-2.0 |
| PyQt6 | GPL-3.0 / Commercial |
| transformers | Apache-2.0 |
| torch | BSD-3-Clause |
| sentence-transformers | Apache-2.0 |
| wtpsplit | MIT |
| soundfile | BSD-3-Clause |
| librosa | ISC |
| numpy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| onnxruntime | MIT |
| pydub | MIT |
| speechbrain | Apache-2.0 |

**Lưu ý về thương mại**: 
- PyQt6 sử dụng GPL v3, yêu cầu open source nếu phân phối
- Để dùng thương mại closed-source, hãy:
  - Mua commercial license PyQt6, HOẶC
  - Thay thế bằng PySide6 (LGPL)

---

## 🙏 Ghi nhận

### ASR Models
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - ASR Engine
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) - Vietnamese ASR Model (offline)
- [hynt/Zipformer-30M-RNNT-Streaming-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h) - Vietnamese ASR Model (streaming)
- [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) - Vietnamese ASR Model (main)

### NLP & Segmentation
- [ViBERT-capu](https://huggingface.co/dragonSwing/vibert-capu) - Punctuation Restoration
- [Wikipedia Segmentation](https://huggingface.co/segment-any-text/sat-12l-sm) - SAT Segmentation

### Speaker Diarization
- [csukuangfj/speaker-embedding-models](https://huggingface.co/csukuangfj/speaker-embedding-models) - Speaker Embedding Models

### Audio Quality
- [DNSMOS](https://github.com/microsoft/DNS-Challenge) - Audio Quality Assessment

---
---


**Liên hệ**: Nếu có vấn đề, vui lòng tạo [Issue](https://github.com/welcomyou/sherpa-vietnamese-asr/issues) trên GitHub.
