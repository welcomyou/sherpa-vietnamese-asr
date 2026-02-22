# Build Portable Scripts

Thư mục này chứa các script để tạo bản **portable** (không cần cài Python) và tải models.

> **Lưu ý**: Các script này tạo ra folder portable chạy bằng Python embedded, **KHÔNG** tạo file .exe đơn lẻ.

---

## 📋 Danh sách file

| File | Mô tả | Cách chạy |
|------|-------|-----------|
| `build.bat` | File batch chạy build (dùng cho Windows) | Double-click hoặc `build.bat` |
| `build_portable.py` | Script build portable chính | `python build-portable/build_portable.py` |
| `prepare_offline_build.py` | Script tải tất cả models tự động | `python build-portable/prepare_offline_build.py` |
| `setup_build_env.py` | Script setup môi trường build | `python build-portable/setup_build_env.py` |

---

## 🚀 Quy trình Build Portable

### Bước 1: Setup môi trường (chỉ chạy 1 lần)

```bash
python build-portable/setup_build_env.py
```

Script sẽ:
- Tạo virtual environment tại `.envtietkiem/`
- Cài đặt tất cả dependencies
- Cài đặt sherpa-onnx
- Test imports

> ⏱️ Thờigian: 10-15 phút tùy kết nối mạng

### Bước 2: Tải Models

```bash
python build-portable/prepare_offline_build.py
```

Script sẽ:
- Kiểm tra 9 models cần thiết
- Tự động tải những models còn thiếu
- Đặt vào đúng thư mục `models/`

Danh sách models:
1. `sherpa-onnx-zipformer-vi-2025-04-20` - ASR chính
2. `zipformer-30m-rnnt-6000h` - ASR nhẹ
3. `zipformer-30m-rnnt-streaming-6000h` - ASR streaming
4. `sat-12l-sm` - Tách câu
5. `vibert-capu` - Thêm dấu câu
6. `nemo_en_titanet_small` - Speaker embedding
7. `eres2netv2_zh` - Speaker embedding (ZH+EN)
8. `sherpa-onnx-pyannote-segmentation-3-0` - Voice segmentation
9. `dnsmos` - Audio quality assessment

> ⏱️ Thờigian: 10-30 phút tùy tốc độ mạng (tổng ~2GB)

### Bước 3: Build Portable

**Cách 1: Dùng file batch (khuyến nghị)**
```bash
build-portable/build.bat
```
Hoặc double-click file `build.bat`

**Cách 2: Chạy Python script**
```bash
python build-portable/build_portable.py
```

Script sẽ:
- Tải Python embedded (nếu chưa có)
- Copy source code và thư viện
- Copy models
- Tạo launcher script

> ⏱️ Thờigian: 5-10 phút

Kết quả sẽ nằm tại: `dist/Lightweight_ASR/`

---

## 📁 Cấu trúc sau khi build

```
dist/Lightweight_ASR/
├── Lightweight_ASR.bat     # ← File chạy chính (double-click vào đây)
├── python/                 # Python embedded runtime
│   ├── python.exe
│   └── Lib/site-packages/  # Thư viện đã cài
├── models/                 # AI models
├── vocabulary/             # Vocabulary files
├── app.py                  # Source code chính
├── *.py                    # Các module Python khác
└── README.txt              # Hướng dẫn
```

---

## ▶️ Cách chạy bản portable

Sau khi build xong:

1. Vào thư mục `dist/Lightweight_ASR/`
2. **Double-click file `Lightweight_ASR.bat`**
3. Ứng dụng sẽ khởi động

> **Lưu ý**: Không cần cài Python trên máy đích. Copy cả folder `Lightweight_ASR/` sang máy khác vẫn chạy được.

---

## 🔧 Troubleshooting

### Lỗi "Virtual environment not found"

Chạy lại bước 1:
```bash
python build-portable/setup_build_env.py
```

### Lỗi "Model not found"

Chạy lại bước 2:
```bash
python build-portable/prepare_offline_build.py
```

### Lỗi khi build

1. Xóa thư mục build cũ:
```bash
rmdir /s /q build
rmdir /s /q dist
```

2. Chạy lại build:
```bash
build-portable/build.bat
```

### Lỗi khi chạy Lightweight_ASR.bat

1. Kiểm tra file `python/python.exe` có tồn tại không
2. Kiểm tra Windows Defender có chặn không
3. Chạy bằng quyền Administrator thử

---

## 📝 Lưu ý quan trọng

| Vấn đề | Giải thích |
|--------|-----------|
| **Không phải .exe** | Đây là bản portable dùng Python embedded, chạy bằng `.bat` |
| **Dung lượng lớn** | ~3-4GB vì bao gồm cả Python runtime và models |
| **Chỉ Windows** | Chỉ chạy được trên Windows 10/11 64-bit |
| **Không cần cài Python** | Máy đích không cần cài Python |
| **Copy là chạy** | Copy cả folder sang máy khác, double-click .bat là chạy |

---

## 🆚 So sánh: Source vs Portable

| | Chạy từ Source | Bản Portable |
|---|---|---|
| **Cần cài Python** | Có | Không |
| **Cần cài dependencies** | Có (pip install) | Không |
| **Dung lượng** | ~500MB (chỉ models) | ~3-4GB (cả Python) |
| **Cách chạy** | `python app.py` | Double-click `.bat` |
| **Copy sang máy khác** | Khó (phải cài lại môi trường) | Dễ (copy folder là xong) |
