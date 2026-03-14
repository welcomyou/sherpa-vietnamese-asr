# Hướng dẫn sử dụng Pyannote 3.1 Speaker Diarization

## Tổng quan

Model **Pyannote 3.1 + ONNX** cung cấp độ chính xác cao nhất (SOTA) cho speaker diarization trên CPU.

## So sánh các Model

| Model | Kích thước | Tốc độ | Độ chính xác | Đặc điểm |
|-------|-----------|--------|--------------|----------|
| **TitaNet Small** | 38.4 MB | Nhanh | Tốt | Tiếng Anh, nhẹ |
| **ERes2NetV2** | 68.1 MB | Trung bình | Cao | Đa ngôn ngữ (ZH+EN) |
| **Pyannote 3.1** | ~110 MB | Trung bình | **SOTA** | Xử lý overlapping speech |

## Yêu cầu cài đặt

### 1. Cài đặt thư viện

```bash
pip install pyannote.audio
```

### 2. Lấy HuggingFace Token

1. Truy cập: https://huggingface.co/settings/tokens
2. Tạo token mới (quyền `read`)
3. Copy token để dùng ở bước sau

### 3. Chấp nhận Licenses

Truy cập và chấp nhận điều khoản tại:
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### 4. Cấu hình Token

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
python app.py
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
python app.py
```

**Linux/Mac:**
```bash
export HF_TOKEN=your_token_here
python app.py
```

## Sử dụng trong ứng dụng

1. Mở ứng dụng
2. Check "Phân tách Người nói (Speaker diarization)"
3. Chọn model **"🏆 Pyannote 3.1 + ONNX (SOTA)"** từ dropdown
4. Chọn file audio và bắt đầu xử lý

## Lưu ý

- **Lần chạy đầu tiên**: Model sẽ tự động tải về (~110MB)
- **Tốc độ**: Chậm hơn ~2-3x so với TitaNet nhưng chính xác hơn đáng kể
- **RTF**: ~0.4-0.6 (1 giờ audio xử lý trong 25-35 phút)
- **RAM**: Cần ~4GB RAM trống

## Xử lý lỗi

### Lỗi "HuggingFace token not provided"
→ Chưa set biến môi trường `HF_TOKEN`

### Lỗi "You need to accept the license"
→ Chưa chấp nhận license tại các link ở bước 3

### Lỗi "pyannote.audio not installed"
→ Chạy `pip install pyannote.audio`

## Tham khảo

- Paper: https://arxiv.org/abs/2310.00032 (Pyannote 3.1)
- GitHub: https://github.com/pyannote/pyannote-audio
- Benchmarks: DER ~11% (VoxConverse), ~19% (AMI), ~27% (DIHARD)
