# Project Rules & System Documentation

## 1. Tổng quan (Overview)
Phần mềm là một ứng dụng **Speech-to-Text (ASR) Offline** dành cho tiếng Việt, hỗ trợ cả xử lý file âm thanh (Offline) và nhận dạng trực tiếp qua Microphone (Live). Ứng dụng được tối ưu cho máy tính cấu hình thấp, chạy hoàn toàn offline trên CPU, không cần GPU.

## 2. Giao diện (User Interface)

### 2.1. Cửa sổ chính (`Main Window` - `app.py`)
Giao diện chính được chia thành các tab chức năng:

*   **Tab "Xử lý tập tin" (Offline Processing):**
    *   **Khu vực nhập liệu:** `DragDropLabel` - Cho phép kéo thả hoặc bấm để chọn file âm thanh.
    *   **Khu vực hiển thị:** `ClickableTextEdit` - Hiển thị văn bản kết quả dưới dạng hội thoại (người nói + nội dung). Hỗ trợ click vào câu để nghe lại đoạn âm thanh tương ứng.
    *   **Thanh điều khiển (Media Player):** Play/Pause, Seek bar, Volume để nghe lại file đã tải lên.
    *   **Thanh Search (`SearchWidget`):** Tìm kiếm nội dung trong văn bản (hỗ trợ tìm kiếm có dấu/không dấu - fuzzy search).
    *   **Panel Cấu hình (Collapsible):**
        *   Chọn Model (Zipformer, v.v.).
        *   Số luồng CPU (`cpu_threads`).
        *   Ngưỡng tách câu (SAT Threshold).
        *   Mức độ thêm dấu câu (Punctuation Confidence).

*   **Tab "Thu âm trực tiếp" (Live Recording):**
    *   Sử dụng cơ chế Dual-Stream (Draft/Commit) để hiển thị văn bản theo thời gian thực.
    *   Hiển thị dạng dòng chảy văn bản (streaming text).

### 2.2. Các Widget Tùy chỉnh
*   **DragDropLabel:** Widget nhận sự kiện kéo thả file âm thanh.
*   **ClickableTextEdit:** `QTextEdit` tùy biến, bắt sự kiện click vào anchor `s_{index}` để điều khiển Media Player nhảy đến thời gian của câu.
*   **SearchWidget:** Widget tìm kiếm nhỏ gọn, nổi (floating) hoặc gắn liền, cho phép tìm kiếm và highlight kết quả.

## 3. Thuật toán & Luồng xử lý (Algorithms & Workflows)

### 3.1. Pipeline Xử lý File (Offline ASR)
Quy trình xử lý file được thực hiện trong `TranscriberThread` (`transcriber.py`):

1.  **Input Loading & Preprocessing:**
    *   Đọc file âm thanh (hỗ trợ mp3, m4a, wav, ogg...).
    *   Nếu file nén (mp3, m4a), dùng `ffmpeg` (qua `pydub`) convert sang WAV 16kHz mono.
    *   **Smart Splitting:** Phân tích năng lượng âm thanh để tìm khoảng lặng (Silence Detection), chia file lớn thành các segment nhỏ (khoảng 30s) để tránh tràn RAM và giới hạn context của model.

2.  **Speech Recognition (ASR):**
    *   Core: **Sherpa-ONNX** với model **Zipformer** (Transducer).
    *   Decoding Method: `greedy_search`.
    *   Output: Raw text & Token timestamps.

3.  **Speaker Diarization (Phân tách người nói) - *Optional*:**
    *   Module: `speaker_diarization.py`.
    *   Model: `titanet_small` hoặc `eres2netv2` (Embeddings).
    *   Logic:
        *   Sử dụng model Segmentation để tìm các đoạn có tiếng nói.
        *   Trích xuất Embedding vector cho từng đoạn.
        *   Clustering (Gom nhóm) các vector để xác định người nói (Speaker 1, Speaker 2...).
        *   Merge: Gán nhãn người nói vào các đoạn văn bản ASR dựa trên độ chồng lấn thời gian (Overlap).

4.  **Post-Process (Hậu xử lý):**
    *   **Punctuation Restoration:** Dùng model BERT (`punctuation_restorer_improved.py`) để thêm dấu câu (chấm, phẩy) vào văn bản raw.
    *   **Sentencing (SAT Algorithm):** Tách đoạn văn thành các câu dựa trên ngữ nghĩa và thời gian (nếu dùng SAT Pipeline).
    *   **Alignment:** Căn chỉnh lại thời gian (timestamps) của từng câu dựa trên timestamp của từng từ.

5.  **Output Generation:**
    *   Tạo HTML formatting cho cuộc hội thoại.
    *   Hiển thị lên UI.

### 3.2. Pipeline Thu âm trực tiếp (Live ASR)
Quy trình xử lý thu âm thực hiện trong `streaming_asr.py`:

*   **Dual-Stream Architecture:**
    1.  **Draft Stream:** Chạy nhanh, độ trễ thấp, đưa ra kết quả tạm thời (có thể sai sót). Model: *Reduced/Quantized Zipformer*.
    2.  **Commit Stream:** Chạy chậm hơn, chính xác hơn, "chốt" kết quả cuối cùng. Sử dụng logic **Text Stabilization** (so khớp chuỗi chồng lấn) để sửa lại kết quả từ Draft stream.
*   **VAD (Voice Activity Detection):** Sử dụng `Silero VAD` để phát hiện tiếng nói, cắt bỏ khoảng lặng đầu vào.
*   **Noise Reduction:** Hỗ trợ `DeepFilterNet` để lọc ồn môi trường (nếu có binary `deep-filter.exe`).

## 4. Input & Output

### 4.1. Input
*   **File:** Hỗ trợ đa định dạng (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.wma`, `.flac`, `.aac`, `.opus`).
    *   Tự động convert về chuẩn: 16kHz, 1 channel (Mono), 16-bit PCM.
*   **Microphone:** Audio Stream (16kHz, Mono).

### 4.2. Output
*   **Giao diện (Visual):** Văn bản hiển thị dạng hội thoại (Chat view), phân màu theo người nói. Highlight từ/câu đang phát.
*   **HTML:** Cấu trúc HTML nội bộ dùng để hiển thị trong `QTextEdit` (có thể copy-paste sang Word/Docs vẫn giữ format).
*   **Data Structure (Internal):** Danh sách các segments:
    ```json
    {
      "text": "Nội dung câu nói",
      "start": 10.5,    // Thời gian bắt đầu (giây)
      "end": 15.2,      // Thời gian kết thúc
      "speaker": "Người nói 1",
      "speaker_id": 0
    }
    ```

## 5. Chức năng phần mềm (Software Functions)
1.  **Chuyển đổi Tiếng nói thành Văn bản (ASR):** Độ chính xác cao với tiếng Việt, tối ưu cho giọng đời sống/hội họp.
2.  **Phân tách người nói (Speaker Diarization):** Tự động nhận diện có bao nhiêu người và ai đang nói câu nào.
3.  **Thêm dấu câu tự động:** Biến văn bản thô thành câu hoàn chỉnh đúng ngữ pháp.
4.  **Phát lại & Đồng bộ (Playback Sync):** Nghe lại file âm thanh và highlight chạy theo lời thoại. Click vào chữ để tua đến đoạn âm thanh.
5.  **Thu âm trực tiếp (Live Dictation):** Hỗ trợ gõ văn bản bằng giọng nói thời gian thực.
6.  **Xử lý Offline:** An toàn dữ liệu, không gửi audio lên server, chạy được khi không có internet.
