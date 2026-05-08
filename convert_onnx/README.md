# Convert Models to ONNX — Loại bỏ hoàn toàn PyTorch

Hướng dẫn export tất cả model từ PyTorch sang ONNX Runtime.
Sau khi export, toàn bộ pipeline chạy **không cần PyTorch** (~2GB).

Dự án đã loại bỏ PyTorch khỏi 4 pipeline chính:

| Pipeline | Model gốc | Trạng thái | Script |
|----------|-----------|------------|--------|
| **Thêm dấu câu + viết hoa** | ViBERT ([dragonSwing/vibert-capu](https://huggingface.co/dragonSwing/vibert-capu)) | ONNX — xong (FP32 + INT8) | [`export_vibert_onnx.py`](export_vibert_onnx.py) |
| **Phân biệt người nói** | pyannote Community-1 (seg + emb + PLDA) | ONNX — xong (Pure ORT pipeline + masked-pool split) | [`split_pyannote_embedding.py`](split_pyannote_embedding.py) |
| **Speaker embedding (Senko)** | CAM++ 3D-Speaker 200k ([ModelScope](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)) | ONNX — xong | [`export_campplus_onnx.py`](export_campplus_onnx.py) |
| **Tách giọng overlap (2-speaker)** | Conv-TasNet ([JorisCos/ConvTasNet_Libri2Mix_sepclean_16k](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)) | ONNX — xong (FP32, INT8 broken) | [`export_convtasnet_onnx.py`](export_convtasnet_onnx.py) |

## Models đã upload lên HuggingFace

Tất cả 4 ONNX export đã được publish (license inherited từ source) — `prepare_offline_build.py` tự download và verify SHA256:

| HF Repo | License | Source |
|---------|---------|--------|
| [welcomyou/vibert-capu-onnx](https://huggingface.co/welcomyou/vibert-capu-onnx) | CC-BY-SA-4.0 | dragonSwing/vibert-capu |
| [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | Apache-2.0 | github.com/modelscope/3D-Speaker |
| [welcomyou/pyannote-community-1-onnx-split](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split) | CC-BY-4.0 | pyannote/speaker-diarization-community-1 (via altunenes ONNX) |
| [welcomyou/convtasnet-libri2mix-16k-onnx](https://huggingface.co/welcomyou/convtasnet-libri2mix-16k-onnx) | CC-BY-SA-4.0 | JorisCos/ConvTasNet_Libri2Mix_sepclean_16k (asteroid) |

Collection: [welcomyou/sherpa-vietnamese-asr](https://huggingface.co/collections/welcomyou/sherpa-vietnamese-asr-69fd2a837f318846d84c15d4)

---

# A. ViBERT — Thêm dấu câu + viết hoa

Export model ViBERT Seq2Labels (thêm dấu câu + viết hoa tiếng Việt) từ PyTorch sang ONNX.

## Tại sao cần ONNX?

| | PyTorch | ONNX Runtime |
|---|---------|-------------|
| Load time | ~6s | ~0.8s |
| Runtime dependency | torch (~2GB) | onnxruntime (~50MB) |
| Quantization | Cần code riêng | Tự tối ưu |
| Portable build | Rất nặng | Nhẹ |

## Bước 1: Tải model gốc

Model gốc: **[dragonSwing/vibert-capu](https://huggingface.co/dragonSwing/vibert-capu)** trên HuggingFace.

Base model: FPTAI/vibert-base-cased (12-layer BERT, 768 hidden, tiếng Việt).
Fine-tuned trên 5.6M mẫu văn bản OSCAR-2109 cho task thêm dấu câu + viết hoa.

### Cách tải

**Cách 1 — git clone (khuyến nghị):**

```bash
cd models/
git lfs install
git clone https://huggingface.co/dragonSwing/vibert-capu
```

**Cách 2 — huggingface-cli:**

```bash
pip install huggingface-hub
huggingface-cli download dragonSwing/vibert-capu --local-dir models/vibert-capu
```

**Cách 3 — Python:**

```python
from huggingface_hub import snapshot_download
snapshot_download("dragonSwing/vibert-capu", local_dir="models/vibert-capu")
```

Sau khi tải, thư mục phải có đủ:

```
models/vibert-capu/
    config.json             # Model config
    pytorch_model.bin       # PyTorch weights (~440 MB)
    vocab.txt               # BERT tokenizer vocabulary
```

## Bước 2: Cài dependencies (chỉ cần khi export)

```bash
pip install torch transformers onnxruntime numpy
```

> Sau khi export xong, runtime chỉ cần `onnxruntime`, `transformers` (tokenizer), `numpy`.
> Không cần `torch` nữa.

## Bước 3: Export sang ONNX

### Export cơ bản

```bash
python convert_onnx/export_vibert_onnx.py
```

Mặc định đọc từ `models/vibert-capu/` và ghi ra `models/vibert-capu/vibert-capu.onnx`.

### Export + verify (khuyến nghị)

```bash
python convert_onnx/export_vibert_onnx.py --verify
```

So sánh output ONNX vs PyTorch để đảm bảo kết quả khớp.

### Tuỳ chỉnh đường dẫn

```bash
python convert_onnx/export_vibert_onnx.py \
    --model_dir path/to/vibert-capu \
    --output path/to/output.onnx \
    --opset 14 \
    --verify
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--model_dir` | `models/vibert-capu` | Thư mục chứa `config.json`, `pytorch_model.bin`, `vocab.txt` |
| `--output` | `<model_dir>/vibert-capu.onnx` | Đường dẫn file ONNX output |
| `--opset` | 14 | ONNX opset version |
| `--verify` | off | So sánh output PyTorch vs ONNX |

## Kết quả sau khi export

```
models/vibert-capu/
    config.json             # Model config (giữ nguyên)
    pytorch_model.bin       # PyTorch weights (440 MB) — có thể xóa sau khi export
    vibert-capu.onnx        # ONNX model (438 MB) — dùng ở runtime
    vocab.txt               # BERT tokenizer vocabulary (giữ nguyên)
```

## Cấu trúc ONNX model

### Inputs (tất cả int64)

| Tên | Shape | Mô tả |
|-----|-------|-------|
| `input_ids` | (batch, seq_len) | Token IDs từ tokenizer |
| `attention_mask` | (batch, seq_len) | 1 = token thật, 0 = padding |
| `token_type_ids` | (batch, seq_len) | Segment IDs (luôn 0) |
| `input_offsets` | (batch, num_words) | Vị trí subword đầu tiên của mỗi từ |

### Outputs (float32)

| Tên | Shape | Mô tả |
|-----|-------|-------|
| `logits` | (batch, num_words, 15) | Xác suất 15 actions |
| `detect_logits` | (batch, num_words, 4) | Xác suất error detection |

### 15 label actions

```
$KEEP                      — Giữ nguyên
$TRANSFORM_CASE_CAPITAL    — Viết hoa chữ cái đầu (ví dụ: hà nội → Hà Nội)
$APPEND_,                  — Thêm dấu phẩy
$APPEND_.                  — Thêm dấu chấm
$TRANSFORM_VERB_VB_VBN     — (không dùng cho tiếng Việt)
$TRANSFORM_CASE_UPPER      — Viết hoa toàn bộ (ví dụ: who → WHO)
$APPEND_:                  — Thêm dấu hai chấm
$APPEND_?                  — Thêm dấu hỏi
$TRANSFORM_VERB_VB_VBC     — (không dùng cho tiếng Việt)
$TRANSFORM_CASE_LOWER      — Viết thường
$TRANSFORM_CASE_CAPITAL_1  — Viết hoa ký tự thứ 2
$TRANSFORM_CASE_UPPER_-1   — Viết hoa trừ ký tự cuối
$MERGE_SPACE               — Nối từ
@@UNKNOWN@@                — Unknown token
@@PADDING@@                — Padding token
```

## Gọi từ code khác

```python
from convert_onnx.export_vibert_onnx import export

export(
    model_dir="models/vibert-capu",
    output_path="models/vibert-capu/vibert-capu.onnx",
    verify=True,
)
```

---

# B. Pyannote Community-1 — Phân biệt người nói (Speaker Diarization)

Pipeline gốc **pyannote/speaker-diarization-community-1** gồm 3 model PyTorch + thuật toán VBx clustering.
Đã được chuyển toàn bộ sang pure ONNX Runtime + NumPy/SciPy, không còn phụ thuộc `torch` hay `pyannote.audio`.

## Source code đang dùng

| File | Vai trò |
|------|---------|
| `core/speaker_diarization.py` | Entry point — dispatcher, gọi Pure ORT backend |
| `core/speaker_diarization_pure_ort.py` | **Implementation chính** — toàn bộ pipeline Pure ORT (884 dòng) |

Các file legacy (không dùng ở runtime, giữ để tham khảo):

| File | Vai trò |
|------|---------|
| `core/speaker_diarization_pyannote.py` | Bản gốc dùng PyTorch + pyannote.audio |
| `core/speaker_diarization_onnx_altunenes.py` | Hybrid ONNX + pyannote (bước chuyển tiếp) |

## Tổng quan pipeline

```
Audio (16kHz mono)
  │
  ├── [1] Segmentation ONNX ──→ Ai nói ở đâu? (VAD + speaker boundaries)
  │       segmentation-community-1.onnx (5.7 MB)
  │
  ├── [2] Embedding ONNX ─────→ Giọng mỗi người khác nhau thế nào?
  │       embedding_encoder.onnx (26 MB) + Gemm weights (5 MB)
  │       hoặc embedding_model.onnx (26 MB) — full model
  │
  ├── [3] PLDA + VBx ─────────→ Gom nhóm: đoạn nào cùng 1 người?
  │       plda.npz + xvec_transform.npz (0.3 MB)
  │       Thuật toán: numpy + scipy (không cần torch)
  │
  └── [4] Reconstruction ─────→ Output: [(start, end, speaker_id), ...]
```

## Nguồn gốc — PyTorch phụ thuộc gì?

Pipeline gốc `pyannote.audio` kéo theo:

| Package | Kích thước | Vai trò | Thay thế bằng |
|---------|-----------|---------|----------------|
| `torch` | ~2 GB | Chạy segmentation + embedding model | `onnxruntime` (~50 MB) |
| `torchaudio` | ~10 MB | Load audio, mel spectrogram | `soundfile` + `librosa` |
| `pyannote.audio` | ~50 MB | Pipeline orchestration, VBx, binarize | `numpy` + `scipy` (code tự viết) |
| `pyannote.core` | ~5 MB | SlidingWindow, Segment, Annotation | Tự viết class nhỏ (~50 dòng) |
| `pytorch-lightning` | ~30 MB | Dependency của pyannote | Không cần |

**Tổng loại bỏ: ~2.1 GB dependencies.**

## Bước 1: Tải ONNX models (đã convert sẵn)

### Segmentation + Embedding ONNX

Tải trực tiếp từ HuggingFace — **đã được convert sẵn**, không cần tự export:

```bash
cd models/
git lfs install
git clone https://huggingface.co/altunenes/speaker-diarization-community-1-onnx pyannote-onnx
```

Hoặc:

```bash
huggingface-cli download altunenes/speaker-diarization-community-1-onnx --local-dir models/pyannote-onnx
```

Kết quả:

```
models/pyannote-onnx/
    segmentation-community-1.onnx    # 5.7 MB — Voice segmentation
    embedding_model.onnx             # 26 MB  — Full speaker embedding
    embedding_encoder.onnx           # 26 MB  — Encoder-only (tối ưu hơn)
    embedding_model_split.onnx       # 26 MB  — Variant khác
    resnet_seg_1_weight.npy          # 5.1 MB — Gemm projection weight
    resnet_seg_1_bias.npy            # 1.2 KB — Gemm projection bias
```

### PLDA + config (từ model gốc pyannote)

```bash
huggingface-cli download pyannote/speaker-diarization-community-1 --local-dir models/pyannote/speaker-diarization-community-1
```

> Cần chấp nhận license tại https://huggingface.co/pyannote/speaker-diarization-community-1

Chỉ cần 3 file (PyTorch .bin **không cần** ở runtime):

```
models/pyannote/speaker-diarization-community-1/
    config.yaml                           # Pipeline params (threshold, Fa, Fb)
    plda/plda.npz                         # 0.13 MB — PLDA model params
    plda/xvec_transform.npz              # 0.13 MB — X-vector transform
    segmentation/pytorch_model.bin        # ❌ KHÔNG CẦN — đã có ONNX
    embedding/pytorch_model.bin           # ❌ KHÔNG CẦN — đã có ONNX
```

## Bước 2: Runtime dependencies

Chỉ cần (không cần `torch`, `pyannote.audio`):

```
onnxruntime          # ONNX inference
numpy                # Array operations
scipy                # VBx clustering, PLDA, linear algebra
kaldi-native-fbank   # Trích xuất fbank features (C++ library, ~2 MB)
soundfile            # Đọc audio
librosa              # Resample audio
```

## Chi tiết từng model ONNX

### Segmentation model

| | Chi tiết |
|---|---------|
| **File** | `segmentation-community-1.onnx` (5.7 MB) |
| **Nguồn** | pyannote/segmentation-3.0 (PyanNet architecture) |
| **Input** | `(batch, 1, 160000)` — 10 giây audio 16kHz mono |
| **Output** | `(batch, 589, 7)` — 589 frames x 7 classes (powerset encoding) |
| **7 classes** | 0=Silence, 1=Spk0, 2=Spk1, 3=Spk2, 4=Spk0+1, 5=Spk0+2, 6=Spk1+2 |
| **Xử lý** | Sliding window 10s, step 1s, batch_size=32, aggregate overlaps |

Cách export (nếu cần tự convert):

```python
# Cần: pip install pyannote.audio torch onnx
from pyannote.audio import Model
import torch

model = Model.from_pretrained("segmentation/pytorch_model.bin")
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 1, 160000),  # 10s @ 16kHz
    "segmentation-community-1.onnx",
    opset_version=13,
    input_names=["x"], output_names=["y"],
    dynamic_axes={"x": {0: "N", 2: "T"}, "y": {0: "N", 1: "T"}},
)
```

Script đầy đủ với metadata: `models/speaker_diarization/sherpa-onnx-pyannote-segmentation-3-0/export-onnx.py`

### Embedding model (WeSpeaker ResNet34)

| | Chi tiết |
|---|---------|
| **File** | `embedding_model.onnx` (26 MB) hoặc `embedding_encoder.onnx` + `.npy` |
| **Nguồn** | WeSpeaker ResNet34 (altunenes convert) |
| **Input** | `(batch, num_frames, 80)` — 80-dim fbank features |
| **Output** | `(batch, 256)` — L2-normalized speaker embedding |
| **Feature extraction** | kaldi-native-fbank: 25ms frame, 10ms shift, Hamming, 80 mel bins |
| **Normalization** | Per-utterance CMVN (trừ mean) |

**Hai variant:**

| Variant | File | Ưu điểm |
|---------|------|---------|
| **Full** | `embedding_model.onnx` | Đơn giản, 1 file duy nhất |
| **Split** | `embedding_encoder.onnx` + `resnet_seg_1_weight.npy` + `resnet_seg_1_bias.npy` | Nhanh hơn — tách encoder ra để batch tốt hơn, stats pooling + Gemm projection bằng NumPy |

Code hiện tại ưu tiên **split variant** (nếu có `embedding_encoder.onnx`).

### PLDA model

| | Chi tiết |
|---|---------|
| **Files** | `plda.npz` (0.13 MB) + `xvec_transform.npz` (0.13 MB) |
| **Format** | NumPy `.npz` — load trực tiếp, không cần torch |
| **Vai trò** | Transform embeddings → PLDA space → VBx clustering |
| **Params** | `xvec_transform.npz`: mean1, mean2, LDA matrix. `plda.npz`: mu, psi, sigma |

## Các thuật toán đã thay thế pyannote bằng numpy/scipy

Toàn bộ logic dưới đây được viết lại trong `core/speaker_diarization_pure_ort.py`:

| Chức năng pyannote | Thay thế bằng | Mô tả |
|---------------------|---------------|-------|
| `SlidingWindow` | `_SW` class (~20 dòng) | Mapping frame ↔ thời gian |
| `Inference.aggregate()` | `_pyannote_aggregate()` | Gộp overlapping chunk predictions |
| `Binarize` (hysteresis) | `_binarize()` | onset/offset thresholding |
| `VBxClustering` | `vbx_cluster()` | Variational Bayes clustering |
| `PLDA transform` | `xvec_transform()` + `plda_transform()` | Embedding → PLDA space |
| `to_diarization()` | `_reconstruct_and_diarize()` | Cluster labels → time segments |
| `speaker_count()` | NumPy operations | Đếm số speaker tức thời |
| `AHC init` | `scipy.cluster.hierarchy.linkage` + `fcluster` | Khởi tạo clusters cho VBx |
| Hungarian assignment | `scipy.optimize.linear_sum_assignment` | Gán speaker ID nhất quán |

## Pipeline params (từ config.yaml)

```yaml
params:
  clustering:
    threshold: 0.6    # Ngưỡng khoảng cách VBx (thấp = nhiều speaker hơn)
    Fa: 0.07          # False alarm weight
    Fb: 0.8           # False negative weight
  segmentation:
    min_duration_off: 0.0  # Gap tối thiểu giữa 2 segment
```

Code hiện tại dùng `threshold=0.7` (hơi cao hơn mặc định cho kết quả sạch hơn).

## So sánh hiệu năng

| | PyTorch (pyannote) | Pure ORT |
|---|---------------------|----------|
| **Dependencies** | torch + pyannote (~2.1 GB) | onnxruntime + scipy (~60 MB) |
| **RAM** | ~2-3 GB | ~500 MB |
| **Tốc độ** | RTF 0.4-0.6 | RTF 0.3-0.4 |
| **Chất lượng (DER)** | 11-19% | Tương đương (cùng thuật toán) |
| **Portable build** | Rất nặng | Nhẹ |

---

# C. Tổng kết — Dependencies runtime (không cần PyTorch)

Sau khi export xong tất cả model, runtime chỉ cần:

```
# Core inference
onnxruntime              # ~50 MB — Chạy tất cả ONNX models
numpy                    # ~25 MB — Array operations
scipy                    # ~40 MB — Clustering, linear algebra

# Audio
soundfile                # ~1 MB  — Đọc WAV/FLAC
librosa                  # ~5 MB  — Resample audio
kaldi-native-fbank       # ~2 MB  — Feature extraction (C++)

# Tokenizer (cho ViBERT)
transformers             # ~20 MB — AutoTokenizer (KHÔNG dùng torch models)

# KHÔNG CẦN:
# torch                  ❌ (~2 GB)
# torchaudio             ❌
# pyannote.audio         ❌
# pyannote.core          ❌
# pytorch-lightning      ❌
```
