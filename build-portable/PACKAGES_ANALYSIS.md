# Phân tích thư viện trong .envtietkiem

## Tổng quan

Build portable hiện tại copy toàn bộ thư viện từ `.envtietkiem/Lib/site-packages`. Dung lượng có thể giảm đáng kể nếu xóa các thư viện không dùng.

## Các thư viện CHẮC CHẮN KHÔNG DÙNG (có thể xóa an toàn)

### 1. AI Models không dùng
| Package | Dung lượng ước tính | Lý do |
|---------|---------------------|-------|
| `moonshine_voice` | ~50 MB | Đã exclude, dùng sherpa-onnx |
| `faster_whisper` | ~30 MB | Không dùng Whisper |
| `chunkformer` | ~5 MB | Chỉ dùng trong test |
| `silero_vad` | ~10 MB | Không dùng VAD này |
| `wespeakerruntime` | ~15 MB | Không dùng WeSpeaker runtime |

**Tổng: ~110 MB**

### 2. PyTorch Lightning (không dùng)
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `lightning` | ~20 MB | Không dùng PyTorch Lightning |
| `lightning_fabric` | ~10 MB | Không dùng |
| `lightning_utilities` | ~1 MB | Dependency |
| `pytorch_lightning` | ~15 MB | Không dùng |
| `pytorch_metric_learning` | ~5 MB | Không dùng metric learning |

**Tổng: ~50 MB**

### 3. Monitoring/Logging không cần
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `tensorboardX` | ~5 MB | Không logging |
| `opentelemetry*` | ~20 MB | Không tracing (7 packages) |
| `colorama` | ~0.5 MB | Terminal colors |
| `coloredlogs` | ~0.5 MB | Colored logging |
| `colorlog` | ~0.5 MB | Color logging |
| `humanfriendly` | ~1 MB | CLI friendly |
| `rich` | ~5 MB | Rich terminal |
| `pygments` | ~5 MB | Syntax highlighting |

**Tổng: ~35 MB**

### 4. Text processing không dùng
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `jiwer` | ~1 MB | WER metrics (chỉ đánh giá) |
| `langid` | ~2 MB | Language detection |
| `mosestokenizer` | ~1 MB | Moses tokenizer |
| `wtpsplit` | ~2 MB | Text splitting |
| `textgrid` | ~1 MB | TextGrid format |
| `markdown_it` | ~2 MB | Markdown parser |
| `mdurl` | ~0.5 MB | URL parser |

**Tổng: ~10 MB**

### 5. Database (chỉ dependency)
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `sqlalchemy` | ~10 MB | ORM (pyannote dep) |
| `alembic` | ~5 MB | DB migration |
| `greenlet` | ~1 MB | Coroutine |

**Tổng: ~15 MB**

### 6. Math/Utils không dùng
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `optuna` | ~10 MB | Hyperparameter tuning |
| `primePy` | ~0.5 MB | Prime numbers |
| `mpmath` | ~5 MB | Precision math |
| `skops` | ~2 MB | sklearn ops |
| `prettytable` | ~1 MB | CLI tables |

**Tổng: ~20 MB**

### 7. Video/Codegen không dùng
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `torchcodec` | ~10 MB | Video decoding |
| `torchgen` | ~5 MB | Codegen |

**Tổng: ~15 MB**

### 8. HTTP/gRPC không cần
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `grpc` | ~20 MB | gRPC (dùng HTTP) |
| `googleapis_common_protos` | ~5 MB | gRPC proto |
| `grpcio` | ~30 MB | gRPC core |

**Tổng: ~55 MB**

### 9. Image processing (nếu không plot)
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `PIL` (pillow) | ~15 MB | Image processing |
| `contourpy` | ~2 MB | Matplotlib dep |
| `kiwisolver` | ~1 MB | Matplotlib dep |
| `fontTools` | ~5 MB | Font tools |
| `cycler` | ~0.5 MB | Matplotlib dep |
| `pyparsing` | ~2 MB | Parsing |
| `matplotlib` | ~50 MB | **XEM XÉT** |
| `mpl_toolkits` | ~5 MB | Matplotlib |

**Tổng: ~80 MB (nếu xóa matplotlib)**

### 10. Audio augmentation
| Package | Dung lượng | Lý do |
|---------|------------|-------|
| `torch_audiomentations` | ~5 MB | Augmentation |
| `torch_pitch_shift` | ~2 MB | Pitch shift |
| `pyrubberband` | ~2 MB | Time-stretch |

**Tổng: ~10 MB**

---

## TỔNG KẾT

### Có thể xóa an toàn
| Nhóm | Dung lượng |
|------|------------|
| AI Models thừa | ~110 MB |
| PyTorch Lightning | ~50 MB |
| Monitoring/Logging | ~35 MB |
| Text processing | ~10 MB |
| Database | ~15 MB |
| Math/Utils | ~20 MB |
| Video/Codegen | ~15 MB |
| HTTP/gRPC | ~55 MB |
| Audio augmentation | ~10 MB |
| **TỔNG** | **~320 MB** |

### Tùy chọn (cần xem xét)
| Package | Dung lượng | Ghi chú |
|---------|------------|---------|
| `matplotlib` | ~50 MB | Có thể cần cho plotting |
| `pandas` | ~30 MB | Có thể cần cho data |

---

## Cách sử dụng

### 1. Chạy cleanup script
```bash
# Dọn dẹp tự động
python build-portable/cleanup_unused_packages.py
```

### 2. Manual cleanup (nếu script lỗi)
```bash
# Kích hoạt venv
.envtietkiem\Scripts\activate

# Xóa từng nhóm
pip uninstall -y moonshine_voice faster_whisper chunkformer silero_vad wespeakerruntime
pip uninstall -y lightning lightning_fabric lightning_utilities pytorch_lightning pytorch_metric_learning
pip uninstall -y tensorboardX opentelemetry opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
pip uninstall -y colorama coloredlogs colorlog humanfriendly rich pygments
pip uninstall -y jiwer langid mosestokenizer wtpsplit textgrid markdown-it-py mdurl
pip uninstall -y sqlalchemy alembic
pip uninstall -y optuna primePy mpmath skops prettytable
pip uninstall -y torchcodec torchgen
pip uninstall -y grpcio googleapis-common-protos
pip uninstall -y torch-audiomentations torch-pitch-shift pyrubberband

# Tùy chọn
pip uninstall -y matplotlib pillow pandas
```

### 3. Build portable sau khi dọn dẹp
```bash
build-portable\build.bat
```

---

## Lưu ý quan trọng

1. **Sao lưu**: Nên backup `.envtietkiem` trước khi xóa
2. **Test**: Sau khi xóa, test app kỹ trước khi build
3. **Matplotlib**: Nếu `audio_analyzer.py` có plotting thì giữ lại
4. **Pandas**: Nếu có xử lý DataFrame thì giữ lại
5. **Dependencies**: Một số package có thể là dependencies của nhau

## Kiểm tra nhanh

```python
# Test imports sau khi cleanup
python -c "
import numpy, torch, transformers, librosa, soundfile
import sherpa_onnx, pyannote.audio, pydub, sklearn
import PyQt6, psutil, sentencepiece, filelock
print('All critical imports OK!')
"
```
