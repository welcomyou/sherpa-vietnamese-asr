# Sherpa Vietnamese ASR

Ứng dụng nhận dạng giọng nói tiếng Việt offline cho Windows. Bản mặc định chạy CPU-only để tương thích rộng, không cần GPU và không gửi âm thanh ra internet. Từ bản 2.6.2, Desktop và Web Service có thể cài thêm GPU add-on rồi bấm **Tối ưu thiết bị** để benchmark và tự chọn GPU theo từng stage nếu thật sự nhanh hơn CPU.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)
![Version](https://img.shields.io/badge/Version-2.6.3-orange)
![Runtime](https://img.shields.io/badge/Runtime-ONNX%20Runtime-success)

## Gói phát hành

Tải đúng các file cùng version trong [GitHub Releases](https://github.com/welcomyou/sherpa-vietnamese-asr/releases).

| File | Dùng cho | Ghi chú |
|---|---|---|
| `sherpa-vietnamese-asr-2.6.3.zip` | Desktop app | Base CPU-only, có GUI PyQt6, ffmpeg/ffprobe và model CPU |
| `sherpa-vietnamese-asr-service-2.6.3.zip` | Web Service + PWA offline | Base CPU-only, có FastAPI, Web UI, PWA offline, ffmpeg/ffprobe |
| `gpu-addon-directml-win64-2.6.3.zip` | GPU NVIDIA hoặc AMD | ONNX Runtime DirectML, app-local add-on |
| `gpu-addon-intel-openvino-win64-2.6.3.zip` | Intel GPU / Intel iGPU | ONNX Runtime OpenVINO, app-local add-on |
| `gpu-models-win64-2.6.3.zip` | Model cần cho GPU | Hiện chứa `models/vibert-capu/vibert-capu.onnx`; CAM++ GPU graph tự sinh khi cần |

Base zip là gói cần tải đầu tiên. GPU add-on và GPU models chỉ cần tải khi màn hình **Tối ưu thiết bị** báo máy có GPU phù hợp và cần cài thêm.

## Cách chạy bản phát hành

Desktop:

```text
sherpa-vietnamese-asr-2.6.3\sherpa-vietnamese-asr.bat
```

Web Service:

```text
sherpa-vietnamese-asr-service-2.6.3\sherpa-vietnamese-asr-service.bat
```

Web Service mặc định chạy HTTPS tại `https://<IP-may-chu>:8443`. Tài khoản admin mặc định: `admin` / `admin`.

## Cài GPU add-on

Ví dụ bạn đã giải nén desktop vào:

```text
D:\Apps\sherpa-vietnamese-asr-2.6.3\
```

Khi cài add-on, hãy giải nén zip vào đúng thư mục gốc này, tức là sau khi giải nén phải có các đường dẫn sau:

```text
D:\Apps\sherpa-vietnamese-asr-2.6.3\gpu_addons\directml\Lib\site-packages\onnxruntime\
D:\Apps\sherpa-vietnamese-asr-2.6.3\models\vibert-capu\vibert-capu.onnx
```

Với Intel OpenVINO, đường dẫn add-on phải là:

```text
D:\Apps\sherpa-vietnamese-asr-2.6.3\gpu_addons\intel-openvino\Lib\site-packages\onnxruntime\
```

Không giải nén kiểu tạo thêm một lớp thư mục như:

```text
D:\Apps\sherpa-vietnamese-asr-2.6.3\gpu-addon-directml-win64-2.6.3\gpu_addons\...
```

Sau khi cài add-on, đóng hẳn app, mở lại, bấm **Tối ưu thiết bị**. Nếu muốn quay về CPU, chọn **CPU-only** trong hộp thoại tối ưu. Kết quả calibration đã lưu sẽ không bị xóa khi chuyển qua lại giữa CPU-only và GPU auto.

## GPU policy

| Phần cứng | Provider ưu tiên | Gói cần tải |
|---|---|---|
| Không có GPU | CPUExecutionProvider | Không cần gói GPU |
| NVIDIA GPU | DirectML | `gpu-addon-directml-win64-2.6.3.zip` + `gpu-models-win64-2.6.3.zip` nếu app báo thiếu |
| AMD GPU / AMD iGPU | DirectML | `gpu-addon-directml-win64-2.6.3.zip` + `gpu-models-win64-2.6.3.zip` nếu app báo thiếu |
| Intel GPU / Intel iGPU | OpenVINO | `gpu-addon-intel-openvino-win64-2.6.3.zip` + `gpu-models-win64-2.6.3.zip` nếu app báo thiếu |
| AMD CPU, không có GPU | CPUExecutionProvider | Không cần gói GPU |

Calibration chỉ chọn GPU khi inference nhanh hơn CPU ít nhất 20% và sai khác số học nằm trong ngưỡng cho phép. Nếu GPU không đạt, stage đó vẫn chạy CPU.

## Stage GPU hiện tại

Các stage có thể được benchmark GPU:

- Speaker Diarization: CAM++ embedding. Model GPU được sinh tự động từ model CPU khi cần.
- Speaker Diarization: Pyannote embedding encoder.
- DNSMOS quality.
- Punctuation: ViBERT FP32.

Các stage giữ CPU theo thiết kế hiện tại:

- ASR encoder / decoder / joiner.
- Audio decode / resample bằng ffmpeg/ffprobe.
- Silero VAD.
- Speaker segmentation, VBx, clustering và postprocess.

Số tốc độ trong hộp thoại **Tối ưu thiết bị** là thời gian inference riêng của từng stage, không phải tổng thời gian xử lý toàn bộ file.

## Tính năng chính

- Chuyển giọng nói thành văn bản từ MP3, WAV, M4A, FLAC, AAC, OGG, MP4, MKV, AVI, MOV, WEBM.
- 3 chế độ ASR: Zipformer 30M, Zipformer 68M, ROVER.
- Speaker diarization bằng Pure ONNX Runtime: Pyannote Community-1 và Senko CAM++.
- Tách overlap 2 người nói bằng Conv-TasNet ONNX.
- Tự động thêm dấu câu và viết hoa bằng ViBERT-capu ONNX.
- Hotwords cho tên riêng và thuật ngữ chuyên ngành.
- Đánh giá chất lượng âm thanh bằng DNSMOS.
- Desktop có thu âm trực tiếp, chỉnh tên/gộp/tách người nói, tua theo câu, theme sáng/tối.
- Web Service có Web UI, queue, tài khoản, admin GUI, Windows Service và PWA offline.

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| ASR | Sherpa-ONNX, Zipformer RNN-T 30M + 68M |
| Diarization | Pyannote Community-1 + Senko CAM++ bằng Pure ONNX Runtime |
| Speaker embedding | CAM++ 192-dim, Pyannote/WeSpeaker ResNet34-LM |
| Overlap separation | Conv-TasNet 16k Libri2Mix ONNX |
| NaturalTurn | Backchannel detection theo Cychosz et al. 2025 |
| Dấu câu | ViBERT-capu ONNX FP32 / INT8 |
| VAD | Silero VAD + Pyannote Segmentation ONNX |
| Audio quality | Microsoft DNSMOS ONNX |
| Resampling | SoXR HQ/VHQ |
| GPU acceleration | ONNX Runtime DirectML, OpenVINO; CUDA không publish mặc định từ 2.6.2 |
| Desktop GUI | PyQt6 |
| Web backend | FastAPI, WebSocket, SQLite |
| PWA offline | ONNX Runtime Web, WASM/WebGPU |
| Summarizer | Gemma 4 E2B GGUF qua llama-cpp-python |

## PWA offline

PWA offline chạy trong trình duyệt bằng WASM/WebGPU. PWA không dùng GPU add-on của desktop/server, không cần DirectML hoặc OpenVINO. Khi trình duyệt hỗ trợ WebGPU, PWA tự dùng WebGPU cho những phần đã được frontend hỗ trợ; nếu không, PWA tự fallback về WASM/CPU.

## Yêu cầu

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| OS | Windows 10 64-bit | Windows 10/11 64-bit |
| RAM | 8 GB | 16 GB+ |
| CPU | Intel i3 / Ryzen 3 | Intel i7 / Ryzen 7+ |
| Storage | 2 GB+ | 5 GB+ |

Máy đích không cần cài Python khi dùng bản phát hành zip.

## Chạy từ source

```bash
git clone https://github.com/welcomyou/sherpa-vietnamese-asr.git
cd sherpa-vietnamese-asr
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-online.txt
```

```bash
python app.py
python server_gui.py
python server_launcher.py --no-gui
```

## Build

Chuẩn bị môi trường và model:

```bash
python build-portable/setup_build_env.py
python build-portable/prepare_offline_build.py
```

Build base CPU-only:

```bash
python build-portable/build_portable.py
python build-portable/build_portable_online.py
```

Build GPU add-on và model GPU:

```bash
python build-portable/build_gpu_addons.py directml intel-openvino
python build-portable/build_gpu_models.py
```

Output nằm trong `dist/`:

```text
sherpa-vietnamese-asr-<version>.zip
sherpa-vietnamese-asr-service-<version>.zip
gpu-addon-directml-win64-<version>.zip
gpu-addon-intel-openvino-win64-<version>.zip
gpu-models-win64-<version>.zip
```

## Model nguồn

| Repo | License | Source |
|---|---|---|
| [welcomyou/vibert-capu-onnx](https://huggingface.co/welcomyou/vibert-capu-onnx) | CC-BY-SA-4.0 | dragonSwing/vibert-capu |
| [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx) | Apache-2.0 | 3D-Speaker CAM++ |
| [welcomyou/pyannote-community-1-onnx-split](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split) | CC-BY-4.0 | pyannote Community-1 |
| [welcomyou/convtasnet-libri2mix-16k-onnx](https://huggingface.co/welcomyou/convtasnet-libri2mix-16k-onnx) | CC-BY-SA-4.0 | JorisCos Conv-TasNet |

Script tải model có pin SHA256 trong `build-portable/prepare_offline_build.py`.

## Ghi nhận

### Thư viện & Models

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) — ASR engine.
- [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) — Vietnamese ASR model 30M.
- [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) — Vietnamese ASR model 68M.
- [ViBERT-capu](https://huggingface.co/dragonSwing/vibert-capu) — punctuation restoration; bản ONNX convert publish tại [welcomyou/vibert-capu-onnx](https://huggingface.co/welcomyou/vibert-capu-onnx).
- [Senko](https://github.com/narcotic-sh/senko) — speaker diarization pipeline với CAM++ và clustering.
- [Pyannote](https://github.com/pyannote/pyannote-audio) — speaker diarization pipeline Community-1.
- [altunenes/speaker-diarization-community-1-onnx](https://huggingface.co/altunenes/speaker-diarization-community-1-onnx) — ONNX models cho Pyannote Community-1.
- [3D-Speaker](https://github.com/modelscope/3D-Speaker) — CAM++ 192-dim speaker embedding; bản ONNX convert publish tại [welcomyou/campplus-3dspeaker-200k-onnx](https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx).
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) — ResNet34-LM speaker embedding.
- [Asteroid](https://github.com/asteroid-team/asteroid) và [JorisCos/ConvTasNet_Libri2Mix_sepclean_16k](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k) — overlap separation; bản ONNX convert publish tại [welcomyou/convtasnet-libri2mix-16k-onnx](https://huggingface.co/welcomyou/convtasnet-libri2mix-16k-onnx).
- [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge) — audio quality assessment.
- [ONNX Runtime](https://onnxruntime.ai/) — CPU, DirectML, OpenVINO và Web inference runtime.
- [FFmpeg](https://ffmpeg.org/) — decode, probe và chuyển đổi audio/video.
- [SoX Resampler Library](https://sourceforge.net/projects/soxr/) — resampling chất lượng cao.
- [FastAPI](https://fastapi.tiangolo.com/), [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) và [llama.cpp](https://github.com/ggml-org/llama.cpp) — web service, desktop GUI và local summarizer.

### Papers

- Cychosz et al., "Natural conversational turn-taking" — *Scientific Reports* 2025 ([doi](https://www.nature.com/articles/s41598-025-24381-1)) — NaturalTurn backchannel detection.
- Bredin et al., "Pyannote.audio 2.1" — *arXiv* 2023 ([2310.00032](https://arxiv.org/abs/2310.00032)) — speaker diarization pipeline.
- Chen et al., "3D-Speaker" — *arXiv* 2024 ([2403.19971](https://arxiv.org/abs/2403.19971)) — CAM++ speaker embedding và clustering.
- Prabhavalkar et al., "Automatic gain control and multi-style training for robust ASR" — *ICASSP* 2015 ([pdf](https://research.google.com/pubs/archive/43289.pdf)) — per-segment RMS normalization.

## License

[MIT License](LICENSE)

Lưu ý: PyQt6 dùng GPL v3. Nếu cần closed-source thương mại, hãy dùng license thương mại của Qt hoặc thay bằng PySide6/LGPL.
