"""
Pyannote Community-1 Speaker Diarization with ONNX Inference

Uses the official pyannote.audio pipeline (VBx clustering, PLDA scoring,
masked pooling, Hamming-weighted reconstruction) but replaces PyTorch model
inference with ONNX Runtime for faster CPU performance (~20-40% speedup).

Architecture:
  1. Segmentation: segmentation-community-1.onnx (powerset, 7 classes)
  2. Embedding: embedding_model.onnx (WeSpeaker, 256-dim)
  3. Clustering: VBx (Variational Bayes) with PLDA scoring — from pyannote
  4. Reconstruction: Hamming-weighted + masked pooling — from pyannote

ONNX Models: https://huggingface.co/altunenes/speaker-diarization-community-1-onnx
Pipeline: https://huggingface.co/pyannote/speaker-diarization-community-1
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, cast
import time
from dataclasses import dataclass

_onnx_available = False
_kaldi_available = False
_pyannote_available = False

def _check_deps():
    global _onnx_available, _kaldi_available, _pyannote_available
    try:
        import onnxruntime as ort
        _onnx_available = True
    except ImportError:
        pass
    try:
        import kaldi_native_fbank as knf
        _kaldi_available = True
    except ImportError:
        pass
    try:
        import pyannote.audio
        _pyannote_available = True
    except ImportError:
        pass

_check_deps()

def decode_powerset_soft(logits):
    """Decode pyannote powerset log-softmax to 3 independent soft probability tracks"""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e_x / e_x.sum(axis=1, keepdims=True)

    # Powerset mapping for 3 speakers:
    # 0: [], 1: [0], 2: [1], 3: [2], 4: [0,1], 5: [0,2], 6: [1,2]
    p_0 = probs[:, 1] + probs[:, 4] + probs[:, 5]
    p_1 = probs[:, 2] + probs[:, 4] + probs[:, 6]
    p_2 = probs[:, 3] + probs[:, 5] + probs[:, 6]
    return np.stack([p_0, p_1, p_2], axis=1)

def decode_powerset_speaker_count(logits):
    """Estimate instantaneous speaker count from powerset probabilities."""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e_x / e_x.sum(axis=1, keepdims=True)

    speaker_counts = np.array([0, 1, 1, 1, 2, 2, 2], dtype=np.float32)
    expected_count = probs @ speaker_counts
    return expected_count

def find_continuous_regions(binary_array):
    """Find start and end indices of contiguous blocks of 1s"""
    arr = binary_array.astype(np.int8)
    padded = np.pad(arr, (1, 1), mode='constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts, ends))

def get_longest_region(regions):
    """Get the longest continuous region"""
    if not regions:
        return 0.0, None
    durs = [(e - s, (s, e)) for s, e in regions]
    durs.sort(key=lambda x: x[0], reverse=True)
    return durs[0][0], durs[0][1]

def hungarian_assignment(cost_matrix):
    """Solve assignment problem using Hungarian algorithm."""
    try:
        from scipy.optimize import linear_sum_assignment
        n_tracks, n_speakers = cost_matrix.shape

        if n_tracks <= n_speakers:
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            return list(zip(row_ind, col_ind))
        else:
            # More tracks than speakers - assign best matches first
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            assignments = list(zip(row_ind, col_ind))

            # Assign remaining tracks to best available
            assigned_tracks = set(row_ind)
            for ti in range(n_tracks):
                if ti not in assigned_tracks:
                    best_speaker = np.argmax(cost_matrix[ti])
                    assignments.append((ti, best_speaker))

            return assignments
    except ImportError:
        # Fallback to greedy
        return greedy_assignment(cost_matrix)

def greedy_assignment(cost_matrix):
    """Fallback greedy assignment."""
    n_tracks, n_speakers = cost_matrix.shape
    used_speakers = set()
    assignments = []

    track_order = sorted(range(n_tracks),
                        key=lambda ti: np.max(cost_matrix[ti]), reverse=True)

    for ti in track_order:
        available_scores = cost_matrix[ti].copy()
        for s in used_speakers:
            available_scores[s] = -1.0
        best_speaker = int(np.argmax(available_scores))
        if available_scores[best_speaker] > -0.5:
            assignments.append((ti, best_speaker))
            used_speakers.add(best_speaker)

    return assignments


def merge_short_islands(segments: List[Dict], max_short_duration: float = 1.5) -> List[Dict]:
    """
    Gộp các đoạn ngắn (< max_short_duration) nằm giữa cùng 1 người nói.
    Pattern: SpkA -> [SpkX(<1.5s)] -> [SpkY(<1.5s)] -> ... -> SpkA
    → Gộp tất cả vào SpkA
    """
    if len(segments) < 3:
        return segments

    # Sắp xếp theo thời gian
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    n = len(sorted_segs)

    # Đánh dấu các segment nào sẽ bị gộp (bỏ qua)
    to_skip = set()

    i = 0
    while i < n:
        if i in to_skip:
            i += 1
            continue

        # Tìm pattern: SpkA -> [ngắn] -> [ngắn] -> SpkA
        # Bắt đầu từ SpkA tại vị trí i
        spk_a = sorted_segs[i]["speaker"]

        # Tìm tất cả các đoạn ngắn liên tiếp phía sau
        j = i + 1
        short_segments_indices = []

        while j < n:
            duration = sorted_segs[j]["end"] - sorted_segs[j]["start"]
            if duration < max_short_duration:
                short_segments_indices.append(j)
                j += 1
            else:
                break

        # Kiểm tra xem sau các đoạn ngắn có phải SpkA không
        if len(short_segments_indices) > 0 and j < n:
            if sorted_segs[j]["speaker"] == spk_a:
                # Gộp các đoạn ngắn vào to_skip
                for idx in short_segments_indices:
                    to_skip.add(idx)
                # Chuyển sang xử lý tiếp từ j+1 (sau SpkA)
                i = j + 1
                continue

        i += 1

    # Tạo kết quả: mở rộng các đoạn SpkA để bao phủ các đoạn bị gộp
    result = []
    i = 0
    while i < n:
        if i in to_skip:
            i += 1
            continue

        current = dict(sorted_segs[i])

        # Nếu đoạn sau là ngắn và sẽ bị gộp vào đoạn này
        # thì mở rộng đoạn này đến hết chuỗi đoạn ngắn
        if i + 1 < n and (i + 1) in to_skip:
            # Tìm điểm cuối của chuỗi đoạn ngắn
            j = i + 1
            while j < n and j in to_skip:
                j += 1

            # j là đoạn SpkA tiếp theo (hoặc hết danh sách)
            if j < n and sorted_segs[j]["speaker"] == current["speaker"]:
                # Mở rộng đến cuối đoạn SpkA thứ hai
                current["end"] = sorted_segs[j]["end"]
                current["activation"] = max(
                    current.get("activation", 0),
                    max(sorted_segs[k].get("activation", 0) for k in range(i+1, j+1))
                )
                i = j + 1  # Nhảy qua cả chuỗi đã gộp
                result.append(current)
                continue

        result.append(current)
        i += 1

    return result


# ===== Output Classes =====

@dataclass
class Segment:
    """Represents a time segment with start and end times."""
    start: float
    end: float

    def __str__(self):
        return f"[{self.start:.3f}, {self.end:.3f}]"


class Annotation:
    """Simplified pyannote Annotation-like class."""

    def __init__(self):
        self._segments = []

    def add_segment(self, start: float, end: float, label: str, track: int = 0):
        self._segments.append((Segment(start, end), track, label))

    def itertracks(self, yield_label: bool = False):
        for segment, track, label in self._segments:
            if yield_label:
                yield segment, track, label
            else:
                yield segment, track

    def __len__(self):
        return len(self._segments)


class DiarizeOutput:
    """Output format with both overlapping and exclusive diarization."""

    def __init__(self):
        self.speaker_diarization = Annotation()
        self.exclusive_speaker_diarization = Annotation()
        self._segments = []

    def add_segments(self, segments: List[Dict], num_speakers: int,
                    global_activations: Optional[np.ndarray] = None,
                    frame_duration: float = 0.017,
                    exclusive_min_duration: float = 0.6):
        self._segments = sorted(segments, key=lambda x: (x['start'], x['speaker']))

        # Build speaker_diarization (with overlaps)
        for seg in self._segments:
            self.speaker_diarization.add_segment(
                seg['start'],
                seg['end'],
                f"SPEAKER_{seg['speaker']:02d}",
                track=seg['speaker']
            )

        # Build exclusive_speaker_diarization (no overlaps)
        self._compute_exclusive_diarization(num_speakers,
                                            global_activations=global_activations,
                                            frame_duration=frame_duration,
                                            exclusive_min_duration=exclusive_min_duration)

    def _compute_exclusive_diarization(self, num_speakers: int,
                                       frame_duration: float = 0.017,
                                       global_activations: Optional[np.ndarray] = None,
                                       exclusive_min_duration: float = 0.6):
        if not self._segments:
            return

        all_times = set()
        for seg in self._segments:
            all_times.add(seg['start'])
            all_times.add(seg['end'])
        time_points = sorted(all_times)

        exclusive_segments = []

        for i in range(len(time_points) - 1):
            start = time_points[i]
            end = time_points[i + 1]
            f_start, f_end = 0, 0

            if global_activations is not None:
                f_start = int(start / frame_duration)
                f_end = int(end / frame_duration)
                f_start = min(f_start, global_activations.shape[0] - 1)
                f_end = min(f_end, global_activations.shape[0])

            active_speakers = []
            for seg in self._segments:
                if seg['start'] <= start and seg['end'] >= end:
                    if global_activations is not None and f_end > f_start:
                        speaker_id = seg['speaker']
                        if speaker_id < global_activations.shape[1]:
                            activation = np.mean(global_activations[f_start:f_end, speaker_id])
                        else:
                            activation = seg.get('activation', 0.5)
                    else:
                        activation = seg.get('activation', 0.5)
                    active_speakers.append((seg, activation))

            if active_speakers:
                best_speaker_seg = max(active_speakers, key=lambda x: x[1])[0]
                best_speaker_id = best_speaker_seg['speaker']
                exclusive_segments.append({
                    "start": start,
                    "end": end,
                    "speaker": best_speaker_id,
                })

        exclusive_segments = self._merge_adjacent_segments(exclusive_segments)

        # Post-processing: Gộp các đảo ngắn kẹp giữa (< 1.5s)
        exclusive_segments = merge_short_islands(exclusive_segments, max_short_duration=1.5)

        for seg in exclusive_segments:
            self.exclusive_speaker_diarization.add_segment(
                seg["start"], seg["end"], f"SPEAKER_{seg['speaker']:02d}",
                track=seg["speaker"]
            )

    def _merge_adjacent_segments(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []

        merged = [dict(segments[0])]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"] and seg["start"] <= prev["end"] + 1e-6:
                prev["end"] = max(prev["end"], seg["end"])
            else:
                merged.append(dict(seg))

        return merged


# ===== ONNX Session Adapter =====

class _OnnxSessionAdapter:
    """Wraps ONNX session to remap I/O names for pyannote compatibility.

    altunenes ONNX embedding model uses:
        input: 'fbank_features', output: 'embeddings'
    pyannote ONNXWeSpeakerPretrainedSpeakerEmbedding expects:
        input: 'feats', output: 'embs'
    """

    def __init__(self, real_session, input_remap=None):
        self._real = real_session
        self._input_remap = input_remap or {}

    def run(self, output_names=None, input_feed=None):
        remapped = {}
        for k, v in input_feed.items():
            remapped[self._input_remap.get(k, k)] = v
        return self._real.run(None, remapped)

    def get_inputs(self):
        return self._real.get_inputs()

    def get_outputs(self):
        return self._real.get_outputs()


# ===== Main Diarizer =====

class AltunenesONNXDiarizer:
    """Pyannote Community-1 pipeline with ONNX inference.

    Uses the official pyannote.audio SpeakerDiarization pipeline for:
    - VBx clustering with PLDA scoring
    - Masked pooling for speaker embedding extraction
    - Hamming-weighted sliding window reconstruction
    - Powerset-to-multilabel conversion

    Replaces PyTorch model inference with ONNX Runtime:
    - segmentation-community-1.onnx (replaces PyanNet PyTorch)
    - embedding_model.onnx (replaces WeSpeaker PyTorch)
    """

    def __init__(self,
                 model_dir: Optional[str] = None,
                 num_speakers: int = -1,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 num_threads: int = 4,
                 min_duration_on: float = 0.0,
                 min_duration_off: float = 0.15,
                 threshold: float = 0.7,
                 # Kept for backward compat (not used in pyannote pipeline mode)
                 onset_threshold: float = 0.5,
                 offset_threshold: float = 0.35,
                 min_embedding_duration: float = 0.2,
                 min_prob_for_embedding: float = 0.4,
                 **kwargs):
        if not _onnx_available:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
        if not _pyannote_available:
            raise RuntimeError(
                "pyannote.audio not installed. pip install pyannote.audio\n"
                "Required for VBx clustering, PLDA scoring, and pipeline orchestration."
            )

        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_threads = num_threads
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.threshold = threshold
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold

        # ONNX model paths
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models", "pyannote-onnx")
        self.model_dir = model_dir
        self.segmentation_model_path = os.path.join(model_dir, "segmentation-community-1.onnx")
        self.embedding_model_path = os.path.join(model_dir, "embedding_model.onnx")

        # Pyannote pipeline model path (config.yaml + plda/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pyannote_model_dir = os.path.join(
            base_dir, "models", "pyannote", "speaker-diarization-community-1"
        )

        self.pipeline = None
        self.sample_rate = 16000

    def initialize(self):
        """Load pyannote pipeline, then replace PyTorch models with ONNX."""
        # Disable pyannote telemetry BEFORE importing pyannote.audio
        # (telemetry module is loaded at import time and tries OTLP network calls)
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
        import torch
        import onnxruntime as ort
        from pyannote.audio import Pipeline

        print(f"[AltunenesONNX] Loading pyannote pipeline + ONNX inference...")
        start_time = time.time()

        # Validate all required files
        if not os.path.exists(os.path.join(self.pyannote_model_dir, "config.yaml")):
            raise FileNotFoundError(
                f"Pyannote model not found: {self.pyannote_model_dir}\n"
                "Need: config.yaml, segmentation/pytorch_model.bin, "
                "embedding/pytorch_model.bin, plda/plda.npz"
            )
        if not os.path.exists(self.segmentation_model_path):
            raise FileNotFoundError(f"Segmentation ONNX not found: {self.segmentation_model_path}")
        if not os.path.exists(self.embedding_model_path):
            raise FileNotFoundError(f"Embedding ONNX not found: {self.embedding_model_path}")

        # === Step 1: Load pyannote pipeline (PyTorch models + VBx + PLDA) ===
        # Pass local directory directly — pyannote 4.0 detects os.path.isdir()
        # and loads purely from local files (no HuggingFace Hub calls needed)
        self.pipeline = Pipeline.from_pretrained(self.pyannote_model_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            torch.set_num_threads(self.num_threads)
            print(f"[AltunenesONNX] Using CPU with {self.num_threads} threads")
        else:
            print(f"[AltunenesONNX] Using CUDA")
        self.pipeline.to(device)

        # === Step 2: Replace segmentation model inference → ONNX ===
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        seg_session = ort.InferenceSession(
            self.segmentation_model_path, sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                if torch.cuda.is_available() else ['CPUExecutionProvider']
        )

        seg_model = self.pipeline._segmentation.model

        # Monkey-patch forward() to use ONNX instead of PyTorch
        def onnx_seg_forward(waveforms, **kwargs):
            np_input = waveforms.detach().cpu().numpy()
            output = seg_session.run(None, {"input_values": np_input})
            return torch.from_numpy(output[0]).to(waveforms.device)

        seg_model.forward = onnx_seg_forward
        # Keep ONNX session alive (prevent garbage collection)
        seg_model._onnx_session = seg_session

        print(f"[AltunenesONNX] Segmentation -> ONNX")

        # === Step 3: Replace embedding model → ONNX WeSpeaker ===
        from pyannote.audio.pipelines.speaker_verification import (
            ONNXWeSpeakerPretrainedSpeakerEmbedding
        )

        onnx_embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding(
            self.embedding_model_path, device=device
        )

        # Fix: pyannote hardcodes intra_op=1, inter_op=1 for embedding ONNX session.
        # Re-create session with proper thread count for much faster embedding extraction.
        emb_sess_options = ort.SessionOptions()
        emb_sess_options.intra_op_num_threads = self.num_threads
        emb_sess_options.inter_op_num_threads = self.num_threads
        emb_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        fast_emb_session = ort.InferenceSession(
            self.embedding_model_path, emb_sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                if torch.cuda.is_available() else ['CPUExecutionProvider']
        )

        # Adapt I/O names: pyannote expects feats/embs, our ONNX has fbank_features/embeddings
        # AND use the fast multi-threaded session instead of the original 1-thread session
        onnx_embedding.session_ = _OnnxSessionAdapter(
            fast_emb_session,
            input_remap={"feats": "fbank_features"}
        )

        # Replace in pipeline
        self.pipeline._embedding = onnx_embedding

        # Update audio handler to match embedding's sample rate
        from pyannote.audio.core.io import Audio
        self.pipeline._audio = Audio(sample_rate=onnx_embedding.sample_rate, mono="downmix")

        print(f"[AltunenesONNX] Embedding -> ONNX (dim={onnx_embedding.dimension})")

        # === Step 4: Configure pipeline parameters ===
        # Pipeline.from_pretrained() already instantiates with config.yaml defaults.
        # Re-instantiate with our custom values.
        params = self.pipeline.parameters()
        params["clustering"]["threshold"] = self.threshold
        params["segmentation"]["min_duration_off"] = self.min_duration_off
        try:
            self.pipeline.instantiate(params)
            print(f"[AltunenesONNX] Params: threshold={self.threshold}, "
                  f"min_duration_off={self.min_duration_off}")
        except Exception as e:
            print(f"[AltunenesONNX] Warning: Could not apply params: {e}")

        elapsed = time.time() - start_time
        print(f"[AltunenesONNX] Initialized in {elapsed:.2f}s")
        print(f"[AltunenesONNX] Pipeline: Segmentation(ONNX) -> Embedding(ONNX+masked) -> VBx+PLDA -> Reconstruction")

    def _load_audio(self, audio_file: str):
        """Load audio using torchaudio (best quality resampling)"""
        import torch
        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform, sample_rate
        except Exception as e:
            print(f"[AltunenesONNX] torchaudio failed: {e}, trying soundfile...")
            import soundfile as sf
            import torch
            data, sr = sf.read(audio_file, dtype='float32')
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
            return waveform, sr

    def process(self,
                audio_file: str,
                progress_callback: Optional[Callable[[int, int], int]] = None,
                audio_data: Optional[np.ndarray] = None,
                audio_sample_rate: Optional[int] = None):
        """Run speaker diarization using pyannote pipeline with ONNX inference.

        Returns pyannote DiarizeOutput with:
        - speaker_diarization: Annotation (with overlapping speech)
        - exclusive_speaker_diarization: Annotation (no overlap, best for ASR)
        """
        if self.pipeline is None:
            self.initialize()

        import torch

        print(f"[AltunenesONNX] Processing: {audio_file}")
        start_time = time.time()

        # Prepare audio as tensor dict (bypasses torchcodec issues in pyannote 4.0)
        if audio_data is not None:
            waveform = audio_data.astype(np.float32)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            sample_rate = audio_sample_rate or self.sample_rate
        else:
            waveform_tensor, sample_rate = self._load_audio(audio_file)

        audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
        duration = waveform_tensor.shape[-1] / sample_rate

        # Build pipeline kwargs for speaker count
        pipeline_kwargs = {}
        if self.num_speakers > 0:
            pipeline_kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers is not None:
                pipeline_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                pipeline_kwargs["max_speakers"] = self.max_speakers

        if progress_callback:
            progress_callback(5, 100)

        # Run pyannote pipeline: Segmentation → Embedding → VBx Clustering → Reconstruction
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                result = self.pipeline(audio_input, hook=hook, **pipeline_kwargs)
        except ImportError:
            result = self.pipeline(audio_input, **pipeline_kwargs)

        elapsed = time.time() - start_time
        print(f"[AltunenesONNX] Done in {elapsed:.2f}s "
              f"({duration:.1f}s audio, RTF={elapsed/duration:.2f})")

        if progress_callback:
            progress_callback(100, 100)

        return result

    def process_with_transcription(self, audio_file: str, transcribed_segments: List[Dict],
                                   speaker_segments: Optional[List[Dict]] = None) -> List[Dict]:
        """Merge speaker diarization with transcription segments."""
        if speaker_segments is None:
            result = self.process(audio_file)
            speaker_segments = []

            # Extract segments from pyannote DiarizeOutput
            annotation = None
            if hasattr(result, 'exclusive_speaker_diarization'):
                annotation = result.exclusive_speaker_diarization
            elif hasattr(result, 'speaker_diarization'):
                annotation = result.speaker_diarization
            elif hasattr(result, 'itertracks'):
                annotation = result

            if annotation is not None:
                for item in annotation.itertracks(yield_label=True):
                    if len(item) != 3:
                        continue
                    raw_turn, _, speaker_label = cast(Tuple[object, object, object], item)
                    if isinstance(speaker_label, str) and speaker_label.startswith("SPEAKER_"):
                        spk_id = int(speaker_label.split("_")[1])
                    elif isinstance(speaker_label, int):
                        spk_id = speaker_label
                    else:
                        continue
                    speaker_segments.append({
                        "start": raw_turn.start,
                        "end": raw_turn.end,
                        "speaker": spk_id
                    })

        if not speaker_segments:
            return transcribed_segments

        results = []
        for trans_seg in transcribed_segments:
            trans_start = trans_seg.get("start", 0)
            trans_end = trans_seg.get("end", trans_start + 1)
            speaker_votes: Dict[int, float] = {}
            for spk_seg in speaker_segments:
                overlap_dur = max(0, min(trans_end, spk_seg["end"]) - max(trans_start, spk_seg["start"]))
                if overlap_dur > 0:
                    spk_id = spk_seg["speaker"]
                    speaker_votes[spk_id] = speaker_votes.get(spk_id, 0) + overlap_dur

            if speaker_votes:
                best_speaker_id = max(speaker_votes.items(), key=lambda item: item[1])[0]
                best_speaker = f"Người nói {best_speaker_id + 1}"
            else:
                best_speaker_id = results[-1].get("speaker_id", 0) if results else 0
                best_speaker = f"Người nói {best_speaker_id + 1}"

            seg_copy = dict(trans_seg)
            seg_copy.update({"speaker": best_speaker, "speaker_id": best_speaker_id})
            results.append(seg_copy)
        return results

    def format_by_speaker(self, segments: List[Dict]) -> str:
        if not segments:
            return "Không có nội dung"
        grouped, current_speaker, current_texts = [], None, []
        for seg in segments:
            speaker = seg.get("speaker", "Người nói 1")
            text = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                if current_speaker and current_texts:
                    grouped.append({"speaker": current_speaker, "text": " ".join(current_texts)})
                current_speaker, current_texts = speaker, [text]
            else:
                current_texts.append(text)
        if current_speaker and current_texts:
            grouped.append({"speaker": current_speaker, "text": " ".join(current_texts)})

        return "\n".join(f"{g['speaker']}:\n{g['text']}\n" for g in grouped).strip()

    def unload(self):
        import gc
        if self.pipeline is not None:
            print("[AltunenesONNX] Unloading pipeline...")
            self.pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("[AltunenesONNX] Unloaded")


def check_available() -> bool:
    """Check if all dependencies and model files are available."""
    if not _onnx_available or not _pyannote_available:
        return False
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_dir = os.path.join(base_dir, "models", "pyannote-onnx")
    pyannote_dir = os.path.join(base_dir, "models", "pyannote", "speaker-diarization-community-1")
    return (
        os.path.exists(os.path.join(onnx_dir, "segmentation-community-1.onnx")) and
        os.path.exists(os.path.join(onnx_dir, "embedding_model.onnx")) and
        os.path.exists(os.path.join(pyannote_dir, "config.yaml"))
    )

def get_model_info() -> Dict:
    return {
        "id": "community1_onnx",
        "name": "Community-1 ONNX (Pyannote Pipeline)",
        "description": "Official pyannote pipeline (VBx + PLDA + masked pooling) with ONNX model inference",
        "size": "~32MB ONNX + ~40MB pipeline",
        "language": "Multilingual",
        "speed": "Fast (ONNX inference, pyannote pipeline)",
        "accuracy": "High (identical to pyannote PyTorch)",
        "features": [
            "VBx Clustering with PLDA scoring",
            "Masked pooling for speaker embeddings",
            "Hamming-weighted sliding window reconstruction",
            "Powerset decoding (7 classes, max 3 speakers/chunk)",
            "Overlapping speech detection",
            "Exclusive speaker diarization (for ASR)",
            "ONNX Runtime inference (CPU/CUDA)",
            "WeSpeaker embedding (256-dim)",
            "min_speakers/max_speakers support",
        ],
        "requirements": ["pyannote.audio", "onnxruntime", "torch", "torchaudio"]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Community-1 ONNX diarization (pyannote pipeline)")
    parser.add_argument("audio", help="Audio path")
    parser.add_argument("--speakers", type=int, default=-1, help="Number of speakers (-1=auto)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers")
    parser.add_argument("--threshold", type=float, default=0.7, help="VBx clustering threshold")
    args = parser.parse_args()

    dr = AltunenesONNXDiarizer(
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        threshold=args.threshold
    )

    output = dr.process(args.audio)

    # Handle pyannote DiarizeOutput
    annotation = None
    if hasattr(output, 'exclusive_speaker_diarization'):
        annotation = output.exclusive_speaker_diarization
        print("\n=== EXCLUSIVE SPEAKER DIARIZATION ===")
    elif hasattr(output, 'speaker_diarization'):
        annotation = output.speaker_diarization
        print("\n=== SPEAKER DIARIZATION ===")

    if annotation is not None:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            print(f"{turn.start:.2f}s - {turn.end:.2f}s: {speaker}")
