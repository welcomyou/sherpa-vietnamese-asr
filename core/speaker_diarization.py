"""
[DEPRECATED - KHÔNG DÙNG TRONG PRODUCTION]
Lý do: NeMo TitaNet Small chậm hơn và kém chính xác hơn so với
  speaker_diarization_pure_ort.py (ResNet34-LM + VBx + PLDA).
  Segmentation 3.0 cũng kém hơn community-1 segmentation.
Dùng thay thế: core/speaker_diarization_pure_ort.py

Người nói Diarization module using sherpa-onnx
Based on: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html

Available speaker embedding models:
- nemo_en_titanet_small.onnx (38.4MB) - Fast, good accuracy, English


Segmentation model:
- sherpa-onnx-pyannote-segmentation-3-0

Sample reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import time
import re

# sherpa_onnx is imported lazily to ensure DLL paths are set up first
_sherpa_onnx = None
SHERPA_AVAILABLE = False  # Default to False, will be set to True on successful import

def get_sherpa_onnx():
    """Lazy import sherpa_onnx to ensure DLL paths are set up first."""
    global _sherpa_onnx, SHERPA_AVAILABLE
    if _sherpa_onnx is None:
        try:
            import sherpa_onnx as so
            _sherpa_onnx = so
            SHERPA_AVAILABLE = True
        except ImportError:
            SHERPA_AVAILABLE = False
            print("Warning: sherpa_onnx not available")
            raise
    return _sherpa_onnx

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _setup_ffmpeg_path():
    """Thiết lập đường dẫn ffmpeg cho pydub"""
    if not PYDUB_AVAILABLE:
        return
        
    from pydub.utils import which
    
    # Kiểm tra ffmpeg đã có trong PATH chưa
    if which("ffmpeg"):
        return
    
    # Các vị trí phổ biến trên Windows
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
        os.path.join(base_dir, "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    
    for ffmpeg_path in possible_paths:
        if os.path.exists(ffmpeg_path):
            # Thiết lập cho pydub
            import pydub
            pydub.AudioSegment.converter = ffmpeg_path
            pydub.AudioSegment.ffmpeg = ffmpeg_path
            pydub.AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
            print(f"[SpeakerDiarizer] Found ffmpeg at: {ffmpeg_path}")
            return


# Speaker embedding model registry
SPEAKER_EMBEDDING_MODELS = {
    "community1_pure_ort": {
        "name": "Pyannote Community-1 (ResNet34-LM + PLDA + VBx - Chậm, chính xác)",
        "file": "community1_pure_ort",
        "size": "~32 MB",
        "language": "Multilingual",
        "speed": "Chính xác nhất",
        "accuracy": "Excellent",
        "sample_rate": 16000,
        "description": "Pyannote Community-1 via ONNX Runtime — ResNet34-LM + PLDA + VBx, no PyTorch"
    },
    "senko_campp": {
        "name": "Senko CAM++ (Tiết kiệm RAM)",
        "file": "senko_campp",
        "size": "~27 MB",
        "language": "Multilingual (ZH+EN)",
        "speed": "Nhanh gấp 3x",
        "accuracy": "Tốt",
        "sample_rate": 16000,
        "has_threshold": False,
        "description": "Senko pipeline — CAM++ 192-dim + pyannote VAD + spectral + mer_cos merge (github.com/narcotic-sh/senko)"
    },
    "senko_campp_optimized": {
        "name": "Senko CAM++ (Optimized - Nhanh)",
        "file": "senko_campp_optimized",
        "size": "~27 MB",
        "language": "Multilingual (ZH+EN)",
        "speed": "Nhanh gấp 6-7x",
        "accuracy": "Tốt",
        "sample_rate": 16000,
        "has_threshold": False,
        "description": "Senko optimized — batch inference + fbank once + VAD step 5s, 2.5x nhanh hơn bản thường"
    },
}


def get_available_models(base_dir: str = None) -> Dict[str, str]:
    """
    Get list of available (downloaded) speaker embedding models
    
    Returns:
        Dict mapping model_id to full path
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_dir = os.path.join(base_dir, "models", "speaker_embedding")
    available = {}
    
    for model_id, info in SPEAKER_EMBEDDING_MODELS.items():
        if model_id == "community1_pure_ort":
            onnx_model_dir = os.path.join(base_dir, "models", "pyannote-onnx")
            seg_path = os.path.join(onnx_model_dir, "segmentation-community-1.onnx")
            emb_path = os.path.join(onnx_model_dir, "embedding_model.onnx")
            enc_path = os.path.join(onnx_model_dir, "embedding_encoder.onnx")
            if os.path.exists(seg_path) and (os.path.exists(emb_path) or os.path.exists(enc_path)):
                available[model_id] = "pure_ort"
        elif model_id in ("senko_campp", "senko_campp_optimized"):
            campp_path = os.path.join(base_dir, "models", "campp-3dspeaker", "campplus_cn_en_common_200k.onnx")
            seg_path = os.path.join(base_dir, "models", "pyannote-onnx", "segmentation-community-1.onnx")
            if os.path.exists(campp_path) and os.path.exists(seg_path):
                available[model_id] = model_id
        else:
            model_path = os.path.join(models_dir, info["file"])
            if os.path.exists(model_path):
                available[model_id] = model_path
    
    return available


def check_pyannote_available() -> bool:
    """Check if Pyannote 3.1 is available"""
    try:
        import pyannote.audio
        return True
    except ImportError:
        return False


def get_model_info(model_id: str) -> Optional[Dict]:
    """Get information about a speaker embedding model"""
    return SPEAKER_EMBEDDING_MODELS.get(model_id)


def get_model_path(model_id: str, base_dir: str = None) -> Optional[str]:
    """Get full path to a model file"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    info = SPEAKER_EMBEDDING_MODELS.get(model_id)
    if info is None:
        return None
    
    model_path = os.path.join(base_dir, "models", "speaker_embedding", info["file"])
    if os.path.exists(model_path):
        return model_path
    return None


class Segment:
    """Người nói segment with start, end, speaker id"""
    
    _call_count = 0  # Track process() calls
    
    def __init__(self, start, end, speaker):
        assert start < end
        self.start = start
        self.end = end
        self.speaker = speaker

    def merge(self, other, gap=0.5):
        assert self.speaker == other.speaker
        if self.end < other.start and self.end + gap >= other.start:
            return Segment(start=self.start, end=other.end, speaker=self.speaker)
        elif other.end < self.start and other.end + gap >= self.start:
            return Segment(start=other.start, end=self.end, speaker=self.speaker)
        else:
            return None

    @property
    def duration(self):
        return self.end - self.start

    def __str__(self):
        return f"{self.start:.3f}s --> {self.end:.3f}s speaker_{self.speaker:02d}"


def resample_audio(audio, sample_rate, target_sample_rate):
    """
    Resample audio to target sample rate using librosa
    """
    if sample_rate != target_sample_rate:
        print(f"[SpeakerDiarizer] Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        audio = audio.astype(np.float32)  # Ensure float32 for model compatibility
        print(f"[SpeakerDiarizer] Resampling completed. New audio shape: {audio.shape}")
        return audio, target_sample_rate
    return audio, sample_rate


def load_audio(filename, target_sample_rate) -> Tuple[np.ndarray, int]:
    """Load audio file and resample if needed using torchaudio (best for Speaker Diarization)
    
    Supports: wav, mp3, m4a, flac, ogg, wma, aac, opus
    
    Args:
        filename: Path to audio file
        target_sample_rate: Expected sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # For WAV files, use soundfile directly to avoid torchaudio/ffmpeg init hang
    if filename.lower().endswith('.wav') and SOUNDFILE_AVAILABLE:
        try:
            audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # mono
            if sample_rate != target_sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
            print(f"[SpeakerDiarizer] Loaded with soundfile: {len(audio)} samples at {sample_rate}Hz")
            return audio, sample_rate
        except Exception as e:
            print(f"[SpeakerDiarizer] soundfile failed: {e}, falling back to torchaudio...")

    import torch
    import torchaudio

    try:
        # Load audio with torchaudio (supports many formats via ffmpeg backend)
        waveform, sample_rate = torchaudio.load(filename, normalize=True)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed using torchaudio (high quality Kaiser-Sinc)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate,
                lowpass_filter_width=6  # High quality
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Convert to numpy (squeeze to 1D)
        audio = waveform.squeeze().numpy().astype(np.float32)
        
        print(f"[SpeakerDiarizer] Loaded with torchaudio: {len(audio)} samples at {sample_rate}Hz")
        return audio, sample_rate
        
    except Exception as e:
        print(f"[SpeakerDiarizer] torchaudio failed: {e}, falling back to soundfile...")
        # Fallback to soundfile
        try:
            audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # mono
            if sample_rate != target_sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
            return audio, sample_rate
        except Exception:
            raise RuntimeError(f"Failed to load audio: {filename}")




class SpeakerDiarizer:
    """Người nói diarization using sherpa-onnx OfflineSpeakerDiarization API"""
    
    
    _call_count = 0  # Track process() calls
    
    def __init__(self, 
                 segmentation_model: str = None,
                 embedding_model: str = None,
                 embedding_model_id: str = None,
                 num_clusters: int = -1,
                 min_speakers: int = None,
                 max_speakers: int = None,
                 num_threads: int = 6,
                 threshold: float = 0.6,
                 min_duration_on: float = 0.3, # updated to realistic defaults
                 min_duration_off: float = 0.0,
                 auth_token: str = None):
        """
        Initialize speaker diarizer
        
        Args:
            segmentation_model: Path to segmentation model (.onnx)
            embedding_model: Path to speaker embedding model (.onnx) - can use embedding_model_id instead
            embedding_model_id: Model ID from SPEAKER_EMBEDDING_MODELS registry
            num_clusters: Number of speakers (-1 for auto-detect using threshold)
            min_speakers: Minimum number of speakers (for Community-1, overrides num_clusters if set)
            max_speakers: Maximum number of speakers (for Community-1, overrides num_clusters if set)
            num_threads: Number of threads for inference
            threshold: Threshold for clustering when num_clusters=-1 (default: 0.8)
                      Smaller threshold = more speakers, Larger threshold = fewer speakers
            min_duration_on: Minimum duration for a speaker to be active (default: 0.3s)
            min_duration_off: Minimum silence between segments (default: 0.5s)
        """
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.embedding_model_id = embedding_model_id
        self.num_clusters = num_clusters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_threads = num_threads
        self.threshold = threshold if threshold is not None else 0.8
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.sd = None  # sherpa_onnx OfflineSpeakerDiarization instance
        self.model_info = None  # Store model info for reference
        self._pyannote_backend = None  # Community1Diarizer instance
        self.auth_token = auth_token or os.environ.get('HF_TOKEN', None)
        
    # Default thresholds for each model
    MODEL_DEFAULT_THRESHOLDS = {
        "community1_pure_ort": 0.7,
        "senko_campp": 0.5,
        "senko_campp_optimized": 0.5,
    }

    @classmethod
    def get_default_threshold(cls, model_id: str) -> float:
        """Get default threshold for a specific model"""
        return cls.MODEL_DEFAULT_THRESHOLDS.get(model_id, 0.7)

    def initialize(self):
        """Initialize speaker diarization backend based on embedding_model_id"""
        if self.embedding_model_id == "campp_pure_ort":
            self._init_campp()
        else:
            self._init_pure_ort()

    def _init_campp(self):
        """Initialize CAM++ Pure ORT backend (2.5× faster than ResNet34)"""
        try:
            from core.speaker_diarization_pure_ort_campp import PureOrtDiarizerCampp
        except ImportError as e:
            raise RuntimeError(
                f"CAM++ Pure ORT diarizer not available: {e}\n"
                "Install with: pip install onnxruntime scipy kaldi-native-fbank"
            )

        num_spk = self.num_clusters if self.num_clusters > 0 else -1
        max_spk = None
        if self.num_clusters > 0:
            max_spk = self.num_clusters + 1
            print(f"[SpeakerDiarizer] CAM++: User selected {self.num_clusters} speakers -> max={max_spk}")
        else:
            print(f"[SpeakerDiarizer] CAM++: Auto-detect speakers")

        self._pyannote_backend = PureOrtDiarizerCampp(
            num_threads=self.num_threads,
            threshold=self.threshold,
            min_duration_off=self.min_duration_off,
            num_speakers=num_spk,
            max_speakers=max_spk,
        )
        self._pyannote_backend.initialize()
        self.model_info = get_model_info("senko_campp")

    def _init_pure_ort(self):
        """Initialize Pure ORT backend (no PyTorch/pyannote dependency)"""
        try:
            from core.speaker_diarization_pure_ort import PureOrtDiarizer
        except ImportError as e:
            raise RuntimeError(
                f"Pure ORT diarizer not available: {e}\n"
                "Install with: pip install onnxruntime scipy kaldi-native-fbank"
            )

        # Nếu user chọn số speaker cụ thể
        num_spk = self.num_clusters if self.num_clusters > 0 else -1
        max_spk = None
        if self.num_clusters > 0:
            max_spk = self.num_clusters + 1
            print(f"[SpeakerDiarizer] PureORT: User selected {self.num_clusters} speakers -> max={max_spk}")
        else:
            print(f"[SpeakerDiarizer] PureORT: Auto-detect speakers")

        self._pyannote_backend = PureOrtDiarizer(
            num_threads=self.num_threads,
            threshold=self.threshold,
            min_duration_off=self.min_duration_off,
            num_speakers=num_spk,
            max_speakers=max_spk,
        )
        self._pyannote_backend.initialize()
        self.model_info = get_model_info("community1_pure_ort")

    def process(self,
                audio_file: str,
                progress_callback: Optional[Callable[[int, int], int]] = None,
                audio_data: Optional[np.ndarray] = None,
                audio_sample_rate: Optional[int] = None,
                asr_words: List[Dict] = None) -> List[Segment]:
        """Perform speaker diarization on audio file

        Args:
            asr_words: word-level timestamps từ ASR (optional). Nếu có,
                       NaturalTurn dùng word count để classify backchannel
                       chính xác hơn (duration < 0.8s VÀ word_count <= 3).
        """
        SpeakerDiarizer._call_count += 1

        if self._pyannote_backend is None:
            self.initialize()

        return self._process_pyannote(audio_file, progress_callback, audio_data, audio_sample_rate, asr_words)

    def _clip_segments_to_speech(self, segments: List[Segment],
                                 audio: np.ndarray, sample_rate: int) -> List[Segment]:
        """Clip diarization segments tại ranh giới speech/silence từ VAD.

        Diarization model có thể kéo dài segment qua khoảng lặng.
        Dùng VAD để xác định vùng có tiếng nói, rồi clip segment —
        chỉ giữ phần giao với speech regions.
        """
        if not segments:
            return segments

        try:
            from core.asr_engine import get_vad_segments
            vad_segs = get_vad_segments(audio, sample_rate=sample_rate,
                                        min_silence_ms=300, padding_ms=200)
            speech_regions = [(s / sample_rate, e / sample_rate) for s, e in vad_segs]
        except Exception as e:
            print(f"[SpeakerDiarizer] VAD clip failed: {e}, using original segments")
            return segments

        if not speech_regions:
            return segments

        total_speech = sum(e - s for s, e in speech_regions)
        print(f"[SpeakerDiarizer] VAD: {len(speech_regions)} speech regions "
              f"({total_speech:.1f}s speech / {len(audio)/sample_rate:.1f}s audio)")

        # Clip: giữ phần giao giữa mỗi diarization segment và speech regions
        clipped = []
        for seg in segments:
            for sp_start, sp_end in speech_regions:
                clip_start = max(seg.start, sp_start)
                clip_end = min(seg.end, sp_end)
                if clip_end - clip_start > 0.05:
                    clipped.append(Segment(
                        start=clip_start, end=clip_end, speaker=seg.speaker
                    ))

        clipped.sort(key=lambda s: s.start)

        if len(clipped) != len(segments):
            print(f"[SpeakerDiarizer] Clipped segments: {len(segments)} -> {len(clipped)}")

        return clipped if clipped else segments

    def _post_process_diarization_segments(self, segments: List[Segment],
                                           asr_words: List[Dict] = None) -> List[Segment]:
        """
        Post-processing cho exclusive_speaker_diarization trước khi dùng cho ASR.

        Thứ tự:
        1. Merge adjacent + small gaps (0.3s) cùng speaker
        2. Resolve fragment zones (nhiều segment ngắn xen kẽ → dominant speaker)
        3. NaturalTurn: xác định floor-holder, gộp secondary speech vào primary
        4. Final merge

        Args:
            asr_words: word-level timestamps từ ASR (optional, nếu có sẽ dùng
                       word count để classify backchannel chính xác hơn)
        """
        if not segments:
            return segments

        original_count = len(segments)

        # 1. Merge đoạn liền kề + gap nhỏ (0.3s) cùng speaker
        segments = self._merge_segments_with_gap(segments, max_gap=0.3)

        # 2. Resolve fragment zones: vùng nhiều segment ngắn xen kẽ → dominant speaker
        segments = self._resolve_fragment_zones(segments, short_thresh=0.5, min_zone_size=3)

        # 3. NaturalTurn: floor-holding detection + secondary speech absorption
        segments = self._natural_turn_merge(segments, max_pause=2.0, asr_words=asr_words)

        # 4. Final merge
        segments = self._merge_segments_with_gap(segments, max_gap=0.3)

        final_count = len(segments)
        if final_count < original_count:
            print(f"[SpeakerDiarizer] Post-process: {original_count} -> {final_count} segments")

        return segments

    def _natural_turn_merge(self, segments: List[Segment], max_pause: float = 1.5,
                            asr_words: List[Dict] = None) -> List[Segment]:
        """
        NaturalTurn algorithm (Cychosz et al., Scientific Reports 2025).

        Xác định ai đang "giữ sàn" (floor-holder) dựa trên timing,
        rồi gộp secondary speech (backchannel, nói chêm) vào primary turn.

        Thuật toán 4 bước (theo paper):
          1. Per speaker: collapse segments có gap < max_pause thành "turns"
             (speaker tiếp tục giữ floor qua pause ngắn)
          2. Sort tất cả turns theo thời gian, nếu turn T2 nằm HOÀN TOÀN
             bên trong boundary của turn T1 → T2 là secondary
          3. Classify secondary: BACKCHANNEL nếu (duration < 0.8s VÀ word_count <= 3),
             ngược lại SECONDARY_TURN (giữ nguyên, phản hồi có nội dung)
             Nếu có asr_words → đếm word count chính xác
             Nếu không → dùng duration-only (pre-ASR fallback)
          4. Gán lại segments: chỉ BACKCHANNEL → speaker của primary turn

        Ví dụ (max_pause=2.0s):
          A(0-10s) → B(10.5-11s) → A(11.5-20s)
          Step 1: A turn = [0-20s] (gap 1.5s < max_pause), B turn = [10.5-11s]
          Step 2: B turn [10.5-11s] nằm trong A turn [0-20s] → secondary
          Step 3: B segment gán lại thành A → output A(0-20s)

        Paper: https://www.nature.com/articles/s41598-025-24381-1
        """
        if len(segments) < 3:
            return segments

        sorted_segs = sorted(segments, key=lambda s: s.start)

        # === Step 1: Build virtual turns per speaker ===
        # Collapse segments cùng speaker có gap < max_pause thành 1 turn
        speakers = set(s.speaker for s in sorted_segs)
        turns = []  # list of (turn_start, turn_end, speaker, [segment_indices])

        for spk in speakers:
            spk_indices = [i for i, s in enumerate(sorted_segs) if s.speaker == spk]
            if not spk_indices:
                continue

            # Start first turn
            turn_start = sorted_segs[spk_indices[0]].start
            turn_end = sorted_segs[spk_indices[0]].end
            turn_seg_indices = [spk_indices[0]]

            for k in range(1, len(spk_indices)):
                idx = spk_indices[k]
                gap = sorted_segs[idx].start - turn_end
                if gap < max_pause:
                    # Same turn: extend boundary
                    turn_end = max(turn_end, sorted_segs[idx].end)
                    turn_seg_indices.append(idx)
                else:
                    # New turn: save current, start new
                    turns.append((turn_start, turn_end, spk, turn_seg_indices))
                    turn_start = sorted_segs[idx].start
                    turn_end = sorted_segs[idx].end
                    turn_seg_indices = [idx]

            turns.append((turn_start, turn_end, spk, turn_seg_indices))

        # Sort turns by start time
        turns.sort(key=lambda t: t[0])

        # === Step 2: Label primary vs secondary ===
        # Turn T2 nằm hoàn toàn trong T1 → T2 là secondary
        n_turns = len(turns)
        is_secondary = [False] * n_turns
        primary_of = [None] * n_turns  # secondary turn i thuộc primary turn nào

        for i in range(n_turns):
            if is_secondary[i]:
                continue
            t1_start, t1_end, t1_spk, _ = turns[i]
            for j in range(i + 1, n_turns):
                if is_secondary[j]:
                    continue
                t2_start, t2_end, t2_spk, _ = turns[j]
                # T2 bắt đầu sau T1 kết thúc → không còn overlap
                if t2_start >= t1_end:
                    break
                # T2 nằm hoàn toàn trong T1 và khác speaker
                if t2_end <= t1_end and t2_spk != t1_spk:
                    is_secondary[j] = True
                    primary_of[j] = i

        # === Step 3: Classify + Reassign secondary turns ===
        # Paper: BACKCHANNEL nếu word_count <= 3 + match backchannel cues
        # Implementation: duration < 0.8s VÀ word_count <= 3 (nếu có ASR text)
        max_backchannel_dur = 2.0  # tiếng Việt backchannel có thể tới ~2s
        backchannel_word_max = 3

        def _count_words_in_range(start, end):
            """Đếm số ASR words có timestamp rơi vào [start, end]."""
            if not asr_words:
                return None  # không có text → trả None
            count = 0
            for w in asr_words:
                w_mid = (w.get("start", 0) + w.get("end", 0)) / 2
                if start <= w_mid <= end:
                    count += 1
            return count

        reassign = {}
        for j in range(n_turns):
            if is_secondary[j] and primary_of[j] is not None:
                t2_start, t2_end, _, _ = turns[j]
                turn_dur = t2_end - t2_start

                # Check duration
                if turn_dur >= max_backchannel_dur:
                    continue  # quá dài → SECONDARY_TURN

                # Check word count (nếu có ASR text)
                wc = _count_words_in_range(t2_start, t2_end)
                if wc is not None and wc > backchannel_word_max:
                    continue  # > 3 từ → nói nhanh nhưng có nội dung

                # BACKCHANNEL: duration < 0.8s VÀ (word_count <= 3 hoặc không có text)
                primary_spk = turns[primary_of[j]][2]
                for seg_idx in turns[j][3]:
                    reassign[seg_idx] = primary_spk

        # Build result
        result = []
        for i, seg in enumerate(sorted_segs):
            new_spk = reassign.get(i, seg.speaker)
            result.append(Segment(start=seg.start, end=seg.end, speaker=new_spk))

        # Merge adjacent same-speaker segments tạo bởi reassignment
        # Chỉ merge khi gần kề (gap < 0.5s) — tránh gộp quá mạnh
        result = self._merge_segments_with_gap(result, max_gap=0.5)

        return result

    def _merge_segments_with_gap(self, segments: List[Segment], max_gap: float = 0.3) -> List[Segment]:
        """Gộp các đoạn liền kề hoặc có gap nhỏ cùng speaker."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: (s.start, s.speaker))
        merged = [Segment(start=sorted_segs[0].start, end=sorted_segs[0].end, speaker=sorted_segs[0].speaker)]

        for seg in sorted_segs[1:]:
            prev = merged[-1]
            gap = seg.start - prev.end
            if seg.speaker == prev.speaker and gap <= max_gap:
                prev.end = max(prev.end, seg.end)
            else:
                merged.append(Segment(start=seg.start, end=seg.end, speaker=seg.speaker))

        return merged

    def _resolve_fragment_zones(self, segments: List[Segment],
                                short_thresh: float = 0.5,
                                min_zone_size: int = 3) -> List[Segment]:
        """
        Phát hiện vùng fragment (nhiều segment ngắn xen kẽ) và gán toàn bộ
        cho speaker dominant (tổng duration lớn nhất) trong vùng đó.

        Tránh cascade bug của reassign từng segment:
          spk_01(0.05) → spk_02(0.03) → spk_01(0.07) → ...
          → gán hết cho speaker có tổng duration lớn nhất trong vùng

        Fragment zone = chuỗi liên tiếp >= min_zone_size segments có duration < short_thresh.
        """
        if len(segments) < min_zone_size:
            return segments

        result = []
        n = len(segments)
        i = 0

        while i < n:
            # Tìm đầu fragment zone: segment ngắn
            if segments[i].duration < short_thresh:
                # Scan vùng liên tiếp các segment ngắn
                j = i
                while j < n and segments[j].duration < short_thresh:
                    j += 1

                zone_size = j - i

                if zone_size >= min_zone_size:
                    # Fragment zone: tính dominant speaker theo tổng duration
                    spk_dur = {}
                    for k in range(i, j):
                        s = segments[k]
                        spk_dur[s.speaker] = spk_dur.get(s.speaker, 0) + s.duration
                    dominant_spk = max(spk_dur, key=spk_dur.get)

                    # Gán toàn bộ zone cho dominant speaker
                    zone_start = segments[i].start
                    zone_end = segments[j - 1].end
                    result.append(Segment(
                        start=zone_start, end=zone_end, speaker=dominant_spk))
                    i = j
                    continue

            result.append(Segment(
                start=segments[i].start, end=segments[i].end,
                speaker=segments[i].speaker))
            i += 1

        return result

    def _process_pyannote(self, audio_file, progress_callback, audio_data, audio_sample_rate,
                          asr_words=None):
        """Process using Pyannote/Community-1 ONNX backend"""
        # Call Pyannote backend
        result = self._pyannote_backend.process(
            audio_file, progress_callback, audio_data, audio_sample_rate
        )
        
        # Handle DiarizeOutput (new format with exclusive_speaker_diarization)
        # or list of dicts (old format)
        segments = []
        
        if hasattr(result, 'exclusive_speaker_diarization'):
            # New format: DiarizeOutput object with Annotation
            # Use exclusive_speaker_diarization for cleaner ASR mapping
            for turn, _, speaker_label in result.exclusive_speaker_diarization.itertracks(yield_label=True):
                # Extract speaker ID from label (e.g., "SPEAKER_00" -> 0)
                if isinstance(speaker_label, str) and speaker_label.startswith("SPEAKER_"):
                    speaker_id = int(speaker_label.split("_")[1])
                else:
                    speaker_id = int(speaker_label)
                
                segment = Segment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker_id
                )
                segments.append(segment)
        elif hasattr(result, 'itertracks'):
            # Annotation object directly
            for turn, _, speaker_label in result.itertracks(yield_label=True):
                if isinstance(speaker_label, str) and speaker_label.startswith("SPEAKER_"):
                    speaker_id = int(speaker_label.split("_")[1])
                else:
                    speaker_id = int(speaker_label)
                
                segment = Segment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker_id
                )
                segments.append(segment)
        else:
            # Old format: list of dicts (PyTorch — already post-processed internally)
            for seg_dict in result:
                segment = Segment(
                    start=seg_dict["start"],
                    end=seg_dict["end"],
                    speaker=seg_dict["speaker"]
                )
                segments.append(segment)

        # Post-process: merge gaps, gộp nói chêm, reassign short segments
        if segments:
            segments = self._post_process_diarization_segments(segments, asr_words=asr_words)

        if segments:
            num_speakers = max(s.speaker for s in segments) + 1
            print(f"[SpeakerDiarizer] Found {len(segments)} segments from {num_speakers} speakers")
            print(f"\n[SpeakerDiarizer] ===== SPEAKER SEGMENTS (Post-Processed) =====")
            for i, seg in enumerate(segments, 1):
                print(f"[{i:3d}] {seg}")
            print(f"[SpeakerDiarizer] ===== END OF SEGMENTS =====\n")

        return segments

    def process_with_transcription(self,
                                   audio_file: str,
                                   transcribed_segments: List[Dict],
                                   speaker_segments: Optional[List[Segment]] = None) -> List[Dict]:
        """
        Merge speaker diarization with transcription segments
        
        Args:
            audio_file: Path to audio file (kept for compatibility, not used if speaker_segments provided)
            transcribed_segments: List of transcription segments with timing
            speaker_segments: Pre-computed speaker segments (to avoid running diarization twice)
        
        Returns:
            List of transcription segments with speaker labels
        """
        # Use provided speaker segments if available, otherwise error
        if speaker_segments is None:
            raise ValueError("speaker_segments must be provided. Call process() first, then pass the result here.")
        
        print(f"[SpeakerDiarizer] Merging {len(speaker_segments)} speaker segments with {len(transcribed_segments)} transcription segments")
        
        if not speaker_segments:
            print("[SpeakerDiarizer] No speaker segments found, returning original segments")
            return transcribed_segments
        
        # Assign speaker to each transcription segment
        results = []
        for trans_seg in transcribed_segments:
            trans_start = trans_seg.get("start", 0)
            trans_end = trans_seg.get("end", trans_start + 1)
            raw_words = trans_seg.get("raw_words", [])
            
            # Find overlapping speaker segments
            speaker_votes = {}
            for spk_seg in speaker_segments:
                overlap_start = max(trans_start, spk_seg.start)
                overlap_end = min(trans_end, spk_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker_id = spk_seg.speaker
                    if speaker_id not in speaker_votes:
                        speaker_votes[speaker_id] = 0
                    speaker_votes[speaker_id] += overlap_duration
            
            # Nếu không có raw_words → segment-level voting (không thể word-level)
            if not raw_words:
                if speaker_votes:
                    best_speaker_id = max(speaker_votes, key=speaker_votes.get)
                    best_speaker = f"Người nói {best_speaker_id + 1}"
                else:
                    if results:
                        best_speaker = results[-1].get("speaker", "Người nói 1")
                        best_speaker_id = results[-1].get("speaker_id", 0)
                    else:
                        best_speaker = "Người nói 1"
                        best_speaker_id = 0
                
                seg_copy = dict(trans_seg)
                seg_copy.update({
                    "speaker": best_speaker,
                    "speaker_id": best_speaker_id
                })
                results.append(seg_copy)
                continue

            # Word-level speaker assignment
            # Tạo adjusted_segments: co boundary 150ms tại chỗ ĐỔI speaker
            # Giữ nguyên boundary giữa các segment cùng speaker
            BOUNDARY_MARGIN = 0.15   # 150ms margin tại speaker transition

            adjusted_segments = []
            n_segs = len(speaker_segments)
            for idx, seg in enumerate(speaker_segments):
                adj_start = seg.start
                adj_end = seg.end

                # Co start nếu segment trước là speaker khác
                if idx > 0 and speaker_segments[idx - 1].speaker != seg.speaker:
                    adj_start = seg.start + BOUNDARY_MARGIN

                # Co end nếu segment sau là speaker khác
                if idx < n_segs - 1 and speaker_segments[idx + 1].speaker != seg.speaker:
                    adj_end = seg.end - BOUNDARY_MARGIN

                # Đảm bảo segment còn hợp lệ (>50ms)
                if adj_end - adj_start > 0.05:
                    adjusted_segments.append(Segment(
                        start=adj_start, end=adj_end, speaker=seg.speaker))
                else:
                    adjusted_segments.append(seg)

            word_groups = []
            current_speaker_id = None
            current_group = []

            # Pre-build sorted list cho binary search trên original segments
            from bisect import bisect_right
            _orig_starts = [s.start for s in speaker_segments]

            for w in raw_words:
                w_start = w.get("start", 0)
                w_spk_id = None

                # 1) Check ORIGINAL segments trước (binary search O(log n))
                #    Quan trọng: từ ở rìa segment gốc PHẢI gán đúng, ko bị margin đẩy ra
                idx = bisect_right(_orig_starts, w_start) - 1
                if idx >= 0 and speaker_segments[idx].start <= w_start <= speaker_segments[idx].end:
                    w_spk_id = speaker_segments[idx].speaker

                # 2) Check adjusted segments (có margin 150ms tại speaker transitions)
                if w_spk_id is None:
                    for spk_seg in adjusted_segments:
                        if spk_seg.start <= w_start <= spk_seg.end:
                            w_spk_id = spk_seg.speaker
                            break

                # 3) Fallback: từ rơi vào TRUE gap (ngoài cả segment gốc)
                if w_spk_id is None:
                    prev_seg = None
                    next_seg = None
                    for spk_seg in adjusted_segments:
                        if spk_seg.end <= w_start:
                            if prev_seg is None or spk_seg.end > prev_seg.end:
                                prev_seg = spk_seg
                        elif spk_seg.start > w_start:
                            if next_seg is None or spk_seg.start < next_seg.start:
                                next_seg = spk_seg

                    if prev_seg and next_seg:
                        dist_prev = w_start - prev_seg.end
                        dist_next = next_seg.start - w_start
                        if prev_seg.speaker != next_seg.speaker:
                            # Khác speaker: gán cho speaker GẦN HƠN
                            if dist_next <= dist_prev:
                                w_spk_id = next_seg.speaker
                            else:
                                w_spk_id = prev_seg.speaker
                        else:
                            # Cùng speaker → gán luôn
                            w_spk_id = prev_seg.speaker
                    elif prev_seg:
                        w_spk_id = prev_seg.speaker
                    elif next_seg:
                        w_spk_id = next_seg.speaker
                    else:
                        w_spk_id = current_speaker_id if current_speaker_id is not None else max(speaker_votes, key=speaker_votes.get)

                if w_spk_id != current_speaker_id:
                    if current_group:
                        word_groups.append((current_speaker_id, current_group))
                    current_speaker_id = w_spk_id
                    current_group = [w]
                else:
                    current_group.append(w)
            
            if current_group:
                word_groups.append((current_speaker_id, current_group))

            # If ended up with only 1 group, no split needed
            if len(word_groups) == 1:
                spk_id = word_groups[0][0]
                seg_copy = dict(trans_seg)
                seg_copy.update({
                    "speaker": f"Người nói {spk_id + 1}",
                    "speaker_id": spk_id
                })
                results.append(seg_copy)
                continue

            # Split text proportionally
            punct_words = trans_seg.get("text", "").split()
            total_raw = len(raw_words)
            punct_idx = 0

            for i, (spk_id, group_words) in enumerate(word_groups):
                g_start = group_words[0].get("start", trans_start)
                g_end = group_words[-1].get("end", trans_end)
                
                # Assign punctuated words to this group
                if i == len(word_groups) - 1:
                    g_punct = punct_words[punct_idx:]
                else:
                    num_punct = int(round(len(group_words) / float(total_raw) * len(punct_words)))
                    # Ensure at least 1 word if possible
                    if num_punct == 0 and punct_idx < len(punct_words):
                        num_punct = 1
                    g_punct = punct_words[punct_idx:punct_idx + num_punct]
                    punct_idx += num_punct
                
                g_text = " ".join(g_punct)
                
                seg_copy = dict(trans_seg)
                seg_copy.update({
                    "text": g_text,
                    "start": g_start,
                    "end": g_end,
                    "speaker": f"Người nói {spk_id + 1}",
                    "speaker_id": spk_id,
                    "raw_words": group_words,  # Giữ raw_words cho từng group
                })
                results.append(seg_copy)
        
        # ── Cross-segment speech continuity correction ──
        # Diarization model có thể cắt boundary sai: speaker A vẫn đang nói
        # nhưng model đã chuyển sang segment speaker B.
        # Nguyên tắc: nếu words đầu segment B liên tục (gap < 0.3s) với word cuối
        # segment A VÀ words đó nằm trong diarization segment gốc của B (không phải gap),
        # → chúng thực ra là speech tiếp nối của A, diarization cắt sai → chuyển về A.
        # Chạy trên toàn bộ results (cross-sentence) vì sentence segmentation có thể
        # tách phrase thành nhiều trans_segs khác nhau.
        from bisect import bisect_right as _br
        _orig_starts = [s.start for s in speaker_segments]
        SPEECH_CONT_GAP = 0.3

        i = 0
        while i < len(results) - 1:
            seg_a = results[i]
            seg_b = results[i + 1]
            spk_a = seg_a.get("speaker_id")
            spk_b = seg_b.get("speaker_id")
            rw_a = seg_a.get("raw_words", [])
            rw_b = seg_b.get("raw_words", [])

            if spk_a is None or spk_b is None or spk_a == spk_b or not rw_a or not rw_b:
                i += 1
                continue

            last_end = rw_a[-1].get("end", 0)
            move_count = 0
            for w in rw_b:
                ws = w.get("start", 0)
                if ws - last_end < SPEECH_CONT_GAP:
                    # Chỉ move nếu word KHÔNG nằm trong segment gốc của spk_b
                    # (word rơi vào gap hoặc thuộc segment spk_a → ASR cắt sai)
                    idx2 = _br(_orig_starts, ws) - 1
                    word_in_spk_b_seg = (idx2 >= 0
                        and speaker_segments[idx2].start <= ws <= speaker_segments[idx2].end
                        and speaker_segments[idx2].speaker == spk_b)
                    if word_in_spk_b_seg:
                        # Word đúng là của spk_b theo diarization → dừng, không move
                        break
                    move_count += 1
                    last_end = w.get("end", 0)
                else:
                    break

            if move_count > 0 and move_count < len(rw_b):
                # Di chuyển words từ đầu seg_b sang cuối seg_a
                moved_words = rw_b[:move_count]
                remaining_words = rw_b[move_count:]
                moved_text = " ".join(w.get("text", "") for w in moved_words)
                remaining_text = " ".join(w.get("text", "") for w in remaining_words)

                seg_a["raw_words"] = rw_a + moved_words
                seg_a["text"] = (seg_a.get("text", "") + " " + moved_text).strip()
                seg_a["end"] = moved_words[-1].get("end", seg_a["end"])

                seg_b["raw_words"] = remaining_words
                seg_b["text"] = remaining_text
                seg_b["start"] = remaining_words[0].get("start", seg_b["start"])
                # Không tăng i — check lại transition mới
            elif move_count > 0 and move_count == len(rw_b):
                # Toàn bộ seg_b chuyển sang seg_a → merge hoàn toàn
                seg_a["raw_words"] = rw_a + rw_b
                seg_a["text"] = (seg_a.get("text", "") + " " + seg_b.get("text", "")).strip()
                seg_a["end"] = rw_b[-1].get("end", seg_b["end"])
                results.pop(i + 1)
                # Không tăng i — check segment tiếp theo
            else:
                i += 1

        # ── Fix trailing word at speaker boundary ──
        # Khi ASR timestamp drift, word cuối segment A thực ra thuộc segment B.
        # Pattern: word cuối A + word đầu B tạo thành cụm cố định (VD: "kính thưa",
        # "xin mời", "xin chào") → chuyển word cuối A sang B.
        # Tổng quát hơn: nếu word cuối A nằm NGOÀI diarization segment của A
        # (timestamp > segment.end) → nên thuộc B.
        i = 0
        while i < len(results) - 1:
            seg_a = results[i]
            seg_b = results[i + 1]
            spk_a = seg_a.get("speaker_id")
            spk_b = seg_b.get("speaker_id")
            rw_a = seg_a.get("raw_words", [])
            rw_b = seg_b.get("raw_words", [])

            if spk_a is None or spk_b is None or spk_a == spk_b or not rw_a or not rw_b:
                i += 1
                continue

            # Check: word cuối A nằm ngoài diarization segment của A?
            last_word = rw_a[-1]
            lw_start = last_word.get("start", 0)

            # Tìm diarization segment chứa last word
            idx_lw = _br(_orig_starts, lw_start) - 1
            word_in_spk_a = (idx_lw >= 0
                and speaker_segments[idx_lw].start <= lw_start <= speaker_segments[idx_lw].end
                and speaker_segments[idx_lw].speaker == spk_a)

            if not word_in_spk_a and len(rw_a) > 1:
                # Word cuối A không thuộc segment A → chuyển sang B
                moved_word = rw_a.pop()
                seg_a["end"] = rw_a[-1].get("end", seg_a["end"])
                seg_a["text"] = " ".join(w.get("text", "") for w in rw_a)

                rw_b.insert(0, moved_word)
                seg_b["start"] = moved_word.get("start", seg_b["start"])
                seg_b["raw_words"] = rw_b
                seg_b["text"] = " ".join(w.get("text", "") for w in rw_b)
                # Don't increment — check again
                continue

            i += 1

        return results

    def format_by_speaker(self, segments: List[Dict]) -> str:
        """
        Format segments grouped by speaker
        
        Người nói 1:
        <câu 1> <câu 2> <câu 3>
        
        Người nói 2:
        <câu 4> <câu 5> <câu 6>
        """
        if not segments:
            return "Không có nội dung"
        
        # Group consecutive segments by speaker
        grouped = []
        current_speaker = None
        current_texts = []
        
        for seg in segments:
            speaker = seg.get("speaker", "Người nói 1")
            text = seg.get("text", "").strip()
            
            if not text:
                continue
            
            if speaker != current_speaker:
                if current_speaker and current_texts:
                    grouped.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_texts)
                    })
                current_speaker = speaker
                current_texts = [text]
            else:
                current_texts.append(text)
        
        # Don't forget the last group
        if current_speaker and current_texts:
            grouped.append({
                "speaker": current_speaker,
                "text": " ".join(current_texts)
            })
        
        # Format output with speaker names in Vietnamese
        lines = []
        for group in grouped:
            # Extract speaker number safely
            match = re.search(r'(\d+)$', group['speaker'])
            speaker_num = match.group(1) if match else "1"
            lines.append(f"Người nói {speaker_num}:")
            lines.append(group['text'])
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def unload(self):
        """
        Giải phóng bộ nhớ bằng cách unload model speaker diarization.
        Gọi method này khi muốn tiết kiệm RAM sau khi xử lý xong.
        """
        import gc
        
        if self.sd is not None:
            print("[SpeakerDiarizer] Unloading speaker diarization model...")
            del self.sd
            self.sd = None
            gc.collect()
            print("[SpeakerDiarizer] Model unloaded successfully")
        
        if self._pyannote_backend is not None:
            self._pyannote_backend.unload()
            self._pyannote_backend = None


def default_progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    """Default progress callback that prints progress"""
    progress = num_processed_chunk / num_total_chunks * 100
    print(f"[SpeakerDiarizer] Progress: {progress:.1f}%")
    return 0


def test_diarization(model_id: str = "titanet_small", audio_file: str = None):
    """Test speaker diarization with specified model"""
    print(f"\n{'='*60}")
    print(f"Testing Speaker Diarization")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")
    
    if audio_file is None:
        print("Error: No audio file provided.")
        print(f"Usage: python speaker_diarization.py --model {model_id} --audio <path>")
        return None
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return None
    
    # Get model info
    model_info = get_model_info(model_id)
    if model_info:
        print(f"Model: {model_info['name']}")
        print(f"Size: {model_info['size']}")
        print(f"Description: {model_info['description']}")
        print()
    
    # Initialize diarizer with model
    start_time = time.time()
    diarizer = SpeakerDiarizer(
        embedding_model_id=model_id,
        num_clusters=-1,
        num_threads=4
    )
    
    print(f"Using test file: {audio_file}")
    
    # Process
    try:
        segments = diarizer.process(audio_file, progress_callback=default_progress_callback)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"{'='*60}")
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Found {len(segments)} segments")
        
        if segments:
            num_speakers = max(s.speaker for s in segments) + 1
            print(f"Number of speakers: {num_speakers}")
            print(f"\nFirst 10 segments:")
            for seg in segments[:10]:
                print(f"  {seg}")
        
        return segments
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_all_models(audio_file: str = None):
    """Test all available models"""
    print("\n" + "="*70)
    print("TESTING ALL SPEAKER EMBEDDING MODELS")
    print("="*70 + "\n")
    
    available = get_available_models()
    
    if not available:
        print("No models found! Please download models first.")
        return
    
    print(f"Found {len(available)} available models:\n")
    for model_id in available.keys():
        info = get_model_info(model_id)
        if info:
            print(f"  - {model_id}: {info['name']} ({info['size']})")
    print()
    
    results = {}
    for model_id in available.keys():
        segments = test_diarization(model_id, audio_file)
        results[model_id] = segments
        print("\n" + "-"*70 + "\n")
    
    return results


def _remap_speakers_to_sentences(original_segments, word_speaker_map):
    """
    Gán lại speaker labels từ word-level mapping vào các câu đã tách sẵn.
    Nếu 1 câu chứa words từ nhiều speakers → split câu tại ranh giới speaker.
    """
    results = []
    for sent_seg in original_segments:
        raw_words = sent_seg.get("raw_words", [])
        if not raw_words:
            results.append(dict(sent_seg))
            continue

        # Group consecutive words by speaker
        word_groups = []
        cur_spk = None
        cur_spk_id = None
        cur_words = []

        for w in raw_words:
            info = word_speaker_map.get(id(w))
            if info:
                spk, spk_id = info
            else:
                spk = cur_spk or "Người nói 1"
                spk_id = cur_spk_id if cur_spk_id is not None else 0

            if spk_id != cur_spk_id:
                if cur_words:
                    word_groups.append((cur_spk, cur_spk_id, cur_words))
                cur_spk = spk
                cur_spk_id = spk_id
                cur_words = [w]
            else:
                cur_words.append(w)

        if cur_words:
            word_groups.append((cur_spk, cur_spk_id, cur_words))

        if len(word_groups) <= 1:
            seg_copy = dict(sent_seg)
            if word_groups:
                seg_copy["speaker"] = word_groups[0][0]
                seg_copy["speaker_id"] = word_groups[0][1]
            results.append(seg_copy)
        else:
            punct_words = sent_seg.get("text", "").split()
            total_raw = len(raw_words)
            punct_idx = 0

            for i, (spk, spk_id, group_words) in enumerate(word_groups):
                g_start = group_words[0].get("start", 0)
                g_end = group_words[-1].get("end", 0)

                if i == len(word_groups) - 1:
                    g_punct = punct_words[punct_idx:]
                else:
                    num_punct = int(round(len(group_words) / float(total_raw) * len(punct_words)))
                    if num_punct == 0 and punct_idx < len(punct_words):
                        num_punct = 1
                    g_punct = punct_words[punct_idx:punct_idx + num_punct]
                    punct_idx += num_punct

                seg_copy = dict(sent_seg)
                seg_copy.update({
                    "text": " ".join(g_punct),
                    "start": g_start,
                    "end": g_end,
                    "speaker": spk,
                    "speaker_id": spk_id,
                    "raw_words": group_words,
                })
                results.append(seg_copy)

    return results


def _diarize_and_remap(diarizer_instance, audio_file, segments, raw_segments):
    """
    Chạy word-level speaker assignment trên TOÀN BỘ raw_words (giống initial pipeline),
    rồi remap kết quả về các câu đã tách sẵn.
    """
    all_raw_words = []
    for seg in segments:
        all_raw_words.extend(seg.get("raw_words", []))

    if not all_raw_words:
        return diarizer_instance.process_with_transcription(
            audio_file=audio_file,
            transcribed_segments=segments,
            speaker_segments=raw_segments
        )

    all_raw_words.sort(key=lambda w: w.get("start", 0))
    one_seg = [{
        "text": " ".join(w.get("text", "") for w in all_raw_words),
        "start": all_raw_words[0].get("start", 0),
        "end": all_raw_words[-1].get("end", 0),
        "raw_words": all_raw_words,
    }]

    diar_results = diarizer_instance.process_with_transcription(
        audio_file=audio_file,
        transcribed_segments=one_seg,
        speaker_segments=raw_segments
    )

    if len(segments) <= 1:
        return diar_results

    word_speaker_map = {}
    for dseg in diar_results:
        spk = dseg.get("speaker")
        spk_id = dseg.get("speaker_id")
        for w in dseg.get("raw_words", []):
            word_speaker_map[id(w)] = (spk, spk_id)

    return _remap_speakers_to_sentences(segments, word_speaker_map)


def run_diarization(audio_file, segments, speaker_model_id, num_speakers, num_threads,
                    threshold=0.6, progress_callback=None, cancel_check=None):
    """
    Chạy speaker diarization trên file audio (high-level orchestration).

    Args:
        audio_file: Đường dẫn file audio
        segments: List[dict] - transcribed segments với keys: text, start, end
        speaker_model_id: str - ID model embedding
        num_speakers: int - số người nói (0 = auto)
        num_threads: int - số CPU threads
        threshold: float - ngưỡng phân biệt người nói (0.0-1.0)
        progress_callback: callable(str) - callback báo tiến trình
        cancel_check: callable() -> bool - trả True nếu cần hủy

    Returns:
        tuple: (speaker_segments_raw, elapsed, result_segments)
    """
    import time as _time

    emit = progress_callback or (lambda msg: None)
    is_cancelled = cancel_check or (lambda: False)

    start_time = _time.time()

    try:
        _setup_ffmpeg_path()
    except Exception:
        pass

    emit("PHASE:Diarization|Đang khởi tạo model|0")

    # --- CAM++ Senko pipeline (best for all audio lengths) ---
    if speaker_model_id in ("senko_campp", "senko_campp_optimized"):
        if speaker_model_id == "senko_campp_optimized":
            from core.speaker_diarization_senko_campp_optimized import SenkoCamppDiarizerOptimized
            diarizer_3d = SenkoCamppDiarizerOptimized(
                num_speakers=num_speakers, num_threads=num_threads)
            label = "Senko CAM++ OPT"
        else:
            from core.speaker_diarization_senko_campp import SenkoCamppDiarizer
            diarizer_3d = SenkoCamppDiarizer(
                num_speakers=num_speakers, num_threads=num_threads)
            label = "Senko CAM++"
        diarizer_3d.initialize()

        def campp_progress(pct):
            emit(f"PHASE:Diarization|Đang phân tách Người nói ({label})|{int(pct)}")

        emit(f"PHASE:Diarization|Đang phân tách Người nói ({label})|10")
        raw_dict_segments = diarizer_3d.process(
            audio_file=audio_file, progress_callback=campp_progress)

        speaker_segments_raw = [
            {
                "speaker": f"Người nói {seg['speaker'] + 1}",
                "speaker_id": seg['speaker'],
                "start": seg['start'],
                "end": seg['end'],
                "duration": seg['end'] - seg['start']
            }
            for seg in raw_dict_segments
        ]

        emit("PHASE:Diarization|Đang gán nhãn Người nói|90")

        raw_segments = [Segment(s['start'], s['end'], s['speaker']) for s in raw_dict_segments]
        merger = SpeakerDiarizer()
        raw_segments = merger._post_process_diarization_segments(raw_segments)

        speaker_segments_raw = [
            {
                "speaker": f"Người nói {seg.speaker + 1}",
                "speaker_id": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "duration": seg.duration
            }
            for seg in raw_segments
        ]

        results = _diarize_and_remap(merger, audio_file, segments, raw_segments)

        elapsed = _time.time() - start_time
        emit("PHASE:Diarization|Hoàn thành|100")
        return speaker_segments_raw, elapsed, results

    # --- Pyannote Community-1 pipeline (default) ---
    diarizer = SpeakerDiarizer(
        embedding_model_id=speaker_model_id,
        num_clusters=num_speakers,
        num_threads=num_threads,
        threshold=threshold
    )
    diarizer.initialize()

    emit("PHASE:Diarization|Đang phân tách Người nói|10")

    _last_progress = [0]

    def internal_progress_callback(num_processed, num_total):
        if num_total == 0:
            return 0
        progress = int(num_processed / num_total * 100)
        if progress >= _last_progress[0] + 5 or num_processed == num_total:
            _last_progress[0] = progress
            phase_progress = 10 + int(progress * 0.75)
            emit(f"PHASE:Diarization|Đang phân tách Người nói|{phase_progress}")
            _time.sleep(0.001)
        return 1 if is_cancelled() else 0

    raw_segments = diarizer.process(audio_file, progress_callback=internal_progress_callback)

    speaker_segments_raw = [
        {
            "speaker": f"Người nói {seg.speaker + 1}",
            "speaker_id": seg.speaker,
            "start": seg.start,
            "end": seg.end,
            "duration": seg.duration
        }
        for seg in raw_segments
    ]

    emit("PHASE:Diarization|Đang gán nhãn Người nói|90")

    results = _diarize_and_remap(diarizer, audio_file, segments, raw_segments)

    # Giải phóng model — chỉ dùng kết quả data từ đây
    diarizer.unload()
    del diarizer

    elapsed = _time.time() - start_time
    emit("PHASE:Diarization|Hoàn thành|100")

    return speaker_segments_raw, elapsed, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test speaker diarization models")
    parser.add_argument("--model", "-m", default="titanet_small",
                       help="Model ID to test (default: titanet_small)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Test all available models")
    parser.add_argument("--audio", default=None,
                       help="Path to audio file for testing")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available models")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Speaker Embedding Models:")
        print("="*70)
        available = get_available_models()
        for model_id, info in SPEAKER_EMBEDDING_MODELS.items():
            status = "[OK]" if model_id in available else "[MISSING]"
            print(f"\n{status} {model_id}:")
            print(f"   Name: {info['name']}")
            print(f"   File: {info['file']}")
            print(f"   Size: {info['size']}")
            print(f"   Language: {info['language']}")
            print(f"   Speed: {info['speed']}")
            print(f"   Accuracy: {info['accuracy']}")
            print(f"   Description: {info['description']}")
    elif args.all:
        test_all_models(args.audio)
    else:
        test_diarization(args.model, args.audio)
