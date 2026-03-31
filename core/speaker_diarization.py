"""
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
        "name": "Community-1 ONNX (Pure ORT)",
        "file": "community1_pure_ort",  # Special marker
        "size": "~32 MB",
        "language": "Multilingual",
        "speed": "Fastest",
        "accuracy": "Excellent",
        "sample_rate": 16000,
        "description": "Pyannote Community-1 via ONNX Runtime, no PyTorch dependency"
    }
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
                 auth_token: str = None,
                 merge_short_speaker: bool = True):
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
        self.merge_short_speaker = merge_short_speaker
        
    # Default thresholds for each model
    MODEL_DEFAULT_THRESHOLDS = {
        "community1_pure_ort": 0.7,
    }

    @classmethod
    def get_default_threshold(cls, model_id: str) -> float:
        """Get default threshold for a specific model"""
        return cls.MODEL_DEFAULT_THRESHOLDS.get(model_id, 0.7)

    def initialize(self):
        """Initialize Pure ORT speaker diarization"""
        self._init_pure_ort()
    
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
                audio_sample_rate: Optional[int] = None) -> List[Segment]:
        """Perform speaker diarization on audio file"""
        SpeakerDiarizer._call_count += 1

        if self._pyannote_backend is None:
            self.initialize()

        return self._process_pyannote(audio_file, progress_callback, audio_data, audio_sample_rate)

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

    def _merge_short_islands(self, segments: List[Segment], max_short_duration: float = 1.5) -> List[Segment]:
        """
        Gộp các đoạn ngắn (< max_short_duration) nằm giữa cùng 1 người nói.
        Pattern: SpkA -> [SpkX(<1.5s)] -> [SpkY(<1.5s)] -> ... -> SpkA
        → Gộp tất cả vào SpkA
        """
        if len(segments) < 3:
            return segments
            
        n = len(segments)
        to_skip = set()
        
        i = 0
        while i < n:
            if i in to_skip:
                i += 1
                continue
                
            spk_a = segments[i].speaker
            
            # Tìm chuỗi các đoạn ngắn liên tiếp phía sau
            j = i + 1
            short_segments_indices = []
            
            while j < n:
                if segments[j].duration < max_short_duration:
                    short_segments_indices.append(j)
                    j += 1
                else:
                    break
            
            # Nếu sau chuỗi ngắn là SpkA, đánh dấu gộp
            if short_segments_indices and j < n:
                if segments[j].speaker == spk_a:
                    for idx in short_segments_indices:
                        to_skip.add(idx)
                    # j là đoạn SpkA kết thúc chuỗi, ta sẽ xử lý tiếp từ j
                    i = j
                    continue
            
            i += 1
            
        # Tạo kết quả cuối cùng
        result = []
        i = 0
        while i < n:
            if i in to_skip:
                i += 1
                continue
                
            current = segments[i]
            
            # Kiểm tra xem có chuỗi gộp bắt đầu từ đây không
            if i + 1 < n and (i + 1) in to_skip:
                j = i + 1
                while j < n and j in to_skip:
                    j += 1
                
                # j là đoạn SpkA tiếp theo
                if j < n and segments[j].speaker == current.speaker:
                    # Tạo segment mới kéo dài từ đầu SpkA này đến hết SpkA kia
                    current = Segment(start=current.start, end=segments[j].end, speaker=current.speaker)
                    i = j + 1
                    result.append(current)
                    continue
            
            result.append(current)
            i += 1
            
        return result
    
    def _post_process_diarization_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Post-processing cho exclusive_speaker_diarization trước khi dùng cho ASR.
        Áp dụng cho ONNX path (PyTorch đã tự xử lý trong Community1Diarizer).

        Thứ tự:
        1. Merge adjacent + small gaps (0.3s)
        2. Smooth A-B-A với ngưỡng 0.5s
        3. Reassign cực ngắn (<0.2s)
        4. Final merge
        """
        if not segments:
            return segments

        original_count = len(segments)

        # 1. Merge đoạn liền kề + gap nhỏ (0.3s) cùng speaker
        segments = self._merge_segments_with_gap(segments, max_gap=0.3)

        # 2. Smooth A-B-A với ngưỡng 0.5s
        segments = self._smooth_aba_segments(segments, max_middle_duration=0.5)

        # 3. Reassign segment cực ngắn (<0.2s) vào neighbor
        segments = self._reassign_short(segments, min_duration=0.3)

        # 4. Final merge sau khi smooth
        segments = self._merge_segments_with_gap(segments, max_gap=0.0)

        final_count = len(segments)
        if final_count < original_count:
            print(f"[SpeakerDiarizer] Post-process ASR: {original_count} -> {final_count} segments")

        return segments

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

    def _smooth_aba_segments(self, segments: List[Segment], max_middle_duration: float = 0.5) -> List[Segment]:
        """Loại bỏ chuyển đổi giả kiểu A-B-A. B < max_middle_duration → gộp vào A."""
        if len(segments) < 3:
            return segments

        smoothed = list(segments)
        changes_made = True
        iterations = 0

        while changes_made and iterations < 3:
            changes_made = False
            iterations += 1
            new_smoothed = []
            i = 0

            while i < len(smoothed):
                if i == 0 or i >= len(smoothed) - 1:
                    new_smoothed.append(smoothed[i])
                    i += 1
                    continue

                prev_seg = smoothed[i - 1]
                cur_seg = smoothed[i]
                next_seg = smoothed[i + 1]

                if (cur_seg.duration <= max_middle_duration and
                        prev_seg.speaker == next_seg.speaker and
                        cur_seg.speaker != prev_seg.speaker):
                    merged_seg = Segment(
                        start=prev_seg.start, end=next_seg.end, speaker=prev_seg.speaker)
                    if new_smoothed:
                        new_smoothed[-1] = merged_seg
                    else:
                        new_smoothed.append(merged_seg)
                    changes_made = True
                    i += 2
                else:
                    new_smoothed.append(cur_seg)
                    i += 1

            smoothed = new_smoothed

        return smoothed

    def _reassign_short(self, segments: List[Segment], min_duration: float = 0.2) -> List[Segment]:
        """Reassign segment cực ngắn vào speaker lân cận."""
        if not segments:
            return segments

        result = []
        n = len(segments)

        for i, seg in enumerate(segments):
            if seg.duration >= min_duration:
                result.append(Segment(start=seg.start, end=seg.end, speaker=seg.speaker))
                continue

            prev_speaker = result[-1].speaker if result else None
            next_speaker = segments[i + 1].speaker if i < n - 1 else None

            if prev_speaker is not None:
                result[-1].end = seg.end
            elif next_speaker is not None:
                result.append(Segment(start=seg.start, end=seg.end, speaker=next_speaker))
            else:
                result.append(Segment(start=seg.start, end=seg.end, speaker=seg.speaker))

        return result

    def _process_pyannote(self, audio_file, progress_callback, audio_data, audio_sample_rate):
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

        # Post-process: merge, smooth A-B-A, reassign short segments
        if segments:
            segments = self._post_process_diarization_segments(segments)

        # _clip_segments_to_speech đã bỏ: Fix 1,2,3 word-level xử lý boundary
        # chính xác hơn, và clip tạo ra segments ngắn không mong muốn

        # Post-processing: Merge short islands (< 1.5s) between same speaker
        if segments and self.merge_short_speaker:
            segments = self._merge_short_islands(segments, max_short_duration=1.5)

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
                            # Speech continuity: trong 1 lượt nói, inter-word gap < 0.3s.
                            # Nếu từ cách segment trước > 0.5s → speaker trước đã dừng
                            # (turn-taking pause) → từ thuộc speaker sau.
                            TURN_GAP_THRESHOLD = 0.5
                            if dist_prev > TURN_GAP_THRESHOLD:
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
                    # Chỉ move nếu word nằm trong segment gốc của spk_b
                    idx2 = _br(_orig_starts, ws) - 1
                    if idx2 >= 0 and speaker_segments[idx2].start <= ws <= speaker_segments[idx2].end:
                        if speaker_segments[idx2].speaker == spk_b:
                            move_count += 1
                            last_end = w.get("end", 0)
                            continue
                    break
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

    results = diarizer.process_with_transcription(
        audio_file=audio_file,
        transcribed_segments=segments,
        speaker_segments=raw_segments
    )

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
