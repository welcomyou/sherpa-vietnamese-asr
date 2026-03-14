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
    "titanet_small": {
        "name": "Nvidia NeMo Titanet Small", 
        "file": "nemo_en_titanet_small.onnx",
        "size": "38.4 MB",
        "language": "English",
        "speed": "Fast",
        "accuracy": "Good",
        "sample_rate": 16000,
        "description": "Fast English model, good balance"
    },
    "community1": {
        "name": "Pyannote Community-1 (SOTA Accuracy)",
        "file": "community1_pipeline",  # Special marker - not a file
        "size": "~40 MB",
        "language": "Multilingual",
        "speed": "Medium",
        "accuracy": "Excellent",
        "sample_rate": 16000,
        "description": "State-of-the-art diarization, better than 3.1, requires HF token"
    },
    "community1_onnx": {
        "name": "Community-1 ONNX (Altunenes)",
        "file": "community1_onnx",  # Special marker - ONNX models
        "size": "~32 MB",
        "language": "Multilingual",
        "speed": "Fast",
        "accuracy": "Very Good",
        "sample_rate": 16000,
        "description": "Pure ONNX - No pyannote.audio needed, faster inference"
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
        model_path = os.path.join(models_dir, info["file"])
        # Special handling for pyannote - check if pyannote.audio is installed
        if model_id == "community1":
            try:
                import pyannote.audio
                available[model_id] = "pyannote_pipeline"
            except ImportError:
                pass
        # Special handling for Altunenes ONNX - check if ONNX models exist
        elif model_id == "community1_onnx":
            onnx_model_dir = os.path.join(base_dir, "models", "pyannote-onnx")
            seg_path = os.path.join(onnx_model_dir, "segmentation-community-1.onnx")
            emb_path = os.path.join(onnx_model_dir, "embedding_model.onnx")
            if os.path.exists(seg_path) and os.path.exists(emb_path):
                available[model_id] = "altunenes_onnx"
        elif os.path.exists(model_path):
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
        "titanet_small": 0.85,      # Nemo Titanet - 0.85 for better accuracy
        "community1": 0.7,          # Pyannote PyTorch - 0.7 for better accuracy
        "community1_onnx": 0.7,     # Pyannote ONNX - default
    }
    
    @classmethod
    def get_default_threshold(cls, model_id: str) -> float:
        """Get default threshold for a specific model"""
        return cls.MODEL_DEFAULT_THRESHOLDS.get(model_id, 0.6)
    
    def _is_pyannote_model(self) -> bool:
        """Check if using Pyannote 3.1 ONNX model"""
        return self.embedding_model_id == "community1"
    
    def _is_altunenes_onnx_model(self) -> bool:
        """Check if using Altunenes ONNX model"""
        return self.embedding_model_id == "community1_onnx"
        
    def initialize(self):
        """Initialize models using sherpa-onnx or Pyannote ONNX"""
        # Check if using Pyannote 3.1
        if self._is_pyannote_model():
            self._init_pyannote()
            return
        
        # Check if using Altunenes ONNX
        if self._is_altunenes_onnx_model():
            self._init_altunenes_onnx()
            return
            
        # Standard sherpa-onnx initialization
        self._init_sherpa_onnx()
    
    def _init_pyannote(self):
        """Initialize Pyannote 3.1 ONNX backend"""
        try:
            from core.speaker_diarization_pyannote import Community1Diarizer
        except ImportError as e:
            raise RuntimeError(
                f"Pyannote ONNX diarizer not available: {e}\n"
                "Install with: pip install pyannote.audio"
            )
        
        # Nếu user chọn số speaker cụ thể, tạo range linh hoạt xung quanh số đó
        # min = max(2, num-1), max = num+1
        min_spk, max_spk = self.min_speakers, self.max_speakers
        num_spk = self.num_clusters
        if self.num_clusters > 0:
            min_spk = max(2, self.num_clusters - 1)
            max_spk = self.num_clusters + 1
            num_spk = -1  # Để model dùng min/max thay vì ép cứng
            print(f"[SpeakerDiarizer] User selected {self.num_clusters} speakers -> Range: min={min_spk}, max={max_spk}")
        else:
            print(f"[SpeakerDiarizer] Auto-detect speakers")

        self._pyannote_backend = Community1Diarizer(
            num_speakers=num_spk,
            min_speakers=min_spk,
            max_speakers=max_spk,
            num_threads=self.num_threads,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            auth_token=self.auth_token,
            threshold=self.threshold
        )
        self._pyannote_backend.initialize()
        self.model_info = get_model_info("community1")
    
    def _init_altunenes_onnx(self):
        """Initialize Altunenes ONNX backend (pure ONNX, no pyannote.audio)"""
        try:
            from core.speaker_diarization_onnx_altunenes import AltunenesONNXDiarizer
        except ImportError as e:
            raise RuntimeError(
                f"Altunenes ONNX diarizer not available: {e}\n"
                "Install with: pip install onnxruntime scikit-learn"
            )
        
        # Nếu user chọn số speaker cụ thể, tạo range linh hoạt xung quanh số đó
        # min = max(2, num-1), max = num+1
        min_spk, max_spk = self.min_speakers, self.max_speakers
        num_spk = self.num_clusters
        if self.num_clusters > 0:
            min_spk = max(2, self.num_clusters - 1)
            max_spk = self.num_clusters + 1
            num_spk = -1  # Để model dùng min/max thay vì ép cứng
            print(f"[SpeakerDiarizer] User selected {self.num_clusters} speakers -> Range: min={min_spk}, max={max_spk}")
        else:
            print(f"[SpeakerDiarizer] Auto-detect speakers")

        self._pyannote_backend = AltunenesONNXDiarizer(
            num_speakers=num_spk,
            min_speakers=min_spk,
            max_speakers=max_spk,
            num_threads=self.num_threads,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            threshold=self.threshold
        )
        self._pyannote_backend.initialize()
        self.model_info = get_model_info("community1_onnx")
    
    def _init_sherpa_onnx(self):
        """Initialize sherpa-onnx backend"""
        # Try to import sherpa_onnx here (for embedded Python compatibility)
        try:
            import sherpa_onnx as so
            global _sherpa_onnx, SHERPA_AVAILABLE
            _sherpa_onnx = so
            SHERPA_AVAILABLE = True
        except ImportError:
            pass
            
        if not all([SHERPA_AVAILABLE, LIBROSA_AVAILABLE, SOUNDFILE_AVAILABLE]):
            missing = []
            if not SHERPA_AVAILABLE: missing.append("sherpa_onnx")
            if not LIBROSA_AVAILABLE: missing.append("librosa")
            if not SOUNDFILE_AVAILABLE: missing.append("soundfile")
            raise RuntimeError(f"Missing dependencies: {missing}")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Default segmentation model path
        if self.segmentation_model is None:
            self.segmentation_model = os.path.join(
                base_dir, "models", "speaker_diarization",
                "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"
            )
        
        # Resolve embedding model path
        if self.embedding_model is None:
            if self.embedding_model_id:
                # Use model ID to get path
                self.embedding_model = get_model_path(self.embedding_model_id, base_dir)
                self.model_info = get_model_info(self.embedding_model_id)
                if self.embedding_model is None:
                    raise FileNotFoundError(
                        f"Model '{self.embedding_model_id}' not found. "
                        f"Please download it first or choose another model."
                    )
            else:
                # Default to titanet_small (Fast English model)
                self.embedding_model = os.path.join(
                    base_dir, "models", "speaker_embedding",
                    "nemo_en_titanet_small.onnx"
                )
                self.embedding_model_id = "titanet_small"
                self.model_info = get_model_info("titanet_small")
        
        # Check model files exist
        if not os.path.exists(self.segmentation_model):
            raise FileNotFoundError(f"Segmentation model not found: {self.segmentation_model}")
        if not os.path.exists(self.embedding_model):
            raise FileNotFoundError(f"Embedding model not found: {self.embedding_model}")
        
        print(f"[SpeakerDiarizer] Loading segmentation model: {self.segmentation_model}")
        print(f"[SpeakerDiarizer] Loading embedding model: {self.embedding_model}")
        if self.model_info:
            print(f"[SpeakerDiarizer] Model info: {self.model_info['name']} ({self.model_info['size']})")
        
        # Model Sherpa chỉ có 1 tham số num_clusters nên dùng trực tiếp
        num_clusters = self.num_clusters
        if self.num_clusters > 0:
            print(f"[SpeakerDiarizer] User selected {self.num_clusters} speakers (Sherpa fixed count)")
        else:
            print(f"[SpeakerDiarizer] Auto-detect speakers")
        
        # Create config following the sample code pattern
        # Note: num_threads needs to be set in BOTH segmentation and embedding configs
        so = get_sherpa_onnx()
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=self.segmentation_model
                ),
                num_threads=self.num_threads,  # Set threads for segmentation model
            ),
            embedding=so.SpeakerEmbeddingExtractorConfig(
                model=self.embedding_model,
                num_threads=self.num_threads,  # Set threads for embedding extractor
            ),
            clustering=so.FastClusteringConfig(
                num_clusters=num_clusters,  # Number of speakers (-1 for auto-detect)
                threshold=self.threshold          # Clustering threshold (smaller = more speakers)
            ),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        
        if not config.validate():
            raise RuntimeError(
                "Please check your config and make sure all required files exist"
            )
        
        self.sd = so.OfflineSpeakerDiarization(config)
        print(f"[SpeakerDiarizer] Initialized successfully (sample_rate={self.sd.sample_rate})")
    
    def process(self, 
                audio_file: str, 
                progress_callback: Optional[Callable[[int, int], int]] = None,
                audio_data: Optional[np.ndarray] = None,
                audio_sample_rate: Optional[int] = None) -> List[Segment]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_file: Path to audio file (used if audio_data not provided)
            progress_callback: Optional callback function(num_processed_chunk, num_total_chunks) -> int
                            Return 0 to continue, non-zero to stop
            audio_data: Pre-loaded audio data (optional, to avoid loading twice)
            audio_sample_rate: Sample rate of pre-loaded audio data
        
        Returns:
            List of Segment objects with speaker labels
        """
        # Track and log process() calls
        SpeakerDiarizer._call_count += 1
        print(f"[SpeakerDiarizer] *** process() called #{SpeakerDiarizer._call_count} ***")
        
        if self.sd is None and self._pyannote_backend is None:
            self.initialize()
        
        # Delegate to Pyannote backend if using pyannote model
        if self._pyannote_backend is not None:
            return self._process_pyannote(audio_file, progress_callback, audio_data, audio_sample_rate)
        
        # Use pre-loaded audio if provided, otherwise load from file
        if audio_data is not None:
            print(f"[SpeakerDiarizer] Using pre-loaded audio data")
            audio = audio_data
            sample_rate = audio_sample_rate if audio_sample_rate is not None else self.sd.sample_rate
        else:
            print(f"[SpeakerDiarizer] Loading audio: {audio_file}")
            # Load audio using load_audio helper (supports m4a, mp3, etc.)
            target_sample_rate = self.sd.sample_rate
            audio, sample_rate = load_audio(audio_file, target_sample_rate)
        
        if sample_rate != self.sd.sample_rate:
            raise RuntimeError(
                f"Expected sample rate: {self.sd.sample_rate}, given: {sample_rate}"
            )
        
        print(f"[SpeakerDiarizer] Audio duration: {len(audio)/sample_rate:.2f}s")
        
        # Process with optional progress callback
        if progress_callback:
            result = self.sd.process(audio, callback=progress_callback).sort_by_start_time()
        else:
            result = self.sd.process(audio).sort_by_start_time()
        
        # Convert sherpa-onnx result to our Segment format
        segments = []
        for r in result:
            segment = Segment(
                start=r.start,
                end=r.end,
                speaker=r.speaker
            )
            segments.append(segment)
        
        # Post-processing: Clip segments to speech regions (VAD)
        segments = self._clip_segments_to_speech(segments, audio, sample_rate)

        # Post-processing: Merge short islands (< 1.5s) between same speaker
        if segments:
            segments = self._merge_short_islands(segments, max_short_duration=1.5)

        if segments:
            num_speakers = max(s.speaker for s in segments) + 1
            print(f"[SpeakerDiarizer] Found {len(segments)} segments from {num_speakers} speakers")
            # Print all segments to console for debugging
            print(f"\n[SpeakerDiarizer] ===== SPEAKER SEGMENTS (Post-Processed) =====")
            for i, seg in enumerate(segments, 1):
                print(f"[{i:3d}] {seg}")
            print(f"[SpeakerDiarizer] ===== END OF SEGMENTS =====\n")
        else:
            print("[SpeakerDiarizer] No speakers found!")

        return segments

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
            # Old format: list of dicts
            for seg_dict in result:
                segment = Segment(
                    start=seg_dict["start"],
                    end=seg_dict["end"],
                    speaker=seg_dict["speaker"]
                )
                segments.append(segment)

        # Post-processing: Clip segments to speech regions (VAD)
        if audio_data is not None:
            segments = self._clip_segments_to_speech(
                segments, audio_data, audio_sample_rate or 16000)
        else:
            try:
                vad_audio, vad_sr = load_audio(audio_file, 16000)
                segments = self._clip_segments_to_speech(segments, vad_audio, vad_sr)
            except Exception as e:
                print(f"[SpeakerDiarizer] Could not load audio for VAD clip: {e}")

        # Post-processing: Merge short islands (< 1.5s) between same speaker
        if segments:
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
            NEXT_SPEAKER_BIAS = 2.0  # Bias 2:1 về speaker sau khi từ rơi vào gap

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

            for w in raw_words:
                w_start = w.get("start", 0)
                w_spk_id = None

                # Dùng START time trên adjusted_segments (đã co margin)
                for spk_seg in adjusted_segments:
                    if spk_seg.start <= w_start <= spk_seg.end:
                        w_spk_id = spk_seg.speaker
                        break

                # Fallback: từ rơi vào gap (có thể do margin tạo ra)
                # Bias 2:1 về speaker SAU nếu 2 bên khác speaker
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
                            # Khác speaker → bias về speaker sau
                            w_spk_id = next_seg.speaker if dist_next <= dist_prev * NEXT_SPEAKER_BIAS else prev_seg.speaker
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
