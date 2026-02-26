"""
Người nói Diarization module using sherpa-onnx
Based on: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html

Available speaker embedding models:
- nemo_en_titanet_small.onnx (38.4MB) - Fast, good accuracy, English
- 3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx (68.1MB) - Chinese + English

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
    "eres2netv2_zh": {
        "name": "3D Speaker ERes2NetV2 (ZH+EN)",
        "file": "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        "size": "68.1 MB",
        "language": "Chinese + English",
        "speed": "Medium",
        "accuracy": "High",
        "sample_rate": 16000,
        "description": "Chinese and English multilingual"
    }
}


def get_available_models(base_dir: str = None) -> Dict[str, str]:
    """
    Get list of available (downloaded) speaker embedding models
    
    Returns:
        Dict mapping model_id to full path
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    models_dir = os.path.join(base_dir, "models", "speaker_embedding")
    available = {}
    
    for model_id, info in SPEAKER_EMBEDDING_MODELS.items():
        model_path = os.path.join(models_dir, info["file"])
        if os.path.exists(model_path):
            available[model_id] = model_path
    
    return available


def get_model_info(model_id: str) -> Optional[Dict]:
    """Get information about a speaker embedding model"""
    return SPEAKER_EMBEDDING_MODELS.get(model_id)


def get_model_path(model_id: str, base_dir: str = None) -> Optional[str]:
    """Get full path to a model file"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    """Load audio file and resample if needed
    
    Supports: wav, mp3, m4a, flac, ogg, wma, aac, opus
    
    Args:
        filename: Path to audio file
        target_sample_rate: Expected sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import os
    file_ext = os.path.splitext(filename)[1].lower()
    
    # For compressed formats (m4a, mp3, wma, etc.), use pydub to convert to wav first
    compressed_formats = ['.m4a', '.mp3', '.wma', '.aac', '.opus']
    
    # Setup ffmpeg path for pydub
    _setup_ffmpeg_path()
    
    if file_ext in compressed_formats and PYDUB_AVAILABLE:
        try:
            print(f"[SpeakerDiarizer] Converting {file_ext} to wav using pydub...")
            
            # Map extension to format
            format_map = {'.m4a': 'm4a', '.mp3': 'mp3', '.wma': 'wma', '.aac': 'aac', '.opus': 'opus'}
            audio_format = format_map.get(file_ext, file_ext[1:])
            
            # Load and convert
            audio_segment = AudioSegment.from_file(filename, format=audio_format)
            audio_segment = audio_segment.set_frame_rate(target_sample_rate).set_channels(1)
            
            # Get audio data as numpy array
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range (pydub uses int16)
            audio = audio / 32768.0
            
            print(f"[SpeakerDiarizer] Converted successfully: {len(audio)} samples at {target_sample_rate}Hz")
            return audio, target_sample_rate
            
        except Exception as e:
            print(f"[SpeakerDiarizer] Pydub conversion failed: {e}, falling back to librosa...")
    
    # Try soundfile first (faster for wav/flac/ogg)
    try:
        audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # only use the first channel
        
        if sample_rate != target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
            )
            audio = audio.astype(np.float32)
            sample_rate = target_sample_rate
        return audio, sample_rate
    except Exception as sf_error:
        # Fall back to librosa for other formats
        if LIBROSA_AVAILABLE:
            print(f"[SpeakerDiarizer] Soundfile failed, using librosa for {filename}")
            audio, sample_rate = librosa.load(filename, sr=target_sample_rate, mono=True)
            audio = audio.astype(np.float32)
            return audio, target_sample_rate
        else:
            raise sf_error


class SpeakerDiarizer:
    """Người nói diarization using sherpa-onnx OfflineSpeakerDiarization API"""
    
    
    _call_count = 0  # Track process() calls
    
    def __init__(self, 
                 segmentation_model: str = None,
                 embedding_model: str = None,
                 embedding_model_id: str = None,
                 num_clusters: int = -1,
                 num_threads: int = 6,
                 threshold: float = 0.7,
                 min_duration_on: float = 0.8,
                 min_duration_off: float = 1.0):
        """
        Initialize speaker diarizer
        
        Args:
            segmentation_model: Path to segmentation model (.onnx)
            embedding_model: Path to speaker embedding model (.onnx) - can use embedding_model_id instead
            embedding_model_id: Model ID from SPEAKER_EMBEDDING_MODELS registry
            num_clusters: Number of speakers (-1 for auto-detect using threshold)
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
        self.num_threads = num_threads
        self.threshold = threshold if threshold is not None else 0.8
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.sd = None  # sherpa_onnx OfflineSpeakerDiarization instance
        self.model_info = None  # Store model info for reference
        
    def initialize(self):
        """Initialize models using sherpa-onnx OfflineSpeakerDiarizationConfig"""
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
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
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
                num_clusters=self.num_clusters,  # Number of speakers (-1 for auto-detect)
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
        
        if self.sd is None:
            self.initialize()
        
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
        
        if segments:
            num_speakers = max(s.speaker for s in segments) + 1
            print(f"[SpeakerDiarizer] Found {len(segments)} segments from {num_speakers} speakers")
            # Print all segments to console for debugging
            print(f"\n[SpeakerDiarizer] ===== SPEAKER SEGMENTS =====")
            for i, seg in enumerate(segments, 1):
                print(f"[{i:3d}] {seg}")
            print(f"[SpeakerDiarizer] ===== END OF SEGMENTS =====\n")
        else:
            print("[SpeakerDiarizer] No speakers found!")
        
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
            
            # If there's no clear overlap, or no raw_words, or only one speaker vote, fallback to segment-level
            if not raw_words or len(speaker_votes) <= 1:
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
                if "raw_words" in seg_copy:
                    del seg_copy["raw_words"]  # Clean up memory
                seg_copy.update({
                    "speaker": best_speaker,
                    "speaker_id": best_speaker_id
                })
                results.append(seg_copy)
                continue

            # Word-level speaker assignment
            word_groups = []
            current_speaker_id = None
            current_group = []

            for w in raw_words:
                w_start = w.get("start", 0)
                w_end = w.get("end", 0)
                
                # Find best speaker for this word
                w_votes = {}
                for spk_seg in speaker_segments:
                    o_start = max(w_start, spk_seg.start)
                    o_end = min(w_end, spk_seg.end)
                    o_dur = max(0, o_end - o_start)
                    if o_dur > 0:
                        w_votes[spk_seg.speaker] = w_votes.get(spk_seg.speaker, 0) + o_dur
                
                if w_votes:
                    w_spk_id = max(w_votes, key=w_votes.get)
                else:
                    # Fallback to previous word's speaker or segment's best speaker
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
                if "raw_words" in seg_copy:
                    del seg_copy["raw_words"]
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
                if "raw_words" in seg_copy:
                    del seg_copy["raw_words"]
                seg_copy.update({
                    "text": g_text,
                    "start": g_start,
                    "end": g_end,
                    "speaker": f"Người nói {spk_id + 1}",
                    "speaker_id": spk_id
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
    
    # Find test file
    if audio_file is None:
        test_files = [
            r"D:\App\asr-vn\test_20min.wav",
            r"D:\App\asr-vn\test_audio.wav",
            "./test_20min.wav",
            "./test_audio.wav",
        ]
        
        for f in test_files:
            if os.path.exists(f):
                audio_file = f
                break
    
    if not audio_file:
        print("Test file not found!")
        return None
    
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
