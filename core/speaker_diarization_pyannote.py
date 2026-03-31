"""
Pyannote Community-1 Speaker Diarization Module (PyTorch)
High-accuracy speaker diarization using Pyannote Audio (PyTorch)

╔══════════════════════════════════════════════════════════════════════════════╗
║                            INSTALLATION GUIDE                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 1. Install Pyannote Audio:                                                   ║
║    pip install pyannote.audio                                                ║
║                                                                              ║
║ 2. Get HuggingFace Token:                                                    ║
║    - Visit: https://huggingface.co/settings/tokens                           ║
║    - Create a read token                                                     ║
║                                                                              ║
║ 3. Accept Licenses:                                                          ║
║    - https://huggingface.co/pyannote/speaker-diarization-community-1         ║
║    - https://huggingface.co/pyannote/segmentation-3.0                        ║
║    - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM          ║
║                                                                              ║
║ 4. Set Environment Variable:                                                 ║
║    set HF_TOKEN=your_token_here    (Windows)                                 ║
║    export HF_TOKEN=your_token_here (Linux/Mac)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
- SOTA accuracy (DER ~11-19% on benchmarks)
- RTF ~0.4-0.6 on modern CPU
- Handles overlapping speech
- Multilingual support

Performance (estimated on modern 8-core CPU):
- 1 hour audio → ~25-35 minutes processing
- DER: 11-19% (depends on audio quality)
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
from pathlib import Path
import time
import warnings

# Lazy imports for heavy dependencies
_pyannote_available = False

def _check_dependencies():
    """Check if required dependencies are available"""
    global _pyannote_available
    try:
        import pyannote.audio
        _pyannote_available = True
    except ImportError:
        pass
    return _pyannote_available

# Check on module load
_check_dependencies()


class Community1Diarizer:
    """
    High-accuracy speaker diarization using Pyannote Community-1 (PyTorch)
    
    This diarizer provides SOTA accuracy while maintaining reasonable CPU performance.
    RTF (Real-Time Factor): ~0.4-0.6 on modern multi-core CPU
    """
    
    def __init__(self, 
                 auth_token: Optional[str] = None,
                 num_speakers: int = -1,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 num_threads: int = 4,
                 min_duration_on: float = 0.3,
                 min_duration_off: float = 0.0,
                 threshold: float = 0.7):
        """
        Initialize Pyannote Community-1 diarizer (PyTorch)
        
        Uses VBxClustering (NOT AgglomerativeClustering from 3.1).
        Config reference: pyannote/speaker-diarization-community-1/config.yaml
        
        Args:
            auth_token: HuggingFace token (required for first download)
            num_speakers: Exact number of speakers (-1 for auto-detect)
            min_speakers: Minimum number of speakers (overrides num_speakers if set)
            max_speakers: Maximum number of speakers (overrides num_speakers if set)
            num_threads: Number of threads for PyTorch inference
            min_duration_on: Minimum duration for active speech (may not be supported by all models)
            min_duration_off: Minimum silence between segments (default: 0.0 per config.yaml)
            threshold: VBxClustering threshold (default: 0.7)
                      Higher = fewer speakers, Lower = more speakers
                      NOTE: This is VBx threshold, NOT AHC threshold (3.1 used 0.715)
        """
        self.auth_token = auth_token or os.getenv("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_threads = num_threads
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.threshold = threshold
        
        self.pipeline = None
        self._pipeline_type = "pyannote_pytorch"
        
        # Validate dependencies
        if not _pyannote_available:
            raise RuntimeError(
                "pyannote.audio not installed. "
                "Install with: pip install pyannote.audio"
            )
        
        # Check for auth token
        if not self.auth_token:
            warnings.warn(
                "HuggingFace token not provided. "
                "Please set HF_TOKEN environment variable or pass auth_token. "
                "Required to download pyannote models."
            )
    
    def initialize(self):
        """Initialize the Pyannote pipeline"""
        # Disable pyannote telemetry BEFORE importing pyannote.audio
        # (telemetry module is loaded at import time and tries OTLP network calls)
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
        from pyannote.audio import Pipeline
        import torch

        print("[Community1] Initializing Pyannote Community-1 pipeline (PyTorch)...")
        start_time = time.time()

        # Load from local model directory (no HuggingFace Hub needed)
        # __file__ = core/speaker_diarization_pyannote.py → go up 2 levels to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(base_dir, "models", "pyannote", "speaker-diarization-community-1")
        if not os.path.exists(os.path.join(local_model_path, "config.yaml")):
            raise RuntimeError(
                f"Local Community-1 model not found at: {local_model_path}\n"
                "Expected: models/pyannote/speaker-diarization-community-1/config.yaml"
            )
        print(f"[Community1] Loading local Community-1 model from: {local_model_path}")
        # pyannote 4.0: from_pretrained detects local directory via os.path.isdir()
        # and loads purely from local files (no HuggingFace Hub calls)
        self.pipeline = Pipeline.from_pretrained(local_model_path)
        
        # Configure for CPU inference
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("[Community1] Using CUDA")
        else:
            device = torch.device("cpu")
            # Set thread count for CPU
            torch.set_num_threads(self.num_threads)
            print(f"[Community1] Using CPU with {self.num_threads} threads")
        
        self.pipeline.to(device)
        
        # Configure hyperparameters for VBxClustering
        # Community-1 uses VBxClustering (NOT AgglomerativeClustering from 3.1)
        # Reference: models/pyannote/speaker-diarization-community-1/config.yaml
        #
        # Pipeline.from_pretrained() already instantiates with config.yaml defaults.
        # Only update changed parameters to preserve Fa, Fb optimized values from model.
        needs_update = False
        params = self.pipeline.parameters()
        
        # Debug: log available parameters
        seg_params = params.get("segmentation", {})
        clustering_params = params.get("clustering", {})
        print(f"[Community1] Available segmentation params: {list(seg_params.keys())}")
        print(f"[Community1] Available clustering params: {list(clustering_params.keys())}")
        
        # Update clustering threshold if different from default
        current_threshold = params.get("clustering", {}).get("threshold")
        if current_threshold is not None and self.threshold != current_threshold:
            params["clustering"]["threshold"] = self.threshold
            needs_update = True
        
        # Update segmentation parameters if they exist and different from defaults
        # min_duration_off (usually exists in config)
        if "min_duration_off" in seg_params and self.min_duration_off != seg_params["min_duration_off"]:
            params["segmentation"]["min_duration_off"] = self.min_duration_off
            needs_update = True
            
        # min_duration_on only if exists in model config
        if "min_duration_on" in seg_params and self.min_duration_on != seg_params["min_duration_on"]:
            params["segmentation"]["min_duration_on"] = self.min_duration_on
            needs_update = True
        
        # Re-instantiate pipeline only if parameters changed
        if needs_update:
            try:
                self.pipeline.instantiate(params)
                print(f"[Community1] Override params: threshold={self.threshold}, min_duration_off={self.min_duration_off}")
            except Exception as e:
                print(f"[Community1] Warning: Could not apply custom params: {e}")
                print(f"[Community1] Falling back to default params from config.yaml")
        else:
            print(f"[Community1] Using default params from config.yaml")
        
        elapsed = time.time() - start_time
        print(f"[Community1] Initialized in {elapsed:.2f}s")
    
    def process(self, 
                audio_file: str,
                progress_callback: Optional[Callable[[int, int], int]] = None,
                audio_data: Optional[np.ndarray] = None,
                audio_sample_rate: Optional[int] = None) -> List[Dict]:
        """
        Perform speaker diarization following Community-1 standard pipeline.
        
        Reference: https://huggingface.co/pyannote/speaker-diarization-community-1
        
        Args:
            audio_file: Path to audio file
            progress_callback: Optional progress callback(current, total) -> int
            audio_data: Pre-loaded audio numpy array (mono float32, optional)
            audio_sample_rate: Sample rate of pre-loaded audio
            
        Returns:
            List of segments with speaker labels
        """
        if self.pipeline is None:
            self.initialize()
        
        print(f"[Community1] Processing: {audio_file}")
        start_time = time.time()
        
        import torch
        
        # Prepare audio input for pipeline
        # Community-1 standard: pipeline({"waveform": tensor, "sample_rate": sr})
        # Reference: https://huggingface.co/pyannote/speaker-diarization-community-1#processing-from-memory
        audio_input = None
        
        # Option 1: Use pre-loaded audio data (avoid loading file twice)
        if audio_data is not None and audio_sample_rate is not None:
            print(f"[Community1] Using pre-loaded audio data: {len(audio_data)/audio_sample_rate:.2f}s @ {audio_sample_rate}Hz")
            # Ensure mono float32
            waveform = audio_data.astype(np.float32)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            # Convert to torch tensor: (1, time) - standard pyannote format
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            audio_input = {"waveform": waveform_tensor, "sample_rate": audio_sample_rate}
        
        # Option 2: Load from file using torchaudio (best for pyannote)
        if audio_input is None:
            try:
                import torchaudio
                waveform_tensor, sample_rate = torchaudio.load(audio_file, normalize=True)
                # Convert to mono if stereo (torchaudio returns [channels, time])
                if waveform_tensor.shape[0] > 1:
                    waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)
                audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}
                print(f"[Community1] Loaded audio with torchaudio: {waveform_tensor.shape[1]/sample_rate:.2f}s @ {sample_rate}Hz")
            except Exception as load_err:
                print(f"[Community1] Warning: Could not load audio with torchaudio: {load_err}")
                # Fallback: let pyannote handle file loading directly
                audio_input = audio_file
        
        
        # Run diarization with cancellation support via hook
        # Pyannote pipeline accepts a `hook=` parameter called at each processing step
        # (segmentation chunks, embeddings, clustering iterations).
        # If the hook raises an exception, the pipeline stops immediately.
        try:
            pipeline_kwargs = {}

            # Support num_speakers, min_speakers, max_speakers as per official API
            if self.num_speakers is not None and self.num_speakers > 0:
                pipeline_kwargs["num_speakers"] = self.num_speakers
            elif self.min_speakers is not None or self.max_speakers is not None:
                if self.min_speakers is not None:
                    pipeline_kwargs["min_speakers"] = self.min_speakers
                if self.max_speakers is not None:
                    pipeline_kwargs["max_speakers"] = self.max_speakers

            class _CancellableHook:
                """Pyannote hook with weighted step progress and cancellation support.
                Maps pyannote internal steps to weighted overall progress:
                  segmentation (0-15%), speaker_counting (15-25%),
                  embeddings (25-85%), discrete_diarization (85-100%).
                Handles both old-style (step, completed, total) and new-style
                (step, artefact, file=, completed=, total=) calling conventions."""

                STEP_WEIGHTS = {
                    "segmentation":         (0, 15),
                    "speaker_counting":     (15, 25),
                    "embeddings":           (25, 85),   # Chiếm phần lớn thời gian
                    "discrete_diarization": (85, 100),
                }

                def __init__(self, cb):
                    self.cb = cb

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def __call__(self, step_name, *args, **kwargs):
                    # Trích completed/total linh hoạt cho mọi pyannote version
                    completed = kwargs.get("completed")
                    total = kwargs.get("total")
                    # Fallback: old-style positional (step_name, completed, total)
                    if completed is None and len(args) >= 1 and isinstance(args[0], (int, float)):
                        completed = args[0]
                    if total is None and len(args) >= 2 and isinstance(args[1], (int, float)):
                        total = args[1]

                    if self.cb:
                        if completed is not None and total is not None and total > 0:
                            start_pct, end_pct = self.STEP_WEIGHTS.get(step_name, (0, 100))
                            step_progress = completed / total
                            overall = int(start_pct + (end_pct - start_pct) * step_progress)
                            ret = self.cb(overall, 100)
                        else:
                            # Step không có thông tin tiến độ - chỉ check cancel
                            start_pct = self.STEP_WEIGHTS.get(step_name, (0, 100))[0]
                            ret = self.cb(start_pct, 100)
                        if ret != 0:
                            print(f"[Community1] Cancellation at step '{step_name}'")
                            raise InterruptedError("Cancelled by user")

            hook = _CancellableHook(progress_callback) if progress_callback else None

            if hook:
                diarization = self.pipeline(audio_input, hook=hook, **pipeline_kwargs)
            else:
                diarization = self.pipeline(audio_input, **pipeline_kwargs)

        except InterruptedError:
            raise  # Propagate cancellation
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")
        
        # Extract annotation from output
        # Community-1 v4 returns DiarizeOutput with both:
        #   - output.speaker_diarization (with overlapping speech)
        #   - output.exclusive_speaker_diarization (no overlap, better for ASR)
        # Reference: https://huggingface.co/pyannote/speaker-diarization-community-1#exclusive-speaker-diarization
        annotation = None
        
        if hasattr(diarization, 'exclusive_speaker_diarization'):
            # Prefer exclusive_speaker_diarization for ASR reconciliation
            # Each time point has exactly 1 speaker → cleaner mapping to transcription
            annotation = diarization.exclusive_speaker_diarization
            print(f"[Community1] Using exclusive_speaker_diarization (optimized for ASR)")
        elif hasattr(diarization, 'speaker_diarization'):
            # Fallback to standard speaker_diarization
            annotation = diarization.speaker_diarization
            print(f"[Community1] Using speaker_diarization from DiarizeOutput")
        elif hasattr(diarization, 'itertracks'):
            # Direct Annotation object
            annotation = diarization
            print(f"[Community1] Processing Annotation with itertracks")
        else:
            print(f"[Community1] Warning: Unknown diarization output type: {type(diarization)}")
        
        # Extract segments from annotation with proper speaker ID mapping
        segments = []
        if annotation is not None and hasattr(annotation, 'itertracks'):
            speaker_map = {}  # Map speaker label to consistent integer ID
            speaker_counter = 0
            
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                # Assign consistent integer ID to each unique speaker label
                if speaker not in speaker_map:
                    speaker_map[speaker] = speaker_counter
                    speaker_counter += 1
                speaker_id = speaker_map[speaker]
                
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker_id
                }
                segments.append(segment)
        else:
            print(f"[Community1] Warning: Could not extract segments from output")
        
        elapsed = time.time() - start_time
        audio_duration = segments[-1]["end"] if segments else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        
        print(f"[Community1] Found {len(segments)} segments before post-processing")
        
        # Post-processing cho ASR: chỉ áp dụng cho exclusive_speaker_diarization
        # Giữ nguyên raw output, chỉ smooth cái dùng cho ASR
        if hasattr(diarization, 'exclusive_speaker_diarization') and annotation is diarization.exclusive_speaker_diarization:
            segments = self._post_process_for_asr(segments)
            print(f"[Community1] After post-processing: {len(segments)} segments")
        
        print(f"[Community1] Processing time: {elapsed:.2f}s, RTF: {rtf:.3f}")
        
        if progress_callback:
            progress_callback(100, 100)
        
        return segments
    
    def _post_process_for_asr(self, segments: List[Dict]) -> List[Dict]:
        """
        Post-processing nhẹ nhàng cho exclusive_speaker_diarization.
        Giữ nguyên raw output, chỉ smooth cái dùng cho ASR.
        
        Thứ tự:
        1. Merge adjacent + small gaps (0.3s)
        2. Smooth A-B-A với ngưỡng 0.5s  
        3. Reassign cực ngắn (<0.2s) thay vì xóa
        """
        if not segments:
            return segments
        
        original_count = len(segments)
        
        # 1. Merge đoạn liền kề + gap nhỏ (0.3s)
        segments = self._merge_with_gap(segments, max_gap=0.3)
        
        # 2. Smooth A-B-A với ngưỡng 0.5s
        segments = self._smooth_aba(segments, max_middle_duration=0.5)
        
        # 3. Reassign segment cực ngắn (<0.2s) vào neighbor
        segments = self._reassign_short_segments(segments, min_duration=0.2)
        
        # Final merge sau khi smooth
        segments = self._merge_with_gap(segments, max_gap=0.0)
        
        final_count = len(segments)
        if final_count < original_count:
            print(f"[Community1] Post-processing: {original_count} -> {final_count} segments")
        
        return segments
    
    def _merge_with_gap(self, segments: List[Dict], max_gap: float = 0.3) -> List[Dict]:
        """
        Gộp các đoạn liền kề hoặc có gap nhỏ cùng speaker.
        A(0-10) + [gap 0.2s] + A(10.2-20) -> A(0-20)
        """
        if not segments:
            return []
        
        # Sắp xếp theo thởi gian
        sorted_segs = sorted(segments, key=lambda x: (x['start'], x['speaker']))
        merged = [dict(sorted_segs[0])]
        
        for seg in sorted_segs[1:]:
            prev = merged[-1]
            gap = seg['start'] - prev['end']
            
            # Nếu cùng speaker và gap <= max_gap, gộp lại
            if seg['speaker'] == prev['speaker'] and gap <= max_gap:
                prev['end'] = max(prev['end'], seg['end'])
            else:
                merged.append(dict(seg))
        
        return merged
    
    def _smooth_aba(self, segments: List[Dict], max_middle_duration: float = 0.5) -> List[Dict]:
        """
        Loại bỏ chuyển đổi giả kiểu A-B-A.
        A(10s) + B(0.3s) + A(5s) -> A(15.3s)
        Chỉ áp dụng khi B rất ngắn (thường là nhận diện sai hoặc backchannel).
        """
        if len(segments) < 3:
            return segments
        
        smoothed = [dict(seg) for seg in segments]
        changes_made = True
        iterations = 0
        max_iterations = 3  # Tránh vòng lặp vô hạn
        
        while changes_made and iterations < max_iterations:
            changes_made = False
            iterations += 1
            new_smoothed = []
            i = 0
            
            while i < len(smoothed):
                if i == 0 or i >= len(smoothed) - 1:
                    new_smoothed.append(dict(smoothed[i]))
                    i += 1
                    continue
                
                prev_seg = smoothed[i - 1]
                cur_seg = smoothed[i]
                next_seg = smoothed[i + 1]
                
                cur_duration = cur_seg['end'] - cur_seg['start']
                
                # Nếu đoạn hiện tại ngắn và bị kẹp giữa 2 đoạn cùng speaker
                if (cur_duration <= max_middle_duration and 
                    prev_seg['speaker'] == next_seg['speaker'] and 
                    cur_seg['speaker'] != prev_seg['speaker']):
                    
                    # Gộp vào đoạn trước (speaker A)
                    merged_seg = {
                        'start': prev_seg['start'],
                        'end': next_seg['end'],
                        'speaker': prev_seg['speaker']
                    }
                    # Thay thế 3 đoạn bằng 1 đoạn gộp
                    if new_smoothed:
                        new_smoothed[-1] = merged_seg
                    else:
                        new_smoothed.append(merged_seg)
                    changes_made = True
                    i += 2  # Skip next_seg vì đã gộp
                else:
                    new_smoothed.append(dict(cur_seg))
                    i += 1
            
            smoothed = new_smoothed
        
        return smoothed
    
    def _reassign_short_segments(self, segments: List[Dict], min_duration: float = 0.2) -> List[Dict]:
        """
        Reassign segment cực ngắn vào speaker lân cận thay vì xóa.
        Ưu tiên: speaker trước > speaker sau > xóa (nếu không có lân cận).
        """
        if not segments:
            return segments
        
        result = []
        n = len(segments)
        
        for i, seg in enumerate(segments):
            duration = seg['end'] - seg['start']
            
            if duration >= min_duration:
                result.append(dict(seg))
                continue
            
            # Segment quá ngắn, tìm neighbor để reassign
            prev_speaker = result[-1]['speaker'] if result else None
            next_speaker = segments[i + 1]['speaker'] if i < n - 1 else None
            
            # Reassign vào speaker trước (thường là ngườI đang nói)
            if prev_speaker is not None:
                if result:
                    result[-1]['end'] = seg['end']  # Kéo dài đoạn trước
                else:
                    result.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'speaker': prev_speaker
                    })
            # Nếu không có trước, assign vào speaker sau
            elif next_speaker is not None:
                result.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': next_speaker
                })
            # Nếu không có cả 2, bỏ qua (cô lập hoàn toàn)
            else:
                result.append(dict(seg))  # Giữ lại nếu là segment duy nhất
        
        return result
    
    def process_with_transcription(self,
                                   audio_file: str,
                                   transcribed_segments: List[Dict],
                                   speaker_segments: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Merge speaker diarization with transcription segments
        
        Args:
            audio_file: Path to audio file
            transcribed_segments: List of transcription segments
            speaker_segments: Pre-computed speaker segments (optional)
            
        Returns:
            List of transcription segments with speaker labels
        """
        if speaker_segments is None:
            speaker_segments = self.process(audio_file)
        
        if not speaker_segments:
            print("[Community1] No speaker segments found")
            return transcribed_segments
        
        print(f"[Community1] Merging {len(speaker_segments)} speaker segments with {len(transcribed_segments)} transcription segments")
        
        results = []
        for trans_seg in transcribed_segments:
            trans_start = trans_seg.get("start", 0)
            trans_end = trans_seg.get("end", trans_start + 1)
            
            # Find overlapping speaker segments
            speaker_votes = {}
            for spk_seg in speaker_segments:
                overlap_start = max(trans_start, spk_seg["start"])
                overlap_end = min(trans_end, spk_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker_id = spk_seg["speaker"]
                    speaker_votes[speaker_id] = speaker_votes.get(speaker_id, 0) + overlap_duration
            
            # Assign best speaker
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
        
        return results
    
    def format_by_speaker(self, segments: List[Dict]) -> str:
        """Format segments grouped by speaker"""
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
        
        # Format output
        lines = []
        for group in grouped:
            lines.append(f"{group['speaker']}:")
            lines.append(group['text'])
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def unload(self):
        """Unload model to free memory"""
        import gc
        import torch
        
        if self.pipeline is not None:
            print("[Community1] Unloading model...")
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("[Community1] Model unloaded")


class Segment:
    """Compatibility class with existing speaker_diarization.py"""
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker
    
    @property
    def duration(self):
        return self.end - self.start
    
    def __str__(self):
        return f"{self.start:.3f}s --> {self.end:.3f}s speaker_{self.speaker:02d}"


def check_pyannote_available() -> bool:
    """Check if Pyannote Community-1 is available"""
    try:
        import pyannote.audio
        return True
    except ImportError:
        return False


def get_model_info() -> Dict:
    """Get information about Pyannote Community-1 model"""
    return {
        "id": "community1",
        "name": "Pyannote Community-1 (SOTA Accuracy)",
        "description": "State-of-the-art speaker diarization using PyTorch",
        "size": "~40MB",
        "language": "Multilingual",
        "speed": "Medium (RTF ~0.5)",
        "accuracy": "Excellent (DER ~11-19%)",
        "features": [
            "Overlapping speech detection",
            "High accuracy on all benchmarks",
            "PyTorch inference",
            "CPU-optimized"
        ],
        "requirements": [
            "pip install pyannote.audio",
            "HuggingFace token (accept license)",
            "~4GB RAM recommended"
        ]
    }


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Pyannote Community-1 diarization")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument("--speakers", type=int, default=-1, help="Number of speakers")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    
    args = parser.parse_args()
    
    # Check availability
    if not check_pyannote_available():
        print("Error: pyannote.audio not installed")
        print("Install with: pip install pyannote.audio")
        sys.exit(1)
    
    # Run diarization
    diarizer = Community1Diarizer(
        auth_token=args.token,
        num_speakers=args.speakers,
        num_threads=args.threads
    )
    
    segments = diarizer.process(args.audio)
    
    print("\n=== RESULTS ===")
    for seg in segments:
        print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: Speaker {seg['speaker']}")
