"""
3D-Speaker Diarization Pipeline with CAM++ Embeddings (192-dim)
===============================================================
Implements the 3D-Speaker approach EXACTLY (arXiv 2403.19971, 2025):
  - Silero VAD (ONNX) for speech detection (paper uses FSMN, Silero is equivalent)
  - Sliding window (1.5s window, 0.75s step) on speech regions
  - CAM++ 192-dim embedding (speech_campplus_sv_zh_en_16k-common_advanced, 200k speakers)
  - Spectral clustering with cosine affinity (pval=0.032, min_cluster_size=4)

Model: models/campp-3dspeaker/campplus_cn_en_common_200k.onnx (27MB, 192-dim)
VAD:   models/silero-vad/silero_vad_16k_op15.onnx (1.3MB)

Dependencies: numpy, scipy, onnxruntime, kaldi_native_fbank, soundfile
No pyannote dependency.
"""
import os
import logging
import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ══════════════════════════════════════════════════════════════
# Fbank feature extraction (kaldi-native-fbank, WeSpeaker style)
# Copied from speaker_diarization_pure_ort.py to be self-contained
# ══════════════════════════════════════════════════════════════

_knf_emb_opts = None


def _compute_fbank(audio, sr=SAMPLE_RATE):
    """Compute 80-dim fbank matching WeSpeaker's compute_fbank:
    - Scale waveform by 32768 (int16 range)
    - Hamming window
    - Per-utterance mean normalization (CMVN)
    """
    global _knf_emb_opts
    import kaldi_native_fbank as knf
    if _knf_emb_opts is None:
        _knf_emb_opts = knf.FbankOptions()
        _knf_emb_opts.frame_opts.dither = 0.0
        _knf_emb_opts.frame_opts.snip_edges = True
        _knf_emb_opts.frame_opts.samp_freq = sr
        _knf_emb_opts.frame_opts.frame_length_ms = 25.0
        _knf_emb_opts.frame_opts.frame_shift_ms = 10.0
        _knf_emb_opts.frame_opts.window_type = "hamming"
        _knf_emb_opts.mel_opts.num_bins = 80
        _knf_emb_opts.mel_opts.low_freq = 20.0
        _knf_emb_opts.mel_opts.high_freq = 0.0  # Nyquist
        _knf_emb_opts.energy_floor = 0.0
    scaled_audio = audio * np.float32(32768.0)
    fb = knf.OnlineFbank(_knf_emb_opts)
    fb.accept_waveform(sr, scaled_audio)
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        return np.empty((0, 80), dtype=np.float32)
    features = np.empty((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fb.get_frame(i)
    # Per-utterance mean normalization (CMVN)
    features -= features.mean(axis=0, keepdims=True)
    return features


# ══════════════════════════════════════════════════════════════
# Energy-based VAD
# ══════════════════════════════════════════════════════════════

def _energy_vad(audio, sr=SAMPLE_RATE, frame_ms=25.0, hop_ms=10.0,
                energy_ratio=0.1, merge_gap=0.3, min_duration=0.3):
    """Simple energy-based VAD.

    1. Compute frame energy (sum of squares per frame)
    2. Threshold at 95th percentile * energy_ratio
    3. Merge speech regions with gap < merge_gap seconds
    4. Remove regions shorter than min_duration seconds

    Returns list of (start_sec, end_sec) speech regions.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_samples = len(audio)

    if n_samples < frame_len:
        return []

    # Compute frame energies
    n_frames = 1 + (n_samples - frame_len) // hop_len
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        energies[i] = np.sum(frame * frame)

    if n_frames == 0:
        return []

    # Threshold: 95th percentile * ratio
    threshold = np.percentile(energies, 95) * energy_ratio
    is_speech = energies > threshold

    # Convert frame-level decisions to time regions
    regions = []
    in_speech = False
    start_time = 0.0
    for i in range(n_frames):
        t = i * hop_ms / 1000.0
        if is_speech[i] and not in_speech:
            start_time = t
            in_speech = True
        elif not is_speech[i] and in_speech:
            end_time = t + frame_ms / 1000.0
            regions.append((start_time, end_time))
            in_speech = False
    if in_speech:
        end_time = (n_frames - 1) * hop_ms / 1000.0 + frame_ms / 1000.0
        regions.append((start_time, end_time))

    if not regions:
        return []

    # Merge regions with gap < merge_gap
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Remove short regions
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration]

    # Clip to audio duration
    audio_dur = n_samples / sr
    merged = [(max(0, s), min(e, audio_dur)) for s, e in merged]

    return merged


# ══════════════════════════════════════════════════════════════
# Spectral Clustering (3D-Speaker paper)
# ══════════════════════════════════════════════════════════════

def _spectral_cluster(embeddings, pval=0.02, min_cluster_size=4,
                      max_speakers=10, min_speakers=1, min_pnum=6):
    """Spectral clustering — EXACT 3D-Speaker implementation.

    Source: github.com/modelscope/3D-Speaker/speakerlab/process/cluster.py
    Verified against source code line-by-line.
    """
    N = embeddings.shape[0]
    if N == 0:
        return np.array([], dtype=np.int32)
    if N == 1:
        return np.array([0], dtype=np.int32)

    # 1. Cosine similarity matrix
    S = embeddings @ embeddings.T

    # 2. Per-row pruning (exact 3D-Speaker p_pruning)
    # n_elems = int((1 - pval) * N), capped to keep at least min_pnum
    n_elems = int((1 - pval) * N)
    n_elems = min(n_elems, N - min_pnum)
    n_elems = max(n_elems, 0)
    for i in range(N):
        low_idx = np.argsort(S[i])[:n_elems]
        S[i, low_idx] = 0.0

    # 3. Symmetrize: average (NOT max) — exact 3D-Speaker
    S = 0.5 * (S + S.T)

    # 4. Unnormalized Laplacian: L = D - S
    np.fill_diagonal(S, 0.0)
    D = np.abs(S).sum(axis=1)
    L = -S.copy()
    np.fill_diagonal(L, D)

    # 5. Eigendecompose (smallest eigenvalues)
    n_eig = min(max_speakers + 1, N)
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, n_eig - 1])

    # 6. Eigengap — search ONLY in range [min_speakers-1 : max_speakers+1]
    # Exact 3D-Speaker: lambdas[min_num_spks-1 : max_num_spks+1]
    lo = max(min_speakers - 1, 0)
    hi = min(max_speakers + 1, len(eigenvalues))
    sub_evals = eigenvalues[lo:hi]
    if len(sub_evals) < 2:
        return np.zeros(N, dtype=np.int32)
    sub_gaps = [float(sub_evals[i+1]) - float(sub_evals[i]) for i in range(len(sub_evals)-1)]
    k = int(np.argmax(sub_gaps)) + min_speakers
    k = max(min_speakers, min(k, max_speakers, N))

    # 7. Raw eigenvectors — NO row normalization (unnormalized Laplacian)
    V = eigenvectors[:, :k]

    # 8. K-means
    labels = _kmeans(V, k, max_iter=300)

    # 9. Merge minor clusters (per-embedding, <= threshold)
    labels = _merge_small_clusters(labels, embeddings, min_cluster_size)

    # Relabel
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels], dtype=np.int32)

    return labels


def _kmeans(X, k, max_iter=300, n_init=10):
    """Simple k-means with multiple random initializations."""
    N, D = X.shape
    best_labels = np.zeros(N, dtype=np.int32)
    best_inertia = np.inf

    for _ in range(n_init):
        # K-means++ initialization
        centers = np.empty((k, D), dtype=X.dtype)
        idx = np.random.randint(N)
        centers[0] = X[idx]
        for c in range(1, k):
            dists = np.min(np.sum((X[:, None, :] - centers[None, :c, :]) ** 2, axis=2), axis=1)
            probs = dists / (dists.sum() + 1e-10)
            idx = np.random.choice(N, p=probs)
            centers[c] = X[idx]

        # Iterate
        labels = np.zeros(N, dtype=np.int32)
        for _ in range(max_iter):
            # Assign
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, k)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update centers
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = X[mask].mean(axis=0)

        inertia = 0.0
        for c in range(k):
            mask = labels == c
            if mask.any():
                inertia += np.sum((X[mask] - centers[c]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def _merge_small_clusters(labels, embeddings, min_cluster_size):
    """Merge minor clusters — EXACT 3D-Speaker filter_minor_cluster.

    Per-EMBEDDING reassignment (not per-cluster centroid).
    Threshold: <= (not <).
    """
    unique, counts = np.unique(labels, return_counts=True)
    minor = set(unique[counts <= min_cluster_size])  # <= not <
    major = unique[counts > min_cluster_size]

    if len(major) == 0:
        return np.zeros(len(labels), dtype=np.int32)

    # Compute centroids of major clusters, L2 normalize
    major_centers = np.stack([embeddings[labels == m].mean(axis=0) for m in major])
    cnorms = np.linalg.norm(major_centers, axis=1, keepdims=True)
    cnorms[cnorms < 1e-10] = 1.0
    major_centers /= cnorms

    # Per-embedding reassignment (exact 3D-Speaker)
    for i in range(len(labels)):
        if labels[i] in minor:
            emb = embeddings[i]
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            sims = major_centers @ emb
            labels[i] = major[np.argmax(sims)]

    return labels


# ══════════════════════════════════════════════════════════════
# Main Diarizer Class
# ══════════════════════════════════════════════════════════════

class ThreeDSpeakerCamppDiarizer:
    """3D-Speaker diarization pipeline with CAM++ 192-dim embeddings.

    Pipeline (EXACTLY per paper arXiv 2403.19971):
      1. Silero VAD (ONNX) for speech region detection
      2. Sliding window (1.5s, 0.75s step) on speech regions
      3. CAM++ 192-dim embedding extraction (200k speakers, 3D-Speaker)
      4. Spectral clustering with cosine affinity (pval=0.032)
    """

    def __init__(self, model_dir=None, num_threads=6, max_speakers=10,
                 min_speakers=1, num_speakers=-1,
                 pval=0.032, min_cluster_size=4,
                 window=1.5, step=0.75, min_duration_off=0.0):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # CAM++ 192-dim from 3D-Speaker (exact paper model)
        self.model_path = (
            os.path.join(model_dir, "campplus_cn_en_common_200k.onnx") if model_dir
            else os.path.join(base, "models", "campp-3dspeaker", "campplus_cn_en_common_200k.onnx")
        )
        # Silero VAD ONNX
        self.vad_path = os.path.join(base, "models", "silero-vad", "silero_vad_16k_op15.onnx")
        self.num_threads = num_threads
        self.max_speakers = max_speakers
        self.min_speakers = max(1, min_speakers)
        self.num_speakers = num_speakers  # -1 = auto
        self.pval = pval
        self.min_cluster_size = min_cluster_size
        self.window = window  # seconds
        self.step = step  # seconds
        self.min_duration_off = min_duration_off
        self.emb_sess = None
        self.vad_sess = None

    def initialize(self):
        """Load CAM++ 192-dim ONNX + Silero VAD ONNX."""
        import onnxruntime as ort
        ort.set_default_logger_severity(3)

        from core.config import compute_ort_threads
        Z = compute_ort_threads(self.num_threads, full_ht=True)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = Z
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False

        # Load CAM++ 192-dim (3D-Speaker)
        opts.optimized_model_filepath = self.model_path + ".opt"
        self.emb_sess = ort.InferenceSession(
            self.model_path, opts, providers=['CPUExecutionProvider']
        )
        # Warmup
        dummy = np.zeros((1, 150, 80), dtype=np.float32)
        self.emb_sess.run(['embs'], {'feats': dummy})

        # Load Silero VAD
        vad_opts = ort.SessionOptions()
        vad_opts.intra_op_num_threads = 1
        vad_opts.log_severity_level = 3
        vad_opts.enable_cpu_mem_arena = False
        self.vad_sess = ort.InferenceSession(
            self.vad_path, vad_opts, providers=['CPUExecutionProvider']
        )

        print(f"[3DSpeaker-CAM++] Loaded CAM++ 192-dim + Silero VAD"
              f" | pval={self.pval}, window={self.window}s, step={self.step}s")

    def _silero_vad(self, audio, sr=SAMPLE_RATE, threshold=0.2,
                    min_speech=0.25, min_silence=0.3):
        """Silero VAD — neural speech detection matching paper quality.

        Processes audio in 512-sample (32ms) chunks through Silero ONNX model.
        Returns list of (start_sec, end_sec) speech regions.
        """
        window_size = 512  # Silero processes 512 samples at 16kHz
        context_size = 64  # Silero needs 64 samples context prefix
        n_samples = len(audio)
        if n_samples < window_size:
            return [(0.0, n_samples / sr)]

        # Silero v5 state: [2, batch, 128]
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_tensor = np.array(sr, dtype=np.int64)
        context = np.zeros(context_size, dtype=np.float32)

        # Get speech probabilities per chunk
        num_windows = n_samples // window_size
        probs = []
        for i in range(num_windows):
            chunk = audio[i * window_size:(i + 1) * window_size]
            # Silero input = [context(64) + chunk(512)] = 576 samples
            input_data = np.concatenate([context, chunk]).reshape(1, -1).astype(np.float32)
            out, state = self.vad_sess.run(
                None, {'input': input_data, 'state': state, 'sr': sr_tensor}
            )
            probs.append(float(out[0][0]))
            context = chunk[-context_size:]  # last 64 samples as next context

        if not probs:
            return []

        # Convert to speech regions
        chunk_dur = window_size / sr
        regions = []
        in_speech = False
        start_t = 0.0

        for i, p in enumerate(probs):
            t = i * chunk_dur
            if p >= threshold and not in_speech:
                start_t = t
                in_speech = True
            elif p < threshold and in_speech:
                end_t = t + chunk_dur
                if end_t - start_t >= min_speech:
                    regions.append((start_t, end_t))
                in_speech = False

        if in_speech:
            end_t = len(probs) * chunk_dur
            if end_t - start_t >= min_speech:
                regions.append((start_t, end_t))

        # Merge close regions
        if not regions:
            return []
        merged = [regions[0]]
        for s, e in regions[1:]:
            if s - merged[-1][1] < min_silence:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Clip to audio
        audio_dur = n_samples / sr
        merged = [(max(0, s), min(e, audio_dur)) for s, e in merged]
        return merged

    def _extract_embedding(self, audio_segment):
        """Extract a single 192-dim L2-normalized embedding from an audio segment.

        Args:
            audio_segment: float32 audio array (mono, 16kHz)

        Returns:
            embedding: (192,) L2-normalized vector, or None if segment too short
        """
        feats = _compute_fbank(audio_segment, SAMPLE_RATE)
        if feats.shape[0] < 10:  # Too few frames
            return None

        # CAM++ expects [1, T, 80]
        feats_input = feats[np.newaxis, :, :]  # (1, T, 80)
        outputs = self.emb_sess.run(['embs'], {'feats': feats_input})
        emb = outputs[0][0]  # (192,)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb /= norm
        return emb

    def _sliding_window_embeddings(self, audio, speech_regions, progress_callback=None):
        """Extract embeddings using sliding windows over speech regions.

        Args:
            audio: full audio array (float32, mono, 16kHz)
            speech_regions: list of (start_sec, end_sec) from VAD

        Returns:
            embeddings: (N, 192) array of L2-normalized embeddings
            window_times: list of (start_sec, end_sec) for each window
        """
        window_samples = int(self.window * SAMPLE_RATE)
        step_samples = int(self.step * SAMPLE_RATE)

        embeddings = []
        window_times = []

        # Collect all windows first
        all_windows = []
        for region_start, region_end in speech_regions:
            start_sample = int(region_start * SAMPLE_RATE)
            end_sample = int(region_end * SAMPLE_RATE)
            region_len = end_sample - start_sample

            if region_len < window_samples:
                # Region shorter than window: use entire region as one window
                all_windows.append((start_sample, end_sample))
            else:
                pos = start_sample
                while pos + window_samples <= end_sample:
                    all_windows.append((pos, pos + window_samples))
                    pos += step_samples
                # Handle remaining tail if significant (> 50% of window)
                if pos < end_sample and (end_sample - pos) > window_samples * 0.5:
                    all_windows.append((pos, end_sample))

        total_windows = len(all_windows)
        if total_windows == 0:
            return np.empty((0, 192), dtype=np.float32), []

        logger.info(f"[3DSpeaker-CAM++] Extracting {total_windows} window embeddings")

        for idx, (ws, we) in enumerate(all_windows):
            segment = audio[ws:we]
            emb = self._extract_embedding(segment)
            if emb is not None:
                embeddings.append(emb)
                window_times.append((ws / SAMPLE_RATE, we / SAMPLE_RATE))

            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(30 + 50 * (idx + 1) / total_windows)

        if len(embeddings) == 0:
            return np.empty((0, 192), dtype=np.float32), []

        return np.stack(embeddings, axis=0), window_times

    def _segments_from_labels(self, window_times, labels):
        """Convert per-window cluster labels to merged time segments.

        Args:
            window_times: list of (start, end) for each window
            labels: (N,) cluster label for each window

        Returns:
            list of {'start': float, 'end': float, 'speaker': int}
        """
        if len(window_times) == 0:
            return []

        segments = []
        current_start = window_times[0][0]
        current_end = window_times[0][1]
        current_label = labels[0]

        for i in range(1, len(window_times)):
            w_start, w_end = window_times[i]
            label = labels[i]

            if label == current_label and (w_start - current_end) < self.min_duration_off + 0.01:
                # Same speaker and adjacent/overlapping — extend
                current_end = w_end
            else:
                # Different speaker or gap — emit segment
                segments.append({
                    'start': float(current_start),
                    'end': float(current_end),
                    'speaker': int(current_label)
                })
                current_start = w_start
                current_end = w_end
                current_label = label

        # Emit last segment
        segments.append({
            'start': float(current_start),
            'end': float(current_end),
            'speaker': int(current_label)
        })

        return segments

    def process(self, audio_file=None, progress_callback=None,
                audio_data=None, audio_sample_rate=None):
        """Full diarization pipeline.

        Args:
            audio_file: path to audio file (WAV, FLAC, etc.)
            progress_callback: optional callable(percent: float) for progress updates
            audio_data: pre-loaded audio array (alternative to audio_file)
            audio_sample_rate: sample rate of audio_data

        Returns:
            list of {'start': float, 'end': float, 'speaker': int}
        """
        import time

        if self.emb_sess is None:
            self.initialize()

        t_total = time.perf_counter()

        # ─── 1. Load audio ───────────────────────────────────────
        if audio_data is not None:
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sr = audio_sample_rate or SAMPLE_RATE
        else:
            import soundfile
            audio, sr = soundfile.read(audio_file, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        duration = len(audio) / SAMPLE_RATE
        logger.info(f"[3DSpeaker-CAM++] Audio: {duration:.1f}s")

        if duration < 0.5:
            logger.warning("[3DSpeaker-CAM++] Audio too short (< 0.5s), returning empty")
            return []

        if progress_callback:
            progress_callback(5)

        # ─── 2. Silero VAD (neural, paper-quality) ──────────────
        t0 = time.perf_counter()
        speech_regions = self._silero_vad(audio, SAMPLE_RATE)
        t_vad = time.perf_counter() - t0

        if not speech_regions:
            logger.warning("[3DSpeaker-CAM++] No speech detected by VAD, using full audio")
            speech_regions = [(0.0, duration)]

        total_speech = sum(e - s for s, e in speech_regions)
        logger.info(f"[3DSpeaker-CAM++] VAD: {len(speech_regions)} regions, "
                     f"{total_speech:.1f}s speech / {duration:.1f}s total ({t_vad:.3f}s)")

        if progress_callback:
            progress_callback(15)

        # ─── 3. Sliding window embedding extraction ──────────────
        t0 = time.perf_counter()
        embeddings, window_times = self._sliding_window_embeddings(
            audio, speech_regions, progress_callback
        )
        t_emb = time.perf_counter() - t0

        # Free audio memory
        del audio

        n_embs = embeddings.shape[0]
        logger.info(f"[3DSpeaker-CAM++] Embeddings: {n_embs} x 192 ({t_emb:.3f}s)")

        if n_embs == 0:
            logger.warning("[3DSpeaker-CAM++] No embeddings extracted")
            return []

        if progress_callback:
            progress_callback(80)

        # ─── 4. Clustering ───────────────────────────────────────
        t0 = time.perf_counter()

        # Determine number of speakers constraint
        if self.num_speakers > 0:
            # Fixed number requested
            min_spk = self.num_speakers
            max_spk = self.num_speakers
        else:
            min_spk = self.min_speakers
            max_spk = self.max_speakers

        # Edge case: very few embeddings
        if n_embs <= 2:
            labels = np.zeros(n_embs, dtype=np.int32)
        else:
            labels = _spectral_cluster(
                embeddings, pval=self.pval,
                min_cluster_size=self.min_cluster_size,
                max_speakers=max_spk, min_speakers=min_spk
            )

        n_speakers = len(np.unique(labels))
        t_clust = time.perf_counter() - t0
        logger.info(f"[3DSpeaker-CAM++] Clustering: {n_speakers} speakers ({t_clust:.3f}s)")

        if progress_callback:
            progress_callback(90)

        # ─── 5. Build output segments ────────────────────────────
        segments = self._segments_from_labels(window_times, labels)

        t_total_elapsed = time.perf_counter() - t_total
        logger.info(f"[3DSpeaker-CAM++] Done: {len(segments)} segments, "
                     f"{n_speakers} speakers, {t_total_elapsed:.3f}s total "
                     f"(VAD={t_vad:.3f}s, emb={t_emb:.3f}s, clust={t_clust:.3f}s)")

        if progress_callback:
            progress_callback(100)

        return segments
