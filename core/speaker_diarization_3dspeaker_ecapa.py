"""
SpeechBrain-style Speaker Diarization with ECAPA-TDNN Embeddings
================================================================
Follows the SpeechBrain AMI diarization recipe (Dawalatabad et al., Interspeech 2021).
Uses spectral clustering with cosine affinity instead of VBx/PLDA.

Pipeline:
  1. Simple energy-based VAD to detect speech regions
  2. Sliding window (1.5s window, 0.75s step) on speech regions
  3. ECAPA-TDNN ONNX embedding extraction (192-dim, one per window)
  4. Spectral clustering with cosine affinity (SpeechBrain parameters)

Dependencies: numpy, scipy, onnxruntime, kaldi_native_fbank, soundfile
"""
import os
import logging
import time
import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
EMB_DIM = 192

# ══════════════════════════════════════════════════════════════
# Fbank (kaldi-native-fbank) — same approach as pure_ort
# ══════════════════════════════════════════════════════════════

_knf_emb_opts = None


def _compute_fbank(audio, sr=SAMPLE_RATE):
    """Compute 80-dim fbank matching WeSpeaker's compute_fbank:
    - Scale waveform by 32768 (int16 range)
    - Hamming window
    - Per-utterance mean normalization (CMVN)
    Source: wespeaker/bin/infer_onnx.py
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
        _knf_emb_opts.mel_opts.high_freq = 0.0  # 0 = Nyquist
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

def _energy_vad(audio, sr=SAMPLE_RATE, frame_ms=25.0, shift_ms=10.0,
                threshold_ratio=0.1, percentile=95,
                min_gap=0.3, min_duration=0.3):
    """Simple energy-based VAD.

    1. Compute frame energy (squared amplitude sum per frame)
    2. Threshold at percentile(energy, 95th) * threshold_ratio
    3. Merge speech regions with gap < min_gap seconds
    4. Remove regions shorter than min_duration seconds

    Returns list of (start_sec, end_sec) tuples.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    frame_shift = int(sr * shift_ms / 1000.0)
    n_samples = len(audio)

    if n_samples < frame_len:
        return []

    # Compute frame energies
    n_frames = 1 + (n_samples - frame_len) // frame_shift
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_shift
        frame = audio[start:start + frame_len].astype(np.float64)
        energies[i] = np.sum(frame * frame)

    if n_frames == 0:
        return []

    # Threshold: 95th percentile * 0.1
    thr = np.percentile(energies, percentile) * threshold_ratio
    is_speech = energies > thr

    # Convert frame indices to time and collect speech regions
    regions = []
    in_speech = False
    speech_start = 0.0
    for i in range(n_frames):
        t = i * shift_ms / 1000.0
        if is_speech[i] and not in_speech:
            speech_start = t
            in_speech = True
        elif not is_speech[i] and in_speech:
            speech_end = t + frame_ms / 1000.0
            regions.append((speech_start, speech_end))
            in_speech = False
    if in_speech:
        speech_end = (n_frames - 1) * shift_ms / 1000.0 + frame_ms / 1000.0
        regions.append((speech_start, speech_end))

    if not regions:
        return []

    # Merge regions with gap < min_gap
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Remove regions shorter than min_duration
    merged = [(s, e) for s, e in merged if (e - s) >= min_duration]

    return merged


# ══════════════════════════════════════════════════════════════
# Spectral Clustering (SpeechBrain parameters)
# ══════════════════════════════════════════════════════════════

def _spectral_cluster(embeddings, pval=0.05, min_cluster_size=4,
                      max_speakers=10, min_speakers=1, min_pnum=6):
    """Spectral clustering — EXACT 3D-Speaker implementation.

    Source: github.com/modelscope/3D-Speaker/speakerlab/process/cluster.py
    Verified against source code line-by-line.
    """
    N = embeddings.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=np.int32)

    # 1. Cosine similarity matrix
    S = embeddings @ embeddings.T

    # 2. Per-row pruning (exact 3D-Speaker p_pruning)
    n_elems = int((1 - pval) * N)
    n_elems = min(n_elems, N - min_pnum)
    n_elems = max(n_elems, 0)
    for i in range(N):
        low_idx = np.argsort(S[i])[:n_elems]
        S[i, low_idx] = 0.0

    # 3. Symmetrize: average (exact 3D-Speaker)
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
    lo = max(min_speakers - 1, 0)
    hi = min(max_speakers + 1, len(eigenvalues))
    sub_evals = eigenvalues[lo:hi]
    if len(sub_evals) < 2:
        return np.zeros(N, dtype=np.int32)
    sub_gaps = [float(sub_evals[i+1]) - float(sub_evals[i]) for i in range(len(sub_evals)-1)]
    k = int(np.argmax(sub_gaps)) + min_speakers
    k = max(min_speakers, min(k, max_speakers, N))

    if k <= 0:
        return np.zeros(N, dtype=np.int32)

    # 7. Raw eigenvectors — NO row normalization (unnormalized Laplacian)
    V = eigenvectors[:, :k]

    # 8. K-means
    labels = _kmeans(V, k, max_iters=300, n_init=10)

    # 10. Post-processing: merge clusters smaller than min_cluster_size into nearest
    labels = _merge_small_clusters(embeddings, labels, min_cluster_size)

    # Relabel sequentially from 0
    labels = _relabel_sequential(labels)

    return labels


def _kmeans(X, k, max_iters=300, n_init=10):
    """K-means clustering with multiple random initializations.
    Returns best labels (lowest inertia).
    """
    N, D = X.shape
    best_labels = np.zeros(N, dtype=np.int32)
    best_inertia = np.inf

    rng = np.random.RandomState(42)

    for _ in range(n_init):
        # K-means++ initialization
        centers = np.empty((k, D), dtype=np.float64)
        idx = rng.randint(N)
        centers[0] = X[idx]

        for c in range(1, k):
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :c, :]) ** 2,
                       axis=2),
                axis=1
            )
            probs = dists / (dists.sum() + 1e-30)
            idx = rng.choice(N, p=probs)
            centers[c] = X[idx]

        # Iterate
        labels = np.zeros(N, dtype=np.int32)
        for _ in range(max_iters):
            # Assign
            dists = np.sum(
                (X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
                axis=2
            )  # (N, k)
            new_labels = np.argmin(dists, axis=1)

            # Update centers
            changed = False
            for c in range(k):
                mask = new_labels == c
                if mask.any():
                    new_center = X[mask].mean(axis=0)
                    if not np.allclose(centers[c], new_center, atol=1e-8):
                        changed = True
                    centers[c] = new_center

            labels = new_labels
            if not changed:
                break

        # Compute inertia
        inertia = 0.0
        for c in range(k):
            mask = labels == c
            if mask.any():
                inertia += np.sum((X[mask] - centers[c]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def _merge_small_clusters(embeddings, labels, min_cluster_size):
    """Merge minor clusters — EXACT 3D-Speaker filter_minor_cluster.

    Per-EMBEDDING reassignment (not per-cluster centroid).
    Threshold: <= (not <).
    """
    unique, counts = np.unique(labels, return_counts=True)
    minor = set(unique[counts <= min_cluster_size])
    major = unique[counts > min_cluster_size]

    if len(major) == 0:
        return np.zeros(len(labels), dtype=np.int32)

    major_centers = np.stack([embeddings[labels == m].mean(axis=0) for m in major])
    cnorms = np.linalg.norm(major_centers, axis=1, keepdims=True)
    cnorms[cnorms < 1e-10] = 1.0
    major_centers /= cnorms

    for i in range(len(labels)):
        if labels[i] in minor:
            emb = embeddings[i]
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            sims = major_centers @ emb
            labels[i] = major[np.argmax(sims)]

    return labels


def _relabel_sequential(labels):
    """Relabel clusters sequentially starting from 0."""
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[lbl] for lbl in labels], dtype=np.int32)


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

class ThreeDSpeakerEcapaDiarizer:
    """SpeechBrain-style diarization: Silero VAD + sliding window embeddings
    + spectral clustering with ECAPA-TDNN (192-dim) embeddings.
    Paper: Dawalatabad et al., Interspeech 2021.
    """

    def __init__(self, model_dir=None, num_threads=6, max_speakers=10,
                 min_speakers=1, num_speakers=-1,
                 pval=0.05, min_cluster_size=4,
                 window=1.5, step=0.75, min_duration_off=0.0):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(base, "models", "ecapa-wespeaker")
        self.vad_path = os.path.join(base, "models", "silero-vad", "silero_vad_16k_op15.onnx")
        self.num_threads = num_threads
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.num_speakers = num_speakers  # -1 = auto
        self.pval = pval
        self.min_cluster_size = min_cluster_size
        self.window = window
        self.step = step
        self.min_duration_off = min_duration_off
        self.emb_sess = None
        self.vad_sess = None

    def initialize(self):
        """Load ECAPA-TDNN ONNX + Silero VAD ONNX."""
        import onnxruntime as ort
        ort.set_default_logger_severity(3)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False

        ecapa_path = os.path.join(self.model_dir, "voxceleb_ECAPA512_LM.onnx")
        if not os.path.isfile(ecapa_path):
            raise FileNotFoundError(f"ECAPA-TDNN not found: {ecapa_path}")
        opts.optimized_model_filepath = ecapa_path + ".opt"

        self.emb_sess = ort.InferenceSession(
            ecapa_path, opts, providers=['CPUExecutionProvider'])

        # Load Silero VAD
        vad_opts = ort.SessionOptions()
        vad_opts.intra_op_num_threads = 1
        vad_opts.log_severity_level = 3
        vad_opts.enable_cpu_mem_arena = False
        self.vad_sess = ort.InferenceSession(
            self.vad_path, vad_opts, providers=['CPUExecutionProvider'])

        print(f"[3DSpeaker-ECAPA] Loaded ECAPA-TDNN + Silero VAD"
              f" | pval={self.pval}, window={self.window}s, step={self.step}s")

    def _silero_vad(self, audio, sr=SAMPLE_RATE, threshold=0.2,
                    min_speech=0.25, min_silence=0.3):
        """Silero VAD — neural speech detection."""
        window_size = 512
        context_size = 64  # Silero needs 64 samples context prefix
        n_samples = len(audio)
        if n_samples < window_size:
            return [(0.0, n_samples / sr)]

        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_tensor = np.array(sr, dtype=np.int64)
        context = np.zeros(context_size, dtype=np.float32)

        num_windows = n_samples // window_size
        probs = []
        for i in range(num_windows):
            chunk = audio[i * window_size:(i + 1) * window_size]
            input_data = np.concatenate([context, chunk]).reshape(1, -1).astype(np.float32)
            out, state = self.vad_sess.run(
                None, {'input': input_data, 'state': state, 'sr': sr_tensor})
            probs.append(float(out[0][0]))
            context = chunk[-context_size:]

        if not probs:
            return []

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

        if not regions:
            return []
        merged = [regions[0]]
        for s, e in regions[1:]:
            if s - merged[-1][1] < min_silence:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        audio_dur = n_samples / sr
        return [(max(0, s), min(e, audio_dur)) for s, e in merged]

    def process(self, audio_file=None, progress_callback=None,
                audio_data=None, audio_sample_rate=None):
        """Full pipeline: load audio -> VAD -> sliding windows -> embeddings
        -> spectral clustering -> output segments.

        Args:
            audio_file: path to audio file (WAV, FLAC, etc.)
            progress_callback: callable(current, total) for progress updates
            audio_data: pre-loaded audio as numpy array (float32, mono)
            audio_sample_rate: sample rate of audio_data

        Returns:
            List of dicts: [{'start': float, 'end': float, 'speaker': int}, ...]
        """
        import soundfile

        t_total = time.perf_counter()

        if self.emb_sess is None:
            self.initialize()

        # ── 1. Load audio ──────────────────────────────────────
        if audio_data is not None:
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sr = audio_sample_rate or SAMPLE_RATE
        else:
            if audio_file is None:
                raise ValueError("Either audio_file or audio_data must be provided")
            audio, sr = soundfile.read(audio_file, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, SAMPLE_RATE, quality='HQ')

        duration = len(audio) / SAMPLE_RATE
        logger.info(f"[3DSpeaker-ECAPA] Audio: {duration:.1f}s")

        if duration < 0.5:
            logger.warning("[3DSpeaker-ECAPA] Audio too short (<0.5s), returning empty")
            return []

        if progress_callback:
            progress_callback(5, 100)

        # ── 2. Energy-based VAD ────────────────────────────────
        t0 = time.perf_counter()
        speech_regions = self._silero_vad(audio, SAMPLE_RATE)
        t_vad = time.perf_counter() - t0

        if not speech_regions:
            logger.warning("[3DSpeaker-ECAPA] No speech detected by VAD")
            # Fallback: treat entire audio as one speech region
            speech_regions = [(0.0, duration)]

        total_speech = sum(e - s for s, e in speech_regions)
        logger.info(f"[3DSpeaker-ECAPA] VAD: {len(speech_regions)} regions, "
                     f"{total_speech:.1f}s speech / {duration:.1f}s total")

        if progress_callback:
            progress_callback(10, 100)

        # ── 3. Sliding window on speech regions ────────────────
        t0 = time.perf_counter()
        window_samples = int(self.window * SAMPLE_RATE)
        step_samples = int(self.step * SAMPLE_RATE)

        windows = []  # list of (start_sec, end_sec)
        for reg_start, reg_end in speech_regions:
            reg_start_samp = int(reg_start * SAMPLE_RATE)
            reg_end_samp = int(reg_end * SAMPLE_RATE)
            reg_len = reg_end_samp - reg_start_samp

            if reg_len < window_samples:
                # Region shorter than window: use entire region as one window
                windows.append((reg_start, reg_end))
            else:
                pos = reg_start_samp
                while pos + window_samples <= reg_end_samp:
                    w_start = pos / SAMPLE_RATE
                    w_end = (pos + window_samples) / SAMPLE_RATE
                    windows.append((w_start, w_end))
                    pos += step_samples
                # Handle last partial window if it extends past the last full window
                last_full_start = pos - step_samples
                if last_full_start + window_samples < reg_end_samp:
                    # There's remaining audio after the last full window
                    remaining = reg_end_samp - (last_full_start + window_samples)
                    if remaining >= step_samples // 2:
                        # Enough remaining audio: add a window aligned to the end
                        w_end = reg_end_samp / SAMPLE_RATE
                        w_start = max(reg_start, (reg_end_samp - window_samples) / SAMPLE_RATE)
                        # Avoid duplicating if it would be too close to last window
                        if not windows or (w_start - windows[-1][0]) >= self.step * 0.5:
                            windows.append((w_start, w_end))

        n_windows = len(windows)
        logger.info(f"[3DSpeaker-ECAPA] {n_windows} windows "
                     f"({self.window}s window, {self.step}s step)")

        if n_windows == 0:
            logger.warning("[3DSpeaker-ECAPA] No windows generated")
            return []

        t_window = time.perf_counter() - t0

        if progress_callback:
            progress_callback(15, 100)

        # ── 4. Embedding extraction ────────────────────────────
        t0 = time.perf_counter()
        embeddings = np.empty((n_windows, EMB_DIM), dtype=np.float32)
        out_name = self.emb_sess.get_outputs()[0].name

        for i, (w_start, w_end) in enumerate(windows):
            s_samp = int(w_start * SAMPLE_RATE)
            e_samp = int(w_end * SAMPLE_RATE)
            e_samp = min(e_samp, len(audio))
            segment = audio[s_samp:e_samp]

            if len(segment) < 400:  # less than 25ms, skip
                embeddings[i] = 0.0
                continue

            # Compute fbank features
            fbank = _compute_fbank(segment, SAMPLE_RATE)

            if fbank.shape[0] < 2:
                embeddings[i] = 0.0
                continue

            # Run ECAPA ONNX: input 'feats' [1, T, 80] -> output 'embs' [1, 192]
            emb = self.emb_sess.run(
                [out_name], {"feats": fbank[np.newaxis]}
            )[0][0]  # (192,)

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            embeddings[i] = emb

            if progress_callback and (i + 1) % 20 == 0:
                progress_callback(
                    15 + int((i + 1) / n_windows * 55), 100)

        t_emb = time.perf_counter() - t0

        # Filter out zero embeddings (failed extractions)
        valid_mask = np.linalg.norm(embeddings, axis=1) > 0.5
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logger.warning("[3DSpeaker-ECAPA] No valid embeddings extracted")
            return [{'start': 0.0, 'end': duration, 'speaker': 0}]

        valid_embeddings = embeddings[valid_indices]
        valid_windows = [windows[i] for i in valid_indices]

        if progress_callback:
            progress_callback(75, 100)

        # ── 5. Spectral clustering ─────────────────────────────
        t0 = time.perf_counter()

        # Determine number of speakers
        if self.num_speakers > 0:
            effective_max = self.num_speakers
        else:
            effective_max = self.max_speakers

        if len(valid_embeddings) <= 1:
            labels = np.zeros(len(valid_embeddings), dtype=np.int32)
        elif self.num_speakers == 1:
            labels = np.zeros(len(valid_embeddings), dtype=np.int32)
        else:
            # Determine min/max speakers (same logic as CAM++)
            if self.num_speakers > 0:
                min_spk = self.num_speakers
                max_spk = self.num_speakers
            else:
                min_spk = max(1, self.min_speakers)
                max_spk = effective_max

            labels = _spectral_cluster(
                valid_embeddings,
                pval=self.pval,
                min_cluster_size=self.min_cluster_size,
                max_speakers=max_spk,
                min_speakers=min_spk,
            )

        t_clust = time.perf_counter() - t0

        if progress_callback:
            progress_callback(90, 100)

        # ── 6. Build output segments ───────────────────────────
        t0 = time.perf_counter()
        raw_segments = []
        for i, (w_start, w_end) in enumerate(valid_windows):
            raw_segments.append({
                'start': w_start,
                'end': w_end,
                'speaker': int(labels[i])
            })

        # Sort by start time
        raw_segments.sort(key=lambda x: x['start'])

        # Merge overlapping/adjacent segments of the same speaker
        segments = _merge_segments(raw_segments, self.min_duration_off)

        # Clip to audio duration
        for seg in segments:
            seg['start'] = max(0.0, seg['start'])
            seg['end'] = min(duration, seg['end'])

        # Remove zero-length segments
        segments = [s for s in segments if s['end'] > s['start'] + 0.01]

        t_post = time.perf_counter() - t0
        t_total = time.perf_counter() - t_total

        n_spk = len(set(s['speaker'] for s in segments)) if segments else 0
        print(f"[3DSpeaker-ECAPA] {duration:.0f}s audio -> {t_total:.2f}s "
              f"(vad={t_vad:.2f} emb={t_emb:.2f} clust={t_clust:.2f} "
              f"post={t_post:.2f}) RTF={t_total / duration:.3f} "
              f"windows={n_windows} segs={len(segments)} spk={n_spk}")

        if progress_callback:
            progress_callback(100, 100)

        return segments


    # _forced_k_cluster removed: khi num_speakers cố định, truyền
    # min_speakers=max_speakers=k vào _spectral_cluster (đúng reference).


def _merge_segments(segments, min_duration_off=0.0):
    """Merge overlapping or adjacent segments of the same speaker.

    For overlapping windows assigned to different speakers, split at the midpoint.
    Then merge consecutive same-speaker segments with small gaps.
    """
    if not segments:
        return []

    # First pass: resolve overlaps between different speakers
    # by splitting at midpoint of overlap
    resolved = []
    for seg in segments:
        if not resolved:
            resolved.append(dict(seg))
            continue

        prev = resolved[-1]
        if seg['start'] < prev['end']:
            # Overlap exists
            if seg['speaker'] == prev['speaker']:
                # Same speaker: extend
                prev['end'] = max(prev['end'], seg['end'])
            else:
                # Different speaker: split at midpoint
                mid = (seg['start'] + min(prev['end'], seg['end'])) / 2.0
                prev['end'] = mid
                new_seg = dict(seg)
                new_seg['start'] = mid
                resolved.append(new_seg)
        else:
            resolved.append(dict(seg))

    # Second pass: merge consecutive same-speaker segments
    if not resolved:
        return []

    merged = [resolved[0]]
    for seg in resolved[1:]:
        if (seg['speaker'] == merged[-1]['speaker'] and
                seg['start'] - merged[-1]['end'] <= max(min_duration_off, 0.01)):
            merged[-1]['end'] = max(merged[-1]['end'], seg['end'])
        else:
            merged.append(seg)

    return merged
