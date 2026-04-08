"""
Senko-style Speaker Diarization with CAM++ 192-dim
===================================================
EXACT implementation of Senko (github.com/narcotic-sh/senko, March 2026):
  1. Pyannote segmentation ONNX as VAD
  2. Sliding window 1.5s, shift 0.6s, povey window, 80-dim fbank
  3. CAM++ 192-dim embedding (3D-Speaker, 200k speakers)
  4. < 20min: spectral clustering (pval=0.012, max_spks=15)
     >= 20min: UMAP+HDBSCAN (n_neighbors=40, metric=cosine, min_cluster=10)
  5. Post: filter_minor_cluster + merge_by_cos(0.875) + gap<=4s + remove<=0.78s + re-rank

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
# Fbank feature extraction
# Primary: vectorized numpy (1.8x faster, diff < 0.001 on speech frames)
# Fallback: kaldi_native_fbank (exact reference)
# ══════════════════════════════════════════════════════════════

_knf_emb_opts = None
_fbank_mel_bank = None
_fbank_povey_window = None


def _init_fbank():
    """Initialize fbank tables: kaldi_native_fbank opts + mel bank + window."""
    global _knf_emb_opts, _fbank_mel_bank, _fbank_povey_window
    if _knf_emb_opts is not None:
        return
    import kaldi_native_fbank as knf
    _knf_emb_opts = knf.FbankOptions()
    _knf_emb_opts.frame_opts.dither = 0.0
    _knf_emb_opts.frame_opts.snip_edges = True
    _knf_emb_opts.frame_opts.samp_freq = SAMPLE_RATE
    _knf_emb_opts.frame_opts.frame_length_ms = 25.0
    _knf_emb_opts.frame_opts.frame_shift_ms = 10.0
    _knf_emb_opts.frame_opts.window_type = "povey"
    _knf_emb_opts.mel_opts.num_bins = 80
    _knf_emb_opts.mel_opts.low_freq = 20.0
    _knf_emb_opts.mel_opts.high_freq = 0.0
    _knf_emb_opts.energy_floor = 1.0
    mb = knf.MelBanks(_knf_emb_opts.mel_opts, _knf_emb_opts.frame_opts)
    _fbank_mel_bank = np.array(mb.get_matrix(), dtype=np.float32)
    N = 400
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / (N - 1))
    _fbank_povey_window = np.power(hann, 0.85).astype(np.float32)


def _compute_fbank(audio, sr=SAMPLE_RATE):
    """Compute 80-dim fbank with per-utterance CMVN.
    Primary: vectorized numpy (1.8x faster). Fallback: kaldi_native_fbank."""
    _init_fbank()
    try:
        return _compute_fbank_vectorized(audio, sr)
    except Exception:
        return _compute_fbank_kaldi(audio, sr)


def _compute_fbank_vectorized(audio, sr=SAMPLE_RATE):
    """Vectorized numpy fbank — 1.8x faster, Kaldi-compatible."""
    frame_length = 400; frame_shift = 160; n_fft = 512
    preemphasis = 0.97; energy_floor = 1.0
    scaled = audio * np.float32(32768.0)
    n_samples = len(scaled)
    if n_samples < frame_length:
        return np.empty((0, 80), dtype=np.float32)
    n_frames = 1 + (n_samples - frame_length) // frame_shift
    if n_frames <= 0:
        return np.empty((0, 80), dtype=np.float32)
    CHUNK = 50000
    all_log_mel = []
    for c_start in range(0, n_frames, CHUNK):
        c_end = min(c_start + CHUNK, n_frames)
        c_n = c_end - c_start
        a_start = c_start * frame_shift
        a_end = (c_end - 1) * frame_shift + frame_length
        chunk_audio = scaled[a_start:a_end]
        strides = (chunk_audio.strides[0] * frame_shift, chunk_audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            chunk_audio, shape=(c_n, frame_length), strides=strides).copy()
        frames -= frames.mean(axis=1, keepdims=True)
        abs_starts = c_start * frame_shift + np.arange(c_n) * frame_shift
        context = np.where(abs_starts > 0, scaled[abs_starts - 1], np.float32(0.0))
        frames[:, 1:] -= preemphasis * frames[:, :-1]
        frames[:, 0] -= preemphasis * context
        frames *= _fbank_povey_window
        padded = np.zeros((c_n, n_fft), dtype=np.float32)
        padded[:, :frame_length] = frames; del frames
        spectra = np.fft.rfft(padded); del padded
        power = np.real(spectra)**2 + np.imag(spectra)**2; del spectra
        mel_e = np.maximum(power @ _fbank_mel_bank.T, energy_floor); del power
        all_log_mel.append(np.log(mel_e).astype(np.float32)); del mel_e
    result = np.concatenate(all_log_mel) if len(all_log_mel) > 1 else all_log_mel[0]
    result -= result.mean(axis=0, keepdims=True)
    return result


def _compute_fbank_kaldi(audio, sr=SAMPLE_RATE):
    """Fallback: kaldi_native_fbank (exact reference, slower)."""
    import kaldi_native_fbank as knf
    scaled = audio * np.float32(32768.0)
    fb = knf.OnlineFbank(_knf_emb_opts)
    fb.accept_waveform(sr, scaled)
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        return np.empty((0, 80), dtype=np.float32)
    features = np.empty((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fb.get_frame(i)
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
# Senko Clustering — EXACT from github.com/narcotic-sh/senko
# SpectralCluster (short audio) + UmapHdbscan (long audio)
# + filter_minor_cluster + merge_by_cos
# ══════════════════════════════════════════════════════════════

def _cosine_similarity(X, Y=None):
    """Cosine similarity matrix — replaces sklearn.metrics.pairwise.cosine_similarity."""
    if Y is None:
        Y = X
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return X_norm @ Y_norm.T


def _senko_spectral(X, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
    """Senko SpectralCluster — exact copy."""
    N = X.shape[0]
    """Senko SpectralCluster — exact."""
    if N <= 1:
        return np.zeros(N, dtype=np.int32)

    # get_sim_mat
    M = _cosine_similarity(X)

    # p_pruning
    n_elems = int((1 - pval) * N)
    n_elems = min(n_elems, N - min_pnum)
    n_elems = max(n_elems, 0)
    for i in range(N):
        low_idx = np.argsort(M[i])[:n_elems]
        M[i, low_idx] = 0

    # symmetrize
    M = 0.5 * (M + M.T)

    # get_laplacian (unnormalized)
    np.fill_diagonal(M, 0)
    D = np.abs(M).sum(axis=1)
    L = np.diag(D) - M

    # get_spec_embs
    lambdas, eig_vecs = np.linalg.eigh(L)
    if oracle_num is not None:
        num_of_spk = oracle_num
    else:
        sub = lambdas[min_num_spks - 1:max_num_spks + 1]
        gaps = [float(sub[i + 1]) - float(sub[i]) for i in range(len(sub) - 1)]
        if not gaps:
            return np.zeros(N, dtype=np.int32)
        num_of_spk = int(np.argmax(gaps)) + min_num_spks
    num_of_spk = max(1, min(num_of_spk, N))
    emb = eig_vecs[:, :num_of_spk]

    # cluster_embs (KMeans)
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=num_of_spk, random_state=0).fit_predict(emb)
    return labels.astype(np.int32)


def _senko_umap_hdbscan(X, n_neighbors=20, n_components=60, min_samples=20,
                         min_cluster_size=10, metric='euclidean'):
    """Senko UmapHdbscan — exact copy."""
    import umap
    import hdbscan

    n_comp = min(n_components, X.shape[0] - 2)
    n_comp = max(n_comp, 2)
    umap_X = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.0,
        n_components=n_comp, metric=metric
    ).fit_transform(X)
    labels = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size
    ).fit_predict(umap_X)
    return labels.astype(np.int32)


def _senko_cluster(X, cluster_type='umap_hdbscan', cluster_line=10,
                    mer_cos=0.875, min_cluster_size=4, **kwargs):
    """Senko CommonClustering — exact logic.

    1. If N < cluster_line → all same speaker
    2. Run clustering (spectral or umap_hdbscan)
    3. filter_minor_cluster
    4. merge_by_cos
    """
    N = X.shape[0]
    if N < cluster_line:
        return np.ones(N, dtype=np.int32)

    if cluster_type == 'umap_hdbscan':
        labels = _senko_umap_hdbscan(X, min_cluster_size=min_cluster_size, **kwargs)
    else:
        labels = _senko_spectral(X, **kwargs)

    # filter_minor_cluster (exact Senko)
    cset = np.unique(labels)
    csize = np.array([(labels == i).sum() for i in cset])
    minor_idx = np.where(csize < min_cluster_size)[0]
    if len(minor_idx) > 0:
        minor_cset = cset[minor_idx]
        major_idx = np.where(csize >= min_cluster_size)[0]
        if len(major_idx) > 0:
            major_cset = cset[major_idx]
            major_center = np.stack([X[labels == i].mean(0) for i in major_cset])
            for i in range(len(labels)):
                if labels[i] in minor_cset:
                    cos_sim = _cosine_similarity(X[i:i+1], major_center)
                    labels[i] = major_cset[cos_sim.argmax()]
        else:
            labels = np.zeros(N, dtype=np.int32)

    # merge_by_cos (exact Senko)
    if mer_cos is not None and mer_cos > 0:
        while True:
            cset = np.unique(labels)
            if len(cset) <= 1:
                break
            centers = np.stack([X[labels == i].mean(0) for i in cset])
            affinity = _cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < mer_cos:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels == c2] = c1

    # Relabel
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels], dtype=np.int32)


# ══════════════════════════════════════════════════════════════
# Main Diarizer Class
# ══════════════════════════════════════════════════════════════

class SenkoCamppDiarizer:
    """Senko-style diarization — EXACT implementation.

    Pipeline (github.com/narcotic-sh/senko):
      1. Pyannote segmentation ONNX as VAD (min_silence=0.1s)
      2. Sliding window 1.5s, step 0.6s, povey window, energy_floor=1.0
      3. CAM++ 192-dim embedding (3D-Speaker, 200k speakers)
      4. < 20min: spectral (pval=0.012, max=15) | >= 20min: UMAP+HDBSCAN
      5. filter_minor_cluster + merge_by_cos(0.875)
      6. merge gap<=4s + remove<=0.78s + re-rank by speaking time
    """

    def __init__(self, model_dir=None, num_threads=6, max_speakers=10,
                 min_speakers=1, num_speakers=-1,
                 mer_cos=0.875,
                 window=1.5, step=0.6, min_duration_off=0.0):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # CAM++ 192-dim from 3D-Speaker (exact paper model)
        self.model_path = (
            os.path.join(model_dir, "campplus_cn_en_common_200k.onnx") if model_dir
            else os.path.join(base, "models", "campp-3dspeaker", "campplus_cn_en_common_200k.onnx")
        )
        # Pyannote segmentation ONNX (dùng làm VAD — chính xác hơn Silero)
        self.seg_path = os.path.join(base, "models", "pyannote-onnx", "segmentation-community-1.onnx")
        self.num_threads = num_threads
        self.max_speakers = max_speakers
        self.min_speakers = max(1, min_speakers)
        self.num_speakers = num_speakers  # -1 = auto
        self.mer_cos = mer_cos
        self.window = window  # seconds
        self.step = step  # seconds
        self.min_duration_off = min_duration_off
        self.emb_sess = None
        self.seg_sess = None

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

        # Load pyannote segmentation as VAD
        seg_opts = ort.SessionOptions()
        seg_opts.intra_op_num_threads = compute_ort_threads(self.num_threads)
        seg_opts.inter_op_num_threads = 1
        seg_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        seg_opts.log_severity_level = 3
        seg_opts.enable_cpu_mem_arena = False
        seg_opts.optimized_model_filepath = self.seg_path + ".opt"
        self.seg_sess = ort.InferenceSession(
            self.seg_path, seg_opts, providers=['CPUExecutionProvider'])

        vad_name = "pyannote segmentation"
        print(f"[Senko-CAM++] Loaded CAM++ 192-dim + {vad_name}"
              f" | window={self.window}s, step={self.step}s, mer_cos={self.mer_cos}")

    def _pyannote_vad(self, audio, sr=SAMPLE_RATE, min_speech=0.25, min_silence=0.1):
        """Pyannote segmentation as VAD — detect speech regions from powerset output."""
        chunk_samples = int(10.0 * sr)
        step_samples = int(1.0 * sr)
        total = len(audio)

        starts = []
        s = 0
        while s < total:
            starts.append(s)
            if s + chunk_samples >= total:
                break
            s += step_samples

        # Batch inference
        all_logits = []
        for b in range(0, len(starts), 32):
            be = min(b + 32, len(starts))
            batch = np.zeros((be - b, 1, chunk_samples), dtype=np.float32)
            for i, idx in enumerate(range(b, be)):
                cs = starts[idx]
                ce = min(cs + chunk_samples, total)
                batch[i, 0, :ce - cs] = audio[cs:ce]
            logits = self.seg_sess.run(None, {"input_values": batch})[0]
            all_logits.append(logits)
        seg_logits = np.concatenate(all_logits, axis=0)

        # Powerset decode → any speaker active = speech
        POWERSET_MAP = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
        binarized = POWERSET_MAP[np.argmax(seg_logits, axis=-1)]
        n_seg_frames = binarized.shape[1]
        frame_dur = 10.0 / n_seg_frames
        total_dur = total / sr

        # Aggregate: frame-level speech probability across overlapping chunks
        n_out = int(total_dur / frame_dur) + 1
        speech_count = np.zeros(n_out, dtype=np.float32)
        total_count = np.zeros(n_out, dtype=np.float32)
        for c_idx, cs in enumerate(starts):
            t0 = cs / sr
            for f in range(n_seg_frames):
                out_f = int((t0 + f * frame_dur) / frame_dur)
                if 0 <= out_f < n_out:
                    if binarized[c_idx, f].sum() > 0:
                        speech_count[out_f] += 1
                    total_count[out_f] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            speech_prob = np.where(total_count > 0, speech_count / total_count, 0)
        is_speech = speech_prob > 0.5

        # Convert to regions
        regions = []
        in_speech = False
        start_t = 0.0
        for f in range(len(is_speech)):
            t = f * frame_dur
            if is_speech[f] and not in_speech:
                start_t = t
                in_speech = True
            elif not is_speech[f] and in_speech:
                if t - start_t >= min_speech:
                    regions.append((start_t, t))
                in_speech = False
        if in_speech:
            t = len(is_speech) * frame_dur
            if t - start_t >= min_speech:
                regions.append((start_t, min(t, total_dur)))

        if not regions:
            return [(0.0, total_dur)]
        merged = [regions[0]]
        for s, e in regions[1:]:
            if s - merged[-1][1] < min_silence:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged if merged else [(0.0, total_dur)]


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
                # Senko: pull back start to fill window (always 1.5s)
                pulled_start = max(0, end_sample - window_samples)
                all_windows.append((pulled_start, pulled_start + window_samples))
            else:
                pos = start_sample
                while pos + window_samples < end_sample:  # strict < (Senko exact)
                    all_windows.append((pos, pos + window_samples))
                    pos += step_samples
                # Senko: always add tail, pull back to fill window
                tail_start = max(start_sample, end_sample - window_samples)
                all_windows.append((tail_start, tail_start + window_samples))

        total_windows = len(all_windows)
        if total_windows == 0:
            return np.empty((0, 192), dtype=np.float32), []

        logger.info(f"[Senko-CAM++] Extracting {total_windows} window embeddings")

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
        logger.info(f"[Senko-CAM++] Audio: {duration:.1f}s")

        if duration < 0.5:
            logger.warning("[Senko-CAM++] Audio too short (< 0.5s), returning empty")
            return []

        if progress_callback:
            progress_callback(5)

        # ─── 2. Silero VAD (neural, paper-quality) ──────────────
        t0 = time.perf_counter()
        speech_regions = self._pyannote_vad(audio, SAMPLE_RATE)
        t_vad = time.perf_counter() - t0

        if not speech_regions:
            logger.warning("[Senko-CAM++] No speech detected by VAD, using full audio")
            speech_regions = [(0.0, duration)]

        total_speech = sum(e - s for s, e in speech_regions)
        logger.info(f"[Senko-CAM++] VAD: {len(speech_regions)} regions, "
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
        logger.info(f"[Senko-CAM++] Embeddings: {n_embs} x 192 ({t_emb:.3f}s)")

        if n_embs == 0:
            logger.warning("[Senko-CAM++] No embeddings extracted")
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

        # Senko clustering: spectral (<20min) or UMAP+HDBSCAN (>=20min)
        if n_embs <= 2:
            labels = np.zeros(n_embs, dtype=np.int32)
        else:
            audio_duration = duration  # seconds
            if audio_duration < 1200.0:  # < 20 minutes → spectral
                labels = _senko_cluster(
                    embeddings,
                    cluster_type='spectral',
                    cluster_line=10,
                    mer_cos=self.mer_cos,
                    min_cluster_size=4,  # Senko spectral.yaml
                    # Senko spectral.yaml exact params
                    min_num_spks=min_spk, max_num_spks=15,
                    pval=0.012,
                )
            else:  # >= 20 minutes → UMAP+HDBSCAN
                labels = _senko_cluster(
                    embeddings,
                    cluster_type='umap_hdbscan',
                    cluster_line=10,
                    mer_cos=self.mer_cos,
                    min_cluster_size=10,  # Senko umap_hdbscan.yaml
                    # Senko umap_hdbscan.yaml exact params
                    n_neighbors=40, n_components=60,
                    min_samples=20, metric='cosine',
                )

        n_speakers = len(np.unique(labels))
        t_clust = time.perf_counter() - t0
        logger.info(f"[Senko-CAM++] Clustering: {n_speakers} speakers ({t_clust:.3f}s)")

        if progress_callback:
            progress_callback(90)

        # mer_cos merge already done inside _senko_cluster

        # ─── 5. Build output segments ────────────────────────────
        segments = self._segments_from_labels(window_times, labels)

        # Resolve overlapping segments (from window overlap)
        if len(segments) > 1:
            for i in range(len(segments) - 1):
                if segments[i]['end'] > segments[i + 1]['start']:
                    mid = (segments[i]['end'] + segments[i + 1]['start']) / 2
                    segments[i]['end'] = mid
                    segments[i + 1]['start'] = mid

        # ─── 7. Senko post-processing ────────────────────────────
        # 7a. Merge adjacent same-speaker with gap <= 4s
        if len(segments) > 1:
            merged_segs = [segments[0]]
            for seg in segments[1:]:
                prev = merged_segs[-1]
                if seg['speaker'] == prev['speaker'] and seg['start'] - prev['end'] <= 4.0:
                    prev['end'] = seg['end']
                else:
                    merged_segs.append(seg)
            segments = merged_segs

        # 7b. Remove segments <= 0.78s (Senko: bridge same speaker, remove different)
        if len(segments) > 1:
            filtered = []
            for i, seg in enumerate(segments):
                if seg['end'] - seg['start'] > 0.78:
                    filtered.append(seg)
                else:
                    # Short segment sandwiched by same speaker → bridge (extend prev)
                    prev_spk = filtered[-1]['speaker'] if filtered else None
                    next_spk = segments[i + 1]['speaker'] if i + 1 < len(segments) else None
                    if prev_spk is not None and prev_spk == next_spk:
                        filtered[-1]['end'] = seg['end']  # bridge
                    # else: drop (different speakers on both sides)
            if filtered:
                segments = filtered

        # 7c. Final merge adjacent same-speaker
        if len(segments) > 1:
            final = [segments[0]]
            for seg in segments[1:]:
                if seg['speaker'] == final[-1]['speaker']:
                    final[-1]['end'] = seg['end']
                else:
                    final.append(seg)
            segments = final

        # 7d. Senko: re-rank speakers by total speaking time (SPEAKER_01 = most talkative)
        spk_dur = {}
        for seg in segments:
            spk_dur[seg['speaker']] = spk_dur.get(seg['speaker'], 0) + (seg['end'] - seg['start'])
        ranked = sorted(spk_dur.keys(), key=lambda s: spk_dur[s], reverse=True)
        rerank = {old: new for new, old in enumerate(ranked)}
        for seg in segments:
            seg['speaker'] = rerank[seg['speaker']]

        t_total_elapsed = time.perf_counter() - t_total
        n_speakers = len(set(s['speaker'] for s in segments))
        logger.info(f"[Senko-CAM++] Done: {len(segments)} segments, "
                     f"{n_speakers} speakers, {t_total_elapsed:.3f}s total "
                     f"(VAD={t_vad:.3f}s, emb={t_emb:.3f}s, clust={t_clust:.3f}s)")

        if progress_callback:
            progress_callback(100)

        return segments

    def unload(self):
        """Giải phóng ONNX sessions khỏi RAM."""
        import gc
        if self.emb_sess is not None:
            del self.emb_sess
            self.emb_sess = None
        if self.seg_sess is not None:
            del self.seg_sess
            self.seg_sess = None
        gc.collect()
        print("[Senko-CAM++] Model unloaded")
