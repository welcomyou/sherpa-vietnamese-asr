"""
[PRODUCTION - ĐANG DÙNG CHÍNH TRONG MAIN CODE]
ResNet34-LM 256-dim + PLDA + VBx clustering. Zero PyTorch dependency.
Nhanh nhất và ổn định nhất trong tất cả các variant.

Pure ORT Speaker Diarization — No pyannote runtime dependency
═══════════════════════════════════════════════════════════════
Replicates pyannote Community-1 pipeline EXACTLY using:
  - onnxruntime (segmentation + embedding ONNX inference)
  - kaldi-native-fbank (feature extraction, same C++ as sherpa-onnx)
  - numpy + scipy (VBx clustering, PLDA, reconstruction)

Every step matches pyannote's SpeakerDiarization.apply() precisely:
  SlidingWindow alignment, Inference.aggregate(), to_diarization(),
  speaker_count(), reconstruct() — all replicated from pyannote source.
"""
import os
import math
import logging
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Lightweight replacements for pyannote.core (zero dependencies)
# ══════════════════════════════════════════════════════════════

class _Seg:
    """Minimal Segment: time span [start, end)."""
    __slots__ = ('start', 'end')
    def __init__(self, start, end):
        self.start = start
        self.end = end
    @property
    def duration(self):
        return self.end - self.start
    @property
    def middle(self):
        return 0.5 * (self.start + self.end)
    def __and__(self, other):
        return _Seg(max(self.start, other.start), min(self.end, other.end))
    def __bool__(self):
        return self.end > self.start


class _SW:
    """Minimal SlidingWindow: frame↔time mapping."""
    __slots__ = ('start', 'duration', 'step')
    def __init__(self, start=0.0, duration=0.0, step=0.0):
        self.start = start
        self.duration = duration
        self.step = step
    def __getitem__(self, i):
        s = self.start + i * self.step
        return _Seg(s, s + self.duration)
    def closest_frame(self, t):
        return int(np.rint((t - self.start - 0.5 * self.duration) / self.step))
    def range_to_segment(self, i0, n):
        s = self.start + (i0 - 0.5) * self.step + 0.5 * self.duration
        e = s + n * self.step
        if i0 == 0:
            s = self.start
        return _Seg(s, e)
    def crop_loose(self, focus):
        """Loose crop: return (i, j) frame range [i, j)."""
        i = int(np.ceil((focus.start - self.duration - self.start) / self.step))
        j = int(np.floor((focus.end - self.start) / self.step))
        return i, j + 1


class _SWF:
    """Minimal SlidingWindowFeature: data array + SlidingWindow."""
    def __init__(self, data, sliding_window):
        self.data = data
        self.sliding_window = sliding_window
    def __iter__(self):
        sw = self.sliding_window
        for i in range(len(self.data)):
            yield sw[i], self.data[i]
    def __len__(self):
        return len(self.data)
    @property
    def extent(self):
        return self.sliding_window.range_to_segment(0, len(self.data))
    def crop(self, focus, return_data=True):
        sw = self.sliding_window
        n = len(self.data)
        i, j = sw.crop_loose(focus)
        i, j = max(i, 0), min(j, n)
        if i >= j:
            empty = np.empty((0,) + self.data.shape[1:], dtype=self.data.dtype)
            return empty if return_data else _SWF(empty, sw)
        data = self.data[i:j]
        if return_data:
            return data
        return _SWF(data, _SW(start=sw[i].start, duration=sw.duration, step=sw.step))


# ══════════════════════════════════════════════════════════════
# Model architecture constants (from Community-1 PyanNet model)
# These are FIXED properties of the trained model, not tunable.
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0          # segmentation window (seconds)
CHUNK_STEP = 1.0               # segmentation step (seconds)
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
STEP_SAMPLES = int(CHUNK_STEP * SAMPLE_RATE)
NUM_SEG_FRAMES = 589           # output frames per 10s chunk
MAX_SPEAKERS_PER_CHUNK = 3

# Receptive field of PyanNet segmentation model (computed from conv chain)
RF_START = 0.0
RF_DURATION = 0.0619375        # seconds per output frame (receptive field size)
RF_STEP = 0.016875             # seconds between output frames

# Powerset mapping: Community-1 uses max_classes_per_frame=2
# Order: silence, single speakers, then pairs (from pyannote Powerset class)
POWERSET_MAP = np.array([
    [0, 0, 0],  # 0: silence
    [1, 0, 0],  # 1: spk 0
    [0, 1, 0],  # 2: spk 1
    [0, 0, 1],  # 3: spk 2
    [1, 1, 0],  # 4: spk 0+1
    [1, 0, 1],  # 5: spk 0+2
    [0, 1, 1],  # 6: spk 1+2
], dtype=np.float32)

# VBx defaults from config.yaml
DEFAULT_THRESHOLD = 0.6
DEFAULT_FA = 0.07
DEFAULT_FB = 0.8


# ══════════════════════════════════════════════════════════════
# Aggregate (replicated from pyannote Inference.aggregate)
# ══════════════════════════════════════════════════════════════

def _pyannote_aggregate(data, chunk_sw, frame_sw,
                        hamming=False, skip_average=False, missing=np.nan,
                        warm_up=(0.0, 0.0)):
    """Aggregate overlapping chunk scores into frame-level scores.
    Exact replication of pyannote Inference.aggregate() — no pyannote dependency."""

    scores = _SWF(data, chunk_sw)
    frames = _SW(start=chunk_sw.start, duration=frame_sw.duration,
                 step=frame_sw.step)

    num_chunks, num_frames_per_chunk, num_classes = data.shape
    epsilon = 1e-12

    hamming_window = (np.hamming(num_frames_per_chunk).reshape(-1, 1)
                      if hamming else np.ones((num_frames_per_chunk, 1)))

    warm_up_window = np.ones((num_frames_per_chunk, 1))
    wu_left = round(warm_up[0] / chunk_sw.duration * num_frames_per_chunk)
    if wu_left > 0:
        warm_up_window[:wu_left] = epsilon
    wu_right = round(warm_up[1] / chunk_sw.duration * num_frames_per_chunk)
    if wu_right > 0:
        warm_up_window[num_frames_per_chunk - wu_right:] = epsilon

    num_frames = (frames.closest_frame(
        chunk_sw.start + chunk_sw.duration
        + (num_chunks - 1) * chunk_sw.step
        + 0.5 * frames.duration) + 1)

    agg_output = np.zeros((num_frames, num_classes), dtype=np.float32)
    agg_weight = np.zeros((num_frames, num_classes), dtype=np.float32)
    agg_mask = np.zeros((num_frames, num_classes), dtype=np.float32)

    for chunk, score in scores:
        mask = 1.0 - np.isnan(score).astype(np.float32)
        np.nan_to_num(score, copy=False, nan=0.0)
        start_frame = frames.closest_frame(chunk.start + 0.5 * frames.duration)
        n = min(num_frames_per_chunk, num_frames - start_frame)
        if n <= 0:
            continue
        w = mask[:n] * hamming_window[:n] * warm_up_window[:n]
        agg_output[start_frame:start_frame + n] += score[:n] * w
        agg_weight[start_frame:start_frame + n] += w
        agg_mask[start_frame:start_frame + n] = np.maximum(
            agg_mask[start_frame:start_frame + n], mask[:n])

    if skip_average:
        result = agg_output
    else:
        result = agg_output / np.maximum(agg_weight, epsilon)

    result[agg_mask == 0.0] = missing
    return _SWF(result, frames)


def _trim(data, warm_up=(0.1, 0.1)):
    """Replicate pyannote Inference.trim()"""
    num_chunks, num_frames, num_classes = data.shape
    nl = round(num_frames * warm_up[0])
    nr = round(num_frames * warm_up[1])
    trimmed = data[:, nl:num_frames - nr]

    new_duration = (1 - warm_up[0] - warm_up[1]) * CHUNK_DURATION
    new_start = warm_up[0] * CHUNK_DURATION
    return trimmed, new_start, new_duration


def _binarize(data, sliding_window, onset=0.5, offset=0.5,
              min_duration_on=0.0, min_duration_off=0.0):
    """Replacement for pyannote.audio.utils.signal.Binarize — no pyannote at all.
    Hysteresis thresholding per speaker, time via frame midpoints.
    Returns list of (start, end, label) sorted by start time."""

    num_frames, num_classes = data.shape
    timestamps = [sliding_window[i].middle for i in range(num_frames)]

    all_segments = []

    for k in range(num_classes):
        k_scores = data[:, k]
        segs = []

        start = timestamps[0]
        is_active = k_scores[0] > onset

        t = timestamps[0]
        for t, y in zip(timestamps[1:], k_scores[1:]):
            if is_active:
                if y < offset:
                    segs.append([start, t])
                    start = t
                    is_active = False
            else:
                if y > onset:
                    start = t
                    is_active = True

        if is_active:
            segs.append([start, t])

        # Fill gaps ≤ collar (matches Annotation.support)
        if min_duration_off > 0.0 and len(segs) > 1:
            merged = [segs[0]]
            for s, e in segs[1:]:
                if s - merged[-1][1] <= min_duration_off:
                    merged[-1][1] = e
                else:
                    merged.append([s, e])
            segs = merged

        # Remove short segments
        if min_duration_on > 0.0:
            segs = [s for s in segs if s[1] - s[0] >= min_duration_on]

        for s, e in segs:
            all_segments.append((s, e, k))

    all_segments.sort(key=lambda x: x[0])
    return all_segments


# ══════════════════════════════════════════════════════════════
# Fbank (kaldi-native-fbank)
# ══════════════════════════════════════════════════════════════

_knf_emb_opts = None

def compute_emb_fbank(audio, sr=SAMPLE_RATE):
    """Compute fbank matching WeSpeaker's compute_fbank EXACTLY:
    - Scale waveform by 32768 (int16 range)
    - Hamming window (not povey)
    - No energy (use_energy=False → energy_floor=0)
    - Per-utterance mean normalization (CMVN)
    Source: wespeaker/bin/infer_onnx.py
    """
    global _knf_emb_opts
    import kaldi_native_fbank as knf
    if _knf_emb_opts is None:
        _knf_emb_opts = knf.FbankOptions()
        _knf_emb_opts.frame_opts.dither = 0.0
        _knf_emb_opts.frame_opts.snip_edges = True  # torchaudio default (kaldi default)
        _knf_emb_opts.frame_opts.samp_freq = sr
        _knf_emb_opts.frame_opts.frame_length_ms = 25.0
        _knf_emb_opts.frame_opts.frame_shift_ms = 10.0
        _knf_emb_opts.frame_opts.window_type = "hamming"  # WeSpeaker uses hamming
        _knf_emb_opts.mel_opts.num_bins = 80
        _knf_emb_opts.mel_opts.low_freq = 20.0
        _knf_emb_opts.mel_opts.high_freq = 0.0  # 0 = Nyquist (8000Hz at 16kHz), torchaudio default
        _knf_emb_opts.energy_floor = 0.0  # WeSpeaker: use_energy=False
    # Scale by 32768 (WeSpeaker convention: float [-1,1] → int16 range)
    scaled_audio = audio * np.float32(32768.0)
    fb = knf.OnlineFbank(_knf_emb_opts)
    fb.accept_waveform(sr, scaled_audio)
    fb.input_finished()
    n = fb.num_frames_ready
    features = np.empty((n, 80), dtype=np.float32)
    for i in range(n):
        features[i] = fb.get_frame(i)
    # Per-utterance mean normalization (CMVN)
    features -= features.mean(axis=0, keepdims=True)
    return features


# ══════════════════════════════════════════════════════════════
# PLDA + VBx (from pyannote/vbx.py — exact copy)
# ══════════════════════════════════════════════════════════════

def l2_norm(x):
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-10)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def load_plda(model_dir):
    plda_dir = os.path.join(model_dir, "plda")
    x = np.load(os.path.join(plda_dir, "xvec_transform.npz"))
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]
    p = np.load(os.path.join(plda_dir, "plda.npz"))
    plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]
    W = np.linalg.inv(plda_tr.T @ plda_tr)
    B = np.linalg.inv((plda_tr.T / plda_psi) @ plda_tr)
    acvar, wccn = eigh(B, W)
    return {
        'mean1': mean1, 'mean2': mean2, 'lda': lda,
        'plda_mu': plda_mu, 'plda_tr': wccn.T[::-1],
        'plda_psi': acvar[::-1],
    }


def xvec_transform(embeddings, pd):
    D_out = pd['lda'].shape[1]
    return l2_norm(
        (l2_norm(embeddings - pd['mean1']) * np.sqrt(pd['lda'].shape[0])) @ pd['lda'] - pd['mean2']
    ) * np.sqrt(D_out)


def plda_transform(embeddings, pd, lda_dim=128):
    return (embeddings - pd['plda_mu']) @ pd['plda_tr'].T[:, :lda_dim]


def vbx_cluster(fea, plda_psi, ahc_labels, Fa, Fb, max_iters=20):
    """VBx clustering — exact from pyannote/vbx.py"""
    T, D = fea.shape
    n_clusters = ahc_labels.max() + 1
    qinit = np.zeros((T, n_clusters))
    qinit[np.arange(T), ahc_labels.astype(int)] = 1.0
    gamma = softmax(qinit * 7.0, axis=1)
    pi = np.ones(n_clusters) / n_clusters
    G = -0.5 * (np.sum(fea**2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(plda_psi)
    rho = fea * V
    prev_elbo = -np.inf
    for ii in range(max_iters):
        invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * plda_psi)
        alpha = Fa / Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(plda_psi) + G)
        lpi = np.log(pi + 1e-8)
        log_p_x = logsumexp(log_p_ + lpi, axis=-1)
        gamma = np.exp(log_p_ + lpi - log_p_x[:, None])
        pi = gamma.sum(axis=0)
        pi = pi / pi.sum()
        elbo = np.sum(log_p_x) + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)
        if ii > 0 and elbo - prev_elbo < 1e-4:
            break
        prev_elbo = elbo
    return gamma, pi


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

class PureOrtDiarizer:
    def __init__(self, model_dir=None, onnx_dir=None, num_threads=6,
                 threshold=DEFAULT_THRESHOLD, Fa=DEFAULT_FA, Fb=DEFAULT_FB,
                 min_duration_off=0.0, num_speakers=-1,
                 min_speakers=None, max_speakers=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(base, "models", "pyannote",
                                                     "speaker-diarization-community-1")
        self.onnx_dir = onnx_dir or os.path.join(base, "models", "pyannote-onnx")
        self.num_threads = num_threads
        self.threshold = threshold
        self.Fa = Fa
        self.Fb = Fb
        self.min_duration_off = min_duration_off
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.seg_sess = None
        self.emb_sess = None
        self.emb_W = None  # Gemm weight for manual linear after masked pooling
        self.emb_b = None  # Gemm bias
        self.plda_data = None
        self.speaker_centroids = None  # (n_speakers, 256) — saved after clustering

    def initialize(self):
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        prov = ['CPUExecutionProvider']

        from core.config import compute_ort_threads
        Z_seg = compute_ort_threads(self.num_threads)           # conservative (50% HT)
        Z_emb = compute_ort_threads(self.num_threads, full_ht=True)  # full HT cho embedding
        print(f"[PureORT] Threads: cpu_threads={self.num_threads}, seg={Z_seg}, emb={Z_emb}")

        opts_seg = ort.SessionOptions()
        opts_seg.intra_op_num_threads = Z_seg
        opts_seg.inter_op_num_threads = 1
        opts_seg.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts_seg.log_severity_level = 3
        opts_seg.enable_cpu_mem_arena = False  # Tránh arena leak 800+ MB không shrink

        opts_emb = ort.SessionOptions()
        opts_emb.intra_op_num_threads = Z_emb
        opts_emb.inter_op_num_threads = 2  # Pipeline operators (benchmark: +3-5% throughput)
        opts_emb.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts_emb.log_severity_level = 3
        opts_emb.enable_cpu_mem_arena = False  # Tránh arena leak 1800+ MB không shrink

        seg_model = os.path.join(self.onnx_dir, "segmentation-community-1.onnx")
        # Cache optimized graph: giảm disk I/O lần load sau
        opts_seg.optimized_model_filepath = seg_model + ".opt"
        self.seg_sess = ort.InferenceSession(seg_model, opts_seg, providers=prov)

        # Use encoder-only model (frame features) + external masked pooling + Gemm
        encoder_path = os.path.join(self.onnx_dir, "embedding_encoder.onnx")
        split_path = os.path.join(self.onnx_dir, "embedding_model_split.onnx")
        if os.path.exists(encoder_path):
            opts_emb.optimized_model_filepath = encoder_path + ".opt"
            self.emb_sess = ort.InferenceSession(encoder_path, opts_emb, providers=prov)
            self.emb_W = np.load(os.path.join(self.onnx_dir, "resnet_seg_1_weight.npy"))
            self.emb_b = np.load(os.path.join(self.onnx_dir, "resnet_seg_1_bias.npy"))
        elif os.path.exists(split_path):
            self.emb_sess = ort.InferenceSession(split_path, opts_emb, providers=prov)
            self.emb_W = np.load(os.path.join(self.onnx_dir, "resnet_seg_1_weight.npy"))
            self.emb_b = np.load(os.path.join(self.onnx_dir, "resnet_seg_1_bias.npy"))
        else:
            self.emb_sess = ort.InferenceSession(
                os.path.join(self.onnx_dir, "embedding_model.onnx"),
                opts_emb, providers=prov)

        self.plda_data = load_plda(self.model_dir)

    def process(self, audio_file, progress_callback=None,
                audio_data=None, audio_sample_rate=None):
        import soundfile
        import time

        t_total = time.perf_counter()

        # 1. Load audio (or use pre-loaded data)
        if audio_data is not None:
            audio = np.asarray(audio_data, dtype=np.float32)  # không copy nếu đã float32
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sr = audio_sample_rate or SAMPLE_RATE
        else:
            audio, sr = soundfile.read(audio_file, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE

        # 2. Segmentation
        t0 = time.perf_counter()
        seg_logits, chunk_starts = self._segment(audio)
        num_chunks = seg_logits.shape[0]
        num_seg_frames = seg_logits.shape[1]
        t_seg = time.perf_counter() - t0

        # 3. Powerset decode (argmax → one_hot → mapping = HARD binary)
        binarized = POWERSET_MAP[np.argmax(seg_logits, axis=-1)]
        del seg_logits  # đã dùng xong, giải phóng ~119MB (2hr)

        # 4. Speaker count (pyannote: trim(0.0,0.0) → aggregate → round)
        t0 = time.perf_counter()
        count = self._speaker_count(binarized, num_chunks, num_seg_frames)
        t_count = time.perf_counter() - t0

        # 5. Embedding extraction
        # Community-1 config: embedding_exclude_overlap=True
        # → prefer clean (non-overlap) mask, fallback to full mask
        t0 = time.perf_counter()
        clean_frames = (binarized.sum(axis=2, keepdims=True) < 2).astype(np.float32)
        clean_binarized = binarized * clean_frames
        emb_min_num_samples = 1680
        min_seg_frames = math.ceil(
            num_seg_frames * emb_min_num_samples / CHUNK_SAMPLES)
        embeddings = self._extract_embeddings(
            audio, binarized, clean_binarized, num_seg_frames,
            chunk_starts, min_seg_frames, progress_callback)
        t_emb = time.perf_counter() - t0

        # Audio không cần nữa — clustering + reconstruct không dùng
        del audio

        # 5b. Filter embeddings (matching pyannote filter_embeddings)
        # VBx uses filtered embeddings, constrained assignment uses ALL.
        single_active = (binarized.sum(axis=2, keepdims=True) == 1).astype(np.float32)
        num_clean_per_spk = (binarized * single_active).sum(axis=1)  # (nc, 3)
        active_enough = num_clean_per_spk >= 0.2 * num_seg_frames
        valid_emb = ~np.isnan(embeddings[:, :, 0])
        train_mask = active_enough & valid_emb

        # 6. VBx Clustering
        t0 = time.perf_counter()
        max_cl = None
        if self.num_speakers > 0:
            max_cl = self.num_speakers
        elif self.max_speakers is not None:
            max_cl = self.max_speakers
        hard_clusters = self._cluster(embeddings, train_mask, binarized,
                                       max_clusters=max_cl)
        t_clust = time.perf_counter() - t0

        # 7. Post-clustering processing (matching pyannote exactly)
        # Cap count at number of detected speakers
        num_detected = int(hard_clusters.max()) + 1
        count.data = np.minimum(count.data, num_detected).astype(np.int8)

        # Force-assign inactive speakers to throw-away cluster -2
        inactive = np.sum(binarized, axis=1) == 0  # (num_chunks, 3)
        hard_clusters[inactive] = -2

        # For exclusive diarization: cap count at 1
        count.data = np.minimum(count.data, 1).astype(np.int8)

        # 8. Reconstruct + to_diarization
        t0 = time.perf_counter()
        segments = self._reconstruct_and_diarize(
            binarized, hard_clusters, count, num_chunks,
            num_seg_frames, duration)
        t_recon = time.perf_counter() - t0

        # 8. min_duration_off
        if self.min_duration_off > 0 and len(segments) > 1:
            merged = [segments[0]]
            for seg in segments[1:]:
                if (seg['speaker'] == merged[-1]['speaker'] and
                    seg['start'] - merged[-1]['end'] <= self.min_duration_off):
                    merged[-1]['end'] = seg['end']
                else:
                    merged.append(seg)
            segments = merged

        t_total = time.perf_counter() - t_total
        n_spk = len(set(s['speaker'] for s in segments))
        print(f"[PureORT] {duration:.0f}s audio → {t_total:.2f}s "
              f"(seg={t_seg:.2f} emb={t_emb:.2f} clust={t_clust:.2f} "
              f"recon={t_recon:.2f}) RTF={t_total/duration:.3f} "
              f"segs={len(segments)} spk={n_spk}")
        return segments

    def compute_single_embedding(self, audio_segment):
        """Compute 256-dim speaker embedding for a short audio segment.
        Used for verifying speaker identity of words in diarization gaps.
        Returns None if audio too short (<9 frames ~= 0.1s).
        """
        if self.emb_sess is None:
            return None
        audio_segment = np.asarray(audio_segment, dtype=np.float32)
        fbank = compute_emb_fbank(audio_segment, SAMPLE_RATE)
        if fbank.shape[0] < 9:
            return None

        if self.emb_W is None:
            # Full embedding model: fbank → embedding directly
            return self.emb_sess.run(
                None, {"fbank_features": fbank[np.newaxis]})[0][0]
        else:
            # Encoder-only: fbank → frame features → stats pool → project
            frame_feat = self.emb_sess.run(
                None, {"fbank_features": fbank[np.newaxis]})[0][0]  # (D, F)
            # Stats pooling (no mask — entire segment is the target speaker)
            mean = frame_feat.mean(axis=1)
            dx2 = (frame_feat - mean[:, np.newaxis]) ** 2
            var = dx2.mean(axis=1)
            std = np.sqrt(var)
            stats = np.concatenate([mean, std])
            return stats @ self.emb_W.T + self.emb_b

    def _segment(self, audio):
        total_samples = len(audio)
        duration = total_samples / SAMPLE_RATE
        # Match pyannote Inference.slide(): process chunks, include last
        # partial chunk (padded with zeros) when chunk.end > duration
        starts = []
        s = 0
        has_last = False
        while True:
            if has_last:
                break
            chunk_end = (s + CHUNK_SAMPLES) / SAMPLE_RATE
            if chunk_end > duration:
                has_last = True  # include this chunk (with padding), then stop
            starts.append(s)
            s += STEP_SAMPLES
        if not starts:
            starts = [0]

        batch_size = 32
        all_logits = []
        for b in range(0, len(starts), batch_size):
            be = min(b + batch_size, len(starts))
            batch = np.zeros((be - b, 1, CHUNK_SAMPLES), dtype=np.float32)
            for i, idx in enumerate(range(b, be)):
                s = starts[idx]
                e = min(s + CHUNK_SAMPLES, total_samples)
                batch[i, 0, :e - s] = audio[s:e]
            logits = self.seg_sess.run(None, {"input_values": batch})[0]
            all_logits.append(logits)

        return np.concatenate(all_logits, axis=0), starts

    def _speaker_count(self, binarized, num_chunks, num_seg_frames):
        """pyannote: speaker_count(binarized, frames, warm_up=(0.0, 0.0))"""
        chunk_sw = _SW(start=0.0, duration=CHUNK_DURATION, step=CHUNK_STEP)
        frame_sw = _SW(start=RF_START, duration=RF_DURATION, step=RF_STEP)

        count_per_frame = binarized.sum(axis=-1, keepdims=True)  # (C, F, 1)
        count_swf = _pyannote_aggregate(
            count_per_frame, chunk_sw, frame_sw,
            hamming=False, skip_average=False, missing=0.0,
            warm_up=(0.0, 0.0))

        count_swf.data = np.rint(count_swf.data).astype(np.uint8)
        return count_swf

    def _masked_stats_pool(self, frame_feat, weights):
        """Weighted statistics pooling matching pyannote StatsPool._pool().
        frame_feat: (features, frames), weights: (frames,) binary/float mask.
        Returns (2*features,) = concat(weighted_mean, weighted_std)."""
        w = weights[np.newaxis, :]  # (1, frames)
        v1 = w.sum() + 1e-8
        mean = (frame_feat * w).sum(axis=1) / v1  # (features,)
        dx2 = (frame_feat - mean[:, np.newaxis]) ** 2
        v2 = (w * w).sum()
        var = (dx2 * w).sum(axis=1) / (v1 - v2 / v1 + 1e-8)
        std = np.sqrt(var)
        return np.concatenate([mean, std])

    def _extract_embeddings(self, audio, binarized, clean_binarized,
                             num_seg_frames, chunk_starts, min_seg_frames,
                             progress_callback):
        num_chunks = binarized.shape[0]
        total_samples = len(audio)
        embeddings = np.full((num_chunks, MAX_SPEAKERS_PER_CHUNK, 256),
                             np.nan, dtype=np.float32)

        if self.emb_W is None:
            # Fallback: no encoder-only model
            for c in range(num_chunks):
                cs = chunk_starts[c]
                ce = min(cs + CHUNK_SAMPLES, total_samples)
                ca = audio[cs:ce]
                if len(ca) < CHUNK_SAMPLES:
                    ca = np.pad(ca, (0, CHUNK_SAMPLES - len(ca)))
                fbank = compute_emb_fbank(ca, SAMPLE_RATE)
                for s in range(MAX_SPEAKERS_PER_CHUNK):
                    mask = binarized[c, :, s]
                    cm = clean_binarized[c, :, s]
                    um = cm if cm.sum() > min_seg_frames else mask
                    if um.sum() < 1:
                        continue
                    fb_idx = np.floor(np.arange(fbank.shape[0]) * num_seg_frames / fbank.shape[0]).astype(int)
                    fb_idx = np.clip(fb_idx, 0, num_seg_frames - 1)
                    mfb = fbank[um[fb_idx] > 0.5]
                    if mfb.shape[0] < 9:
                        continue
                    embeddings[c, s, :] = self.emb_sess.run(None, {"fbank_features": mfb[np.newaxis]})[0][0]
            return embeddings

        emb_W, emb_b = self.emb_W, self.emb_b
        frames_per_chunk = int(CHUNK_DURATION * 1000 / 10) - 2  # 998 for 10s

        # --- 1. Single-stream fbank (numpy direct, không .tolist()) ---
        import kaldi_native_fbank as knf
        compute_emb_fbank(np.zeros(1600, dtype=np.float32), SAMPLE_RATE)

        scaled = audio * np.float32(32768.0)
        fb = knf.OnlineFbank(_knf_emb_opts)
        fb.accept_waveform(SAMPLE_RATE, scaled)  # numpy direct — không tạo Python list
        del scaled  # giải phóng ngay
        # Pad cuối (thay vì np.pad toàn bộ audio trước)
        fb.accept_waveform(SAMPLE_RATE, np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        fb.input_finished()
        n_total_frames = fb.num_frames_ready
        all_raw_frames = np.empty((n_total_frames, 80), dtype=np.float32)
        for i in range(n_total_frames):
            all_raw_frames[i] = fb.get_frame(i)
        del fb  # giải phóng C++ internal buffer (~230MB cho 2hr)

        # --- 2. Batched: windowing + encoder + pooling (streamed per-batch) ---
        frame_shift_samples = int(SAMPLE_RATE * 0.01)  # 160
        batch_size = 64  # tăng từ 32 → giảm số batch iterations
        io_binding = self.emb_sess.io_binding()
        out_name = self.emb_sess.get_outputs()[0].name
        feat_idx = None  # lazy compute ở batch đầu

        for b_start in range(0, num_chunks, batch_size):
            b_end = min(b_start + batch_size, num_chunks)
            bs = b_end - b_start

            # Windowing: extract fbank per batch từ all_raw_frames + CMVN
            batch_fbanks = np.zeros((bs, frames_per_chunk, 80), dtype=np.float32)
            for i, c in enumerate(range(b_start, b_end)):
                fbank_start = chunk_starts[c] // frame_shift_samples
                fbank_end = min(fbank_start + frames_per_chunk, n_total_frames)
                n = fbank_end - fbank_start
                if n > 0:
                    batch_fbanks[i, :n] = all_raw_frames[fbank_start:fbank_end]
                batch_fbanks[i] -= batch_fbanks[i].mean(axis=0, keepdims=True)

            # Encoder inference
            io_binding.bind_cpu_input("fbank_features", batch_fbanks)
            io_binding.bind_output(out_name)
            self.emb_sess.run_with_iobinding(io_binding)
            out = io_binding.get_outputs()[0].numpy()

            # Lazy compute feat_idx (1 lần duy nhất)
            if feat_idx is None:
                n_feat_frames = out[0].shape[1]
                feat_idx = np.floor(
                    np.arange(n_feat_frames) * num_seg_frames / n_feat_frames
                ).astype(int)
                feat_idx = np.clip(feat_idx, 0, num_seg_frames - 1)

            # Masked pooling + Gemm NGAY (không lưu all_frame_feats)
            for i, c in enumerate(range(b_start, b_end)):
                frame_feat = out[i]
                for s in range(MAX_SPEAKERS_PER_CHUNK):
                    mask = binarized[c, :, s]
                    cm = clean_binarized[c, :, s]
                    used_mask = cm if cm.sum() > min_seg_frames else mask
                    weights = used_mask[feat_idx].astype(np.float32)

                    w = weights[np.newaxis, :]
                    v1 = w.sum() + 1e-8
                    mean = (frame_feat * w).sum(axis=1) / v1
                    dx2 = (frame_feat - mean[:, np.newaxis]) ** 2
                    v2 = (w * w).sum()
                    var = (dx2 * w).sum(axis=1) / (v1 - v2 / v1 + 1e-8)
                    std = np.sqrt(var)
                    stats = np.concatenate([mean, std])
                    embeddings[c, s, :] = stats @ emb_W.T + emb_b

            if progress_callback:
                progress_callback(25 + int(b_end / num_chunks * 60), 100)

        del all_raw_frames, io_binding  # giải phóng

        return embeddings

    def _cluster(self, all_embeddings, train_mask, segmentations,
                  max_clusters=None):
        """Exact replication of pyannote VBxClustering.__call__.
        all_embeddings: (nc, 3, dim) — ALL embeddings (non-NaN for all speakers)
        train_mask: (nc, 3) bool — filter_embeddings result for VBx training
        segmentations: (nc, frames, 3) — binarized segmentation for inactive detection
        """
        num_chunks, num_speakers, dimension = all_embeddings.shape
        train_emb = all_embeddings[train_mask]

        if len(train_emb) < 2:
            return np.zeros((num_chunks, num_speakers), dtype=np.int8)

        # AHC on L2-normed train embeddings
        train_normed = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-10)
        dendrogram = linkage(train_normed, method="centroid", metric="euclidean")
        ahc_labels = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        _, ahc_labels = np.unique(ahc_labels, return_inverse=True)

        # VBx
        emb_tf = xvec_transform(train_emb, self.plda_data)
        emb_plda = plda_transform(emb_tf, self.plda_data)
        gamma, pi = vbx_cluster(emb_plda, self.plda_data['plda_psi'],
                                 ahc_labels, Fa=self.Fa, Fb=self.Fb)

        # Soft centroids (pyannote: W.T @ train / W.sum)
        active_spk = np.where(pi > 1e-7)[0]
        if len(active_spk) == 0:
            active_spk = np.array([0])
        W = gamma[:, active_spk]
        centroids = (W.T @ train_emb) / (W.sum(axis=0, keepdims=True).T + 1e-8)

        # Optional K-Means re-clustering
        n_auto = centroids.shape[0]
        min_cl = max_clusters if max_clusters else 1
        max_cl = max_clusters if max_clusters else num_chunks * num_speakers
        if max_clusters and n_auto > max_cl:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=max_cl, n_init=3, random_state=42, copy_x=False)
            km.fit_predict(train_normed)
            centroids = np.vstack([
                train_emb[km.labels_ == k].mean(axis=0) for k in range(max_cl)])

        # Save centroids for later speaker verification (gap words)
        self.speaker_centroids = centroids.copy()

        # Distance ALL embeddings → centroids (pyannote: rearrange + cdist)
        flat_emb = all_embeddings.reshape(-1, dimension)  # (nc*3, dim)
        e2k_dist = cdist(flat_emb, centroids, metric="cosine")
        soft_clusters = (2.0 - e2k_dist).reshape(num_chunks, num_speakers, -1)

        # Suppress inactive speakers (pyannote: segmentations.data.sum(1)==0)
        const = soft_clusters.min() - 1.0
        inactive = segmentations.sum(axis=1) == 0  # (nc, 3)
        soft_clusters[inactive] = const

        # Constrained assignment (pyannote constrained_argmax)
        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def _reconstruct_and_diarize(self, activities, hard_clusters,
                                  count_swf, num_chunks, num_seg_frames,
                                  duration):
        num_clusters = int(hard_clusters.max()) + 1
        if num_clusters <= 0:
            return []

        chunk_sw = _SW(start=0.0, duration=CHUNK_DURATION, step=CHUNK_STEP)
        frame_sw = _SW(start=RF_START, duration=RF_DURATION, step=RF_STEP)

        # reconstruct: remap local→global via np.max
        # (matching pyannote: iterate 2D segmentation per chunk, not 3D index)
        clustered = np.full((num_chunks, num_seg_frames, num_clusters),
                            np.nan, dtype=np.float32)
        for c in range(num_chunks):
            segmentation = activities[c]  # (num_frames, local_speakers) — 2D
            cluster_c = hard_clusters[c]
            for k in np.unique(cluster_c):
                if k == -2:
                    continue
                selected = segmentation[:, cluster_c == k]
                clustered[c, :, k] = selected.max(axis=1) if selected.ndim > 1 else selected.ravel()

        # to_diarization: aggregate(skip_average=True, hamming=False)
        activations = _pyannote_aggregate(
            clustered, chunk_sw, frame_sw,
            hamming=False, skip_average=True, missing=0.0,
            warm_up=(0.0, 0.0))

        # Pad if needed
        _, n_spk = activations.data.shape
        max_spk = int(np.max(count_swf.data))
        if n_spk < max_spk:
            activations.data = np.pad(activations.data,
                                       ((0, 0), (0, max_spk - n_spk)))

        # Crop to common extent (exact pyannote logic)
        extent = activations.extent & count_swf.extent
        activations = activations.crop(extent, return_data=False)
        count_cropped = count_swf.crop(extent, return_data=False)

        # Top-k binarization (exclusive: cap count at 1)
        sorted_speakers = np.argsort(-activations.data, axis=-1)
        binary = np.zeros_like(activations.data)
        for t, ((_, c), speakers) in enumerate(
                zip(count_cropped, sorted_speakers)):
            c_val = min(int(c.item()), 1)  # exclusive
            for i in range(c_val):
                binary[t, speakers[i]] = 1.0

        # Binarize → segments (no pyannote at all)
        raw_segs = _binarize(binary, activations.sliding_window,
                             onset=0.5, offset=0.5,
                             min_duration_on=0.0,
                             min_duration_off=self.min_duration_off)

        # Extract segments
        segments = []
        speaker_map = {}
        speaker_counter = 0
        for start, end, speaker in raw_segs:
            if speaker not in speaker_map:
                speaker_map[speaker] = speaker_counter
                speaker_counter += 1
            segments.append({
                'start': round(start, 4),
                'end': round(end, 4),
                'speaker': speaker_map[speaker],
            })

        segments.sort(key=lambda s: s['start'])

        # Re-index centroids to match remapped speaker IDs
        if self.speaker_centroids is not None and speaker_map:
            inv_map = {v: k for k, v in speaker_map.items()}
            n_speakers = len(speaker_map)
            reindexed = np.zeros((n_speakers, self.speaker_centroids.shape[1]),
                                 dtype=np.float32)
            for new_id, old_id in inv_map.items():
                if old_id < self.speaker_centroids.shape[0]:
                    reindexed[new_id] = self.speaker_centroids[old_id]
            self.speaker_centroids = reindexed

        return segments

    def unload(self):
        """Giải phóng ONNX sessions khỏi RAM."""
        import gc
        if self.seg_sess is not None:
            del self.seg_sess
            self.seg_sess = None
        if self.emb_sess is not None:
            del self.emb_sess
            self.emb_sess = None
        self.emb_W = None
        self.emb_b = None
        self.plda_data = None
        self.speaker_centroids = None
        gc.collect()
        print("[PureORT] Model unloaded")
