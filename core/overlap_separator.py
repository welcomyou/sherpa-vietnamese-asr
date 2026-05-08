"""Overlap speaker separator — Phase 1 POC.

Kiến trúc:
  1. Input: audio + diarization segments (global speaker labels) + overlap_regions
  2. Cho mỗi overlap region:
     a. participants = set speakers active trong region (từ segments)
     b. Nếu đúng 2 participants → Conv-TasNet 2-speaker separate
     c. Extract CAM++ embedding cho mỗi stream
     d. Hungarian match với centroids của 2 participants → gán stream ↔ speaker
     e. Ghép context audio trước/sau (clean solo của chính speaker đó) cho ASR
        để compensate ngữ cảnh ngắn của region overlap
  3. Output: dict với mỗi region:
     { (start, end, spk_A, spk_B): {
         'audio_A': np.ndarray (context_before + clean_A + context_after),
         'audio_B': np.ndarray (same format),
         'real_start_A': offset of overlap start within audio_A (sec)
         'real_end_A': offset of overlap end,
         'real_start_B', 'real_end_B': same for B
       } }

Downstream (asr_engine) sẽ:
  - ASR từng audio_A/B → word timestamps
  - Map word timestamp qua offset → giữ lại words trong [real_start, real_end]
  - Tạo 2 segment parallel trong .asr.json với overlap=True
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence, Any

SAMPLE_RATE = 16000
CONTEXT_SEC_DEFAULT = 3.0
MIN_REGION_SEC = 0.4     # region ngắn hơn → skip (ROI thấp, artifact cao)
MIN_REF_SEC = 1.0        # cần >= 1s clean audio của speaker để lấy centroid reliable
MIN_OVERLAP_SEC = 1.0    # filter backchannel FP — pyannote thường detect "dạ/ạ/ok" <1s
                         # là overlap (93% false-positive trên VNPT phone call).
                         # min=1.0s giữ được real overlap + loại backchannel.
                         # Test WER: 30M giảm 19.2%, 68M giảm 5.0% khi bật filter.
FADE_MS = 15             # fade ở boundary khi ghép context + separated


class OverlapSeparator:
    """Per-region speech separation cho vùng overlap 2-người nói.

    Backend: Conv-TasNet 16k ONNX Runtime (welcomyou/convtasnet-libri2mix-16k-onnx,
    gốc JorisCos/ConvTasNet_Libri2Mix_sepclean_16k). Output 2 streams + Hungarian
    matching via CAM++ embedding để gán stream ↔ speaker.

    Pure ORT — không cần PyTorch / asteroid ở runtime. Speed tương đương
    PyTorch, SNR ~62 dB so với output PyTorch (xem convert_onnx/export_convtasnet_onnx.py).

    Note: USEF-TSE backend đã loại bỏ (xem temp/usef/USEF_TSE_JOURNEY.md cho
    chi tiết — Conv-TasNet win rõ rệt trên VNPT phone call: +12 dB vs +2 dB
    SI-SDRi, 10x nhanh hơn).
    """

    def __init__(self,
                 campp_onnx_path: Optional[str] = None,
                 convtasnet_onnx_path: Optional[str] = None,
                 num_threads: int = 4,
                 context_sec: float = CONTEXT_SEC_DEFAULT):
        self.num_threads = num_threads
        self.context_sec = context_sec
        self.context_samples = int(context_sec * SAMPLE_RATE)
        self.fade_n = int(FADE_MS / 1000.0 * SAMPLE_RATE)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.campp_path = campp_onnx_path or os.path.join(
            base, "models", "campp-3dspeaker", "campplus_cn_en_common_200k.onnx")
        self.convtasnet_path = convtasnet_onnx_path or os.path.join(
            base, "models", "convtasnet-libri2mix-16k", "convtasnet_16k.onnx")
        self._cam_sess = None
        self._convtasnet = None   # lazy ORT session
        self._fbank_opts = None

    # ------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------
    def _init_cam(self):
        if self._cam_sess is not None:
            return
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self._cam_sess = ort.InferenceSession(
            self.campp_path, opts, providers=['CPUExecutionProvider'])
        import kaldi_native_fbank as knf
        self._knf = knf
        fopts = knf.FbankOptions()
        fopts.frame_opts.samp_freq = SAMPLE_RATE
        fopts.frame_opts.dither = 0.0
        fopts.frame_opts.snip_edges = False
        fopts.frame_opts.window_type = "povey"
        fopts.mel_opts.num_bins = 80
        self._fbank_opts = fopts

    def _init_convtasnet(self):
        if self._convtasnet is not None:
            return
        import onnxruntime as ort
        if not os.path.exists(self.convtasnet_path):
            raise FileNotFoundError(
                f"Conv-TasNet ONNX không tìm thấy: {self.convtasnet_path}\n"
                "Chạy: python build-portable/prepare_offline_build.py để tải về."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        opts.optimized_model_filepath = self.convtasnet_path + ".opt"
        self._convtasnet = ort.InferenceSession(
            self.convtasnet_path, opts, providers=["CPUExecutionProvider"])
        # Warmup
        self._convtasnet.run(None, {"mixture": np.zeros((1, SAMPLE_RATE), dtype=np.float32)})

    # ------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------
    def compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """CAM++ 192-dim embedding trên audio (mono 16kHz float32).

        Return None nếu audio < 300ms (quá ngắn cho fbank).
        """
        self._init_cam()
        if len(audio) < int(0.3 * SAMPLE_RATE):
            return None
        f = self._knf.OnlineFbank(self._fbank_opts)
        f.accept_waveform(SAMPLE_RATE, audio.astype(np.float32).tolist())
        f.input_finished()
        n = f.num_frames_ready
        if n < 10:
            return None
        feats = np.stack([f.get_frame(i) for i in range(n)]).astype(np.float32)
        feats -= feats.mean(axis=0, keepdims=True)
        emb = self._cam_sess.run(['embs'], {'feats': feats[None, :, :]})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)

    # ------------------------------------------------------------
    # Centroid per speaker from clean solo segments
    # ------------------------------------------------------------
    def compute_centroids(self,
                          audio: np.ndarray,
                          segments: Sequence[Any],
                          overlap_regions: Sequence[Tuple[float, float]]
                          ) -> Dict[int, np.ndarray]:
        """Tính centroid embedding cho mỗi speaker_id.

        segments: iterable of objects with .start, .end, .speaker attributes OR dicts
                  with 'start', 'end', 'speaker' keys (both supported).
        Chỉ dùng segments KHÔNG trùng overlap region để lấy embedding sạch.
        """
        def _get(s, attr, default=None):
            if isinstance(s, dict):
                return s.get(attr, default)
            return getattr(s, attr, default)

        overlap_sorted = sorted(overlap_regions)

        def _intersects_overlap(s, e):
            for os_, oe_ in overlap_sorted:
                if oe_ < s:
                    continue
                if os_ > e:
                    break
                if max(s, os_) < min(e, oe_):
                    return True
            return False

        centroid_embs: Dict[int, List[np.ndarray]] = {}
        for seg in segments:
            s = float(_get(seg, 'start', 0))
            e = float(_get(seg, 'end', s))
            spk = int(_get(seg, 'speaker', -1))
            if spk < 0 or (e - s) < MIN_REF_SEC:
                continue
            if _intersects_overlap(s, e):
                continue
            a_s = int(s * SAMPLE_RATE)
            a_e = min(int(e * SAMPLE_RATE), len(audio))
            emb = self.compute_embedding(audio[a_s:a_e])
            if emb is not None:
                centroid_embs.setdefault(spk, []).append(emb)

        centroids: Dict[int, np.ndarray] = {}
        for spk, embs in centroid_embs.items():
            c = np.mean(embs, axis=0)
            n = np.linalg.norm(c)
            if n > 1e-10:
                c /= n
            centroids[spk] = c.astype(np.float32)
        return centroids

    # ------------------------------------------------------------
    # Participants in a region (from segments)
    # ------------------------------------------------------------
    @staticmethod
    def participants_in_region(region: Tuple[float, float],
                               segments: Sequence[Any]) -> List[int]:
        def _get(s, attr, default=None):
            if isinstance(s, dict):
                return s.get(attr, default)
            return getattr(s, attr, default)
        t_s, t_e = region
        parts = set()
        for seg in segments:
            s = float(_get(seg, 'start', 0))
            e = float(_get(seg, 'end', s))
            spk = int(_get(seg, 'speaker', -1))
            if spk < 0:
                continue
            if max(s, t_s) < min(e, t_e):
                parts.add(spk)
        return sorted(parts)

    # ------------------------------------------------------------
    # Find closest clean segment of a speaker (for context)
    # ------------------------------------------------------------
    @staticmethod
    def _closest_clean_segment(audio_len_sec: float,
                               segments: Sequence[Any],
                               overlap_regions: Sequence[Tuple[float, float]],
                               spk: int,
                               target_t: float,
                               direction: str) -> Optional[Tuple[float, float]]:
        """Tìm segment clean (non-overlap) của speaker gần nhất TRƯỚC hoặc SAU target_t.

        direction='before' → end <= target_t, pick segment có end lớn nhất (gần target)
        direction='after'  → start >= target_t, pick segment có start nhỏ nhất
        """
        def _get(s, attr, default=None):
            if isinstance(s, dict):
                return s.get(attr, default)
            return getattr(s, attr, default)

        def _intersects_overlap(s, e):
            for os_, oe_ in overlap_regions:
                if max(s, os_) < min(e, oe_):
                    return True
            return False

        candidates = []
        for seg in segments:
            s = float(_get(seg, 'start', 0))
            e = float(_get(seg, 'end', s))
            if int(_get(seg, 'speaker', -1)) != spk:
                continue
            if _intersects_overlap(s, e):
                continue
            if direction == 'before' and e <= target_t:
                candidates.append((s, e))
            elif direction == 'after' and s >= target_t:
                candidates.append((s, e))
        if not candidates:
            return None
        if direction == 'before':
            return max(candidates, key=lambda x: x[1])
        else:
            return min(candidates, key=lambda x: x[0])

    # ------------------------------------------------------------
    # Separate one overlap region (Conv-TasNet)
    # ------------------------------------------------------------
    def separate_region(self,
                        audio: np.ndarray,
                        region: Tuple[float, float],
                        participants: Sequence[int],
                        centroids: Dict[int, np.ndarray],
                        ) -> Optional[Dict[int, np.ndarray]]:
        """Conv-TasNet tách 2 streams, Hungarian match stream ↔ speaker via
        centroid embedding. Trả dict {speaker_id: separated_audio}.

        Return None nếu region quá ngắn hoặc không đủ 2 participants.
        """
        if len(participants) != 2:
            return None
        if participants[0] not in centroids or participants[1] not in centroids:
            return None
        t_s, t_e = region
        if t_e - t_s < MIN_REGION_SEC:
            return None
        a_s = int(t_s * SAMPLE_RATE)
        a_e = min(int(t_e * SAMPLE_RATE), len(audio))
        region_audio = audio[a_s:a_e]
        if len(region_audio) < int(MIN_REGION_SEC * SAMPLE_RATE):
            return None

        # Conv-TasNet forward (ORT)
        self._init_convtasnet()
        inp = region_audio.astype(np.float32)[np.newaxis]  # (1, T)
        ests = self._convtasnet.run(None, {"mixture": inp})[0][0]  # (2, T)
        # Crop output to input length (Conv-TasNet may pad output by a few samples)
        ests = ests[..., :len(region_audio)]

        # Rescale each stream by mixture peak (SI-SDR loss scale-invariant → output scale tùy ý)
        mix_peak = float(np.abs(region_audio).max())
        if mix_peak < 1e-6:
            return None
        for j in range(ests.shape[0]):
            p = float(np.abs(ests[j]).max())
            if p > 0:
                ests[j] = ests[j] * (mix_peak * 0.9 / p)

        # Embedding + Hungarian match
        e0 = self.compute_embedding(ests[0])
        e1 = self.compute_embedding(ests[1])
        if e0 is None or e1 is None:
            # Fallback: best-guess (stream idx → participant idx)
            return {participants[0]: ests[0], participants[1]: ests[1]}
        ps = list(participants)
        try:
            from scipy.optimize import linear_sum_assignment
            cost = np.array([
                [1.0 - float(e0 @ centroids[ps[0]]), 1.0 - float(e0 @ centroids[ps[1]])],
                [1.0 - float(e1 @ centroids[ps[0]]), 1.0 - float(e1 @ centroids[ps[1]])],
            ])
            row_ind, col_ind = linear_sum_assignment(cost)
            out: Dict[int, np.ndarray] = {}
            for stream_idx, pair_idx in zip(row_ind, col_ind):
                out[ps[pair_idx]] = ests[stream_idx]
            return out
        except Exception:
            # Fallback: check 2 permutations by cost sum manually
            s_id = float(e0 @ centroids[ps[0]]) + float(e1 @ centroids[ps[1]])
            s_sw = float(e0 @ centroids[ps[1]]) + float(e1 @ centroids[ps[0]])
            if s_id >= s_sw:
                return {ps[0]: ests[0], ps[1]: ests[1]}
            else:
                return {ps[1]: ests[0], ps[0]: ests[1]}

    # ------------------------------------------------------------
    # Build audio with context for ASR
    # ------------------------------------------------------------
    def build_context_audio(self,
                             audio: np.ndarray,
                             audio_len_sec: float,
                             segments: Sequence[Any],
                             overlap_regions: Sequence[Tuple[float, float]],
                             region: Tuple[float, float],
                             spk: int,
                             separated: np.ndarray
                             ) -> Tuple[np.ndarray, float, float]:
        """Ghép: context_before + separated + context_after → audio cho ASR.

        Return: (concat_audio, real_start_sec, real_end_sec)
                real_start/end = offset (giây) trong concat_audio tương ứng với
                                 overlap region. Dùng để filter word timestamps về sau.
        """
        ctx = self.context_sec
        # Tìm clean segment gần nhất trước region
        before = self._closest_clean_segment(
            audio_len_sec, segments, overlap_regions, spk,
            target_t=region[0], direction='before')
        after = self._closest_clean_segment(
            audio_len_sec, segments, overlap_regions, spk,
            target_t=region[1], direction='after')

        # Extract audio chunks (truncate to ctx seconds)
        chunks = []
        real_start = 0.0

        if before is not None:
            b_s, b_e = before
            b_s = max(b_s, b_e - ctx)   # lấy ctx giây cuối của segment before
            a_s = int(b_s * SAMPLE_RATE)
            a_e = min(int(b_e * SAMPLE_RATE), len(audio))
            if a_e > a_s:
                chunks.append(audio[a_s:a_e])
                real_start += (a_e - a_s) / SAMPLE_RATE

        # Separated audio in middle
        chunks.append(separated.astype(np.float32))
        real_end = real_start + len(separated) / SAMPLE_RATE

        if after is not None:
            a_s_t, a_e_t = after
            a_e_t = min(a_e_t, a_s_t + ctx)   # lấy ctx giây đầu của segment after
            a_s = int(a_s_t * SAMPLE_RATE)
            a_e = min(int(a_e_t * SAMPLE_RATE), len(audio))
            if a_e > a_s:
                chunks.append(audio[a_s:a_e])

        # Concat với fade-in/out giữa các chunks để tránh click
        result = self._concat_with_fade(chunks)
        return result, real_start, real_end

    def _concat_with_fade(self, chunks: List[np.ndarray]) -> np.ndarray:
        """Concat các chunks, fade ngắn ở boundary để tránh click/pop."""
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0].astype(np.float32)
        fn = self.fade_n
        fade_in = np.linspace(0, 1, fn, dtype=np.float32)
        fade_out = np.linspace(1, 0, fn, dtype=np.float32)
        out_parts = []
        for i, ch in enumerate(chunks):
            ch = ch.astype(np.float32).copy()
            if i > 0 and len(ch) > fn:
                ch[:fn] *= fade_in
            if i < len(chunks) - 1 and len(ch) > fn:
                ch[-fn:] *= fade_out
            out_parts.append(ch)
        return np.concatenate(out_parts)

    # ------------------------------------------------------------
    # Full pipeline entry point — build per-speaker audio for each overlap region
    # ------------------------------------------------------------
    def process(self,
                audio: np.ndarray,
                segments: Sequence[Any],
                overlap_regions: Sequence[Tuple[float, float]],
                progress_callback=None
                ) -> List[Dict]:
        """Process all overlap regions.

        Returns list of dicts:
          {
            'start': float, 'end': float,
            'participants': [spk_X, spk_Y],
            'audio_per_speaker': {spk_X: np.ndarray, spk_Y: np.ndarray},
            'real_start_per_speaker': {spk_X: float, spk_Y: float},
            'real_end_per_speaker':   {spk_X: float, spk_Y: float},
          }

        Regions không đủ điều kiện (participants != 2, thiếu centroid, region <0.4s)
        sẽ bị skip (không có trong output).

        Filter MIN_OVERLAP_SEC=1.0s: loại bỏ backchannel FP (dạ/ạ/ok <1s). Test
        trên VNPT phone call (1 thật / 14 detect) cho thấy 93% overlap <1s là
        false-positive backchannel, filter giúp giảm cpWER 5-19% relative.
        """
        if not overlap_regions:
            return []
        # Filter backchannel false-positive (trên VNPT phone call giảm WER 5-19%)
        filtered_regions = [r for r in overlap_regions if (r[1] - r[0]) >= MIN_OVERLAP_SEC]
        if len(filtered_regions) < len(overlap_regions):
            import logging
            logging.getLogger(__name__).info(
                f"[OverlapSep] Filtered {len(overlap_regions) - len(filtered_regions)}/"
                f"{len(overlap_regions)} short regions (<{MIN_OVERLAP_SEC}s, likely backchannel)")
        overlap_regions = filtered_regions
        if not overlap_regions:
            return []

        audio_len_sec = len(audio) / SAMPLE_RATE
        centroids = self.compute_centroids(audio, segments, overlap_regions)

        results: List[Dict] = []
        total = len(overlap_regions)
        for i, region in enumerate(overlap_regions):
            if progress_callback:
                try:
                    progress_callback(int(i / max(1, total) * 100))
                except Exception:
                    pass

            participants = self.participants_in_region(region, segments)
            if len(participants) != 2:
                continue
            if not all(p in centroids for p in participants):
                continue

            streams = self.separate_region(audio, region, participants, centroids)
            if streams is None:
                continue

            audio_per_spk = {}
            real_start_per = {}
            real_end_per = {}
            for spk, sep_audio in streams.items():
                concat, rs, re = self.build_context_audio(
                    audio, audio_len_sec, segments, overlap_regions,
                    region, spk, sep_audio)
                audio_per_spk[spk] = concat
                real_start_per[spk] = rs
                real_end_per[spk] = re

            results.append({
                'start': region[0], 'end': region[1],
                'participants': participants,
                'audio_per_speaker': audio_per_spk,
                'real_start_per_speaker': real_start_per,
                'real_end_per_speaker': real_end_per,
            })

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass
        return results

    # ------------------------------------------------------------
    # Helper: filter words từ ASR output vào window [real_start, real_end]
    # ------------------------------------------------------------
    @staticmethod
    def filter_words_in_window(words: Sequence[Dict],
                                real_start: float,
                                real_end: float,
                                real_offset: float = 0.0
                                ) -> List[Dict]:
        """Filter ASR words chỉ giữ những chữ có midpoint nằm trong window
        [real_start, real_end] (theo concat time).

        real_offset: nếu cần shift word times về global time trong file gốc,
                      caller truyền offset (global_start_of_region - real_start).
        """
        out = []
        for w in words:
            ws = float(w.get('start', 0))
            we = float(w.get('end', ws))
            mid = (ws + we) / 2.0
            if real_start <= mid <= real_end:
                new_w = dict(w)
                new_w['start'] = ws + real_offset
                new_w['end'] = we + real_offset
                out.append(new_w)
        return out
