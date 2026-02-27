# -*- coding: utf-8 -*-
"""
audio_analyzer.py - Module đánh giá chất lượng âm thanh và ASR-Proxy
Tích hợp: VAD, DNSMOS, ASR-Proxy cho cả offline và online models

Lưu ý sự khác biệt ASR-Proxy:
- Offline: result.ys_log_probs (log probabilities)
- Online: recognizer.ys_probs(stream) (cũng là log probabilities)
Cả hai đều cần exp() để chuyển về probability
"""

import os
import sys
import json
import math
import time
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import librosa

# PyQt imports cho threading
from PyQt6.QtCore import QThread, pyqtSignal

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DNSMOS_DIR = os.path.join(BASE_DIR, "models", "dnsmos")
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
DNSMOS_MODEL_NAME = "sig_bak_ovr.onnx"

SAMPLE_RATE = 16000
VAD_PAD_SEC = 0.6
VAD_MIN_SILENCE_MS = 300
VAD_MIN_SPEECH_MS = 250


@dataclass
class QualityMetrics:
    """Kết quả đánh giá chất lượng"""
    dnsmos_sig: float = 0.0  # Signal quality (1-5)
    dnsmos_bak: float = 0.0  # Background quality (1-5)
    dnsmos_ovrl: float = 0.0  # Overall quality (1-5)
    asr_confidence: float = 0.0  # ASR confidence (0-1)
    sample_text: str = ""  # Text sample từ ASR
    duration_analyzed: float = 0.0  # Thờigian thực tế phân tích
    num_segments: int = 0  # Số đoạn speech tìm thấy


@dataclass  
class AnalysisResult:
    """Kết quả phân tích đầy đủ"""
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    suggestions: List[str] = field(default_factory=list)
    is_ready: bool = False  # Sẵn sàng cho ASR?
    error_message: Optional[str] = None


class DNSMOSDownloader(QThread):
    """Thread download DNSMOS model"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int)
    
    def run(self):
        try:
            os.makedirs(DNSMOS_DIR, exist_ok=True)
            model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
            
            if os.path.exists(model_path):
                self.finished.emit(True, "Model đã tồn tại")
                return
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(int(downloaded * 100 / total_size), 100)
                self.progress.emit(percent)
            
            urllib.request.urlretrieve(
                DNSMOS_URL, 
                model_path + ".tmp",
                reporthook=download_progress
            )
            
            os.rename(model_path + ".tmp", model_path)
            self.finished.emit(True, "Download thành công")
            
        except Exception as e:
            self.finished.emit(False, str(e))


class AudioQualityAnalyzer:
    """
    Analyzer chính cho chất lượng âm thanh và ASR-Proxy
    
    Usage:
        analyzer = AudioQualityAnalyzer(offline_recognizer, online_recognizer)
        result = analyzer.analyze_file("path/to/file.wav")
        # hoặc
        result = analyzer.analyze_microphone(device_index=0, duration=8)
    """
    
    def __init__(self, 
                 offline_recognizer=None, 
                 online_recognizer=None,
                 use_gpu=False):
        self.offline_recognizer = offline_recognizer
        self.online_recognizer = online_recognizer
        self.use_gpu = use_gpu
        
        # VAD model (lazy load)
        self._vad_model = None
        self._vad_utils = None
        
        # DNSMOS session (lazy load)
        self._dnsmos_session = None
        
    def _load_vad(self):
        """Load Silero VAD model"""
        if self._vad_model is None:
            try:
                import torch
                self._vad_model, self._vad_utils = torch.hub.load(
                    'snakers4/silero-vad',
                    'silero_vad',
                    force_reload=False,
                    onnx=False
                )
                print("[AudioAnalyzer] Loaded Silero VAD")
            except Exception as e:
                print(f"[AudioAnalyzer] Failed to load VAD: {e}")
                raise
        return self._vad_model, self._vad_utils
    
    def _load_dnsmos(self):
        """Load DNSMOS ONNX model"""
        if self._dnsmos_session is None:
            try:
                import onnxruntime as ort
                model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
                
                if not os.path.exists(model_path):
                    print(f"[AudioAnalyzer] DNSMOS model not found at {model_path}")
                    print("[AudioAnalyzer] Please download manually or run DNSMOSDownloader")
                    return None
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self._dnsmos_session = ort.InferenceSession(model_path, providers=providers)
                print("[AudioAnalyzer] Loaded DNSMOS model")
                
            except ImportError:
                print("[AudioAnalyzer] onnxruntime not available, DNSMOS disabled")
                return None
            except Exception as e:
                print(f"[AudioAnalyzer] Failed to load DNSMOS: {e}")
                return None
        
        return self._dnsmos_session
    
    def ensure_dnsmos_model(self) -> bool:
        """Đảm bảo DNSMOS model đã tải, trả về True nếu sẵn sàng"""
        model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
        return os.path.exists(model_path)
    
    def stratified_sample(self, audio: np.ndarray, sr: int = 16000, 
                          segment_sec: int = 10) -> List[np.ndarray]:
        """
        Lấy mẫu stratified: 3 đoạn ở vị trí 15%, 50%, 85%
        Đảm bảo đại diện đầu, giữa, cuối file
        """
        if len(audio) < sr * 2:  # File quá ngắn < 2s
            return [audio]
        
        samples = []
        positions = [0.15, 0.50, 0.85]
        segment_samples = segment_sec * sr
        
        for pos in positions:
            center = int(len(audio) * pos)
            start = max(0, center - segment_samples // 2)
            end = min(len(audio), start + segment_samples)
            
            if end - start > sr:  # Ít nhất 1s
                samples.append(audio[start:end])
        
        return samples if samples else [audio]
    
    def vad_segment(self, audio: np.ndarray, sr: int = 16000,
                    padding_sec: float = VAD_PAD_SEC) -> List[np.ndarray]:
        """
        Dùng Silero VAD để cắt các đoạn speech
        Thêm padding trước/sau để không mất âm đầu/cuối
        
        Nếu VAD default không phát hiện speech (audio quá nhỏ),
        sẽ retry với threshold thấp hơn và boost amplitude.
        """
        try:
            vad_model, vad_utils = self._load_vad()
            (get_speech_timestamps, _, _, _, _) = vad_utils
            
            # Chuyển về tensor
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            
            # Lấy timestamps với settings mặc định
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                vad_model,
                sampling_rate=sr,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                min_speech_duration_ms=VAD_MIN_SPEECH_MS
            )
            
            # Retry với threshold thấp hơn nếu không tìm thấy speech
            if not speech_timestamps:
                print("[AudioAnalyzer] VAD default found no speech, retrying with lower threshold...")
                speech_timestamps = get_speech_timestamps(
                    audio_tensor,
                    vad_model,
                    sampling_rate=sr,
                    min_silence_duration_ms=200,
                    min_speech_duration_ms=150,
                    threshold=0.3  # Giảm threshold từ default 0.5 xuống 0.3
                )
            
            # Retry lần 2: boost amplitude nếu audio quá nhỏ
            if not speech_timestamps:
                max_amp = np.max(np.abs(audio))
                if max_amp > 1e-6 and max_amp < 0.5:
                    print(f"[AudioAnalyzer] Audio amplitude low (max={max_amp:.4f}), boosting and retrying VAD...")
                    boosted = audio / max_amp  # Normalize to [-1, 1]
                    boosted_tensor = torch.from_numpy(boosted.astype(np.float32))
                    speech_timestamps = get_speech_timestamps(
                        boosted_tensor,
                        vad_model,
                        sampling_rate=sr,
                        min_silence_duration_ms=200,
                        min_speech_duration_ms=150,
                        threshold=0.3
                    )
                    if speech_timestamps:
                        print(f"[AudioAnalyzer] VAD found {len(speech_timestamps)} segments after boosting")
            
            if not speech_timestamps:
                return []
            
            # Apply padding và merge gần nhau
            segments = []
            pad_samples = int(padding_sec * sr)
            
            for ts in speech_timestamps:
                start = max(0, ts['start'] - pad_samples)
                end = min(len(audio), ts['end'] + pad_samples)
                segments.append(audio[start:end])
            
            return segments
            
        except Exception as e:
            print(f"[AudioAnalyzer] VAD error: {e}")
            # Fallback: trả về cả audio nếu VAD lỗi
            return [audio]
    
    def compute_dnsmos(self, audio: np.ndarray, sr: int = 16000) -> Optional[Dict[str, float]]:
        """
        Tính DNSMOS score: SIG, BAK, OVRL
        Trả về dict hoặc None nếu lỗi
        """
        session = self._load_dnsmos()
        if session is None:
            return None
        
        try:
            # DNSMOS yêu cầu 9.01s @ 16kHz = 144160 samples
            target_len = 144160
            
            if len(audio) < target_len:
                # Pad zero
                audio_padded = np.zeros(target_len, dtype=np.float32)
                audio_padded[:len(audio)] = audio[:target_len].astype(np.float32)
            else:
                # Chỉ lấy 9.01s đầu (có thể cải thiện bằng sliding window trong tương lai)
                audio_padded = audio[:target_len].astype(np.float32)
            
            # KHÔNG normalize (peak normalization) ở đây, vì DNSMOS rất nhạy với âm lượng thật.
            # model được train với audio float [-1.0, 1.0] nên level nguyên bản là quan trọng nhất.
            
            # Input shape: (1, samples)
            input_data = audio_padded.reshape(1, -1)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            
            # Output: [SIG, BAK, OVRL] hoặc [[SIG, BAK, OVRL]]
            output_arr = outputs[0]
            if len(output_arr.shape) == 2:
                scores = output_arr[0]
            else:
                scores = output_arr
            
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = scores[0], scores[1], scores[2]
            
            # Áp dụng hàm đa thức (polynomial fit) chuẩn của Microsoft DNSMOS để map ra điểm MOS (1-5)
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
            
            sig_poly = p_sig(mos_sig_raw)
            bak_poly = p_bak(mos_bak_raw)
            ovr_poly = p_ovr(mos_ovr_raw)
            
            return {
                'SIG': float(np.clip(sig_poly, 1.0, 5.0)),
                'BAK': float(np.clip(bak_poly, 1.0, 5.0)),
                'OVRL': float(np.clip(ovr_poly, 1.0, 5.0))
            }
            
        except Exception as e:
            print(f"[AudioAnalyzer] DNSMOS error: {e}")
            return None
    
    def compute_dnsmos_average(self, audio: np.ndarray, sr: int = 16000) -> Optional[Dict[str, float]]:
        """
        Tính DNSMOS score trung bình trên toàn bộ audio bằng sliding window.
        Mỗi window 9.01s với overlap 50%.
        """
        target_len = 144160  # 9.01s @ 16kHz
        
        if len(audio) <= target_len:
            return self.compute_dnsmos(audio, sr)
        
        all_scores = []
        step = target_len // 2  # 50% overlap
        
        for start in range(0, len(audio) - target_len + 1, step):
            chunk = audio[start:start + target_len]
            score = self.compute_dnsmos(chunk, sr)
            if score:
                all_scores.append(score)
        
        if not all_scores:
            return None
        
        return {
            'SIG': float(np.mean([s['SIG'] for s in all_scores])),
            'BAK': float(np.mean([s['BAK'] for s in all_scores])),
            'OVRL': float(np.mean([s['OVRL'] for s in all_scores])),
        }
    
    def compute_asr_proxy_offline(self, audio: np.ndarray, 
                                   recognizer) -> Tuple[Optional[float], str]:
        """
        ASR-Proxy dùng offline model
        Trả về: (confidence, text)
        
        Sử dụng: stream.result.ys_log_probs
        """
        if recognizer is None:
            return None, ""
        
        try:
            import sherpa_onnx as so
            
            # Tạo stream và transcribe
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
            recognizer.decode_stream(stream)
            
            result = stream.result
            text = result.text.strip()
            
            # Lấy ys_log_probs - QUAN TRỌNG: Đây là log probabilities (âm)
            if hasattr(result, 'ys_log_probs') and result.ys_log_probs:
                log_probs = result.ys_log_probs
                # Chuyển log prob -> prob (0-1)
                mean_log_prob = float(np.mean(log_probs))
                confidence = float(np.exp(mean_log_prob))
                return confidence, text
            else:
                # Không có confidence data → return None (không đoán)
                return None, text
                
        except Exception as e:
            print(f"[AudioAnalyzer] ASR-Proxy offline error: {e}")
            return None, ""
    
    def compute_asr_proxy_online(self, audio: np.ndarray,
                                  recognizer) -> Tuple[Optional[float], str]:
        """
        ASR-Proxy dùng online/streaming model
        Trả về: (confidence, text)
        
        Sử dụng: recognizer.get_result_all(stream).ys_probs
        
        Lưu ý: ys_probs cũng là LOG PROBABILITIES (âm), không phải 0-1!
        """
        if recognizer is None:
            return None, ""
        
        try:
            import sherpa_onnx as so
            
            # Tạo streaming session
            stream = recognizer.create_stream()
            
            # Feed audio theo chunks
            chunk_size = int(0.1 * SAMPLE_RATE)  # 100ms
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                stream.accept_waveform(SAMPLE_RATE, chunk.astype(np.float32))
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
            
            # IMPORTANT: Gọi input_finished() để flush buffer cuối
            stream.input_finished()
            
            # Flush remaining
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            
            # Lấy result - API đúng: get_result(stream) trả về str
            text = recognizer.get_result(stream)
            
            # Lấy ys_probs - API đúng: ys_probs(stream) trả về List[float]
            ys_probs = recognizer.ys_probs(stream)
            
            # Tính confidence - ys_probs là LOG PROBABILITIES
            if ys_probs and len(ys_probs) > 0:
                mean_log_prob = float(np.mean(ys_probs))
                confidence = float(np.exp(mean_log_prob))
                return confidence, text.strip()
            else:
                # Không có confidence data → return None
                return None, text.strip()
                
        except Exception as e:
            print(f"[AudioAnalyzer] ASR-Proxy online error: {e}")
            return None, ""
    
    def analyze_file(self, file_path: str, 
                     use_offline_asr: bool = True) -> AnalysisResult:
        """
        Phân tích file âm thanh
        
        Args:
            file_path: Đường dẫn file audio
            use_offline_asr: True=dùng offline model (chính xác hơn), 
                           False=dùng online model (nếu có)
        
        Returns:
            AnalysisResult với metrics và suggestions
        """
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, res_type="soxr_vhq")
            
            # Stratified sampling
            samples = self.stratified_sample(audio)
            
            all_dnsmos_scores = []
            all_asr_confidences = []
            sample_texts = []
            total_segments = 0
            total_duration = 0
            
            # Phân tích từng sample
            vad_found_speech = False
            for sample in samples:
                # VAD segment
                segments = self.vad_segment(sample)
                if not segments:
                    continue
                
                vad_found_speech = True
                total_segments += len(segments)
                
                # DNSMOS: tính trên từng segment riêng để tránh ảnh hưởng bởi việc concat
                for seg in segments:
                    if len(seg) >= SAMPLE_RATE * 0.5:  # Chỉ tính segment >= 0.5s
                        dnsmos = self.compute_dnsmos_average(seg)
                        if dnsmos:
                            all_dnsmos_scores.append(dnsmos)
                
                # Ghép tất cả segments cho ASR-Proxy
                main_segment = np.concatenate(segments)
                if len(main_segment) < SAMPLE_RATE * 0.5:
                    continue
                
                total_duration += len(main_segment) / SAMPLE_RATE
                
                # ASR-Proxy - ưu tiên offline cho file analysis (chính xác hơn)
                conf, text = None, ""
                if use_offline_asr and self.offline_recognizer:
                    conf, text = self.compute_asr_proxy_offline(main_segment, 
                                                                 self.offline_recognizer)
                elif self.online_recognizer:
                    conf, text = self.compute_asr_proxy_online(main_segment,
                                                                self.online_recognizer)
                
                if conf is not None:
                    all_asr_confidences.append(conf)
                if text and not sample_texts:  # Chỉ lấy text từ segment đầu
                    sample_texts.append(text)
            
            # Fallback: Nếu VAD không phát hiện speech, tính DNSMOS trực tiếp trên raw samples
            if not vad_found_speech and not all_dnsmos_scores:
                print("[AudioAnalyzer] VAD found no speech in any sample, computing DNSMOS on raw audio...")
                for sample in samples:
                    if len(sample) >= SAMPLE_RATE * 0.5:
                        dnsmos = self.compute_dnsmos_average(sample)
                        if dnsmos:
                            all_dnsmos_scores.append(dnsmos)
                            total_duration += len(sample) / SAMPLE_RATE
                
                # ASR-Proxy trên raw audio (dùng sample đầu tiên)
                if samples:
                    asr_sample = samples[0]
                    conf, text = None, ""
                    if use_offline_asr and self.offline_recognizer:
                        conf, text = self.compute_asr_proxy_offline(asr_sample, 
                                                                     self.offline_recognizer)
                    elif self.online_recognizer:
                        conf, text = self.compute_asr_proxy_online(asr_sample,
                                                                    self.online_recognizer)
                    if conf is not None:
                        all_asr_confidences.append(conf)
                    if text:
                        sample_texts.append(text)
            
            # Tính trung bình
            metrics = QualityMetrics()
            
            if all_dnsmos_scores:
                metrics.dnsmos_sig = np.mean([s['SIG'] for s in all_dnsmos_scores])
                metrics.dnsmos_bak = np.mean([s['BAK'] for s in all_dnsmos_scores])
                metrics.dnsmos_ovrl = np.mean([s['OVRL'] for s in all_dnsmos_scores])
            
            if all_asr_confidences:
                metrics.asr_confidence = np.mean(all_asr_confidences)
            
            metrics.sample_text = sample_texts[0] if sample_texts else ""
            metrics.duration_analyzed = total_duration
            metrics.num_segments = total_segments
            
            # Tạo suggestions
            suggestions = self._generate_suggestions(metrics)
            
            # Xác định ready - chỉ dựa ASR nếu không có DNSMOS
            if all_dnsmos_scores:
                is_ready = (metrics.asr_confidence >= 0.60 and 
                           metrics.dnsmos_ovrl >= 2.5)
            else:
                is_ready = (metrics.asr_confidence >= 0.60)
            
            return AnalysisResult(
                metrics=metrics,
                suggestions=suggestions,
                is_ready=is_ready
            )
            
        except Exception as e:
            return AnalysisResult(
                error_message=f"Lỗi phân tích: {str(e)}"
            )
    
    def analyze_microphone(self, audio: np.ndarray) -> AnalysisResult:
        """
        Phân tích audio từ microphone
        
        Args:
            audio: Audio array từ microphone
            
        Note: Tự động chọn recognizer theo thứ tự: online -> offline
        
        Returns:
            AnalysisResult
        """
        try:
            # Resample nếu cần
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # VAD segment
            segments = self.vad_segment(audio)
            vad_found = bool(segments)
            
            if not segments:
                # Fallback: dùng raw audio nếu VAD không phát hiện speech
                print("[AudioAnalyzer] Mic VAD found no speech, using raw audio as fallback")
                if len(audio) >= SAMPLE_RATE * 0.5:
                    segments = [audio]  # Dùng raw audio
                else:
                    return AnalysisResult(
                        error_message="Giọng nói quá ngắn. Vui lòng nói lâu hơn."
                    )
            
            # Kiểm tra tổng độ dài
            total_len = sum(len(s) for s in segments)
            if total_len < SAMPLE_RATE * 0.5:
                return AnalysisResult(
                    error_message="Giọng nói quá ngắn. Vui lòng nói lâu hơn."
                )
            
            # DNSMOS: tính trên từng segment riêng
            all_dnsmos = []
            for seg in segments:
                if len(seg) >= SAMPLE_RATE * 0.3:  # >= 0.3s
                    d = self.compute_dnsmos_average(seg)
                    if d:
                        all_dnsmos.append(d)
            
            dnsmos = None
            if all_dnsmos:
                dnsmos = {
                    'SIG': float(np.mean([d['SIG'] for d in all_dnsmos])),
                    'BAK': float(np.mean([d['BAK'] for d in all_dnsmos])),
                    'OVRL': float(np.mean([d['OVRL'] for d in all_dnsmos])),
                }
            
            # Ghép segments cho ASR-Proxy
            main_segment = np.concatenate(segments)
            
            # ASR-Proxy - tự động chọn recognizer có sẵn
            # Note: conf = None nghĩa là không đo được (không phải confidence thấp)
            conf, text = None, ""
            if self.online_recognizer:
                conf, text = self.compute_asr_proxy_online(main_segment,
                                                            self.online_recognizer)
            elif self.offline_recognizer:
                conf, text = self.compute_asr_proxy_offline(main_segment,
                                                             self.offline_recognizer)
            
            # Tạo metrics
            metrics = QualityMetrics()
            if dnsmos:
                metrics.dnsmos_sig = dnsmos['SIG']
                metrics.dnsmos_bak = dnsmos['BAK']
                metrics.dnsmos_ovrl = dnsmos['OVRL']
            
            # Convert None → 0.0 cho UI (None = không đo được, 0.0 = hiển thị 0%)
            metrics.asr_confidence = conf if conf is not None else 0.0
            metrics.sample_text = text
            metrics.duration_analyzed = len(main_segment) / SAMPLE_RATE
            metrics.num_segments = len(segments)
            
            # Suggestions
            suggestions = self._generate_suggestions(metrics)
            # Xác định ready - chỉ dựa ASR nếu không có DNSMOS
            if dnsmos:
                is_ready = (metrics.asr_confidence >= 0.60 and 
                           metrics.dnsmos_ovrl >= 2.5)
            else:
                is_ready = (metrics.asr_confidence >= 0.60)
            
            return AnalysisResult(
                metrics=metrics,
                suggestions=suggestions,
                is_ready=is_ready
            )
            
        except Exception as e:
            return AnalysisResult(
                error_message=f"Lỗi phân tích: {str(e)}"
            )
    
    def _generate_suggestions(self, metrics: QualityMetrics) -> List[str]:
        """Tạo gợi ý cải thiện dựa trên metrics"""
        suggestions = []
        
        # BAK (Background noise)
        if metrics.dnsmos_bak < 2.5:
            suggestions.append("🔴 Nhiễu nền cao: Tắt quạt, điều hòa, hoặc chuyển nơi yên tĩnh hơn")
        elif metrics.dnsmos_bak < 3.5:
            suggestions.append("🟡 Có nhiễu nền: Cố gắng giảm âm thanh xung quanh")
        
        # SIG (Speech signal)
        if metrics.dnsmos_sig < 2.5:
            suggestions.append("🔴 Giọng nói kém: Đưa microphone gần miệng hơn (15-20cm)")
        elif metrics.dnsmos_sig < 3.5:
            suggestions.append("🟡 Chất lượng giọng nói trung bình: Điều chỉnh vị trí microphone")
        
        # OVRL (Overall)
        if metrics.dnsmos_ovrl < 2.5:
            suggestions.append("🔴 Chất lượng tổng thể kém: Kiểm tra lại thiết bị và môi trường")
        
        # ASR Confidence
        if metrics.asr_confidence < 0.60:
            suggestions.append("🔴 ASR khó nhận diện: Nói chậm rãi, phát âm rõ ràng từng từ")
        elif metrics.asr_confidence < 0.75:
            suggestions.append("🟡 ASR có thể sai sót: Kiểm tra kết quả sau khi nhận dạng")
        
        # Nếu tất cả tốt
        if not suggestions:
            suggestions.append("✅ Chất lượng tốt! Sẵn sàng cho nhận dạng.")
        
        return suggestions
    
    @staticmethod
    def get_confidence_label(confidence: float) -> Tuple[str, str]:
        """
        Trả về (label, color) dựa trên confidence
        """
        if confidence >= 0.85:
            return "Xuất sắc", "#28a745"  # Xanh lá
        elif confidence >= 0.75:
            return "Tốt", "#5cb85c"  # Xanh nhạt
        elif confidence >= 0.60:
            return "Trung bình", "#ffc107"  # Vàng
        else:
            return "Kém", "#dc3545"  # Đỏ
    
    @staticmethod
    def get_dnsmos_label(score: float) -> Tuple[str, str]:
        """
        Trả về (label, color) dựa trên DNSMOS score (1-5)
        """
        if score >= 4.0:
            return "Tốt", "#28a745"
        elif score >= 3.0:
            return "Khá", "#5cb85c"
        elif score >= 2.0:
            return "Trung bình", "#ffc107"
        else:
            return "Kém", "#dc3545"


class AnalysisThread(QThread):
    """Thread để phân tích không block UI"""
    finished = pyqtSignal(AnalysisResult)
    progress = pyqtSignal(str, int)  # message, percent
    
    def __init__(self, analyzer: AudioQualityAnalyzer, 
                 file_path: str = None,
                 audio: np.ndarray = None,
                 use_offline: bool = True):
        super().__init__()
        self.analyzer = analyzer
        self.file_path = file_path
        self.audio = audio
        self.use_offline = use_offline
    
    def run(self):
        try:
            if self.file_path:
                self.progress.emit("Đang phân tích file...", 30)
                result = self.analyzer.analyze_file(self.file_path, use_offline_asr=self.use_offline)
            else:
                self.progress.emit("Đang phân tích audio...", 30)
                result = self.analyzer.analyze_microphone(self.audio)
            
            self.progress.emit("Hoàn tất", 100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit(AnalysisResult(
                error_message=f"Lỗi: {str(e)}"
            ))


# Utility functions
def check_dnsmos_model_exists() -> bool:
    """Kiểm tra DNSMOS model đã tải chưa"""
    model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
    return os.path.exists(model_path)


def download_dnsmos_model_sync() -> bool:
    """Tải DNSMOS model đồng bộ (blocking)"""
    try:
        os.makedirs(DNSMOS_DIR, exist_ok=True)
        model_path = os.path.join(DNSMOS_DIR, DNSMOS_MODEL_NAME)
        
        if os.path.exists(model_path):
            return True
        
        print("[AudioAnalyzer] Downloading DNSMOS model...")
        urllib.request.urlretrieve(DNSMOS_URL, model_path)
        print("[AudioAnalyzer] DNSMOS model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"[AudioAnalyzer] Failed to download DNSMOS: {e}")
        return False


if __name__ == "__main__":
    # Test script
    print("Audio Quality Analyzer Test")
    print("="*60)
    
    # Check DNSMOS
    if not check_dnsmos_model_exists():
        print("DNSMOS model not found. Downloading...")
        download_dnsmos_model_sync()
    
    # Tạo analyzer (không cần ASR model để test DNSMOS và VAD)
    analyzer = AudioQualityAnalyzer()
    
    # Test VAD
    test_file = r"D:\App\asr-vn\test_2min_wpe_3.wav"
    if os.path.exists(test_file):
        print(f"\nTesting with: {test_file}")
        result = analyzer.analyze_file(test_file, use_offline_asr=False)
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        else:
            print(f"\nResults:")
            print(f"  DNSMOS SIG: {result.metrics.dnsmos_sig:.2f}")
            print(f"  DNSMOS BAK: {result.metrics.dnsmos_bak:.2f}")
            print(f"  DNSMOS OVRL: {result.metrics.dnsmos_ovrl:.2f}")
            print(f"  ASR Confidence: {result.metrics.asr_confidence:.2%}")
            print(f"  Sample text: {result.metrics.sample_text[:50]}...")
            print(f"\nSuggestions:")
            for s in result.suggestions:
                print(f"  {s}")
