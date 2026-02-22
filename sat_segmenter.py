"""
SAT (Segment Any Text) Sentence and Paragraph Segmenter
Sử dụng sat-12l-sm model để tách câu và tách đoạn văn bản tiếng Việt
"""

import os
import re
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SATSegmenter:
    """
    Segmenter sử dụng SAT (Segment Any Text) model để tách câu và đoạn
    Model: sat-12l-sm - optimized cho sentence segmentation
    Language: vi (Vietnamese)
    Style: ted (phù hợp với văn nói/ASR transcripts)
    
    Để build .exe offline: Đặt model vào models/sat-12l-sm/
    """
    
    def __init__(
        self, 
        model_name: str = "sat-12l-sm",
        language: str = "vi",
        style_or_domain: str = "ted",
        threshold: float = 0.3,
        paragraph_threshold: float = 0.3,
        device: str = None,
        local_model_path: str = None  # Đường dẫn local cho offline mode
    ):
        """
        Khởi tạo SAT Segmenter
        
        Args:
            model_name: Tên model (mặc định: 'sat-12l-sm')
            language: Mã ngôn ngữ ISO 639-1 (mặc định: 'vi')
            style_or_domain: Phong cách/domain (mặc định: 'ted')
            threshold: Ngưỡng tách câu (mặc định: 0.5)
            paragraph_threshold: Ngưỡng tách đoạn (mặc định: 0.5)
            device: Thiết bị chạy ('cpu', 'cuda', hoặc None)
            local_model_path: Đường dẫn local đến model (cho offline mode).
                             Mặc định: tự động tìm trong models/sat-12l-sm/
        """
        self.model_name = model_name
        self.language = language
        self.style_or_domain = style_or_domain
        self.threshold = threshold
        self.paragraph_threshold = paragraph_threshold
        self.device = device
        self._sat_model = None
        self._initialized = False
        
        # Xác định đường dẫn model
        if local_model_path and os.path.exists(local_model_path):
            self._model_path = local_model_path
        else:
            # Tự động tìm trong thư mục models/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_dir, "models", model_name)
            if os.path.exists(os.path.join(local_path, "model_optimized.onnx")) or \
               os.path.exists(os.path.join(local_path, "model.safetensors")):
                self._model_path = local_path
                logger.info(f"Found local model at: {local_path}")
            else:
                self._model_path = None  # Sẽ tải từ HuggingFace
                
    def initialize(self):
        """Khởi tạo model SAT với LoRA adaptation"""
        if self._initialized:
            return
            
        try:
            from wtpsplit import SaT
            
            logger.info(f"Đang tải SAT model: {self.model_name}")
            logger.info(f"  - Language: {self.language}")
            logger.info(f"  - Style/Domain: {self.style_or_domain}")
            logger.info(f"  - Threshold: {self.threshold}")
            logger.info(f"  - Paragraph Threshold: {self.paragraph_threshold}")
            
            # Load model từ local hoặc HuggingFace
            if self._model_path and os.path.exists(self._model_path):
                logger.info(f"  - Loading from local path: {self._model_path}")
                # Load từ local path
                model_identifier = self._model_path
            else:
                logger.info("  - Loading from HuggingFace Hub")
                model_identifier = self.model_name
            
            # Khởi tạo model
            # Kiểm tra xem có file ONNX không để dùng ort_providers
            onnx_path = os.path.join(model_identifier, "model_optimized.onnx") if os.path.isdir(model_identifier) else None
            has_onnx = onnx_path and os.path.exists(onnx_path)
            
            if has_onnx:
                # Load với ONNX Runtime (nhẹ và nhanh hơn)
                logger.info("  - Using ONNX Runtime")
                if self.device == "cuda":
                    ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    ort_providers = ["CPUExecutionProvider"]
                
                self._sat_model = SaT(model_identifier, ort_providers=ort_providers)
                logger.info(f"  - Loaded ONNX model from: {model_identifier}")
            elif self._model_path:
                # Offline mode nhưng không có ONNX -> load PyTorch model
                self._sat_model = SaT(model_identifier)
                logger.info("  - Loaded PyTorch model (no LoRA)")
            else:
                # Online mode: load với LoRA adaptation
                self._sat_model = SaT(
                    model_identifier,
                    style_or_domain=self.style_or_domain,
                    language=self.language
                )
            
            # Chuyển sang device nếu cần (chỉ cho PyTorch mode, ONNX tự xử lý qua ort_providers)
            if self.device and not has_onnx:
                logger.info(f"Chuyển model sang device: {self.device}")
                if self.device == "cuda":
                    self._sat_model.half().to(self.device)
                else:
                    self._sat_model.to(self.device)
            
            self._initialized = True
            logger.info("Đã tải SAT model thành công")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải SAT model: {e}")
            raise
    
    def segment_sentences(self, text: str, threshold: float = None) -> List[str]:
        """
        Tách văn bản thành các câu
        
        Args:
            text: Văn bản cần tách
            threshold: Ngưỡng tùy chọn (nếu None dùng giá trị mặc định)
            
        Returns:
            List các câu
        """
        if not self._initialized:
            self.initialize()
        
        if not text or not text.strip():
            return []
        
        try:
            thresh = threshold if threshold is not None else self.threshold
            sentences = self._sat_model.split(text, threshold=thresh)
            
            cleaned_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    cleaned_sentences.append(sent)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"Lỗi khi tách câu: {e}")
            return self._fallback_sentence_split(text)
    
    def segment_paragraphs(
        self, 
        text: str, 
        do_paragraph_segmentation: bool = True,
        paragraph_threshold: float = None
    ) -> List[dict]:
        """
        Tách văn bản thành các đoạn (paragraphs) với metadata
        
        Args:
            text: Văn bản cần tách
            do_paragraph_segmentation: Bật/tắt tách đoạn
            paragraph_threshold: Ngưỡng tách đoạn
            
        Returns:
            List các đoạn
        """
        if not self._initialized:
            self.initialize()
        
        if not text or not text.strip():
            return []
        
        try:
            para_thresh = paragraph_threshold if paragraph_threshold is not None else self.paragraph_threshold
            
            if do_paragraph_segmentation:
                result = self._sat_model.split(
                    text, 
                    do_paragraph_segmentation=True,
                    paragraph_threshold=para_thresh,
                    threshold=self.threshold
                )
                
                paragraphs = []
                for para_sentences in result:
                    if para_sentences:
                        para_text = " ".join(s.strip() for s in para_sentences if s.strip())
                        if para_text:
                            paragraphs.append({
                                'text': para_text,
                                'sentences': [s.strip() for s in para_sentences if s.strip()]
                            })
                
                return paragraphs
            else:
                sentences = self.segment_sentences(text)
                return [{
                    'text': ' '.join(sentences),
                    'sentences': sentences
                }]
            
        except Exception as e:
            logger.error(f"Lỗi khi tách đoạn: {e}")
            return self._fallback_paragraph_split(text)
    
    def predict_proba(self, text: str) -> List[float]:
        """Lấy xác suất boundary"""
        if not self._initialized:
            self.initialize()
        
        try:
            return self._sat_model.predict_proba(text)
        except Exception as e:
            logger.error(f"Lỗi khi predict_proba: {e}")
            return []
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Phương pháp tách câu dự phòng"""
        sentence_endings = r'[.!?。！？;]+\s*'
        sentences = re.split(sentence_endings, text)
        
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 3:
                cleaned.append(sent)
        
        return cleaned if cleaned else [text]
    
    def _fallback_paragraph_split(self, text: str) -> List[dict]:
        """Phương pháp tách đoạn dự phòng"""
        raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        paragraphs = []
        for para_text in raw_paragraphs:
            sentences = self._fallback_sentence_split(para_text)
            paragraphs.append({
                'text': para_text,
                'sentences': sentences
            })
        
        return paragraphs
    
    def unload(self):
        """
        Giải phóng bộ nhớ bằng cách unload model SAT.
        Gọi method này khi muốn tiết kiệm RAM sau khi xử lý xong.
        """
        import gc
        
        if self._sat_model is not None:
            logger.info("Unloading SAT model...")
            del self._sat_model
            self._sat_model = None
            self._initialized = False
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("SAT model unloaded successfully")


class ChunkedSATProcessor:
    """
    Xử lý văn bản theo chunks để tiết kiệm bộ nhớ.
    Mỗi chunk khoảng 500 từ, với logic overlap để đảm bảo câu không bị cắt.
    """
    
    def __init__(self, chunk_size: int = 500, segmenter: SATSegmenter = None):
        """
        Args:
            chunk_size: Số từ mỗi chunk (mặc định 500)
            segmenter: Instance của SATSegmenter (nếu None sẽ tạo mới)
        """
        self.chunk_size = chunk_size
        self.segmenter = segmenter
        
    def _split_into_word_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chia text thành các chunks dựa theo số từ.
        
        Returns:
            List of (chunk_text, start_char_idx, end_char_idx)
        """
        words = text.split()
        total_words = len(words)
        
        if total_words <= self.chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        word_idx = 0
        char_idx = 0
        
        while word_idx < total_words:
            # Lấy chunk_size từ tiếp theo
            end_word_idx = min(word_idx + self.chunk_size, total_words)
            chunk_words = words[word_idx:end_word_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Tính vị trí ký tự trong text gốc (xấp xỉ)
            start_char = char_idx
            end_char = start_char + len(chunk_text)
            
            chunks.append((chunk_text, start_char, end_char))
            
            word_idx = end_word_idx
            char_idx = end_char + 1  # +1 for space
        
        return chunks
    
    def process_chunked(
        self,
        text: str,
        threshold: float = 0.3,
        progress_callback=None,
        should_stop=None
    ) -> List[str]:
        """
        Xử lý văn bản theo chunks với overlap handling.
        
        Args:
            text: Văn bản cần xử lý
            threshold: Ngưỡng tách câu
            progress_callback: Callable(current_chunk, total_chunks, sentences_so_far)
            should_stop: Callable() -> bool để kiểm tra dừng sớm
            
        Returns:
            List các câu đã tách
            
        Algorithm:
        1. Chia text thành chunks ~500 từ
        2. Với mỗi chunk (trừ chunk đầu), thêm câu cuối của chunk trước vào đầu
        3. Chạy SAT trên chunk
        4. So sánh câu đầu của chunk hiện tại với câu cuối của chunk trước:
           - Nếu gộp: thay thế câu cuối của chunk trước bằng câu gộp
           - Nếu tách: giữ nguyên cả hai, xóa duplicate
        5. Tiếp tục cho đến hết chunks
        """
        if not text or not text.strip():
            return []
        
        # Initialize segmenter if needed
        if self.segmenter is None:
            self.segmenter = SATSegmenter(threshold=threshold)
        
        if not self.segmenter._initialized:
            self.segmenter.initialize()
        
        # Split into chunks
        chunks = self._split_into_word_chunks(text)
        total_chunks = len(chunks)
        
        logger.info(f"ChunkedSATProcessor: Chia thành {total_chunks} chunks ({self.chunk_size} từ/chunk)")
        
        all_sentences = []
        previous_last_sentence = None
        
        for chunk_idx, (chunk_text, start_char, end_char) in enumerate(chunks):
            # Check for early exit
            if should_stop and should_stop():
                logger.info("ChunkedSATProcessor: Dừng theo yêu cầu")
                return all_sentences
            
            # Prepare text to segment
            if previous_last_sentence:
                # Thêm câu cuối của chunk trước vào đầu chunk này
                text_to_segment = previous_last_sentence + " " + chunk_text
            else:
                text_to_segment = chunk_text
            
            # Segment with SAT
            chunk_sentences = self.segmenter.segment_sentences(text_to_segment, threshold=threshold)
            
            if not chunk_sentences:
                continue
            
            # Handle overlap
            if previous_last_sentence and len(all_sentences) > 0:
                # Kiểm tra xem câu đầu của chunk này có bằng câu cuối chunk trước không
                first_sentence = chunk_sentences[0].strip() if chunk_sentences else ""
                
                # Normalize để so sánh
                prev_normalized = previous_last_sentence.strip().lower()
                first_normalized = first_sentence.strip().lower()
                
                # Kiểm tra 3 trường hợp:
                # 1. Câu đầu chunk mới = câu cuối chunk cũ (tách riêng) -> bỏ duplicate
                # 2. Câu đầu chunk mới chứa câu cuối chunk cũ (gộp) -> thay thế
                # 3. Câu đầu chunk mới khác hoàn toàn -> SAT quyết định gộp
                
                if first_normalized == prev_normalized:
                    # Trường hợp 1: Câu không bị gộp, bỏ duplicate
                    chunk_sentences = chunk_sentences[1:]
                    logger.debug(f"Chunk {chunk_idx}: Không gộp, bỏ duplicate")
                elif prev_normalized in first_normalized and len(first_normalized) > len(prev_normalized):
                    # Trường hợp 2: SAT gộp câu cuối chunk cũ vào câu đầu chunk mới
                    # Thay thế câu cuối của all_sentences bằng câu gộp
                    all_sentences[-1] = chunk_sentences[0]
                    chunk_sentences = chunk_sentences[1:]
                    logger.debug(f"Chunk {chunk_idx}: SAT đã gộp câu")
                else:
                    # Trường hợp 3: SAT quyết định khác -> bỏ ref của prev sentence nếu có
                    # Kiểm tra xem câu đầu có bắt đầu bằng nội dung của prev không
                    if first_sentence.startswith(previous_last_sentence[:min(20, len(previous_last_sentence))]):
                        # Có thể là gộp, thay thế
                        all_sentences[-1] = chunk_sentences[0]
                        chunk_sentences = chunk_sentences[1:]
                    # Else: giữ nguyên, SAT đã tách riêng
            
            # Add remaining sentences from this chunk
            all_sentences.extend(chunk_sentences)
            
            # Remember last sentence for next chunk
            if all_sentences:
                previous_last_sentence = all_sentences[-1]
            
            # Progress callback
            if progress_callback:
                progress_callback(chunk_idx + 1, total_chunks, len(all_sentences))
            
            logger.info(f"Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_sentences)} câu (tổng: {len(all_sentences)})")
        
        logger.info(f"ChunkedSATProcessor: Hoàn thành - {len(all_sentences)} câu")
        return all_sentences


class SATPunctuationPipeline:
    """
    Pipeline kết hợp SAT (tách câu/đoạn) + Vibert (gán dấu câu)
    """
    
    def __init__(
        self,
        sat_model: str = "sat-12l-sm",
        language: str = "vi",
        style_or_domain: str = "ted",
        threshold: float = 0.5,
        paragraph_threshold: float = 0.5,
        vibert_model: str = "dragonSwing/vibert-capu",
        device: str = None,
        local_sat_path: str = None,  # Đường dẫn local SAT model
        local_vibert_path: str = None  # Đường dẫn local Vibert model
    ):
        self.segmenter = None
        self.sat_model_name = sat_model
        self.language = language
        self.style_or_domain = style_or_domain
        self.threshold = threshold
        self.paragraph_threshold = paragraph_threshold
        self.vibert_model_name = vibert_model
        self.device = device
        self.local_sat_path = local_sat_path
        self.local_vibert_path = local_vibert_path
        self.punctuation_restorer = None
        
    def initialize(self):
        """Khởi tạo cả hai model"""
        # Khởi tạo SAT
        self.segmenter = SATSegmenter(
            model_name=self.sat_model_name,
            language=self.language,
            style_or_domain=self.style_or_domain,
            threshold=self.threshold,
            paragraph_threshold=self.paragraph_threshold,
            device=self.device,
            local_model_path=self.local_sat_path
        )
        self.segmenter.initialize()
        
        # Khởi tạo Vibert
        try:
            from punctuation_restorer_improved import ImprovedPunctuationRestorer
            
            logger.info(f"Đang tải Vibert model: {self.vibert_model_name}")
            
            # Kiểm tra local Vibert model
            if self.local_vibert_path and os.path.exists(self.local_vibert_path):
                logger.info(f"  - Loading from local path: {self.local_vibert_path}")
                # Tạo restorer với model local
                self.punctuation_restorer = self._create_local_vibert_restorer(self.local_vibert_path)
            else:
                # Tự động tìm trong models/
                base_dir = os.path.dirname(os.path.abspath(__file__))
                auto_local = os.path.join(base_dir, "models", "vibert-capu")
                if os.path.exists(os.path.join(auto_local, "pytorch_model.bin")):
                    logger.info(f"  - Loading from local path: {auto_local}")
                    self.punctuation_restorer = self._create_local_vibert_restorer(auto_local)
                else:
                    logger.info("  - Loading from HuggingFace Hub")
                    self.punctuation_restorer = ImprovedPunctuationRestorer(
                        model_name=self.vibert_model_name,
                        device=self.device
                    )
            
            logger.info("Đã tải Vibert model thành công")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải Vibert model: {e}")
            raise
    
    def _create_local_vibert_restorer(self, model_path: str):
        """Tạo Vibert restorer từ local path"""
        import torch
        from gec_model import GecBERTModel
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(base_dir, "vocabulary")
        
        gec_model = GecBERTModel(
            vocab_path=vocab_path,
            model_paths=[model_path],
            split_chunk=True,
            device=self.device,
            confidence=0.3
        )
        
        if self.device == "cpu":
            import gc
            for i, model in enumerate(gec_model.models):
                gec_model.models[i] = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                del model
                gc.collect()
        
        import re
        # Tạo restorer tạm thờii với post-processing
        restorer = type('LocalVibertRestorer', (), {})()
        restorer.gec_model = gec_model
        
        def _post_process(text):
            """Làm sạch dấu câu sau khi model predict (giống ImprovedPunctuationRestorer)."""
            # 1. Xóa dấu phẩy liên tiếp (,, -> ,)
            text = re.sub(r',+', ',', text)
            
            # 2. Xóa dấu chấm liên tiếp (... giữ nguyên ...)
            text = re.sub(r'\.{4,}', '...', text)
            
            # 3. Không để dấu phẩy trước dấu chấm (,. -> .)
            text = re.sub(r',\s*\.', '.', text)
            
            # 4. Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        def restore(text):
            if not text or not text.strip():
                return ""
            try:
                with torch.no_grad():
                    results = gec_model(text)
                if isinstance(results, list):
                    result = results[0]
                    return _post_process(result)
                return _post_process(results)
            except Exception as e:
                print(f"Error during restoration: {e}")
                return text
        
        restorer.restore = restore
        return restorer
    
    def process(
        self, 
        text: str,
        do_paragraph_segmentation: bool = False,
        paragraph_threshold: float = None,
        threshold: float = None
    ) -> dict:
        """Xử lý pipeline: Tách câu/đoạn → Gán dấu"""
        if self.segmenter is None or self.punctuation_restorer is None:
            self.initialize()
        
        result = {
            'segments': [],
            'sentences': [],
            'paragraphs': [],
            'full_text': '',
            'raw_sentences': []
        }
        
        # Bước 1: Tách câu/đoạn
        # Sử dụng threshold tùy chỉnh nếu được truyền vào
        sentence_threshold = threshold if threshold is not None else self.threshold
        
        if do_paragraph_segmentation:
            logger.info("Bước 1: Tách đoạn và câu bằng SAT...")
            logger.info(f"  - Sentence threshold: {sentence_threshold}")
            paragraphs = self.segmenter.segment_paragraphs(
                text,
                do_paragraph_segmentation=True,
                paragraph_threshold=paragraph_threshold
            )
            logger.info(f"Đã tách được {len(paragraphs)} đoạn")
            
            all_sentences = []
            for para in paragraphs:
                all_sentences.extend(para['sentences'])
            
            result['paragraphs'] = paragraphs
            result['raw_sentences'] = all_sentences
        else:
            logger.info("Bước 1: Tách câu bằng...")
            logger.info(f"  - Threshold: {sentence_threshold}")
            sentences = self.segmenter.segment_sentences(text, threshold=sentence_threshold)
            logger.info(f"Đã tách được {len(sentences)} câu")
            result['raw_sentences'] = sentences
        
        # Bước 2: Gán dấu
        raw_sentences = result['raw_sentences']
        logger.info(f"Bước 2: Gán dấu cho {len(raw_sentences)} câu...")
        punctuated_sentences = []
        
        for i, sentence in enumerate(raw_sentences):
            try:
                punctuated = self.punctuation_restorer.restore(sentence)
                punctuated_sentences.append(punctuated)
                if (i + 1) % 10 == 0:
                    logger.info(f"Đã xử lý {i + 1}/{len(raw_sentences)} câu")
            except Exception as e:
                logger.error(f"Lỗi khi gán dấu câu {i}: {e}")
                punctuated_sentences.append(sentence)
        
        # Post-process: Ensure all sentences have ending punctuation
        final_sentences = []
        valid_endings = ('.', '?', '!', '...', '…')
        
        for sent in punctuated_sentences:
            sent = sent.strip()
            if sent:
                if not sent.endswith(valid_endings):
                    # Check if it ends with a quote or parenthesis that might contain punctuation
                    if sent.endswith(('"', "'", ')', ']', '}')):
                        # Checking loosely before the closing char is simpler/safer to just append '.' if unsure
                        # But typically "Start." is good. "Start". is also okay.
                        # Let's simple check the very last char first.
                        pass 
                    
                    # Force append period
                    sent += '.'
                final_sentences.append(sent)
        
        result['sentences'] = final_sentences
        punctuated_sentences = final_sentences
        
        # Cập nhật paragraphs
        if do_paragraph_segmentation and result['paragraphs']:
            sent_idx = 0
            for para in result['paragraphs']:
                para_sentences = []
                for _ in para['sentences']:
                    if sent_idx < len(punctuated_sentences):
                        para_sentences.append(punctuated_sentences[sent_idx])
                        sent_idx += 1
                para['sentences'] = para_sentences
                para['text'] = '. '.join(para_sentences)
                if para['text'] and not para['text'].endswith('.'):
                    para['text'] += '.'
        
        result['segments'] = self._group_into_segments(punctuated_sentences)
        
        full_text = '. '.join(punctuated_sentences)
        if full_text and not full_text.endswith('.'):
            full_text += '.'
        result['full_text'] = full_text
        
        return result
    
    def _group_into_segments(self, sentences: List[str], max_sentences_per_segment: int = 5) -> List[dict]:
        """Nhóm các câu thành segments"""
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            if len(current_segment) >= max_sentences_per_segment:
                segments.append({
                    'sentences': current_segment.copy(),
                    'text': '. '.join(current_segment) + '.'
                })
                current_segment = []
        
        if current_segment:
            segments.append({
                'sentences': current_segment.copy(),
                'text': '. '.join(current_segment) + '.'
            })
        
        return segments
    
    def unload(self):
        """
        Giải phóng bộ nhớ bằng cách unload cả SAT và Punctuation model.
        Gọi method này khi muốn tiết kiệm RAM sau khi xử lý xong.
        """
        import gc
        
        if self.segmenter is not None:
            self.segmenter.unload()
            self.segmenter = None
        
        if self.punctuation_restorer is not None:
            if hasattr(self.punctuation_restorer, 'unload'):
                self.punctuation_restorer.unload()
            del self.punctuation_restorer
            self.punctuation_restorer = None
        
        gc.collect()
        logger.info("SATPunctuationPipeline: Đã unload tất cả các model")


# Singleton instances
_sat_segmenter = None
_sat_pipeline = None

def get_sat_segmenter(
    model_name: str = "sat-12l-sm",
    language: str = "vi",
    style_or_domain: str = "ted",
    local_model_path: str = None
) -> SATSegmenter:
    """Lấy singleton instance của SATSegmenter"""
    global _sat_segmenter
    if _sat_segmenter is None:
        _sat_segmenter = SATSegmenter(
            model_name=model_name,
            language=language,
            style_or_domain=style_or_domain,
            local_model_path=local_model_path
        )
        _sat_segmenter.initialize()
    return _sat_segmenter


def get_sat_pipeline(
    vibert_model: str = "dragonSwing/vibert-capu",
    do_paragraph_segmentation: bool = False,
    local_sat_path: str = None,
    local_vibert_path: str = None
) -> SATPunctuationPipeline:
    """Lấy singleton instance của SATPunctuationPipeline"""
    global _sat_pipeline
    if _sat_pipeline is None:
        _sat_pipeline = SATPunctuationPipeline(
            vibert_model=vibert_model,
            local_sat_path=local_sat_path,
            local_vibert_path=local_vibert_path
        )
        _sat_pipeline.initialize()
    return _sat_pipeline


# Test
if __name__ == "__main__":
    test_text = """xin chao cac ban hom nay toi se gioi thieu ve cong nghe nhan dang giong noi

cong nghe nay rat huu ich trong nhieu linh vuc nhu y te giao duc va kinh doanh

chung ta cung tim hieu nhe"""
    
    print("=== Test SAT Segmenter (sat-12l-sm) ===")
    print(f"Model: sat-12l-sm")
    print(f"Language: vi")
    print(f"Style: ted (phù hợp với văn nói)")
    print()
    
    segmenter = SATSegmenter(
        model_name="sat-12l-sm",
        language="vi",
        style_or_domain="ted",
        paragraph_threshold=0.5
    )
    segmenter.initialize()
    
    print("\nVan ban goc:")
    print(test_text)
    
    print("\n--- Tach cau ---")
    sentences = segmenter.segment_sentences(test_text)
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    print(f"\nTong so cau: {len(sentences)}")
