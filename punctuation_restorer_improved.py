"""
Punctuation Restorer với độ chính xác cao hơn.
- Tăng confidence threshold
- Post-processing để giảm dấu phẩy thừa
- Giữ dấu chấm cho các điểm kết thúc rõ ràng
"""

import os
import re

# Lazy import torch to avoid DLL conflicts
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

class ImprovedPunctuationRestorer:
    def __init__(self, device="cpu", confidence=0.3, model_name="dragonSwing/vibert-capu", case_confidence=0.0): 
        self.device = device
        self.model_name = model_name
        self.confidence = confidence
        
        # Load model giống như cũ
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(base_dir, "vocabulary")
        
        img_model_path = os.path.join(base_dir, "models", "vibert-capu")
        if os.path.exists(img_model_path) and os.path.exists(os.path.join(img_model_path, "pytorch_model.bin")):
            model_to_load = img_model_path
        else:
            model_to_load = self.model_name
        
        print(f"Loading GecBERTModel (confidence={confidence}): {model_to_load}...")
        
        from gec_model import GecBERTModel
        self.gec_model = GecBERTModel(
            vocab_path=vocab_path,
            model_paths=[model_to_load],
            split_chunk=True,
            chunk_size=48,
            overlap_size=12,
            max_len=64,
            device=device,
            confidence=confidence,
            case_confidence=case_confidence
        )
        
        if self.device == "cpu":
            print("Applying dynamic quantization...")
            import gc
            torch = get_torch()
            for i, model in enumerate(self.gec_model.models):
                self.gec_model.models[i] = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                del model
                gc.collect()
    
    def restore(self, text, progress_callback=None):
        """Thêm dấu với post-processing để tăng độ chính xác."""
        if not text or not text.strip():
            return ""
        
        try:
            torch = get_torch()
            with torch.no_grad():
                results = self.gec_model(text, progress_callback=progress_callback)
            
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            # Post-processing để làm sạch dấu câu
            result = self._post_process(result)
            
            return result
            
        except Exception as e:
            print(f"Error during restoration: {e}")
            return text
    
    def _post_process(self, text):
        """Làm sạch dấu câu sau khi model predict."""
        # 1. Xóa dấu phẩy liên tiếp (,, -> ,)
        text = re.sub(r',+', ',', text)
        
        # 2. Xóa dấu chấm liên tiếp (... giữ nguyên ...)
        text = re.sub(r'\.{4,}', '...', text)
        
        # 3. Không để dấu phẩy trước dấu chấm (,. -> .)
        text = re.sub(r',\s*\.', '.', text)
        
        # 4. Giảm dấu phẩy giữa các từ ngắn (không quá 1 dấu phẩy trong 5 từ)
        # Tách thành các câu để xử lý
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned_sentences = []
        
        for sent in sentences:
            # Đếm số dấu phẩy trong câu
            comma_count = sent.count(',')
            words = sent.split()
            
            # Nếu câu ngắn (< 8 từ) mà có nhiều dấu phẩy, giảm bớt
            if len(words) < 8 and comma_count > 1:
                # Giữ lại chỉ 1 dấu phẩy đầu tiên
                parts = sent.split(',', 1)
                if len(parts) > 1:
                    # Tìm vị trí dấu phẩy thứ 2 và xóa các dấu sau đó
                    second_comma_pos = parts[1].find(',')
                    if second_comma_pos != -1:
                        parts[1] = parts[1][:second_comma_pos] + parts[1][second_comma_pos+1:].replace(',', '')
                    sent = parts[0] + ',' + parts[1]
            
            cleaned_sentences.append(sent)
        
        text = ' '.join(cleaned_sentences)
        
        # 5. Đảm bảo khoảng trắng sau dấu câu
        text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)
        
        # 6. Xóa khoảng trắng thừa trước dấu câu
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        # 7. Không để dấu phẩy ở đầu câu
        text = re.sub(r'^,\s*', '', text)
        text = re.sub(r'\.\s*,', '. ', text)
        
        # 8. Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        
        # 9. Viết hoa chữ cái đầu và sau dấu chấm câu (an toàn thủ công)
        def capitalize_match(match):
            return match.group(1) + match.group(2).upper()
        text = re.sub(r'(^|[.!?]\s+)([^\W_])', capitalize_match, text)
        
        return text.strip()
    
    def unload(self):
        """
        Giải phóng bộ nhớ bằng cách unload GEC model.
        Gọi method này khi muốn tiết kiệm RAM sau khi xử lý xong.
        """
        import gc
        
        if hasattr(self, 'gec_model') and self.gec_model is not None:
            print("Unloading GEC model...")
            del self.gec_model
            self.gec_model = None
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                torch = get_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            print("GEC model unloaded successfully")


# Class cũ để tương thích ngược
class PunctuationRestorer(ImprovedPunctuationRestorer):
    """Tương thích với code cũ."""
    pass


if __name__ == "__main__":
    # Test
    restorer = ImprovedPunctuationRestorer(confidence=0.3)
    
    test_texts = [
        "kính thưa các đồng chí tiếp nối chương trình chất vấn",
        "xin cảm ơn đại biểu mờI đại biểu phát biểu",
        "về vấn đề quy hoạch treo chúng tôi đã có nhiều giải pháp",
    ]
    
    print("\nTest improved punctuation restoration:")
    print("=" * 60)
    for text in test_texts:
        result = restorer.restore(text)
        print(f"\nInput:  {text}")
        print(f"Output: {result}")
