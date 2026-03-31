"""
Meeting summarizer - 3-pass pipeline (Extract → Summarize → Self-check).
Backend: Ollama API (local) hoặc llama-cpp-python (nếu có).
Chỉ dùng cho web app. Tích hợp vào queue_manager.
"""

import os
import re
import json
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger("asr.summarizer")

# === Model download ===

DEFAULT_GGUF_REPO = "unsloth/Qwen3.5-4B-GGUF"
DEFAULT_GGUF_FILE = "Qwen3.5-4B-Q4_K_M.gguf"


def get_default_model_path() -> str:
    """Đường dẫn mặc định cho model GGUF trong thư mục models/."""
    from core.config import BASE_DIR
    return os.path.join(BASE_DIR, "models", DEFAULT_GGUF_FILE)


def download_model(progress_cb: Optional[Callable[[str, int], None]] = None) -> str:
    """
    Tải model GGUF từ HuggingFace nếu chưa có.
    Trả về đường dẫn file GGUF.
    progress_cb(message, percent) — callback tiến độ.
    """
    dest = get_default_model_path()
    if os.path.isfile(dest):
        logger.info(f"Model already exists: {dest}")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    logger.info(f"Downloading model {DEFAULT_GGUF_REPO}/{DEFAULT_GGUF_FILE} → {dest}")
    if progress_cb:
        progress_cb(f"Đang tải model {DEFAULT_GGUF_FILE}...", 0)

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=DEFAULT_GGUF_REPO,
            filename=DEFAULT_GGUF_FILE,
            local_dir=os.path.dirname(dest),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded: {path} ({os.path.getsize(path)/1e9:.1f} GB)")
        if progress_cb:
            progress_cb("Tải model hoàn tất", 100)
        return path
    except ImportError:
        # Fallback: dùng requests tải trực tiếp từ HF
        import requests
        url = f"https://huggingface.co/{DEFAULT_GGUF_REPO}/resolve/main/{DEFAULT_GGUF_FILE}"
        logger.info(f"Downloading via requests: {url}")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_cb:
                    pct = min(99, int(downloaded / total * 100))
                    progress_cb(f"Đang tải model... {downloaded/1e9:.1f}/{total/1e9:.1f} GB", pct)
        if progress_cb:
            progress_cb("Tải model hoàn tất", 100)
        logger.info(f"Model downloaded: {dest} ({os.path.getsize(dest)/1e9:.1f} GB)")
        return dest


def _get_physical_cores() -> int:
    """Lấy số physical cores (giống ALLOWED_THREADS trong core/config.py)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 4
    except ImportError:
        import os
        return max(1, (os.cpu_count() or 4) // 2)


# === Prompts ===

EXTRACT_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích biên bản họp. Nhiệm vụ: đọc bản ghi cuộc họp và trích xuất TẤT CẢ thông tin quan trọng.

QUY TẮC:
1. Đọc KỸ TỪNG DÒNG. Không được bỏ sót bất kỳ quyết định, nhiệm vụ, hay thông tin quan trọng nào.
2. CHỈ trích xuất thông tin CÓ TRONG bản ghi. KHÔNG thêm, suy diễn, hay bịa đặt.
3. Mỗi mục PHẢI ghi rõ số đoạn nguồn (ví dụ: đoạn 5, 12).
4. Bỏ qua: lời chào, câu lặp lại, nói dở dang vô nghĩa, tiếng ồn.
5. Giữ nguyên tên riêng, con số, ngày tháng chính xác như trong bản ghi.
6. Nếu không chắc chắn → giữ nguyên câu gốc thay vì diễn giải lại.
7. Trả lời bằng tiếng Việt.\
"""

EXTRACT_USER_TEMPLATE = """\
CUỘC HỌP: {duration_min} phút, {num_speakers} người ({speaker_list}).

BẢN GHI:
{transcript}

---
Trích xuất TẤT CẢ thông tin quan trọng theo các nhóm sau:

THÔNG TIN CHÍNH:
- Liệt kê mọi thông tin, báo cáo, cập nhật quan trọng được đề cập.
- Format: "• [Nội dung] — {{tên người nói}} (đoạn X, Y)"

QUYẾT ĐỊNH:
- Liệt kê mọi quyết định, thống nhất, phê duyệt.
- Format: "• [Quyết định gì] (đoạn X)"

CÔNG VIỆC:
- Liệt kê mọi nhiệm vụ, deadline, cam kết được giao.
- Format: "• [Việc gì] → [Ai làm] — deadline: [nếu có] (đoạn X)"

VẤN ĐỀ CHƯA THỐNG NHẤT:
- Liệt kê vấn đề còn tranh cãi hoặc chưa có kết luận.
- Format: "• [Vấn đề gì] (đoạn X)"

KẾT LUẬN:
- Tóm gọn kết luận cuối cuộc họp (nếu có).
- Format: "• [Kết luận] (đoạn X)"\
"""

SUMMARIZE_SYSTEM_PROMPT = """\
Bạn là chuyên gia viết báo cáo tóm tắt cuộc họp.

QUY TẮC:
1. CHỈ sử dụng thông tin trong phần TRÍCH XUẤT bên dưới. KHÔNG thêm gì.
2. Viết ngắn gọn, chuyên nghiệp, đi thẳng vào trọng tâm.
3. Giữ nguyên số đoạn refs từ phần trích xuất.
4. Trả về DUY NHẤT JSON hợp lệ, không có text bên ngoài.
5. Không bắt đầu bằng ```json hay markdown, chỉ trả JSON thuần.
6. Dùng tiếng Việt.\
"""

SUMMARIZE_USER_TEMPLATE = """\
THÔNG TIN CUỘC HỌP:
- Thời lượng: {duration_min} phút
- Người tham gia: {speaker_list}

TRÍCH XUẤT TỪ BIÊN BẢN:
{extracted_facts}

---
Viết tóm tắt cuộc họp theo JSON format sau:
{{
  "title": "Tiêu đề ngắn gọn (dưới 15 từ)",
  "summary": "Tóm tắt tổng quan 2-3 câu",
  "key_points": [
    {{"text": "Nội dung", "speaker": "Ai nói", "refs": [0, 3]}}
  ],
  "decisions": [
    {{"text": "Quyết định", "refs": [12]}}
  ],
  "action_items": [
    {{"text": "Việc cần làm", "assignee": "Ai", "deadline": "nếu có hoặc null", "refs": [15]}}
  ],
  "open_issues": [
    {{"text": "Vấn đề chưa thống nhất", "refs": [20]}}
  ],
  "conclusion": "Kết luận cuối (null nếu không có)"
}}

Lưu ý:
- key_points: sắp xếp theo thứ tự thời gian trong cuộc họp.
- Gộp các ý liên quan thành 1 điểm (không lặp).
- Mỗi mục PHẢI có refs hợp lệ.
- Nếu section rỗng thì trả mảng rỗng [].
- conclusion là string hoặc null.\
"""

SELFCHECK_SYSTEM_PROMPT = """\
Bạn là chuyên gia kiểm tra chất lượng biên bản họp.

Nhiệm vụ: so sánh BẢN TÓM TẮT với BẢN TRÍCH XUẤT GỐC, tìm lỗi.\
"""

SELFCHECK_USER_TEMPLATE = """\
BẢN TRÍCH XUẤT GỐC:
{extracted_facts}

BẢN TÓM TẮT:
{summary_text}

---
Kiểm tra và trả lời NGẮN GỌN:

1. CÓ Ý NÀO TRONG TRÍCH XUẤT BỊ THIẾU trong tóm tắt không? Liệt kê.
2. CÓ Ý NÀO TRONG TÓM TẮT KHÔNG CÓ trong trích xuất không? (hallucination) Liệt kê.
3. Có sai tên, sai số, sai ngày tháng không?

Nếu tất cả đều OK, trả lời: "OK - không có lỗi."
Nếu có lỗi, liệt kê từng lỗi cụ thể.\
"""

SELFCHECK_FIX_SYSTEM = """\
Sửa lại JSON tóm tắt cuộc họp dựa trên phản hồi kiểm tra.
CHỈ sửa các lỗi được chỉ ra. Giữ nguyên phần đúng.
Trả về DUY NHẤT JSON hợp lệ, không có text bên ngoài.\
"""

SELFCHECK_FIX_USER = """\
JSON TÓM TẮT HIỆN TẠI:
{summary_json}

LỖI CẦN SỬA:
{errors}

TRÍCH XUẤT GỐC (để tham chiếu):
{extracted_facts}

---
Sửa JSON và trả về JSON hoàn chỉnh đã fix.\
"""


# === Transcript formatting ===

def format_transcript_for_llm(asr_data: dict) -> tuple:
    """
    Chuyển ASR JSON thành text cho LLM prompt.
    Dùng text-only index (khớp với playerSegments trong frontend).

    Returns:
        (transcript_text, text_seg_map, speaker_list, duration_min)
    """
    segments = asr_data.get("segments", [])
    duration_sec = asr_data.get("duration_sec", 0)

    lines = []
    current_speaker = "?"
    text_idx = 0
    text_seg_map = {}
    speakers_seen = set()

    for seg in segments:
        seg_type = seg.get("type", "")
        if seg_type == "speaker":
            current_speaker = seg.get("speaker", "?")
            speakers_seen.add(current_speaker)
        elif seg_type == "text":
            start = seg.get("start_time", 0)
            h = int(start // 3600)
            m = int(start % 3600 // 60)
            s = int(start % 60)
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {current_speaker} (đoạn {text_idx}): {text}")
                text_seg_map[text_idx] = {
                    "start_time": start,
                    "text": text,
                    "speaker": current_speaker,
                }
                text_idx += 1

    transcript = "\n".join(lines)
    speaker_list = sorted(speakers_seen)
    duration_min = round(duration_sec / 60, 1)

    return transcript, text_seg_map, speaker_list, duration_min


def chunk_transcript(transcript: str, text_seg_map: dict,
                     max_chars: int = 12000) -> list:
    """
    Chia transcript thành chunks theo ranh giới speaker.
    Mỗi chunk giữ context liên tục, không cắt giữa câu.

    Returns: list of (chunk_text, chunk_seg_ids)
    """
    if len(transcript) <= max_chars:
        return [(transcript, list(text_seg_map.keys()))]

    lines = transcript.split("\n")
    chunks = []
    current_chunk = []
    current_ids = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        # Cắt tại ranh giới khi đủ lớn
        if current_len + line_len > max_chars and current_chunk:
            chunks.append(("\n".join(current_chunk), current_ids[:]))
            current_chunk = []
            current_ids = []
            current_len = 0

        current_chunk.append(line)
        current_len += line_len

        # Extract segment id from line
        match = re.search(r'\(đoạn (\d+)\)', line)
        if match:
            current_ids.append(int(match.group(1)))

    if current_chunk:
        chunks.append(("\n".join(current_chunk), current_ids[:]))

    return chunks


# === JSON parsing ===

def parse_llm_json(raw: str) -> dict:
    """Parse JSON từ LLM output với fallback strategies."""
    raw = raw.strip()

    # Loại bỏ thinking tags nếu có (Qwen3.5 thinking mode)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # 1. Parse trực tiếp
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Tìm {} block ngoài cùng
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 3. Tìm trong ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 4. Fallback
    logger.warning(f"Failed to parse LLM JSON output (len={len(raw)})")
    return {}


def validate_summary(summary: dict, text_seg_map: dict) -> dict:
    """Validate refs, loại bỏ refs không hợp lệ."""
    if not summary:
        return _empty_summary("LLM không trả về kết quả hợp lệ")

    max_idx = max(text_seg_map.keys()) if text_seg_map else -1

    for section in ("key_points", "decisions", "action_items", "open_issues"):
        items = summary.get(section, [])
        if not isinstance(items, list):
            summary[section] = []
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            refs = item.get("refs", [])
            if not isinstance(refs, list):
                refs = []
            valid = [r for r in refs if isinstance(r, int) and 0 <= r <= max_idx]
            item["refs"] = valid

    summary.setdefault("title", "Tóm tắt cuộc họp")
    summary.setdefault("summary", "")
    summary.setdefault("key_points", [])
    summary.setdefault("decisions", [])
    summary.setdefault("action_items", [])
    summary.setdefault("open_issues", [])
    summary.setdefault("conclusion", None)

    return summary


def _empty_summary(reason: str) -> dict:
    return {
        "title": "Không thể tạo tóm tắt",
        "summary": reason,
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "open_issues": [],
        "conclusion": None,
    }


# === LLM Backend ===

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"


class MeetingSummarizer:
    """
    3-pass meeting summarizer:
      1. Extract: trích xuất facts từ transcript
      2. Summarize: tổng hợp thành JSON cấu trúc
      3. Self-check: model tự kiểm tra thiếu/thừa ý → fix

    Backend: Ollama API hoặc llama-cpp-python (auto-detect).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, ollama_url: str = OLLAMA_DEFAULT_URL,
                 model: str = OLLAMA_DEFAULT_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._backend = None  # 'ollama' or 'llama_cpp'
        self._llm = None  # llama_cpp.Llama instance (nếu dùng)

        self._init_backend()
        logger.info(f"Summarizer ready: backend={self._backend}, model={model}")

    def _init_backend(self):
        """Auto-detect backend: ưu tiên llama-cpp-python, fallback Ollama."""
        # Thử llama-cpp-python trước (control tốt hơn)
        try:
            from llama_cpp import Llama
            # Kiểm tra nếu ollama_url trỏ tới file GGUF local
            if os.path.isfile(self.ollama_url):
                n_threads = _get_physical_cores()
                self._llm = Llama(
                    model_path=self.ollama_url,
                    n_ctx=8192,           # Giảm từ 16K → 8K (đủ cho ~2h họp, nhanh hơn ~35%)
                    n_threads=n_threads,
                    n_threads_batch=n_threads,
                    n_batch=2048,         # Batch size lớn hơn → prompt processing nhanh hơn
                    n_ubatch=512,         # Physical batch → tối ưu compute
                    verbose=False,
                )
                self._backend = "llama_cpp"
                logger.info(f"Using llama-cpp-python backend (threads={n_threads})")
                return
        except (ImportError, Exception) as e:
            logger.debug(f"llama-cpp-python not available: {e}")

        # Fallback: Ollama API
        import requests
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            self._backend = "ollama"
            logger.info("Using Ollama API backend")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._backend = "ollama"  # Set anyway, will fail at runtime with clear error

    def _chat(self, system: str, user: str, max_tokens: int = 4096,
              temperature: float = 0.1, **kwargs) -> str:
        """Gọi LLM - auto-dispatch theo backend."""
        if self._backend == "llama_cpp" and self._llm:
            return self._chat_llama_cpp(system, user, max_tokens, temperature, **kwargs)
        return self._chat_ollama(system, user, max_tokens, temperature, **kwargs)

    def _chat_ollama(self, system: str, user: str, max_tokens: int = 4096,
                     temperature: float = 0.1, **kwargs) -> str:
        """Gọi Ollama API."""
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # Tắt thinking mode (Qwen3.5) — nhanh hơn, trả content trực tiếp
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": kwargs.get("top_p", 0.9),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "num_ctx": 16384,  # Context đủ cho ~1h họp
                "num_thread": _get_physical_cores(),  # Dùng hết physical cores
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=1800,  # 30 phút max
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        # Loại bỏ thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

    def _chat_llama_cpp(self, system: str, user: str, max_tokens: int = 4096,
                        temperature: float = 0.1, **kwargs) -> str:
        """Gọi llama-cpp-python trực tiếp (control tốt hơn)."""
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 0.9),
            repeat_penalty=kwargs.get("repeat_penalty", 1.1),
        )
        content = response["choices"][0]["message"]["content"]
        # Loại bỏ thinking tags nếu có
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

    def summarize(self, asr_data: dict,
                  progress_cb: Optional[Callable[[str], None]] = None) -> dict:
        """
        Pipeline 3 bước: Extract → Summarize → Self-check.
        Có chunking cho transcript dài.
        """
        # Format transcript
        transcript, text_seg_map, speaker_list, duration_min = format_transcript_for_llm(asr_data)

        if not transcript.strip():
            return _empty_summary("Bản ghi trống, không có nội dung để tóm tắt.")

        speaker_list_str = ", ".join(speaker_list) if speaker_list else "không xác định"
        num_speakers = len(speaker_list)

        # === Bước 1: TRÍCH XUẤT (extractive) ===
        if progress_cb:
            progress_cb("Đang trích xuất thông tin quan trọng...")

        # Chunking nếu transcript quá dài
        chunks = chunk_transcript(transcript, text_seg_map, max_chars=12000)

        all_facts = []
        for i, (chunk_text, chunk_ids) in enumerate(chunks):
            if len(chunks) > 1 and progress_cb:
                progress_cb(f"Đang trích xuất phần {i+1}/{len(chunks)}...")

            extract_prompt = EXTRACT_USER_TEMPLATE.format(
                duration_min=duration_min,
                num_speakers=num_speakers,
                speaker_list=speaker_list_str,
                transcript=chunk_text,
            )

            logger.info(f"Step 1 chunk {i+1}/{len(chunks)}: {len(chunk_text)} chars, "
                        f"{len(chunk_ids)} segments")

            facts = self._chat(
                system=EXTRACT_SYSTEM_PROMPT,
                user=extract_prompt,
                max_tokens=4096,
                temperature=0.1,
            )
            all_facts.append(facts)

        extracted_facts = "\n\n".join(all_facts)
        logger.info(f"Step 1 done: extracted {len(extracted_facts)} chars total")

        # === Bước 2: TÓM TẮT (abstractive từ facts) ===
        if progress_cb:
            progress_cb("Đang viết báo cáo tóm tắt...")

        summarize_prompt = SUMMARIZE_USER_TEMPLATE.format(
            duration_min=duration_min,
            speaker_list=speaker_list_str,
            extracted_facts=extracted_facts,
        )

        logger.info("Step 2: Generating structured summary JSON")

        summary_raw = self._chat(
            system=SUMMARIZE_SYSTEM_PROMPT,
            user=summarize_prompt,
            max_tokens=2048,
            temperature=0.05,
            top_p=0.85,
        )

        logger.info(f"Step 2 done: raw output {len(summary_raw)} chars")

        summary = parse_llm_json(summary_raw)
        if not summary:
            logger.warning("First JSON parse failed, retrying...")
            summary_raw = self._chat(
                system=SUMMARIZE_SYSTEM_PROMPT,
                user=summarize_prompt,
                max_tokens=2048,
                temperature=0.01,
            )
            summary = parse_llm_json(summary_raw)

        summary = validate_summary(summary, text_seg_map)

        # === Bước 3: SELF-CHECK (kiểm tra thiếu/thừa ý) ===
        if progress_cb:
            progress_cb("Đang kiểm tra chất lượng tóm tắt...")

        logger.info("Step 3: Self-check")

        # Format summary thành text để model check
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

        check_result = self._chat(
            system=SELFCHECK_SYSTEM_PROMPT,
            user=SELFCHECK_USER_TEMPLATE.format(
                extracted_facts=extracted_facts,
                summary_text=summary_text,
            ),
            max_tokens=1024,
            temperature=0.1,
        )

        logger.info(f"Step 3 check result: {check_result[:200]}...")

        # Nếu có lỗi → fix
        check_lower = check_result.lower()
        has_errors = not any(ok in check_lower for ok in [
            "ok", "không có lỗi", "không lỗi", "tất cả đều ok", "chính xác",
        ])

        if has_errors and len(check_result) > 20:
            if progress_cb:
                progress_cb("Đang sửa lỗi tóm tắt...")

            logger.info("Step 3: Fixing errors found by self-check")

            fix_raw = self._chat(
                system=SELFCHECK_FIX_SYSTEM,
                user=SELFCHECK_FIX_USER.format(
                    summary_json=summary_text,
                    errors=check_result,
                    extracted_facts=extracted_facts,
                ),
                max_tokens=2048,
                temperature=0.05,
            )

            fixed = parse_llm_json(fix_raw)
            if fixed:
                summary = validate_summary(fixed, text_seg_map)
                logger.info("Step 3: Summary fixed successfully")
            else:
                logger.warning("Step 3: Fix parse failed, keeping original")
        else:
            logger.info("Step 3: No errors found, summary OK")

        # Thêm metadata
        summary["_meta"] = {
            "duration_min": duration_min,
            "num_speakers": num_speakers,
            "num_segments": len(text_seg_map),
            "speakers": speaker_list,
            "backend": self._backend,
            "model": self.model,
            "chunks": len(chunks),
            "self_check": check_result[:200],
        }

        # Thêm text_seg_map cho frontend
        summary["_segments"] = {
            str(k): {"start_time": v["start_time"], "speaker": v["speaker"]}
            for k, v in text_seg_map.items()
        }

        return summary

    @classmethod
    def get_instance(cls, ollama_url: str = OLLAMA_DEFAULT_URL,
                     model: str = OLLAMA_DEFAULT_MODEL) -> "MeetingSummarizer":
        """Singleton getter."""
        with cls._lock:
            if cls._instance is not None:
                if (cls._instance.ollama_url == ollama_url.rstrip("/")
                        and cls._instance.model == model):
                    return cls._instance
                cls._instance = None
            cls._instance = cls(ollama_url, model)
            return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        with cls._lock:
            return cls._instance is not None


def is_summarizer_available() -> bool:
    """
    Kiểm tra summarizer khả dụng.
    True nếu: GGUF file tồn tại, hoặc Ollama có model, hoặc có thể tải model.
    """
    from web_service.config import server_config
    url_or_path = server_config.get("summarizer_model_path") or ""
    model = server_config.get("summarizer_ollama_model") or OLLAMA_DEFAULT_MODEL

    # Nếu là file GGUF local
    if url_or_path and os.path.isfile(url_or_path):
        return True

    # Nếu chưa cấu hình → check model mặc định có sẵn (hoặc có thể tải)
    if not url_or_path:
        default = get_default_model_path()
        # Model đã có sẵn hoặc có internet để tải
        return os.path.isfile(default) or True  # Luôn True — sẽ tải khi cần

    # Nếu là URL → check Ollama
    if url_or_path.startswith("http"):
        try:
            import requests
            resp = requests.get(f"{url_or_path.rstrip('/')}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False

    return False


def run_summarization(file_id: int, session_id: str,
                      send_ws: Callable, progress_cb: Callable):
    """Hàm chạy trong worker thread của queue_manager."""
    from web_service.config import server_config
    from web_service.database import db

    logger.info(f"Summarization started: file_id={file_id}")

    try:
        file_record = db.get_file(file_id)
        if not file_record or not file_record.get("asr_result_json"):
            raise ValueError("Không có kết quả ASR để tóm tắt")

        asr_data = json.loads(file_record["asr_result_json"])

        url_or_path = server_config.get("summarizer_model_path") or ""
        model = server_config.get("summarizer_ollama_model") or OLLAMA_DEFAULT_MODEL

        # Nếu chưa cấu hình → thử dùng model mặc định, tải nếu cần
        if not url_or_path:
            default_path = get_default_model_path()
            if not os.path.isfile(default_path):
                progress_cb("PHASE:Summary|Đang tải model (~2.7 GB)...|5")
                def _dl_progress(msg, pct):
                    progress_cb(f"PHASE:Summary|{msg}|{min(pct // 10, 9)}")
                default_path = download_model(progress_cb=_dl_progress)
            url_or_path = default_path

        progress_cb("PHASE:Summary|Đang khởi tạo...|10")

        summarizer = MeetingSummarizer.get_instance(url_or_path, model)

        def _step_progress(msg):
            if "trích xuất" in msg.lower():
                progress_cb(f"PHASE:Summary|{msg}|25")
            elif "báo cáo" in msg.lower() or "tóm tắt" in msg.lower():
                progress_cb(f"PHASE:Summary|{msg}|55")
            elif "kiểm tra" in msg.lower():
                progress_cb(f"PHASE:Summary|{msg}|75")
            elif "sửa" in msg.lower():
                progress_cb(f"PHASE:Summary|{msg}|85")
            else:
                progress_cb(f"PHASE:Summary|{msg}|50")

        summary = summarizer.summarize(asr_data, progress_cb=_step_progress)

        progress_cb("PHASE:Summary|Đang lưu kết quả...|95")

        summary_json = json.dumps(summary, ensure_ascii=False)
        db.update_file(file_id, summary_json=summary_json)

        logger.info(f"Summarization completed: file_id={file_id}")

        send_ws(session_id, {
            "type": "summary_complete",
            "file_id": file_id,
            "summary": summary,
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Summarization failed: file_id={file_id}: {error_msg}", exc_info=True)
        send_ws(session_id, {
            "type": "summary_error",
            "file_id": file_id,
            "error": error_msg,
        })
        raise
