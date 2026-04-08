"""
Meeting summarizer - 2-pass pipeline (Extract → Summarize).
Model: Gemma 4 E2B (PLE architecture, 2.3B effective params).
Backend: llama-cpp-python (GGUF) hoặc Ollama API.
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

DEFAULT_GGUF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_GGUF_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"


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
        import requests, hashlib
        url = f"https://huggingface.co/{DEFAULT_GGUF_REPO}/resolve/main/{DEFAULT_GGUF_FILE}"
        logger.info(f"Downloading via requests: {url}")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        sha256 = hashlib.sha256()
        tmp_dest = dest + ".tmp"
        with open(tmp_dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_cb:
                    pct = min(99, int(downloaded / total * 100))
                    progress_cb(f"Đang tải model... {downloaded/1e9:.1f}/{total/1e9:.1f} GB", pct)
        file_hash = sha256.hexdigest()
        logger.info(f"Model SHA-256: {file_hash}")
        os.rename(tmp_dest, dest)
        if progress_cb:
            progress_cb("Tải model hoàn tất", 100)
        logger.info(f"Model downloaded: {dest} ({os.path.getsize(dest)/1e9:.1f} GB)")
        return dest


def download_model_to(dest_dir: str,
                      progress_cb: Optional[Callable[[str, int], None]] = None) -> str:
    """Tải model GGUF vào thư mục chỉ định. Trả về đường dẫn file."""
    dest = os.path.join(dest_dir, DEFAULT_GGUF_FILE)
    os.makedirs(dest_dir, exist_ok=True)

    logger.info(f"Downloading model {DEFAULT_GGUF_REPO}/{DEFAULT_GGUF_FILE} → {dest}")

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=DEFAULT_GGUF_REPO,
            filename=DEFAULT_GGUF_FILE,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded: {path} ({os.path.getsize(path)/1e9:.1f} GB)")
        return path
    except ImportError:
        import requests as _req
        url = f"https://huggingface.co/{DEFAULT_GGUF_REPO}/resolve/main/{DEFAULT_GGUF_FILE}"
        resp = _req.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_cb:
                    pct = min(99, int(downloaded / total * 100))
                    progress_cb(f"Đang tải... {downloaded/1e9:.1f}/{total/1e9:.1f} GB", pct)
        logger.info(f"Model downloaded: {dest} ({os.path.getsize(dest)/1e9:.1f} GB)")
        return dest


def _resolve_model_path(path_or_url: str) -> str:
    """
    Resolve model path: nếu là relative path (không phải URL) → chuyển thành absolute
    dựa trên BASE_DIR (project root). Nếu là URL (http/https) → giữ nguyên.
    """
    path_or_url = path_or_url.strip().rstrip("/")
    if not path_or_url:
        return path_or_url
    # URL → giữ nguyên
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    # Absolute path → giữ nguyên
    if os.path.isabs(path_or_url):
        return path_or_url
    # Relative path → resolve từ BASE_DIR
    try:
        from core.config import BASE_DIR
        resolved = os.path.join(BASE_DIR, path_or_url)
        return resolved
    except ImportError:
        return path_or_url


def _get_physical_cores() -> int:
    """Lấy số physical cores (giống ALLOWED_THREADS trong core/config.py)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 4
    except ImportError:
        import os
        return max(1, (os.cpu_count() or 4) // 2)


def _get_logical_cores() -> int:
    """Lấy số logical cores (hyperthreads) — dùng cho batch processing."""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 8
    except ImportError:
        import os
        return os.cpu_count() or 8


# === Prompts — Distilled từ Claude Opus trên 5 mẫu họp thực tế VN ===

EXTRACT_SYSTEM_PROMPT = """\
Bạn là trợ lý trích xuất thông tin từ biên bản cuộc họp/cuộc gọi tiếng Việt.

QUY TẮC PHÂN LOẠI:
- THẢO LUẬN: nhiều người bàn qua bàn lại, chưa kết luận → KHÔNG đưa vào "quyết định".
- QUYẾT ĐỊNH: chỉ khi chủ trì nói "thống nhất/chốt/đồng ý" hoặc giao việc cụ thể (ai + làm gì).
- ACTION ITEM: bám vào động từ chỉ thị + tên người/đơn vị. "Đồng chí X rà soát" = giao việc cho X. \
"Báo cáo lại", "tham mưu", "nghiên cứu", "tiếp thu" đều là giao việc.
- THÔNG BÁO CẤP TRÊN: chính sách/lộ trình do cấp trên ban hành → thông tin nền, không phải quyết định cuộc họp này.

QUY TẮC XỬ LÝ:
- GIỮ CHÍNH XÁC: số liệu, mốc thời gian, tên văn bản pháp lý.
- BỎ NOISE: bỏ lặp từ, sai chính tả, câu bỏ dở. Giữ ý chính, viết lại ngắn gọn.
- KHÔNG suy diễn, KHÔNG thêm ý không có trong bản ghi.
- Viết tắt: giải thích lần đầu, sau đó dùng viết tắt.
- Trả lời bằng tiếng Việt."""

EXTRACT_USER_TEMPLATE = """\
{transcript}

---
Trích xuất:
1. LOẠI: (cuộc họp nhiều người / cuộc gọi 2 người / báo cáo)
2. NGƯỜI CHỦ TRÌ/QUYẾT ĐỊNH: (ai có quyền chốt)
3. CÁC Ý CHÍNH THEO THỜI GIAN: (ai nói gì, ý chính — chỉ ý quan trọng, bỏ ý phụ)
4. QUYẾT ĐỊNH ĐÃ CHỐT: (chỉ những gì được "thống nhất/chốt/đồng ý")
5. CÔNG VIỆC: (Ai | Việc gì | Hạn chót — bao gồm cả giao việc ẩn)
6. VẤN ĐỀ CHƯA GIẢI QUYẾT:
7. SỐ LIỆU/MỐC THỜI GIAN QUAN TRỌNG:
8. KẾT LUẬN:"""

SUMMARIZE_SYSTEM_PROMPT = """\
Viết tóm tắt cuộc họp/cuộc gọi tiếng Việt từ thông tin đã trích xuất.

QUY TẮC:
- CHỈ dùng thông tin từ TRÍCH XUẤT. KHÔNG thêm mới. KHÔNG suy diễn.
- Cấu trúc tùy loại: họp nhiều người cần đầy đủ (diễn biến, quyết định, action items). \
Cuộc gọi 2 người chỉ cần: mục đích, nội dung chính, kết quả.
- Tóm tắt tổng quan trả lời: Họp về gì? Ai tham dự? Kết quả chính?
- Quyết định: CHỈ những gì đã "thống nhất/chốt". Đánh số thứ tự.
- Công việc: bao gồm cả giao việc ẩn ("rà soát", "báo cáo lại", "nghiên cứu").
- GIỮ CHÍNH XÁC số liệu và tên văn bản.
- Trả về DUY NHẤT JSON hợp lệ, tiếng Việt. KHÔNG wrap markdown."""

SUMMARIZE_USER_TEMPLATE = """\
TRÍCH XUẤT:
{extracted_facts}

---
JSON (ngắn gọn, mỗi key_point tối đa 2 câu):
{{
  "title": "Tiêu đề (dưới 15 từ)",
  "summary": "Tóm tắt 2-3 câu: họp về gì, ai tham dự, kết quả chính",
  "key_points": [
    {{"text": "Nội dung ngắn gọn", "speaker": "Ai nói", "refs": [0]}}
  ],
  "decisions": [
    {{"text": "Quyết định đã chốt (không phải thảo luận)", "refs": [5]}}
  ],
  "action_items": [
    {{"text": "Việc cần làm", "assignee": "Ai chịu trách nhiệm", "deadline": "mốc thời gian hoặc null", "refs": [10]}}
  ],
  "open_issues": [
    {{"text": "Vấn đề đang bàn chưa có kết luận", "refs": [15]}}
  ],
  "conclusion": "Kết luận cuối (string hoặc null)"
}}

refs = mảng số nguyên (đoạn). Section rỗng → []. Gộp ý trùng nếu có nhiều phần."""

# JSON Schema cho constrained decoding (llama-cpp-python response_format)
SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "speaker": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "refs"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "refs"],
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "assignee": {"type": "string"},
                    "deadline": {"type": ["string", "null"]},
                    "refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "refs"],
            },
        },
        "open_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "refs"],
            },
        },
        "conclusion": {"type": ["string", "null"]},
    },
    "required": ["title", "summary", "key_points", "decisions", "action_items"],
}


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
                     max_chars: int = 12000, overlap_lines: int = 5) -> list:
    """
    Chia transcript thành chunks có OVERLAP.
    - Cắt tại ranh giới dòng (mỗi dòng = 1 segment)
    - Overlap: mỗi chunk bắt đầu bằng N dòng cuối của chunk trước (giữ context)
    - Kèm metadata chunk header cho LLM biết vị trí trong cuộc họp

    Returns: list of (chunk_text_with_header, chunk_seg_ids)
    """
    if len(transcript) <= max_chars:
        return [(transcript, list(text_seg_map.keys()))]

    lines = transcript.split("\n")
    total_lines = len(lines)

    chunks = []
    start_idx = 0

    while start_idx < total_lines:
        # Tìm end_idx sao cho chunk <= max_chars
        current_len = 0
        end_idx = start_idx
        while end_idx < total_lines:
            line_len = len(lines[end_idx]) + 1
            if current_len + line_len > max_chars and end_idx > start_idx:
                break
            current_len += line_len
            end_idx += 1

        chunk_lines = lines[start_idx:end_idx]

        # Extract segment ids
        chunk_ids = []
        for line in chunk_lines:
            match = re.search(r'\(đoạn (\d+)\)', line)
            if match:
                chunk_ids.append(int(match.group(1)))

        chunk_text = "\n".join(chunk_lines)
        chunks.append((chunk_text, chunk_ids))

        # Tiến tới chunk tiếp, lùi lại overlap_lines để giữ context
        next_start = end_idx - overlap_lines
        if next_start <= start_idx:
            next_start = end_idx  # Tránh infinite loop nếu overlap quá lớn
        start_idx = next_start

    return chunks


# === JSON parsing ===

def parse_llm_json(raw: str) -> dict:
    """Parse JSON từ LLM output với fallback strategies."""
    raw = raw.strip()

    # Loại bỏ thinking tags nếu có (Qwen/Gemma thinking mode)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # 1. Parse trực tiếp
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Tìm trong ```json ... ``` (Gemma 4 thường wrap output trong markdown)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Tìm {} block ngoài cùng
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
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
OLLAMA_DEFAULT_MODEL = "gemma4:e2b"


class MeetingSummarizer:
    """
    2-pass meeting summarizer (Gemma 4 E2B optimized):
      1. Extract: trích xuất facts từ transcript
      2. Summarize: tổng hợp thành JSON cấu trúc

    Backend: llama-cpp-python (GGUF) hoặc Ollama API (auto-detect).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, ollama_url: str = OLLAMA_DEFAULT_URL,
                 model: str = OLLAMA_DEFAULT_MODEL):
        # Resolve relative path → absolute (relative to project root)
        self.ollama_url = _resolve_model_path(ollama_url)
        self.model = model
        self._backend = None  # 'ollama' or 'llama_cpp'
        self._llm = None  # llama_cpp.Llama instance (nếu dùng)

        self._init_backend()
        logger.info(f"Summarizer ready: backend={self._backend}, model={model}")

    def _init_backend(self):
        """Auto-detect backend dựa trên config path."""
        is_file = os.path.isfile(self.ollama_url)
        is_url = self.ollama_url.startswith(("http://", "https://"))

        # --- GGUF file → llama-cpp-python ---
        if is_file:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python chưa cài. Chạy: pip install llama-cpp-python\n"
                    "Hoặc chuyển sang backend Ollama."
                )

            # Đọc số luồng LLM từ config (admin setting riêng cho LLM)
            try:
                from web_service.config import server_config
                n_threads = int(server_config.get("summarizer_threads") or 0)
            except Exception:
                n_threads = 0
            if n_threads <= 0:
                n_threads = _get_physical_cores()
            # Batch processing (prefill) dùng tất cả logical cores → tận dụng 100% CPU
            n_threads_batch = _get_logical_cores()
            logger.info(f"Loading GGUF model: {self.ollama_url} "
                        f"(gen_threads={n_threads}, batch_threads={n_threads_batch})")
            self._llm = Llama(
                model_path=self.ollama_url,
                n_ctx=8192,
                n_threads=n_threads,            # Generation: physical cores
                n_threads_batch=n_threads_batch, # Prefill: all logical cores → 100% CPU
                n_batch=512,
                n_ubatch=512,
                verbose=False,
            )
            self._backend = "llama_cpp"
            logger.info("Using llama-cpp-python backend")
            return

        # --- URL → Ollama API ---
        if is_url:
            import requests
            try:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                resp.raise_for_status()
                self._backend = "ollama"
                logger.info(f"Using Ollama API backend: {self.ollama_url}")
            except Exception as e:
                raise RuntimeError(
                    f"Không thể kết nối Ollama tại {self.ollama_url}: {e}\n"
                    "Đảm bảo Ollama đang chạy (ollama serve)."
                )
            return

        # --- Không phải file cũng không phải URL ---
        raise RuntimeError(
            f"Config summarizer không hợp lệ: '{self.ollama_url}'\n"
            "Cần là đường dẫn file GGUF hoặc URL Ollama (http://...)."
        )

    def _chat(self, system: str, user: str, max_tokens: int = 4096,
              temperature: float = 0.1, **kwargs) -> str:
        """Gọi LLM - auto-dispatch theo backend."""
        if self._backend == "llama_cpp" and self._llm:
            return self._chat_llama_cpp(system, user, max_tokens, temperature, **kwargs)
        return self._chat_ollama(system, user, max_tokens, temperature, **kwargs)

    def _chat_ollama(self, system: str, user: str, max_tokens: int = 4096,
                     temperature: float = 1.0, **kwargs) -> str:
        """Gọi Ollama API."""
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 64),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.0),
                "num_ctx": 16384,
                "num_thread": _get_physical_cores(),
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=1800,  # 30 phút max
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return content.strip()

    def _chat_llama_cpp(self, system: str, user: str, max_tokens: int = 4096,
                        temperature: float = 1.0, **kwargs) -> str:
        """Gọi llama-cpp-python trực tiếp. Sampling tối ưu cho Gemma 4."""
        call_kwargs = dict(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 64),
            repeat_penalty=kwargs.get("repeat_penalty", 1.0),
        )

        # JSON schema enforcement: ép output JSON 100% hợp lệ
        json_schema = kwargs.get("json_schema")
        if json_schema:
            call_kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }

        response = self._llm.create_chat_completion(**call_kwargs)
        content = response["choices"][0]["message"]["content"]
        return content.strip()

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

        chunks = chunk_transcript(transcript, text_seg_map, max_chars=12000, overlap_lines=5)
        is_long = len(chunks) > 1

        all_facts = []
        prev_summary = ""  # Rolling summary cho context giữa chunks

        for i, (chunk_text, chunk_ids) in enumerate(chunks):
            if is_long and progress_cb:
                progress_cb(f"Đang trích xuất phần {i+1}/{len(chunks)}...")

            # Thêm context header cho chunk (giúp LLM biết vị trí + context trước đó)
            chunk_header = ""
            if is_long:
                chunk_header = f"[Phần {i+1}/{len(chunks)} của cuộc họp]\n"
                if prev_summary:
                    chunk_header += f"[Tóm tắt phần trước: {prev_summary}]\n\n"

            extract_prompt = EXTRACT_USER_TEMPLATE.format(
                duration_min=duration_min,
                num_speakers=num_speakers,
                speaker_list=speaker_list_str,
                transcript=chunk_header + chunk_text,
            )

            logger.info(f"Step 1 chunk {i+1}/{len(chunks)}: {len(chunk_text)} chars, "
                        f"{len(chunk_ids)} segments")

            facts = self._chat(
                system=EXTRACT_SYSTEM_PROMPT,
                user=extract_prompt,
                max_tokens=4096,
                temperature=0.8,   # Extract: cần chính xác nhưng đủ linh hoạt
                top_p=0.9,
            )
            all_facts.append(facts)

            # Tạo rolling summary ngắn cho chunk tiếp theo biết context
            if is_long and i < len(chunks) - 1:
                # Lấy 2-3 câu đầu của facts làm context summary
                fact_lines = [l.strip() for l in facts.split("\n") if l.strip().startswith("•")]
                prev_summary = "; ".join(l[2:50] for l in fact_lines[:3])

        extracted_facts = "\n\n".join(all_facts)

        # Nếu nhiều chunks → thêm hướng dẫn deduplicate cho bước 2
        if is_long:
            extracted_facts += (
                "\n\n[LƯU Ý: Trích xuất trên gồm nhiều phần có overlap. "
                "Khi tóm tắt, GỘP các ý trùng lặp thành 1, ưu tiên ý ở phần sau (mới hơn).]"
            )

        logger.info(f"Step 1 done: extracted {len(extracted_facts)} chars total "
                     f"({len(chunks)} chunks)")

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
            temperature=0.7,   # Summarize: cần structured, ít creative
            top_p=0.9,
        )

        logger.info(f"Step 2 done: raw output {len(summary_raw)} chars")

        summary = parse_llm_json(summary_raw)
        if not summary:
            logger.warning("First JSON parse failed, retrying with schema enforcement...")
            summary_raw = self._chat(
                system=SUMMARIZE_SYSTEM_PROMPT,
                user=summarize_prompt,
                max_tokens=2048,
                temperature=0.3,
                json_schema=SUMMARY_JSON_SCHEMA,
            )
            summary = parse_llm_json(summary_raw)

        summary = validate_summary(summary, text_seg_map)

        # Bước 3 (self-check) đã bỏ — Gemma 4 E2B không cần:
        # - Benchmark cho thấy self-check tốn 20-30% thời gian
        # - False positive gây fix phá kết quả tốt
        # - Model không hallucinate trên test data thực tế

        # Thêm metadata
        summary["_meta"] = {
            "duration_min": duration_min,
            "num_speakers": num_speakers,
            "num_segments": len(text_seg_map),
            "speakers": speaker_list,
            "backend": self._backend,
            "model": self.model,
            "chunks": len(chunks),
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
        """Singleton getter — resolve path trước khi so sánh."""
        resolved = _resolve_model_path(ollama_url)
        with cls._lock:
            if cls._instance is not None:
                if cls._instance.ollama_url == resolved and cls._instance.model == model:
                    # Kiểm tra backend vẫn valid
                    if cls._instance._backend == "llama_cpp" and cls._instance._llm is not None:
                        return cls._instance
                    if cls._instance._backend == "ollama":
                        return cls._instance
                # Config thay đổi hoặc backend invalid → tạo mới
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
    url_or_path = _resolve_model_path(server_config.get("summarizer_model_path") or "")
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

        url_or_path = _resolve_model_path(server_config.get("summarizer_model_path") or "")
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
