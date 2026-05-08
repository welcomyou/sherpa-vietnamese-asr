# core/ - Thư viện dùng chung giữa desktop app và web service
# KHÔNG import PyQt6 - pure Python

from core.config import (
    DEBUG_LOGGING,
    BASE_DIR,
    CONFIG_FILE,
    COLORS,
    ALLOWED_THREADS,
    DEFAULT_THREADS,
    compute_ort_threads,
    MODEL_DOWNLOAD_INFO,
    get_speaker_embedding_models,
    is_diarization_available,
    ensure_bpe_vocab,
    prepare_hotwords_file,
    get_hotwords_config,
)

from core.utils import (
    normalize_vietnamese,
    fuzzy_score,
    find_fuzzy_matches,
)

from core.asr_json import (
    serialize_segments,
    deserialize_segments,
    deserialize_overlap_segments,
    load_asr_json,
    save_asr_json,
)

from core.asr_engine import (
    TranscriberPipeline,
    merge_chunks_with_overlap,
    split_long_segments,
    load_audio,
    _find_ffmpeg,
    find_silent_regions,
    get_ort,
)

from core.speaker_diarization import (
    run_diarization,
)
