let installPrompt = null;
let manifest = null;
let selectedAudioFile = null;
let vadSession = null;
let ortConfigured = false;
let lastAudioPcm = null;
let asrWorker = null;
let asrWorkerScript = null;
let asrInitPromise = null;
let asrLoadedModelId = null;
let asrLoadedConfigKey = null;
let asrRequestId = 0;
const asrRequests = new Map();
let punctuationSession = null;
let punctuationVocab = null;
let punctuationInitPromise = null;
let punctuationExecutionProvider = "wasm";
let diarizationSession = null;
let diarizationInitPromise = null;
let vadExecutionProvider = "wasm";
let diarizationExecutionProvider = "wasm";
let camppSession = null;
let camppInitPromise = null;
let camppMelBank = null;
let camppWindow = null;
let wespeakerMelBank = null;
let wespeakerWindow = null;
let fftTables = null;
let overlapSession = null;
let overlapInitPromise = null;
let pyannoteEmbeddingSession = null;
let pyannoteCommunityInitPromise = null;
let pyannoteCommunityAssets = null;
let dnsmosSession = null;
let dnsmosInitPromise = null;
let camppExecutionProvider = "wasm";
let pyannoteEmbeddingExecutionProvider = "wasm";
let dnsmosExecutionProvider = "wasm";
let overlapExecutionProvider = "wasm";
let offlineBootstrapReady = false;
let offlineBootstrapBusy = false;
let editorState = null;
let editorPreviewAudioUrl = "";
let speakerDialogContext = null;
let editorLastAutoScrollAt = 0;
let selectedLibraryItemId = null;
let selectedLibraryImportPromise = null;
let libraryDbPromise = null;
let zstdModulePromise = null;
let libraryAutosaveTimer = null;
let _allMeetings = [];
let _meetingsPage = 1;
let _searchTimeout = null;
let applyingUserConfig = false;
let userConfigSaveTimer = null;
let autoBootstrapAttempted = false;
let offlineBootstrapError = "";
let bootstrapReloadScheduled = false;
let pipelineLogLines = [];
let debugLogEntries = [];
let benchmarkProviderMode = null;
let calibrationProfile = null;
let calibrationBusy = false;
let autoCalibrationAttempted = false;
let calibrationSkipRequested = false;
let screenWakeLock = null;
let screenWakeLockRequest = null;
let screenWakeLockUnsupportedLogged = false;
const screenWakeLockReasons = new Set();
let activeResumeAfterKillContext = null;
let opfsWritableSupportPromise = null;
const RESUME_AFTER_KILL_PREFIX = "resume_after_kill_";
const RESUME_AFTER_KILL_KEEP_SNAPSHOTS = 2;
const RESUME_AFTER_KILL_DURABLE_STAGES = new Set([
  "vad",
  "asr_chunks",
  "asr",
  "quality",
  "campp_speech_regions",
  "pyannote_segmentation",
  "diarization",
  "punctuation",
  "overlap",
]);

// --- Buffer Pool: preallocated TypedArrays để giảm GC pressure trong inference loops ---
let diarBatchBuf = null;         // Float32Array[batchSize * DIAR_CHUNK_SAMPLES] — dùng cho cả 2 diar fns
let camppDataBuf = null;         // Float32Array[batch * maxFrames * 80] — CAM++ batch input
let wespeakerInputBuf = null;    // Float32Array[batchSize * maxFrames * 80] — Pyannote emb input
let vadInputBuf = null;          // Float32Array[VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE]
let vadStateBuf = null;          // Float32Array[2 * 1 * 128] — Silero VAD state

function ensureBuf(current, needed) {
  if (current && current.length >= needed) return current;
  return new Float32Array(Math.ceil(needed * 1.25));
}

const VAD_SAMPLE_RATE = 16000;
const AUDIO_DECODER_WORKER = "/js/ffmpeg-decode-worker.js";
// FFmpeg WASM must not force soxr/swr: some browser builds do not expose those engine names.
const AUDIO_DECODER_RESAMPLER = "ffmpeg-default";
const VAD_WINDOW_SIZE = 512;
const VAD_CONTEXT_SIZE = 64;
const ASR_CHUNK_SECONDS = 30;
const ASR_CHUNK_OVERLAP_SECONDS = 3;
const ASR_MERGE_GAP_SECONDS = 5;
const ASR_MIN_CHUNK_SECONDS = 1;
const ASR_SILENCE_THRESHOLD = 0.01;
const ASR_MIN_SILENCE_SECONDS = 0.3;
const ASR_SPLIT_SEARCH_SECONDS = 2;
const ASR_MIN_SPLIT_ADVANCE_SECONDS = 20;
const WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;
const ASR_FILLER_WORDS = new Set(["à", "ờ", "ừ", "ơ", "uh", "um"]);
const PUNCT_MAX_LEN = 80;
const PUNCT_MIN_LEN = 3;
const PUNCT_CHUNK_SIZE = 56;
const PUNCT_OVERLAP_SIZE = 16;
const PUNCT_MIN_WORDS_CUT = 6;
const PUNCT_STRIDE = PUNCT_CHUNK_SIZE - PUNCT_OVERLAP_SIZE;
const PUNCT_ITERATIONS = 3;
const PUNCT_MINI_BATCH_SIZE = 8;
const DEFAULT_PUNCT_SLIDER = 6;
const DEFAULT_CASE_SLIDER = 6;
const DEFAULT_CPU_THREADS = 4;
const DEFAULT_PUNCT_CONFIDENCE = 0.5 - (DEFAULT_PUNCT_SLIDER - 1) * (1.3 / 9);
const DEFAULT_PUNCT_CASE_CONFIDENCE = -1.5 + (DEFAULT_CASE_SLIDER - 1) * (2.0 / 9);
const DNSMOS_SAMPLE_LENGTH = 144160;
const DNSMOS_MIN_SAMPLES = 8000;
const REQUIRED_OFFLINE_PACK_IDS = ["base_asr_30m", "required_pwa_offline_models"];
const ASR_JSON_VERSION = 1;
const LIBRARY_DB_NAME = "asr-vn-offline-library";
const LIBRARY_DB_VERSION = 2;
const LIBRARY_STORE = "items";
const FILE_STORE = "files";
const MODEL_CACHE_NAME = "asr-vn-offline-model-files-v1";
const LIBRARY_SOURCE_FILE = "source.bin";
const LIBRARY_RESULT_FILE = "result.asr.json.zst";
const MEETINGS_PER_PAGE = 20;
const USER_CONFIG_KEY = "asr-vn-offline-config-v1";
const USER_CONFIG_SCHEMA = 8;
const CALIBRATION_PROFILE_KEY = "asr-vn-offline-calibration-v1";
const CALIBRATION_LAST_REPORT_KEY = "asr-vn-offline-calibration-report-v1";
const MANIFEST_STORAGE_KEY = "asr-vn-offline-model-manifest-v1";
const OFFLINE_PWA_CODE_VERSION = "offline-pwa-v123";
const OFFLINE_RUNTIME_CACHE_NAME = OFFLINE_PWA_CODE_VERSION;
const CALIBRATION_CODE_VERSION = "calibration-v1-provider-stage-tolerance";
const CALIBRATION_SAMPLE_URL = "/calibration/1hour_qh_10min.mp3";
const CALIBRATION_SAMPLE_NAME = "1hour_qh_10min.mp3";
const STARTUP_FETCH_TIMEOUT_MS = 1500;
const DOWNLOAD_RESPONSE_TIMEOUT_MS = 15000;
const MODEL_DOWNLOAD_STALL_TIMEOUT_MS = 45000;
const PWA_INSTALLED_FLAG_KEY = "asr-vn-offline-installed-v1";
const WEBGPU_CALIBRATION_STAGE_NAMES = new Set([
  "CAM++ speaker embedding",
  "Pyannote Community-1 embedding encoder",
  "DNSMOS quality",
  "ViBERT punctuation",
  "ViBERT punctuation fp32",
]);
const OFFLINE_RUNTIME_ASSET_URLS = Object.freeze([
  "/",
  "/index.html",
  "/manifest.json",
  "/api/model-manifest",
  "/hotword.txt",
  "/shared/css/style.css",
  "/shared/js/about.js",
  "/shared/js/status.js",
  "/css/app.css",
  "/js/app.js",
  "/js/asr-worker.js",
  "/js/ffmpeg-decode-worker.js",
  "/js/pure-ort-asr-worker.js",
  "/vendor/ffmpeg/ffmpeg/classes.js",
  "/vendor/ffmpeg/ffmpeg/const.js",
  "/vendor/ffmpeg/ffmpeg/errors.js",
  "/vendor/ffmpeg/ffmpeg/index.js",
  "/vendor/ffmpeg/ffmpeg/types.js",
  "/vendor/ffmpeg/ffmpeg/utils.js",
  "/vendor/ffmpeg/ffmpeg/worker.js",
  "/vendor/ffmpeg/core/ffmpeg-core.js",
  "/vendor/ffmpeg/core/ffmpeg-core.wasm",
  "/vendor/onnxruntime-web/ort.wasm.min.js",
  "/vendor/onnxruntime-web/ort.webgpu.min.js",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs",
  "/vendor/longform-clustering/longform-clustering.js",
  "/vendor/mpg123-decoder/mpg123-decoder.min.js",
  "/vendor/zstd-wasm/zstd-wrapper.js",
  "/vendor/zstd-wasm/zstd.js",
  "/vendor/zstd-wasm/zstd.wasm",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-asr.js",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.js",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.wasm",
  "/calibration/1hour_qh_10min.mp3",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
]);
const DEFAULT_THEME = "dark";
const DEFAULT_UI_TEXT_SCALE = 110;
const DEFAULT_HOTWORDS_SCORE = 1.5;
const MAX_HOTWORDS = 500;
const SPEAKER_COLORS = [
  "#2ea3f2",
  "#31c48d",
  "#f4b740",
  "#f05252",
  "#a78bfa",
  "#14b8a6",
  "#f97316",
  "#e879f9",
  "#60a5fa",
  "#84cc16",
];
const START_TOKEN = "$START";
const PAD_TOKEN = "@@PADDING@@";
const UNK_TOKEN = "@@UNKNOWN@@";
const PUNCT_LABELS = [
  "$KEEP",
  "$TRANSFORM_CASE_CAPITAL",
  "$APPEND_,",
  "$APPEND_.",
  "$TRANSFORM_VERB_VB_VBN",
  "$TRANSFORM_CASE_UPPER",
  "$APPEND_:",
  "$APPEND_?",
  "$TRANSFORM_VERB_VB_VBC",
  "$TRANSFORM_CASE_LOWER",
  "$TRANSFORM_CASE_CAPITAL_1",
  "$TRANSFORM_CASE_UPPER_-1",
  "$MERGE_SPACE",
  UNK_TOKEN,
  PAD_TOKEN,
];
const PUNCT_D_TAGS = ["CORRECT", "INCORRECT", UNK_TOKEN, PAD_TOKEN];
const PUNCT_ALLOWED = new Set([":", ".", ",", "?"]);
const PUNCT_NOOP_INDEX = PUNCT_LABELS.indexOf("$KEEP");
const PUNCT_INCORR_INDEX = PUNCT_D_TAGS.indexOf("INCORRECT");
const PUNCT_APPEND_PERIOD_INDEX = PUNCT_LABELS.indexOf("$APPEND_.");
const PUNCT_APPEND_COMMA_INDEX = PUNCT_LABELS.indexOf("$APPEND_,");
const PUNCT_CASE_INDEXES = PUNCT_LABELS
  .map((label, index) => (label.startsWith("$TRANSFORM_CASE_") ? index : -1))
  .filter((index) => index >= 0);
const DIAR_CHUNK_SECONDS = 10;
const DIAR_STEP_SECONDS = 5;
const DIAR_CHUNK_SAMPLES = DIAR_CHUNK_SECONDS * VAD_SAMPLE_RATE;
const DIAR_STEP_SAMPLES = DIAR_STEP_SECONDS * VAD_SAMPLE_RATE;
const DIAR_BATCH_SIZE = 16;
const WEBGPU_DIAR_BATCH_SIZE = 64;
const PYANNOTE_STEP_SECONDS = 1;
const PYANNOTE_STEP_SAMPLES = PYANNOTE_STEP_SECONDS * VAD_SAMPLE_RATE;
const PYANNOTE_MAX_SPEAKERS_PER_CHUNK = 3;
const PYANNOTE_RF_START = 0;
const PYANNOTE_RF_DURATION = 0.0619375;
const PYANNOTE_RF_STEP = 0.016875;
const PYANNOTE_DEFAULT_THRESHOLD = 0.6;
const PYANNOTE_FA = 0.07;
const PYANNOTE_FB = 0.8;
const PYANNOTE_EMB_MIN_NUM_SAMPLES = 1680;
const WESPEAKER_FRAME_LENGTH = 400;
const WESPEAKER_FRAME_SHIFT = 160;
const WESPEAKER_N_FFT = 512;
const WESPEAKER_NUM_MEL_BINS = 80;
const WESPEAKER_LOW_FREQ = 20;
const WESPEAKER_PREEMPHASIS = 0.97;
const WESPEAKER_BATCH_SIZE = 8;
const DIAR_POWERSET = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 0],
  [1, 0, 1],
  [0, 1, 1],
];
const CAMPP_WINDOW_SECONDS = 1.5;
const CAMPP_STEP_SECONDS = 0.6;
const CAMPP_FRAME_LENGTH = 400;
const CAMPP_FRAME_SHIFT = 160;
const CAMPP_N_FFT = 512;
const CAMPP_NUM_MEL_BINS = 80;
const CAMPP_LOW_FREQ = 20;
const CAMPP_PREEMPHASIS = 0.97;
const CAMPP_ENERGY_FLOOR = 1.0;
const CAMPP_BATCH_SIZE = 32;
const CAMPP_MERGE_COS = 0.875;
const CAMPP_MIN_CLUSTER_SIZE = 4;
const CAMPP_JACOBI_MAX_WINDOWS = 256;
const CAMPP_SPECTRAL_MAX_WINDOWS = 2400;
const CAMPP_LANCZOS_ITERATIONS = 256;
const OVERLAP_MIN_REGION_SECONDS = 0.4;
const OVERLAP_MIN_DECODE_SECONDS = 1.0;
const OVERLAP_MIN_REF_SECONDS = 1.0;
const DEFAULT_ASR_MODEL_ID = "sherpa-onnx-zipformer-vi-2025-04-20";
const ROVER_MODEL_ID = "rover-voting";
const ASR_BACKEND_WORKERS = {
  pure_ort: "/js/pure-ort-asr-worker.js",
  sherpa_wasm: "/js/asr-worker.js",
};
const ASR_MODELS = {
  "zipformer-30m-rnnt-6000h": {
    type: "single",
    backend: "pure_ort",
    label: "hynt/Zipformer-30M (nhanh)",
    files: {
      encoder: "asr30.encoder",
      decoder: "asr30.decoder",
      joiner: "asr30.joiner",
      tokens: "asr30.tokens",
      bpeVocab: "asr30.bpe_vocab",
    },
    decodingMethod: "modified_beam_search",
    maxActivePaths: 8,
  },
  "sherpa-onnx-zipformer-vi-2025-04-20": {
    type: "single",
    backend: "pure_ort",
    label: "Zipformer-Vi 2025 (68M)",
    files: {
      encoder: "asr68.encoder",
      decoder: "asr68.decoder",
      joiner: "asr68.joiner",
      tokens: "asr68.tokens",
      bpeVocab: "asr68.bpe_vocab",
    },
    decodingMethod: "modified_beam_search",
    maxActivePaths: 8,
  },
  "sherpa-onnx-zipformer-vi-2025-04-20-sherpa-wasm": {
    type: "single",
    backend: "sherpa_wasm",
    label: "Zipformer-Vi 2025 (68M, sherpa-wasm benchmark)",
    files: {
      encoder: "asr68.encoder",
      decoder: "asr68.decoder",
      joiner: "asr68.joiner",
      tokens: "asr68.tokens",
    },
    decodingMethod: "modified_beam_search",
    maxActivePaths: 8,
  },
  "rover-voting": {
    type: "rover",
    label: "ROVER (chậm, chính xác)",
    modelIds: ["zipformer-30m-rnnt-6000h", "sherpa-onnx-zipformer-vi-2025-04-20"],
  },
};
const SUPPORTED_SPEAKER_MODELS = new Set(["senko_campp_optimized", "pyannote_community1_vbx"]);
const DEFAULT_USER_CONFIG = {
  schemaVersion: USER_CONFIG_SCHEMA,
  theme: DEFAULT_THEME,
  uiTextScale: DEFAULT_UI_TEXT_SCALE,
  asrModel: DEFAULT_ASR_MODEL_ID,
  cpuThreads: null,
  punctuationLevel: DEFAULT_PUNCT_SLIDER,
  caseLevel: DEFAULT_CASE_SLIDER,
  speakerDiarization: true,
  speakerCount: "0",
  speakerModel: "senko_campp_optimized",
  rmsNormalize: false,
  bypassVad: false,
  resumeAfterKill: true,
  overlapSeparation: false,
  saveRam: true,
  hotwordsEnabled: true,
  hotwordsScore: DEFAULT_HOTWORDS_SCORE,
  hotwords: [],
};

const $ = (id) => document.getElementById(id);

function boundedNumber(value, fallback, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function boundedFloat(value, fallback, min, max) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function logicalThreadCount() {
  const reported = Number(navigator.hardwareConcurrency || 0);
  return Number.isFinite(reported) && reported > 0 ? reported : DEFAULT_CPU_THREADS;
}

function maxWasmThreads() {
  if (!window.crossOriginIsolated) return 1;
  return Math.max(1, Math.min(8, logicalThreadCount()));
}

function defaultCpuThreads() {
  const logical = logicalThreadCount();
  const estimatedPhysical = logical > DEFAULT_CPU_THREADS ? Math.ceil(logical / 2) : logical;
  return Math.max(1, Math.min(maxWasmThreads(), estimatedPhysical));
}

function getDiarBatchSize() {
  const threads = getRequestedThreads();
  if (threads >= 6) return 32;
  if (threads >= 4) return DIAR_BATCH_SIZE;
  return Math.max(4, Math.min(DIAR_BATCH_SIZE, threads * 2));
}

function getDiarBatchSizeForProvider(provider) {
  if (provider !== "webgpu") return getDiarBatchSize();
  const deviceMemory = Number(navigator.deviceMemory || 0);
  if (Number.isFinite(deviceMemory) && deviceMemory > 0 && deviceMemory < 8) return 32;
  return WEBGPU_DIAR_BATCH_SIZE;
}

function deviceMemoryGb() {
  const value = Number(navigator.deviceMemory || 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function uniqueSortedNumbers(values) {
  return [...new Set(values.filter((value) => Number.isFinite(value) && value > 0))]
    .sort((a, b) => a - b);
}

function webgpuBatchCandidates(kind, fallback) {
  const memory = deviceMemoryGb();
  if (kind === "pyannote_embedding") {
    const max = memory >= 12 ? 32 : (memory >= 8 ? 24 : (memory >= 4 ? 16 : 8));
    return uniqueSortedNumbers([fallback, 8, 16, 24, 32].filter((value) => value <= max));
  }
  if (kind === "campp_embedding") {
    const max = memory >= 12 ? 128 : (memory >= 8 ? 96 : (memory >= 4 ? 64 : 32));
    return uniqueSortedNumbers([fallback, 32, 64, 96, 128].filter((value) => value <= max));
  }
  const max = memory >= 12 ? 128 : (memory >= 8 ? 96 : (memory >= 4 ? 64 : 32));
  return uniqueSortedNumbers([fallback, 32, 64, 96, 128].filter((value) => value <= max));
}

function autotuneSampleCount(candidates, total) {
  const maxCandidate = Math.max(...candidates, 1);
  return Math.max(1, Math.min(total, Math.max(maxCandidate, Math.min(total, maxCandidate * 2))));
}

function webGpuAutotuneGuardOptions(context = {}) {
  const baseline = Number(context.wasmBaselineSeconds || 0);
  return {
    wasmBaselineSeconds: baseline,
    abortWhenSlowerThanWasm: Boolean(context.abortWebGpuTuneWhenSlower && baseline > 0),
  };
}

function webGpuRuntimeAbortOptions(context = {}, tuning = null) {
  const baseline = Number(context.wasmBaselineSeconds || 0);
  const tuneSeconds = Number(tuning?.totalSeconds || 0);
  const remaining = baseline > 0 ? baseline - tuneSeconds : 0;
  return {
    wasmBaselineSeconds: baseline,
    maxRuntimeSeconds: remaining > 0 ? remaining : 0,
  };
}

function throwIfWebGpuRuntimeSlower(options, started, completed = null, total = null) {
  const maxRuntimeSeconds = Number(options.maxRuntimeSeconds || 0);
  if (!(maxRuntimeSeconds > 0)) return;
  const elapsed = benchmarkSeconds(started);
  if (elapsed < maxRuntimeSeconds) return;
  const stageName = options.abortStageName || "WebGPU stage";
  const baseline = Number(options.wasmBaselineSeconds || 0);
  const tuneSeconds = Number(options.batchTuning?.totalSeconds || 0);
  const totalWithTune = Number((elapsed + tuneSeconds).toFixed(3));
  const error = new Error(
    `${stageName} WebGPU run stopped because runtime ${elapsed.toFixed(2)}s` +
    `${tuneSeconds ? ` plus tune ${tuneSeconds.toFixed(2)}s` : ""} exceeded WASM baseline ${baseline.toFixed(2)}s.`
  );
  error.benchmarkSummary = {
    provider: "webgpu",
    batchSize: options.batchSize || null,
    batchTuning: options.batchTuning || null,
    earlyStopped: true,
    stopReason: "webgpu_run_slower_than_wasm_baseline",
    elapsedBeforeStopSeconds: elapsed,
    totalWithTuneBeforeStopSeconds: totalWithTune,
    wasmBaselineSeconds: baseline,
    completed,
    total,
  };
  throw error;
}

function webGpuTuneAbortError(stageName, tuning, baselineSeconds) {
  const error = new Error(
    `${stageName} WebGPU tuning stopped because tuning time ` +
    `${Number(tuning.totalSeconds || 0).toFixed(2)}s exceeded WASM baseline ` +
    `${Number(baselineSeconds || 0).toFixed(2)}s.`
  );
  error.benchmarkSummary = {
    provider: "webgpu",
    batchSize: tuning.selectedBatchSize || null,
    batchTuning: tuning,
    earlyStopped: true,
  };
  return error;
}

async function autotuneWebGpuBatch(stageName, candidates, runCandidate, options = {}) {
  const totalStarted = performance.now();
  const baselineSeconds = Number(options.wasmBaselineSeconds || 0);
  const abortWhenSlower = Boolean(options.abortWhenSlowerThanWasm && baselineSeconds > 0);
  const attempts = [];
  let best = null;
  for (const batchSize of candidates) {
    const started = performance.now();
    try {
      await runCandidate(batchSize);
      const elapsedSeconds = benchmarkSeconds(started);
      const attempt = { batchSize, elapsedSeconds };
      attempts.push(attempt);
      if (!best || elapsedSeconds < best.elapsedSeconds) best = attempt;
    } catch (error) {
      attempts.push({
        batchSize,
        error: { message: error.message || String(error) },
      });
    }
    const totalSeconds = benchmarkSeconds(totalStarted);
    if (abortWhenSlower && totalSeconds >= baselineSeconds) {
      const tuning = {
        selectedBatchSize: best?.batchSize || null,
        candidates,
        attempts,
        totalSeconds,
        deviceMemoryGb: deviceMemoryGb() || null,
        earlyStopped: true,
        stopReason: "webgpu_tune_slower_than_wasm_baseline",
        wasmBaselineSeconds: Number(baselineSeconds.toFixed(3)),
      };
      log(
        `[Benchmark] ${stageName} WebGPU batch autotune stopped early: ` +
        `${totalSeconds.toFixed(2)}s >= WASM ${baselineSeconds.toFixed(2)}s.`
      );
      throw webGpuTuneAbortError(stageName, tuning, baselineSeconds);
    }
  }
  if (!best) {
    throw new Error(`${stageName} WebGPU batch autotune failed for all candidates.`);
  }
  log(`[Benchmark] ${stageName} WebGPU batch autotune selected batch=${best.batchSize}.`);
  return {
    selectedBatchSize: best.batchSize,
    candidates,
    attempts,
    totalSeconds: benchmarkSeconds(totalStarted),
    deviceMemoryGb: deviceMemoryGb() || null,
  };
}

function sliderValue(id, fallback) {
  return boundedNumber($(id)?.value, fallback, 1, 10);
}

function confidenceLabel(value) {
  const level = boundedNumber(value, 6, 1, 10);
  if (level <= 2) return "Rất ít";
  if (level <= 4) return "Ít";
  if (level <= 6) return "Vừa";
  if (level <= 8) return "Nhiều";
  return "Rất nhiều";
}

function formatConfidenceLabel(value) {
  const level = boundedNumber(value, 6, 1, 10);
  return `${confidenceLabel(level)} (${level})`;
}

function hashString(value) {
  let hash = 2166136261;
  const text = String(value || "");
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

function normalizeHotwordText(value) {
  return String(value || "")
    .normalize("NFC")
    .replace(/\s+/g, " ")
    .trim()
    .toLocaleUpperCase("vi-VN");
}

function parseHotwordsText(text, defaultScore = DEFAULT_HOTWORDS_SCORE) {
  const items = [];
  const lines = String(text || "").split(/\r?\n/);
  for (const rawLine of lines) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    let score = defaultScore;
    const scoreMatch = line.match(/^(.*?)(?:\s*:([+-]?(?:\d+(?:\.\d*)?|\.\d+)))$/);
    if (scoreMatch) {
      line = scoreMatch[1].trim();
      score = boundedFloat(scoreMatch[2], defaultScore, 0, 8);
    }

    const hotword = normalizeHotwordText(line);
    if (!hotword) continue;
    items.push({ text: hotword, score });
    if (items.length >= MAX_HOTWORDS) break;
  }
  return items;
}

function normalizeHotwordItems(items, defaultScore = DEFAULT_HOTWORDS_SCORE) {
  if (!Array.isArray(items)) return [];
  const parsed = [];
  for (const item of items) {
    const text = normalizeHotwordText(typeof item === "string" ? item : item?.text);
    if (!text) continue;
    parsed.push({
      text,
      score: boundedFloat(item?.score, defaultScore, 0, 8),
    });
    if (parsed.length >= MAX_HOTWORDS) break;
  }
  return parsed;
}

function currentHotwordsDefaultScore() {
  return boundedFloat($("hotwords-score")?.value, DEFAULT_HOTWORDS_SCORE, 0, 8);
}

function readHotwordItemsFromUi() {
  const rows = Array.from(document.querySelectorAll("[data-hotword-row]"));
  return normalizeHotwordItems(
    rows.map((row) => ({
      text: row.querySelector("[data-hotword-text]")?.value,
      score: row.querySelector("[data-hotword-score]")?.value,
    })),
    currentHotwordsDefaultScore()
  );
}

function buildHotwordsText(items = readHotwordItemsFromUi(), defaultScore = currentHotwordsDefaultScore()) {
  return normalizeHotwordItems(items, defaultScore)
    .map((item) => {
      const score = boundedFloat(item.score, defaultScore, 0, 8);
      return `${item.text} :${score.toFixed(2).replace(/\.?0+$/, "")}`;
    })
    .join("\n");
}

function updateHotwordsSummary() {
  const summary = $("hotwords-summary");
  if (!summary) return;
  const items = readHotwordItemsFromUi();
  const enabled = Boolean($("hotwords-enabled")?.checked);
  const score = currentHotwordsDefaultScore();
  const scoreValue = $("hotwords-score-value");
  if (scoreValue) scoreValue.textContent = score.toFixed(1);
  const rows = Array.from(document.querySelectorAll("[data-hotword-row]"));
  const visible = rows.filter((row) => !row.hidden).length;
  const query = $("hotwords-search")?.value?.trim() || "";
  const countLabel = query ? `${visible}/${items.length}` : `${items.length}`;
  summary.textContent = enabled
    ? `${countLabel} hotword`
    : `${countLabel} hotword, đang tắt`;
}

function updateSpeakerQuickSummary() {
  const summary = $("speaker-quick-summary");
  if (!summary) return;
  const enabled = Boolean($("speaker-diarization")?.checked);
  if (!enabled) {
    summary.textContent = "Off: transcript will not be automatically split by speaker.";
    return;
  }
  const model = $("speaker-model")?.value || DEFAULT_USER_CONFIG.speakerModel;
  const count = boundedNumber($("speaker-count")?.value, 0, 0, 20);
  const modelLabel = model === "pyannote_community1_vbx" ? "Pyannote Community-1" : "CAM++";
  summary.textContent = count > 0
    ? `${modelLabel}, fixed ${count} speaker(s)`
    : `${modelLabel}, auto speaker detection`;
}

function refreshHotwordRowNumbers() {
  Array.from(document.querySelectorAll("[data-hotword-row]")).forEach((row, index) => {
    const number = row.querySelector("[data-hotword-index]");
    if (number) number.textContent = String(index + 1);
  });
}

function updateHotwordFilter() {
  const query = normalizeVietnamese($("hotwords-search")?.value?.trim() || "");
  const rows = Array.from(document.querySelectorAll("[data-hotword-row]"));
  for (const row of rows) {
    const text = row.querySelector("[data-hotword-text]")?.value || "";
    row.hidden = Boolean(query) && !normalizeVietnamese(text).includes(query);
  }
  updateHotwordsSummary();
}

function commitHotwordRowEdit(row, textInput, scoreInput) {
  const previous = row.dataset.hotwordValue || "";
  const normalized = normalizeHotwordText(textInput?.value);
  if (textInput) textInput.value = normalized;
  if (scoreInput) scoreInput.value = boundedFloat(scoreInput.value, currentHotwordsDefaultScore(), 0, 8).toFixed(1);
  updateHotwordFilter();
  saveUserConfig();
  if (previous !== normalized) {
    row.dataset.hotwordValue = normalized;
    resetAsrWorker("hotwords changed");
  }
}

function createHotwordRow(item = {}, index = 0) {
  const row = document.createElement("div");
  row.className = "hotword-row";
  row.dataset.hotwordRow = "1";
  row.dataset.hotwordValue = normalizeHotwordText(item.text);

  const indexCell = document.createElement("span");
  indexCell.className = "hotword-index";
  indexCell.dataset.hotwordIndex = "1";
  indexCell.textContent = String(index + 1);

  const textInput = document.createElement("input");
  textInput.type = "text";
  textInput.autocomplete = "off";
  textInput.value = normalizeHotwordText(item.text);
  textInput.dataset.hotwordText = "1";

  const scoreInput = document.createElement("input");
  scoreInput.type = "hidden";
  scoreInput.min = "0";
  scoreInput.max = "8";
  scoreInput.step = "0.1";
  scoreInput.value = boundedFloat(item.score, currentHotwordsDefaultScore(), 0, 8).toFixed(1);
  scoreInput.dataset.hotwordScore = "1";

  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "btn btn-sm btn-danger hotword-remove";
  removeButton.textContent = "×";
  removeButton.title = "Xóa hotword";
  removeButton.setAttribute("aria-label", "Xóa hotword");
  removeButton.dataset.hotwordRemove = "1";

  row.append(indexCell, textInput, scoreInput, removeButton);

  textInput.addEventListener("input", () => {
    updateHotwordFilter();
    scheduleUserConfigSave();
  });
  textInput.addEventListener("change", () => commitHotwordRowEdit(row, textInput, scoreInput));
  textInput.addEventListener("blur", () => commitHotwordRowEdit(row, textInput, scoreInput));
  textInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      textInput.blur();
    }
  });
  removeButton.addEventListener("click", () => {
    row.remove();
    refreshHotwordRowNumbers();
    updateHotwordFilter();
    saveUserConfig();
    resetAsrWorker("hotwords changed");
  });
  return row;
}

function renderHotwordRows(items = []) {
  const root = $("hotword-list");
  if (!root) return;
  root.textContent = "";
  normalizeHotwordItems(items, currentHotwordsDefaultScore()).forEach((item, index) => {
    root.appendChild(createHotwordRow(item, index));
  });
  updateHotwordFilter();
}

function addHotwordFromControls() {
  const textInput = $("hotword-new-text");
  const scoreInput = $("hotword-new-score");
  const hotword = normalizeHotwordText(textInput?.value);
  if (!hotword) return;
  const existing = new Set(readHotwordItemsFromUi().map((item) => item.text));
  if (existing.has(hotword)) {
    log(`Hotword already exists: ${hotword}`);
    return;
  }
  const score = boundedFloat(scoreInput?.value, currentHotwordsDefaultScore(), 0, 8);
  const root = $("hotword-list");
  root?.appendChild(createHotwordRow({ text: hotword, score }, root.querySelectorAll("[data-hotword-row]").length));
  if (textInput) textInput.value = "";
  if (scoreInput) scoreInput.value = "";
  refreshHotwordRowNumbers();
  updateHotwordFilter();
  saveUserConfig();
  resetAsrWorker("hotwords changed");
}

function openHotwordDialog() {
  closeMeetingsPanel();
  updateHotwordFilter();
  const panel = $("hotword-panel");
  if (!panel) return;
  panel.style.display = "flex";
  $("hotword-new-text")?.focus();
}

function closeHotwordDialog() {
  const panel = $("hotword-panel");
  if (panel) panel.style.display = "none";
  saveUserConfig();
}

function exportHotwordTxt() {
  const text = buildHotwordsText(readHotwordItemsFromUi(), currentHotwordsDefaultScore());
  const blob = new Blob([`${text}${text ? "\n" : ""}`], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hotword.txt";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function importHotwordTxt(file) {
  if (!file) return;
  const text = await file.text();
  const items = parseHotwordsText(text, currentHotwordsDefaultScore());
  renderHotwordRows(items);
  saveUserConfig();
  resetAsrWorker("hotwords changed");
  log(`Imported hotword.txt: ${items.length} hotword(s).`);
}

async function resetHotwordsToDefaults() {
  const defaults = await loadDefaultHotwords();
  renderHotwordRows(defaults);
  saveUserConfig();
  resetAsrWorker("hotwords changed");
  log(`Restored default hotword.txt: ${defaults.length} hotword(s).`);
}

function applyTheme(theme) {
  const resolved = theme === "light" ? "light" : DEFAULT_THEME;
  document.body.dataset.theme = resolved;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = resolved === "light" ? "#f5f7fa" : "#3a3a3a";
  const button = $("btn-theme");
  if (button) button.textContent = resolved === "light" ? "Dark" : "Light";
}

function normalizedUiTextScale(value) {
  return boundedNumber(value, DEFAULT_UI_TEXT_SCALE, 90, 130);
}

function formatUiTextScaleLabel(value) {
  const scale = normalizedUiTextScale(value);
  if (scale < 100) return `Nhỏ (${scale}%)`;
  if (scale === 100) return `Vừa (${scale}%)`;
  if (scale >= 125) return `Rất lớn (${scale}%)`;
  return `Lớn (${scale}%)`;
}

function applyUiTextScale(value) {
  const scale = normalizedUiTextScale(value);
  document.documentElement.style.setProperty("--ui-text-scale", (scale / 100).toFixed(2));
  document.body.dataset.uiTextScale = String(scale);
  const slider = $("ui-text-scale");
  if (slider) slider.value = String(scale);
  const label = $("ui-text-scale-value");
  if (label) label.textContent = formatUiTextScaleLabel(scale);
}

function readStoredUserConfig() {
  try {
    const raw = window.localStorage?.getItem(USER_CONFIG_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    log(`Read config failed: ${error.message}`);
    return null;
  }
}

function normalizeUserConfig(config = {}) {
  const migrated = { ...(config || {}) };
  const sourceSchemaVersion = migrated.schemaVersion || 1;
  if ((migrated.schemaVersion || 1) < 2 && migrated.speakerDiarization !== true) {
    migrated.speakerDiarization = true;
  }
  if ((migrated.schemaVersion || 1) < 3) {
    if (!migrated.asrModel || migrated.asrModel === "zipformer-30m-rnnt-6000h") {
      migrated.asrModel = DEFAULT_ASR_MODEL_ID;
    }
    if (!migrated.punctuationLevel || Number(migrated.punctuationLevel) === 7) {
      migrated.punctuationLevel = DEFAULT_PUNCT_SLIDER;
    }
    if (!migrated.caseLevel || Number(migrated.caseLevel) === 3) {
      migrated.caseLevel = DEFAULT_CASE_SLIDER;
    }
  }
  if (sourceSchemaVersion < 4) {
    const previousThreads = Number.parseInt(migrated.cpuThreads, 10);
    if (!Number.isFinite(previousThreads) || previousThreads === DEFAULT_CPU_THREADS) {
      migrated.cpuThreads = defaultCpuThreads();
    }
  }
  const merged = { ...DEFAULT_USER_CONFIG, ...migrated };
  const modelId = ASR_MODELS[merged.asrModel] ? merged.asrModel : DEFAULT_ASR_MODEL_ID;
  const speakerModel = SUPPORTED_SPEAKER_MODELS.has(merged.speakerModel)
    ? merged.speakerModel
    : DEFAULT_USER_CONFIG.speakerModel;
  const fallbackThreads = defaultCpuThreads();
  const maxThreads = maxWasmThreads();
  return {
    ...DEFAULT_USER_CONFIG,
    ...merged,
    schemaVersion: USER_CONFIG_SCHEMA,
    theme: merged.theme === "light" ? "light" : DEFAULT_THEME,
    uiTextScale: normalizedUiTextScale(merged.uiTextScale),
    asrModel: modelId,
    cpuThreads: boundedNumber(merged.cpuThreads, fallbackThreads, 1, maxThreads),
    punctuationLevel: DEFAULT_PUNCT_SLIDER,
    caseLevel: DEFAULT_CASE_SLIDER,
    speakerDiarization: Boolean(merged.speakerDiarization),
    speakerCount: String(boundedNumber(merged.speakerCount, 0, 0, 20)),
    speakerModel,
    rmsNormalize: Boolean(merged.rmsNormalize),
    bypassVad: Boolean(merged.bypassVad),
    resumeAfterKill: true,
    overlapSeparation: false,
    saveRam: merged.saveRam !== false,
    hotwordsEnabled: merged.hotwordsEnabled !== false,
    hotwordsScore: boundedFloat(merged.hotwordsScore, DEFAULT_HOTWORDS_SCORE, 0, 8),
    hotwords: normalizeHotwordItems(merged.hotwords, boundedFloat(merged.hotwordsScore, DEFAULT_HOTWORDS_SCORE, 0, 8)),
  };
}

function setControlValue(id, value) {
  const node = $(id);
  if (node) node.value = String(value);
}

function setControlChecked(id, value) {
  const node = $(id);
  if (node) node.checked = Boolean(value);
}

function applyUserConfig(config) {
  const normalized = normalizeUserConfig(config);
  applyingUserConfig = true;
  try {
    applyTheme(normalized.theme);
    applyUiTextScale(normalized.uiTextScale);
    setControlValue("asr-model", normalized.asrModel);
    setControlValue("cpu-threads", normalized.cpuThreads);
    setControlValue("punctuation-level", normalized.punctuationLevel);
    setControlValue("case-level", normalized.caseLevel);
    setControlChecked("speaker-diarization", normalized.speakerDiarization);
    setControlValue("speaker-count", normalized.speakerCount);
    setControlValue("speaker-model", normalized.speakerModel);
    setControlChecked("rms-normalize", normalized.rmsNormalize);
    setControlChecked("bypass-vad", normalized.bypassVad);
    setControlChecked("overlap-separation", false);
    setControlChecked("save-ram", normalized.saveRam);
    setControlChecked("hotwords-enabled", normalized.hotwordsEnabled);
    setControlValue("hotwords-score", normalized.hotwordsScore);
    renderHotwordRows(normalized.hotwords);
    syncPipelineControls();
    updateHotwordsSummary();
  } finally {
    applyingUserConfig = false;
  }
}

function collectUserConfigFromUi() {
  return normalizeUserConfig({
    theme: document.body.dataset.theme || DEFAULT_THEME,
    uiTextScale: $("ui-text-scale")?.value || DEFAULT_UI_TEXT_SCALE,
    asrModel: $("asr-model")?.value || DEFAULT_ASR_MODEL_ID,
    cpuThreads: getRequestedThreads(),
    punctuationLevel: DEFAULT_PUNCT_SLIDER,
    caseLevel: DEFAULT_CASE_SLIDER,
    speakerDiarization: Boolean($("speaker-diarization")?.checked),
    speakerCount: $("speaker-count")?.value || "0",
    speakerModel: $("speaker-model")?.value || DEFAULT_USER_CONFIG.speakerModel,
    rmsNormalize: Boolean($("rms-normalize")?.checked),
    bypassVad: Boolean($("bypass-vad")?.checked),
    resumeAfterKill: true,
    overlapSeparation: false,
    saveRam: Boolean($("save-ram")?.checked),
    hotwordsEnabled: Boolean($("hotwords-enabled")?.checked),
    hotwordsScore: currentHotwordsDefaultScore(),
    hotwords: readHotwordItemsFromUi(),
  });
}

function saveUserConfig() {
  if (applyingUserConfig) return;
  try {
    window.localStorage?.setItem(USER_CONFIG_KEY, JSON.stringify(collectUserConfigFromUi()));
  } catch (error) {
    log(`Save config failed: ${error.message}`);
  }
}

function scheduleUserConfigSave() {
  if (applyingUserConfig) return;
  window.clearTimeout(userConfigSaveTimer);
  userConfigSaveTimer = window.setTimeout(saveUserConfig, 200);
}

async function loadDefaultHotwords() {
  try {
    const response = await fetch("/hotword.txt", { cache: "force-cache" });
    if (!response.ok) return [];
    return parseHotwordsText(await response.text(), DEFAULT_HOTWORDS_SCORE);
  } catch (_) {
    return [];
  }
}

async function initializeUserConfig() {
  const stored = readStoredUserConfig();
  const config = normalizeUserConfig(stored || DEFAULT_USER_CONFIG);
  const storedSchema = stored?.schemaVersion || 1;
  if (!stored) {
    const defaults = await loadDefaultHotwords();
    if (defaults.length) config.hotwords = defaults;
  } else if (storedSchema < 5 && Array.isArray(stored.hotwords) && stored.hotwords.length === 251) {
    const defaults = await loadDefaultHotwords();
    if (defaults.length === 252) config.hotwords = defaults;
  }
  applyUserConfig(config);
  saveUserConfig();
}

function readStoredCalibrationProfile() {
  try {
    const raw = window.localStorage?.getItem(CALIBRATION_PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    log(`Read calibration profile failed: ${error.message}`);
    return null;
  }
}

function calibrationManifestSignature() {
  const packs = (manifest?.packs || []).map((pack) => ({
    id: pack.id,
    files: (pack.files || []).map((file) => ({
      id: file.id,
      bytes: file.bytes || 0,
      target_path: file.target_path || "",
      local_path: file.local_path || "",
      sha256: file.sha256 || null,
    })),
  }));
  return hashString(JSON.stringify({
    schema_version: manifest?.schema_version || null,
    runtime: manifest?.runtime || {},
    packs,
  }));
}

function plainCalibrationAdapterInfo(environment = null) {
  const info = environment?.webgpu?.adapterInfo || {};
  return {
    vendor: info.vendor || "",
    architecture: info.architecture || "",
    device: info.device || "",
    description: info.description || "",
    subgroupMinSize: info.subgroupMinSize || null,
    subgroupMaxSize: info.subgroupMaxSize || null,
  };
}

async function currentCalibrationSignature(environment = null) {
  const env = environment || await collectBenchmarkEnvironment();
  return hashString(JSON.stringify({
    schema: 1,
    appCodeVersion: OFFLINE_PWA_CODE_VERSION,
    codeVersion: CALIBRATION_CODE_VERSION,
    sampleUrl: CALIBRATION_SAMPLE_URL,
    manifestSignature: calibrationManifestSignature(),
    userAgent: navigator.userAgent,
    platform: navigator.platform || "",
    logicalThreads: logicalThreadCount(),
    maxWasmThreads: maxWasmThreads(),
    deviceMemoryGb: deviceMemoryGb() || null,
    webgpuSupported: Boolean(env.webgpu?.supported),
    webgpuAdapter: plainCalibrationAdapterInfo(env),
  }));
}

async function loadCalibrationProfileForCurrentDevice() {
  const stored = readStoredCalibrationProfile();
  if (!stored?.signature) {
    calibrationProfile = null;
    return null;
  }
  const signature = await currentCalibrationSignature().catch(() => null);
  if (!signature || stored.signature !== signature) {
    calibrationProfile = null;
    log("Device calibration profile is stale or for another runtime; calibration will run again when possible.");
    return null;
  }
  calibrationProfile = stored;
  log(`Device calibration profile loaded (${stored.createdAt || "unknown"}).`);
  return calibrationProfile;
}

function saveCalibrationProfile(profile, report = null) {
  calibrationProfile = profile;
  try {
    window.localStorage?.setItem(CALIBRATION_PROFILE_KEY, JSON.stringify(profile));
    if (report) {
      const slimReport = {
        ...report,
        pipelineLog: (report.pipelineLog || []).slice(-120),
      };
      window.localStorage?.setItem(CALIBRATION_LAST_REPORT_KEY, JSON.stringify(slimReport));
    }
  } catch (error) {
    log(`Save calibration profile failed: ${error.message}`);
  }
}

function providerAliasesForCalibration(stageName) {
  if (stageName === "CAM++ speech regions (pyannote segmentation)") return ["Pyannote segmentation"];
  if (stageName === "Pyannote Community-1 segmentation") return ["Pyannote segmentation"];
  if (stageName === "ViBERT punctuation") return ["ViBERT punctuation fp32"];
  if (stageName === "ASR full pipeline") return ["ASR"];
  return [];
}

function selectedAttemptForStage(stage) {
  if (!stage?.attempts?.length) return null;
  return stage.attempts.find((attempt) => (
    !attempt.error &&
    (attempt.runtime === stage.selectedRuntime || attempt.provider === stage.selectedProvider)
  )) || stage.attempts.find((attempt) => !attempt.error) || null;
}

function deriveCalibrationProfile(report, environment, signature) {
  const stages = {};
  const selectedProviders = {};
  const batchSizes = {};
  for (const stage of report.stages || []) {
    const selected = selectedAttemptForStage(stage);
    const decision = {
      provider: stage.selectedProvider || selected?.provider || null,
      runtime: stage.selectedRuntime || selected?.runtime || null,
      speedupWebgpuOverWasm: stage.comparison?.speedupWebgpuOverWasm ?? null,
      outputHashEqual: stage.comparison?.outputHashEqual ?? null,
      webgpuAccepted: stage.comparison?.webgpuAccepted ?? null,
      webgpuAcceptedByTolerance: stage.comparison?.webgpuAcceptedByTolerance ?? false,
      acceptanceMode: stage.comparison?.acceptanceMode ?? null,
      rejectionReason: stage.comparison?.rejectionReason ?? null,
      selectedElapsedSeconds: selected?.elapsedSeconds ?? null,
      selectedBatchSize: selected?.summary?.batchSize ?? null,
    };
    stages[stage.name] = decision;
    if (decision.provider) {
      selectedProviders[stage.name] = decision.provider;
      for (const alias of providerAliasesForCalibration(stage.name)) {
        selectedProviders[alias] = decision.provider;
      }
    }
    if (decision.selectedBatchSize) {
      batchSizes[stage.name] = decision.selectedBatchSize;
      for (const alias of providerAliasesForCalibration(stage.name)) {
        batchSizes[alias] = decision.selectedBatchSize;
      }
    }
  }
  return {
    schemaVersion: 1,
    kind: "offline_pwa_device_calibration_profile",
    codeVersion: CALIBRATION_CODE_VERSION,
    appCodeVersion: OFFLINE_PWA_CODE_VERSION,
    signature,
    createdAt: new Date().toISOString(),
    sample: {
      url: CALIBRATION_SAMPLE_URL,
      name: CALIBRATION_SAMPLE_NAME,
    },
    environment: {
      userAgent: navigator.userAgent,
      platform: navigator.platform || "",
      logicalThreads: logicalThreadCount(),
      maxWasmThreads: maxWasmThreads(),
      deviceMemoryGb: deviceMemoryGb() || null,
      webgpu: {
        supported: Boolean(environment?.webgpu?.supported),
        adapterInfo: plainCalibrationAdapterInfo(environment),
      },
    },
    manifestSignature: calibrationManifestSignature(),
    selectedProviders,
    batchSizes,
    stages,
    comparison: report.comparison || {},
  };
}

function defaultWebGpuBatchSize(kind, fallback) {
  const candidates = webgpuBatchCandidates(kind, fallback);
  return candidates[candidates.length - 1] || fallback;
}

function defaultCalibrationSelectedProviders(webgpuAvailable) {
  const gpu = webgpuAvailable ? "webgpu" : "wasm";
  return {
    "ASR full pipeline": "wasm",
    ASR: "wasm",
    "Silero VAD": "wasm",
    "Pyannote Community-1 segmentation": "wasm",
    "CAM++ speech regions (pyannote segmentation)": "wasm",
    "CAM++ speaker embedding": gpu,
    "CAM++ clustering": "js",
    "CAM++ long-form clustering": "js",
    "Pyannote Community-1 embedding encoder": gpu,
    "Pyannote Community-1 clustering": "js",
    "DNSMOS quality": gpu,
    "ViBERT punctuation": gpu,
    "ViBERT punctuation fp32": gpu,
    "Overlap separation": "wasm",
  };
}

function defaultCalibrationBatchSizes(webgpuAvailable) {
  if (!webgpuAvailable) return {};
  return {
    "CAM++ speaker embedding": defaultWebGpuBatchSize("campp_embedding", CAMPP_BATCH_SIZE),
    "Pyannote Community-1 embedding encoder": defaultWebGpuBatchSize("pyannote_embedding", WESPEAKER_BATCH_SIZE),
  };
}

function shouldBenchmarkWebGpuStage(options, stageName) {
  return Array.isArray(options?.benchmarkStages) && WEBGPU_CALIBRATION_STAGE_NAMES.has(stageName);
}

async function buildDefaultCalibrationProfile(signature = null, environment = null) {
  const env = environment || await collectBenchmarkEnvironment().catch(() => null);
  const resolvedSignature = signature || await currentCalibrationSignature(env).catch(() => null);
  const webgpuAvailable = Boolean(env?.webgpu?.supported && env?.webgpu?.adapterAvailable && isWebGpuRuntimeAvailable());
  return {
    schemaVersion: 1,
    kind: "offline_pwa_device_calibration_profile",
    source: "default_skip",
    codeVersion: CALIBRATION_CODE_VERSION,
    appCodeVersion: OFFLINE_PWA_CODE_VERSION,
    signature: resolvedSignature,
    createdAt: new Date().toISOString(),
    sample: {
      url: CALIBRATION_SAMPLE_URL,
      name: CALIBRATION_SAMPLE_NAME,
      skipped: true,
    },
    environment: {
      userAgent: navigator.userAgent,
      platform: navigator.platform || "",
      logicalThreads: logicalThreadCount(),
      maxWasmThreads: maxWasmThreads(),
      deviceMemoryGb: deviceMemoryGb() || null,
      webgpu: {
        supported: Boolean(env?.webgpu?.supported),
        adapterInfo: plainCalibrationAdapterInfo(env),
      },
    },
    manifestSignature: calibrationManifestSignature(),
    selectedProviders: defaultCalibrationSelectedProviders(webgpuAvailable),
    batchSizes: defaultCalibrationBatchSizes(webgpuAvailable),
    stages: {},
    comparison: {
      source: "default_skip",
      note: "User skipped device calibration; using conservative defaults from prior local tests.",
      webgpuDefaultEnabled: webgpuAvailable,
    },
  };
}

async function saveDefaultCalibrationProfile(reason = "skip") {
  const environment = await collectBenchmarkEnvironment().catch(() => null);
  const signature = await currentCalibrationSignature(environment).catch(() => null);
  if (!signature) {
    throw new Error("Cannot build calibration signature for this device.");
  }
  const profile = await buildDefaultCalibrationProfile(signature, environment);
  saveCalibrationProfile(profile, {
    kind: "offline_pwa_device_calibration_default",
    reason,
    environment,
    signature,
    createdAt: profile.createdAt,
    profileSummary: {
      selectedProviders: profile.selectedProviders,
      batchSizes: profile.batchSizes,
    },
  });
  log(
    `[Calibration] Default profile saved: ` +
    `CAM++=${profile.selectedProviders["CAM++ speaker embedding"] || "wasm"}, ` +
    `pyannoteEmb=${profile.selectedProviders["Pyannote Community-1 embedding encoder"] || "wasm"}, ` +
    `dnsmos=${profile.selectedProviders["DNSMOS quality"] || "wasm"}, ` +
    `punct=${profile.selectedProviders["ViBERT punctuation"] || profile.selectedProviders["ViBERT punctuation fp32"] || "wasm"}.`
  );
  return profile;
}

function calibratedProviderForStage(stageName, fallback = null) {
  const provider = calibrationProfile?.selectedProviders?.[stageName] || fallback;
  if (provider === "webgpu" && !WEBGPU_CALIBRATION_STAGE_NAMES.has(stageName)) return fallback;
  return provider;
}

function calibratedBatchSizeForStage(stageName, fallback) {
  const value = Number(calibrationProfile?.batchSizes?.[stageName]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

async function exportUserConfig() {
  try {
    const data = collectUserConfigFromUi();
    await writeTextFileWithPicker(
      "asr-vn-pwa.config.json",
      JSON.stringify(data, null, 2),
      "application/json",
      "PWA config",
      ".json"
    );
    log("Exported PWA config.");
  } catch (error) {
    if (error?.name === "AbortError") return;
    log(`Export config failed: ${error.message}`);
  }
}

async function importUserConfig(file) {
  try {
    const data = JSON.parse(await file.text());
    const config = normalizeUserConfig(data);
    applyUserConfig(config);
    saveUserConfig();
    resetAsrWorker("config imported");
    log(`Imported PWA config: ${file.name}`);
  } catch (error) {
    log(`Import config failed: ${error.message}`);
  }
}

function punctuationConfidenceFromSlider(value) {
  return 0.5 - (value - 1) * (1.3 / 9);
}

function caseConfidenceFromSlider(value) {
  return -1.5 + (value - 1) * (2.0 / 9);
}

function getRequestedThreads() {
  const maxThreads = maxWasmThreads();
  return boundedNumber($("cpu-threads")?.value, defaultCpuThreads(), 1, maxThreads);
}

function getSelectedAsrModel() {
  const modelId = $("asr-model")?.value || DEFAULT_ASR_MODEL_ID;
  const model = ASR_MODELS[modelId];
  if (!model) {
    throw new Error(`ASR model is not supported in offline PWA: ${modelId}`);
  }
  return { id: modelId, ...model };
}

function getPipelineOptions() {
  const punctuationLevel = sliderValue("punctuation-level", DEFAULT_PUNCT_SLIDER);
  const caseLevel = sliderValue("case-level", DEFAULT_CASE_SLIDER);
  const speakerDiarization = Boolean($("speaker-diarization")?.checked);
  const speakerModel = $("speaker-model")?.value || "senko_campp_optimized";
  const numSpeakers = boundedNumber($("speaker-count")?.value, 0, 0, 20);
  const overlapSeparation = false;
  const hotwords = readHotwordItemsFromUi();
  const hotwordsEnabled = Boolean($("hotwords-enabled")?.checked);
  const hotwordsScore = currentHotwordsDefaultScore();
  const hotwordsText = hotwordsEnabled ? buildHotwordsText(hotwords, hotwordsScore) : "";

  if (speakerDiarization && !SUPPORTED_SPEAKER_MODELS.has(speakerModel)) {
    throw new Error(`Speaker model is not supported in offline PWA: ${speakerModel}`);
  }
  if (overlapSeparation && !speakerDiarization) {
    throw new Error("Overlap separation requires speaker diarization in offline PWA.");
  }

  return {
    asrModel: getSelectedAsrModel(),
    cpuThreads: getRequestedThreads(),
    punctuationLevel,
    caseLevel,
    bypassPunctuation: punctuationLevel === 1,
    punctuationConfidence: punctuationConfidenceFromSlider(punctuationLevel),
    caseConfidence: caseConfidenceFromSlider(caseLevel),
    speakerDiarization,
    speakerModel,
    numSpeakers,
    overlapSeparation,
    rmsNormalize: Boolean($("rms-normalize")?.checked),
    bypassVad: Boolean($("bypass-vad")?.checked),
    resumeAfterKill: true,
    saveRam: Boolean($("save-ram")?.checked),
    hotwordsEnabled,
    hotwordsScore,
    hotwords,
    hotwordsText,
    hotwordCount: hotwordsText ? hotwords.length : 0,
  };
}

function syncPipelineControls() {
  const threadInput = $("cpu-threads");
  if (threadInput) {
    const maxThreads = maxWasmThreads();
    threadInput.max = String(maxThreads);
    threadInput.value = String(boundedNumber(threadInput.value, defaultCpuThreads(), 1, maxThreads));
    $("cpu-threads-value").textContent = threadInput.value;
  }
  const punctuationLevel = $("punctuation-level");
  if (punctuationLevel) {
    punctuationLevel.value = String(DEFAULT_PUNCT_SLIDER);
    $("punctuation-level-value").textContent = formatConfidenceLabel(DEFAULT_PUNCT_SLIDER);
  }
  const caseLevel = $("case-level");
  if (caseLevel) {
    caseLevel.value = String(DEFAULT_CASE_SLIDER);
    $("case-level-value").textContent = formatConfidenceLabel(DEFAULT_CASE_SLIDER);
  }
  const uiTextScale = $("ui-text-scale");
  if (uiTextScale) {
    const scale = normalizedUiTextScale(uiTextScale.value);
    uiTextScale.value = String(scale);
    const label = $("ui-text-scale-value");
    if (label) label.textContent = formatUiTextScaleLabel(scale);
  }
  const hotwordsScore = $("hotwords-score");
  if (hotwordsScore) $("hotwords-score-value").textContent = Number(hotwordsScore.value || DEFAULT_HOTWORDS_SCORE).toFixed(1);

  const diarizationEnabled = Boolean($("speaker-diarization")?.checked);
  ["speaker-count", "speaker-model"].forEach((id) => {
    const node = $(id);
    if (node) node.disabled = !diarizationEnabled;
  });
  const overlapNode = $("overlap-separation");
  if (overlapNode) {
    overlapNode.disabled = !diarizationEnabled;
    if (!diarizationEnabled) overlapNode.checked = false;
  }
  updateSpeakerQuickSummary();
}

function setPipelineControlsDisabled(disabled) {
  [
    "asr-model",
    "cpu-threads",
    "ui-text-scale",
    "punctuation-level",
    "case-level",
    "speaker-diarization",
    "speaker-count",
    "speaker-model",
    "overlap-separation",
    "rms-normalize",
    "bypass-vad",
    "save-ram",
    "hotwords-enabled",
    "hotwords-score",
    "hotword-new-text",
    "hotword-new-score",
    "btn-add-hotword",
  ].forEach((id) => {
    const node = $(id);
    if (node) {
      const needsDiarization = (id.startsWith("speaker-") && id !== "speaker-diarization") || id === "overlap-separation";
      node.disabled = disabled || (needsDiarization && !$("speaker-diarization")?.checked);
    }
  });
  document.querySelectorAll("[data-hotword-row] input, [data-hotword-row] button").forEach((node) => {
    node.disabled = disabled;
  });
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00.00";
  const minutes = Math.floor(seconds / 60);
  const wholeSeconds = Math.floor(seconds % 60);
  const centiseconds = Math.floor((seconds - Math.floor(seconds)) * 100);
  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(centiseconds).padStart(2, "0")}`;
}

function safeFileBaseName(name, fallback = "result") {
  const base = (name || fallback)
    .replace(/\.asr\.json$/i, "")
    .replace(/\.[^.]+$/i, "")
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "_")
    .trim();
  return base || fallback;
}

function log(message) {
  const line = `${new Date().toLocaleTimeString()}  ${message}`;
  pipelineLogLines.push(line);
  if (pipelineLogLines.length > 2000) {
    pipelineLogLines = pipelineLogLines.slice(-2000);
  }
  const box = $("pipeline-log");
  if (box) {
    box.textContent += `${line}\n`;
    box.scrollTop = box.scrollHeight;
  }
}

function debugRound(value, digits = 3) {
  const num = Number(value);
  return Number.isFinite(num) ? Number(num.toFixed(digits)) : null;
}

function appendDebugLog(event, data = {}) {
  const entry = {
    at: new Date().toISOString(),
    event,
    data,
  };
  debugLogEntries.push(entry);
  if (debugLogEntries.length > 5000) {
    debugLogEntries = debugLogEntries.slice(-5000);
  }
  return entry;
}

function debugAudioSnapshot() {
  const audio = $("editor-audio");
  if (!audio) return null;
  return {
    currentTime: debugRound(audio.currentTime),
    duration: debugRound(audio.duration),
    paused: Boolean(audio.paused),
    readyState: audio.readyState,
    srcType: audio.src ? (audio.src.startsWith("blob:") ? "blob" : "url") : "none",
  };
}

function debugSegmentSnapshot(index) {
  const segment = editorState?.segments?.[index];
  if (!segment) return null;
  const rawWords = Array.isArray(segment.raw_words) ? segment.raw_words : [];
  const previous = editorState.segments[index - 1] || null;
  const next = editorState.segments[index + 1] || null;
  return {
    index,
    speaker: normalizeSpeakerId(segment.speaker),
    start: debugRound(segment.start),
    end: debugRound(segment.end),
    playbackStart: debugRound(editorSegmentPlaybackStart(index)),
    rawWordCount: rawWords.length,
    rawFirst: rawWords.length ? {
      text: String(rawWords[0].text || ""),
      start: debugRound(rawWords[0].start),
      end: debugRound(rawWords[0].end),
    } : null,
    rawLast: rawWords.length ? {
      text: String(rawWords[rawWords.length - 1].text || ""),
      start: debugRound(rawWords[rawWords.length - 1].start),
      end: debugRound(rawWords[rawWords.length - 1].end),
    } : null,
    prevEnd: previous ? debugRound(previous.end) : null,
    nextStart: next ? debugRound(next.start) : null,
    textPreview: String(segment.text || "").slice(0, 180),
  };
}

function resetPipelineLog() {
  pipelineLogLines = [];
  const box = $("pipeline-log");
  if (box) box.textContent = "";
}

function setProgress(el, loaded, total) {
  if (!el) return;
  const pct = total ? Math.max(0, Math.min(100, (loaded / total) * 100)) : 0;
  el.style.width = `${pct.toFixed(1)}%`;
}

function setPipelineProgress(stage, pct) {
  const bounded = Math.max(0, Math.min(100, pct || 0));
  if (document.body.classList.contains("calibration-busy")) {
    setCalibrationSetupProgress(stage || "Đang tính toán cấu hình tối ưu", bounded);
  }
  const container = $("process-progress");
  if (container) container.style.display = bounded > 0 || stage ? "flex" : "none";
  const bar = $("pipeline-progress-bar");
  const label = $("pipeline-progress-text");
  const stageNode = $("pipeline-stage");
  if (bar) bar.style.width = `${bounded.toFixed(1)}%`;
  if (label) label.textContent = `${Math.round(bounded)}%`;
  if (stageNode && stage) stageNode.textContent = stage;
}

function hidePipelineProgress() {
  const container = $("process-progress");
  if (container) container.style.display = "none";
}

function screenWakeLockSupported() {
  return Boolean(navigator.wakeLock?.request);
}

function wakeLockReasonLabel(reason) {
  if (reason === "calibration") return "Calibrate thiết bị";
  if (reason === "benchmark") return "Benchmark";
  if (reason === "processing") return "Xử lý file";
  if (reason === "bootstrap") return "Tải dữ liệu lần đầu";
  return reason || "Tác vụ nền";
}

function screenWakeLockMessage() {
  if (screenWakeLockSupported()) {
    return "Ứng dụng sẽ yêu cầu giữ màn hình sáng trong lúc chạy; vui lòng để PWA ở foreground và không khóa màn hình.";
  }
  return "Trình duyệt này không hỗ trợ Screen Wake Lock; vui lòng mở PWA ở foreground, tắt tự khóa màn hình tạm thời và không đóng ứng dụng.";
}

async function ensureScreenWakeLock() {
  if (!screenWakeLockReasons.size) return false;
  if (document.visibilityState !== "visible") return false;
  if (screenWakeLock) return true;
  if (!screenWakeLockSupported()) {
    if (!screenWakeLockUnsupportedLogged) {
      screenWakeLockUnsupportedLogged = true;
      log("[WakeLock] Screen Wake Lock is not supported by this browser.");
    }
    return false;
  }
  if (screenWakeLockRequest) return screenWakeLockRequest;

  screenWakeLockRequest = navigator.wakeLock.request("screen")
    .then((lock) => {
      if (!screenWakeLockReasons.size || document.visibilityState !== "visible") {
        lock.release().catch(() => null);
        return false;
      }
      screenWakeLock = lock;
      const labels = [...screenWakeLockReasons].map(wakeLockReasonLabel).join(", ");
      log(`[WakeLock] Screen wake lock active (${labels || "active task"}).`);
      lock.addEventListener("release", () => {
        if (screenWakeLock === lock) screenWakeLock = null;
        log("[WakeLock] Screen wake lock released.");
        if (screenWakeLockReasons.size && document.visibilityState === "visible") {
          window.setTimeout(() => {
            ensureScreenWakeLock().catch((error) => {
              log(`[WakeLock] Resume failed: ${error.message || String(error)}`);
            });
          }, 250);
        }
      }, { once: true });
      return true;
    })
    .catch((error) => {
      log(`[WakeLock] Cannot request screen wake lock: ${error.message || String(error)}`);
      return false;
    })
    .finally(() => {
      screenWakeLockRequest = null;
    });
  return screenWakeLockRequest;
}

async function requestScreenWakeLockFor(reason) {
  screenWakeLockReasons.add(reason || "task");
  document.body.classList.add("wake-lock-requested");
  return ensureScreenWakeLock();
}

async function releaseScreenWakeLockFor(reason) {
  screenWakeLockReasons.delete(reason || "task");
  if (screenWakeLockReasons.size) return;
  document.body.classList.remove("wake-lock-requested");
  const lock = screenWakeLock;
  screenWakeLock = null;
  if (lock) {
    try {
      await lock.release();
    } catch (error) {
      log(`[WakeLock] Release failed: ${error.message || String(error)}`);
    }
  }
}

function setupScreenWakeLockResume() {
  const resume = () => {
    if (!screenWakeLockReasons.size || document.visibilityState !== "visible") return;
    ensureScreenWakeLock().catch((error) => {
      log(`[WakeLock] Resume failed: ${error.message || String(error)}`);
    });
  };
  const handleVisibility = () => {
    if (document.visibilityState === "hidden") {
      armResumeAfterKill("app_hidden").catch((error) => {
        log(`[resume_after_kill] Visibility arm failed: ${error.message || String(error)}`);
      });
      return;
    }
    resume();
  };
  document.addEventListener("visibilitychange", handleVisibility);
  document.addEventListener("resume", resume);
  window.addEventListener("focus", resume);
  window.addEventListener("pageshow", resume);
}

function setCalibrationSetupProgress(stage = "", pct = 0) {
  const bounded = Math.max(0, Math.min(100, Number(pct) || 0));
  const badge = $("offline-bootstrap-status");
  if (badge) {
    badge.textContent = "Đang Calibrate thiết bị";
    badge.classList.add("warn");
    badge.classList.remove("ok");
  }
  const message = $("offline-bootstrap-message");
  if (message) {
    const stageText = stage ? ` ${stage}.` : "";
    message.textContent = `Ứng dụng đang Calibrate để chọn cấu hình xử lý tối ưu cho thiết bị này.${stageText} ${screenWakeLockMessage()}`;
  }
  setProgress($("offline-bootstrap-progress-bar"), bounded, 100);
}

function clearCalibrationSetupClasses() {
  document.body.classList.remove("calibration-pending", "calibration-busy");
}

function syncCalibrationSetupUi(stage = "", pct = 0) {
  const standalone = isStandaloneApp();
  const pending = Boolean(
    standalone &&
    offlineBootstrapReady &&
    !calibrationBusy &&
    !calibrationProfile &&
    !autoCalibrationAttempted &&
    !selectedAudioFile &&
    !selectedLibraryImportPromise
  );
  document.body.classList.toggle("calibration-pending", pending);
  document.body.classList.toggle("calibration-busy", Boolean(standalone && calibrationBusy));
  if (standalone && calibrationBusy) {
    setCalibrationSetupProgress(stage || "Đang Calibrate cấu hình tối ưu", pct);
  } else if (pending) {
    const badge = $("offline-bootstrap-status");
    if (badge) {
      badge.textContent = "Chuẩn bị tối ưu";
      badge.classList.add("warn");
      badge.classList.remove("ok");
    }
    const message = $("offline-bootstrap-message");
    if (message) {
      message.textContent = `Đã tải đủ dữ liệu offline. Ứng dụng sẽ Calibrate cấu hình tối ưu trước khi mở giao diện xử lý. ${screenWakeLockMessage()}`;
    }
    setProgress($("offline-bootstrap-progress-bar"), 0, 100);
  } else if (!pending && !calibrationBusy) {
    clearCalibrationSetupClasses();
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  window.setTimeout(() => toast.remove(), 5000);
}

function idbRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB request failed."));
  });
}

function libraryDb() {
  if (libraryDbPromise) return libraryDbPromise;
  if (!window.indexedDB) {
    throw new Error("IndexedDB is not supported by this browser.");
  }
  libraryDbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(LIBRARY_DB_NAME, LIBRARY_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(LIBRARY_STORE)) {
        const store = db.createObjectStore(LIBRARY_STORE, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt");
        store.createIndex("status", "status");
      }
      if (!db.objectStoreNames.contains(FILE_STORE)) {
        db.createObjectStore(FILE_STORE, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB open failed."));
  });
  return libraryDbPromise;
}

async function libraryStore(mode = "readonly") {
  const db = await libraryDb();
  return db.transaction(LIBRARY_STORE, mode).objectStore(LIBRARY_STORE);
}

async function libraryGetAllItems() {
  const store = await libraryStore("readonly");
  const items = await idbRequest(store.getAll());
  return items.sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")));
}

async function libraryGetItem(id) {
  const store = await libraryStore("readonly");
  return idbRequest(store.get(id));
}

async function libraryPutItem(item) {
  const store = await libraryStore("readwrite");
  await idbRequest(store.put(item));
  return item;
}

async function libraryDeleteItemRecord(id) {
  const store = await libraryStore("readwrite");
  await idbRequest(store.delete(id));
}

async function fileBlobStore(mode = "readonly") {
  const db = await libraryDb();
  return db.transaction(FILE_STORE, mode).objectStore(FILE_STORE);
}

function idbFileKey(scope, name) {
  return `${String(scope || "").replace(/^\/+|\/+$/g, "")}/${String(name || "").replace(/^\/+/, "")}`;
}

async function idbPutFile(scope, name, data) {
  const blob = data instanceof Blob ? data : new Blob([data]);
  const store = await fileBlobStore("readwrite");
  await idbRequest(store.put({
    key: idbFileKey(scope, name),
    scope,
    name,
    blob,
    size: blob.size,
    updatedAt: new Date().toISOString(),
  }));
}

async function idbGetFile(scope, name) {
  const store = await fileBlobStore("readonly");
  const record = await idbRequest(store.get(idbFileKey(scope, name)));
  if (!record?.blob) throw new Error(`Stored file not found: ${scope}/${name}`);
  return new File([record.blob], name, {
    type: record.blob.type || "application/octet-stream",
    lastModified: record.updatedAt ? Date.parse(record.updatedAt) || Date.now() : Date.now(),
  });
}

async function idbDeleteFile(scope, name) {
  const store = await fileBlobStore("readwrite");
  await idbRequest(store.delete(idbFileKey(scope, name)));
}

async function idbListFileNames(scope) {
  const store = await fileBlobStore("readonly");
  const records = await idbRequest(store.getAll());
  const prefix = `${String(scope || "").replace(/^\/+|\/+$/g, "")}/`;
  return records
    .filter((record) => String(record?.key || "").startsWith(prefix))
    .map((record) => record.name || String(record.key).slice(prefix.length));
}

async function idbDeleteScope(scope) {
  const store = await fileBlobStore("readwrite");
  const records = await idbRequest(store.getAll());
  const prefix = `${String(scope || "").replace(/^\/+|\/+$/g, "")}/`;
  await Promise.all(records
    .filter((record) => String(record?.key || "").startsWith(prefix))
    .map((record) => idbRequest(store.delete(record.key))));
}

function modelCacheRequest(name) {
  return new Request(`/__asr_vn_model_cache__/${encodeURIComponent(name)}`, { method: "GET" });
}

async function cacheModelFile(name, response) {
  if (!window.caches) throw new Error("Cache Storage is not supported.");
  const cache = await caches.open(MODEL_CACHE_NAME);
  await cache.put(modelCacheRequest(name), response);
}

async function readCachedModelFile(name) {
  if (!window.caches) throw new Error("Cache Storage is not supported.");
  const cache = await caches.open(MODEL_CACHE_NAME);
  const response = await cache.match(modelCacheRequest(name));
  if (!response) throw new Error(`Cached model not found: ${name}`);
  return response.blob();
}

async function deleteCachedModelFile(name) {
  if (!window.caches) return;
  const cache = await caches.open(MODEL_CACHE_NAME);
  await cache.delete(modelCacheRequest(name));
}

async function clearCachedModels() {
  if (window.caches) await caches.delete(MODEL_CACHE_NAME);
}

async function opfsWritableSupported() {
  if (opfsWritableSupportPromise) return opfsWritableSupportPromise;
  opfsWritableSupportPromise = (async () => {
    if (!navigator.storage?.getDirectory) return false;
    try {
      const root = await navigator.storage.getDirectory();
      const handle = await root.getFileHandle("__asr_vn_write_probe__", { create: true });
      const supported = typeof handle.createWritable === "function";
      await root.removeEntry("__asr_vn_write_probe__").catch(() => null);
      return supported;
    } catch (_) {
      return false;
    }
  })();
  return opfsWritableSupportPromise;
}

async function recoverInterruptedLibraryItems() {
  const items = await libraryGetAllItems();
  const now = new Date().toISOString();
  let recovered = 0;
  for (const item of items) {
    if (item.resultStored || item.status === "result_ready" || item.status === "completed") {
      if (resumeAfterKillMeta(item)) {
        await clearResumeAfterKillForItem(item.id).catch(() => null);
      } else {
        await cleanupResumeArtifactsForItem(item.id).catch(() => null);
      }
      continue;
    }
    await cleanupResumeArtifactsForItem(item.id, resumeKeepNamesFromMeta(resumeAfterKillMeta(item))).catch(() => null);
    if (item.status !== "processing" && item.status !== "importing") continue;
    const patch = {
      updatedAt: now,
      interruptedAt: now,
      interruptedStatus: item.status,
    };
    if (item.sourceStored) {
      const meta = resumeAfterKillMeta(item);
      const hasResumeSnapshots = hasDurableResumeSnapshots(meta);
      await libraryPutItem({
        ...item,
        ...patch,
        status: item.resultStored ? "result_ready" : (hasResumeSnapshots ? "processing" : "source_ready"),
        resumeAfterKill: meta ? { ...meta, snapshots: durableResumeSnapshots(meta) } : undefined,
      });
    } else {
      await libraryPutItem({
        ...item,
        ...patch,
        status: "error",
        errorMessage: "Tác vụ trước bị gián đoạn trước khi lưu xong file nguồn.",
      });
    }
    recovered += 1;
  }
  if (recovered) {
    log(`Recovered ${recovered} interrupted offline library item(s).`);
  }
  return recovered;
}

async function zstdModule() {
  if (!zstdModulePromise) {
    zstdModulePromise = import("/vendor/zstd-wasm/zstd-wrapper.js").then(async (module) => {
      await module.init("/vendor/zstd-wasm/zstd.wasm");
      return module;
    });
  }
  return zstdModulePromise;
}

async function compressJsonZstd(data) {
  const module = await zstdModule();
  const json = JSON.stringify(data, null, 2);
  return module.compress(new TextEncoder().encode(json), 6);
}

async function decompressJsonZstd(file) {
  const module = await zstdModule();
  const compressed = new Uint8Array(await file.arrayBuffer());
  const decoded = module.decompress(compressed);
  return JSON.parse(new TextDecoder().decode(decoded));
}

async function updateRuntimeStatus() {
  $("secure-context").textContent = window.isSecureContext ? "yes" : "no";
  $("isolated").textContent = window.crossOriginIsolated ? "yes" : "no";
  $("onnx-runtime").textContent = window.ort ? "loaded" : "missing";

  const persistence = $("storage-persistence");
  if (persistence) {
    if (navigator.storage?.persisted) {
      persistence.textContent = await navigator.storage.persisted() ? "persistent" : "best effort";
    } else {
      persistence.textContent = "unsupported";
    }
  }

  if (navigator.storage?.estimate) {
    const estimate = await navigator.storage.estimate();
    $("storage-quota").textContent = formatBytes(estimate.quota || 0);
    $("storage-used").textContent = formatBytes(estimate.usage || 0);
  }

  const status = $("runtime-status");
  status.textContent = offlineBootstrapReady ? "Offline ready" : (navigator.onLine ? "Bootstrap needed" : "Offline incomplete");
  status.classList.toggle("ok", offlineBootstrapReady);
  status.classList.toggle("warn", !offlineBootstrapReady);
}

async function requestPersistentStorage(noisy = false) {
  if (!navigator.storage?.persist || !navigator.storage?.persisted) {
    if (noisy) log("Persistent storage API is not supported.");
    return false;
  }
  if (await navigator.storage.persisted()) {
    await updateRuntimeStatus();
    return true;
  }
  const ok = await navigator.storage.persist();
  if (noisy) {
    log(ok ? "Persistent storage granted." : "Persistent storage was not granted by the browser.");
  }
  await updateRuntimeStatus();
  return ok;
}

async function fetchWithTimeout(url, init = {}, timeoutMs = STARTUP_FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

async function readStreamChunkWithTimeout(reader, label, timeoutMs = MODEL_DOWNLOAD_STALL_TIMEOUT_MS) {
  let timer = null;
  try {
    return await Promise.race([
      reader.read(),
      new Promise((_, reject) => {
        timer = window.setTimeout(() => {
          reject(new Error(`${label} stalled for ${Math.round(timeoutMs / 1000)}s.`));
        }, timeoutMs);
      }),
    ]);
  } catch (error) {
    try {
      await reader.cancel(error);
    } catch (_) {}
    throw error;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    log("Service worker is not supported by this browser.");
    return;
  }
  let reloadingForServiceWorkerUpdate = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloadingForServiceWorkerUpdate) return;
    reloadingForServiceWorkerUpdate = true;
    log("Service worker updated; reloading app to use the latest offline code.");
    window.location.reload();
  });
  const registrationPromise = navigator.serviceWorker.register("/sw.js", {
    scope: "/",
    updateViaCache: "none",
  });
  registrationPromise.catch((error) => {
    log(`Service worker registration failed after startup continued: ${error.message || String(error)}`);
  });
  const registration = await Promise.race([
    registrationPromise,
    new Promise((resolve) => window.setTimeout(() => resolve(null), STARTUP_FETCH_TIMEOUT_MS)),
  ]);
  if (!registration) {
    log("Service worker registration timed out; continuing with cached app shell.");
    return;
  }
  registration.addEventListener("updatefound", () => {
    const worker = registration.installing;
    if (!worker) return;
    worker.addEventListener("statechange", () => {
      if (worker.state === "installed" && navigator.serviceWorker.controller) {
        log("New offline app version installed; switching to it now.");
      }
    });
  });
  await Promise.race([
    registration.update(),
    new Promise((resolve) => window.setTimeout(resolve, STARTUP_FETCH_TIMEOUT_MS)),
  ]).catch((error) => log(`Service worker update check failed: ${error.message}`));
  log("Service worker registered.");
}

function normalizeManifestForOfflineUse(data) {
  if (!data || typeof data !== "object") return data;
  for (const pack of data.packs || []) {
    for (const file of pack.files || []) {
      if (!file.download_url && file.id) {
        file.download_url = `/api/model-files/${encodeURIComponent(file.id)}`;
      }
    }
    if (pack.server_ready === undefined) pack.server_ready = true;
    if (pack.server_available === undefined) pack.server_available = (pack.files || []).length;
  }
  if (!data.server_model_bundle) {
    const total = (data.packs || []).reduce((sum, pack) => sum + (pack.files || []).length, 0);
    data.server_model_bundle = {
      source: "cached-client-manifest",
      ready: true,
      available: total,
      total,
      missing: [],
      missing_optional: [],
      remote_downloads_enabled: false,
    };
  }
  return data;
}

function saveManifestSnapshot(data) {
  try {
    window.localStorage?.setItem(MANIFEST_STORAGE_KEY, JSON.stringify(data));
  } catch (error) {
    log(`Save manifest snapshot failed: ${error.message || String(error)}`);
  }
}

async function loadCachedManifestSnapshot() {
  if (window.caches) {
    try {
      const response = await caches.match("/api/model-manifest");
      if (response?.ok) return normalizeManifestForOfflineUse(await response.json());
    } catch (error) {
      log(`Read cached model manifest failed: ${error.message || String(error)}`);
    }
  }
  try {
    const raw = window.localStorage?.getItem(MANIFEST_STORAGE_KEY);
    if (raw) return normalizeManifestForOfflineUse(JSON.parse(raw));
  } catch (error) {
    log(`Read manifest snapshot failed: ${error.message || String(error)}`);
  }
  return null;
}

async function loadManifest() {
  let loaded = null;
  const cachedFirst = await loadCachedManifestSnapshot();
  if (cachedFirst) {
    loaded = cachedFirst;
    log("Using cached model manifest.");
  }
  try {
    const response = await fetchWithTimeout(
      "/api/model-manifest",
      { cache: "reload" },
      STARTUP_FETCH_TIMEOUT_MS
    );
    if (!response.ok) throw new Error(`model manifest failed: ${response.status}`);
    loaded = normalizeManifestForOfflineUse(await response.json());
    saveManifestSnapshot(loaded);
    log(cachedFirst ? "Updated model manifest from server." : "Loaded model manifest from server.");
  } catch (error) {
    if (loaded) {
      log(`Using cached model manifest because server manifest is unavailable: ${error.message || String(error)}`);
    } else {
      loaded = normalizeManifestForOfflineUse({
        packs: [],
        server_model_bundle: {
          source: "unavailable-offline",
          ready: false,
          available: 0,
          total: 0,
          missing: [...REQUIRED_OFFLINE_PACK_IDS],
          missing_optional: [],
          remote_downloads_enabled: false,
        },
      });
      log(`Model manifest is unavailable and no cached snapshot exists: ${error.message || String(error)}`);
    }
  }
  manifest = loaded;
  const bundle = manifest.server_model_bundle;
  if (bundle && !bundle.ready) {
    log(`Server model bundle is incomplete: ${bundle.available}/${bundle.total} file(s) available.`);
  }
  await pruneUnusedModelFiles().catch((error) => {
    log(`Prune unused model files failed: ${error.message || String(error)}`);
  });
  await renderPacks();
  await refreshOfflineBootstrapState();
}

function runtimeAssetRequest(url) {
  return new Request(url, { method: "GET" });
}

async function runtimeAssetCache() {
  if (!window.caches) {
    throw new Error("Cache Storage is required for full offline PWA runtime assets.");
  }
  return caches.open(OFFLINE_RUNTIME_CACHE_NAME);
}

async function runtimeAssetStatus(url, cache = null) {
  if (!window.caches) return { ready: false, url, reason: "cache-storage-unavailable" };
  const runtimeCache = cache || await runtimeAssetCache();
  const response = await runtimeCache.match(runtimeAssetRequest(url));
  if (response?.ok) return { ready: true, url, reason: "cached" };
  return { ready: false, url, reason: "missing" };
}

async function requiredRuntimeAssetsStatus() {
  if (!window.caches) {
    return {
      ready: 0,
      total: OFFLINE_RUNTIME_ASSET_URLS.length,
      missing: [...OFFLINE_RUNTIME_ASSET_URLS],
      statuses: OFFLINE_RUNTIME_ASSET_URLS.map((url) => ({ ready: false, url, reason: "cache-storage-unavailable" })),
    };
  }
  const cache = await runtimeAssetCache();
  const statuses = await Promise.all(OFFLINE_RUNTIME_ASSET_URLS.map((url) => runtimeAssetStatus(url, cache)));
  return {
    ready: statuses.filter((item) => item.ready).length,
    total: statuses.length,
    missing: statuses.filter((item) => !item.ready).map((item) => item.url),
    statuses,
  };
}

async function cacheRequiredRuntimeAsset(url, progress = null) {
  const cache = await runtimeAssetCache();
  const cached = await runtimeAssetStatus(url, cache);
  if (cached.ready) {
    if (progress) progress.done += 1;
    return { url, cached: true };
  }
  if (!navigator.onLine) {
    throw new Error(`Cannot cache offline runtime asset while offline: ${url}`);
  }
  const response = await fetchWithTimeout(
    url,
    { cache: "reload", headers: { "X-ASR-Offline-Cache-Fill": "1" } },
    DOWNLOAD_RESPONSE_TIMEOUT_MS
  );
  if (!response.ok || !response.body) {
    throw new Error(`runtime asset download failed for ${url}: ${response.status}`);
  }
  const request = runtimeAssetRequest(url);
  try {
    const reader = response.body.getReader();
    const stream = new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await readStreamChunkWithTimeout(reader, `runtime ${url}`);
          if (done) {
            controller.close();
            return;
          }
          controller.enqueue(value);
        } catch (error) {
          controller.error(error);
        }
      },
      cancel(reason) {
        reader.cancel(reason).catch(() => null);
      },
    });
    await cache.put(request, new Response(stream, {
      headers: response.headers,
      status: response.status,
      statusText: response.statusText,
    }));
    const stored = await runtimeAssetStatus(url, cache);
    if (!stored.ready) throw new Error(`runtime asset did not persist in Cache Storage: ${url}`);
    if (progress) progress.done += 1;
    return { url, cached: false };
  } catch (error) {
    await cache.delete(request).catch(() => null);
    throw error;
  }
}

async function ensureRequiredRuntimeAssetsCached(options = {}) {
  const progressEl = $(options.progressId || "offline-bootstrap-progress-bar");
  const onProgress = options.onProgress || null;
  const initial = await requiredRuntimeAssetsStatus();
  if (initial.ready === initial.total) return initial;

  const progress = { done: initial.ready, total: initial.total };
  setProgress(progressEl, progress.done, progress.total || 1);
  if (onProgress) onProgress(progress.done, progress.total);

  for (const url of OFFLINE_RUNTIME_ASSET_URLS) {
    const status = await runtimeAssetStatus(url);
    if (status.ready) continue;
    log(`Caching offline runtime asset: ${url}`);
    await cacheRequiredRuntimeAsset(url, progress);
    setProgress(progressEl, progress.done, progress.total || 1);
    if (onProgress) onProgress(progress.done, progress.total);
  }

  const finalStatus = await requiredRuntimeAssetsStatus();
  if (finalStatus.ready !== finalStatus.total) {
    throw new Error(`Offline runtime cache is incomplete: ${finalStatus.ready}/${finalStatus.total}. Missing: ${finalStatus.missing.slice(0, 5).join(", ")}`);
  }
  log("Required PWA runtime assets are ready in Cache Storage.");
  return finalStatus;
}

function packSize(pack) {
  return pack.files.reduce((sum, file) => sum + (file.bytes || 0), 0);
}

async function fileStatus(file) {
  try {
    const dir = await opfsModelDir();
    let stored = null;
    if (dir.kind === "idb") {
      stored = await readFallbackModelFile(file);
      const sizeOk = !file.bytes || stored.size === file.bytes;
      if (!sizeOk) {
        clearModelIntegrity(file);
        return {
          ready: false,
          size: stored.size,
          reason: `size mismatch (${formatBytes(stored.size)})`,
        };
      }
      if (file.sha256 && !modelIntegrityIsRecorded(file, stored.size)) {
        try {
          await verifyModelBlobIntegrity(file, stored);
        } catch (error) {
          clearModelIntegrity(file);
          await deleteCachedModelFile(modelFileName(file)).catch(() => null);
          await idbDeleteFile(dir.scope, modelFileName(file)).catch(() => null);
          return { ready: false, size: stored.size, reason: error.message || "sha256 mismatch" };
        }
      }
      return {
        ready: true,
        size: stored.size,
        reason: file.sha256 ? "stored+sha256" : "stored",
      };
    }
    const handle = await dir.getFileHandle(modelFileName(file));
    stored = await handle.getFile();
    const sizeOk = !file.bytes || stored.size === file.bytes;
    if (!sizeOk) {
      clearModelIntegrity(file);
      return {
        ready: false,
        size: stored.size,
        reason: `size mismatch (${formatBytes(stored.size)})`,
      };
    }
    if (file.sha256 && !modelIntegrityIsRecorded(file, stored.size)) {
      try {
        await verifyModelBlobIntegrity(file, stored);
      } catch (error) {
        clearModelIntegrity(file);
        await removeOpfsFile(dir, modelFileName(file)).catch(() => null);
        return { ready: false, size: stored.size, reason: error.message || "sha256 mismatch" };
      }
    }
    return {
      ready: true,
      size: stored.size,
      reason: file.sha256 ? "stored+sha256" : "stored",
    };
  } catch (error) {
    return { ready: false, size: 0, reason: "missing" };
  }
}

async function packStatus(pack) {
  const statuses = await Promise.all(pack.files.map(fileStatus));
  const ready = statuses.filter((item) => item.ready).length;
  return { ready, total: statuses.length, statuses };
}

function requiredOfflinePacks() {
  if (!manifest?.packs) return [];
  return REQUIRED_OFFLINE_PACK_IDS.map((packId) => manifest.packs.find((pack) => pack.id === packId)).filter(Boolean);
}

async function requiredOfflineStatus() {
  const packs = requiredOfflinePacks();
  const runtimeAssets = await requiredRuntimeAssetsStatus();
  const status = {
    ready: runtimeAssets.ready,
    total: runtimeAssets.total,
    bytes: 0,
    readyBytes: 0,
    serverReady: true,
    missingPacks: [],
    runtimeAssets,
  };
  for (const packId of REQUIRED_OFFLINE_PACK_IDS) {
    if (!packs.some((pack) => pack.id === packId)) {
      status.serverReady = false;
      status.missingPacks.push(packId);
    }
  }
  for (const pack of packs) {
    const packState = await packStatus(pack);
    status.ready += packState.ready;
    status.total += packState.total;
    status.bytes += packSize(pack);
    status.serverReady = status.serverReady && pack.server_ready !== false;
    for (const [index, file] of pack.files.entries()) {
      if (packState.statuses[index]?.ready) status.readyBytes += file.bytes || 0;
    }
  }
  status.complete = status.total > 0 && status.ready === status.total;
  return status;
}

function updateProcessButtonState() {
  const button = $("btn-process");
  const benchmarkButton = $("btn-benchmark");
  const disabled = !selectedAudioFile || !offlineBootstrapReady || offlineBootstrapBusy || calibrationBusy || Boolean(selectedLibraryImportPromise);
  const title = selectedLibraryImportPromise
    ? "Đang lưu file nguồn."
    : (calibrationBusy ? "Ứng dụng đang tối ưu cấu hình cho thiết bị." : (offlineBootstrapReady ? "" : "Ứng dụng đang chuẩn bị dữ liệu offline."));
  if (button) {
    button.disabled = disabled;
    button.title = title;
  }
  if (benchmarkButton) {
    benchmarkButton.disabled = disabled;
    benchmarkButton.title = title || "Chạy benchmark WASM/WebGPU và xuất file log.";
  }
  const recalibrationButton = $("btn-recalibration");
  if (recalibrationButton) {
    recalibrationButton.disabled = !offlineBootstrapReady || offlineBootstrapBusy || calibrationBusy;
    recalibrationButton.title = calibrationBusy
      ? "Đang chạy Re-Calibration."
      : (offlineBootstrapReady ? "Đo lại cấu hình tối ưu cho thiết bị này." : "Cần tải đủ model trước.");
  }
}

function updateInstallButtonState() {
  const button = $("btn-install-app");
  const message = $("install-status-msg");
  if (!button) return;
  if (isStandaloneApp()) {
    button.disabled = true;
    button.textContent = "Đã cài đặt ứng dụng";
    button.classList.remove("btn-install-ready");
    if (message) message.style.display = "none";
    return;
  }
  button.hidden = false;
  button.disabled = false;
  button.textContent = "B2: Cài đặt ứng dụng";
  button.title = installPrompt
    ? "Cài ứng dụng bằng Chrome."
    : "Bấm để cài hoặc xem hướng dẫn nếu Chrome chưa hiện native install prompt.";
  button.classList.add("btn-install-ready");
  if (message) {
    message.style.display = installPrompt ? "none" : "";
    message.textContent = "Bấm B2 để cài. Nếu Chrome chưa hiện hộp cài tự động, app sẽ mở hướng dẫn cài thủ công.";
  }
}

function hasRuntimeStandaloneSignal() {
  return window.matchMedia?.("(display-mode: standalone)")?.matches ||
         window.matchMedia?.("(display-mode: minimal-ui)")?.matches ||
         window.matchMedia?.("(display-mode: window-controls-overlay)")?.matches ||
         window.navigator.standalone === true ||
         window.location.search.includes("source=pwa");
}

function hasStoredStandaloneSignal() {
  try {
    return window.localStorage?.getItem(PWA_INSTALLED_FLAG_KEY) === "1";
  } catch (_) {
    return false;
  }
}

function markStandaloneInstalled() {
  try {
    window.localStorage?.setItem(PWA_INSTALLED_FLAG_KEY, "1");
  } catch (_) {}
}

function hasOfflineAppShellSignal() {
  return navigator.onLine === false;
}

function isStandaloneApp() {
  return hasRuntimeStandaloneSignal() || hasStoredStandaloneSignal() || hasOfflineAppShellSignal();
}

function isIOS() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

function detectDevice() {
  const ua = navigator.userAgent;
  if (/Samsung/i.test(ua)) return "samsung";
  if (/Android/i.test(ua)) return "android";
  if (isIOS()) return "ios";
  if (/Macintosh|Mac OS/i.test(ua)) return "macos";
  if (/Windows/i.test(ua)) return "windows";
  return "other";
}

function updateStandaloneUi() {
  const runtimeStandalone = hasRuntimeStandaloneSignal();
  if (runtimeStandalone) markStandaloneInstalled();
  const standalone = runtimeStandalone || hasStoredStandaloneSignal() || hasOfflineAppShellSignal();
  document.body.classList.toggle("standalone-app", standalone);
  document.body.classList.toggle("browser-install-mode", !standalone);
  if (standalone && !offlineBootstrapReady) {
    document.body.classList.add("bootstrap-needed");
    document.body.classList.remove("offline-ready");
  }
  syncCalibrationSetupUi();
  const installPanel = $("install-panel");
  if (installPanel) installPanel.style.display = standalone ? "none" : "";
}

function mountAdvancedSettings() {
  const mount = $("advanced-settings-mount");
  const template = $("advanced-settings-template");
  if (!mount || !template || $("advanced-settings")) return;
  mount.appendChild(template.content.cloneNode(true));
}

function showIOSInstallGuide() {
  window.alert("Trên iPhone/iPad: bấm nút Chia sẻ trong Safari, chọn Thêm vào Màn hình chính, rồi mở app từ biểu tượng vừa tạo.");
}

function showManualInstallGuide() {
  const message = $("install-status-msg");
  if (!message) return;

  // Try to find the active cert tab
  let activeDevice = "windows";
  const certGuide = $("cert-guide");
  if (certGuide) {
      const activeBtn = certGuide.querySelector(".cert-tab-btn.active");
      if (activeBtn) activeDevice = activeBtn.dataset.certTab;
  } else {
      if (/Android/i.test(navigator.userAgent)) activeDevice = "android";
      else if (isIOS()) activeDevice = "ios";
      else if (/Mac/i.test(navigator.userAgent)) activeDevice = "macos";
  }

  const browser = /Edg\//i.test(navigator.userAgent) ? "Edge" : "Chrome";
  let steps = [];

  if (activeDevice === "samsung" || activeDevice === "android") {
      steps = [
        `Mở menu 3 chấm của ${browser} (hoặc trình duyệt hiện tại).`,
        "Chọn Cài đặt ứng dụng (Install app) hoặc Thêm vào màn hình chính (Add to home screen).",
        "Sau khi cài xong, mở Sherpa Vietnamese ASR từ icon ứng dụng trên màn hình để app tự tải model offline."
      ];
  } else if (activeDevice === "ios") {
      steps = [
        "Trên Safari, bấm nút Chia sẻ (Share) ở dưới đáy màn hình.",
        "Cuộn xuống và chọn 'Thêm vào Màn hình chính' (Add to Home Screen).",
        "Sau khi thêm xong, quay ra màn hình chính và mở app từ biểu tượng vừa tạo để tải model offline."
      ];
  } else {
      steps = [
        `Mở menu 3 chấm góc phải trên cùng của ${browser}.`,
        "Chọn Lưu và chia sẻ (Save and share) -> Cài đặt trang dưới dạng ứng dụng (Install page as app) hoặc Tạo lối tắt (Create shortcut) -> Mở dưới dạng cửa sổ.",
        "Nếu không thấy mục cài, hãy kiểm tra lại B1 cert SSL xem đã cài đúng chưa, đóng hoàn toàn trình duyệt rồi thử lại. Nếu app đã được cài trước đó, hãy mở từ Start menu/Desktop thay vì trình duyệt."
      ];
  }

  message.style.display = "";
  message.innerHTML = `
    <div class="manual-install-guide">
      <strong>Trình duyệt chưa gửi hộp cài tự động.</strong>
      <ol>${steps.map((step) => `<li>${step}</li>`).join("")}</ol>
    </div>
  `;
}

function showCertGuide() {
  const guide = $("cert-guide");
  if (!guide) return;
  if (guide.style.display !== "none" && guide.textContent.trim()) {
    guide.style.display = "none";
    return;
  }
  guide.innerHTML = buildCertGuideHTML(detectDevice());
  guide.querySelectorAll("[data-cert-tab]").forEach((button) => {
    button.addEventListener("click", () => switchCertTab(button.dataset.certTab || "android"));
  });
  guide.style.display = "";
}

function buildCertGuideHTML(activeDevice) {
  const devices = [
    ["samsung", "Samsung"],
    ["android", "Android khác"],
    ["ios", "iPhone / iPad"],
    ["windows", "Windows"],
    ["macos", "macOS"],
  ];
  const guides = {
    samsung: [
      "Nhấn Tải cert bên dưới.",
      "Mở Cài đặt -> Bảo mật và riêng tư -> Cài đặt bảo mật khác.",
      "Chọn Cài đặt từ bộ nhớ thiết bị, chọn Chứng chỉ CA, rồi chọn file sherpa-asr-vn.crt.",
      "Đóng hoàn toàn trình duyệt rồi mở lại địa chỉ PWA.",
    ],
    android: [
      "Nhấn Tải cert bên dưới.",
      "Mở Cài đặt -> Bảo mật -> Mã hóa và thông tin xác thực -> Cài chứng chỉ.",
      "Chọn Chứng chỉ CA, không chọn VPN và ứng dụng, rồi chọn file sherpa-asr-vn.crt.",
      "Đóng hoàn toàn trình duyệt rồi mở lại địa chỉ PWA.",
    ],
    ios: [
      "Nhấn Tải cert bên dưới và cho phép tải hồ sơ.",
      "Mở Cài đặt -> Cài đặt chung -> VPN & Quản lý thiết bị, cài hồ sơ vừa tải.",
      "Vào Cài đặt chung -> Giới thiệu -> Cài đặt tin cậy chứng nhận, bật tin cậy cho Sherpa Vietnamese ASR.",
      "Đóng hoàn toàn Safari rồi mở lại địa chỉ PWA.",
    ],
    windows: [
      "Nhấn Tải cert bên dưới và mở file sherpa-asr-vn.crt.",
      "Chọn Install Certificate, Current User.",
      "Chọn Place all certificates in the following store -> Trusted Root Certification Authorities.",
      "Hoàn tất cài đặt rồi khởi động lại trình duyệt.",
    ],
    macos: [
      "Nhấn Tải cert bên dưới và mở file sherpa-asr-vn.crt.",
      "Thêm vào Keychain, mở chứng chỉ Sherpa Vietnamese ASR.",
      "Trong Trust, đặt When using this certificate thành Always Trust.",
      "Đóng cửa sổ, xác nhận mật khẩu rồi khởi động lại trình duyệt.",
    ],
  };
  const selected = guides[activeDevice] ? activeDevice : "android";
  const tabs = devices.map(([id, label]) => (
    `<button class="cert-tab${id === selected ? " active" : ""}" type="button" data-cert-tab="${id}">${label}</button>`
  )).join("");
  const contents = devices.map(([id]) => {
    const lines = (guides[id] || []).map((line) => `<li>${line}</li>`).join("");
    return `<div class="cert-tab-content" id="cert-tab-${id}"${id === selected ? "" : " style=\"display:none\""}><ol>${lines}</ol></div>`;
  }).join("");
  return `
    <div class="cert-guide-content">
      <div class="cert-device-tabs">${tabs}</div>
      ${contents}
      <a href="/install-cert" class="btn btn-sm btn-install" style="display:inline-flex;text-decoration:none;margin-top:8px;width:100%;justify-content:center">Tải cert</a>
    </div>
  `;
}

function switchCertTab(deviceId) {
  document.querySelectorAll(".cert-tab-content").forEach((el) => {
    el.style.display = el.id === `cert-tab-${deviceId}` ? "" : "none";
  });
  document.querySelectorAll(".cert-tab").forEach((el) => {
    el.classList.toggle("active", el.dataset.certTab === deviceId);
  });
}

async function refreshOfflineBootstrapState() {
  const status = await requiredOfflineStatus();
  offlineBootstrapReady = status.complete;
  document.body.classList.toggle("offline-ready", offlineBootstrapReady);
  document.body.classList.toggle("bootstrap-needed", !offlineBootstrapReady);

  const badge = $("offline-bootstrap-status");
  if (badge) {
    badge.textContent = offlineBootstrapReady ? "Sẵn sàng offline" : "Cần dữ liệu";
    badge.classList.toggle("ok", offlineBootstrapReady);
    badge.classList.toggle("warn", !offlineBootstrapReady);
  }

  const message = $("offline-bootstrap-message");
  if (message) {
    if (offlineBootstrapReady) {
      message.textContent = "Đã tải đủ dữ liệu offline. Ứng dụng sẽ mở lại để bật giao diện xử lý.";
    } else if (offlineBootstrapError) {
      message.textContent = offlineBootstrapError;
    } else if (offlineBootstrapBusy) {
      message.textContent = "Vui lòng giữ kết nối và đợi tải dữ liệu lần đầu. Không đóng ứng dụng cho đến khi thanh tiến trình hoàn tất.";
    } else if (!status.serverReady) {
      message.textContent = `Server thiếu bundle dữ liệu: ${status.missingPacks.join(", ") || "model files"}.`;
    } else if (status.runtimeAssets?.ready !== status.runtimeAssets?.total) {
      message.textContent = `Vui lòng giữ kết nối để tải đủ mã chạy offline. Đã có ${status.runtimeAssets.ready}/${status.runtimeAssets.total} file runtime.`;
    } else if (isStandaloneApp()) {
      message.textContent = `Vui lòng giữ kết nối để tải model lần đầu. Đã có ${status.ready}/${status.total} file cần thiết.`;
    } else {
      message.textContent = "Cài đặt ứng dụng trước. Sau khi cài đặt, app tự tải dữ liệu cần thiết.";
    }
  }

  setProgress($("offline-bootstrap-progress-bar"), status.readyBytes, status.bytes || 1);
  const details = $("model-details");
  const detailSummary = $("model-details-summary");
  if (detailSummary) {
    detailSummary.textContent = offlineBootstrapReady
      ? "All required models are stored."
      : `${status.ready}/${status.total} required file(s) ready.`;
  }
  if (details && offlineBootstrapReady && !offlineBootstrapBusy) {
    details.open = false;
  }

  const runtime = $("runtime-status");
  if (runtime) {
    runtime.textContent = offlineBootstrapReady ? "Sẵn sàng offline" : (navigator.onLine ? "Cần cài dữ liệu" : "Chưa đủ dữ liệu offline");
    runtime.classList.toggle("ok", offlineBootstrapReady);
    runtime.classList.toggle("warn", !offlineBootstrapReady);
  }
  syncCalibrationSetupUi();
  updateProcessButtonState();
  updateInstallButtonState();
  if (offlineBootstrapReady && !offlineBootstrapBusy) {
    window.setTimeout(() => {
      runDeviceCalibrationIfNeeded("offline ready").catch((error) => log(`[Calibration] Auto-run failed: ${error.message}`));
    }, 1200);
  }
  return status;
}

async function renderPacks() {
  const root = $("model-packs");
  root.textContent = "";
  for (const pack of manifest.packs || []) {
    const status = await packStatus(pack);
    const section = document.createElement("section");
    section.className = "pack";

    const header = document.createElement("div");
    header.className = "pack-header";
    const serverReady = pack.server_ready !== false;
    const serverText = serverReady
      ? `${pack.files.length}/${pack.files.length} on server`
      : `${pack.server_available || 0}/${pack.files.length} on server`;
    header.innerHTML = `
      <div>
        <strong>${pack.name}</strong>
        <p>${pack.description || ""}</p>
      </div>
      <div class="pack-actions">
        <div class="pack-status">${status.ready}/${status.total} ready - ${serverText} - ${formatBytes(packSize(pack))}</div>
      </div>
    `;
    section.appendChild(header);

    const progress = document.createElement("div");
    progress.className = "progress";
    progress.innerHTML = `<span id="progress-${pack.id}"></span>`;
    section.appendChild(progress);

    for (const [index, file] of pack.files.entries()) {
      const itemStatus = status.statuses[index];
      const row = document.createElement("div");
      row.className = "file-row";
      const serverLabel = file.available_local ? "server bundle" : file.server_status || "missing on server";
      row.innerHTML = `
        <span>${file.id}</span>
        <span>${formatBytes(file.bytes)}</span>
        <span>${serverLabel}</span>
        <span class="${itemStatus.ready ? "ready" : "missing"}">${itemStatus.ready ? "ready" : itemStatus.reason}</span>
      `;
      section.appendChild(row);
    }

    root.appendChild(section);
  }

}

async function opfsModelDir() {
  if (!await opfsWritableSupported()) {
    return { kind: "idb", scope: "models" };
  }
  if (!navigator.storage?.getDirectory) {
    throw new Error("OPFS is not supported by this browser.");
  }
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("models", { create: true });
}

async function opfsLibraryDir() {
  if (!await opfsWritableSupported()) {
    return { kind: "idb", scope: "library" };
  }
  if (!navigator.storage?.getDirectory) {
    throw new Error("OPFS is not supported by this browser.");
  }
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("library", { create: true });
}

async function opfsLibraryItemDir(id, create = true) {
  const root = await opfsLibraryDir();
  if (root.kind === "idb") {
    return { kind: "idb", scope: `library/${id}` };
  }
  return root.getDirectoryHandle(id, { create });
}

async function writeOpfsFile(dir, name, data) {
  if (dir?.kind === "idb") {
    await idbPutFile(dir.scope, name, data);
    return;
  }
  const handle = await dir.getFileHandle(name, { create: true });
  if (typeof handle.createWritable !== "function") {
    await idbPutFile(dir.scope || "opfs-fallback", name, data);
    return;
  }
  const writable = await handle.createWritable();
  try {
    await writable.write(data);
    await writable.close();
  } catch (error) {
    await writable.abort().catch(() => null);
    throw error;
  }
}

async function readOpfsFile(dir, name) {
  if (dir?.kind === "idb") {
    return idbGetFile(dir.scope, name);
  }
  const handle = await dir.getFileHandle(name);
  return handle.getFile();
}

async function removeOpfsFile(dir, name) {
  if (dir?.kind === "idb") {
    await idbDeleteFile(dir.scope, name).catch(() => null);
    return;
  }
  await dir.removeEntry(name).catch(() => null);
}

function resumeSnapshotBaseName(stage, kind = "json") {
  const safeStage = String(stage || "stage").replace(/[^a-z0-9_-]+/gi, "_").toLowerCase();
  const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
  return `${RESUME_AFTER_KILL_PREFIX}${safeStage}_${Date.now()}_${id}.${kind === "f32" ? "f32" : "json.zst"}`;
}

function resumePipelineSignature(file, pipelineOptions) {
  const options = {
    asrModel: pipelineOptions.asrModel?.id || "",
    punctuationLevel: pipelineOptions.punctuationLevel,
    caseLevel: pipelineOptions.caseLevel,
    speakerDiarization: pipelineOptions.speakerDiarization,
    speakerModel: pipelineOptions.speakerModel,
    numSpeakers: pipelineOptions.numSpeakers,
    overlapSeparation: pipelineOptions.overlapSeparation,
    rmsNormalize: pipelineOptions.rmsNormalize,
    bypassVad: pipelineOptions.bypassVad,
    hotwordsEnabled: pipelineOptions.hotwordsEnabled,
    hotwordsScore: pipelineOptions.hotwordsScore,
    hotwordsHash: hashString(pipelineOptions.hotwordsText || ""),
  };
  return hashString(JSON.stringify({
    sourceName: file?.name || "",
    sourceSize: file?.size || 0,
    sourceLastModified: file?.lastModified || 0,
    options,
  }));
}

function resumeAfterKillMeta(item) {
  return item?.resumeAfterKill && typeof item.resumeAfterKill === "object"
    ? item.resumeAfterKill
    : null;
}

function newestResumeSnapshot(meta, stage, kind = null) {
  const snapshots = meta?.snapshots?.[stage] || [];
  const filtered = kind ? snapshots.filter((snapshot) => snapshot.kind === kind) : snapshots;
  return filtered
    .slice()
    .sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")))[0] || null;
}

function shouldWriteResumeCheckpoint(context) {
  return Boolean(context && !context.disabled && (context.armed || context.hasExistingResume));
}

function waitForUiPaint() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame !== "function") {
      window.setTimeout(resolve, 0);
      return;
    }
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

async function createResumeAfterKillContext(file, pipelineOptions, options = {}) {
  if (options.benchmarkOnly || options.saveLibraryResult === false || !selectedLibraryItemId) return null;
  const item = await libraryGetItem(selectedLibraryItemId).catch(() => null);
  if (!item?.id) return null;
  const signature = resumePipelineSignature(file, pipelineOptions);
  const meta = resumeAfterKillMeta(item);
  const staleResume = Boolean(meta?.signature && meta.signature !== signature);
  const hasExistingResume = Boolean(meta?.signature === signature && hasDurableResumeSnapshots(meta));
  const context = {
    itemId: item.id,
    signature,
    armed: true,
    hasExistingResume,
    disabled: false,
    manual: Boolean(pipelineOptions.resumeAfterKill),
  };
  await libraryPutItem({
    ...item,
    status: "processing",
    resumeAfterKill: {
      schemaVersion: 1,
      enabled: context.armed,
      armedAt: context.armed ? (meta?.armedAt || new Date().toISOString()) : null,
      reason: context.manual ? "manual_config" : (hasExistingResume ? "resume_existing_checkpoint" : "auto_processing_checkpoint"),
      signature,
      latestStage: meta?.signature === signature ? meta.latestStage || null : null,
      snapshots: meta?.signature === signature ? durableResumeSnapshots(meta) : {},
      updatedAt: new Date().toISOString(),
    },
  });
  if (staleResume) {
    await cleanupResumeArtifactsForItem(item.id);
  }
  return context;
}

async function armResumeAfterKill(reason = "hidden") {
  const context = activeResumeAfterKillContext;
  if (!context || context.disabled || context.armed) return;
  context.armed = true;
  try {
    const item = await libraryGetItem(context.itemId);
    if (!item) return;
    const meta = resumeAfterKillMeta(item) || {};
    await libraryPutItem({
      ...item,
      resumeAfterKill: {
        schemaVersion: 1,
        ...meta,
        enabled: true,
        armedAt: meta.armedAt || new Date().toISOString(),
        reason,
        signature: context.signature,
        snapshots: meta.signature === context.signature ? durableResumeSnapshots(meta) : {},
        updatedAt: new Date().toISOString(),
      },
    });
    log(`[resume_after_kill] Armed after ${reason}; checkpoints will be written at stage boundaries.`);
  } catch (error) {
    log(`[resume_after_kill] Arm failed: ${error.message || String(error)}`);
  }
}

async function cleanupResumeArtifactsForItem(itemId, keepNames = new Set()) {
  let dir = null;
  try {
    dir = await opfsLibraryItemDir(itemId, false);
  } catch (_) {
    return;
  }
  if (dir.kind === "idb") {
    const names = await idbListFileNames(dir.scope).catch(() => []);
    for (const name of names) {
      if (!String(name).startsWith(RESUME_AFTER_KILL_PREFIX)) continue;
      if (keepNames.has(name)) continue;
      await idbDeleteFile(dir.scope, name).catch(() => null);
    }
    return;
  }
  for await (const name of dir.keys()) {
    if (!String(name).startsWith(RESUME_AFTER_KILL_PREFIX)) continue;
    if (keepNames.has(name)) continue;
    await removeOpfsFile(dir, name);
  }
}

function resumeKeepNamesFromMeta(meta) {
  const keepNames = new Set();
  for (const [stage, list] of Object.entries(meta?.snapshots || {})) {
    if (!RESUME_AFTER_KILL_DURABLE_STAGES.has(stage)) continue;
    for (const snapshot of list || []) {
      if (snapshot.fileName) keepNames.add(snapshot.fileName);
    }
  }
  return keepNames;
}

function hasDurableResumeSnapshots(meta) {
  return Boolean(
    meta?.enabled &&
    meta?.snapshots &&
    Object.entries(meta.snapshots).some(([stage, snapshots]) => (
      RESUME_AFTER_KILL_DURABLE_STAGES.has(stage) &&
      Array.isArray(snapshots) &&
      snapshots.length
    ))
  );
}

function durableResumeSnapshots(meta) {
  const snapshots = {};
  for (const [stage, list] of Object.entries(meta?.snapshots || {})) {
    if (RESUME_AFTER_KILL_DURABLE_STAGES.has(stage) && Array.isArray(list) && list.length) {
      snapshots[stage] = list;
    }
  }
  return snapshots;
}

function encodeVadCheckpoint(vad, vadElapsed = 0, pipelineOptions = {}) {
  if (!vad) return null;
  const probabilities = Array.isArray(vad.probabilities) ? vad.probabilities : [];
  const probabilitiesU16 = probabilities.length
    ? probabilities.map((value) => Math.max(0, Math.min(65535, Math.round(Number(value || 0) * 65535))))
    : [];
  return {
    segments: (vad.segments || []).map((segment) => ({
      start: Number(segment.start) || 0,
      end: Number(segment.end) || 0,
    })),
    probabilitiesU16,
    probabilityScale: 65535,
    boosted: Boolean(vad.boosted),
    peak: Number(vad.peak || 0),
    scale: Number(vad.scale || 1),
    bypassed: Boolean(vad.bypassed || pipelineOptions.bypassVad),
    elapsed: Number(vadElapsed || 0),
    provider: vad.benchmarkSelectedProvider || (pipelineOptions.bypassVad ? "off" : vadExecutionProvider),
    sampleRate: VAD_SAMPLE_RATE,
    windowSize: VAD_WINDOW_SIZE,
  };
}

function decodeVadCheckpoint(data) {
  if (!data || !Array.isArray(data.segments)) return null;
  const probabilityScale = Number(data.probabilityScale || 65535) || 65535;
  const probabilities = Array.isArray(data.probabilities)
    ? data.probabilities.map((value) => Number(value) || 0)
    : (Array.isArray(data.probabilitiesU16)
        ? data.probabilitiesU16.map((value) => (Number(value) || 0) / probabilityScale)
        : (Array.isArray(data.probabilitiesU8)
            ? data.probabilitiesU8.map((value) => (Number(value) || 0) / (Number(data.probabilityScale || 255) || 255))
            : []));
  return {
    segments: data.segments.map((segment) => ({
      start: Number(segment.start) || 0,
      end: Number(segment.end) || 0,
    })).filter((segment) => segment.end > segment.start),
    probabilities,
    boosted: Boolean(data.boosted),
    peak: Number(data.peak || 0),
    scale: Number(data.scale || 1),
    bypassed: Boolean(data.bypassed),
    benchmarkSelectedProvider: data.provider || null,
  };
}

function encodeCheckpointSegments(segments = [], includeSpeaker = false) {
  if (!Array.isArray(segments)) return [];
  return segments.map((segment) => {
    const output = {
      start: Number(segment.start) || 0,
      end: Number(segment.end) || 0,
    };
    if (includeSpeaker) output.speaker = normalizeSpeakerId(segment.speaker);
    return output;
  }).filter((segment) => segment.end > segment.start);
}

function encodeExecutionProviderCheckpoint(provider = null) {
  if (!provider || typeof provider !== "object") return null;
  return {
    segmentation: provider.segmentation || null,
    embedding: provider.embedding || null,
  };
}

function encodeCamppSpeechRegionsCheckpoint(vad) {
  if (!vad) return null;
  return {
    regions: encodeCheckpointSegments(vad.regions || []),
    overlapRegions: encodeCheckpointSegments(vad.overlapRegions || []),
    chunks: Number(vad.chunks || 0),
    provider: vad.benchmarkSelectedProvider || vad.provider || null,
  };
}

function decodeCamppSpeechRegionsCheckpoint(data) {
  if (!data || !Array.isArray(data.regions)) return null;
  const regions = encodeCheckpointSegments(data.regions || []);
  if (!regions.length) return null;
  return {
    regions,
    overlapRegions: encodeCheckpointSegments(data.overlapRegions || []),
    chunks: Number(data.chunks || 0),
    provider: data.provider || null,
    benchmarkSelectedProvider: data.provider || null,
  };
}

function encodeBinaryMaskRowsRle(mask, chunks, numFrames, speakers) {
  const rows = [];
  for (let chunk = 0; chunk < chunks; chunk += 1) {
    for (let speaker = 0; speaker < speakers; speaker += 1) {
      const runs = [];
      let start = -1;
      let length = 0;
      for (let frame = 0; frame < numFrames; frame += 1) {
        const active = mask[(chunk * numFrames + frame) * speakers + speaker] ? 1 : 0;
        if (active && start < 0) {
          start = frame;
          length = 1;
        } else if (active) {
          length += 1;
        } else if (start >= 0) {
          runs.push([start, length]);
          start = -1;
          length = 0;
        }
      }
      if (start >= 0) runs.push([start, length]);
      if (runs.length) rows.push({ chunk, speaker, runs });
    }
  }
  return rows;
}

function decodeBinaryMaskRowsRle(rows, chunks, numFrames, speakers) {
  const mask = new Uint8Array(chunks * numFrames * speakers);
  for (const row of rows || []) {
    const chunk = Number(row?.chunk);
    const speaker = Number(row?.speaker);
    if (!Number.isInteger(chunk) || chunk < 0 || chunk >= chunks) continue;
    if (!Number.isInteger(speaker) || speaker < 0 || speaker >= speakers) continue;
    for (const run of row.runs || []) {
      const start = Math.max(0, Math.min(numFrames, Number(run?.[0]) || 0));
      const length = Math.max(0, Number(run?.[1]) || 0);
      const end = Math.min(numFrames, start + length);
      for (let frame = start; frame < end; frame += 1) {
        mask[(chunk * numFrames + frame) * speakers + speaker] = 1;
      }
    }
  }
  return mask;
}

function encodePyannoteSegmentationCheckpoint(segmentation, binarized, totalSamples) {
  if (!segmentation || !binarized) return null;
  const chunks = Number(segmentation.chunks || 0);
  const numFrames = Number(segmentation.numFrames || 0);
  if (!chunks || !numFrames) return null;
  return {
    totalSamples: Number(totalSamples || 0),
    chunks,
    numFrames,
    starts: Array.from(segmentation.starts || [], (value) => Number(value) || 0),
    speakersPerChunk: PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
    activeRows: encodeBinaryMaskRowsRle(binarized, chunks, numFrames, PYANNOTE_MAX_SPEAKERS_PER_CHUNK),
    provider: segmentation.benchmarkSelectedProvider || segmentation.provider || null,
  };
}

function decodePyannoteSegmentationCheckpoint(data, totalSamples) {
  if (!data || !Array.isArray(data.starts) || !Array.isArray(data.activeRows)) return null;
  if (Number(data.totalSamples || 0) !== Number(totalSamples || 0)) return null;
  const chunks = Number(data.chunks || 0);
  const numFrames = Number(data.numFrames || 0);
  const speakers = Number(data.speakersPerChunk || PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
  if (!chunks || !numFrames || speakers !== PYANNOTE_MAX_SPEAKERS_PER_CHUNK) return null;
  if (data.starts.length !== chunks) return null;
  return {
    segmentation: {
      chunks,
      numFrames,
      starts: data.starts.map((value) => Number(value) || 0),
      provider: data.provider || null,
      benchmarkSelectedProvider: data.provider || null,
    },
    binarized: decodeBinaryMaskRowsRle(data.activeRows, chunks, numFrames, PYANNOTE_MAX_SPEAKERS_PER_CHUNK),
  };
}

function encodeDiarizationCheckpoint(diarization) {
  if (!diarization) return null;
  const rawSegments = encodeCheckpointSegments(diarization.rawSegments || [], true);
  return {
    segments: encodeCheckpointSegments(diarization.segments || [], true),
    ...(rawSegments.length ? { rawSegments } : {}),
    speakers: Number(diarization.speakers || 0),
    elapsed: Number(diarization.elapsed || 0),
    chunks: Number(diarization.chunks || 0),
    embeddings: Number(diarization.embeddings || 0),
    overlapRegions: encodeCheckpointSegments(diarization.overlapRegions || []),
    backend: diarization.backend || "off",
    executionProvider: encodeExecutionProviderCheckpoint(diarization.executionProvider),
  };
}

function decodeDiarizationCheckpoint(data) {
  if (!data || !Array.isArray(data.segments)) return null;
  const segments = encodeCheckpointSegments(data.segments || [], true);
  const rawSegments = encodeCheckpointSegments(data.rawSegments || [], true);
  return {
    segments,
    ...(rawSegments.length ? { rawSegments } : {}),
    speakers: Number(data.speakers || 0),
    elapsed: Number(data.elapsed || 0),
    chunks: Number(data.chunks || 0),
    embeddings: Number(data.embeddings || 0),
    overlapRegions: encodeCheckpointSegments(data.overlapRegions || []),
    backend: data.backend || "off",
    executionProvider: encodeExecutionProviderCheckpoint(data.executionProvider),
  };
}

function asrChunksSignature(chunks, modelConfig) {
  return hashString(JSON.stringify({
    model: modelConfig?.id || "",
    chunks: (chunks || []).map((chunk) => ({
      index: chunk.index,
      start: chunk.start,
      end: chunk.end,
      sourceStart: chunk.sourceStart ?? null,
      sourceEnd: chunk.sourceEnd ?? null,
    })),
  }));
}

function encodeAsrChunksCheckpoint(modelConfig, chunks, results, speechSeconds) {
  return {
    model: modelConfig?.id || "",
    chunksSignature: asrChunksSignature(chunks, modelConfig),
    totalChunks: chunks.length,
    speechSeconds: Number(speechSeconds || 0),
    results: (results || []).filter(Boolean).map((result) => ({
      ...result,
      result: result.result || {},
    })),
  };
}

function decodeAsrChunksCheckpoint(data, chunks, modelConfig) {
  if (!data || data.model !== modelConfig?.id) return null;
  const signatureMatches = data.chunksSignature === asrChunksSignature(chunks, modelConfig);
  if (!signatureMatches && Number(data.totalChunks || 0) !== chunks.length) return null;
  const output = new Array(chunks.length);
  let mismatched = false;
  for (const item of data.results || []) {
    const index = Number(item?.index || 0);
    if (!Number.isInteger(index) || index < 1 || index > chunks.length) continue;
    if (!signatureMatches) {
      const current = chunks[index - 1];
      const sameChunk =
        Math.abs(Number(item.start || 0) - Number(current.start || 0)) <= 1 &&
        Math.abs(Number(item.end || 0) - Number(current.end || 0)) <= 1 &&
        Math.abs(Number(item.sourceStart ?? -1) - Number(current.sourceStart ?? -1)) <= 1 &&
        Math.abs(Number(item.sourceEnd ?? -1) - Number(current.sourceEnd ?? -1)) <= 1;
      if (!sameChunk) {
        mismatched = true;
        break;
      }
    }
    output[index - 1] = item;
  }
  if (mismatched) return null;
  return output;
}

async function commitResumeSnapshot(context, stage, kind, fileName, bytes, extra = {}) {
  const item = await libraryGetItem(context.itemId);
  if (!item) return null;
  const previousMeta = resumeAfterKillMeta(item) || {};
  const snapshots = previousMeta.signature === context.signature
    ? durableResumeSnapshots(previousMeta)
    : {};
  const stageList = (snapshots[stage] || []).slice();
  const snapshot = {
    stage,
    kind,
    fileName,
    bytes: bytes || 0,
    createdAt: new Date().toISOString(),
    signature: context.signature,
    ...extra,
  };
  stageList.push(snapshot);
  stageList.sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")));
  snapshots[stage] = stageList.slice(0, RESUME_AFTER_KILL_KEEP_SNAPSHOTS);
  const keepNames = resumeKeepNamesFromMeta({ snapshots });
  await libraryPutItem({
    ...item,
    resumeAfterKill: {
      schemaVersion: 1,
      enabled: true,
      armedAt: previousMeta.armedAt || new Date().toISOString(),
      reason: previousMeta.reason || "hidden",
      signature: context.signature,
      latestStage: stage,
      snapshots,
      updatedAt: new Date().toISOString(),
    },
  });
  await cleanupResumeArtifactsForItem(context.itemId, keepNames);
  return snapshot;
}

async function writeResumeJsonCheckpoint(context, stage, data, extra = {}) {
  if (!shouldWriteResumeCheckpoint(context)) return null;
  const dir = await opfsLibraryItemDir(context.itemId, true);
  const fileName = resumeSnapshotBaseName(stage, "json");
  const compressed = await compressJsonZstd({
    schemaVersion: 1,
    kind: "resume_after_kill_checkpoint",
    stage,
    signature: context.signature,
    createdAt: new Date().toISOString(),
    data,
  });
  await writeOpfsFile(dir, fileName, compressed);
  const snapshot = await commitResumeSnapshot(context, stage, "json", fileName, compressed.byteLength, extra);
  log(`[resume_after_kill] Saved ${stage} checkpoint (${formatBytes(compressed.byteLength)}).`);
  return snapshot;
}

async function readResumeJsonCheckpoint(itemId, snapshot) {
  const dir = await opfsLibraryItemDir(itemId, false);
  const file = await readOpfsFile(dir, snapshot.fileName);
  const payload = await decompressJsonZstd(file);
  return payload?.data;
}

async function loadResumeAfterKillCheckpoints(context) {
  if (!context?.hasExistingResume) return {};
  const item = await libraryGetItem(context.itemId).catch(() => null);
  const meta = resumeAfterKillMeta(item);
  if (!meta || meta.signature !== context.signature) return {};
  const output = {};
  for (const stage of RESUME_AFTER_KILL_DURABLE_STAGES) {
    const snapshot = newestResumeSnapshot(meta, stage, "json");
    if (!snapshot) continue;
    try {
      output[stage] = await readResumeJsonCheckpoint(context.itemId, snapshot);
      output[`${stage}Snapshot`] = snapshot;
    } catch (error) {
      log(`[resume_after_kill] Ignoring corrupt ${stage} checkpoint: ${error.message || String(error)}`);
    }
  }
  if (Object.keys(output).length) {
    log(`[resume_after_kill] Loaded checkpoint(s): ${Object.keys(output).filter((key) => !key.endsWith("Snapshot")).join(", ")}.`);
  }
  return output;
}

async function clearResumeAfterKillForItem(itemId) {
  const item = await libraryGetItem(itemId).catch(() => null);
  if (!item) return;
  const { resumeAfterKill, ...rest } = item;
  await libraryPutItem({ ...rest, updatedAt: new Date().toISOString() });
  await cleanupResumeArtifactsForItem(itemId);
}

function libraryItemId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function libraryStatusLabel(item) {
  return libraryStatusMeta(item).label;
}

function libraryStatusMeta(item) {
  if (window.getAsrFileStatus) return window.getAsrFileStatus(item);
  return {
    status: item?.resultStored ? "completed" : (item?.status === "processing" ? "processing" : "waiting"),
    label: item?.resultStored ? "Ho\u00e0n th\u00e0nh" : (item?.status === "processing" ? "\u0110ang x\u1eed l\u00fd" : "Ch\u1edd x\u1eed l\u00fd"),
    message: item?.errorMessage || item?.error_message || "",
  };
}

function libraryDisplayName(item) {
  return item.meetingName || item.meeting_name || safeFileBaseName(item.sourceName || item.original_filename, "Ghi âm");
}

function formatLibraryDate(dateStr) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return String(dateStr);
  return `${date.toLocaleDateString("vi-VN")} ${date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}`;
}

function librarySearchText(item) {
  const status = libraryStatusMeta(item);
  return normalizeVietnamese([
    libraryDisplayName(item),
    item.sourceName,
    status.label,
    item.asrModel,
    item.speakerModel,
  ].filter(Boolean).join(" "));
}

async function librarySourceFile(item) {
  const dir = await opfsLibraryItemDir(item.id, false);
  const stored = await readOpfsFile(dir, LIBRARY_SOURCE_FILE);
  return new File([stored], item.sourceName || stored.name || "source.bin", {
    type: item.sourceType || stored.type || "application/octet-stream",
    lastModified: item.sourceLastModified || stored.lastModified,
  });
}

async function libraryResultJson(item) {
  const dir = await opfsLibraryItemDir(item.id, false);
  const file = await readOpfsFile(dir, LIBRARY_RESULT_FILE);
  return decompressJsonZstd(file);
}

async function createLibraryItemFromSourceFile(file) {
  const now = new Date().toISOString();
  const item = {
    id: libraryItemId(),
    status: "importing",
    meetingName: safeFileBaseName(file.name || "source.bin", "Ghi âm"),
    sourceName: file.name || "source.bin",
    sourceType: file.type || "",
    sourceSize: file.size || 0,
    sourceLastModified: file.lastModified || Date.now(),
    sourceStored: false,
    resultStored: false,
    resultBytes: 0,
    durationSec: 0,
    transcriptChars: 0,
    speakers: 0,
    asrModel: "",
    speakerModel: "",
    createdAt: now,
    updatedAt: now,
  };
  await libraryPutItem(item);
  await renderLibrary();

  try {
    const dir = await opfsLibraryItemDir(item.id, true);
    await writeOpfsFile(dir, LIBRARY_SOURCE_FILE, file);
    item.status = "source_ready";
    item.sourceStored = true;
    item.updatedAt = new Date().toISOString();
    await libraryPutItem(item);
    log(`Stored source in offline library: ${item.sourceName}`);
    return item;
  } catch (error) {
    await deleteLibraryItem(item.id, { quiet: true }).catch(() => null);
    throw error;
  } finally {
    await renderLibrary();
    await updateRuntimeStatus();
  }
}

async function updateLibraryItem(id, patch) {
  const item = await libraryGetItem(id);
  if (!item) throw new Error(`Library item not found: ${id}`);
  const updated = { ...item, ...patch, updatedAt: new Date().toISOString() };
  await libraryPutItem(updated);
  await renderLibrary();
  return updated;
}

async function saveCurrentEditorResultToLibrary() {
  if (!editorState?.libraryItemId || !editorState.segments?.length) return;
  const item = await libraryGetItem(editorState.libraryItemId);
  if (!item) return;
  const data = serializeEditorAsrJson();
  const compressed = await compressJsonZstd(data);
  const dir = await opfsLibraryItemDir(item.id, true);
  await writeOpfsFile(dir, LIBRARY_RESULT_FILE, compressed);
  await updateLibraryItem(item.id, {
    status: "result_ready",
    resultStored: true,
    resultBytes: compressed.byteLength,
    durationSec: editorState.duration || item.durationSec || 0,
    transcriptChars: editorState.text?.length || formatEditorTranscriptText().length,
    speakers: collectEditorSpeakerIds(editorState.segments, editorState.rawSpeakerSegments, editorState.overlapSegments).length,
    asrModel: editorState.asr?.model || editorState.pipelineOptions?.asrModel?.id || item.asrModel || "",
    speakerModel: editorState.pipelineOptions?.speakerModel || item.speakerModel || "",
  });
  log(`Saved library result: ${item.sourceName} (${formatBytes(compressed.byteLength)} .zst)`);
}

function scheduleLibraryResultAutosave() {
  if (!editorState?.libraryItemId) return;
  clearTimeout(libraryAutosaveTimer);
  libraryAutosaveTimer = setTimeout(() => {
    saveCurrentEditorResultToLibrary().catch((error) => {
      log(`Library autosave failed: ${error.message}`);
    });
  }, 700);
}

async function openLibraryItem(id) {
  const item = await libraryGetItem(id);
  if (!item) throw new Error("Library item not found.");
  clearEditorResult();
  hidePipelineProgress();
  selectedLibraryItemId = id;
  const source = await librarySourceFile(item);
  selectedAudioFile = source;
  updateSelectedFileUi({ name: libraryDisplayName(item), size: item.sourceSize });
  updateProcessButtonState();
  revealProcessingPanel();

  if (item.resultStored) {
    const data = await libraryResultJson(item);
    setEditorResultFromAsrJson(data, item.sourceName);
    editorState.libraryItemId = id;
    await attachEditorAudioFile(source, { preserveLibraryItem: true });
    editorState.libraryItemId = id;
    renderEditor();
    log(`Opened library result: ${item.sourceName}`);
    hidePipelineProgress();
  } else {
    showSelectedAudioPreview(source);
    log(`Selected library source: ${item.sourceName}`);
    hidePipelineProgress();
  }
}

async function processLibraryItem(id, options = {}) {
  const item = await libraryGetItem(id);
  if (!item) throw new Error("Library item not found.");
  selectedLibraryItemId = id;
  selectedAudioFile = await librarySourceFile(item);
  updateSelectedFileUi({ name: libraryDisplayName(item), size: item.sourceSize });
  updateProcessButtonState();
  revealProcessingPanel();
  if (options.resume) {
    await doProcessSelectedAudioFile();
  } else {
    await processSelectedAudioFile();
  }
}

async function deleteLibraryResult(id) {
  const item = await libraryGetItem(id);
  if (!item) return;
  const dir = await opfsLibraryItemDir(id, false);
  await removeOpfsFile(dir, LIBRARY_RESULT_FILE);
  await clearResumeAfterKillForItem(id).catch(() => null);
  await updateLibraryItem(id, {
    status: "source_ready",
    resultStored: false,
    resultBytes: 0,
    transcriptChars: 0,
    speakers: 0,
  });
  if (editorState?.libraryItemId === id) {
    editorState.libraryItemId = null;
  }
  log(`Deleted library result: ${item.sourceName}`);
}

async function deleteLibraryItem(id, options = {}) {
  const item = await libraryGetItem(id);
  const wasSelected = selectedLibraryItemId === id;
  const wasEditor = editorState?.libraryItemId === id;
  const root = await opfsLibraryDir();
  if (root.kind === "idb") {
    await idbDeleteScope(`library/${id}`).catch(() => null);
  } else {
    await root.removeEntry(id, { recursive: true }).catch(() => null);
  }
  await libraryDeleteItemRecord(id);
  if (wasSelected) {
    selectedLibraryItemId = null;
    selectedAudioFile = null;
    selectedLibraryImportPromise = null;
    updateSelectedFileUi(null);
    updateProcessButtonState();
  }
  if (activeResumeAfterKillContext?.itemId === id) {
    activeResumeAfterKillContext.disabled = true;
  }
  if (wasEditor) {
    clearEditorResult();
  }
  if (!options.quiet) {
    log(`Deleted library item: ${item?.sourceName || id}`);
    await renderLibrary();
    await updateRuntimeStatus();
  }
}

function updateLibrarySummary(items) {
  const summary = $("library-summary");
  if (!summary) return;
  const totalBytes = items.reduce((sum, item) => sum + (item.sourceSize || 0) + (item.resultBytes || 0), 0);
  summary.textContent = items.length
    ? `${items.length} tập tin, ${formatBytes(totalBytes)} lưu trong bộ nhớ riêng của trình duyệt.`
    : "Chưa có tập tin nào.";
}

function currentMeetingSearchQuery() {
  return $("meetings-search")?.value?.trim() || "";
}

function filteredMeetings() {
  const query = normalizeVietnamese(currentMeetingSearchQuery());
  if (!query) return _allMeetings;
  return _allMeetings.filter((item) => librarySearchText(item).includes(query));
}

function toggleMeetingsPanel() {
  const panel = $("meetings-panel");
  if (!panel) return;
  if (panel.style.display === "none" || !panel.style.display) {
    closeHotwordDialog();
    panel.style.display = "flex";
    _meetingsPage = 1;
    loadMeetings();
  } else {
    closeMeetingsPanel();
  }
}

function closeMeetingsPanel() {
  const panel = $("meetings-panel");
  if (panel) panel.style.display = "none";
}

function revealProcessingPanel() {
  const filePanel = document.querySelector(".file-panel");
  if (filePanel?.scrollIntoView) {
    filePanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function loadMeetings() {
  const listEl = $("meetings-list");
  if (listEl) {
    listEl.textContent = "";
    const loading = document.createElement("div");
    loading.className = "meetings-loading";
    loading.textContent = "Đang tải...";
    listEl.appendChild(loading);
  }

  try {
    _allMeetings = await libraryGetAllItems();
    updateLibrarySummary(_allMeetings);
    renderMeetingsPage();
  } catch (error) {
    if ($("library-summary")) $("library-summary").textContent = "Không đọc được thư viện offline.";
    if (listEl) {
      listEl.textContent = "";
      const message = document.createElement("div");
      message.className = "meetings-loading";
      message.style.color = "var(--danger)";
      message.textContent = `Lỗi: ${error.message}`;
      listEl.appendChild(message);
    }
  }
}

async function renderLibrary() {
  await loadMeetings();
}

function searchMeetings() {
  clearTimeout(_searchTimeout);
  _searchTimeout = setTimeout(() => {
    _meetingsPage = 1;
    renderMeetingsPage();
  }, 300);
}

function renderMeetingsPage() {
  const listEl = $("meetings-list");
  if (!listEl) return;
  const items = filteredMeetings();
  const total = items.length;
  listEl.textContent = "";

  if (!total) {
    const empty = document.createElement("div");
    empty.className = "meetings-loading";
    empty.textContent = _allMeetings.length ? "Không có tập tin phù hợp" : "Chưa có tập tin nào";
    listEl.appendChild(empty);
    updateMeetingsToolbar();
    return;
  }

  const totalPages = Math.ceil(total / MEETINGS_PER_PAGE);
  if (_meetingsPage > totalPages) _meetingsPage = totalPages;
  if (_meetingsPage < 1) _meetingsPage = 1;
  const start = (_meetingsPage - 1) * MEETINGS_PER_PAGE;
  const page = items.slice(start, start + MEETINGS_PER_PAGE);

  const grid = document.createElement("div");
  grid.className = "meetings-grid";
  for (const item of page) {
    grid.appendChild(renderMeetingRow(item));
  }
  listEl.appendChild(grid);

  if (totalPages > 1) {
    const pager = document.createElement("div");
    pager.className = "mg-pager";
    const prev = document.createElement("button");
    prev.className = "btn btn-sm";
    prev.type = "button";
    prev.innerHTML = "&laquo;";
    prev.disabled = _meetingsPage <= 1;
    prev.addEventListener("click", meetingsPagePrev);
    const info = document.createElement("span");
    info.className = "mg-page-info";
    info.textContent = `Trang ${_meetingsPage}/${totalPages} (${total} tập tin)`;
    const next = document.createElement("button");
    next.className = "btn btn-sm";
    next.type = "button";
    next.innerHTML = "&raquo;";
    next.disabled = _meetingsPage >= totalPages;
    next.addEventListener("click", meetingsPageNext);
    pager.append(prev, info, next);
    listEl.appendChild(pager);
  }

  updateMeetingsToolbar();
}

function renderMeetingRow(item) {
  const status = libraryStatusMeta(item);
  const resumable = status.status === "processing" && Boolean(resumeAfterKillMeta(item)?.enabled);
  const canLoadSource = Boolean(item.sourceStored) && status.status !== "error" && item.status !== "importing";
  const clickable = status.status === "completed" || resumable || canLoadSource;
  const row = document.createElement("div");
  row.className = `mg-item${clickable ? " mg-clickable" : ""}`;
  row.dataset.id = item.id;
  if (clickable) {
    row.addEventListener("click", async () => {
      try {
        closeMeetingsPanel();
        setPipelineProgress(resumable ? "Đang nạp checkpoint" : "Đang nạp file", 1);
        await waitForUiPaint();
        if (resumable) {
          await processLibraryItem(item.id, { resume: true });
        } else {
          await openLibraryItem(item.id);
        }
        showToast(resumable ? `Đang tiếp tục xử lý: ${libraryDisplayName(item)}` : `Đã tải: ${libraryDisplayName(item)}`, "success");
      } catch (error) {
        showToast(`Lỗi: ${error.message}`, "error");
      }
    });
  }

  const checkWrap = document.createElement("div");
  checkWrap.className = "mg-check";
  checkWrap.addEventListener("click", (event) => event.stopPropagation());
  if (item.status !== "importing") {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "mg-cb";
    checkbox.value = item.id;
    checkbox.addEventListener("change", updateMeetingsToolbar);
    checkWrap.appendChild(checkbox);
  }
  row.appendChild(checkWrap);

  const info = document.createElement("div");
  info.className = "mg-info";
  const name = document.createElement("div");
  name.className = "mg-name";
  name.textContent = libraryDisplayName(item);
  name.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    startRenameMeeting(item.id, name);
  });
  info.appendChild(name);

  const meta = document.createElement("div");
  meta.className = "mg-meta";
  const file = document.createElement("span");
  file.className = "mg-file";
  file.title = item.sourceName || "";
  file.textContent = item.sourceName || "source.bin";
  meta.appendChild(file);
  for (const value of [
    formatBytes(item.sourceSize || 0),
    formatLibraryDate(item.createdAt || item.updatedAt),
  ]) {
    const span = document.createElement("span");
    span.textContent = value;
    meta.appendChild(span);
  }
  if (status.message) {
    const err = document.createElement("span");
    err.className = "mg-err";
    err.textContent = status.message;
    meta.appendChild(err);
  }
  info.appendChild(meta);
  row.appendChild(info);

  const badge = document.createElement("div");
  badge.className = `mg-status status-${status.status}`;
  badge.textContent = status.label;
  row.appendChild(badge);
  return row;
}

function meetingsPagePrev() {
  if (_meetingsPage > 1) {
    _meetingsPage -= 1;
    renderMeetingsPage();
  }
}

function meetingsPageNext() {
  const totalPages = Math.ceil(filteredMeetings().length / MEETINGS_PER_PAGE);
  if (_meetingsPage < totalPages) {
    _meetingsPage += 1;
    renderMeetingsPage();
  }
}

function updateMeetingsToolbar() {
  const bar = $("meetings-toolbar");
  const master = $("mg-select-all");
  if (!bar) return;
  const checkboxes = Array.from(document.querySelectorAll(".mg-cb"));
  bar.style.display = checkboxes.length ? "flex" : "none";
  if (!master) return;
  const checked = checkboxes.filter((checkbox) => checkbox.checked).length;
  master.checked = checkboxes.length > 0 && checked === checkboxes.length;
  master.indeterminate = checked > 0 && checked < checkboxes.length;
}

function toggleSelectAll() {
  const master = $("mg-select-all");
  document.querySelectorAll(".mg-cb").forEach((checkbox) => {
    checkbox.checked = Boolean(master?.checked);
  });
  updateMeetingsToolbar();
}

async function deleteSelectedMeetings() {
  const ids = Array.from(document.querySelectorAll(".mg-cb:checked")).map((checkbox) => checkbox.value);
  if (!ids.length) {
    showToast("Chưa chọn tập tin nào", "error");
    return;
  }
  if (!window.confirm(`Xóa ${ids.length} tập tin đã chọn? Không thể hoàn tác.`)) return;

  let ok = 0;
  let fail = 0;
  for (const id of ids) {
    try {
      await deleteLibraryItem(id, { quiet: true });
      ok += 1;
    } catch (error) {
      fail += 1;
      log(`Delete library item failed: ${error.message}`);
    }
  }
  showToast(`Đã xóa ${ok} tập tin${fail ? `, ${fail} lỗi` : ""}`, ok ? "success" : "error");
  if ($("mg-select-all")) $("mg-select-all").checked = false;
  await renderLibrary();
  await updateRuntimeStatus();
}

function startRenameMeeting(meetingId, el) {
  const currentName = el.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.value = currentName;
  input.className = "meeting-rename-input";
  const finish = async (commit) => {
    if (!commit) {
      el.textContent = currentName;
      return;
    }
    await saveRenameMeeting(meetingId, input.value.trim(), el, currentName);
  };
  input.addEventListener("click", (event) => event.stopPropagation());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true).catch((error) => showToast(`Lỗi đổi tên: ${error.message}`, "error"));
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false).catch(() => null);
    }
  });
  input.addEventListener("blur", () => {
    finish(true).catch((error) => showToast(`Lỗi đổi tên: ${error.message}`, "error"));
  });
  el.textContent = "";
  el.appendChild(input);
  input.focus();
  input.select();
}

async function saveRenameMeeting(meetingId, newName, el, oldName) {
  if (!newName || newName === oldName) {
    el.textContent = oldName;
    return;
  }
  try {
    await updateLibraryItem(meetingId, { meetingName: newName });
  } catch (error) {
    el.textContent = oldName;
    throw error;
  }
}

function modelFileName(file) {
  return `${file.id.replace(/[^a-zA-Z0-9_.-]/g, "_")}.bin`;
}

function expectedModelSha256(file) {
  const value = String(file?.sha256 || "").trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(value) ? value : "";
}

function modelIntegrityKey(file) {
  return `asr-vn-model-integrity:${modelFileName(file)}`;
}

function modelIntegrityIsRecorded(file, size) {
  const expected = expectedModelSha256(file);
  if (!expected) return true;
  try {
    const record = JSON.parse(localStorage.getItem(modelIntegrityKey(file)) || "{}");
    return record?.sha256 === expected && Number(record?.bytes || 0) === Number(size || 0);
  } catch (_) {
    return false;
  }
}

function markModelIntegrity(file, size, sha256) {
  const expected = expectedModelSha256(file);
  if (!expected || sha256 !== expected) return;
  try {
    localStorage.setItem(modelIntegrityKey(file), JSON.stringify({
      id: file.id,
      bytes: Number(size || 0),
      sha256,
      verifiedAt: new Date().toISOString(),
    }));
  } catch (_) {
    // Integrity is still enforced for the current download; this cache only avoids re-hashing on startup.
  }
}

function clearModelIntegrity(file) {
  try {
    localStorage.removeItem(modelIntegrityKey(file));
  } catch (_) {
    // ignore
  }
}

function clearAllModelIntegrityRecords() {
  try {
    for (let i = localStorage.length - 1; i >= 0; i -= 1) {
      const key = localStorage.key(i);
      if (key?.startsWith("asr-vn-model-integrity:")) localStorage.removeItem(key);
    }
  } catch (_) {
    // ignore
  }
}

function hexFromBytes(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function sha256HexFromBlob(blob) {
  if (!crypto?.subtle?.digest) {
    throw new Error("SHA-256 verification is not supported by this browser.");
  }
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return hexFromBytes(new Uint8Array(digest));
}

async function verifyModelBlobIntegrity(file, blob) {
  if (file.bytes && blob.size !== file.bytes) {
    throw new Error(`${file.id} size mismatch: got ${blob.size}, expected ${file.bytes}`);
  }
  const expected = expectedModelSha256(file);
  if (!expected) return { sha256: null, size: blob.size };
  const actual = await sha256HexFromBlob(blob);
  if (actual !== expected) {
    throw new Error(`${file.id} sha256 mismatch: got ${actual}, expected ${expected}`);
  }
  markModelIntegrity(file, blob.size, actual);
  return { sha256: actual, size: blob.size };
}

function configureOrt() {
  if (ortConfigured) return;
  if (!window.ort) {
    throw new Error("ONNX Runtime Web did not load.");
  }

  window.ort.env.wasm.wasmPaths = "/vendor/onnxruntime-web/";
  window.ort.env.wasm.numThreads = getRequestedThreads();
  window.ort.env.wasm.proxy = false;
  window.ort.env.logLevel = "error";
  ortConfigured = true;
}

function isWebGpuRuntimeAvailable() {
  return Boolean(navigator.gpu && window.ort?.InferenceSession);
}

async function createOrtSession(model, options = {}) {
  configureOrt();
  const name = options.name || "ONNX model";
  const calibratedProvider = !benchmarkProviderMode
    ? calibratedProviderForStage(name, null)
    : null;
  const webgpuRequested = options.webgpuPreferred &&
    (!options.webgpuBenchmarkOnly || benchmarkProviderMode === "webgpu" || calibratedProvider === "webgpu");
  const webgpuAllowed = webgpuRequested && benchmarkProviderMode !== "wasm";
  const preferred = calibratedProvider === "wasm"
    ? ["wasm"]
    : (webgpuAllowed && isWebGpuRuntimeAvailable()
    ? ["webgpu", "wasm"]
    : ["wasm"]);
  const sessionOptions = {
    executionProviders: preferred,
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
    ...(options.sessionOptions || {}),
  };
  try {
    const session = await window.ort.InferenceSession.create(model, sessionOptions);
    return { session, provider: preferred[0] };
  } catch (error) {
    if (preferred[0] !== "webgpu") throw error;
    if (benchmarkProviderMode === "webgpu" || options.strictWebgpu) {
      throw new Error(`${name} WebGPU init failed in benchmark mode: ${error.message}`);
    }
    log(`${name} WebGPU init failed; falling back to WASM: ${error.message}`);
    const session = await window.ort.InferenceSession.create(model, {
      ...sessionOptions,
      executionProviders: ["wasm"],
    });
    return { session, provider: "wasm" };
  }
}

function refreshOrtThreadConfigIfIdle() {
  if (vadSession || punctuationSession || diarizationSession || camppSession || overlapSession || pyannoteEmbeddingSession || dnsmosSession) return;
  ortConfigured = false;
  configureOrt();
}

function findManifestFile(fileId) {
  for (const pack of manifest?.packs || []) {
    for (const file of pack.files || []) {
      if (file.id === fileId) return file;
    }
  }
  throw new Error(`Model id not found in manifest: ${fileId}`);
}

async function ensureModelFile(fileId) {
  const file = findManifestFile(fileId);
  const status = await fileStatus(file);
  if (status.ready) return file;
  if (file.available_local === false) {
    throw new Error(`Required model is not bundled on server: ${file.id} (${file.server_status || "missing on server"}).`);
  }

  log(`Required model is missing; downloading ${file.id}.`);
  if (!navigator.onLine) {
    throw new Error(`Required model is missing locally while offline: ${file.id}. Reconnect once to finish the offline model pack.`);
  }
  const response = await fetchWithTimeout(
    file.download_url,
    { cache: "no-store" },
    DOWNLOAD_RESPONSE_TIMEOUT_MS
  );
  if (!response.ok || !response.body) {
    throw new Error(`download failed for ${file.id}: ${response.status}`);
  }
  await writeModelFile(
    file,
    response,
    null,
    0,
    file.bytes || Number(response.headers.get("Content-Length")) || 0
  );
  await renderPacks();
  await updateRuntimeStatus();
  return file;
}

async function readStoredModelArrayBuffer(file) {
  const dir = await opfsModelDir();
  if (dir.kind === "idb") {
    const stored = await readFallbackModelFile(file);
    return stored.arrayBuffer();
  }
  const handle = await dir.getFileHandle(modelFileName(file));
  const stored = await handle.getFile();
  return stored.arrayBuffer();
}

async function loadModelArrayBuffer(fileId) {
  const file = await ensureModelFile(fileId);
  return readStoredModelArrayBuffer(file);
}

async function readStoredModelText(file) {
  const dir = await opfsModelDir();
  if (dir.kind === "idb") {
    const stored = await readFallbackModelFile(file);
    return stored.text();
  }
  const handle = await dir.getFileHandle(modelFileName(file));
  const stored = await handle.getFile();
  return stored.text();
}

async function loadModelText(fileId) {
  const file = await ensureModelFile(fileId);
  return readStoredModelText(file);
}

async function readFallbackModelFile(file) {
  const name = modelFileName(file);
  try {
    return await readCachedModelFile(name);
  } catch (_) {
    return idbGetFile("models", name);
  }
}

function parseNpyArray(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 16 || bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error("Invalid NPY file.");
  }
  const major = bytes[6];
  let headerLength = 0;
  let headerOffset = 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (major === 1) {
    headerLength = view.getUint16(8, true);
    headerOffset = 10;
  } else if (major === 2 || major === 3) {
    headerLength = view.getUint32(8, true);
    headerOffset = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.`);
  }

  const header = new TextDecoder("utf-8").decode(bytes.slice(headerOffset, headerOffset + headerLength));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === "True";
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] || "";
  if (fortran) throw new Error("Fortran-order NPY arrays are not supported in offline PWA.");
  const shape = shapeText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item));
  const count = shape.reduce((prod, value) => prod * value, 1);
  const dataOffset = headerOffset + headerLength;
  let data = null;
  if (descr === "<f4" || descr === "|f4") {
    data = new Float32Array(bytes.buffer.slice(bytes.byteOffset + dataOffset, bytes.byteOffset + dataOffset + count * 4));
  } else if (descr === "<f8" || descr === "|f8") {
    data = new Float64Array(bytes.buffer.slice(bytes.byteOffset + dataOffset, bytes.byteOffset + dataOffset + count * 8));
  } else {
    throw new Error(`Unsupported NPY dtype ${descr}.`);
  }
  return { data, shape, dtype: descr };
}

function parseStoredNpz(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const arrays = new Map();
  let offset = 0;
  const readZip64Size = (extra, compressed, uncompressed) => {
    let pos = 0;
    let comp = compressed;
    let uncomp = uncompressed;
    while (pos + 4 <= extra.length) {
      const headerId = extra[pos] | (extra[pos + 1] << 8);
      const size = extra[pos + 2] | (extra[pos + 3] << 8);
      pos += 4;
      if (pos + size > extra.length) break;
      if (headerId === 0x0001) {
        let zp = pos;
        const readU64 = () => {
          const lo = BigInt(extra[zp] | (extra[zp + 1] << 8) | (extra[zp + 2] << 16) | (extra[zp + 3] << 24));
          const hi = BigInt(extra[zp + 4] | (extra[zp + 5] << 8) | (extra[zp + 6] << 16) | (extra[zp + 7] << 24));
          zp += 8;
          const value = (hi << 32n) + (lo & 0xffffffffn);
          if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("NPZ ZIP64 entry is too large.");
          return Number(value);
        };
        if (uncomp === 0xffffffff && zp + 8 <= pos + size) uncomp = readU64();
        if (comp === 0xffffffff && zp + 8 <= pos + size) comp = readU64();
        break;
      }
      pos += size;
    }
    return { compressedSize: comp, uncompressedSize: uncomp };
  };
  while (offset + 30 <= bytes.length) {
    const signature = view.getUint32(offset, true);
    if (signature !== 0x04034b50) break;
    const flags = view.getUint16(offset + 6, true);
    const method = view.getUint16(offset + 8, true);
    let compressedSize = view.getUint32(offset + 18, true);
    let uncompressedSize = view.getUint32(offset + 22, true);
    const fileNameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    if (flags & 0x08) {
      throw new Error("NPZ files with data descriptors are not supported.");
    }
    if (method !== 0) {
      throw new Error(`NPZ entry compression method ${method} is not supported.`);
    }
    const nameStart = offset + 30;
    const name = new TextDecoder("utf-8").decode(bytes.slice(nameStart, nameStart + fileNameLength));
    const extra = bytes.slice(nameStart + fileNameLength, nameStart + fileNameLength + extraLength);
    if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff) {
      const sizes = readZip64Size(extra, compressedSize, uncompressedSize);
      compressedSize = sizes.compressedSize;
      uncompressedSize = sizes.uncompressedSize;
    }
    const dataStart = nameStart + fileNameLength + extraLength;
    const dataEnd = dataStart + compressedSize;
    if (dataEnd > bytes.length) throw new Error("Invalid NPZ entry size.");
    const key = name.endsWith(".npy") ? name.slice(0, -4) : name;
    arrays.set(key, parseNpyArray(bytes.buffer.slice(bytes.byteOffset + dataStart, bytes.byteOffset + dataEnd)));
    offset = dataEnd;
  }
  if (!arrays.size) throw new Error("NPZ archive has no readable arrays.");
  return arrays;
}

async function loadNpyArray(fileId) {
  return parseNpyArray(await loadModelArrayBuffer(fileId));
}

async function loadNpzArrays(fileId) {
  return parseStoredNpz(await loadModelArrayBuffer(fileId));
}

function setAsrStatus(text) {
  const node = $("asr-runtime");
  if (node) node.textContent = text;
}

function asrWorkerScriptForModel(modelConfig = null) {
  const backend = modelConfig?.backend || "pure_ort";
  const script = ASR_BACKEND_WORKERS[backend];
  if (!script) {
    throw new Error(`ASR backend is not supported in offline PWA: ${backend}`);
  }
  return script;
}

function getAsrWorker(modelConfig = null) {
  if (asrWorker && !modelConfig) return asrWorker;
  const script = asrWorkerScriptForModel(modelConfig);
  if (asrWorker && asrWorkerScript === script) return asrWorker;
  if (asrWorker && asrWorkerScript !== script) {
    resetAsrWorker("ASR backend changed");
  }

  asrWorker = new Worker(script);
  asrWorkerScript = script;
  asrWorker.addEventListener("message", (event) => {
    const message = event.data || {};
    if (message.type === "status") {
      setAsrStatus(message.status);
      return;
    }
    if (message.type === "runtime-ready") {
      setAsrStatus("runtime ready");
      return;
    }
    if (message.type === "log") {
      log(`ASR worker: ${message.message}`);
      return;
    }

    if (!message.id || !asrRequests.has(message.id)) return;
    const pending = asrRequests.get(message.id);
    asrRequests.delete(message.id);

    if (message.type === "error") {
      pending.reject(new Error(message.message || "ASR worker error"));
      return;
    }
    pending.resolve(message);
  });

  asrWorker.addEventListener("error", (event) => {
    for (const pending of asrRequests.values()) {
      pending.reject(new Error(event.message || "ASR worker crashed"));
    }
    asrRequests.clear();
    setAsrStatus("worker error");
  });

  return asrWorker;
}

function resetAsrWorker(status = "idle") {
  if (asrWorker) {
    asrWorker.terminate();
    asrWorker = null;
  }
  asrWorkerScript = null;
  for (const pending of asrRequests.values()) {
    pending.reject(new Error("ASR worker was reset."));
  }
  asrRequests.clear();
  asrInitPromise = null;
  asrLoadedModelId = null;
  asrLoadedConfigKey = null;
  setAsrStatus(status);
}

function asrRuntimeConfigKey(modelConfig, options = {}) {
  const hotwordsText = String(options.hotwordsText || "");
  const hotwordsScore = boundedFloat(options.hotwordsScore, DEFAULT_HOTWORDS_SCORE, 0, 8).toFixed(2);
  const cpuThreads = boundedNumber(options.cpuThreads, getRequestedThreads(), 1, 8);
  const providerMode = options.providerMode === "webgpu" ? "webgpu" : "wasm";
  return [
    modelConfig.id,
    modelConfig.backend || "pure_ort",
    cpuThreads,
    providerMode,
    hotwordsScore,
    hotwordsText ? hashString(hotwordsText) : "no-hotwords",
  ].join("|");
}

async function loadAsrModelIntoWorker(modelConfig, options = {}) {
  const modelFiles = modelConfig.files;
  const [encoder, decoder, joiner, tokens, bpeVocab] = await Promise.all([
    loadModelArrayBuffer(modelFiles.encoder),
    loadModelArrayBuffer(modelFiles.decoder),
    loadModelArrayBuffer(modelFiles.joiner),
    loadModelArrayBuffer(modelFiles.tokens),
    modelFiles.bpeVocab ? loadModelText(modelFiles.bpeVocab) : Promise.resolve(""),
  ]);
  await callAsrWorker(
    "init",
    {
      files: { encoder, decoder, joiner, tokens, bpeVocab },
      modelId: modelConfig.id,
      modelLabel: modelConfig.label,
      backend: modelConfig.backend || "pure_ort",
      decodingMethod: modelConfig.decodingMethod,
      maxActivePaths: modelConfig.maxActivePaths,
      numThreads: options.cpuThreads || getRequestedThreads(),
      hotwordsText: options.hotwordsText || "",
      hotwordsScore: boundedFloat(options.hotwordsScore, DEFAULT_HOTWORDS_SCORE, 0, 8),
      providerMode: options.providerMode || "wasm",
      stageProviders: options.stageProviders || null,
    },
    [encoder, decoder, joiner, tokens],
    modelConfig
  );
}

function createAsrWorkerHandle(label = "ASR worker", modelConfig = null) {
  const worker = new Worker(asrWorkerScriptForModel(modelConfig));
  const requests = new Map();
  let requestId = 0;
  let closed = false;

  worker.addEventListener("message", (event) => {
    const message = event.data || {};
    if (message.type === "status") {
      setAsrStatus(`${label}: ${message.status}`);
      return;
    }
    if (message.type === "runtime-ready") {
      setAsrStatus(`${label}: runtime ready`);
      return;
    }
    if (message.type === "log") {
      log(`${label}: ${message.message}`);
      return;
    }

    if (!message.id || !requests.has(message.id)) return;
    const pending = requests.get(message.id);
    requests.delete(message.id);

    if (message.type === "error") {
      pending.reject(new Error(message.message || `${label} error`));
      return;
    }
    pending.resolve(message);
  });

  worker.addEventListener("error", (event) => {
    for (const pending of requests.values()) {
      pending.reject(new Error(event.message || `${label} crashed`));
    }
    requests.clear();
  });

  return {
    call(type, payload = {}, transfer = []) {
      if (closed) return Promise.reject(new Error(`${label} is closed.`));
      const id = ++requestId;
      const promise = new Promise((resolve, reject) => {
        requests.set(id, { resolve, reject });
      });
      worker.postMessage({ id, type, ...payload }, transfer);
      return promise;
    },
    terminate() {
      closed = true;
      worker.terminate();
      for (const pending of requests.values()) {
        pending.reject(new Error(`${label} was terminated.`));
      }
      requests.clear();
    },
  };
}

async function loadAsrModelIntoWorkerHandle(handle, modelConfig, options = {}) {
  const modelFiles = modelConfig.files;
  const [encoder, decoder, joiner, tokens, bpeVocab] = await Promise.all([
    loadModelArrayBuffer(modelFiles.encoder),
    loadModelArrayBuffer(modelFiles.decoder),
    loadModelArrayBuffer(modelFiles.joiner),
    loadModelArrayBuffer(modelFiles.tokens),
    modelFiles.bpeVocab ? loadModelText(modelFiles.bpeVocab) : Promise.resolve(""),
  ]);
  await handle.call(
    "init",
    {
      files: { encoder, decoder, joiner, tokens, bpeVocab },
      modelId: modelConfig.id,
      modelLabel: modelConfig.label,
      backend: modelConfig.backend || "pure_ort",
      decodingMethod: modelConfig.decodingMethod,
      maxActivePaths: modelConfig.maxActivePaths,
      numThreads: options.cpuThreads || getRequestedThreads(),
      hotwordsText: options.hotwordsText || "",
      hotwordsScore: boundedFloat(options.hotwordsScore, DEFAULT_HOTWORDS_SCORE, 0, 8),
      providerMode: options.providerMode || "wasm",
      stageProviders: options.stageProviders || null,
    },
    [encoder, decoder, joiner, tokens]
  );
}

function callAsrWorker(type, payload = {}, transfer = [], modelConfig = null) {
  const worker = getAsrWorker(modelConfig);
  const id = ++asrRequestId;
  const promise = new Promise((resolve, reject) => {
    asrRequests.set(id, { resolve, reject });
  });
  worker.postMessage({ id, type, ...payload }, transfer);
  return promise;
}

async function ensureAsrReady(modelConfig = getSelectedAsrModel(), options = {}) {
  const configKey = asrRuntimeConfigKey(modelConfig, options);
  if (asrLoadedConfigKey && asrLoadedConfigKey !== configKey) {
    resetAsrWorker("ASR config changed");
  }
  if (asrInitPromise) return asrInitPromise;

  asrInitPromise = (async () => {
    setAsrStatus(`loading ${modelConfig.label}`);
    if (modelConfig.type === "rover") {
      for (const childId of modelConfig.modelIds || []) {
        const child = ASR_MODELS[childId];
        if (!child || child.type !== "single") {
          throw new Error(`ROVER child model is not supported in offline PWA: ${childId}`);
        }
        setAsrStatus(`loading ${child.label}`);
        await loadAsrModelIntoWorker({ id: childId, ...child }, options);
      }
    } else {
      setAsrStatus("initializing");
      await loadAsrModelIntoWorker(modelConfig, options);
    }
    asrLoadedModelId = modelConfig.id;
    asrLoadedConfigKey = configKey;
    setAsrStatus("ready");
  })().catch((error) => {
    asrInitPromise = null;
    asrLoadedModelId = null;
    asrLoadedConfigKey = null;
    setAsrStatus("error");
    throw error;
  });

  return asrInitPromise;
}

function renderTranscript(text, details) {
  // Internal pipeline preview intentionally hidden. The user-facing surface is
  // the server-style result panel only.
}

function normalizeSpeechSegments(samples, segments) {
  const source = Array.isArray(segments) ? segments : [];
  const clipped = source
    .map((segment) => ({
      start: Math.max(0, Math.min(samples.length, Math.floor(segment.start || 0))),
      end: Math.max(0, Math.min(samples.length, Math.ceil(segment.end || 0))),
    }))
    .filter((segment) => segment.end > segment.start)
    .sort((a, b) => a.start - b.start);

  const merged = [];
  for (const segment of clipped) {
    const previous = merged[merged.length - 1];
    if (previous && segment.start <= previous.end) {
      previous.end = Math.max(previous.end, segment.end);
    } else {
      merged.push({ ...segment });
    }
  }
  return merged;
}

function mergeSpeechSegmentsForAsr(samples, segments) {
  const normalized = normalizeSpeechSegments(samples, segments);
  const maxGapSamples = ASR_MERGE_GAP_SECONDS * VAD_SAMPLE_RATE;
  if (normalized.length <= 1) return normalized;

  const merged = [normalized[0]];
  for (const segment of normalized.slice(1)) {
    const previous = merged[merged.length - 1];
    if (segment.start - previous.end <= maxGapSamples) {
      previous.end = segment.end;
    } else {
      merged.push({ ...segment });
    }
  }
  return merged;
}

function buildSpeechTimeline(samples, segments) {
  const speechSegments = mergeSpeechSegmentsForAsr(samples, segments);
  const entries = [];
  let timelineOffset = 0;

  for (const segment of speechSegments) {
    const length = segment.end - segment.start;
    entries.push({
      sourceStart: segment.start,
      sourceEnd: segment.end,
      timelineStart: timelineOffset,
      timelineEnd: timelineOffset + length,
    });
    timelineOffset += length;
  }

  return { entries, totalSamples: timelineOffset, sourceSegments: speechSegments };
}

function copyTimelineRange(samples, timeline, start, end) {
  const length = Math.max(0, end - start);
  const output = new Float32Array(length);
  let outputOffset = 0;

  for (const entry of timeline.entries) {
    const overlapStart = Math.max(start, entry.timelineStart);
    const overlapEnd = Math.min(end, entry.timelineEnd);
    if (overlapEnd <= overlapStart) continue;

    const sourceStart = entry.sourceStart + (overlapStart - entry.timelineStart);
    const sourceEnd = sourceStart + (overlapEnd - overlapStart);
    output.set(samples.subarray(sourceStart, sourceEnd), outputOffset);
    outputOffset += sourceEnd - sourceStart;
  }
  return output;
}

function timelineSampleToSourceSeconds(timeline, sample) {
  if (!timeline?.entries?.length) {
    return Math.max(0, sample) / VAD_SAMPLE_RATE;
  }

  const clamped = Math.max(0, Math.min(sample, timeline.totalSamples || sample));
  for (const entry of timeline.entries) {
    if (clamped >= entry.timelineStart && clamped <= entry.timelineEnd) {
      const sourceSample = entry.sourceStart + Math.max(0, clamped - entry.timelineStart);
      return sourceSample / VAD_SAMPLE_RATE;
    }
    if (clamped < entry.timelineStart) {
      return entry.sourceStart / VAD_SAMPLE_RATE;
    }
  }

  const last = timeline.entries[timeline.entries.length - 1];
  return last.sourceEnd / VAD_SAMPLE_RATE;
}

function timelineRangeToSourceSeconds(timeline, start, end) {
  const startSec = timelineSampleToSourceSeconds(timeline, start);
  const endSec = timelineSampleToSourceSeconds(timeline, end);
  if (endSec > startSec) return { start: startSec, end: endSec };
  return {
    start: startSec,
    end: startSec + Math.max(0.01, (end - start) / VAD_SAMPLE_RATE),
  };
}

function findAsrSilentRegions(audioData) {
  const frameLength = Math.max(1, Math.floor(VAD_SAMPLE_RATE * 0.01));
  const numFrames = Math.floor((audioData?.length || 0) / frameLength);
  if (!numFrames) return [];

  const isSilent = new Uint8Array(numFrames);
  for (let frame = 0; frame < numFrames; frame += 1) {
    const offset = frame * frameLength;
    let sumSquares = 0;
    for (let i = 0; i < frameLength; i += 1) {
      const value = audioData[offset + i] || 0;
      sumSquares += value * value;
    }
    const rms = Math.sqrt(sumSquares / frameLength);
    isSilent[frame] = rms < ASR_SILENCE_THRESHOLD ? 1 : 0;
  }

  const minFrames = Math.floor(ASR_MIN_SILENCE_SECONDS / 0.01);
  const silentRegions = [];
  let silentStart = -1;
  for (let frame = 0; frame < numFrames; frame += 1) {
    if (isSilent[frame]) {
      if (silentStart < 0) silentStart = frame;
      continue;
    }
    if (silentStart >= 0) {
      if (frame - silentStart >= minFrames) {
        silentRegions.push([
          silentStart * frameLength,
          Math.min(frame * frameLength, audioData.length),
        ]);
      }
      silentStart = -1;
    }
  }

  if (silentStart >= 0 && numFrames - silentStart >= minFrames) {
    silentRegions.push([
      silentStart * frameLength,
      Math.min(numFrames * frameLength, audioData.length),
    ]);
  }

  return silentRegions;
}

function findBestAsrSplitPoint(targetSample, totalSamples, silentRegions) {
  const searchWindow = ASR_SPLIT_SEARCH_SECONDS * VAD_SAMPLE_RATE;
  const searchStart = Math.max(0, targetSample - searchWindow);
  const searchEnd = Math.min(totalSamples, targetSample + searchWindow);
  let bestPoint = targetSample;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const [silentStart, silentEnd] of silentRegions || []) {
    if (silentEnd >= searchStart && silentStart <= searchEnd) {
      const midSilent = Math.floor((silentStart + silentEnd) / 2);
      const distance = Math.abs(midSilent - targetSample);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestPoint = midSilent;
      }
    }
  }

  return bestPoint;
}

function makeAsrChunks(timeline, concatSamples = null) {
  const chunkSamples = ASR_CHUNK_SECONDS * VAD_SAMPLE_RATE;
  const overlapSamples = ASR_CHUNK_OVERLAP_SECONDS * VAD_SAMPLE_RATE;
  const minSplitAdvanceSamples = ASR_MIN_SPLIT_ADVANCE_SECONDS * VAD_SAMPLE_RATE;
  const chunks = [];
  const totalSamples = Math.max(0, timeline?.totalSamples || 0);
  if (!totalSamples) return chunks;

  const silentRegions = concatSamples ? findAsrSilentRegions(concatSamples) : [];
  const boundaries = [0];
  let current = 0;
  while (current + chunkSamples < totalSamples) {
    const target = current + chunkSamples;
    let split = concatSamples
      ? findBestAsrSplitPoint(target, totalSamples, silentRegions)
      : target;
    if (split <= current + minSplitAdvanceSamples) {
      split = target;
    }
    split = Math.max(current + 1, Math.min(split, totalSamples));
    boundaries.push(split);
    current = split;
  }
  if (boundaries[boundaries.length - 1] !== totalSamples) {
    boundaries.push(totalSamples);
  }

  for (let i = 0; i < boundaries.length - 1; i += 1) {
    const logicalStart = boundaries[i];
    const logicalEnd = boundaries[i + 1];
    if (logicalEnd <= logicalStart) continue;
    if (logicalEnd - logicalStart < ASR_MIN_CHUNK_SECONDS * VAD_SAMPLE_RATE && chunks.length) {
      chunks[chunks.length - 1].end = logicalEnd;
      chunks[chunks.length - 1].logicalEnd = logicalEnd;
      break;
    }
    const actualStart = i === 0 ? logicalStart : Math.max(0, logicalStart - overlapSamples);
    chunks.push({
      index: chunks.length + 1,
      start: actualStart,
      end: logicalEnd,
      logicalStart,
      logicalEnd,
      overlapAtStart: logicalStart - actualStart,
    });
  }

  chunks.silentRegionCount = silentRegions.length;
  chunks.chunkingMode = concatSamples ? "desktop_silence" : "fixed_overlap";
  return chunks;
}

function normalizeOverlapToken(token) {
  return token
    .toLocaleLowerCase("vi-VN")
    .replace(/[.,!?;:"'()[\]{}]/g, "")
    .trim();
}

function appendWithOverlap(existing, next) {
  const left = (existing || "").trim();
  const right = (next || "").trim();
  if (!left) return right;
  if (!right) return left;

  const leftWords = left.split(/\s+/);
  const rightWords = right.split(/\s+/);
  const maxOverlap = Math.min(24, leftWords.length, rightWords.length);

  for (let size = maxOverlap; size > 0; size -= 1) {
    const leftTail = leftWords.slice(leftWords.length - size).map(normalizeOverlapToken).join(" ");
    const rightHead = rightWords.slice(0, size).map(normalizeOverlapToken).join(" ");
    if (leftTail && leftTail === rightHead) {
      return `${left} ${rightWords.slice(size).join(" ")}`.trim();
    }
  }

  return `${left} ${right}`.trim();
}

function normalizeWordForAsrOverlap(word) {
  return String(word || "")
    .toLocaleLowerCase("vi-VN")
    .normalize("NFC")
    .replace(/[^\p{L}\p{N}_]/gu, "")
    .trim();
}

function levenshteinRatio(a, b) {
  if (a === b) return 1;
  if (!a || !b) return 0;
  const previous = new Uint16Array(b.length + 1);
  const current = new Uint16Array(b.length + 1);
  for (let j = 0; j <= b.length; j += 1) previous[j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + cost
      );
    }
    previous.set(current);
  }
  return 1 - previous[b.length] / Math.max(a.length, b.length);
}

function asrWordsMatch(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.length > 2 && b.length > 2 && (a.includes(b) || b.includes(a))) return true;
  return levenshteinRatio(a, b) >= 0.8;
}

function findAsrOverlapAlignment(tailWords, headWords) {
  if (!tailWords.length || !headWords.length) {
    return { cutIndex: 0, popCount: 0 };
  }

  const originalTailLen = tailWords.length;
  const tail = tailWords.slice(-100);
  const head = headWords.slice(0, 100);
  const tailNorm = tail.map((word) => normalizeWordForAsrOverlap(word.text));
  const headNorm = head.map((word) => normalizeWordForAsrOverlap(word.text));
  let bestScore = 0;
  let bestCutIndex = 0;
  let bestPopCount = 0;
  const minOffset = -tailNorm.length + 1;
  const maxOffset = headNorm.length;

  for (let offset = minOffset; offset < maxOffset; offset += 1) {
    let score = 0;
    const matchedTail = [];
    const matchedHead = [];
    for (let i = 0; i < tailNorm.length; i += 1) {
      const headIndex = i + offset;
      if (headIndex >= 0 && headIndex < headNorm.length && asrWordsMatch(tailNorm[i], headNorm[headIndex])) {
        score += 1;
        matchedTail.push(i);
        matchedHead.push(headIndex);
      }
    }
    const overlapWindow = Math.min(headNorm.length, tailNorm.length + offset) - Math.max(0, offset);
    const matchRatio = score / Math.max(1, overlapWindow);
    if (score > bestScore && matchRatio >= 0.5) {
      bestScore = score;
      bestCutIndex = matchedHead[matchedHead.length - 1] + 1;
      bestPopCount = tailNorm.length - 1 - matchedTail[matchedTail.length - 1];
    }
  }

  const minLen = Math.min(tailNorm.length, headNorm.length);
  const diverged = bestScore < minLen && bestPopCount > 0;
  if (bestScore === 0 || diverged) {
    const divTail = bestScore === 0 ? tailWords : tailWords.slice(-bestPopCount);
    const divHead = bestScore === 0 ? headWords : headWords.slice(bestCutIndex);
    const tailProb = divTail.reduce((sum, word) => sum + (word.prob ?? 1), 0) / Math.max(1, divTail.length);
    const headProb = divHead.reduce((sum, word) => sum + (word.prob ?? 1), 0) / Math.max(1, divHead.length);
    if (tailProb > headProb) {
      return { cutIndex: headWords.length, popCount: 0 };
    }
    return { cutIndex: 0, popCount: originalTailLen };
  }

  return { cutIndex: bestCutIndex, popCount: bestPopCount };
}

function mergeChunkWordsWithOverlap(results) {
  const ordered = (results || []).filter(Boolean);
  if (!ordered.length || !ordered.every((item) => Array.isArray(item.words))) return null;
  const mergedWords = [];
  let previousWordChunk = null;

  for (let i = 0; i < ordered.length; i += 1) {
    const chunkWords = ordered[i].words;
    if (!chunkWords.length) continue;
    if (!previousWordChunk) {
      mergedWords.push(...chunkWords);
      previousWordChunk = ordered[i];
      continue;
    }

    const previousDuration = ((previousWordChunk.end || 0) - (previousWordChunk.start || 0)) / VAD_SAMPLE_RATE;
    const overlapStartLocal = Math.max(0, previousDuration - ASR_CHUNK_OVERLAP_SECONDS);
    const tailWords = (previousWordChunk.words || []).filter((word) => (word.local_start || 0) >= overlapStartLocal);
    const headWords = chunkWords.filter((word) => (word.local_start || 0) < ASR_CHUNK_OVERLAP_SECONDS);
    const { cutIndex, popCount } = findAsrOverlapAlignment(tailWords, headWords);
    if (popCount > 0) mergedWords.splice(Math.max(0, mergedWords.length - popCount), popCount);
    if (cutIndex < chunkWords.length) mergedWords.push(...chunkWords.slice(cutIndex));
    previousWordChunk = ordered[i];
  }

  if (!mergedWords.length) return null;

  return {
    words: mergedWords,
    text: mergedWords.map((word) => word.text).join(" ").trim(),
  };
}

function mergeChunkTexts(results) {
  const mergedWords = mergeChunkWordsWithOverlap(results);
  if (mergedWords) return mergedWords.text;
  let text = "";
  for (const item of results) {
    text = appendWithOverlap(text, item.text || "");
  }
  return text.trim();
}

function removeAsrFillerWordsFromText(text) {
  const words = normalizeAsrText(text).split(/\s+/).filter(Boolean);
  const kept = [];
  let removed = 0;
  for (const word of words) {
    if (ASR_FILLER_WORDS.has(normalizeOverlapToken(word))) {
      removed += 1;
    } else {
      kept.push(word);
    }
  }
  return { text: kept.join(" "), removed, original: words.length };
}

function removeAsrFillerWords(words = []) {
  const kept = [];
  let removed = 0;
  for (const word of words || []) {
    if (ASR_FILLER_WORDS.has(normalizeOverlapToken(word.text))) {
      removed += 1;
    } else {
      kept.push(word);
    }
  }
  return { words: kept, removed, original: words?.length || 0 };
}

function asrConfidenceFromWords(words = []) {
  const probs = [];
  for (const word of words || []) {
    const prob = Number(word?.prob);
    if (Number.isFinite(prob) && prob >= 0 && prob <= 1) probs.push(prob);
  }
  if (!probs.length) return null;
  return probs.reduce((sum, value) => sum + value, 0) / probs.length;
}

function collectAsrConfidenceWords(results = []) {
  const words = [];
  for (const item of results || []) {
    if (Array.isArray(item?.words)) words.push(...item.words);
    const primary = item?.result?.primary?.words;
    const secondary = item?.result?.secondary?.words;
    if (Array.isArray(primary)) words.push(...primary);
    if (Array.isArray(secondary)) words.push(...secondary);
  }
  return words;
}

function mapAsrWordsToSourceTimeline(words = [], timeline = null) {
  if (!Array.isArray(words)) return [];
  return words.map((word) => {
    const start = Number(word.start);
    const end = Number(word.end);
    const sourceStart = timeline
      ? timelineSampleToSourceSeconds(timeline, Math.round(Math.max(0, start) * VAD_SAMPLE_RATE))
      : start;
    const sourceEnd = timeline
      ? timelineSampleToSourceSeconds(timeline, Math.round(Math.max(start, end) * VAD_SAMPLE_RATE))
      : end;
    return {
      ...word,
      timeline_start: start,
      timeline_end: end,
      start: sourceStart,
      end: Math.max(sourceStart + 0.01, sourceEnd),
    };
  });
}

function countEnergyPeaks(audioSegment, sampleRate = VAD_SAMPLE_RATE) {
  const frameLen = Math.max(1, Math.floor(sampleRate * 0.010));
  const hopLen = Math.max(1, Math.floor(sampleRate * 0.005));
  if (!audioSegment?.length || audioSegment.length < frameLen) return [];
  const numFrames = Math.max(1, Math.floor((audioSegment.length - frameLen) / hopLen) + 1);
  const energy = new Float32Array(numFrames);
  let maxEnergy = 0;
  for (let i = 0; i < numFrames; i += 1) {
    const start = i * hopLen;
    let sum = 0;
    for (let j = 0; j < frameLen && start + j < audioSegment.length; j += 1) {
      const value = audioSegment[start + j];
      sum += value * value;
    }
    const rms = Math.sqrt(sum / frameLen);
    energy[i] = rms;
    if (rms > maxEnergy) maxEnergy = rms;
  }
  if (maxEnergy <= 0) return [];

  const kernel = [0, 0.25, 0.5, 0.25, 0];
  const smooth = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i += 1) {
    let value = 0;
    let weight = 0;
    for (let k = 0; k < kernel.length; k += 1) {
      const idx = i + k - 2;
      if (idx >= 0 && idx < numFrames) {
        value += energy[idx] * kernel[k];
        weight += kernel[k];
      }
    }
    smooth[i] = weight ? value / weight : energy[i];
  }

  const nonSilence = [];
  const silenceFloor = maxEnergy * 0.05;
  for (const value of smooth) {
    if (value > silenceFloor) nonSilence.push(value);
  }
  if (!nonSilence.length) return [];
  const threshold = nonSilence.reduce((sum, value) => sum + value, 0) / nonSilence.length;
  const minDistance = Math.max(1, Math.floor(90 / ((hopLen / sampleRate) * 1000)));
  const peaks = [];
  let lastPeak = -minDistance;
  for (let i = 1; i < smooth.length - 1; i += 1) {
    if (i - lastPeak < minDistance) continue;
    const value = smooth[i];
    if (value < threshold || value < smooth[i - 1] || value < smooth[i + 1]) continue;
    const left = Math.max(0, i - minDistance);
    const right = Math.min(smooth.length - 1, i + minDistance);
    const localMin = Math.min(smooth[left], smooth[right]);
    if (value - localMin < threshold * 0.3) continue;
    peaks.push((i * hopLen) / sampleRate);
    lastPeak = i;
  }
  return peaks;
}

function computeGapEnergyRange(audioSegment, sampleRate = VAD_SAMPLE_RATE) {
  const frameLen = Math.max(1, Math.floor(sampleRate * 0.010));
  const hopLen = Math.max(1, Math.floor(sampleRate * 0.005));
  if (!audioSegment?.length || audioSegment.length < 50) return 0;
  const numFrames = Math.max(1, Math.floor((audioSegment.length - frameLen) / hopLen) + 1);
  let minEnergy = Infinity;
  let maxEnergy = 0;
  for (let i = 0; i < numFrames; i += 1) {
    const start = i * hopLen;
    let sum = 0;
    let count = 0;
    for (let j = 0; j < frameLen && start + j < audioSegment.length; j += 1) {
      const value = audioSegment[start + j];
      sum += value * value;
      count += 1;
    }
    const rms = Math.sqrt(sum / Math.max(1, count));
    minEnergy = Math.min(minEnergy, rms);
    maxEnergy = Math.max(maxEnergy, rms);
  }
  return Number.isFinite(minEnergy) ? maxEnergy - minEnergy : 0;
}

function maxVadProbabilityInSamples(probabilities = [], startSample = 0, endSample = 0) {
  if (!probabilities?.length || endSample <= startSample) return 0;
  const first = Math.max(0, Math.min(probabilities.length - 1, Math.floor(startSample / VAD_WINDOW_SIZE)));
  const last = Math.max(first + 1, Math.min(probabilities.length, Math.ceil(endSample / VAD_WINDOW_SIZE)));
  let max = 0;
  for (let i = first; i < last; i += 1) {
    const value = Number(probabilities[i]);
    if (Number.isFinite(value) && value > max) max = value;
  }
  return max;
}

function annotateAsrSuspects(words = [], audio = null, vadProbabilities = []) {
  if (!Array.isArray(words) || words.length < 2) return { words, stats: { total: 0, disagree: 0, entropy: 0, gap: 0 } };
  const annotated = words.map((word) => ({ ...word }));
  const stats = { total: 0, disagree: 0, entropy: 0, gap: 0 };
  const hasTsallis = annotated.some((word) => word.tsallis_max !== null && word.tsallis_max !== undefined);
  const hasMargin = annotated.some((word) => word.margin_min !== null && word.margin_min !== undefined);
  const hasEntropy = annotated.some((word) => word.entropy_norm !== null && word.entropy_norm !== undefined);
  const gapSuspects = new Set();

  for (let i = 0; i < annotated.length; i += 1) {
    const word = annotated[i];
    let suspect = false;
    if (word._disagree) {
      suspect = true;
      stats.disagree += 1;
    } else if (hasTsallis) {
      const tsallis = Number(word.tsallis_max);
      const margin = Number(word.margin_min);
      if (Number.isFinite(tsallis) && tsallis > 0.04) {
        if (hasMargin && Number.isFinite(margin)) {
          suspect = margin < 0.6;
        } else {
          suspect = tsallis > 0.12;
        }
      }
      if (suspect) stats.entropy += 1;
    } else if (hasEntropy) {
      const entropy = Number(word.entropy_norm);
      if (Number.isFinite(entropy) && entropy > 0.10) {
        suspect = true;
        stats.entropy += 1;
      }
    }
    if (suspect) word._suspect_level = "warning";
  }

  if (audio?.length) {
    for (let i = 0; i < annotated.length - 1; i += 1) {
      const current = annotated[i];
      const next = annotated[i + 1];
      const gapStart = Number(current.end);
      const gapEnd = Number(next.start);
      const gapMs = (gapEnd - gapStart) * 1000;
      if (!Number.isFinite(gapMs) || gapMs < 200) continue;
      const startSample = Math.max(0, Math.floor(gapStart * VAD_SAMPLE_RATE));
      const endSample = Math.min(audio.length, Math.ceil(gapEnd * VAD_SAMPLE_RATE));
      if (endSample - startSample < 80) continue;
      const gapAudio = audio.subarray(startSample, endSample);
      const peaks = countEnergyPeaks(gapAudio, VAD_SAMPLE_RATE);
      const energyRange = computeGapEnergyRange(gapAudio, VAD_SAMPLE_RATE);
      const vadMax = maxVadProbabilityInSamples(vadProbabilities, startSample, endSample);
      if (vadMax >= 0.90 && (gapMs >= 500 || peaks.length >= 3) && energyRange >= 0.04) {
        gapSuspects.add(i);
        current.gap_after_ms = Math.round(gapMs);
        next.gap_before_ms = Math.round(gapMs);
      }
    }
  }

  for (let i = 0; i < annotated.length; i += 1) {
    if (!annotated[i]._suspect_level && (gapSuspects.has(i) || gapSuspects.has(i - 1))) {
      annotated[i]._suspect_level = "warning";
      stats.gap += 1;
    }
  }
  stats.total = annotated.filter((word) => word._suspect_level).length;
  return { words: annotated, stats };
}

function normalizeAsrText(text) {
  return (text || "")
    .toLocaleLowerCase("vi-VN")
    .replace(/\s+/g, " ")
    .trim();
}

function splitRoverWords(text) {
  return normalizeAsrText(text)
    .split(/\s+/)
    .map((word) => ({ text: word, norm: normalizeOverlapToken(word) }))
    .filter((word) => word.norm);
}

function splitRoverWordObjects(wordsOrText) {
  if (!Array.isArray(wordsOrText)) return splitRoverWords(wordsOrText);
  return wordsOrText
    .map((word) => ({
      ...word,
      text: normalizeAsrText(word?.text || ""),
      norm: normalizeOverlapToken(word?.text || ""),
    }))
    .filter((word) => word.norm);
}

function roverSubstitutionCost(a, b) {
  if (a.norm === b.norm) return 0;
  if (a.norm.length > 2 && b.norm.length > 2 && (a.norm.includes(b.norm) || b.norm.includes(a.norm))) {
    return 0.45;
  }
  return 1.15;
}

function roverAlignmentOps(wordsA, wordsB) {
  const n = wordsA.length;
  const m = wordsB.length;
  const width = m + 1;
  const dp = new Float32Array((n + 1) * (m + 1));
  const bt = new Uint8Array((n + 1) * (m + 1));
  const idx = (i, j) => i * width + j;

  for (let i = 1; i <= n; i += 1) {
    dp[idx(i, 0)] = i;
    bt[idx(i, 0)] = 2;
  }
  for (let j = 1; j <= m; j += 1) {
    dp[idx(0, j)] = j;
    bt[idx(0, j)] = 3;
  }

  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      const subCost = roverSubstitutionCost(wordsA[i - 1], wordsB[j - 1]);
      let best = dp[idx(i - 1, j - 1)] + subCost;
      let op = subCost === 0 ? 1 : 4;
      const del = dp[idx(i - 1, j)] + 1;
      if (del < best) {
        best = del;
        op = 2;
      }
      const ins = dp[idx(i, j - 1)] + 1;
      if (ins < best) {
        best = ins;
        op = 3;
      }
      dp[idx(i, j)] = best;
      bt[idx(i, j)] = op;
    }
  }

  const ops = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    const op = bt[idx(i, j)];
    if (op === 1) {
      ops.push({ type: "equal", a: wordsA[i - 1], b: wordsB[j - 1] });
      i -= 1;
      j -= 1;
    } else if (op === 4) {
      ops.push({ type: "replace", a: wordsA[i - 1], b: wordsB[j - 1] });
      i -= 1;
      j -= 1;
    } else if (op === 2) {
      ops.push({ type: "delete", a: wordsA[i - 1] });
      i -= 1;
    } else {
      ops.push({ type: "insert", b: wordsB[j - 1] });
      j -= 1;
    }
  }
  return ops.reverse();
}

function roverBlockScore(words) {
  if (!words.length) return -Infinity;
  let score = 0;
  let previous = "";
  for (const word of words) {
    const norm = word.norm;
    score += Math.min(8, norm.length);
    if (norm.length <= 1 && !["a", "à", "ạ", "ờ", "ừ", "ừm"].includes(norm)) score -= 2.5;
    if (norm === previous) score -= 1.5;
    if (["à", "ờ", "ừ", "ừm", "uh", "um"].includes(norm)) score -= 2;
    previous = norm;
  }
  return score / Math.max(1, words.length);
}

function roverDedupAdjacent(words) {
  const output = [];
  for (const word of words) {
    const previous = output[output.length - 1];
    if (previous && previous.norm === word.norm) continue;
    output.push(word);
  }
  return output;
}

function roverMergeTexts(textA, textB) {
  return roverMergeWords(textA, textB);
}

function roverOutputWord(word, disagree = false) {
  const output = { ...(word || {}), text: word?.text || "" };
  if (disagree) output._disagree = true;
  return output;
}

function roverMergeWords(inputA, inputB) {
  const wordsA = splitRoverWordObjects(inputA);
  const wordsB = splitRoverWordObjects(inputB);
  if (!wordsA.length && !wordsB.length) {
    return { text: "", words: [], stats: { replacements: 0, replacementB: 0, inserts: 0, deletes: 0 } };
  }
  if (!wordsA.length) {
    const words = wordsB.map((word) => roverOutputWord(word, true));
    return { text: words.map((word) => word.text).join(" "), words, stats: { replacements: 0, replacementB: 0, inserts: wordsB.length, deletes: 0 } };
  }
  if (!wordsB.length) {
    const words = wordsA.map((word) => roverOutputWord(word, false));
    return { text: words.map((word) => word.text).join(" "), words, stats: { replacements: 0, replacementB: 0, inserts: 0, deletes: 0 } };
  }

  const ops = roverAlignmentOps(wordsA, wordsB);
  const merged = [];
  const stats = { replacements: 0, replacementB: 0, inserts: 0, deletes: 0 };
  let blockA = [];
  let blockB = [];

  function flushBlock() {
    if (!blockA.length && !blockB.length) return;
    if (blockA.length && blockB.length) {
      stats.replacements += 1;
      const scoreA = roverBlockScore(blockA);
      const scoreB = roverBlockScore(blockB);
      if (scoreB >= scoreA - 0.25) {
        merged.push(...blockB.map((word) => roverOutputWord(word, true)));
        stats.replacementB += 1;
      } else {
        merged.push(...blockA.map((word) => roverOutputWord(word, true)));
      }
    } else if (blockA.length) {
      merged.push(...blockA.map((word) => roverOutputWord(word, true)));
      stats.deletes += blockA.length;
    } else {
      const insertScore = roverBlockScore(blockB);
      if (insertScore > 1.5) {
        merged.push(...blockB.map((word) => roverOutputWord(word, true)));
        stats.inserts += blockB.length;
      }
    }
    blockA = [];
    blockB = [];
  }

  for (const op of ops) {
    if (op.type === "equal") {
      flushBlock();
      merged.push(roverOutputWord(op.a, false));
    } else {
      if (op.a) blockA.push(op.a);
      if (op.b) blockB.push(op.b);
    }
  }
  flushBlock();

  const deduped = roverDedupAdjacent(merged);
  return {
    text: deduped.map((word) => word.text).join(" "),
    words: deduped,
    stats,
  };
}

function shouldUseParallelAsr(chunks, modelConfig, options = {}) {
  if (modelConfig.type !== "single") return false;
  if (!window.crossOriginIsolated) return false;
  if ((chunks?.length || 0) < 4) return false;
  return boundedNumber(options.cpuThreads, getRequestedThreads(), 1, maxWasmThreads()) >= 4;
}

async function decodeAsrChunkWithCall(call, samples, timeline, chunk, modelConfig, concatSamples = null) {
  const chunkSamples = concatSamples
    ? new Float32Array(concatSamples.subarray(chunk.start, chunk.end))
    : copyTimelineRange(samples, timeline, chunk.start, chunk.end);
  const response = await call(
    "decode",
    { modelId: modelConfig.id, samples: chunkSamples, timeOffset: chunk.start / VAD_SAMPLE_RATE },
    [chunkSamples.buffer]
  );
  const result = response.result || {};
  return {
    ...chunk,
    text: normalizeAsrText(result.text),
    words: Array.isArray(result.words) ? result.words : null,
    result,
  };
}

async function runAsrChunkSequence(sequence, call, results, samples, timeline, modelConfig, options, state, concatSamples = null) {
  for (const chunk of sequence) {
    if (results[chunk.index - 1]) {
      if (options.progress) options.progress(state.done, state.total);
      continue;
    }
    setAsrStatus(`decoding ${chunk.index}/${state.total}`);
    const decoded = await decodeAsrChunkWithCall(call, samples, timeline, chunk, modelConfig, concatSamples);
    results[chunk.index - 1] = decoded;
    state.done += 1;
    if (options.onAsrChunkDecoded) {
      options.onAsrChunkDecoded(decoded, results).catch((error) => {
        log(`[resume_after_kill] ASR chunk checkpoint failed: ${error.message || String(error)}`);
      });
    }

    const partial = results.filter(Boolean);
    renderTranscript(
      mergeChunkTexts(partial),
      `ASR chunk ${state.done}/${state.total}; ${(state.speechSeconds).toFixed(1)}s speech`
    );
    log(`ASR chunk ${chunk.index}/${state.total}: ${decoded.text || "[empty]"}`);
    if (options.progress) options.progress(state.done, state.total);
  }
}

async function decodeAsrChunksParallel(samples, timeline, chunks, modelConfig, options, speechSeconds, concatSamples = null, initialResults = null) {
  const secondary = createAsrWorkerHandle("ASR worker 2", modelConfig);
  try {
    await loadAsrModelIntoWorkerHandle(secondary, modelConfig, options);
    const results = initialResults || new Array(chunks.length);
    const state = { done: results.filter(Boolean).length, total: chunks.length, speechSeconds };
    const even = chunks.filter((_, index) => index % 2 === 0);
    const odd = chunks.filter((_, index) => index % 2 === 1);
    log(`ASR parallel decode: 2 worker(s), ${options.cpuThreads || getRequestedThreads()} thread(s) per worker.`);
    if (options.progress) options.progress(state.done, chunks.length);
    await Promise.all([
      runAsrChunkSequence(even, callAsrWorker, results, samples, timeline, modelConfig, options, state, concatSamples),
      runAsrChunkSequence(odd, secondary.call, results, samples, timeline, modelConfig, options, state, concatSamples),
    ]);
    return results;
  } finally {
    secondary.terminate();
    setAsrStatus("ready");
  }
}

async function runFullAsr(samples, vadSegments, options = {}) {
  const modelConfig = options.asrModel || getSelectedAsrModel();
  if (modelConfig.type === "rover") {
    return runFullRoverAsr(samples, vadSegments, { ...options, asrModel: modelConfig });
  }
  const timeline = buildSpeechTimeline(samples, vadSegments);
  const concatSamples = copyTimelineRange(samples, timeline, 0, timeline.totalSamples);
  const chunks = makeAsrChunks(timeline, concatSamples);
  if (!chunks.length) {
    throw new Error("ASR has no speech samples to decode.");
  }

  const speechSeconds = timeline.totalSamples / VAD_SAMPLE_RATE;
  log(
    `ASR input (${modelConfig.label}): ${speechSeconds.toFixed(1)}s speech, ` +
    `${timeline.sourceSegments.length} merged segment(s), ${chunks.length} chunk(s), ` +
    `chunking=${chunks.chunkingMode}, silenceRegions=${chunks.silentRegionCount || 0}.`
  );
  await ensureAsrReady(modelConfig, {
    cpuThreads: options.cpuThreads,
    hotwordsText: options.hotwordsText,
    hotwordsScore: options.hotwordsScore,
    providerMode: "wasm",
  });
  const started = performance.now();
  const hadAsrChunkCheckpoint = Boolean(options.resumeAsrChunks?.results?.length);
  let results = decodeAsrChunksCheckpoint(options.resumeAsrChunks, chunks, modelConfig) || new Array(chunks.length);
  const resumedChunks = results.filter(Boolean).length;
  if (resumedChunks) {
    renderTranscript(mergeChunkTexts(results.filter(Boolean)), `ASR resumed ${resumedChunks}/${chunks.length} chunk(s)`);
    log(`[resume_after_kill] Resumed ASR chunks: ${resumedChunks}/${chunks.length}.`);
  } else if (hadAsrChunkCheckpoint) {
    log("[resume_after_kill] ASR chunk checkpoint was present but did not match the current model/chunk layout; decoding from the beginning.");
  }
  let checkpointQueue = Promise.resolve();
  if (options.resumeContext) {
    options.onAsrChunkDecoded = (_decoded, currentResults) => {
      checkpointQueue = checkpointQueue.catch(() => null).then(() => writeResumeJsonCheckpoint(
        options.resumeContext,
        "asr_chunks",
        encodeAsrChunksCheckpoint(modelConfig, chunks, currentResults, speechSeconds),
        {
          completedChunks: currentResults.filter(Boolean).length,
          totalChunks: chunks.length,
          model: modelConfig.id,
        }
      )).catch((error) => {
        log(`[resume_after_kill] ASR chunk checkpoint failed: ${error.message || String(error)}`);
      });
      return checkpointQueue;
    };
  }

  if (shouldUseParallelAsr(chunks, modelConfig, options)) {
    results = await decodeAsrChunksParallel(samples, timeline, chunks, modelConfig, options, speechSeconds, concatSamples, results);
  } else {
    for (const chunk of chunks) {
      if (results[chunk.index - 1]) {
        if (options.progress) options.progress(chunk.index, chunks.length);
        continue;
      }
      setAsrStatus(`decoding ${chunk.index}/${chunks.length}`);
      if (options.progress) options.progress(chunk.index - 1, chunks.length);
      const decoded = await decodeAsrChunkWithCall(callAsrWorker, samples, timeline, chunk, modelConfig, concatSamples);
      results[chunk.index - 1] = decoded;
      if (options.onAsrChunkDecoded) await options.onAsrChunkDecoded(decoded, results);

      const mergedText = mergeChunkTexts(results);
      renderTranscript(
        mergedText,
        `ASR chunk ${chunk.index}/${chunks.length}; ${(speechSeconds).toFixed(1)}s speech`
      );
      log(`ASR chunk ${chunk.index}/${chunks.length}: ${decoded.text || "[empty]"}`);
      if (options.progress) options.progress(chunk.index, chunks.length);
    }
  }
  await checkpointQueue.catch(() => null);

  const elapsed = (performance.now() - started) / 1000;
  setAsrStatus("ready");

  const mergedWords = mergeChunkWordsWithOverlap(results);
  let filtered = null;
  let words = [];
  let text = "";
  if (mergedWords?.words?.length) {
    filtered = removeAsrFillerWords(mergedWords.words);
    words = mapAsrWordsToSourceTimeline(filtered.words, timeline);
    const suspect = annotateAsrSuspects(words, samples, options.vadProbabilities || []);
    words = suspect.words;
    if (suspect.stats.total) {
      log(
        `ASR suspect metadata: ${suspect.stats.total}/${words.length} word(s) ` +
        `(${suspect.stats.disagree} disagree, ${suspect.stats.entropy} entropy, ${suspect.stats.gap} gap).`
      );
    }
    text = words.map((word) => word.text).join(" ").trim();
  } else {
    filtered = removeAsrFillerWordsFromText(mergeChunkTexts(results));
    text = filtered.text;
  }
  const confidence = asrConfidenceFromWords(words.length ? words : collectAsrConfidenceWords(results));
  if (filtered.removed) {
    log(`ASR filler removal: removed ${filtered.removed} word(s) (${filtered.original} -> ${text ? text.split(/\s+/).length : 0}).`);
  }
  renderTranscript(text, `ASR full run ${elapsed.toFixed(2)}s`);
  log(`ASR finished in ${elapsed.toFixed(2)}s: ${text || "[empty]"}`);
  return {
    text,
    elapsed,
    results,
    chunks,
    timeline,
    words,
    confidence,
    speechSeconds,
    mergedSegments: timeline.sourceSegments.length,
    model: modelConfig.id,
    modelLabel: modelConfig.label,
  };
}

function summarizeAsrWorkerTimings(results = []) {
  const totals = {
    fbankMs: 0,
    encoderMs: 0,
    decoderMs: 0,
    joinerMs: 0,
    searchMs: 0,
  };
  for (const item of results || []) {
    const timings = item?.result?.timings || {};
    for (const key of Object.keys(totals)) {
      totals[key] += Number(timings[key] || 0);
    }
  }
  return {
    fbankSeconds: Number((totals.fbankMs / 1000).toFixed(3)),
    encoderSeconds: Number((totals.encoderMs / 1000).toFixed(3)),
    decoderSeconds: Number((totals.decoderMs / 1000).toFixed(3)),
    joinerSeconds: Number((totals.joinerMs / 1000).toFixed(3)),
    searchSeconds: Number((totals.searchMs / 1000).toFixed(3)),
  };
}

function asrOutputSummaryForBenchmark(asr, provider = "wasm") {
  const words = asr?.words || [];
  return {
    provider,
    model: asr?.model || null,
    chunks: asr?.chunks?.length || 0,
    speechSeconds: Number(asr?.speechSeconds?.toFixed?.(3) || 0),
    textChars: String(asr?.text || "").length,
    textHash: hashString(asr?.text || ""),
    words: words.length,
    wordTimingHash: hashString(words.map((word) => (
      `${word.text}:${Number(word.start).toFixed(3)}-${Number(word.end).toFixed(3)}`
    )).join("|")),
    timings: summarizeAsrWorkerTimings(asr?.results || []),
    providerDetails: asr?.providerDetails || null,
  };
}

async function runFullAsrWithDedicatedWorker(samples, vadSegments, options = {}) {
  const modelConfig = options.asrModel || getSelectedAsrModel();
  if (modelConfig.type === "rover") {
    throw new Error("ASR WebGPU benchmark branch does not support ROVER models yet.");
  }
  const timeline = buildSpeechTimeline(samples, vadSegments);
  const concatSamples = copyTimelineRange(samples, timeline, 0, timeline.totalSamples);
  const chunks = makeAsrChunks(timeline, concatSamples);
  if (!chunks.length) {
    throw new Error("ASR has no speech samples to decode.");
  }

  const providerMode = options.providerMode === "webgpu" ? "webgpu" : "wasm";
  const speechSeconds = timeline.totalSamples / VAD_SAMPLE_RATE;
  const started = performance.now();
  const results = new Array(chunks.length);
  const state = { done: 0, total: chunks.length, speechSeconds };
  const handles = [];
  const workerCount = providerMode === "webgpu"
    ? 1
    : (shouldUseParallelAsr(chunks, modelConfig, options) ? 2 : 1);
  try {
    for (let i = 0; i < workerCount; i += 1) {
      const handle = createAsrWorkerHandle(`ASR ${providerMode} benchmark worker ${i + 1}`, modelConfig);
      handles.push(handle);
      await loadAsrModelIntoWorkerHandle(handle, modelConfig, {
        ...options,
        providerMode,
      });
    }

    if (workerCount > 1) {
      const even = chunks.filter((_, index) => index % 2 === 0);
      const odd = chunks.filter((_, index) => index % 2 === 1);
      log(`ASR ${providerMode} benchmark decode: 2 worker(s), ${options.cpuThreads || getRequestedThreads()} thread(s) per worker.`);
      await Promise.all([
        runAsrChunkSequence(even, handles[0].call, results, samples, timeline, modelConfig, options, state, concatSamples),
        runAsrChunkSequence(odd, handles[1].call, results, samples, timeline, modelConfig, options, state, concatSamples),
      ]);
    } else {
      await runAsrChunkSequence(chunks, handles[0].call, results, samples, timeline, modelConfig, options, state, concatSamples);
    }
  } finally {
    for (const handle of handles) handle.terminate();
    setAsrStatus("ready");
  }

  const elapsed = (performance.now() - started) / 1000;
  const mergedWords = mergeChunkWordsWithOverlap(results);
  let filtered = null;
  let words = [];
  let text = "";
  if (mergedWords?.words?.length) {
    filtered = removeAsrFillerWords(mergedWords.words);
    words = mapAsrWordsToSourceTimeline(filtered.words, timeline);
    const suspect = annotateAsrSuspects(words, samples, options.vadProbabilities || []);
    words = suspect.words;
    text = words.map((word) => word.text).join(" ").trim();
  } else {
    filtered = removeAsrFillerWordsFromText(mergeChunkTexts(results));
    text = filtered.text;
  }
  const firstProviderDetails = results.find((item) => item?.result?.providers)?.result?.providers || null;
  return {
    text,
    elapsed,
    results,
    chunks,
    timeline,
    words,
    confidence: asrConfidenceFromWords(words.length ? words : collectAsrConfidenceWords(results)),
    speechSeconds,
    mergedSegments: timeline.sourceSegments.length,
    model: modelConfig.id,
    modelLabel: modelConfig.label,
    providerDetails: firstProviderDetails,
  };
}

async function addAsrBenchmarkWebGpuBranch(options, asr, asrAudio, vadSegments, pipelineOptions, vadProbabilities = []) {
  if (!Array.isArray(options?.benchmarkStages)) return;
  addBenchmarkStage(options, {
    name: "ASR full pipeline",
    capability: "wasm-only",
    attempts: [{
      runtime: "wasm",
      provider: "wasm",
      elapsedSeconds: Number(asr.elapsed.toFixed(3)),
      summary: asrOutputSummaryForBenchmark(asr, "wasm"),
    }],
    selectedRuntime: "wasm",
    selectedProvider: "wasm",
    webgpuNotApplicable: {
      reason: "ASR is intentionally fixed to the full WASM path for offline PWA calibration and benchmark.",
    },
  });
  if (options && typeof options === "object") {
    options.benchmarkSelectedProviders = options.benchmarkSelectedProviders || {};
    options.benchmarkSelectedProviders["ASR full pipeline"] = "wasm";
  }
  log(`[Benchmark] Stage ASR full pipeline: WASM ${Number(asr.elapsed || 0).toFixed(2)}s; WebGPU branch disabled.`);
  return;
  const attempts = [{
    runtime: "wasm",
    provider: "wasm",
    elapsedSeconds: Number(asr.elapsed.toFixed(3)),
    summary: asrOutputSummaryForBenchmark(asr, "wasm"),
  }];
  let webgpuResult = null;
  try {
    const webgpuStarted = performance.now();
    webgpuResult = await runFullAsrWithDedicatedWorker(asrAudio, vadSegments, {
      asrModel: pipelineOptions.asrModel,
      cpuThreads: pipelineOptions.cpuThreads,
      hotwordsText: pipelineOptions.hotwordsText,
      hotwordsScore: pipelineOptions.hotwordsScore,
      vadProbabilities,
      providerMode: "webgpu",
      progress: (done, total) => {
        setPipelineProgress("ASR WebGPU benchmark", 30 + (done / Math.max(1, total)) * 35);
      },
    });
    const provider = webgpuResult.providerDetails &&
      Object.values(webgpuResult.providerDetails).some((value) => value === "webgpu")
      ? "webgpu"
      : "wasm";
    attempts.push({
      runtime: "webgpu",
      provider,
      elapsedSeconds: benchmarkSeconds(webgpuStarted),
      summary: asrOutputSummaryForBenchmark(webgpuResult, provider),
    });
  } catch (error) {
    attempts.push({
      runtime: "webgpu",
      error: { message: error.message || String(error) },
    });
    log(`[Benchmark] Stage ASR full pipeline: WEBGPU failed: ${error.message || String(error)}`);
  }

  const wasmAttempt = attempts.find((attempt) => attempt.runtime === "wasm" && !attempt.error);
  const webgpuAttempt = attempts.find((attempt) => attempt.runtime === "webgpu" && !attempt.error);
  const outputHashEqual = wasmAttempt && webgpuAttempt
    ? benchmarkOutputHash(wasmAttempt.summary) === benchmarkOutputHash(webgpuAttempt.summary) &&
      wasmAttempt.summary.wordTimingHash === webgpuAttempt.summary.wordTimingHash
    : null;
  const speedupWebgpuOverWasm = wasmAttempt?.elapsedSeconds && webgpuAttempt?.elapsedSeconds
    ? Number((wasmAttempt.elapsedSeconds / webgpuAttempt.elapsedSeconds).toFixed(3))
    : null;
  const webgpuAccepted = Boolean(
    webgpuAttempt &&
    webgpuAttempt.provider === "webgpu" &&
    outputHashEqual === true &&
    speedupWebgpuOverWasm > 1
  );
  const mismatchDetails = outputHashEqual === false
    ? benchmarkMismatchDetails(
        "ASR full pipeline",
        asr,
        webgpuResult,
        wasmAttempt?.summary,
        webgpuAttempt?.summary
      )
    : null;
  let rejectionReason = null;
  if (webgpuAttempt && !webgpuAccepted) {
    if (webgpuAttempt.provider !== "webgpu") rejectionReason = `webgpu attempt ran on ${webgpuAttempt.provider}`;
    else if (outputHashEqual !== true) rejectionReason = "webgpu output hash did not match wasm";
    else if (!(speedupWebgpuOverWasm > 1)) rejectionReason = "webgpu was not faster than wasm";
  } else if (!webgpuAttempt) {
    const webgpuError = attempts.find((attempt) => attempt.runtime === "webgpu" && attempt.error);
    if (webgpuError) rejectionReason = webgpuError.error?.message || "webgpu attempt failed";
  }
  addBenchmarkStage(options, {
    name: "ASR full pipeline",
    capability: "wasm-webgpu",
    attempts,
    selectedRuntime: "wasm",
    selectedProvider: "wasm",
    comparison: {
      speedupWebgpuOverWasm,
      outputHashEqual,
      webgpuAccepted,
      rejectionReason,
      ...(mismatchDetails ? { mismatchDetails } : {}),
    },
  });
  if (webgpuAttempt) {
    log(
      `[Benchmark] Stage ASR full pipeline: WASM ${wasmAttempt.elapsedSeconds.toFixed(2)}s, ` +
      `WEBGPU ${webgpuAttempt.elapsedSeconds.toFixed(2)}s (${webgpuAttempt.provider}), ` +
      `hashEqual=${outputHashEqual}.`
    );
  }
}

async function runFullRoverAsr(samples, vadSegments, options = {}) {
  const modelConfig = options.asrModel || ASR_MODELS[ROVER_MODEL_ID];
  const childIds = modelConfig.modelIds || [];
  if (childIds.length !== 2) {
    throw new Error("ROVER requires exactly two ASR models in offline PWA.");
  }
  const modelA = { id: childIds[0], ...ASR_MODELS[childIds[0]] };
  const modelB = { id: childIds[1], ...ASR_MODELS[childIds[1]] };
  if (!modelA.files || !modelB.files) {
    throw new Error("ROVER child ASR model configuration is incomplete.");
  }

  const timeline = buildSpeechTimeline(samples, vadSegments);
  const concatSamples = copyTimelineRange(samples, timeline, 0, timeline.totalSamples);
  const chunks = makeAsrChunks(timeline, concatSamples);
  if (!chunks.length) {
    throw new Error("ASR has no speech samples to decode.");
  }

  const speechSeconds = timeline.totalSamples / VAD_SAMPLE_RATE;
  log(
    `ROVER input: ${speechSeconds.toFixed(1)}s speech, ` +
    `${timeline.sourceSegments.length} merged segment(s), ${chunks.length} chunk(s), ` +
    `chunking=${chunks.chunkingMode}, silenceRegions=${chunks.silentRegionCount || 0}.`
  );
  await ensureAsrReady(modelConfig, {
    cpuThreads: options.cpuThreads,
    hotwordsText: options.hotwordsText,
    hotwordsScore: options.hotwordsScore,
  });
  const started = performance.now();
  const hadAsrChunkCheckpoint = Boolean(options.resumeAsrChunks?.results?.length);
  const results = decodeAsrChunksCheckpoint(options.resumeAsrChunks, chunks, modelConfig) || new Array(chunks.length);
  const totalStats = { replacements: 0, replacementB: 0, inserts: 0, deletes: 0 };
  for (const item of results.filter(Boolean)) {
    const stats = item?.result?.stats || {};
    for (const key of Object.keys(totalStats)) totalStats[key] += stats[key] || 0;
  }
  const resumedChunks = results.filter(Boolean).length;
  if (resumedChunks) {
    renderTranscript(mergeChunkTexts(results.filter(Boolean)), `ROVER resumed ${resumedChunks}/${chunks.length} chunk(s)`);
    log(`[resume_after_kill] Resumed ROVER chunks: ${resumedChunks}/${chunks.length}.`);
  } else if (hadAsrChunkCheckpoint) {
    log("[resume_after_kill] ROVER chunk checkpoint was present but did not match the current model/chunk layout; decoding from the beginning.");
  }
  let checkpointQueue = Promise.resolve();
  const checkpointRoverChunks = () => {
    if (!options.resumeContext) return Promise.resolve(null);
    checkpointQueue = checkpointQueue.catch(() => null).then(() => writeResumeJsonCheckpoint(
      options.resumeContext,
      "asr_chunks",
      encodeAsrChunksCheckpoint(modelConfig, chunks, results, speechSeconds),
      {
        completedChunks: results.filter(Boolean).length,
        totalChunks: chunks.length,
        model: modelConfig.id,
      }
    )).catch((error) => {
      log(`[resume_after_kill] ROVER chunk checkpoint failed: ${error.message || String(error)}`);
    });
    return checkpointQueue;
  };

  for (const chunk of chunks) {
    if (results[chunk.index - 1]) {
      if (options.progress) options.progress(chunk.index, chunks.length);
      continue;
    }
    setAsrStatus(`ROVER decoding ${chunk.index}/${chunks.length}`);
    if (options.progress) options.progress(chunk.index - 1, chunks.length);
    const chunkSamples = new Float32Array(concatSamples.subarray(chunk.start, chunk.end));
    const samplesA = new Float32Array(chunkSamples);
    const timeOffset = chunk.start / VAD_SAMPLE_RATE;
    const responseA = await callAsrWorker("decode", { modelId: modelA.id, samples: samplesA, timeOffset }, [samplesA.buffer]);
    const responseB = await callAsrWorker("decode", { modelId: modelB.id, samples: chunkSamples, timeOffset }, [chunkSamples.buffer]);
    const textA = normalizeAsrText(responseA.result?.text || "");
    const textB = normalizeAsrText(responseB.result?.text || "");
    const merged = Array.isArray(responseA.result?.words) || Array.isArray(responseB.result?.words)
      ? roverMergeWords(responseA.result?.words || [], responseB.result?.words || [])
      : roverMergeTexts(textA, textB);
    for (const key of Object.keys(totalStats)) {
      totalStats[key] += merged.stats[key] || 0;
    }

    results[chunk.index - 1] = {
      ...chunk,
      text: merged.text,
      words: merged.words || null,
      result: {
        rover: true,
        primary: responseA.result || {},
        secondary: responseB.result || {},
        stats: merged.stats,
      },
    };
    await checkpointRoverChunks();

    const mergedText = mergeChunkTexts(results);
    renderTranscript(
      mergedText,
      `ROVER chunk ${chunk.index}/${chunks.length}; ${(speechSeconds).toFixed(1)}s speech`
    );
    log(
      `ROVER chunk ${chunk.index}/${chunks.length}: ` +
      `${merged.text || "[empty]"} ` +
      `(replace ${merged.stats.replacementB}/${merged.stats.replacements}->68M, insert ${merged.stats.inserts})`
    );
    if (options.progress) options.progress(chunk.index, chunks.length);
  }
  await checkpointQueue.catch(() => null);

  const elapsed = (performance.now() - started) / 1000;
  setAsrStatus("ready");

  const mergedWords = mergeChunkWordsWithOverlap(results);
  let filtered = null;
  let words = [];
  let text = "";
  if (mergedWords?.words?.length) {
    filtered = removeAsrFillerWords(mergedWords.words);
    words = mapAsrWordsToSourceTimeline(filtered.words, timeline);
    const suspect = annotateAsrSuspects(words, samples, options.vadProbabilities || []);
    words = suspect.words;
    if (suspect.stats.total) {
      log(
        `ROVER suspect metadata: ${suspect.stats.total}/${words.length} word(s) ` +
        `(${suspect.stats.disagree} disagree, ${suspect.stats.entropy} entropy, ${suspect.stats.gap} gap).`
      );
    }
    text = words.map((word) => word.text).join(" ").trim();
  } else {
    filtered = removeAsrFillerWordsFromText(mergeChunkTexts(results));
    text = filtered.text;
  }
  const confidence = asrConfidenceFromWords(words.length ? words : collectAsrConfidenceWords(results));
  if (filtered.removed) {
    log(`ROVER filler removal: removed ${filtered.removed} word(s) (${filtered.original} -> ${text ? text.split(/\s+/).length : 0}).`);
  }
  renderTranscript(text, `ROVER full run ${elapsed.toFixed(2)}s`);
  log(
    `ROVER finished in ${elapsed.toFixed(2)}s: replace ` +
    `${totalStats.replacementB}/${totalStats.replacements}->68M, insert ${totalStats.inserts}, delete ${totalStats.deletes}.`
  );
  return {
    text,
    elapsed,
    results,
    chunks,
    timeline,
    words,
    confidence,
    speechSeconds,
    mergedSegments: timeline.sourceSegments.length,
    model: modelConfig.id,
    modelLabel: modelConfig.label,
    roverStats: totalStats,
  };
}

function parseWordPieceVocab(text) {
  const tokenToId = new Map();
  const tokens = text.split(/\r?\n/).filter((line) => line.length > 0);
  tokens.forEach((token, index) => tokenToId.set(token, index));
  return {
    tokenToId,
    padId: tokenToId.get("[PAD]") ?? 0,
    unkId: tokenToId.get("[UNK]") ?? 1,
    startId: tokenToId.get(START_TOKEN) ?? tokenToId.size,
  };
}

function wordPieceTokenize(word, vocab) {
  if (word === START_TOKEN) return [START_TOKEN];
  if (!word) return [];
  if (word.length > 100) return ["[UNK]"];

  const pieces = [];
  let start = 0;
  while (start < word.length) {
    let end = word.length;
    let current = null;

    while (start < end) {
      let sub = word.slice(start, end);
      if (start > 0) sub = `##${sub}`;
      if (vocab.tokenToId.has(sub)) {
        current = sub;
        break;
      }
      end -= 1;
    }

    if (current === null) return ["[UNK]"];
    pieces.push(current);
    start = end;
  }
  return pieces;
}

function encodePunctuationBatch(tokenBatch) {
  const encoded = [];
  let maxSeqLen = 0;
  let maxOffsetsLen = 0;

  for (const sequence of tokenBatch) {
    const prefixed = [START_TOKEN, ...sequence.slice(0, PUNCT_MAX_LEN)];
    const inputIds = [];
    const offsets = [];

    for (const word of prefixed) {
      offsets.push(inputIds.length);
      const pieces = wordPieceTokenize(word, punctuationVocab);
      for (const piece of pieces) {
        if (piece === START_TOKEN) {
          inputIds.push(punctuationVocab.startId);
        } else {
          inputIds.push(punctuationVocab.tokenToId.get(piece) ?? punctuationVocab.unkId);
        }
      }
    }

    encoded.push({ inputIds, offsets });
    maxSeqLen = Math.max(maxSeqLen, inputIds.length);
    maxOffsetsLen = Math.max(maxOffsetsLen, offsets.length);
  }

  const batchSize = encoded.length;
  const inputIds = new BigInt64Array(batchSize * maxSeqLen);
  const attentionMask = new BigInt64Array(batchSize * maxSeqLen);
  const tokenTypeIds = new BigInt64Array(batchSize * maxSeqLen);
  const inputOffsets = new BigInt64Array(batchSize * maxOffsetsLen);

  for (let b = 0; b < batchSize; b += 1) {
    const row = encoded[b];
    for (let i = 0; i < maxSeqLen; i += 1) {
      const pos = b * maxSeqLen + i;
      const id = i < row.inputIds.length ? row.inputIds[i] : punctuationVocab.padId;
      inputIds[pos] = BigInt(id);
      attentionMask[pos] = BigInt(i < row.inputIds.length ? 1 : 0);
      tokenTypeIds[pos] = 0n;
    }
    for (let i = 0; i < maxOffsetsLen; i += 1) {
      const offset = i < row.offsets.length ? row.offsets[i] : 0;
      inputOffsets[b * maxOffsetsLen + i] = BigInt(offset);
    }
  }

  return {
    input_ids: new window.ort.Tensor("int64", inputIds, [batchSize, maxSeqLen]),
    attention_mask: new window.ort.Tensor("int64", attentionMask, [batchSize, maxSeqLen]),
    token_type_ids: new window.ort.Tensor("int64", tokenTypeIds, [batchSize, maxSeqLen]),
    input_offsets: new window.ort.Tensor("int64", inputOffsets, [batchSize, maxOffsetsLen]),
  };
}

async function ensurePunctuationReady() {
  if (punctuationInitPromise) return punctuationInitPromise;

  punctuationInitPromise = (async () => {
    configureOrt();
    log("Loading ViBERT punctuation fp32 ONNX session.");
    const [model, vocabText] = await Promise.all([
      loadModelArrayBuffer("punct.vibert_fp32"),
      loadModelText("punct.vocab"),
    ]);
    punctuationVocab = parseWordPieceVocab(vocabText);
    const created = await createOrtSession(model, {
      name: "ViBERT punctuation fp32",
      webgpuPreferred: true,
    });
    punctuationSession = created.session;
    punctuationExecutionProvider = created.provider;
    log(`ViBERT punctuation fp32 session ready (${punctuationExecutionProvider}).`);
  })().catch((error) => {
    punctuationInitPromise = null;
    punctuationSession = null;
    punctuationVocab = null;
    punctuationExecutionProvider = "wasm";
    throw error;
  });

  return punctuationInitPromise;
}

function softmaxArgmax(data, base, count, adjustments = {}) {
  let maxLogit = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const value = data[base + i];
    if (value > maxLogit) maxLogit = value;
  }

  const probs = new Float32Array(count);
  const adjusted = new Float32Array(count);
  let sum = 0;
  for (let i = 0; i < count; i += 1) {
    const value = Math.exp(data[base + i] - maxLogit);
    probs[i] = value;
    sum += value;
  }

  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < count; i += 1) {
    let value = probs[i] / Math.max(sum, 1e-12);
    if (i === PUNCT_NOOP_INDEX) value += adjustments.keep || 0;
    if (PUNCT_CASE_INDEXES.includes(i)) value += adjustments.case || 0;
    adjusted[i] = value;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }

  const gap = Number(adjustments.pauseGap);
  if (Number.isFinite(gap)) {
    if (gap >= 1.0 && bestIndex === PUNCT_NOOP_INDEX && PUNCT_APPEND_PERIOD_INDEX >= 0) {
      adjusted[PUNCT_NOOP_INDEX] -= 0.2;
      adjusted[PUNCT_APPEND_PERIOD_INDEX] += 0.2;
    } else if (gap >= 0.2 && bestIndex === PUNCT_NOOP_INDEX && PUNCT_APPEND_COMMA_INDEX >= 0) {
      adjusted[PUNCT_APPEND_COMMA_INDEX] += 0.2;
    } else if (gap < 0.1 && PUNCT_APPEND_COMMA_INDEX >= 0) {
      adjusted[PUNCT_APPEND_COMMA_INDEX] -= 0.3;
    }

    bestIndex = 0;
    bestValue = -Infinity;
    for (let i = 0; i < count; i += 1) {
      if (adjusted[i] > bestValue) {
        bestValue = adjusted[i];
        bestIndex = i;
      }
    }
  }
  return { index: bestIndex, prob: bestValue };
}

function convertPunctuationOutputs(logitsTensor, detectTensor, options = {}) {
  const logits = logitsTensor.data;
  const detect = detectTensor.data;
  const batchSize = logitsTensor.dims[0];
  const numWords = logitsTensor.dims[1];
  const labelCount = logitsTensor.dims[2];
  const detectCount = detectTensor.dims[2];
  const punctuationConfidence = options.punctuationConfidence ?? DEFAULT_PUNCT_CONFIDENCE;
  const caseConfidence = options.caseConfidence ?? DEFAULT_PUNCT_CASE_CONFIDENCE;
  const allProbabilities = [];
  const allIdxs = [];
  const errorProbs = [];

  for (let b = 0; b < batchSize; b += 1) {
    const rowProbs = [];
    const rowIdxs = [];
    let maxError = 0;

    for (let w = 0; w < numWords; w += 1) {
      const labelBase = (b * numWords + w) * labelCount;
      const pauseGap = Array.isArray(options.pauseHintsBatch?.[b]) && w > 0
        ? options.pauseHintsBatch[b][w - 1]
        : undefined;
      const label = softmaxArgmax(logits, labelBase, labelCount, {
        keep: punctuationConfidence,
        case: caseConfidence,
        pauseGap,
      });
      rowProbs.push(label.prob);
      rowIdxs.push(label.index);

      const detectBase = (b * numWords + w) * detectCount;
      const detectProb = softmaxArgmax(detect, detectBase, detectCount);
      if (detectProb.index === PUNCT_INCORR_INDEX && detectProb.prob > maxError) {
        maxError = detectProb.prob;
      }
    }

    allProbabilities.push(rowProbs);
    allIdxs.push(rowIdxs);
    errorProbs.push(maxError);
  }

  return { allProbabilities, allIdxs, errorProbs };
}

async function predictPunctuation(tokenBatch, progressCallback, options = {}) {
  const allProbabilities = [];
  const allIdxs = [];
  const errorProbs = [];
  const total = tokenBatch.length;

  for (let start = 0; start < total; start += PUNCT_MINI_BATCH_SIZE) {
    const end = Math.min(total, start + PUNCT_MINI_BATCH_SIZE);
    const inputs = encodePunctuationBatch(tokenBatch.slice(start, end));
    const outputs = await punctuationSession.run(inputs);
    const logits = outputs.logits || outputs[punctuationSession.outputNames[0]];
    const detectLogits = outputs.detect_logits || outputs[punctuationSession.outputNames[1]];
    const pauseHintsBatch = Array.isArray(options.pauseHintsBatch)
      ? options.pauseHintsBatch.slice(start, end)
      : null;
    const converted = convertPunctuationOutputs(logits, detectLogits, {
      ...options,
      pauseHintsBatch,
    });
    allProbabilities.push(...converted.allProbabilities);
    allIdxs.push(...converted.allIdxs);
    errorProbs.push(...converted.errorProbs);
    if (progressCallback) progressCallback(end, total);
  }

  return { allProbabilities, allIdxs, errorProbs };
}

function applyCaseTransform(token, label) {
  if (label.endsWith("LOWER")) return token.toLocaleLowerCase("vi-VN");
  if (label.endsWith("UPPER")) return token.toLocaleUpperCase("vi-VN");
  if (label.endsWith("CAPITAL")) {
    return token.charAt(0).toLocaleUpperCase("vi-VN") + token.slice(1).toLocaleLowerCase("vi-VN");
  }
  if (label.endsWith("CAPITAL_1")) {
    const tail = token.slice(1);
    return token.charAt(0) + tail.charAt(0).toLocaleUpperCase("vi-VN") + tail.slice(1).toLocaleLowerCase("vi-VN");
  }
  if (label.endsWith("UPPER_-1")) {
    return token.slice(0, -1).toLocaleUpperCase("vi-VN") + token.slice(-1);
  }
  return token;
}

function getPunctuationAction(index, prob, label) {
  if (!label || label === "$KEEP" || label === UNK_TOKEN || label === PAD_TOKEN) return null;
  if (label === "$DELETE" || label.startsWith("$REPLACE_")) return null;

  if (label.startsWith("$APPEND_")) {
    const added = label.replace("$APPEND_", "");
    if (!PUNCT_ALLOWED.has(added)) return null;
    return { start: index, end: index, label, prob };
  }

  if (label.startsWith("$TRANSFORM_CASE_")) {
    return { start: index - 1, end: index, label, prob };
  }

  return null;
}

function applyPunctuationEdits(tokens, edits) {
  const target = tokens.slice();
  let shift = 0;

  for (const edit of edits) {
    const targetPos = edit.start + shift;
    if (edit.start < 0) continue;

    const sourceToken = targetPos < target.length ? target[targetPos] : "";
    if (edit.label === "") {
      target.splice(targetPos, 1);
      shift -= 1;
    } else if (edit.start === edit.end) {
      const word = edit.label.replace("$APPEND_", "");
      if (
        (targetPos < target.length && target[targetPos] === word) ||
        (targetPos > 0 && target[targetPos - 1] === word)
      ) {
        continue;
      }
      target.splice(targetPos, 0, word);
      shift += 1;
    } else if (edit.label.startsWith("$TRANSFORM_")) {
      target[targetPos] = applyCaseTransform(sourceToken, edit.label);
    }
  }

  return target;
}

function postprocessPunctuationBatch(tokenBatch, probabilities, idxs, errorProbs) {
  const results = [];
  for (let b = 0; b < tokenBatch.length; b += 1) {
    const tokens = tokenBatch[b];
    const length = Math.min(tokens.length, PUNCT_MAX_LEN);
    if (Math.max(...idxs[b]) === PUNCT_NOOP_INDEX || errorProbs[b] < 0) {
      results.push(tokens);
      continue;
    }

    const edits = [];
    for (let i = 0; i <= length; i += 1) {
      if (idxs[b][i] === PUNCT_NOOP_INDEX) continue;
      const label = PUNCT_LABELS[idxs[b][i]];
      const action = getPunctuationAction(i, probabilities[b][i], label);
      if (action) edits.push(action);
    }
    results.push(applyPunctuationEdits(tokens, edits));
  }
  return results;
}

function splitPunctuationChunks(tokens, pauseHints = null) {
  const chunks = [];
  const hintChunks = Array.isArray(pauseHints) ? [] : null;
  const indices = [];
  const start = chunks.length;
  const count = tokens.length;
  const pushChunk = (chunkStart, chunkEnd) => {
    chunks.push(tokens.slice(chunkStart, chunkEnd));
    if (hintChunks) hintChunks.push(pauseHints.slice(chunkStart, chunkEnd));
  };

  if (count <= PUNCT_CHUNK_SIZE) {
    pushChunk(0, count);
  } else if (count < PUNCT_CHUNK_SIZE * 2 - PUNCT_OVERLAP_SIZE) {
    const splitIdx = Math.floor((count + PUNCT_OVERLAP_SIZE + 1) / 2);
    pushChunk(0, splitIdx);
    pushChunk(splitIdx - PUNCT_OVERLAP_SIZE, count);
  } else {
    for (let i = 0; i < count - PUNCT_OVERLAP_SIZE; i += PUNCT_STRIDE) {
      pushChunk(i, Math.min(count, i + PUNCT_CHUNK_SIZE));
    }
  }

  indices.push([start, chunks.length]);
  return { chunks, indices, hintChunks };
}

function isPunctuationToken(token) {
  return PUNCT_ALLOWED.has(token);
}

function applyPunctuationChunkMerging(tokens, nextTokens) {
  if (!tokens.length) return nextTokens.slice();

  const sourceIdx = [];
  const sourceTokens = [];
  for (let i = tokens.length - 1; i >= 0 && sourceIdx.length < PUNCT_OVERLAP_SIZE; i -= 1) {
    if (!isPunctuationToken(tokens[i])) {
      sourceIdx.unshift(i);
      sourceTokens.unshift(tokens[i].toLocaleLowerCase("vi-VN"));
    }
  }

  const targetIdx = [];
  const targetTokens = [];
  for (let i = 0; i < nextTokens.length && targetIdx.length < PUNCT_OVERLAP_SIZE; i += 1) {
    if (!isPunctuationToken(nextTokens[i])) {
      targetIdx.push(i);
      targetTokens.push(nextTokens[i].toLocaleLowerCase("vi-VN"));
    }
  }

  const maxOverlap = Math.min(sourceTokens.length, targetTokens.length);
  for (let size = maxOverlap; size >= PUNCT_MIN_WORDS_CUT; size -= 1) {
    const sourceStart = sourceTokens.length - size;
    const left = sourceTokens.slice(sourceStart).join("\u0001");
    const right = targetTokens.slice(0, size).join("\u0001");
    if (left !== right) continue;

    const keep = Math.min(PUNCT_OVERLAP_SIZE - PUNCT_MIN_WORDS_CUT, size - 1);
    const tailIdx = sourceIdx[sourceStart + keep];
    const headIdx = targetIdx[keep];
    return tokens.slice(0, tailIdx).concat(nextTokens.slice(headIdx));
  }

  const mergeHead = targetIdx[Math.min(targetIdx.length - 1, PUNCT_OVERLAP_SIZE - PUNCT_MIN_WORDS_CUT)] ?? 0;
  return tokens.concat(nextTokens.slice(mergeHead));
}

function mergePunctuationChunks(chunks) {
  let result = [];
  for (const chunk of chunks) {
    result = applyPunctuationChunkMerging(result, chunk);
  }
  return result.join(" ");
}

function postProcessPunctuationText(text) {
  let output = text.replace(/:/g, " ");
  output = output.replace(/,+/g, ",");
  output = output.replace(/\.{4,}/g, "...");
  output = output.replace(/,\s*\./g, ".");

  const sentences = output.split(/(?<=[.!?])\s+/u);
  output = sentences
    .map((sentence) => {
      const commaCount = (sentence.match(/,/g) || []).length;
      const words = sentence.trim().split(/\s+/).filter(Boolean);
      if (words.length >= 8 || commaCount <= 1) return sentence;
      const firstComma = sentence.indexOf(",");
      if (firstComma < 0) return sentence;
      return sentence.slice(0, firstComma + 1) + sentence.slice(firstComma + 1).replace(/,/g, "");
    })
    .join(" ");

  output = output.replace(/([,.!?])([^\s])/g, "$1 $2");
  output = output.replace(/\s+([,.!?])/g, "$1");
  output = output.replace(/^,\s*/, "");
  output = output.replace(/\.\s*,/g, ". ");
  output = output.replace(/\s+/g, " ");
  output = output.replace(/(^|[.!?]\s+)(\p{L})/gu, (_match, prefix, letter) => (
    prefix + letter.toLocaleUpperCase("vi-VN")
  ));
  return output.trim();
}

function syntheticPunctuationBenchmarkText() {
  const vocabulary = [
    "kinh", "thua", "quoc", "hoi", "hom", "nay", "chung", "toi", "trinh", "bay",
    "noi", "dung", "bao", "cao", "ve", "tinh", "hinh", "kinh", "te", "xa", "hoi",
    "dia", "phuong", "du", "an", "dau", "tu", "giao", "thong", "moi", "truong",
    "nguoi", "dan", "doanh", "nghiep", "can", "duoc", "ho", "tro", "kip", "thoi",
    "giai", "phap", "trien", "khai", "dong", "bo", "hieu", "qua", "minh", "bach",
  ];
  const targetMiniBatches = Math.max(4, Math.min(16, logicalThreadCount()));
  const targetChunks = targetMiniBatches * PUNCT_MINI_BATCH_SIZE;
  const targetWords = PUNCT_CHUNK_SIZE + Math.max(0, targetChunks - 1) * PUNCT_STRIDE;
  const random = createMt19937(0x5eed2026);
  const words = [];
  for (let i = 0; i < targetWords; i += 1) {
    words.push(vocabulary[random.randomInt(vocabulary.length)]);
  }
  return words.join(" ");
}

async function restorePunctuation(text, options = {}) {
  const raw = (text || "").trim();
  if (!raw) return { text: "", elapsed: 0, chunks: 0 };

  await ensurePunctuationReady();
  const started = performance.now();
  const words = raw.split(/\s+/).filter(Boolean);
  const pauseHints = Array.isArray(options.pauseHints) && options.pauseHints.length === words.length
    ? options.pauseHints
    : null;
  const { chunks, indices, hintChunks } = splitPunctuationChunks(words, pauseHints);
  let finalBatch = chunks.map((chunk) => chunk.slice());
  const previousPredictions = new Map(finalBatch.map((chunk, index) => [index, new Set([chunk.join("\u0001")])]));
  let predIds = finalBatch
    .map((chunk, index) => ({ chunk, index }))
    .filter((item) => item.chunk.length >= PUNCT_MIN_LEN)
    .map((item) => item.index);

  for (let iter = 0; iter < PUNCT_ITERATIONS && predIds.length; iter += 1) {
    const origBatch = predIds.map((index) => finalBatch[index]);
    const predicted = await predictPunctuation(
      origBatch,
      (done, total) => {
        if (options.progress) options.progress(iter, done, total);
      },
      {
        ...options,
        pauseHintsBatch: iter === 0 && hintChunks
          ? predIds.map((index) => hintChunks[index])
          : null,
      }
    );
    const predBatch = postprocessPunctuationBatch(
      origBatch,
      predicted.allProbabilities,
      predicted.allIdxs,
      predicted.errorProbs
    );

    const nextPredIds = [];
    for (let i = 0; i < predIds.length; i += 1) {
      const originalId = predIds[i];
      const oldKey = finalBatch[originalId].join("\u0001");
      const newKey = predBatch[i].join("\u0001");
      if (oldKey === newKey) continue;

      finalBatch[originalId] = predBatch[i];
      const seen = previousPredictions.get(originalId);
      if (!seen.has(newKey)) {
        seen.add(newKey);
        nextPredIds.push(originalId);
      }
    }
    predIds = nextPredIds;
  }

  const merged = indices.map(([start, end]) => mergePunctuationChunks(finalBatch.slice(start, end)));
  let restored = merged[0] || raw;
  restored = restored.replace(/\s+([:.,?])/g, "$1");
  restored = postProcessPunctuationText(restored);
  const elapsed = (performance.now() - started) / 1000;
  return { text: restored, elapsed, chunks: chunks.length, executionProvider: punctuationExecutionProvider };
}

async function ensureDiarizationReady() {
  if (diarizationInitPromise) return diarizationInitPromise;

  diarizationInitPromise = (async () => {
    configureOrt();
    log("Loading pyannote segmentation ONNX session.");
    const model = await loadModelArrayBuffer("speaker.pyannote_seg");
    const created = await createOrtSession(model, {
      name: "Pyannote segmentation",
      webgpuPreferred: true,
      webgpuBenchmarkOnly: true,
    });
    diarizationSession = created.session;
    diarizationExecutionProvider = created.provider;
    log(`Pyannote segmentation session ready (${diarizationExecutionProvider}).`);
  })().catch((error) => {
    diarizationInitPromise = null;
    diarizationSession = null;
    diarizationExecutionProvider = "wasm";
    throw error;
  });

  return diarizationInitPromise;
}

async function ensureCamppReady() {
  if (camppInitPromise) return camppInitPromise;

  camppInitPromise = (async () => {
    configureOrt();
    log("Loading CAM++ speaker embedding ONNX session.");
    const model = await loadModelArrayBuffer("speaker.campp");
    const created = await createOrtSession(model, {
      name: "CAM++ speaker embedding",
      webgpuPreferred: true,
    });
    camppSession = created.session;
    camppExecutionProvider = created.provider;
    log(`CAM++ speaker embedding session ready (${camppExecutionProvider}).`);
  })().catch((error) => {
    camppInitPromise = null;
    camppSession = null;
    camppExecutionProvider = "wasm";
    throw error;
  });

  return camppInitPromise;
}

async function ensureOverlapReady() {
  throw new Error("Overlap separation is not included in the main offline PWA model pack.");
}

async function ensurePyannoteCommunityReady() {
  if (pyannoteCommunityInitPromise) return pyannoteCommunityInitPromise;

  pyannoteCommunityInitPromise = (async () => {
    await ensureDiarizationReady();
    configureOrt();
    if (!pyannoteCommunityAssets) {
      log("Loading Pyannote Community-1 VBx assets.");
      const [weight, bias, preparedPlda] = await Promise.all([
        loadNpyArray("speaker.pyannote_resnet_weight"),
        loadNpyArray("speaker.pyannote_resnet_bias"),
        loadNpzArrays("speaker.pyannote_plda_prepared"),
      ]);
      pyannoteCommunityAssets = {
        weight: weight.data,
        weightShape: weight.shape,
        bias: bias.data,
        plda: preparePyannotePlda(preparedPlda),
      };
      if (pyannoteCommunityAssets.weightShape[0] !== 256 || pyannoteCommunityAssets.weightShape[1] !== 5120) {
        throw new Error("Pyannote ResNet projection weight has unexpected shape.");
      }
      if (pyannoteCommunityAssets.bias.length !== 256) {
        throw new Error("Pyannote ResNet projection bias has unexpected shape.");
      }
    }
    if (pyannoteEmbeddingSession) return;
    log("Loading Pyannote Community-1 embedding encoder ONNX session.");
    const encoderModel = await loadModelArrayBuffer("speaker.pyannote_embedding_encoder");
    const created = await createOrtSession(encoderModel, {
      name: "Pyannote Community-1 embedding encoder",
      webgpuPreferred: true,
    });
    pyannoteEmbeddingSession = created.session;
    pyannoteEmbeddingExecutionProvider = created.provider;
    log(`Pyannote Community-1 VBx assets ready (${pyannoteEmbeddingExecutionProvider} embedding encoder).`);
  })().catch((error) => {
    pyannoteCommunityInitPromise = null;
    pyannoteEmbeddingSession = null;
    pyannoteEmbeddingExecutionProvider = "wasm";
    throw error;
  });

  return pyannoteCommunityInitPromise;
}

async function ensureDnsmosReady() {
  if (dnsmosInitPromise) return dnsmosInitPromise;

  dnsmosInitPromise = (async () => {
    configureOrt();
    log("Loading DNSMOS quality ONNX session.");
    const model = await loadModelArrayBuffer("quality.dnsmos");
    const created = await createOrtSession(model, {
      name: "DNSMOS quality",
      webgpuPreferred: true,
      webgpuBenchmarkOnly: true,
    });
    dnsmosSession = created.session;
    dnsmosExecutionProvider = created.provider;
    log(`DNSMOS quality session ready (${dnsmosExecutionProvider}).`);
  })().catch((error) => {
    dnsmosInitPromise = null;
    dnsmosSession = null;
    dnsmosExecutionProvider = "wasm";
    throw error;
  });

  return dnsmosInitPromise;
}

function clipQualityScore(value) {
  return Math.max(1, Math.min(5, Number(value)));
}

function mapDnsmosScores(raw) {
  const sig = raw[0];
  const bak = raw[1];
  const ovr = raw[2];
  return {
    SIG: clipQualityScore(-0.08397278 * sig * sig + 1.22083953 * sig + 0.0052439),
    BAK: clipQualityScore(-0.13166888 * bak * bak + 1.60915514 * bak - 0.39604546),
    OVRL: clipQualityScore(-0.06766283 * ovr * ovr + 1.11546468 * ovr + 0.04602535),
  };
}

async function computeDnsmosSingle(audio) {
  await ensureDnsmosReady();
  const padded = new Float32Array(DNSMOS_SAMPLE_LENGTH);
  if (audio?.length) {
    padded.set(audio.subarray(0, Math.min(audio.length, DNSMOS_SAMPLE_LENGTH)));
  }
  const inputName = dnsmosSession.inputNames[0];
  const outputs = await dnsmosSession.run({
    [inputName]: new window.ort.Tensor("float32", padded, [1, DNSMOS_SAMPLE_LENGTH]),
  });
  const tensor = outputs[dnsmosSession.outputNames[0]];
  return mapDnsmosScores(tensor.data);
}

async function computeDesktopDnsmosFromSpeech(speechSamples, options = {}) {
  if (!speechSamples?.length || speechSamples.length < DNSMOS_MIN_SAMPLES) return null;
  const positions = [0.15, 0.50, 0.85];
  const scores = [];

  for (let i = 0; i < positions.length; i += 1) {
    const center = Math.floor(speechSamples.length * positions[i]);
    const start = Math.max(0, center - Math.floor(DNSMOS_SAMPLE_LENGTH / 2));
    const end = Math.min(speechSamples.length, start + DNSMOS_SAMPLE_LENGTH);
    if (end - start < DNSMOS_MIN_SAMPLES) continue;
    if (options.progress) options.progress(i, positions.length);
    const score = await computeDnsmosSingle(speechSamples.subarray(start, end));
    if (score) scores.push(score);
  }

  if (!scores.length) return null;
  return {
    dnsmos_sig: Number((scores.reduce((sum, item) => sum + item.SIG, 0) / scores.length).toFixed(2)),
    dnsmos_bak: Number((scores.reduce((sum, item) => sum + item.BAK, 0) / scores.length).toFixed(2)),
    dnsmos_ovrl: Number((scores.reduce((sum, item) => sum + item.OVRL, 0) / scores.length).toFixed(2)),
  };
}

function withQualityLabels(quality) {
  if (!quality) return null;
  const result = { ...quality };
  if (result.dnsmos_ovrl !== undefined) {
    const ovrl = Number(result.dnsmos_ovrl);
    if (ovrl >= 4.0) result.dnsmos_label = "T\u1ed1t";
    else if (ovrl >= 3.0) result.dnsmos_label = "Kh\u00e1";
    else if (ovrl >= 2.0) result.dnsmos_label = "Trung b\u00ecnh";
    else result.dnsmos_label = "K\u00e9m";
  }
  if (result.asr_confidence !== undefined) {
    const conf = Number(result.asr_confidence);
    if (conf >= 0.85) result.confidence_label = "Xu\u1ea5t s\u1eafc";
    else if (conf >= 0.75) result.confidence_label = "T\u1ed1t";
    else if (conf >= 0.60) result.confidence_label = "Trung b\u00ecnh";
    else result.confidence_label = "K\u00e9m";
  }
  return result;
}
async function computeQualityInfoFromSpeech(speechSamples, asrConfidence, options = {}) {
  const started = performance.now();
  const quality = {};
  try {
    const dnsmos = await computeDesktopDnsmosFromSpeech(speechSamples, options);
    if (dnsmos) Object.assign(quality, dnsmos);
  } catch (error) {
    log(`DNSMOS quality failed: ${error.message}`);
    if (options.strictDnsmos) throw error;
  }
  if (Number.isFinite(asrConfidence)) {
    quality.asr_confidence = Number(asrConfidence.toFixed(4));
  }
  if (!Object.keys(quality).length) return null;
  const result = withQualityLabels(quality);
  result.elapsed = Number(((performance.now() - started) / 1000).toFixed(3));
  return result;
}

async function releaseOrtSession(session) {
  if (!session) return false;
  if (typeof session.release === "function") {
    await session.release();
    return true;
  }
  if (typeof session.dispose === "function") {
    await session.dispose();
    return true;
  }
  return false;
}

async function unloadVadModel() {
  const hadSession = Boolean(vadSession);
  await releaseOrtSession(vadSession).catch((error) => log(`Unload VAD failed: ${error.message}`));
  vadSession = null;
  vadExecutionProvider = "wasm";
  if (hadSession) log("Unloaded Silero VAD session.");
}

async function unloadPunctuationModel() {
  const hadSession = Boolean(punctuationSession || punctuationVocab);
  await releaseOrtSession(punctuationSession).catch((error) => log(`Unload punctuation failed: ${error.message}`));
  punctuationSession = null;
  punctuationVocab = null;
  punctuationInitPromise = null;
  punctuationExecutionProvider = "wasm";
  if (hadSession) log("Unloaded punctuation session.");
}

async function unloadCamppEmbeddingSessionOnly() {
  const hadSession = Boolean(camppSession);
  await releaseOrtSession(camppSession).catch((error) => log(`Unload CAM++ embedding failed: ${error.message}`));
  camppSession = null;
  camppInitPromise = null;
  camppExecutionProvider = "wasm";
  if (hadSession) log("Unloaded CAM++ embedding session.");
}

async function unloadPyannoteEmbeddingSessionOnly() {
  const hadSession = Boolean(pyannoteEmbeddingSession);
  await releaseOrtSession(pyannoteEmbeddingSession).catch((error) => log(`Unload Pyannote embedding failed: ${error.message}`));
  pyannoteEmbeddingSession = null;
  pyannoteCommunityInitPromise = null;
  pyannoteEmbeddingExecutionProvider = "wasm";
  if (hadSession) log("Unloaded Pyannote embedding encoder session.");
}

async function unloadDiarizationSegmentationSessionOnly() {
  const hadSession = Boolean(diarizationSession);
  await releaseOrtSession(diarizationSession).catch((error) => log(`Unload pyannote segmentation failed: ${error.message}`));
  diarizationSession = null;
  diarizationInitPromise = null;
  diarizationExecutionProvider = "wasm";
  if (hadSession) log("Unloaded pyannote segmentation session.");
}

async function unloadSpeakerModels() {
  const hadSession = Boolean(diarizationSession || camppSession || pyannoteEmbeddingSession || pyannoteCommunityAssets);
  await Promise.all([
    releaseOrtSession(diarizationSession),
    releaseOrtSession(camppSession),
    releaseOrtSession(pyannoteEmbeddingSession),
  ]).catch((error) => log(`Unload speaker models failed: ${error.message}`));
  diarizationSession = null;
  diarizationInitPromise = null;
  camppSession = null;
  camppInitPromise = null;
  camppMelBank = null;
  camppWindow = null;
  wespeakerMelBank = null;
  wespeakerWindow = null;
  fftTables = null;
  pyannoteEmbeddingSession = null;
  pyannoteCommunityInitPromise = null;
  pyannoteCommunityAssets = null;
  diarizationExecutionProvider = "wasm";
  camppExecutionProvider = "wasm";
  pyannoteEmbeddingExecutionProvider = "wasm";
  if (hadSession) log("Unloaded speaker diarization sessions/assets.");
}

async function unloadOverlapModel() {
  const hadSession = Boolean(overlapSession);
  await releaseOrtSession(overlapSession).catch((error) => log(`Unload overlap model failed: ${error.message}`));
  overlapSession = null;
  overlapInitPromise = null;
  overlapExecutionProvider = "wasm";
  if (hadSession) log("Unloaded overlap separation session.");
}

async function unloadQualityModel() {
  const hadSession = Boolean(dnsmosSession);
  await releaseOrtSession(dnsmosSession).catch((error) => log(`Unload quality model failed: ${error.message}`));
  dnsmosSession = null;
  dnsmosInitPromise = null;
  dnsmosExecutionProvider = "wasm";
  if (hadSession) log("Unloaded DNSMOS quality session.");
}

async function unloadAsrModel() {
  if (!asrWorker && !asrInitPromise && !asrLoadedConfigKey) return;
  resetAsrWorker("unloaded");
  log("Unloaded ASR worker/model.");
}

async function unloadModelsAfterStep(step, options = {}) {
  if (!options.saveRam) return;
  if (step === "vad") {
    await unloadVadModel();
  } else if (step === "asr") {
    await unloadAsrModel();
  } else if (step === "quality") {
    await unloadQualityModel();
  } else if (step === "punctuation") {
    await unloadPunctuationModel();
  } else if (step === "diarization") {
    await unloadSpeakerModels();
  } else if (step === "overlap") {
    await unloadOverlapModel();
    await unloadSpeakerModels();
    await unloadAsrModel();
  } else if (step === "all") {
    await unloadVadModel();
    await unloadPunctuationModel();
    await unloadOverlapModel();
    await unloadQualityModel();
    await unloadSpeakerModels();
    await unloadAsrModel();
  }
}

function requireNpzArray(npz, key) {
  const array = npz.get(key);
  if (!array?.data || !array?.shape) {
    throw new Error(`Pyannote prepared PLDA archive is missing ${key}.`);
  }
  return array;
}

function preparePyannotePlda(preparedNpz) {
  const mean1 = requireNpzArray(preparedNpz, "mean1");
  const mean2 = requireNpzArray(preparedNpz, "mean2");
  const lda = requireNpzArray(preparedNpz, "lda");
  const mu = requireNpzArray(preparedNpz, "mu");
  const pldaTr = requireNpzArray(preparedNpz, "plda_tr");
  const pldaPsi = requireNpzArray(preparedNpz, "plda_psi");

  if (mean1.data.length !== 256 || mean2.data.length !== 128 || mu.data.length !== 128 || pldaPsi.data.length !== 128) {
    throw new Error("Pyannote prepared PLDA vector shapes are invalid.");
  }
  if (lda.shape[0] !== 256 || lda.shape[1] !== 128 || pldaTr.shape[0] !== 128 || pldaTr.shape[1] !== 128) {
    throw new Error("Pyannote prepared PLDA matrix shapes are invalid.");
  }
  return {
    mean1: mean1.data,
    mean2: mean2.data,
    lda: lda.data,
    mu: mu.data,
    pldaTr: pldaTr.data,
    pldaPsi: pldaPsi.data,
  };
}

function diarizationStarts(totalSamples) {
  const starts = [];
  let start = 0;
  while (start < totalSamples) {
    starts.push(start);
    if (start + DIAR_CHUNK_SAMPLES >= totalSamples) break;
    start += DIAR_STEP_SAMPLES;
  }
  return starts;
}

function hzToMel(freq) {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

function melToHz(mel) {
  return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
}

function getCamppWindow() {
  if (camppWindow) return camppWindow;
  camppWindow = new Float32Array(CAMPP_FRAME_LENGTH);
  for (let i = 0; i < CAMPP_FRAME_LENGTH; i += 1) {
    const hann = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (CAMPP_FRAME_LENGTH - 1));
    camppWindow[i] = Math.pow(hann, 0.85);
  }
  return camppWindow;
}

function getCamppMelBank() {
  if (camppMelBank) return camppMelBank;

  const highFreq = VAD_SAMPLE_RATE / 2;
  const lowMel = hzToMel(CAMPP_LOW_FREQ);
  const highMel = hzToMel(highFreq);
  const melDelta = (highMel - lowMel) / (CAMPP_NUM_MEL_BINS + 1);
  const centers = new Float64Array(CAMPP_NUM_MEL_BINS + 2);
  for (let i = 0; i < centers.length; i += 1) {
    centers[i] = melToHz(lowMel + i * melDelta);
  }

  const bins = CAMPP_N_FFT / 2 + 1;
  camppMelBank = Array.from({ length: CAMPP_NUM_MEL_BINS }, () => new Float32Array(bins));
  for (let bin = 0; bin < bins; bin += 1) {
    const freq = (bin * VAD_SAMPLE_RATE) / CAMPP_N_FFT;
    for (let mel = 0; mel < CAMPP_NUM_MEL_BINS; mel += 1) {
      const left = centers[mel];
      const center = centers[mel + 1];
      const right = centers[mel + 2];
      let weight = 0;
      if (freq > left && freq <= center) {
        weight = (freq - left) / Math.max(center - left, 1e-12);
      } else if (freq > center && freq < right) {
        weight = (right - freq) / Math.max(right - center, 1e-12);
      }
      camppMelBank[mel][bin] = weight;
    }
  }
  return camppMelBank;
}

function getFftTables(size) {
  if (fftTables && fftTables.size === size) return fftTables;

  const levels = Math.log2(size);
  const reverse = new Uint16Array(size);
  for (let i = 0; i < size; i += 1) {
    let value = i;
    let result = 0;
    for (let bit = 0; bit < levels; bit += 1) {
      result = (result << 1) | (value & 1);
      value >>= 1;
    }
    reverse[i] = result;
  }

  const cos = new Float64Array(size / 2);
  const sin = new Float64Array(size / 2);
  for (let i = 0; i < size / 2; i += 1) {
    const angle = (-2.0 * Math.PI * i) / size;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }

  fftTables = { size, reverse, cos, sin };
  return fftTables;
}

function fftInPlace(real, imag) {
  const n = real.length;
  const tables = getFftTables(n);
  for (let i = 0; i < n; i += 1) {
    const j = tables.reverse[i];
    if (j <= i) continue;
    const tr = real[i];
    const ti = imag[i];
    real[i] = real[j];
    imag[i] = imag[j];
    real[j] = tr;
    imag[j] = ti;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const tableStep = n / size;
    for (let start = 0; start < n; start += size) {
      for (let j = 0; j < half; j += 1) {
        const k = j * tableStep;
        const wr = tables.cos[k];
        const wi = tables.sin[k];
        const even = start + j;
        const odd = even + half;
        const tr = wr * real[odd] - wi * imag[odd];
        const ti = wr * imag[odd] + wi * real[odd];
        real[odd] = real[even] - tr;
        imag[odd] = imag[even] - ti;
        real[even] += tr;
        imag[even] += ti;
      }
    }
  }
}

function computeCamppFbank(audio) {
  if (audio.length < CAMPP_FRAME_LENGTH) return [];

  const window = getCamppWindow();
  const melBank = getCamppMelBank();
  const numFrames = 1 + Math.floor((audio.length - CAMPP_FRAME_LENGTH) / CAMPP_FRAME_SHIFT);
  const features = new Array(numFrames);
  const means = new Float64Array(CAMPP_NUM_MEL_BINS);
  const frame = new Float32Array(CAMPP_FRAME_LENGTH);
  const real = new Float64Array(CAMPP_N_FFT);
  const imag = new Float64Array(CAMPP_N_FFT);
  const power = new Float64Array(CAMPP_N_FFT / 2 + 1);
  // Buffer reuse: flat array, tránh new Float32Array(80) mỗi frame
  const flatOut = new Float32Array(numFrames * CAMPP_NUM_MEL_BINS);

  for (let f = 0; f < numFrames; f += 1) {
    const audioStart = f * CAMPP_FRAME_SHIFT;
    let mean = 0;
    for (let i = 0; i < CAMPP_FRAME_LENGTH; i += 1) {
      mean += audio[audioStart + i] * 32768.0;
    }
    mean /= CAMPP_FRAME_LENGTH;

    for (let i = 0; i < CAMPP_FRAME_LENGTH; i += 1) {
      frame[i] = audio[audioStart + i] * 32768.0 - mean;
    }

    real.fill(0);
    imag.fill(0);
    const context = audioStart > 0 ? audio[audioStart - 1] * 32768.0 : 0;
    real[0] = (frame[0] - CAMPP_PREEMPHASIS * context) * window[0];
    for (let i = 1; i < CAMPP_FRAME_LENGTH; i += 1) {
      real[i] = (frame[i] - CAMPP_PREEMPHASIS * frame[i - 1]) * window[i];
    }

    fftInPlace(real, imag);
    for (let i = 0; i < power.length; i += 1) {
      power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    const rowOffset = f * CAMPP_NUM_MEL_BINS;
    for (let mel = 0; mel < CAMPP_NUM_MEL_BINS; mel += 1) {
      const weights = melBank[mel];
      let energy = 0;
      for (let bin = 0; bin < weights.length; bin += 1) {
        energy += power[bin] * weights[bin];
      }
      const value = Math.log(Math.max(energy, CAMPP_ENERGY_FLOOR));
      flatOut[rowOffset + mel] = value;
      means[mel] += value;
    }
    features[f] = flatOut.subarray(rowOffset, rowOffset + CAMPP_NUM_MEL_BINS);
  }

  for (let mel = 0; mel < CAMPP_NUM_MEL_BINS; mel += 1) {
    means[mel] /= numFrames;
  }
  for (const row of features) {
    for (let mel = 0; mel < CAMPP_NUM_MEL_BINS; mel += 1) {
      row[mel] -= means[mel];
    }
  }
  return features;
}

function getWespeakerWindow() {
  if (wespeakerWindow) return wespeakerWindow;
  wespeakerWindow = new Float32Array(WESPEAKER_FRAME_LENGTH);
  for (let i = 0; i < WESPEAKER_FRAME_LENGTH; i += 1) {
    wespeakerWindow[i] = 0.54 - 0.46 * Math.cos((2.0 * Math.PI * i) / (WESPEAKER_FRAME_LENGTH - 1));
  }
  return wespeakerWindow;
}

function getWespeakerMelBank() {
  if (wespeakerMelBank) return wespeakerMelBank;
  const highFreq = VAD_SAMPLE_RATE / 2;
  const lowMel = hzToMel(WESPEAKER_LOW_FREQ);
  const highMel = hzToMel(highFreq);
  const melDelta = (highMel - lowMel) / (WESPEAKER_NUM_MEL_BINS + 1);
  const centers = new Float64Array(WESPEAKER_NUM_MEL_BINS + 2);
  for (let i = 0; i < centers.length; i += 1) {
    centers[i] = melToHz(lowMel + i * melDelta);
  }

  const bins = WESPEAKER_N_FFT / 2 + 1;
  wespeakerMelBank = Array.from({ length: WESPEAKER_NUM_MEL_BINS }, () => new Float32Array(bins));
  for (let bin = 0; bin < bins; bin += 1) {
    const freq = (bin * VAD_SAMPLE_RATE) / WESPEAKER_N_FFT;
    for (let mel = 0; mel < WESPEAKER_NUM_MEL_BINS; mel += 1) {
      const left = centers[mel];
      const center = centers[mel + 1];
      const right = centers[mel + 2];
      let weight = 0;
      if (freq > left && freq <= center) {
        weight = (freq - left) / Math.max(center - left, 1e-12);
      } else if (freq > center && freq < right) {
        weight = (right - freq) / Math.max(right - center, 1e-12);
      }
      wespeakerMelBank[mel][bin] = weight;
    }
  }
  return wespeakerMelBank;
}

function computeWespeakerFbank(audio) {
  if (audio.length < WESPEAKER_FRAME_LENGTH) return [];
  const window = getWespeakerWindow();
  const melBank = getWespeakerMelBank();
  const numFrames = 1 + Math.floor((audio.length - WESPEAKER_FRAME_LENGTH) / WESPEAKER_FRAME_SHIFT);
  const features = new Array(numFrames);
  const means = new Float64Array(WESPEAKER_NUM_MEL_BINS);
  const frame = new Float32Array(WESPEAKER_FRAME_LENGTH);
  const real = new Float64Array(WESPEAKER_N_FFT);
  const imag = new Float64Array(WESPEAKER_N_FFT);
  const power = new Float64Array(WESPEAKER_N_FFT / 2 + 1);
  // Buffer reuse: flat array, tránh new Float32Array(80) mỗi frame
  const flatOut = new Float32Array(numFrames * WESPEAKER_NUM_MEL_BINS);

  for (let f = 0; f < numFrames; f += 1) {
    const audioStart = f * WESPEAKER_FRAME_SHIFT;
    let mean = 0;
    for (let i = 0; i < WESPEAKER_FRAME_LENGTH; i += 1) {
      mean += audio[audioStart + i] * 32768.0;
    }
    mean /= WESPEAKER_FRAME_LENGTH;

    for (let i = 0; i < WESPEAKER_FRAME_LENGTH; i += 1) {
      frame[i] = audio[audioStart + i] * 32768.0 - mean;
    }

    real.fill(0);
    imag.fill(0);
    const context = audioStart > 0 ? audio[audioStart - 1] * 32768.0 : 0;
    real[0] = (frame[0] - WESPEAKER_PREEMPHASIS * frame[0]) * window[0];
    for (let i = 1; i < WESPEAKER_FRAME_LENGTH; i += 1) {
      real[i] = (frame[i] - WESPEAKER_PREEMPHASIS * frame[i - 1]) * window[i];
    }

    fftInPlace(real, imag);
    for (let i = 0; i < power.length; i += 1) {
      power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    const rowOffset = f * WESPEAKER_NUM_MEL_BINS;
    for (let mel = 0; mel < WESPEAKER_NUM_MEL_BINS; mel += 1) {
      const weights = melBank[mel];
      let energy = 0;
      for (let bin = 0; bin < weights.length; bin += 1) {
        energy += power[bin] * weights[bin];
      }
      const value = Math.log(Math.max(energy, 1e-10));
      flatOut[rowOffset + mel] = value;
      means[mel] += value;
    }
    features[f] = flatOut.subarray(rowOffset, rowOffset + WESPEAKER_NUM_MEL_BINS);
  }

  for (let mel = 0; mel < WESPEAKER_NUM_MEL_BINS; mel += 1) {
    means[mel] /= numFrames;
  }
  for (const row of features) {
    for (let mel = 0; mel < WESPEAKER_NUM_MEL_BINS; mel += 1) {
      row[mel] -= means[mel];
    }
  }
  return features;
}

function l2Normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sum) || 1;
  const output = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    output[i] = vector[i] / norm;
  }
  return output;
}

function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function decodeDiarizationLogits(data, base, numFrames, numClasses) {
  const frames = new Array(numFrames);
  for (let frame = 0; frame < numFrames; frame += 1) {
    let bestClass = 0;
    let bestValue = -Infinity;
    const offset = base + frame * numClasses;
    for (let cls = 0; cls < numClasses; cls += 1) {
      const value = data[offset + cls];
      if (value > bestValue) {
        bestValue = value;
        bestClass = cls;
      }
    }
    frames[frame] = DIAR_POWERSET[bestClass].slice();
  }
  return frames;
}

function renderDiarization(segments, details = "") {
  // Raw speaker-turn diagnostics are hidden from the normal offline PWA UI.
}

function renderOverlapSegments(segments, details = "") {
  // Overlap diagnostics are hidden from the normal offline PWA UI.
}

function normalizedColor(value, fallback) {
  return /^#[0-9a-f]{6}$/i.test(value || "") ? value : fallback;
}

function defaultSpeakerName(speaker) {
  return `Người nói ${Number(speaker) + 1}`;
}

function normalizeSpeakerId(value) {
  const id = Number(value);
  return Number.isInteger(id) && id >= 0 ? id : 0;
}

function normalizeDiarizationSegments(segments = []) {
  return segments
    .map((segment, index) => ({
      id: `raw-${index}`,
      start: Number(segment.start),
      end: Number(segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }))
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
}

function normalizeEditorOverlapSegments(segments = []) {
  return segments
    .map((segment, index) => {
      const rawSpeaker = segment.speaker_id ?? segment.speaker;
      const numericSpeaker = typeof rawSpeaker === "string" && /^speaker\s+\d+$/i.test(rawSpeaker)
        ? Number(rawSpeaker.replace(/\D+/g, "")) - 1
        : rawSpeaker;
      return {
        id: `overlap-${index}`,
        start: Number(segment.start ?? segment.start_time),
        end: Number(segment.end ?? segment.end_time),
        speaker: normalizeSpeakerId(numericSpeaker),
        text: segment.text || "",
        raw_words: Array.isArray(segment.raw_words) ? segment.raw_words : [],
        overlap: true,
      };
    })
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
}

function speakerMetaFor(speaker) {
  const id = normalizeSpeakerId(speaker);
  if (!editorState) {
    return { name: defaultSpeakerName(id), color: SPEAKER_COLORS[id % SPEAKER_COLORS.length] };
  }
  if (!editorState.speakers[id]) {
    editorState.speakers[id] = {
      name: defaultSpeakerName(id),
      color: SPEAKER_COLORS[id % SPEAKER_COLORS.length],
    };
  }
  return editorState.speakers[id];
}

function collectEditorSpeakerIds(segments, rawSegments, overlapSegments = []) {
  const ids = new Set();
  for (const segment of segments || []) ids.add(normalizeSpeakerId(segment.speaker));
  for (const segment of rawSegments || []) ids.add(normalizeSpeakerId(segment.speaker));
  for (const segment of overlapSegments || []) ids.add(normalizeSpeakerId(segment.speaker));
  if (!ids.size) ids.add(0);
  return [...ids].sort((a, b) => a - b);
}

function syncEditorSpeakers(existing = {}) {
  if (!editorState) return;
  const next = {};
  for (const id of collectEditorSpeakerIds(editorState.segments, editorState.rawSpeakerSegments, editorState.overlapSegments)) {
    const previous = existing[id] || editorState.speakers?.[id];
    next[id] = {
      name: previous?.name || defaultSpeakerName(id),
      color: normalizedColor(previous?.color, SPEAKER_COLORS[id % SPEAKER_COLORS.length]),
    };
  }
  editorState.speakers = next;
}

function editorTokenize(text) {
  return (text || "").trim().split(/\s+/).filter(Boolean);
}

function splitEditorSentenceGroups(tokens, maxWords = 32) {
  const groups = [];
  let current = [];
  for (const token of tokens) {
    current.push(token);
    if (/[.!?:;]+["')\]]*$/.test(token) || current.length >= maxWords) {
      groups.push(current);
      current = [];
    }
  }
  if (current.length) groups.push(current);
  return groups;
}

function buildEditorChunkRanges(asr, duration) {
  const chunks = asr?.chunks?.length ? asr.chunks : (asr?.results || []);
  const ranges = [];
  let previousEnd = 0;

  for (const chunk of chunks) {
    const chunkStart = Number(chunk.start);
    const chunkEnd = Number(chunk.end);
    if (!Number.isFinite(chunkStart) || !Number.isFinite(chunkEnd) || chunkEnd <= chunkStart) continue;

    const effectiveStart = Math.max(chunkStart, previousEnd);
    const effectiveEnd = Math.max(effectiveStart, chunkEnd);
    previousEnd = Math.max(previousEnd, chunkEnd);
    if (effectiveEnd <= effectiveStart) continue;

    const mapped = timelineRangeToSourceSeconds(asr?.timeline, effectiveStart, effectiveEnd);
    const start = Math.max(0, Math.min(duration, mapped.start));
    const end = Math.max(start + 0.01, Math.min(duration, mapped.end));
    ranges.push({
      start,
      end,
      weight: Math.max(1, effectiveEnd - effectiveStart),
    });
  }

  if (!ranges.length) {
    ranges.push({ start: 0, end: Math.max(0.01, duration || 0.01), weight: Math.max(1, Math.round((duration || 1) * VAD_SAMPLE_RATE)) });
  }

  return ranges;
}

function dominantSpeakerForRange(start, end, rawSegments) {
  if (!rawSegments?.length) return 0;
  const weights = new Map();
  for (const segment of rawSegments) {
    const overlap = Math.min(end, segment.end) - Math.max(start, segment.start);
    if (overlap > 0) {
      const speaker = normalizeSpeakerId(segment.speaker);
      weights.set(speaker, (weights.get(speaker) || 0) + overlap);
    }
  }
  if (weights.size) {
    return [...weights.entries()].sort((a, b) => b[1] - a[1])[0][0];
  }

  const mid = (start + end) / 2;
  let nearest = rawSegments[0];
  let distance = Infinity;
  for (const segment of rawSegments) {
    const candidate = mid < segment.start ? segment.start - mid : (mid > segment.end ? mid - segment.end : 0);
    if (candidate < distance) {
      distance = candidate;
      nearest = segment;
    }
  }
  return normalizeSpeakerId(nearest?.speaker);
}

function splitEditorRangeBySpeakerTurns(range, rawSegments) {
  if (!rawSegments?.length || range.end <= range.start) {
    return [{ ...range, speaker: dominantSpeakerForRange(range.start, range.end, rawSegments) }];
  }

  const intersections = rawSegments
    .map((segment) => ({
      start: Math.max(range.start, segment.start),
      end: Math.min(range.end, segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }))
    .filter((segment) => segment.end - segment.start > 0.03)
    .sort((a, b) => a.start - b.start);

  if (!intersections.length) {
    return [{ ...range, speaker: dominantSpeakerForRange(range.start, range.end, rawSegments) }];
  }

  const pieces = [];
  let cursor = range.start;
  const addPiece = (start, end, speaker) => {
    if (end - start <= 0.03) return;
    const durationRatio = (end - start) / Math.max(0.01, range.end - range.start);
    pieces.push({
      start,
      end,
      speaker: normalizeSpeakerId(speaker),
      weight: Math.max(1, range.weight * durationRatio),
    });
  };

  for (const segment of intersections) {
    if (segment.start > cursor + 0.03) {
      addPiece(cursor, segment.start, dominantSpeakerForRange(cursor, segment.start, rawSegments));
    }
    addPiece(Math.max(cursor, segment.start), segment.end, segment.speaker);
    cursor = Math.max(cursor, segment.end);
  }
  if (cursor < range.end - 0.03) {
    addPiece(cursor, range.end, dominantSpeakerForRange(cursor, range.end, rawSegments));
  }

  const merged = [];
  for (const piece of pieces) {
    const previous = merged[merged.length - 1];
    if (previous && previous.speaker === piece.speaker && piece.start - previous.end <= 0.05) {
      previous.end = piece.end;
      previous.weight += piece.weight;
    } else {
      merged.push(piece);
    }
  }
  return merged.length ? merged : [{ ...range, speaker: dominantSpeakerForRange(range.start, range.end, rawSegments) }];
}

function buildEditorSpeakerRanges(ranges, rawSegments) {
  return ranges.flatMap((range) => splitEditorRangeBySpeakerTurns(range, rawSegments));
}

function cleanAsrRawWordForJson(word) {
  const output = {
    text: word.text || "",
    start: Number(Number(word.start || 0).toFixed(7)),
    end: Number(Number(word.end || 0).toFixed(7)),
  };
  const optional = [
    "local_start", "local_end", "timeline_start", "timeline_end",
    "prob", "tsallis_max", "margin_min", "entropy_norm", "_conf",
    "_suspect_level", "_disagree", "gap_before_ms", "gap_after_ms",
  ];
  for (const key of optional) {
    const value = word[key];
    if (value === undefined || value === null || value === "") continue;
    if (typeof value === "number") {
      output[key] = Number(value.toFixed(key.endsWith("_ms") ? 0 : 4));
    } else {
      output[key] = value;
    }
  }
  return output;
}

function rawWordsForEditorRange(words = [], start = 0, end = 0) {
  if (!Array.isArray(words) || !words.length) return [];
  return words
    .filter((word) => Number(word.end) > start && Number(word.start) < end)
    .map(cleanAsrRawWordForJson);
}

function cleanEditorAlignmentWord(word) {
  const start = Number(word?.start);
  const end = Number(word?.end);
  const text = String(word?.text || "").trim();
  if (!text || !Number.isFinite(start) || !Number.isFinite(end)) return null;
  return {
    ...word,
    text,
    start,
    end: Math.max(start + 0.01, end),
  };
}

function normalizeEditorAlignmentWord(word) {
  return String(word || "")
    .toLocaleLowerCase("vi-VN")
    .trim()
    .replace(/[^\p{L}\p{N}_\s]/gu, "")
    .replace(/\s+/g, "");
}

function editorAlignmentWordsMatch(a, b) {
  if (!a || !b) return false;
  if (a === b) return true;
  return a.length > 2 && b.length > 2 && (a.includes(b) || b.includes(a));
}

function findEditorWordSequenceMatch(asrWords, targetWords, startIndex, maxLookAhead = 50) {
  if (!targetWords?.length || !asrWords?.length) return [null, null];
  const firstTarget = normalizeEditorAlignmentWord(targetWords[0]);
  if (!firstTarget) return [null, null];

  const endSearch = Math.min(startIndex + maxLookAhead, asrWords.length);
  let bestMatch = null;
  let bestScore = 0;

  for (let i = startIndex; i < endSearch; i += 1) {
    const asrWord = normalizeEditorAlignmentWord(asrWords[i]?.text);
    if (!editorAlignmentWordsMatch(asrWord, firstTarget)) continue;

    let matchedCount = 1;
    let lastMatchedIndex = i;
    let asrOffset = 0;

    for (let j = 1; j < targetWords.length; j += 1) {
      const targetWord = normalizeEditorAlignmentWord(targetWords[j]);
      if (!targetWord) {
        matchedCount += 1;
        continue;
      }

      const asrIndex = i + j + asrOffset;
      if (asrIndex >= asrWords.length) break;

      const candidate = normalizeEditorAlignmentWord(asrWords[asrIndex]?.text);
      if (editorAlignmentWordsMatch(candidate, targetWord)) {
        matchedCount += 1;
        lastMatchedIndex = asrIndex;
        continue;
      }

      if (asrIndex + 1 < asrWords.length) {
        const nextCandidate = normalizeEditorAlignmentWord(asrWords[asrIndex + 1]?.text);
        if (editorAlignmentWordsMatch(nextCandidate, targetWord)) {
          matchedCount += 1;
          lastMatchedIndex = asrIndex + 1;
          asrOffset += 1;
          continue;
        }
      }
      break;
    }

    const score = matchedCount / Math.max(1, targetWords.length);
    if (score > bestScore) {
      bestScore = score;
      bestMatch = [i, lastMatchedIndex];
    }
    if (score >= 0.95) break;
  }

  return bestScore >= 0.7 && bestMatch ? bestMatch : [null, null];
}

function intervalOverlap(startA, endA, startB, endB) {
  return Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
}

function effectiveWordInterval(word) {
  let start = Number(word?.start ?? 0);
  if (!Number.isFinite(start)) start = 0;
  let end = Number(word?.end ?? start);
  if (!Number.isFinite(end) || end === 0) end = start;
  if (end < start) {
    const tmp = start;
    start = end;
    end = tmp;
  }
  end = Math.min(end, start + WORD_ASSIGN_MAX_DURATION_SECONDS);
  if (end <= start) end = start + WORD_ASSIGN_MAX_DURATION_SECONDS;
  return { start, end };
}

function speakerForWordByTime(word, rawSegments, fallbackSpeaker = 0) {
  const segments = (rawSegments || [])
    .map((segment) => ({
      ...segment,
      start: Number(segment.start),
      end: Number(segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }))
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  if (!segments.length) return normalizeSpeakerId(fallbackSpeaker);

  const { start, end } = effectiveWordInterval(word);
  const mid = (start + end) / 2;
  let best = null;
  let bestOverlap = 0;
  let bestCenterDistance = Infinity;
  for (const segment of segments) {
    const overlap = intervalOverlap(start, end, segment.start, segment.end);
    if (overlap <= 0) continue;
    const centerDistance = Math.abs(((segment.start + segment.end) / 2) - mid);
    if (overlap > bestOverlap || (overlap === bestOverlap && centerDistance < bestCenterDistance)) {
      best = segment;
      bestOverlap = overlap;
      bestCenterDistance = centerDistance;
    }
  }
  if (best) return normalizeSpeakerId(best.speaker);

  let previous = null;
  let next = null;
  for (const segment of segments) {
    if (segment.end <= mid) {
      if (!previous || segment.end > previous.end) previous = segment;
    } else if (segment.start >= mid) {
      if (!next || segment.start < next.start) next = segment;
    }
  }
  if (previous && next) {
    const prevDistance = mid - previous.end;
    const nextDistance = next.start - mid;
    return normalizeSpeakerId(prevDistance <= nextDistance ? previous.speaker : next.speaker);
  }
  if (previous) return normalizeSpeakerId(previous.speaker);
  if (next) return normalizeSpeakerId(next.speaker);
  return normalizeSpeakerId(fallbackSpeaker);
}

function dominantSpeakerForWords(words, rawSegments, fallbackSpeaker = 0) {
  const segments = (rawSegments || [])
    .map((segment) => ({
      ...segment,
      start: Number(segment.start),
      end: Number(segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }))
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start);
  if (!Array.isArray(words) || !words.length || !segments.length) {
    return normalizeSpeakerId(fallbackSpeaker);
  }
  const weights = new Map();
  for (const word of words) {
    const { start, end } = effectiveWordInterval(word);
    for (const segment of segments) {
      const overlap = intervalOverlap(start, end, segment.start, segment.end);
      if (overlap <= 0) continue;
      weights.set(segment.speaker, (weights.get(segment.speaker) || 0) + overlap);
    }
  }
  if (weights.size) {
    return [...weights.entries()].sort((a, b) => b[1] - a[1])[0][0];
  }
  return speakerForWordByTime(words[0], segments, fallbackSpeaker);
}

function buildEditorWordSpeakerMap(words, rawSegments) {
  const sorted = (rawSegments || [])
    .map((segment) => ({
      ...segment,
      start: Number(segment.start),
      end: Number(segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }))
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const speakers = [];
  let previous = sorted.length ? normalizeSpeakerId(sorted[0].speaker) : 0;
  for (const word of words || []) {
    previous = speakerForWordByTime(word, sorted, previous);
    speakers.push(previous);
  }
  return speakers;
}

function pushEditorAlignedSegment(segments, text, words, speaker) {
  const cleanWords = (words || []).map(cleanEditorAlignmentWord).filter(Boolean);
  const body = String(text || "").trim();
  if (!body || !cleanWords.length) return;
  segments.push({
    id: `seg-${segments.length}`,
    start: cleanWords[0].start,
    end: Math.max(cleanWords[cleanWords.length - 1].end, cleanWords[0].start + 0.01),
    speaker: normalizeSpeakerId(speaker),
    text: body,
    raw_words: cleanWords.map(cleanAsrRawWordForJson),
  });
}

function alignEditorTextToAsrWords(finalText, asrWords, rawSegments = []) {
  const allWords = (asrWords || []).map(cleanEditorAlignmentWord).filter(Boolean);
  if (!finalText || !allWords.length) return [];

  const sentences = String(finalText)
    .split(/(?<=[.?!])\s+/u)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const wordSpeakers = buildEditorWordSpeakerMap(allWords, rawSegments);
  const segments = [];
  let currentWordIndex = 0;

  for (const sentence of sentences.length ? sentences : [finalText]) {
    const sentenceWords = sentence.split(/\s+/).filter(Boolean);
    if (!sentenceWords.length) continue;
    const targetWords = sentenceWords
      .map(normalizeEditorAlignmentWord)
      .filter(Boolean);
    if (!targetWords.length) continue;

    let [matchStart, matchEnd] = findEditorWordSequenceMatch(allWords, targetWords, currentWordIndex);
    if (matchStart === null) {
      const firstWord = targetWords[0];
      let tempIndex = currentWordIndex;
      while (tempIndex < allWords.length) {
        const candidate = normalizeEditorAlignmentWord(allWords[tempIndex]?.text);
        if (editorAlignmentWordsMatch(candidate, firstWord)) break;
        tempIndex += 1;
      }
      if (tempIndex < allWords.length) {
        matchStart = tempIndex;
        matchEnd = Math.min(tempIndex + targetWords.length - 1, allWords.length - 1);
      } else {
        matchStart = Math.min(currentWordIndex, allWords.length - 1);
        matchEnd = Math.min(matchStart + targetWords.length - 1, allWords.length - 1);
      }
    }

    if (matchStart === null || matchEnd === null || matchEnd < matchStart) continue;
    const matchedWords = allWords.slice(matchStart, matchEnd + 1);
    if (!matchedWords.length) continue;

    const subGroups = [];
    let groupSpeaker = wordSpeakers[matchStart] ?? dominantSpeakerForRange(matchedWords[0].start, matchedWords[0].end, rawSegments);
    let groupStart = 0;
    for (let offset = 0; offset < matchedWords.length; offset += 1) {
      const speaker = wordSpeakers[matchStart + offset] ?? groupSpeaker;
      if (speaker !== groupSpeaker) {
        subGroups.push({ speaker: groupSpeaker, start: groupStart, end: offset });
        groupSpeaker = speaker;
        groupStart = offset;
      }
    }
    subGroups.push({ speaker: groupSpeaker, start: groupStart, end: matchedWords.length });

    if (subGroups.length === 1) {
      pushEditorAlignedSegment(segments, sentence, matchedWords, subGroups[0].speaker);
    } else {
      for (const group of subGroups) {
        const groupWords = matchedWords.slice(group.start, group.end);
        if (!groupWords.length) continue;
        const textStart = Math.floor((group.start / matchedWords.length) * sentenceWords.length);
        let textEnd = Math.floor((group.end / matchedWords.length) * sentenceWords.length);
        if (group.end === matchedWords.length) textEnd = sentenceWords.length;
        if (textEnd <= textStart) textEnd = Math.min(sentenceWords.length, textStart + 1);
        pushEditorAlignedSegment(
          segments,
          sentenceWords.slice(textStart, textEnd).join(" "),
          groupWords,
          group.speaker
        );
      }
    }
    currentWordIndex = matchEnd + 1;
  }

  return segments;
}

function normalizeEditorSegmentBoundaries(segments = [], duration = 0) {
  const maxDuration = Number(duration) || Number.POSITIVE_INFINITY;
  const normalized = (segments || [])
    .map((segment, index) => {
      const rawWords = Array.isArray(segment.raw_words)
        ? segment.raw_words.map(cleanEditorAlignmentWord).filter(Boolean)
        : [];
      const wordStart = rawWords.length ? rawWords[0].start : Number(segment.start);
      const wordEnd = rawWords.length ? rawWords[rawWords.length - 1].end : Number(segment.end);
      const start = Math.max(0, Math.min(maxDuration, Number.isFinite(wordStart) ? wordStart : 0));
      const end = Math.max(start + 0.01, Math.min(maxDuration, Number.isFinite(wordEnd) ? wordEnd : start + 1));
      return {
        ...segment,
        id: segment.id || `seg-${index}`,
        start: Number(start.toFixed(3)),
        end: Number(end.toFixed(3)),
        speaker: normalizeSpeakerId(segment.speaker ?? segment.speaker_id),
        raw_words: rawWords.map(cleanAsrRawWordForJson),
      };
    })
    .filter((segment) => segment.text && Number.isFinite(segment.start) && Number.isFinite(segment.end));

  for (let i = 0; i < normalized.length - 1; i += 1) {
    const nextStart = normalized[i + 1].start;
    if (normalized[i].end > nextStart) {
      normalized[i].end = Number(Math.max(normalized[i].start + 0.01, nextStart).toFixed(3));
    }
  }
  return normalized;
}

function splitEditorLongSegmentsDesktopStyle(segments = [], maxDuration = 12.0) {
  const result = [];

  const appendPart = (source, text, start, end, rawWords) => {
    const body = String(text || "").trim();
    if (!body) return;
    const cleanRaw = (rawWords || []).map(cleanEditorAlignmentWord).filter(Boolean);
    result.push({
      ...source,
      id: `seg-${result.length}`,
      text: body,
      start: Number(start.toFixed(3)),
      end: Number(Math.max(start + 0.01, end).toFixed(3)),
      raw_words: cleanRaw.map(cleanAsrRawWordForJson),
    });
  };

  const processSubText = (source, subText, subStart, subEnd, subRawWords) => {
    const subDuration = subEnd - subStart;
    const text = String(subText || "").trim();
    if (subDuration <= maxDuration || !text) {
      appendPart(source, text, subStart, subEnd, subRawWords);
      return;
    }

    const words = text.split(/\s+/).filter(Boolean);
    const numParts = Math.max(2, Math.ceil(subDuration / maxDuration));
    if (words.length < numParts) {
      appendPart(source, text, subStart, subEnd, subRawWords);
      return;
    }

    const wordsPerPart = Math.floor(words.length / numParts);
    const remainder = words.length % numParts;
    const rawWords = (subRawWords || []).map(cleanEditorAlignmentWord).filter(Boolean);
    const rawPerPart = rawWords.length ? Math.floor(rawWords.length / numParts) : 0;
    const rawRemainder = rawWords.length ? rawWords.length % numParts : 0;
    const timePerWord = subDuration / Math.max(1, words.length);
    let wordIndex = 0;
    let rawIndex = 0;

    for (let partIndex = 0; partIndex < numParts; partIndex += 1) {
      const currentWords = wordsPerPart + (partIndex < remainder ? 1 : 0);
      if (!currentWords) continue;
      const partWords = words.slice(wordIndex, wordIndex + currentWords);
      let partStart = subStart + wordIndex * timePerWord;
      let partEnd = subStart + (wordIndex + currentWords) * timePerWord;
      let partRawWords = [];

      if (rawWords.length) {
        const currentRaw = rawPerPart + (partIndex < rawRemainder ? 1 : 0);
        if (currentRaw > 0 && rawIndex < rawWords.length) {
          const lastRawIndex = Math.min(rawIndex + currentRaw - 1, rawWords.length - 1);
          partRawWords = rawWords.slice(rawIndex, lastRawIndex + 1);
          partStart = partRawWords[0].start;
          partEnd = partRawWords[partRawWords.length - 1].end;
          rawIndex += currentRaw;
        }
      }

      partStart = Math.max(subStart, partStart);
      partEnd = Math.min(subEnd, Math.max(partStart + 0.01, partEnd));
      const previous = result[result.length - 1];
      if (previous && partStart < previous.end && previous.speaker === source.speaker) {
        partStart = previous.end;
        partEnd = Math.max(partStart + 0.01, partEnd);
      }
      appendPart(source, partWords.join(" "), partStart, partEnd, partRawWords);
      wordIndex += currentWords;
    }
  };

  for (const segment of segments || []) {
    const duration = Number(segment.end) - Number(segment.start);
    const text = String(segment.text || "").trim();
    if (duration <= maxDuration || !text) {
      result.push({ ...segment, id: `seg-${result.length}` });
      continue;
    }

    if (text.includes(",")) {
      const commaParts = text.split(/(?<=,)\s+/u).map((part) => part.trim()).filter(Boolean);
      if (commaParts.length > 1) {
        const allWords = text.split(/\s+/).filter(Boolean);
        const rawWords = Array.isArray(segment.raw_words) ? segment.raw_words : [];
        const timePerWord = duration / Math.max(1, allWords.length);
        let wordOffset = 0;
        let rawOffset = 0;
        for (const part of commaParts) {
          const count = part.split(/\s+/).filter(Boolean).length;
          const partRaw = rawWords.slice(rawOffset, rawOffset + count);
          const partStart = partRaw.length ? Number(partRaw[0].start) : segment.start + wordOffset * timePerWord;
          const partEnd = partRaw.length ? Number(partRaw[partRaw.length - 1].end) : segment.start + (wordOffset + count) * timePerWord;
          processSubText(segment, part, partStart, partEnd, partRaw);
          wordOffset += count;
          rawOffset += count;
        }
        continue;
      }
    }

    processSubText(segment, text, segment.start, segment.end, segment.raw_words || []);
  }

  return result.map((segment, index) => ({ ...segment, id: `seg-${index}` }));
}

function mapEditorWordTimingFromTimeline(word, timeline) {
  const start = Number(word?.start);
  const end = Number(word?.end);
  if (!timeline || !Number.isFinite(start) || !Number.isFinite(end)) return word;
  const mapped = timelineRangeToSourceSeconds(
    timeline,
    Math.round(Math.max(0, start) * VAD_SAMPLE_RATE),
    Math.round(Math.max(start, end) * VAD_SAMPLE_RATE)
  );
  return {
    ...word,
    timeline_start: word.timeline_start ?? start,
    timeline_end: word.timeline_end ?? end,
    start: Number(mapped.start.toFixed(3)),
    end: Number(Math.max(mapped.start + 0.01, mapped.end).toFixed(3)),
  };
}

function maybeMapEditorSegmentsFromTimeline(segments = [], asr = null, duration = 0) {
  const timeline = asr?.timeline;
  if (!timeline?.totalSamples || !segments.length) return segments;
  const speechDuration = timeline.totalSamples / VAD_SAMPLE_RATE;
  const maxEnd = segments.reduce((max, segment) => Math.max(max, Number(segment.end) || 0), 0);
  const sourceDuration = Number(duration) || maxEnd;
  const compressedByVad = sourceDuration - speechDuration > 1.5 &&
    maxEnd <= speechDuration + 1.0 &&
    sourceDuration - maxEnd > 1.5;
  if (!compressedByVad) return segments;

  log(
    `[Editor] Remapped transcript segment timing from ASR speech timeline to source audio time ` +
    `(speech=${speechDuration.toFixed(2)}s, source=${sourceDuration.toFixed(2)}s).`
  );
  return segments.map((segment) => {
    const mapped = timelineRangeToSourceSeconds(
      timeline,
      Math.round(Math.max(0, Number(segment.start) || 0) * VAD_SAMPLE_RATE),
      Math.round(Math.max(Number(segment.start) || 0, Number(segment.end) || 0) * VAD_SAMPLE_RATE)
    );
    return {
      ...segment,
      timeline_start: segment.timeline_start ?? segment.start,
      timeline_end: segment.timeline_end ?? segment.end,
      start: Number(mapped.start.toFixed(3)),
      end: Number(Math.max(mapped.start + 0.01, mapped.end).toFixed(3)),
      raw_words: Array.isArray(segment.raw_words)
        ? segment.raw_words.map((word) => mapEditorWordTimingFromTimeline(word, timeline))
        : segment.raw_words,
    };
  });
}

function buildEditorTranscriptSegments(finalText, asr, rawSegments, duration) {
  const tokens = editorTokenize(finalText);
  if (!tokens.length) return [];
  const asrWords = Array.isArray(asr?.words) ? asr.words : [];

  const aligned = alignEditorTextToAsrWords(finalText, asrWords, rawSegments || []);
  if (aligned.length) {
    return splitEditorLongSegmentsDesktopStyle(
      normalizeEditorSegmentBoundaries(aligned, duration),
      12.0
    );
  }

  // === Fallback: proportional weight (original logic, used when no word timestamps) ===
  const ranges = buildEditorSpeakerRanges(buildEditorChunkRanges(asr, duration), rawSegments);
  const totalWeight = ranges.reduce((sum, range) => sum + range.weight, 0) || 1;
  const segments = [];
  let cursor = 0;
  let cumulativeWeight = 0;

  ranges.forEach((range, rangeIndex) => {
    cumulativeWeight += range.weight;
    const targetEnd = rangeIndex === ranges.length - 1
      ? tokens.length
      : Math.max(cursor, Math.round((cumulativeWeight / totalWeight) * tokens.length));
    const rangeTokens = tokens.slice(cursor, targetEnd);
    cursor = targetEnd;
    if (!rangeTokens.length) return;

    const groups = splitEditorSentenceGroups(rangeTokens);
    let groupCursor = 0;
    for (const group of groups) {
      const groupStartRatio = groupCursor / Math.max(1, rangeTokens.length);
      groupCursor += group.length;
      const groupEndRatio = groupCursor / Math.max(1, rangeTokens.length);
      const start = range.start + (range.end - range.start) * groupStartRatio;
      const end = Math.max(start + 0.01, range.start + (range.end - range.start) * groupEndRatio);
      segments.push({
        id: `seg-${segments.length}`,
        start,
        end,
        speaker: normalizeSpeakerId(range.speaker ?? dominantSpeakerForRange(start, end, rawSegments)),
        text: group.join(" "),
        raw_words: rawWordsForEditorRange(asrWords, start, end),
      });
    }
  });

  return segments;
}


function pcmToWavBlob(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function revokeEditorPreviewAudioUrl() {
  if (editorPreviewAudioUrl) {
    URL.revokeObjectURL(editorPreviewAudioUrl);
    editorPreviewAudioUrl = "";
  }
}

function showEditorPlayerPanel() {
  const player = $("player-panel");
  if (player) player.style.display = "flex";
}

function setEditorPlayIcon(isPlay) {
  const playIcon = document.querySelector(".play-icon");
  const pauseIcon = document.querySelector(".pause-icon");
  if (playIcon) playIcon.style.display = isPlay ? "block" : "none";
  if (pauseIcon) pauseIcon.style.display = isPlay ? "none" : "block";
}

function setEditorAudioUrl(url, expectedDuration = 0, label = "audio") {
  const audio = $("editor-audio");
  if (!audio || !url) return;
  audio.pause();
  audio.src = url;
  audio.currentTime = 0;
  const seek = $("player-seek");
  if (seek) {
    seek.value = "0";
    seek.max = "100";
  }
  const time = $("player-time");
  if (time) time.textContent = "00:00 / 00:00";
  setEditorPlayIcon(true);
  showEditorPlayerPanel();
  appendDebugLog("player.audio_src_set", {
    label,
    expectedDuration: debugRound(expectedDuration),
    sourceType: url.startsWith("blob:") ? "blob" : "url",
  });
  audio.addEventListener("loadedmetadata", function onLoadedMetadata() {
    audio.removeEventListener("loadedmetadata", onLoadedMetadata);
    const actual = Number(audio.duration) || 0;
    const expected = Number(expectedDuration) || 0;
    if (actual > 0 && expected > 0) {
      const delta = actual - expected;
      log(`[Player] ${label} duration=${actual.toFixed(3)}s, ASR duration=${expected.toFixed(3)}s, delta=${delta.toFixed(3)}s.`);
    }
    appendDebugLog("player.loadedmetadata", {
      label,
      actualDuration: debugRound(actual),
      expectedDuration: debugRound(expected),
      durationDelta: debugRound(actual - expected),
      audio: debugAudioSnapshot(),
    });
  });
  audio.load();
}

function editorPlayableAudioUrl(file, samples) {
  if (file instanceof File || file instanceof Blob) {
    return URL.createObjectURL(file);
  }
  if (samples?.length) {
    return URL.createObjectURL(pcmToWavBlob(samples, VAD_SAMPLE_RATE));
  }
  return "";
}

function showSelectedAudioPreview(file) {
  if (!(file instanceof File || file instanceof Blob)) return;
  revokeEditorPreviewAudioUrl();
  editorPreviewAudioUrl = URL.createObjectURL(file);
  setEditorAudioUrl(editorPreviewAudioUrl, 0, file.name || "source audio");
  appendDebugLog("player.preview_source_selected", {
    fileName: file.name || "source audio",
    fileSize: file.size || 0,
    audio: debugAudioSnapshot(),
  });
}

function clearEditorResult() {
  revokeEditorPreviewAudioUrl();
  if (editorState?.audioUrl) URL.revokeObjectURL(editorState.audioUrl);
  editorState = null;
  const panel = $("result-panel");
  if (panel) panel.style.display = "none";
  const resultContent = $("result-content");
  if (resultContent) resultContent.textContent = "";
  const resultSummary = $("result-summary");
  if (resultSummary) resultSummary.textContent = "";
  const quality = $("quality-strip");
  if (quality) {
    quality.style.display = "none";
    quality.textContent = "";
  }
  const timing = $("result-timing");
  if (timing) {
    timing.style.display = "none";
    timing.textContent = "";
  }
  const search = $("search-input");
  if (search) search.value = "";
  const searchCount = $("search-count");
  if (searchCount) searchCount.textContent = "";
  const audio = $("editor-audio");
  if (audio) {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  }
  const playerSeek = $("player-seek");
  if (playerSeek) {
    playerSeek.value = "0";
    playerSeek.max = "100";
  }
  const playerTime = $("player-time");
  if (playerTime) playerTime.textContent = "00:00 / 00:00";

  const player = $("player-panel");
  if (player) player.style.display = "none";
  const debugButton = $("btn-export-debug-log");
  if (debugButton) debugButton.disabled = !debugLogEntries.length;
}

function setEditorResult(result) {
  const samples = result.samples;
  const duration = samples?.length ? samples.length / VAD_SAMPLE_RATE : (result.duration || 0);
  const previousSpeakers = editorState?.speakers || {};
  clearEditorResult();

  const diarizationSegments = normalizeDiarizationSegments(result.diarization?.segments || []);
  const rawSpeakerSegments = normalizeDiarizationSegments(result.diarization?.rawSegments || result.diarization?.segments || []);
  const speakerTimelineSegments = diarizationSegments.length ? diarizationSegments : rawSpeakerSegments;
  const overlapSegments = normalizeEditorOverlapSegments(result.overlap?.segments || []);
  const speakerDiarizationEnabled = Boolean(
    rawSpeakerSegments.length ||
    result.pipelineOptions?.speakerDiarization ||
    (result.diarization?.backend && result.diarization.backend !== "off")
  );
  let segments = buildEditorTranscriptSegments(result.text || result.asr?.text || "", result.asr, speakerTimelineSegments, duration);
  segments = maybeMapEditorSegmentsFromTimeline(segments, result.asr, duration);
  segments = splitEditorLongSegmentsDesktopStyle(normalizeEditorSegmentBoundaries(segments, duration), 12.0);
  const sourceFile = result.file instanceof File || result.file instanceof Blob ? result.file : null;
  const audioUrl = editorPlayableAudioUrl(sourceFile, samples);
  editorState = {
    fileName: result.file?.name || result.fileName || "audio",
    fileSize: result.file?.size || 0,
    samples,
    duration,
    text: result.text || result.asr?.text || "",
    asr: result.asr || null,
    punctuation: result.punctuation || null,
    diarization: result.diarization || null,
    overlap: result.overlap || null,
    qualityInfo: result.qualityInfo || result.quality_info || null,
    timing: result.timing || null,
    pipelineOptions: result.pipelineOptions || {},
    speakerDiarizationEnabled,
    rawSpeakerSegments,
    overlapSegments,
    segments,
    speakers: {},
    pipelineLog: Array.isArray(result.pipelineLog) ? result.pipelineLog.slice() : pipelineLogLines.slice(),
    audioUrl,
    sourceFile,
    libraryItemId: result.libraryItemId || null,
    activeSegmentIndex: -1,
    activeTab: "transcript",
    searchQuery: "",
    searchMatches: [],
    searchPieces: [],
    searchIndex: -1,
  };
  syncEditorSpeakers(previousSpeakers);
  appendDebugLog("editor.result_set", {
    fileName: editorState.fileName,
    duration: debugRound(duration),
    asrWords: result.asr?.words?.length || 0,
    segmentCount: segments.length,
    rawWordSegments: segments.filter((segment) => Array.isArray(segment.raw_words) && segment.raw_words.length).length,
    rawSpeakerSegments: rawSpeakerSegments.length,
    firstSegment: debugSegmentSnapshot(0),
    lastSegment: debugSegmentSnapshot(Math.max(0, segments.length - 1)),
  });
  if (result.asr?.words?.length && !editorSegmentsHaveRawWordTiming(segments)) {
    log("[Editor] ASR produced word timestamps, but transcript alignment did not attach them to editor segments.");
  }

  const audio = $("editor-audio");
  if (audio && audioUrl) {
    setEditorAudioUrl(audioUrl, duration, sourceFile ? result.file?.name || "source audio" : "canonical PCM");
  }

  const panel = $("result-panel");
  if (panel) panel.style.display = "flex";
  const search = $("search-input");
  if (search) search.value = "";
  renderEditor();
}

function buildRawSpeakerSegmentsFromTranscript(segments) {
  const raw = [];
  for (const segment of segments || []) {
    const previous = raw[raw.length - 1];
    if (previous && previous.speaker === segment.speaker && segment.start - previous.end <= 0.5) {
      previous.end = Math.max(previous.end, segment.end);
    } else {
      raw.push({
        id: `raw-${raw.length}`,
        start: segment.start,
        end: segment.end,
        speaker: normalizeSpeakerId(segment.speaker),
      });
    }
  }
  return raw;
}

function parseEditorSegmentsFromAsrJson(data) {
  if (!data || !Array.isArray(data.segments)) {
    throw new Error("Invalid ASR JSON: missing segments.");
  }

  const segments = [];
  const speakers = {};
  const speakerNames = data.speaker_names || {};
  const speakerColors = data.speaker_colors || {};
  let currentSpeaker = 0;

  for (const item of data.segments) {
    const type = item.type || "text";
    if (type === "speaker") {
      currentSpeaker = normalizeSpeakerId(item.speaker_id);
      speakers[currentSpeaker] = {
        name: speakerNames[String(currentSpeaker)] || item.speaker || defaultSpeakerName(currentSpeaker),
        color: normalizedColor(speakerColors[String(currentSpeaker)], SPEAKER_COLORS[currentSpeaker % SPEAKER_COLORS.length]),
      };
      continue;
    }
    if (type !== "text") continue;

    const speaker = item.speaker_id !== undefined ? normalizeSpeakerId(item.speaker_id) : currentSpeaker;
    const start = Number(item.start_time ?? item.start ?? 0);
    const partials = Array.isArray(item.partials) ? item.partials : [];
    const partialEnd = partials.length ? Number(partials[partials.length - 1].timestamp) : NaN;
    const end = Number(item.end_time ?? item.end ?? partialEnd);
    segments.push({
      id: `seg-${segments.length}`,
      start: Number.isFinite(start) ? start : 0,
      end: Number.isFinite(end) ? end : (Number.isFinite(start) ? start + 1 : segments.length + 1),
      speaker,
      text: item.text || "",
      partials,
      raw_words: Array.isArray(item.raw_words) ? item.raw_words : [],
    });
    if (!speakers[speaker]) {
      speakers[speaker] = {
        name: speakerNames[String(speaker)] || defaultSpeakerName(speaker),
        color: normalizedColor(speakerColors[String(speaker)], SPEAKER_COLORS[speaker % SPEAKER_COLORS.length]),
      };
    }
  }

  for (let i = 0; i < segments.length; i += 1) {
    const next = segments[i + 1];
    if (!Number.isFinite(segments[i].end) || segments[i].end <= segments[i].start) {
      segments[i].end = next ? Math.max(segments[i].start + 0.01, next.start) : segments[i].start + 1;
    }
  }

  for (const [id, name] of Object.entries(speakerNames)) {
    const speaker = normalizeSpeakerId(id);
    speakers[speaker] = {
      name: name || speakers[speaker]?.name || defaultSpeakerName(speaker),
      color: normalizedColor(speakerColors[String(speaker)], speakers[speaker]?.color || SPEAKER_COLORS[speaker % SPEAKER_COLORS.length]),
    };
  }

  return { segments, speakers };
}

function editorSegmentsHaveRawWordTiming(segments = []) {
  return (segments || []).some((segment) => Array.isArray(segment.raw_words) && segment.raw_words.length > 0);
}

function setEditorResultFromAsrJson(data, fileName) {
  const parsed = parseEditorSegmentsFromAsrJson(data);
  const duration = Number(data.duration_sec) || parsed.segments.reduce((max, segment) => Math.max(max, segment.end), 0);
  const asr = data.pwa?.asr_timeline ? { timeline: data.pwa.asr_timeline } : null;
  const hasRawWordTiming = editorSegmentsHaveRawWordTiming(parsed.segments);
  let segments = maybeMapEditorSegmentsFromTimeline(parsed.segments, asr, duration);
  segments = normalizeEditorSegmentBoundaries(segments, duration);
  if (hasRawWordTiming) {
    segments = splitEditorLongSegmentsDesktopStyle(segments, 12.0);
  } else {
    log("[Editor] Loaded ASR JSON has no raw word timestamps; keeping original segment timing. Click-to-seek may be coarse until the file is reprocessed/exported with the current PWA.");
  }
  const rawSpeakerSegments = normalizeDiarizationSegments(
    data.pwa?.raw_speaker_segments ||
    data.raw_speaker_segments ||
    data.diarization_segments ||
    []
  );
  const overlapSegments = normalizeEditorOverlapSegments(data.overlap_segments || data.pwa?.overlap_segments || []);
  clearEditorResult();

  editorState = {
    fileName: data.pwa?.source_file_name || fileName || "result.asr.json",
    fileSize: 0,
    samples: null,
    duration,
    text: parsed.segments.map((segment) => segment.text).filter(Boolean).join(" "),
    asr,
    diarization: data.pwa?.diarization || null,
    overlap: data.pwa?.overlap || { segments: overlapSegments },
    qualityInfo: data.quality_info || null,
    timing: data.timing || null,
    pipelineOptions: data.pwa?.pipeline_options || {},
    rawSpeakerSegments: rawSpeakerSegments.length ? rawSpeakerSegments : buildRawSpeakerSegmentsFromTranscript(parsed.segments),
    overlapSegments,
    segments,
    speakers: parsed.speakers,
    pipelineLog: Array.isArray(data.pwa?.pipeline_log) ? data.pwa.pipeline_log.slice() : [],
    audioUrl: "",
    sourceFile: null,
    libraryItemId: null,
    activeSegmentIndex: -1,
    activeTab: "transcript",
    searchQuery: "",
    searchMatches: [],
    searchPieces: [],
    searchIndex: -1,
  };
  syncEditorSpeakers(parsed.speakers);
  appendDebugLog("editor.asr_json_loaded", {
    fileName: editorState.fileName,
    duration: debugRound(duration),
    segmentCount: segments.length,
    rawSpeakerSegments: editorState.rawSpeakerSegments.length,
    firstSegment: debugSegmentSnapshot(0),
    lastSegment: debugSegmentSnapshot(Math.max(0, segments.length - 1)),
  });

  const audio = $("editor-audio");
  if (audio) {
    audio.removeAttribute("src");
    audio.load();
  }
  const panel = $("result-panel");
  if (panel) panel.style.display = "flex";
  const search = $("search-input");
  if (search) search.value = "";
  renderTranscript(editorState.text, `Loaded ${fileName || ".asr.json"}`);
  renderDiarization(editorState.rawSpeakerSegments);
  renderOverlapSegments(editorState.overlapSegments);
  renderEditor();
}

function editorSpeakerNameMap() {
  const names = {};
  if (!editorState) return names;
  for (const id of collectEditorSpeakerIds(editorState.segments, editorState.rawSpeakerSegments, editorState.overlapSegments)) {
    names[String(id)] = speakerMetaFor(id).name;
  }
  return names;
}

function editorSpeakerColorMap() {
  const colors = {};
  if (!editorState) return colors;
  for (const id of collectEditorSpeakerIds(editorState.segments, editorState.rawSpeakerSegments, editorState.overlapSegments)) {
    colors[String(id)] = speakerMetaFor(id).color;
  }
  return colors;
}

function serializeEditorSegmentsForAsrJson() {
  const output = [];
  let currentSpeaker = null;
  editorState.segments.forEach((segment, index) => {
    const speaker = normalizeSpeakerId(segment.speaker);
    const meta = speakerMetaFor(speaker);
    if (speaker !== currentSpeaker) {
      output.push({
        type: "speaker",
        speaker: meta.name,
        speaker_id: speaker,
        start_time: Number(segment.start.toFixed(3)),
      });
      currentSpeaker = speaker;
    }

    const textSegment = {
      type: "text",
      text: segment.text || "",
      start_time: Number(segment.start.toFixed(3)),
      segment_id: index,
      partials: [{
        text: segment.text || "",
        timestamp: Number(segment.end.toFixed(3)),
      }],
    };
    if (Array.isArray(segment.raw_words) && segment.raw_words.length) {
      textSegment.raw_words = segment.raw_words;
    }
    output.push(textSegment);
  });
  return output;
}

function serializeEditorOverlapSegmentsForAsrJson() {
  return (editorState?.overlapSegments || []).map((segment) => {
    const speaker = normalizeSpeakerId(segment.speaker);
    const entry = {
      speaker: speakerMetaFor(speaker).name,
      speaker_id: speaker,
      start_time: Number(segment.start.toFixed(3)),
      end_time: Number(segment.end.toFixed(3)),
      text: segment.text || "",
    };
    if (Array.isArray(segment.raw_words) && segment.raw_words.length) {
      entry.raw_words = segment.raw_words;
    }
    return entry;
  });
}

function formatEditorRawSpeakerText() {
  if (!editorState?.rawSpeakerSegments?.length) return "";
  return editorState.rawSpeakerSegments
    .map((segment) => (
      `${speakerMetaFor(segment.speaker).name}: ${segment.start.toFixed(2)}-${segment.end.toFixed(2)}s`
    ))
    .join("\n");
}

function formatEditorTranscriptText() {
  if (!editorState?.segments?.length) return "";
  const paragraphs = [];
  let currentSpeaker = null;
  let texts = [];

  function flush() {
    const body = texts.map((text) => text.trim()).filter(Boolean).join(" ");
    if (!body) return;
    if (currentSpeaker !== null) {
      paragraphs.push(`${speakerMetaFor(currentSpeaker).name}:\n${body}`);
    } else {
      paragraphs.push(body);
    }
  }

  for (const segment of editorState.segments) {
    const speaker = normalizeSpeakerId(segment.speaker);
    if (speaker !== currentSpeaker) {
      flush();
      currentSpeaker = speaker;
      texts = [];
    }
    texts.push(segment.text || "");
  }
  flush();
  return paragraphs.join("\n\n").trim();
}

function serializeEditorAsrJson() {
  if (!editorState?.segments?.length) {
    throw new Error("No editor result to save.");
  }
  const timing = { ...(editorState.timing || {}) };
  if (editorState.asr?.elapsed) timing.transcription_detail = editorState.asr.elapsed;
  if (editorState.punctuation?.elapsed) timing.punctuation = editorState.punctuation.elapsed;
  if (editorState.diarization?.elapsed) timing.diarization = editorState.diarization.elapsed;
  if (editorState.overlap?.elapsed) timing.overlap_separation = editorState.overlap.elapsed;

  const data = {
    version: ASR_JSON_VERSION,
    model: editorState.asr?.model || editorState.pipelineOptions?.asrModel?.id || "offline_pwa",
    model_type: "file",
    created_at: new Date().toISOString(),
    duration_sec: Number((editorState.duration || 0).toFixed(2)),
    timing,
    quality_info: editorState.qualityInfo || undefined,
    speaker_names: editorSpeakerNameMap(),
    speaker_colors: editorSpeakerColorMap(),
    segments: serializeEditorSegmentsForAsrJson(),
    speaker_diarization_text: formatEditorRawSpeakerText(),
    pwa: {
      schema_version: 1,
      source_file_name: editorState.fileName,
      source_file_size: editorState.fileSize || 0,
      raw_speaker_segments: editorState.rawSpeakerSegments || [],
      diarization: editorState.diarization || null,
      asr_timeline: editorState.asr?.timeline || null,
      overlap: editorState.overlap || null,
      pipeline_options: editorState.pipelineOptions || {},
      pipeline_log: editorState.pipelineLog || pipelineLogLines.slice(),
    },
  };

  const overlapSegments = serializeEditorOverlapSegmentsForAsrJson();
  if (overlapSegments.length) {
    data.overlap_segments = overlapSegments;
    data.pwa.overlap_segments = editorState.overlapSegments;
  }

  return data;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function writeTextFileWithPicker(filename, text, mimeType, description, extension) {
  const blob = new Blob([text], { type: mimeType });
  if (window.showSaveFilePicker && window.isSecureContext) {
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [{
        description,
        accept: { [mimeType]: [extension] },
      }],
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    return;
  }
  downloadBlob(blob, filename);
}

async function createTextFileWriterFromUserGesture(filename, mimeType, description, extension) {
  if (window.showSaveFilePicker && window.isSecureContext) {
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [{
        description,
        accept: { [mimeType]: [extension] },
      }],
    });
    return async (text) => {
      const blob = new Blob([text], { type: mimeType });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
    };
  }
  return async (text) => {
    downloadBlob(new Blob([text], { type: mimeType }), filename);
  };
}

const WEBGPU_LIMIT_KEYS = [
  "maxTextureDimension1D",
  "maxTextureDimension2D",
  "maxTextureDimension3D",
  "maxTextureArrayLayers",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
  "maxDynamicUniformBuffersPerPipelineLayout",
  "maxDynamicStorageBuffersPerPipelineLayout",
  "maxSampledTexturesPerShaderStage",
  "maxSamplersPerShaderStage",
  "maxStorageBuffersPerShaderStage",
  "maxStorageTexturesPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxUniformBufferBindingSize",
  "maxStorageBufferBindingSize",
  "minUniformBufferOffsetAlignment",
  "minStorageBufferOffsetAlignment",
  "maxVertexBuffers",
  "maxBufferSize",
  "maxVertexAttributes",
  "maxVertexBufferArrayStride",
  "maxInterStageShaderComponents",
  "maxInterStageShaderVariables",
  "maxColorAttachments",
  "maxColorAttachmentBytesPerSample",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
];

function plainWebGpuInfo(info) {
  if (!info) return null;
  const out = {};
  for (const key of ["vendor", "architecture", "device", "description", "subgroupMinSize", "subgroupMaxSize"]) {
    if (info[key] !== undefined && info[key] !== "") out[key] = info[key];
  }
  return Object.keys(out).length ? out : null;
}

function webGpuLimitsObject(limits) {
  if (!limits) return {};
  const out = {};
  for (const key of WEBGPU_LIMIT_KEYS) {
    const value = limits[key];
    if (value !== undefined && value !== null) out[key] = Number(value);
  }
  return out;
}

async function collectBenchmarkEnvironment() {
  const storage = navigator.storage?.estimate
    ? await navigator.storage.estimate().catch(() => null)
    : null;
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection || null;
  const environment = {
    createdAt: new Date().toISOString(),
    app: "offline_pwa",
    userAgent: navigator.userAgent,
    platform: navigator.platform || "",
    language: navigator.language || "",
    secureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency || null,
    physicalCores: null,
    physicalCoresNote: "Browser APIs do not expose physical CPU cores or physical threads.",
    deviceMemoryGB: navigator.deviceMemory || null,
    screen: {
      width: window.screen?.width || null,
      height: window.screen?.height || null,
      devicePixelRatio: window.devicePixelRatio || 1,
    },
    storage: storage ? {
      quotaBytes: storage.quota || null,
      usageBytes: storage.usage || null,
    } : null,
    connection: connection ? {
      effectiveType: connection.effectiveType || null,
      downlink: connection.downlink || null,
      rtt: connection.rtt || null,
      saveData: connection.saveData || false,
    } : null,
    wasm: {
      requestedThreads: getRequestedThreads(),
      maxThreads: maxWasmThreads(),
      logicalThreads: logicalThreadCount(),
      simd: "auto",
      proxy: false,
      crossOriginIsolatedRequiredForThreads: true,
    },
    webgpu: {
      supported: Boolean(navigator.gpu),
      vramBytes: null,
      vramNote: "WebGPU does not expose total/dedicated VRAM to web pages.",
      adapterInfo: null,
      features: [],
      limits: {},
    },
  };

  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      environment.webgpu.adapterAvailable = Boolean(adapter);
      if (adapter) {
        const info = typeof adapter.requestAdapterInfo === "function"
          ? await adapter.requestAdapterInfo().catch(() => null)
          : adapter.info;
        environment.webgpu.adapterInfo = plainWebGpuInfo(info);
        environment.webgpu.features = Array.from(adapter.features || []);
        environment.webgpu.limits = webGpuLimitsObject(adapter.limits);
      }
    } catch (error) {
      environment.webgpu.adapterError = error.message || String(error);
    }
  }
  return environment;
}

function summarizePipelineOptionsForBenchmark(options) {
  return {
    asrModel: options.asrModel?.id || "",
    asrLabel: options.asrModel?.label || "",
    cpuThreads: options.cpuThreads,
    punctuationLevel: options.punctuationLevel,
    caseLevel: options.caseLevel,
    punctuation: !options.bypassPunctuation,
    vad: !options.bypassVad,
    speakerDiarization: options.speakerDiarization,
    speakerModel: options.speakerModel,
    numSpeakers: options.numSpeakers,
    overlapSeparation: options.overlapSeparation,
    rmsNormalize: options.rmsNormalize,
    saveRam: options.saveRam,
    hotwordCount: options.hotwordCount,
  };
}

function summarizeBenchmarkResult(mode, result, elapsedSeconds) {
  const text = String(result?.text || "");
  return {
    mode,
    elapsedSeconds: Number(elapsedSeconds.toFixed(3)),
    providers: {
      asr: "wasm",
      vad: result?.vad?.provider || (result?.vad?.bypassed ? "off" : "wasm"),
      quality: result?.qualityInfo?.benchmarkSelectedProvider || (result?.qualityInfo ? "wasm" : "off"),
      punctuation: result?.punctuation?.executionProvider || "off",
      diarizationSegmentation: result?.diarization?.executionProvider?.segmentation || (result?.diarization?.backend === "off" ? "off" : "wasm"),
      speakerEmbedding: result?.diarization?.executionProvider?.embedding || "off",
      overlap: result?.overlap?.backend === "convtasnet" ? "wasm" : "off",
    },
    timings: {
      vadSeconds: Number(result?.vad?.elapsed?.toFixed?.(3) || 0),
      asrSeconds: Number(result?.asr?.elapsed?.toFixed?.(3) || 0),
      qualitySeconds: Number(result?.qualityInfo?.elapsed?.toFixed?.(3) || 0),
      diarizationSeconds: Number(result?.diarization?.elapsed?.toFixed?.(3) || 0),
      punctuationSeconds: Number(result?.punctuation?.elapsed?.toFixed?.(3) || 0),
      overlapSeconds: Number(result?.overlap?.elapsed?.toFixed?.(3) || 0),
    },
    counts: {
      transcriptChars: text.length,
      transcriptHash: hashString(text),
      vadSegments: result?.vad?.segments || 0,
      vadSpeechSeconds: Number(result?.vad?.speechSeconds?.toFixed?.(3) || 0),
      asrChunks: result?.asr?.chunks?.length || 0,
      speakerTurns: result?.diarization?.segments?.length || 0,
      speakers: result?.diarization?.speakers || 0,
      speakerEmbeddings: result?.diarization?.embeddings || 0,
      punctChunks: result?.punctuation?.chunks || 0,
      overlapLines: result?.overlap?.segments?.length || 0,
    },
  };
}

function benchmarkLogFilename(fileName) {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${safeFileBaseName(fileName, "audio")}.pwa-benchmark.${stamp}.json`;
}

function benchmarkSeconds(started) {
  return Number(((performance.now() - started) / 1000).toFixed(3));
}

function hashFloatValues(values, scale = 1e6, limit = Number.POSITIVE_INFINITY) {
  let hash = 2166136261;
  const count = Math.min(values?.length || 0, limit);
  for (let i = 0; i < count; i += 1) {
    const value = Math.round((Number(values[i]) || 0) * scale);
    hash ^= value & 0xff;
    hash = Math.imul(hash, 16777619);
    hash ^= (value >>> 8) & 0xff;
    hash = Math.imul(hash, 16777619);
    hash ^= (value >>> 16) & 0xff;
    hash = Math.imul(hash, 16777619);
    hash ^= (value >>> 24) & 0xff;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

function hashEmbeddingList(embeddings, limit = 256000) {
  if (!embeddings) return "0";
  if (embeddings instanceof Float32Array || embeddings instanceof Float64Array) {
    return hashFloatValues(embeddings, 1e6, limit);
  }
  let hash = 2166136261;
  let seen = 0;
  for (const vector of embeddings || []) {
    const length = Math.min(vector?.length || 0, Math.max(0, limit - seen));
    const part = hashFloatValues(vector, 1e6, length);
    for (let i = 0; i < part.length; i += 1) {
      hash ^= part.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    seen += length;
    if (seen >= limit) break;
  }
  return (hash >>> 0).toString(16);
}

function addBenchmarkStage(options, stage) {
  if (!Array.isArray(options?.benchmarkStages)) return;
  const record = {
    index: options.benchmarkStages.length + 1,
    ...stage,
  };
  options.benchmarkStages.push(record);
  return record;
}

function benchmarkOutputHash(summary = {}) {
  return summary.outputHash || summary.textHash || summary.embeddingHash || summary.segmentHash || null;
}

function flattenNumericValues(value, limit = Number.POSITIVE_INFINITY) {
  if (!value) return new Float64Array(0);
  if (value instanceof Float32Array || value instanceof Float64Array || value instanceof Int32Array || value instanceof Uint8Array) {
    return value.length > limit ? Float64Array.from(value.subarray(0, limit)) : Float64Array.from(value);
  }
  const output = [];
  const visit = (item) => {
    if (output.length >= limit || item == null) return;
    if (typeof item === "number") {
      output.push(Number(item) || 0);
      return;
    }
    if (item instanceof Float32Array || item instanceof Float64Array || item instanceof Int32Array || item instanceof Uint8Array) {
      const count = Math.min(item.length, limit - output.length);
      for (let i = 0; i < count; i += 1) output.push(Number(item[i]) || 0);
      return;
    }
    if (Array.isArray(item)) {
      for (const child of item) {
        if (output.length >= limit) break;
        visit(child);
      }
    }
  };
  visit(value);
  return Float64Array.from(output);
}

function numericDiffStats(aValue, bValue, limit = 1000000) {
  const a = flattenNumericValues(aValue, limit);
  const b = flattenNumericValues(bValue, limit);
  const count = Math.min(a.length, b.length);
  if (!count) {
    return {
      comparedValues: 0,
      aLength: a.length,
      bLength: b.length,
      reason: "no comparable numeric values",
    };
  }
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let sumSqDiff = 0;
  let finitePairs = 0;
  let bothNonFinite = 0;
  let oneNonFinite = 0;
  let firstDiffIndex = null;
  for (let i = 0; i < count; i += 1) {
    const av = a[i];
    const bv = b[i];
    const aFinite = Number.isFinite(av);
    const bFinite = Number.isFinite(bv);
    if (!aFinite || !bFinite) {
      if (!aFinite && !bFinite) bothNonFinite += 1;
      else {
        oneNonFinite += 1;
        if (firstDiffIndex === null) firstDiffIndex = i;
      }
      continue;
    }
    const diff = Math.abs(av - bv);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    sumAbsDiff += diff;
    sumSqDiff += diff * diff;
    if (firstDiffIndex === null && diff !== 0) firstDiffIndex = i;
    finitePairs += 1;
  }
  const denom = Math.max(1, finitePairs);
  return {
    comparedValues: count,
    finitePairs,
    bothNonFinite,
    oneNonFinite,
    aLength: a.length,
    bLength: b.length,
    firstDiffIndex,
    maxAbsDiff: Number(maxAbsDiff.toExponential(6)),
    meanAbsDiff: Number((sumAbsDiff / denom).toExponential(6)),
    rmsDiff: Number(Math.sqrt(sumSqDiff / denom).toExponential(6)),
    hashScale1e6: {
      a: hashFloatValues(a, 1e6, count),
      b: hashFloatValues(b, 1e6, count),
      equal: hashFloatValues(a, 1e6, count) === hashFloatValues(b, 1e6, count),
    },
    hashScale1e5: {
      a: hashFloatValues(a, 1e5, count),
      b: hashFloatValues(b, 1e5, count),
      equal: hashFloatValues(a, 1e5, count) === hashFloatValues(b, 1e5, count),
    },
    hashScale1e4: {
      a: hashFloatValues(a, 1e4, count),
      b: hashFloatValues(b, 1e4, count),
      equal: hashFloatValues(a, 1e4, count) === hashFloatValues(b, 1e4, count),
    },
  };
}

function embeddingCosineDiffStats(aEmbeddings, bEmbeddings, limitVectors = 4096) {
  if (
    (aEmbeddings instanceof Float32Array || aEmbeddings instanceof Float64Array) &&
    (bEmbeddings instanceof Float32Array || bEmbeddings instanceof Float64Array)
  ) {
    const dim = 256;
    const totalVectors = Math.min(Math.floor(aEmbeddings.length / dim), Math.floor(bEmbeddings.length / dim));
    const count = Math.min(totalVectors, limitVectors);
    if (!count) return { comparedVectors: 0, aVectors: totalVectors, bVectors: totalVectors };
    let minCosine = Infinity;
    let sumCosine = 0;
    let compared = 0;
    let skippedNonFinite = 0;
    let firstCosineBelow9999 = null;
    for (let i = 0; i < count; i += 1) {
      const base = i * dim;
      let dot = 0;
      let na = 0;
      let nb = 0;
      let finite = true;
      for (let d = 0; d < dim; d += 1) {
        const av = Number(aEmbeddings[base + d]);
        const bv = Number(bEmbeddings[base + d]);
        const aFinite = Number.isFinite(av);
        const bFinite = Number.isFinite(bv);
        if (!aFinite || !bFinite) {
          if (!aFinite && !bFinite) continue;
          finite = false;
          break;
        }
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
      }
      if (!finite || (na <= 0 && nb <= 0)) {
        skippedNonFinite += 1;
        continue;
      }
      const cosineValue = dot / Math.max(Math.sqrt(na) * Math.sqrt(nb), 1e-12);
      minCosine = Math.min(minCosine, cosineValue);
      sumCosine += cosineValue;
      if (firstCosineBelow9999 === null && cosineValue < 0.9999) firstCosineBelow9999 = i;
      compared += 1;
    }
    return {
      comparedVectors: compared,
      skippedNonFinite,
      aVectors: totalVectors,
      bVectors: totalVectors,
      minCosine: compared ? Number(minCosine.toFixed(8)) : null,
      meanCosine: compared ? Number((sumCosine / compared).toFixed(8)) : null,
      firstCosineBelow9999,
    };
  }
  const aList = Array.isArray(aEmbeddings) ? aEmbeddings : null;
  const bList = Array.isArray(bEmbeddings) ? bEmbeddings : null;
  if (!aList || !bList) return null;
  const count = Math.min(aList.length, bList.length, limitVectors);
  if (!count) return { comparedVectors: 0 };
  let minCosine = Infinity;
  let sumCosine = 0;
  let firstCosineBelow9999 = null;
  for (let i = 0; i < count; i += 1) {
    const a = aList[i] || [];
    const b = bList[i] || [];
    const dim = Math.min(a.length || 0, b.length || 0);
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let d = 0; d < dim; d += 1) {
      dot += (Number(a[d]) || 0) * (Number(b[d]) || 0);
      na += (Number(a[d]) || 0) ** 2;
      nb += (Number(b[d]) || 0) ** 2;
    }
    const cosineValue = dot / Math.max(Math.sqrt(na) * Math.sqrt(nb), 1e-12);
    minCosine = Math.min(minCosine, cosineValue);
    sumCosine += cosineValue;
    if (firstCosineBelow9999 === null && cosineValue < 0.9999) firstCosineBelow9999 = i;
  }
  return {
    comparedVectors: count,
    aVectors: aList.length,
    bVectors: bList.length,
    minCosine: Number(minCosine.toFixed(8)),
    meanCosine: Number((sumCosine / count).toFixed(8)),
    firstCosineBelow9999,
  };
}

function embeddingDiffWithinTolerance(mismatchDetails, thresholds = {}) {
  const numeric = mismatchDetails?.numericDiff;
  const cosine = mismatchDetails?.embeddingCosine;
  if (!numeric || !cosine) return false;
  if (Number(numeric.oneNonFinite || 0) > Number(thresholds.maxOneNonFinite ?? 0)) return false;
  if (Number(cosine.comparedVectors || 0) <= 0) return false;
  const maxAbs = Number(thresholds.maxAbsDiff ?? 1e-4);
  const meanAbs = Number(thresholds.meanAbsDiff ?? 1e-5);
  const rms = Number(thresholds.rmsDiff ?? 1e-5);
  const minCosine = Number(thresholds.minCosine ?? 0.9999);
  return (
    Number(numeric.maxAbsDiff) <= maxAbs &&
    Number(numeric.meanAbsDiff) <= meanAbs &&
    Number(numeric.rmsDiff) <= rms &&
    Number(cosine.minCosine) >= minCosine
  );
}

function firstTextDiff(aText, bText) {
  const a = String(aText || "");
  const b = String(bText || "");
  const count = Math.min(a.length, b.length);
  let index = null;
  for (let i = 0; i < count; i += 1) {
    if (a[i] !== b[i]) {
      index = i;
      break;
    }
  }
  if (index === null && a.length !== b.length) index = count;
  return {
    aLength: a.length,
    bLength: b.length,
    firstDiffIndex: index,
    aSnippet: index === null ? "" : a.slice(Math.max(0, index - 32), index + 64),
    bSnippet: index === null ? "" : b.slice(Math.max(0, index - 32), index + 64),
  };
}

function compareRegionArrays(aRegions = [], bRegions = []) {
  const count = Math.min(aRegions.length || 0, bRegions.length || 0);
  let firstDiffIndex = null;
  let firstSpeakerDiffIndex = null;
  let maxStartDiff = 0;
  let maxEndDiff = 0;
  for (let i = 0; i < count; i += 1) {
    const a = aRegions[i] || {};
    const b = bRegions[i] || {};
    maxStartDiff = Math.max(maxStartDiff, Math.abs(Number(a.start) - Number(b.start)));
    maxEndDiff = Math.max(maxEndDiff, Math.abs(Number(a.end) - Number(b.end)));
    if (firstSpeakerDiffIndex === null && Number(a.speaker ?? -1) !== Number(b.speaker ?? -1)) {
      firstSpeakerDiffIndex = i;
    }
    if (
      Number(a.speaker ?? -1) !== Number(b.speaker ?? -1) ||
      Math.abs(Number(a.start) - Number(b.start)) > 0.0005 ||
      Math.abs(Number(a.end) - Number(b.end)) > 0.0005
    ) {
      firstDiffIndex = i;
      break;
    }
  }
  if (firstDiffIndex === null && (aRegions.length || 0) !== (bRegions.length || 0)) firstDiffIndex = count;
  return {
    aCount: aRegions.length || 0,
    bCount: bRegions.length || 0,
    firstDiffIndex,
    firstSpeakerDiffIndex,
    maxStartDiff: Number(maxStartDiff.toFixed(6)),
    maxEndDiff: Number(maxEndDiff.toFixed(6)),
    aFirstDiff: firstDiffIndex === null ? null : aRegions[firstDiffIndex] || null,
    bFirstDiff: firstDiffIndex === null ? null : bRegions[firstDiffIndex] || null,
  };
}

function segmentDiffWithinTolerance(diff, thresholds = {}) {
  if (!diff) return false;
  const maxCountDelta = Number(thresholds.maxCountDelta ?? 0);
  const maxTimeDiff = Number(thresholds.maxTimeDiff ?? 0.05);
  return (
    Math.abs(Number(diff.aCount || 0) - Number(diff.bCount || 0)) <= maxCountDelta &&
    diff.firstSpeakerDiffIndex === null &&
    Number(diff.maxStartDiff || 0) <= maxTimeDiff &&
    Number(diff.maxEndDiff || 0) <= maxTimeDiff
  );
}

function benchmarkMismatchDetails(stageName, wasmResult, webgpuResult, wasmSummary = {}, webgpuSummary = {}) {
  const details = {
    stage: stageName,
    summaryHashes: {
      wasm: benchmarkOutputHash(wasmSummary),
      webgpu: benchmarkOutputHash(webgpuSummary),
      wasmEmbedding: wasmSummary.embeddingHash || null,
      webgpuEmbedding: webgpuSummary.embeddingHash || null,
      wasmSegment: wasmSummary.segmentHash || null,
      webgpuSegment: webgpuSummary.segmentHash || null,
      wasmText: wasmSummary.textHash || null,
      webgpuText: webgpuSummary.textHash || null,
    },
  };

  if (wasmSummary.textHash && webgpuSummary.textHash && wasmSummary.textHash !== webgpuSummary.textHash) {
    details.textDiff = firstTextDiff(wasmResult?.text, webgpuResult?.text);
  }
  if (wasmSummary.wordTimingHash && webgpuSummary.wordTimingHash && wasmSummary.wordTimingHash !== webgpuSummary.wordTimingHash) {
    details.wordTimingHash = {
      wasm: wasmSummary.wordTimingHash,
      webgpu: webgpuSummary.wordTimingHash,
    };
  }
  if (wasmSummary.embeddingHash && webgpuSummary.embeddingHash && wasmSummary.embeddingHash !== webgpuSummary.embeddingHash) {
    const wasmEmb = wasmResult?.embeddings || wasmResult;
    const webgpuEmb = webgpuResult?.embeddings || webgpuResult;
    details.numericDiff = numericDiffStats(wasmEmb, webgpuEmb, 1000000);
    const cosine = embeddingCosineDiffStats(wasmEmb, webgpuEmb);
    if (cosine) details.embeddingCosine = cosine;
  }
  if (wasmResult?.logitsData && webgpuResult?.logitsData) {
    details.logitsDiff = numericDiffStats(wasmResult.logitsData, webgpuResult.logitsData, 1000000);
  }
  if (wasmSummary.segmentHash && webgpuSummary.segmentHash && wasmSummary.segmentHash !== webgpuSummary.segmentHash) {
    details.segmentDiff = compareRegionArrays(
      wasmResult?.segments || wasmResult?.regions || [],
      webgpuResult?.segments || webgpuResult?.regions || []
    );
  }
  if (!details.textDiff && !details.numericDiff && !details.logitsDiff && !details.segmentDiff && !details.wordTimingHash) {
    details.summaryDiff = Object.fromEntries(
      [...new Set([...Object.keys(wasmSummary || {}), ...Object.keys(webgpuSummary || {})])]
        .filter((key) => JSON.stringify(wasmSummary?.[key]) !== JSON.stringify(webgpuSummary?.[key]))
        .slice(0, 20)
        .map((key) => [key, { wasm: wasmSummary?.[key] ?? null, webgpu: webgpuSummary?.[key] ?? null }])
    );
  }
  return details;
}

async function runBenchmarkWasmOnlyStage(options, name, run, summarize = () => ({}), metadata = {}) {
  const started = performance.now();
  try {
    const result = await run();
    const elapsedSeconds = benchmarkSeconds(started);
    const summary = summarize(result) || {};
    addBenchmarkStage(options, {
      name,
      capability: "wasm-only",
      attempts: [{
        runtime: "wasm",
        provider: summary.provider || "wasm",
        elapsedSeconds,
        summary,
      }],
      selectedRuntime: "wasm",
      selectedProvider: summary.provider || "wasm",
      ...metadata,
    });
    log(`[Benchmark] Stage ${name}: WASM ${elapsedSeconds.toFixed(2)}s.`);
    return result;
  } catch (error) {
    addBenchmarkStage(options, {
      name,
      capability: "wasm-only",
      attempts: [{
        runtime: "wasm",
        error: { message: error.message || String(error) },
      }],
      error: error.message || String(error),
    });
    throw error;
  }
}

async function runBenchmarkDualProviderStage(options, name, run, unload, summarize = () => ({})) {
  const attempts = [];
  const results = new Map();
  let firstError = null;

  for (const runtime of ["wasm", "webgpu"]) {
    if (unload) {
      await unload().catch((error) => log(`[Benchmark] ${name} pre-unload failed: ${error.message}`));
    }
    const previousProviderMode = benchmarkProviderMode;
    benchmarkProviderMode = runtime;
    const started = performance.now();
    try {
      const wasmBaseline = runtime === "webgpu"
        ? Number(results.get("wasm")?.attempt?.elapsedSeconds || 0)
        : 0;
      const result = await run(runtime, {
        wasmBaselineSeconds: wasmBaseline,
        abortWebGpuTuneWhenSlower: wasmBaseline > 0,
        useCalibratedWebGpuBatch: options.useCalibratedWebGpuBenchmark === true,
      });
      const totalWithTuneSeconds = benchmarkSeconds(started);
      const summary = summarize(result, runtime) || {};
      const tuneSeconds = runtime === "webgpu"
        ? Number(summary.batchTuning?.totalSeconds || 0)
        : 0;
      const elapsedSeconds = Number(Math.max(0, totalWithTuneSeconds - tuneSeconds).toFixed(3));
      const attempt = {
        runtime,
        provider: summary.provider || runtime,
        elapsedSeconds,
        totalWithTuneSeconds,
        tuneSeconds,
        summary,
      };
      attempts.push(attempt);
      results.set(runtime, { result, attempt });
      log(
        `[Benchmark] Stage ${name}: ${runtime.toUpperCase()} ${elapsedSeconds.toFixed(2)}s ` +
        `${tuneSeconds ? `(+ tune ${tuneSeconds.toFixed(2)}s) ` : ""}(${attempt.provider}).`
      );
    } catch (error) {
      if (!firstError) firstError = error;
      const summary = error.benchmarkSummary || null;
      const tuneSeconds = Number(summary?.batchTuning?.totalSeconds || 0);
      attempts.push({
        runtime,
        provider: summary?.provider || runtime,
        totalWithTuneSeconds: benchmarkSeconds(started),
        tuneSeconds,
        ...(summary ? { summary } : {}),
        error: { message: error.message || String(error) },
      });
      log(`[Benchmark] Stage ${name}: ${runtime.toUpperCase()} failed: ${error.message || String(error)}`);
    } finally {
      benchmarkProviderMode = previousProviderMode;
    }
  }

  if (unload) {
    await unload().catch((error) => log(`[Benchmark] ${name} final unload failed: ${error.message}`));
  }

  if (!results.size) {
    addBenchmarkStage(options, {
      name,
      capability: "wasm-webgpu",
      attempts,
      error: firstError?.message || "Stage failed for both providers.",
    });
    throw firstError || new Error(`${name} failed for both providers.`);
  }

  const wasm = results.get("wasm")?.attempt || attempts.find((attempt) => attempt.runtime === "wasm" && !attempt.error);
  const webgpu = results.get("webgpu")?.attempt || attempts.find((attempt) => attempt.runtime === "webgpu" && !attempt.error);
  const wasmHash = benchmarkOutputHash(wasm?.summary);
  const webgpuHash = benchmarkOutputHash(webgpu?.summary);
  const outputHashEqual = wasmHash && webgpuHash ? wasmHash === webgpuHash : null;
  const speedupWebgpuOverWasm = wasm?.elapsedSeconds && webgpu?.elapsedSeconds
    ? Number((wasm.elapsedSeconds / webgpu.elapsedSeconds).toFixed(3))
    : null;
  const webgpuAccepted = Boolean(
    wasm &&
    webgpu &&
    webgpu.provider === "webgpu" &&
    outputHashEqual === true &&
    speedupWebgpuOverWasm > 1
  );
  const mismatchDetails = outputHashEqual === false
    ? benchmarkMismatchDetails(
        name,
        results.get("wasm")?.result,
        results.get("webgpu")?.result,
        wasm?.summary,
        webgpu?.summary
      )
    : null;
  let rejectionReason = null;
  if (webgpu && !webgpuAccepted) {
    if (webgpu.provider !== "webgpu") rejectionReason = `webgpu attempt ran on ${webgpu.provider}`;
    else if (outputHashEqual !== true) rejectionReason = "webgpu output hash did not match wasm";
    else if (!(speedupWebgpuOverWasm > 1)) rejectionReason = "webgpu was not faster than wasm";
  } else if (!webgpu) {
    const webgpuError = attempts.find((attempt) => attempt.runtime === "webgpu" && attempt.error);
    if (webgpuError) rejectionReason = webgpuError.error?.message || "webgpu attempt failed";
  }
  const selectedRuntime = webgpuAccepted ? "webgpu" : (wasm ? "wasm" : "webgpu");
  const selected = results.get(selectedRuntime) || results.values().next().value;
  const comparison = {
    speedupWebgpuOverWasm,
    outputHashEqual,
    webgpuAccepted,
    webgpuEarlyStopped: attempts.some((attempt) => attempt.runtime === "webgpu" && attempt.summary?.earlyStopped),
    rejectionReason,
    ...(mismatchDetails ? { mismatchDetails } : {}),
  };
  const stageRecord = addBenchmarkStage(options, {
    name,
    capability: "wasm-webgpu",
    attempts,
    selectedRuntime,
    selectedProvider: selected.attempt.provider,
    comparison,
  });
  if (options && typeof options === "object") {
    options.benchmarkStageResults = options.benchmarkStageResults || {};
    options.benchmarkStageResults[name] = {
      attempts,
      stage: stageRecord || null,
      wasm: results.get("wasm") || null,
      webgpu: results.get("webgpu") || null,
    };
  }
  if (webgpu && !webgpuAccepted) {
    log(`[Benchmark] Stage ${name}: WebGPU not selected: ${rejectionReason || "not accepted"}.`);
  }
  if (options && typeof options === "object") {
    options.benchmarkSelectedProviders = options.benchmarkSelectedProviders || {};
    options.benchmarkSelectedProviders[name] = selected.attempt.provider;
  }
  if (selected.result && typeof selected.result === "object") {
    try {
      selected.result.benchmarkSelectedRuntime = selected.attempt.runtime;
      selected.result.benchmarkSelectedProvider = selected.attempt.provider;
    } catch (_) {
      // TypedArray/object extensibility is browser-dependent; stage record still has the provider.
    }
  }
  return selected.result;
}

function summarizePunctuationForBenchmark(result) {
  const text = String(result?.text || "");
  return {
    provider: result?.executionProvider || "wasm",
    chunks: result?.chunks || 0,
    textChars: text.length,
    textHash: hashString(text),
  };
}

function summarizeCamppEmbeddingForBenchmark(result) {
  return {
    provider: camppExecutionProvider,
    embeddings: result?.embeddings?.length || 0,
    windows: result?.windowTimes?.length || 0,
    batchSize: result?.batchSize || null,
    batchTuning: result?.batchTuning || null,
    embeddingHash: hashEmbeddingList(result?.embeddings),
  };
}

function summarizePyannoteEmbeddingForBenchmark(result) {
  return {
    provider: result?.provider || pyannoteEmbeddingExecutionProvider,
    embeddings: Math.floor((result?.length || 0) / 256),
    batchSize: result?.batchSize || null,
    batchTuning: result?.batchTuning || null,
    embeddingHash: hashEmbeddingList(result),
  };
}

function summarizeVadForBenchmark(result) {
  const segments = result?.segments || [];
  return {
    provider: result?.bypassed ? "off" : vadExecutionProvider,
    segments: segments.length,
    speechSeconds: Number((segments.reduce((sum, segment) => sum + segment.end - segment.start, 0) / VAD_SAMPLE_RATE).toFixed(3)),
    boosted: Boolean(result?.boosted),
    segmentHash: hashString(segments.map((s) => `${s.start}-${s.end}`).join("|")),
    probabilityHash: hashFloatValues(result?.probabilities || [], 1e6, 20000),
  };
}

function summarizePyannoteSpeechRegionsForBenchmark(result) {
  const regions = result?.regions || [];
  const overlapRegions = result?.overlapRegions || [];
  const regionHash = hashString(regions.map((s) => `${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|"));
  const overlapHash = hashString(overlapRegions.map((s) => `${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|"));
  return {
    provider: result?.provider || diarizationExecutionProvider,
    regions: regions.length,
    overlapRegions: overlapRegions.length,
    chunks: result?.chunks || 0,
    batchSize: result?.batchSize || null,
    batchTuning: result?.batchTuning || null,
    regionHash,
    overlapHash,
    outputHash: `${regionHash}:${overlapHash}`,
  };
}

function summarizePyannoteSegmentationForBenchmark(result) {
  const binarized = result?.logitsData
    ? powerSetBinarize(result.logitsData, result.chunks, result.numFrames, result.numClasses)
    : new Uint8Array(0);
  const discreteHash = hashFloatValues(binarized, 1, Number.POSITIVE_INFINITY);
  const logitsDebugHash = hashFloatValues(result?.logitsData || [], 1e4, 200000);
  return {
    provider: result?.provider || diarizationExecutionProvider,
    chunks: result?.chunks || 0,
    frames: result?.numFrames || 0,
    classes: result?.numClasses || 0,
    batchSize: result?.batchSize || null,
    batchTuning: result?.batchTuning || null,
    discreteHash,
    logitsDebugHash,
    outputHash: discreteHash,
  };
}

function summarizeQualityForBenchmark(result) {
  return {
    provider: result ? dnsmosExecutionProvider : "off",
    enabled: Boolean(result),
    dnsmosOvrl: result?.dnsmos_ovrl ?? null,
    dnsmosSig: result?.dnsmos_sig ?? null,
    dnsmosBak: result?.dnsmos_bak ?? null,
    asrConfidence: result?.asr_confidence ?? null,
    outputHash: result ? hashString(JSON.stringify({
      dnsmos_sig: result.dnsmos_sig ?? null,
      dnsmos_bak: result.dnsmos_bak ?? null,
      dnsmos_ovrl: result.dnsmos_ovrl ?? null,
      asr_confidence: result.asr_confidence ?? null,
    })) : "off",
  };
}

function summarizeDiarSegmentsForBenchmark(result) {
  const segments = result?.segments || [];
  return {
    provider: result?.executionProvider?.embedding || result?.executionProvider?.segmentation || "wasm",
    turns: segments.length,
    speakers: result?.speakers || 0,
    embeddings: result?.embeddings || 0,
    segmentHash: hashString(segments.map((s) => `${s.speaker}:${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|")),
  };
}

function summarizeBenchmarkStageComparisons(stages = []) {
  const dual = stages.filter((stage) => stage.capability === "wasm-webgpu");
  const isWebGpuFaster = (stage) => Number(stage.comparison?.speedupWebgpuOverWasm || 0) > 1;
  const isToleranceAccepted = (stage) => (
    stage.comparison?.webgpuAcceptedByTolerance === true ||
    stage.comparison?.downstreamSegmentDiffWithinTolerance === true && stage.comparison?.webgpuAccepted === true
  );
  return {
    dualStageCount: dual.length,
    webgpuFasterCount: dual.filter(isWebGpuFaster).length,
    webgpuAcceptedCount: dual.filter((stage) => stage.comparison?.webgpuAccepted === true).length,
    webgpuEarlyStoppedCount: dual.filter((stage) => stage.comparison?.webgpuEarlyStopped === true).length,
    outputMismatchCount: dual.filter((stage) => stage.comparison?.outputHashEqual === false).length,
    toleratedMismatchCount: dual.filter((stage) => stage.comparison?.outputHashEqual === false && isToleranceAccepted(stage)).length,
    rejectedMismatchCount: dual.filter((stage) => stage.comparison?.outputHashEqual === false && stage.comparison?.webgpuAccepted !== true).length,
    stages: dual.map((stage) => ({
      name: stage.name,
      selectedRuntime: stage.selectedRuntime,
      selectedProvider: stage.selectedProvider,
      speedupWebgpuOverWasm: stage.comparison?.speedupWebgpuOverWasm ?? null,
      outputHashEqual: stage.comparison?.outputHashEqual ?? null,
      webgpuAccepted: stage.comparison?.webgpuAccepted ?? false,
      webgpuEarlyStopped: stage.comparison?.webgpuEarlyStopped ?? false,
      webgpuAcceptedByTolerance: stage.comparison?.webgpuAcceptedByTolerance ?? false,
      acceptanceMode: stage.comparison?.acceptanceMode ?? null,
      numericToleranceAccepted: stage.comparison?.numericToleranceAccepted ?? null,
      downstreamSegmentHashEqual: stage.comparison?.downstreamSegmentHashEqual ?? null,
      downstreamSegmentDiffWithinTolerance: stage.comparison?.downstreamSegmentDiffWithinTolerance ?? null,
      segmentDiff: stage.comparison?.segmentDiff ?? null,
      rejectionReason: stage.comparison?.rejectionReason ?? null,
      mismatchDetails: stage.comparison?.mismatchDetails ?? null,
      downstreamCheck: stage.comparison?.downstreamCheck ?? null,
      wasmSeconds: stage.attempts?.find((attempt) => attempt.runtime === "wasm" || attempt.runtime === "js")?.elapsedSeconds ?? null,
      webgpuSeconds: stage.attempts?.find((attempt) => attempt.runtime === "webgpu")?.elapsedSeconds ?? null,
      webgpuTuneSeconds: stage.attempts?.find((attempt) => attempt.runtime === "webgpu")?.tuneSeconds ?? null,
    })),
  };
}

async function saveEditorAsrJson() {
  try {
    const data = serializeEditorAsrJson();
    const filename = `${safeFileBaseName(editorState.fileName)}.asr.json`;
    await writeTextFileWithPicker(
      filename,
      JSON.stringify(data, null, 2),
      "application/json",
      "ASR JSON",
      ".json"
    );
    log(`Saved ASR JSON: ${filename}`);
  } catch (error) {
    if (error?.name === "AbortError") return;
    log(`Save ASR JSON failed: ${error.message}`);
  }
}

function serializeDebugLog() {
  const activeIndex = editorState?.activeSegmentIndex ?? -1;
  return {
    schema: 1,
    created_at: new Date().toISOString(),
    app: "offline_pwa",
    app_code_version: OFFLINE_PWA_CODE_VERSION,
    user_agent: navigator.userAgent,
    location: {
      href: window.location.href,
      standalone: isStandaloneApp(),
      secureContext: window.isSecureContext,
      crossOriginIsolated: window.crossOriginIsolated,
    },
    audio: debugAudioSnapshot(),
    editor: editorState ? {
      fileName: editorState.fileName,
      duration: debugRound(editorState.duration),
      segmentCount: editorState.segments?.length || 0,
      activeSegmentIndex: activeIndex,
      activeSegment: activeIndex >= 0 ? debugSegmentSnapshot(activeIndex) : null,
      firstSegment: debugSegmentSnapshot(0),
      lastSegment: debugSegmentSnapshot(Math.max(0, (editorState.segments?.length || 1) - 1)),
      rawSpeakerSegments: (editorState.rawSpeakerSegments || []).map((segment, index) => ({
        index,
        speaker: normalizeSpeakerId(segment.speaker),
        start: debugRound(segment.start),
        end: debugRound(segment.end),
      })),
    } : null,
    events: debugLogEntries.slice(),
    pipeline_log_tail: pipelineLogLines.slice(-300),
  };
}

async function exportDebugLog() {
  try {
    appendDebugLog("debug.export_requested", {
      audio: debugAudioSnapshot(),
      activeSegment: editorState?.activeSegmentIndex >= 0
        ? debugSegmentSnapshot(editorState.activeSegmentIndex)
        : null,
    });
    const filename = `${safeFileBaseName(editorState?.fileName || "pwa")}.debug-log.json`;
    await writeTextFileWithPicker(
      filename,
      JSON.stringify(serializeDebugLog(), null, 2),
      "application/json",
      "PWA debug log",
      ".json"
    );
    log(`Exported debug log: ${filename}`);
  } catch (error) {
    if (error?.name === "AbortError") return;
    log(`Export debug log failed: ${error.message}`);
  }
}

async function copyEditorText() {
  const text = formatEditorTranscriptText();
  if (!text) {
    log("No transcript text to copy.");
    return;
  }
  try {
    if (navigator.clipboard?.writeText && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.focus();
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    log(`Copied transcript text (${text.length} chars).`);
  } catch (error) {
    log(`Copy text failed: ${error.message}`);
  }
}

async function exportEditorTranscript() {
  try {
    const text = formatEditorTranscriptText();
    if (!text) throw new Error("No transcript text to export.");
    const filename = `${safeFileBaseName(editorState.fileName)}.txt`;
    await writeTextFileWithPicker(filename, text, "text/plain", "Transcript text", ".txt");
    log(`Exported transcript: ${filename}`);
  } catch (error) {
    if (error?.name === "AbortError") return;
    log(`Export transcript failed: ${error.message}`);
  }
}

async function loadAsrJsonFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    setEditorResultFromAsrJson(data, file.name);
    log(`Loaded ASR JSON: ${file.name}`);
  } catch (error) {
    log(`Load ASR JSON failed: ${error.message}`);
  }
}

async function openAsrJsonFile() {
  showToast("PWA offline không còn hỗ trợ Upload JSON. Vui lòng chọn file âm thanh và xử lý lại.", "error");
}

function decodeWavPcmDirect16k(file, arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 44) return null;
  const view = new DataView(arrayBuffer);
  const text4 = (offset) => String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
  );
  if (text4(0) !== "RIFF" || text4(8) !== "WAVE") return null;

  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= arrayBuffer.byteLength) {
    const id = text4(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (body + size > arrayBuffer.byteLength) break;
    if (id === "fmt ") {
      fmt = {
        format: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bitsPerSample: view.getUint16(body + 14, true),
      };
      if (size >= 40 && fmt.format === 0xfffe) {
        const subFormatOffset = body + 24;
        const guidRest = Array.from({ length: 12 }, (_, index) => view.getUint8(subFormatOffset + 4 + index));
        const pcmGuidRest = [0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];
        if (pcmGuidRest.every((value, index) => guidRest[index] === value)) {
          fmt.format = view.getUint16(subFormatOffset, true);
        }
      }
    } else if (id === "data") {
      dataOffset = body;
      dataSize = size;
      break;
    }
    offset = body + size + (size & 1);
  }

  if (!fmt || dataOffset < 0 || fmt.sampleRate !== VAD_SAMPLE_RATE || fmt.channels < 1) return null;
  const bytesPerSample = fmt.bitsPerSample / 8;
  const isPcmInt = fmt.format === 1 && [8, 16, 24, 32].includes(fmt.bitsPerSample);
  const isFloat = fmt.format === 3 && [32, 64].includes(fmt.bitsPerSample);
  if (!Number.isInteger(bytesPerSample) || (!isPcmInt && !isFloat)) return null;

  const readSample = (sampleOffset) => {
    if (isFloat) {
      return fmt.bitsPerSample === 32
        ? view.getFloat32(sampleOffset, true)
        : view.getFloat64(sampleOffset, true);
    }
    if (fmt.bitsPerSample === 8) {
      return (view.getUint8(sampleOffset) - 128) / 128.0;
    }
    if (fmt.bitsPerSample === 16) {
      return view.getInt16(sampleOffset, true) / 32768.0;
    }
    if (fmt.bitsPerSample === 24) {
      let value = view.getUint8(sampleOffset)
        | (view.getUint8(sampleOffset + 1) << 8)
        | (view.getUint8(sampleOffset + 2) << 16);
      if (value & 0x800000) value |= 0xff000000;
      return value / 8388608.0;
    }
    return view.getInt32(sampleOffset, true) / 2147483648.0;
  };

  const frameCount = Math.floor(dataSize / (bytesPerSample * fmt.channels));
  const samples = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    const frameOffset = dataOffset + frame * bytesPerSample * fmt.channels;
    for (let ch = 0; ch < fmt.channels; ch += 1) {
      sum += readSample(frameOffset + ch * bytesPerSample);
    }
    samples[frame] = sum / fmt.channels;
  }
  return {
    samples,
    sampleRate: VAD_SAMPLE_RATE,
    originalSampleRate: fmt.sampleRate,
    channels: fmt.channels,
    duration: samples.length / VAD_SAMPLE_RATE,
    decoder: `wav-${isFloat ? "float" : "pcm"}${fmt.bitsPerSample}-direct`,
    resampler: "none",
  };
}

function decodeAudioFileWithFfmpeg(file, arrayBuffer, options = {}) {
  const directWave = decodeWavPcmDirect16k(file, arrayBuffer);
  if (directWave) {
    log(`Audio decoder: ${directWave.decoder} canonical path.`);
    return Promise.resolve(directWave);
  }
  if (shouldUseMpg123Decode(file)) {
    return decodeAudioFileWithMpg123(file, arrayBuffer)
      .then((decoded) => {
        log(`Audio decoder: ${decoded.decoder} canonical path.`);
        return decoded;
      })
      .catch((error) => {
        log(`Audio decoder: mpg123 failed for ${file?.name || "input"}; using browser/FFmpeg decoder. ${error.message}`);
        return decodeAudioFileWithBrowserOrFfmpeg(file, arrayBuffer, options);
      });
  }
  return decodeAudioFileWithBrowserOrFfmpeg(file, arrayBuffer, options);
}

function decodeAudioFileWithBrowserOrFfmpeg(file, arrayBuffer, options = {}) {
  if (shouldPreferWebAudioDecode(file)) {
    return decodeAudioFileWithWebAudio(file, arrayBuffer)
      .then((decoded) => {
        log(`Audio decoder: ${decoded.decoder} canonical path.`);
        return decoded;
      })
      .catch((error) => {
        log(`Audio decoder: WebAudio failed for ${file?.name || "input"}; using FFmpeg WASM. ${error.message}`);
        return decodeAudioFileWithFfmpegWorker(file, arrayBuffer, options);
      });
  }
  if (shouldTryContainerAudioExtract(file)) {
    return decodeAudioFileWithContainerExtract(file, arrayBuffer, options);
  }
  if (isKnownUnsafeFfmpegWasmContainer(file)) {
    return Promise.reject(new Error(`${mediaFileExtension(file) || "Container"} chưa được hỗ trợ ổn định trong PWA offline. Vui lòng dùng MP3, M4A, MP4, WAV, WebM, OGG, OPUS, FLAC hoặc AAC.`));
  }
  return decodeAudioFileWithFfmpegWorker(file, arrayBuffer, options);
}

function mediaFileExtension(file) {
  const name = String(file?.name || "").toLowerCase();
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot) : "";
}

function shouldPreferWebAudioDecode(file) {
  const ext = mediaFileExtension(file);
  return ext === ".wav"
    || ext === ".m4a"
    || ext === ".mp4"
    || ext === ".aac"
    || ext === ".mov"
    || ext === ".webm"
    || ext === ".ogg"
    || ext === ".opus"
    || ext === ".flac";
}

function shouldUseMpg123Decode(file) {
  return mediaFileExtension(file) === ".mp3";
}

function shouldTryContainerAudioExtract(file) {
  return false;
}

function isKnownUnsafeFfmpegWasmContainer(file) {
  const ext = mediaFileExtension(file);
  return ext === ".avi" || ext === ".flv" || ext === ".mkv" || ext === ".wma" || ext === ".wmv";
}

async function extractAudioStreamWithFfmpeg(file, arrayBuffer, outputFormat, options = {}) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(AUDIO_DECODER_WORKER, { type: "module" });
    const id = 1;
    const timeoutMs = options.ffmpegTimeoutMs || options.timeoutMs || 180000;
    let settled = false;
    let timeoutId = null;

    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      if (timeoutId) clearTimeout(timeoutId);
      worker.terminate();
      callback(value);
    };

    worker.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "log") {
        log(`Audio extract: ${message.message}`);
        return;
      }
      if (message.id !== id) return;
      if (message.type === "error") {
        finish(reject, new Error(message.message || "FFmpeg audio extract failed."));
        return;
      }
      if (message.type === "extracted") {
        finish(resolve, {
          audioBuffer: message.audioBuffer,
          outputFormat: message.outputFormat || outputFormat,
        });
      }
    });

    worker.addEventListener("error", (event) => {
      finish(reject, new Error(event.message || "FFmpeg audio extract worker crashed."));
    });

    timeoutId = setTimeout(() => {
      finish(reject, new Error(`FFmpeg audio extract timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs + 1000);

    worker.postMessage({
      id,
      type: "extract",
      fileName: file?.name || "input",
      bytes: arrayBuffer,
      outputFormat,
      timeoutMs,
    }, [arrayBuffer]);
  });
}

async function decodeAudioFileWithContainerExtract(file, arrayBuffer, options = {}) {
  const attempts = [
    { format: "mp3", decode: (buffer) => decodeAudioFileWithMpg123({ name: `${file?.name || "audio"}.mp3` }, buffer) },
    { format: "aac", decode: (buffer) => decodeAudioFileWithWebAudio({ name: `${file?.name || "audio"}.aac` }, buffer) },
  ];
  const errors = [];
  for (const attempt of attempts) {
    try {
      const extracted = await extractAudioStreamWithFfmpeg(
        file,
        arrayBuffer.slice(0),
        attempt.format,
        options
      );
      const decoded = await attempt.decode(extracted.audioBuffer);
      return {
        ...decoded,
        decoder: `${decoded.decoder}+demux-${attempt.format}`,
      };
    } catch (error) {
      errors.push(`${attempt.format}: ${error.message}`);
    }
  }
  throw new Error(`Không trích xuất được audio từ ${file?.name || "container"} (${errors.join("; ")}).`);
}

function downmixChannelDataToMono(channelData, samplesDecoded) {
  const channels = Array.isArray(channelData) ? channelData.filter(Boolean) : [];
  if (!channels.length) throw new Error("MP3 decoder returned no PCM channels.");
  const length = Math.max(0, Math.min(
    Number.isFinite(samplesDecoded) && samplesDecoded > 0 ? samplesDecoded : channels[0].length,
    ...channels.map((data) => data.length || 0)
  ));
  const mono = new Float32Array(length);
  for (let ch = 0; ch < channels.length; ch += 1) {
    const data = channels[ch];
    for (let i = 0; i < length; i += 1) mono[i] += data[i] / channels.length;
  }
  return mono;
}

async function decodeAudioFileWithMpg123(file, arrayBuffer) {
  const library = window["mpg123-decoder"];
  if (!library?.MPEGDecoder) throw new Error("mpg123-decoder is not loaded.");
  const decoder = new library.MPEGDecoder({ enableGapless: true });
  try {
    await decoder.ready;
    const decoded = decoder.decode(new Uint8Array(arrayBuffer));
    const mono = downmixChannelDataToMono(decoded.channelData, decoded.samplesDecoded);
    const sampleRate = Number(decoded.sampleRate) || VAD_SAMPLE_RATE;
    const samples = resampleLinear(mono, sampleRate, VAD_SAMPLE_RATE);
    return {
      samples,
      sampleRate: VAD_SAMPLE_RATE,
      originalSampleRate: sampleRate,
      channels: Array.isArray(decoded.channelData) ? decoded.channelData.length : null,
      duration: samples.length / VAD_SAMPLE_RATE,
      decoder: "mpg123-wasm",
      resampler: sampleRate === VAD_SAMPLE_RATE ? "none" : "js-linear",
    };
  } finally {
    try {
      decoder.free();
    } catch (_) {
      // Best effort release of the MP3 decoder.
    }
  }
}

function decodeAudioDataCompat(context, buffer) {
  const copy = buffer.slice(0);
  const decoded = context.decodeAudioData(copy);
  if (decoded && typeof decoded.then === "function") return decoded;
  return new Promise((resolve, reject) => {
    context.decodeAudioData(copy, resolve, reject);
  });
}

async function decodeAudioFileWithWebAudio(file, arrayBuffer) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("WebAudio decodeAudioData is not available.");
  const context = new AudioContextCtor();
  try {
    const audioBuffer = await decodeAudioDataCompat(context, arrayBuffer);
    const mono = downmixToMono(audioBuffer);
    const samples = resampleLinear(mono, audioBuffer.sampleRate || VAD_SAMPLE_RATE, VAD_SAMPLE_RATE);
    return {
      samples,
      sampleRate: VAD_SAMPLE_RATE,
      originalSampleRate: audioBuffer.sampleRate || null,
      channels: audioBuffer.numberOfChannels || null,
      duration: samples.length / VAD_SAMPLE_RATE,
      decoder: `webaudio:${mediaFileExtension(file).slice(1) || "media"}`,
      resampler: audioBuffer.sampleRate === VAD_SAMPLE_RATE ? "none" : "js-linear",
    };
  } finally {
    try {
      await context.close();
    } catch (_) {
      // Closing AudioContext is a best-effort resource release.
    }
  }
}

function decodeAudioFileWithFfmpegWorker(file, arrayBuffer, options = {}) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(AUDIO_DECODER_WORKER, { type: "module" });
    const id = 1;
    const timeoutMs = options.ffmpegTimeoutMs || options.timeoutMs || 180000;
    let settled = false;
    let timeoutId = null;

    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      if (timeoutId) clearTimeout(timeoutId);
      worker.terminate();
      callback(value);
    };

    worker.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "log") {
        log(`Audio decoder: ${message.message}`);
        return;
      }
      if (message.type === "progress") {
        if (typeof options.progress === "function") {
          options.progress(Number(message.progress) || 0);
        }
        return;
      }
      if (message.id !== id) return;
      if (message.type === "error") {
        finish(reject, new Error(message.message || "FFmpeg audio decode failed."));
        return;
      }
      if (message.type === "decoded") {
        const samples = new Float32Array(message.pcmBuffer);
        finish(resolve, {
          samples,
          sampleRate: message.sampleRate || VAD_SAMPLE_RATE,
          originalSampleRate: message.originalSampleRate || null,
          channels: message.channels || null,
          duration: message.duration || samples.length / VAD_SAMPLE_RATE,
          decoder: message.decoder || "ffmpeg-wasm",
          resampler: message.resampler || AUDIO_DECODER_RESAMPLER,
        });
      }
    });

    worker.addEventListener("error", (event) => {
      finish(reject, new Error(event.message || "FFmpeg audio decoder crashed."));
    });

    timeoutId = setTimeout(() => {
      finish(reject, new Error(`FFmpeg audio decode timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs + 1000);

    worker.postMessage({
      id,
      type: "decode",
      fileName: file?.name || "input",
      bytes: arrayBuffer,
      timeoutMs,
      resampler: AUDIO_DECODER_RESAMPLER,
    }, [arrayBuffer]);
  });
}

async function attachEditorAudioFile(file, options = {}) {
  if (!editorState || !file) return;
  const previousLibraryItemId = editorState.libraryItemId;
  try {
    revokeEditorPreviewAudioUrl();
    if (editorState.audioUrl) URL.revokeObjectURL(editorState.audioUrl);
    editorState.sourceFile = file;
    editorState.fileName = file.name;
    editorState.fileSize = file.size;
    editorState.libraryItemId = options.preserveLibraryItem ? previousLibraryItemId : editorState.libraryItemId;
    editorState.audioUrl = URL.createObjectURL(file);
    setEditorAudioUrl(editorState.audioUrl, editorState.duration, file.name);
    renderEditor();
    log(`Attached source audio for editor playback: ${file.name}.`);
  } catch (error) {
    log(`Attach audio failed: ${error.message}`);
  }
}

async function ensureEditorSamplesForProcessing() {
  if (editorState?.samples?.length) return editorState.samples;
  const file = editorState?.sourceFile || selectedAudioFile;
  if (!file) throw new Error("No loaded editor audio for processing.");
  log(`Decoding source audio for editor processing: ${file.name || "audio"}.`);
  const arrayBuffer = await file.arrayBuffer();
  const decoded = await decodeAudioFileWithFfmpeg(file, arrayBuffer);
  editorState.samples = decoded.samples;
  editorState.duration = decoded.samples.length / VAD_SAMPLE_RATE;
  lastAudioPcm = decoded.samples;
  log(`Decoded editor processing audio: ${file.name || "audio"} (${decoded.decoder}), ${editorState.duration.toFixed(2)}s.`);
  return editorState.samples;
}

function createSpeakerBadge(speaker, segmentIndex) {
  const meta = speakerMetaFor(speaker);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "editor-speaker-badge";
  button.dataset.editorAction = "rename-speaker";
  button.dataset.speakerId = String(normalizeSpeakerId(speaker));
  if (Number.isInteger(segmentIndex)) button.dataset.index = String(segmentIndex);
  button.style.backgroundColor = meta.color;
  button.textContent = meta.name;
  return button;
}

function appendHighlightedText(parent, text, query) {
  const source = text || "";
  const needle = (query || "").trim().toLocaleLowerCase("vi-VN");
  if (!needle) {
    parent.appendChild(document.createTextNode(source));
    return;
  }

  const haystack = source.toLocaleLowerCase("vi-VN");
  let offset = 0;
  while (offset < source.length) {
    const index = haystack.indexOf(needle, offset);
    if (index < 0) break;
    if (index > offset) parent.appendChild(document.createTextNode(source.slice(offset, index)));
    const mark = document.createElement("mark");
    mark.textContent = source.slice(index, index + needle.length);
    parent.appendChild(mark);
    offset = index + needle.length;
  }
  if (offset < source.length) parent.appendChild(document.createTextNode(source.slice(offset)));
}

function renderEmptyState(root, message) {
  const node = document.createElement("div");
  node.className = "empty-state";
  node.textContent = message;
  root.appendChild(node);
}

function renderEditorSegmentRow(root, segment, index, options = {}) {
  const row = document.createElement("article");
  row.className = "editor-segment";
  row.dataset.index = String(index);
  row.dataset.editorSegmentIndex = String(index);
  if (index === editorState.activeSegmentIndex) row.classList.add("active");
  if (editorState.searchMatches.some((match) => match.segmentIndex === index)) row.classList.add("search-hit");

  const head = document.createElement("div");
  head.className = "editor-segment-head";
  const left = document.createElement("div");
  left.className = "editor-segment-actions";
  left.appendChild(createSpeakerBadge(segment.speaker, index));
  const time = document.createElement("span");
  time.className = "editor-time";
  time.textContent = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;
  left.appendChild(time);
  head.appendChild(left);

  const actions = document.createElement("div");
  actions.className = "editor-segment-actions";
  const split = document.createElement("button");
  split.type = "button";
  split.dataset.editorAction = "split-block";
  split.dataset.index = String(index);
  split.textContent = "Split block";
  actions.appendChild(split);

  const mergePrev = document.createElement("button");
  mergePrev.type = "button";
  mergePrev.dataset.editorAction = "merge-prev";
  mergePrev.dataset.index = String(index);
  mergePrev.textContent = "Merge prev";
  actions.appendChild(mergePrev);

  const mergeNext = document.createElement("button");
  mergeNext.type = "button";
  mergeNext.dataset.editorAction = "merge-next";
  mergeNext.dataset.index = String(index);
  mergeNext.textContent = "Merge next";
  actions.appendChild(mergeNext);
  head.appendChild(actions);
  row.appendChild(head);

  const text = document.createElement("div");
  text.className = options.compact ? "editor-block-text" : "editor-segment-text";
  appendHighlightedText(text, segment.text, editorState.searchQuery);
  row.appendChild(text);
  root.appendChild(row);
}

function editorSpeakerBlocks() {
  const blocks = [];
  for (let i = 0; i < editorState.segments.length; i += 1) {
    const segment = editorState.segments[i];
    const previous = blocks[blocks.length - 1];
    if (previous && previous.speaker === segment.speaker) {
      previous.end = segment.end;
      previous.endIndex = i;
      previous.text = `${previous.text} ${segment.text}`.trim();
    } else {
      blocks.push({
        start: segment.start,
        end: segment.end,
        startIndex: i,
        endIndex: i,
        speaker: segment.speaker,
        text: segment.text,
      });
    }
  }
  return blocks;
}

function renderEditorTranscript() {
  const root = $("editor-transcript-list");
  if (!root || !editorState) return;
  root.textContent = "";
  if (!editorState.segments.length) {
    renderEmptyState(root, "No transcript segments.");
    return;
  }
  editorState.segments.forEach((segment, index) => renderEditorSegmentRow(root, segment, index));
}

function renderEditorSpeakerBlocks() {
  const root = $("editor-speaker-blocks");
  if (!root || !editorState) return;
  root.textContent = "";
  if (!editorState.speakerDiarizationEnabled && !editorState.rawSpeakerSegments.length) {
    renderEmptyState(root, "Speaker diarization is off. Enable it and rerun to create speaker blocks.");
    return;
  }
  if (editorState.speakerDiarizationEnabled && !editorState.rawSpeakerSegments.length) {
    renderEmptyState(root, "Speaker diarization produced no raw speaker turns.");
    return;
  }
  const blocks = editorSpeakerBlocks();
  if (!blocks.length) {
    renderEmptyState(root, "No speaker blocks.");
    return;
  }

  for (const block of blocks) {
    renderEditorSegmentRow(root, {
      start: block.start,
      end: block.end,
      speaker: block.speaker,
      text: `${block.endIndex - block.startIndex + 1} sentence(s): ${block.text}`,
    }, block.startIndex, { compact: true });
  }
}

function renderEditorRawSpeakers() {
  const root = $("editor-raw-speakers");
  if (!root || !editorState) return;
  root.textContent = "";
  if (!editorState.rawSpeakerSegments.length) {
    renderEmptyState(root, "No raw speaker segments.");
    return;
  }

  editorState.rawSpeakerSegments.forEach((segment, index) => {
    const row = document.createElement("article");
    row.className = "editor-segment";
    row.dataset.rawIndex = String(index);
    const head = document.createElement("div");
    head.className = "editor-segment-head";
    const left = document.createElement("div");
    left.className = "editor-segment-actions";
    left.appendChild(createSpeakerBadge(segment.speaker));
    const time = document.createElement("span");
    time.className = "editor-time";
    time.textContent = `${formatTime(segment.start)} - ${formatTime(segment.end)}`;
    left.appendChild(time);
    head.appendChild(left);
    row.appendChild(head);

    const text = document.createElement("div");
    text.className = "editor-block-text";
    text.textContent = `Raw turn ${index + 1}`;
    row.appendChild(text);
    root.appendChild(row);
  });
}

function normalizeVietnamese(text) {
  return String(text || "")
    .toLocaleLowerCase("vi-VN")
    .replace(/đ/g, "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function mapNormToOrig(original, normIdx) {
  if (normIdx <= 0) return 0;
  let baseCount = 0;
  for (let i = 0; i < original.length; i += 1) {
    const c = original[i];
    if (c.toLocaleLowerCase("vi-VN") === "đ") {
      baseCount += 1;
      if (baseCount > normIdx) return i;
      continue;
    }
    const decomposed = c.normalize("NFD");
    const firstCode = decomposed.charCodeAt(0);
    const isBase = decomposed.length === 0 || !(firstCode >= 0x0300 && firstCode <= 0x036f);
    if (isBase) baseCount += 1;
    if (baseCount > normIdx) return i;
  }
  return original.length;
}

function refreshEditorSearchMatches() {
  if (!editorState) return;
  const query = editorState.searchQuery.trim();
  editorState.searchPieces = [];
  editorState.searchMatches = [];

  if (query) {
    const boundaries = [];
    const parts = [];
    let globalPos = 0;
    editorState.segments.forEach((segment, index) => {
      const text = segment.text || "";
      boundaries.push({
        index,
        globalStart: globalPos,
        globalEnd: globalPos + text.length,
        text,
      });
      parts.push(text);
      globalPos += text.length + 1;
      parts.push(" ");
    });

    const concatenated = parts.join("");
    const queryLower = query.toLocaleLowerCase("vi-VN");
    const queryNorm = normalizeVietnamese(query);
    const lower = concatenated.toLocaleLowerCase("vi-VN");
    const normalized = normalizeVietnamese(concatenated);
    const seenLogical = new Set();
    let logicalIdx = 0;

    const mapGlobalToSegment = (position) => {
      for (const boundary of boundaries) {
        if (boundary.globalStart <= position && position < boundary.globalEnd) {
          return boundary;
        }
        if (position === boundary.globalEnd) return boundary;
      }
      return null;
    };

    const addMatch = (startGlobal, matchLength, normalizedMatch = false) => {
      if (matchLength <= 0) return;
      const endGlobal = startGlobal + matchLength;
      const first = mapGlobalToSegment(startGlobal);
      const last = mapGlobalToSegment(endGlobal - 1);
      if (!first || !last) return;

      const key = `${first.index}:${startGlobal}:${endGlobal}`;
      if (seenLogical.has(key)) return;
      seenLogical.add(key);

      const currentLogical = logicalIdx++;
      editorState.searchMatches.push({
        segmentIndex: first.index,
        charStart: Math.max(0, startGlobal - first.globalStart),
        logicalIdx: currentLogical,
      });

      for (const boundary of boundaries) {
        if (boundary.index < first.index || boundary.index > last.index) continue;
        const localStart = boundary.index === first.index
          ? Math.max(0, startGlobal - boundary.globalStart)
          : 0;
        const localEnd = boundary.index === last.index
          ? Math.min(boundary.text.length, endGlobal - boundary.globalStart)
          : boundary.text.length;
        if (localEnd <= localStart) continue;

        const charStart = normalizedMatch ? mapNormToOrig(boundary.text, localStart) : localStart;
        const charEnd = normalizedMatch ? mapNormToOrig(boundary.text, localEnd) : localEnd;
        if (charEnd <= charStart) continue;

        editorState.searchPieces.push({
          segmentIndex: boundary.index,
          charStart,
          charEnd,
          logicalIdx: currentLogical,
        });
      }
    };

    let pos = 0;
    while ((pos = lower.indexOf(queryLower, pos)) !== -1) {
      addMatch(pos, query.length, false);
      pos += 1;
    }

    if (queryNorm) {
      pos = 0;
      while ((pos = normalized.indexOf(queryNorm, pos)) !== -1) {
        addMatch(pos, queryNorm.length, true);
        pos += 1;
      }
    }

    editorState.searchMatches.sort((a, b) => a.segmentIndex - b.segmentIndex || a.charStart - b.charStart);
    const logicalOrder = new Map();
    editorState.searchMatches.forEach((match, index) => {
      logicalOrder.set(match.logicalIdx, index);
      match.logicalIdx = index;
    });
    editorState.searchPieces.forEach((piece) => {
      piece.logicalIdx = logicalOrder.get(piece.logicalIdx) ?? piece.logicalIdx;
    });
    editorState.searchPieces.sort((a, b) => a.segmentIndex - b.segmentIndex || a.charStart - b.charStart);
  }

  if (!editorState.searchMatches.length) {
    editorState.searchIndex = -1;
  } else if (editorState.searchIndex < 0) {
    editorState.searchIndex = 0;
  } else if (editorState.searchIndex >= editorState.searchMatches.length) {
    editorState.searchIndex = 0;
  }
}

function renderEditorTabs() {
  if (!editorState) return;
  document.querySelectorAll("[data-editor-tab]").forEach((button) => {
    const active = button.dataset.editorTab === editorState.activeTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-editor-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.editorPanel !== editorState.activeTab;
  });
}

function resultQualityColor(score, thresholds) {
  for (const [value, color] of thresholds) {
    if (score >= value) return color;
  }
  return thresholds[thresholds.length - 1][1];
}

function renderResultQualityStrip(quality) {
  const strip = $("quality-strip");
  if (!strip) return;
  strip.textContent = "";
  if (!quality) {
    strip.style.display = "none";
    return;
  }

  const dnsThresh = [[4.0, "#28a745"], [3.0, "#5cb85c"], [2.0, "#ffc107"], [0, "#dc3545"]];
  const confThresh = [[0.80, "#28a745"], [0.60, "#ffc107"], [0, "#dc3545"]];
  const items = [];

  function addItem(label, value, color) {
    const item = document.createElement("span");
    item.className = "qs-item";
    item.appendChild(document.createTextNode(`${label} `));
    const val = document.createElement("span");
    val.className = "qs-val";
    val.style.color = color;
    val.textContent = value;
    item.appendChild(val);
    items.push(item);
  }

  if (quality.dnsmos_sig !== undefined) {
    addItem("Gi\u1ecdng n\u00f3i", `${Number(quality.dnsmos_sig).toFixed(1)}/5`, resultQualityColor(Number(quality.dnsmos_sig), dnsThresh));
  }
  if (quality.dnsmos_bak !== undefined) {
    addItem("Nhi\u1ec5u n\u1ec1n", `${Number(quality.dnsmos_bak).toFixed(1)}/5`, resultQualityColor(Number(quality.dnsmos_bak), dnsThresh));
  }
  if (quality.dnsmos_ovrl !== undefined) {
    addItem("T\u1ed5ng th\u1ec3", `${Number(quality.dnsmos_ovrl).toFixed(1)}/5`, resultQualityColor(Number(quality.dnsmos_ovrl), dnsThresh));
  }
  if (quality.asr_confidence !== undefined) {
    const confidence = Number(quality.asr_confidence);
    addItem("M\u1ee9c \u0111\u1ed9 t\u1ef1 tin d\u1ecbch ch\u00ednh x\u00e1c", `${(confidence * 100).toFixed(1)}%`, resultQualityColor(confidence, confThresh));
  }

  if (!items.length) {
    strip.style.display = "none";
    return;
  }

  const label = document.createElement("span");
  label.className = "qs-label";
  label.textContent = "Ch\u1ea5t l\u01b0\u1ee3ng:";
  strip.appendChild(label);
  items.forEach((item, index) => {
    if (index > 0) {
      const sep = document.createElement("span");
      sep.className = "qs-sep";
      sep.textContent = "\u00b7";
      strip.appendChild(sep);
    }
    strip.appendChild(item);
  });
  strip.style.display = "flex";
}
function renderResultTiming() {
  const el = $("result-timing");
  if (!el || !editorState) return;
  const sourceTiming = editorState.timing || {};
  const timing = {
    preprocessing: sourceTiming.preprocessing,
    transcription_detail: sourceTiming.transcription_detail ?? sourceTiming.asr ?? editorState.asr?.elapsed,
    diarization: sourceTiming.diarization ?? editorState.diarization?.elapsed,
    punctuation: sourceTiming.punctuation ?? editorState.punctuation?.elapsed,
    overlap_separation: sourceTiming.overlap_separation ?? sourceTiming.overlap ?? editorState.overlap?.elapsed,
    total: sourceTiming.total,
  };
  const labels = {
    preprocessing: "PreProcessing",
    transcription_detail: "ASR",
    diarization: "Ph\u00e2n t\u00e1ch ng\u01b0\u1eddi n\u00f3i",
    punctuation: "D\u1ea5u c\u00e2u",
    overlap_separation: "T\u00e1ch overlap",
    total: "T\u1ed5ng th\u1eddi gian",
  };
  el.textContent = "";
  for (const [key, label] of Object.entries(labels)) {
    const value = Number(timing[key]);
    if (!Number.isFinite(value) || value <= 0) continue;
    const item = document.createElement("div");
    item.className = key === "total" ? "timing-item timing-total" : "timing-item";
    if (key === "total") {
      item.style.fontWeight = "bold";
      item.style.marginLeft = "8px";
      item.style.color = "var(--primary-color, #2ea3f2)";
    }
    const name = document.createElement("span");
    name.className = "timing-label";
    name.textContent = `${label}:`;
    const number = document.createElement("span");
    number.className = "timing-value";
    number.textContent = `${value.toFixed(1)}s`;
    item.append(name, number);
    el.appendChild(item);
  }
  el.style.display = el.children.length ? "flex" : "none";
}
function appendResultHighlightedText(parent, text, segmentIndex) {
  const source = text || "";
  const pieces = (editorState?.searchPieces || [])
    .filter((piece) => piece.segmentIndex === segmentIndex)
    .sort((a, b) => a.charStart - b.charStart);
  if (!pieces.length) {
    parent.appendChild(document.createTextNode(source));
    return;
  }

  let offset = 0;
  for (const piece of pieces) {
    const start = Math.max(0, Math.min(source.length, piece.charStart));
    const end = Math.max(start, Math.min(source.length, piece.charEnd));
    if (start < offset) continue;
    if (start > offset) parent.appendChild(document.createTextNode(source.slice(offset, start)));
    const mark = document.createElement("span");
    mark.className = piece.logicalIdx === editorState.searchIndex ? "search-current-match" : "search-match";
    mark.textContent = source.slice(start, end);
    parent.appendChild(mark);
    offset = end;
  }
  if (offset < source.length) {
    parent.appendChild(document.createTextNode(source.slice(offset)));
  }
}

function createResultSegmentSpan(segment, index) {
  const span = document.createElement("span");
  span.className = "seg-span";
  span.dataset.seg = String(index);
  span.dataset.editorSegmentIndex = String(index);
  if (index === editorState.activeSegmentIndex) span.classList.add("seg-highlight");
  appendResultHighlightedText(span, segment.text || "", index);
  return span;
}

function renderResultSpeakerView(root) {
  const blocks = editorSpeakerBlocks();
  if (!blocks.length) {
    renderEmptyState(root, "Không có nội dung nhận dạng.");
    return;
  }
  for (const block of blocks) {
    const speaker = normalizeSpeakerId(block.speaker);
    const meta = speakerMetaFor(speaker);
    const node = document.createElement("div");
    node.className = "speaker-block";
    node.dataset.block = String(block.startIndex);
    node.dataset.speakerId = String(speaker);
    node.style.borderLeftColor = meta.color;

    const label = document.createElement("div");
    label.className = "speaker-label";
    label.dataset.spk = String(speaker);
    label.dataset.blockIdx = String(block.startIndex);
    label.style.color = meta.color;
    label.textContent = `${meta.name}:`;
    node.appendChild(label);

    const text = document.createElement("div");
    text.className = "speaker-text";
    for (let i = block.startIndex; i <= block.endIndex; i += 1) {
      text.appendChild(createResultSegmentSpan(editorState.segments[i], i));
      text.appendChild(document.createTextNode(" "));
    }
    node.appendChild(text);
    root.appendChild(node);
  }
}

function renderResultPlainView(root) {
  const container = document.createElement("div");
  container.className = "plain-text";
  editorState.segments.forEach((segment, index) => {
    container.appendChild(createResultSegmentSpan(segment, index));
    container.appendChild(document.createTextNode(" "));
  });
  root.appendChild(container);
}

function renderServerStyleResult() {
  const root = $("result-content");
  if (!root || !editorState) return;
  root.textContent = "";
  if (!editorState.segments.length) {
    renderEmptyState(root, "Không có nội dung nhận dạng.");
  } else if (editorState.speakerDiarizationEnabled || editorState.rawSpeakerSegments.length) {
    renderResultSpeakerView(root);
  } else {
    renderResultPlainView(root);
  }

  const panel = $("result-panel");
  if (panel) panel.style.display = "flex";
  renderResultQualityStrip(editorState.qualityInfo);
  renderResultTiming();
}

function renderEditor() {
  if (!editorState) return;
  refreshEditorSearchMatches();
  const searchStatus = $("search-count");
  if (searchStatus) searchStatus.textContent = editorState.searchQuery
    ? (editorState.searchMatches.length
        ? `${Math.max(0, editorState.searchIndex) + 1}/${editorState.searchMatches.length}`
        : "0/0")
    : "";
  const rerun = $("btn-rerun-diarization");
  if (rerun) rerun.disabled = !(editorState.samples?.length || editorState.sourceFile);
  ["btn-save-json", "btn-copy-text", "btn-export-transcript", "btn-export-debug-log"].forEach((id) => {
    const button = $(id);
    if (button) button.disabled = !editorState.segments.length;
  });
  const attach = $("btn-attach-audio");
  if (attach) attach.disabled = false;

  renderServerStyleResult();
  applyEditorActiveClasses(false);
}

function applyEditorActiveClasses(allowScroll = false) {
  if (!editorState) return;
  document.querySelectorAll("[data-editor-segment-index]").forEach((node) => {
    const active = Number(node.dataset.editorSegmentIndex) === editorState.activeSegmentIndex;
    node.classList.toggle("active", active);
    node.classList.toggle("seg-highlight", active);
    if (active && allowScroll) {
      const now = performance.now();
      if (now - editorLastAutoScrollAt > 900) {
        editorLastAutoScrollAt = now;
        node.scrollIntoView({ block: "nearest" });
      }
    }
  });
}

function editorSegmentIndexAtTime(time, preferredIndex = null) {
  if (!editorState?.segments?.length) return;
  const epsilon = 0.035;
  const preferred = editorState.segments[preferredIndex];
  if (preferred && time + epsilon >= preferred.start && time < preferred.end + epsilon) {
    return preferredIndex;
  }

  let index = -1;
  for (let i = 0; i < editorState.segments.length; i += 1) {
    const segment = editorState.segments[i];
    if (Number.isFinite(segment.start) && segment.start <= time + epsilon) index = i;
  }
  return index >= 0 ? index : 0;
}

function setEditorActiveByTime(time, allowScroll = false, preferredIndex = null) {
  if (!editorState?.segments?.length) return;
  const index = editorSegmentIndexAtTime(time, preferredIndex ?? editorState.activeSegmentIndex);
  if (index !== editorState.activeSegmentIndex) {
    const previousIndex = editorState.activeSegmentIndex;
    editorState.activeSegmentIndex = index;
    appendDebugLog("editor.active_segment_changed", {
      time: debugRound(time),
      previousIndex,
      activeSegmentIndex: index,
      preferredIndex,
      allowScroll,
      segment: debugSegmentSnapshot(index),
      audio: debugAudioSnapshot(),
    });
    applyEditorActiveClasses(allowScroll);
  }
}

function seekEditorTo(seconds, autoplay = false, preferredIndex = null) {
  const audio = $("editor-audio");
  if (!audio || !Number.isFinite(seconds)) return;
  const before = debugAudioSnapshot();
  audio.currentTime = Math.max(0, Math.min(seconds, editorState?.duration || seconds));
  setEditorActiveByTime(audio.currentTime, true, preferredIndex);
  appendDebugLog("player.seek_to", {
    requestedSeconds: debugRound(seconds),
    appliedCurrentTime: debugRound(audio.currentTime),
    preferredIndex,
    activeSegmentIndex: editorState?.activeSegmentIndex ?? null,
    autoplay,
    before,
    after: debugAudioSnapshot(),
  });
  if (autoplay) audio.play().catch(() => null);
}

function editorSegmentPlaybackStart(index) {
  const segment = editorState?.segments?.[index];
  if (!segment) return NaN;
  const rawStart = Array.isArray(segment.raw_words) && segment.raw_words.length
    ? Number(segment.raw_words[0]?.start)
    : NaN;
  const segmentStart = Number(segment.start);
  const start = Number.isFinite(rawStart) ? rawStart : segmentStart;
  const duration = Number(editorState?.duration);
  return Math.max(0, Math.min(Number.isFinite(duration) ? duration : start, start));
}

function seekEditorSegment(index, autoplay = true) {
  if (!editorState?.segments[index]) return;
  const before = debugAudioSnapshot();
  const segmentBefore = debugSegmentSnapshot(index);
  editorState.activeSegmentIndex = index;
  applyEditorActiveClasses(true);
  const playbackStart = editorSegmentPlaybackStart(index);
  appendDebugLog("editor.segment_click", {
    index,
    autoplay,
    playbackStart: debugRound(playbackStart),
    segment: segmentBefore,
    activeBeforeSeek: editorState.activeSegmentIndex,
    audioBefore: before,
  });
  seekEditorTo(playbackStart, autoplay, index);
  appendDebugLog("editor.segment_click_after_seek", {
    index,
    activeSegmentIndex: editorState.activeSegmentIndex,
    segment: debugSegmentSnapshot(index),
    audioAfter: debugAudioSnapshot(),
  });
}

function findEditorSpeakerBlock(segmentIndex) {
  if (!editorState?.segments[segmentIndex]) return null;
  const speaker = editorState.segments[segmentIndex].speaker;
  let start = segmentIndex;
  let end = segmentIndex;
  while (start > 0 && editorState.segments[start - 1].speaker === speaker) start -= 1;
  while (end + 1 < editorState.segments.length && editorState.segments[end + 1].speaker === speaker) end += 1;
  return { start, end, speaker };
}

function nextEditorSpeakerId() {
  const ids = collectEditorSpeakerIds(editorState?.segments || [], editorState?.rawSpeakerSegments || []);
  return ids.length ? Math.max(...ids) + 1 : 0;
}

function openSpeakerDialog(context) {
  if (!editorState) return;
  const dialog = $("speaker-dialog");
  if (!dialog) return;
  speakerDialogContext = context;

  const title = $("speaker-dialog-title");
  const name = $("speaker-dialog-name");
  const color = $("speaker-dialog-color");
  if (context.mode === "split") {
    const id = nextEditorSpeakerId();
    if (title) title.textContent = "Split speaker block";
    if (name) name.value = defaultSpeakerName(id);
    if (color) color.value = SPEAKER_COLORS[id % SPEAKER_COLORS.length];
  } else {
    const meta = speakerMetaFor(context.speaker);
    if (title) title.textContent = "Rename speaker";
    if (name) name.value = meta.name;
    if (color) color.value = meta.color;
  }

  if (dialog.showModal) dialog.showModal();
  else dialog.setAttribute("open", "");
}

function closeSpeakerDialog() {
  const dialog = $("speaker-dialog");
  if (!dialog) return;
  if (dialog.close) dialog.close();
  else dialog.removeAttribute("open");
}

function applySpeakerDialog() {
  if (!editorState || !speakerDialogContext) return;
  const nameInput = $("speaker-dialog-name");
  const colorInput = $("speaker-dialog-color");
  const name = (nameInput?.value || "").trim();
  const color = normalizedColor(colorInput?.value, SPEAKER_COLORS[0]);

  if (speakerDialogContext.mode === "split") {
    const block = findEditorSpeakerBlock(speakerDialogContext.segmentIndex);
    if (!block) return;
    const id = nextEditorSpeakerId();
    editorState.speakers[id] = {
      name: name || defaultSpeakerName(id),
      color: normalizedColor(color, SPEAKER_COLORS[id % SPEAKER_COLORS.length]),
    };
    for (let i = block.start; i <= block.end; i += 1) {
      editorState.segments[i].speaker = id;
    }
  } else {
    const id = normalizeSpeakerId(speakerDialogContext.speaker);
    const meta = speakerMetaFor(id);
    meta.name = name || defaultSpeakerName(id);
    meta.color = normalizedColor(color, meta.color);
  }

  speakerDialogContext = null;
  closeSpeakerDialog();
  syncEditorSpeakers(editorState.speakers);
  renderEditor();
  scheduleLibraryResultAutosave();
}

function mergeEditorSpeakerBlock(segmentIndex, direction) {
  if (!editorState) return;
  const block = findEditorSpeakerBlock(segmentIndex);
  if (!block) return;
  const targetIndex = direction === "prev" ? block.start - 1 : block.end + 1;
  const target = editorState.segments[targetIndex];
  if (!target) {
    log(direction === "prev" ? "No previous speaker block to merge." : "No next speaker block to merge.");
    return;
  }
  for (let i = block.start; i <= block.end; i += 1) {
    editorState.segments[i].speaker = target.speaker;
  }
  syncEditorSpeakers(editorState.speakers);
  renderEditor();
  scheduleLibraryResultAutosave();
}

function applyEditorDiarization(diarization) {
  if (!editorState) return;
  const previousSpeakers = editorState.speakers || {};
  editorState.diarization = diarization;
  editorState.rawSpeakerSegments = normalizeDiarizationSegments(diarization?.segments || []);
  for (const segment of editorState.segments) {
    const words = Array.isArray(segment.raw_words) ? segment.raw_words : [];
    segment.speaker = words.length
      ? dominantSpeakerForWords(words, editorState.rawSpeakerSegments, segment.speaker)
      : dominantSpeakerForRange(segment.start, segment.end, editorState.rawSpeakerSegments);
  }
  syncEditorSpeakers(previousSpeakers);
  renderDiarization(editorState.rawSpeakerSegments);
  renderEditor();
  scheduleLibraryResultAutosave();
}

async function rerunEditorDiarization() {
  if (!editorState) return;
  const button = $("btn-rerun-diarization");
  if (button) button.disabled = true;
  setPipelineControlsDisabled(true);
  try {
    const samples = await ensureEditorSamplesForProcessing();
    const options = getPipelineOptions();
    setPipelineProgress("Rerun diarization", 82);
    log(`Rerun diarization on editor result with ${options.speakerModel}.`);
    const asrWords = Array.isArray(editorState.asr?.words) && editorState.asr.words.length
      ? editorState.asr.words
      : editorState.segments.flatMap((segment) => Array.isArray(segment.raw_words) ? segment.raw_words : []);
    const diarization = await runDiarization(samples, {
      speakerModel: options.speakerModel,
      numSpeakers: options.numSpeakers,
      asrWords,
      progress: (done, total) => {
        const ratio = Number.isFinite(total) ? done / Math.max(1, total) : done;
        setPipelineProgress("Rerun diarization", 82 + ratio * 17);
      },
    });
    applyEditorDiarization(diarization);
    setPipelineProgress("Diarization updated", 100);
    log(`Editor diarization updated: ${diarization.segments.length} turn(s), ${diarization.speakers} speaker(s).`);
  } catch (error) {
    setPipelineProgress("Rerun failed", 100);
    log(`Rerun diarization failed: ${error.message}`);
  } finally {
    setPipelineControlsDisabled(false);
    syncPipelineControls();
    updateProcessButtonState();
    if (button) button.disabled = !editorState?.samples?.length;
  }
}

async function getPyannoteSpeechRegions(samples, options = {}) {
  const duration = samples.length / VAD_SAMPLE_RATE;
  const segmentation = await runPyannoteSegmentationLogits(samples, {
    ...options,
    starts: diarizationStarts(samples.length),
    algorithm: "campp_speech_regions",
  });
  const { logitsData, starts, chunks, numFrames, numClasses } = segmentation;
  const frameSeconds = DIAR_CHUNK_SECONDS / numFrames;
  const speechCount = [];
  const totalCount = [];
  const overlapCount = [];

  for (let c = 0; c < chunks; c += 1) {
    const frames = decodeDiarizationLogits(
      logitsData,
      c * numFrames * numClasses,
      numFrames,
      numClasses
    );
    const chunkStartSec = starts[c] / VAD_SAMPLE_RATE;
    for (let f = 0; f < frames.length; f += 1) {
      const outFrame = Math.floor((chunkStartSec + f * frameSeconds) / frameSeconds);
      const active = frames[f].reduce((sum, value) => sum + value, 0);
      speechCount[outFrame] = (speechCount[outFrame] || 0) + (active > 0 ? 1 : 0);
      overlapCount[outFrame] = (overlapCount[outFrame] || 0) + (active >= 2 ? 1 : 0);
      totalCount[outFrame] = (totalCount[outFrame] || 0) + 1;
    }
  }

  const camppIsSpeech = totalCount.map((count, index) => count > 0 && (speechCount[index] || 0) / count > 0.5);
  const camppIsOverlap = totalCount.map((count, index) => count > 0 && (overlapCount[index] || 0) / count > 0.5);
  const camppRegions = frameMaskToRegions(camppIsSpeech, frameSeconds, duration, 0.25, 0.1);
  const camppOverlapRegions = frameMaskToRegions(camppIsOverlap, frameSeconds, duration, 0.3, 0.1);
  return {
    regions: camppRegions.length ? camppRegions : [{ start: 0, end: duration }],
    overlapRegions: camppOverlapRegions,
    chunks,
    provider: segmentation.provider,
    batchSize: segmentation.batchSize,
    batchTuning: segmentation.batchTuning || options.batchTuning || null,
  };

}

function frameMaskToRegions(mask, frameSeconds, duration, minOn, minOff) {
  const regions = [];
  let active = false;
  let start = 0;

  for (let i = 0; i < mask.length; i += 1) {
    const time = Math.min(duration, i * frameSeconds);
    if (mask[i] && !active) {
      start = time;
      active = true;
    } else if (!mask[i] && active) {
      if (time - start >= minOn) {
        regions.push({ start, end: time });
      }
      active = false;
    }
  }
  if (active && duration - start >= minOn) {
    regions.push({ start, end: duration });
  }

  if (!regions.length || minOff <= 0) return regions;
  const merged = [regions[0]];
  for (const region of regions.slice(1)) {
    const previous = merged[merged.length - 1];
    if (region.start - previous.end < minOff) {
      previous.end = region.end;
    } else {
      merged.push({ ...region });
    }
  }
  return merged;
}

async function extractCamppEmbeddings(samples, speechRegions, options = {}) {
  await ensureCamppReady();
  const started = performance.now();
  const windowFrames = Math.round((CAMPP_WINDOW_SECONDS * 1000) / 10);
  const stepFrames = Math.round((CAMPP_STEP_SECONDS * 1000) / 10);
  const batchSizeConfig = options.batchSize || CAMPP_BATCH_SIZE;
  const windows = [];

  for (const region of speechRegions) {
    const startSample = Math.max(0, Math.floor(region.start * VAD_SAMPLE_RATE));
    const endSample = Math.min(samples.length, Math.ceil(region.end * VAD_SAMPLE_RATE));
    if (endSample - startSample < CAMPP_FRAME_LENGTH) continue;
    const regionFbank = computeCamppFbank(samples.subarray(startSample, endSample));
    const nFrames = regionFbank.length;
    if (nFrames < 10) continue;

    if (nFrames < windowFrames) {
      windows.push({
        fbank: regionFbank,
        startFrame: 0,
        frameCount: nFrames,
        start: region.start,
        end: region.end,
      });
      continue;
    }

    let position = 0;
    while (position + windowFrames < nFrames) {
      const start = region.start + position * 0.01;
      windows.push({
        fbank: regionFbank,
        startFrame: position,
        frameCount: windowFrames,
        start,
        end: start + CAMPP_WINDOW_SECONDS,
      });
      position += stepFrames;
    }

    const tailPosition = Math.max(0, nFrames - windowFrames);
    const tailStart = region.start + tailPosition * 0.01;
    windows.push({
      fbank: regionFbank,
      startFrame: tailPosition,
      frameCount: windowFrames,
      start: tailStart,
      end: tailStart + CAMPP_WINDOW_SECONDS,
    });
  }

  if (!windows.length) {
    return { embeddings: [], windowTimes: [], batchSize: batchSizeConfig, batchTuning: options.batchTuning || null };
  }

  const runWindows = Number.isFinite(options.tuneLimitWindows)
    ? windows.slice(0, Math.max(1, Math.min(windows.length, options.tuneLimitWindows)))
    : windows;

  const embeddings = [];
  const windowTimes = [];
  for (let batchStart = 0; batchStart < runWindows.length; batchStart += batchSizeConfig) {
    const batch = runWindows.slice(batchStart, batchStart + batchSizeConfig);
    const maxFrames = Math.max(...batch.map((item) => item.frameCount));
    const data = new Float32Array(batch.length * maxFrames * CAMPP_NUM_MEL_BINS);

    for (let b = 0; b < batch.length; b += 1) {
      const item = batch[b];
      for (let f = 0; f < item.frameCount; f += 1) {
        const row = item.fbank[item.startFrame + f];
        data.set(row, (b * maxFrames + f) * CAMPP_NUM_MEL_BINS);
      }
    }

    const outputs = await camppSession.run({
      feats: new window.ort.Tensor("float32", data, [batch.length, maxFrames, CAMPP_NUM_MEL_BINS]),
    });
    const tensor = outputs.embs || outputs[camppSession.outputNames[0]];
    const dim = tensor.dims[1];
    for (let b = 0; b < batch.length; b += 1) {
      const vector = tensor.data.slice(b * dim, (b + 1) * dim);
      embeddings.push(l2Normalize(vector));
      windowTimes.push({ start: batch[b].start, end: batch[b].end });
    }

    if (options.progress) {
      options.progress(Math.min(runWindows.length, batchStart + batch.length), runWindows.length);
    }
    throwIfWebGpuRuntimeSlower(options, started, Math.min(runWindows.length, batchStart + batch.length), runWindows.length);
  }

  return { embeddings, windowTimes, batchSize: batchSizeConfig, batchTuning: options.batchTuning || null };
}

async function computeCamppEmbeddingForAudio(audio) {
  await ensureCamppReady();
  if (audio.length < Math.floor(0.3 * VAD_SAMPLE_RATE)) return null;
  const fbank = computeCamppFbank(audio);
  if (fbank.length < 10) return null;

  const data = new Float32Array(fbank.length * CAMPP_NUM_MEL_BINS);
  for (let frame = 0; frame < fbank.length; frame += 1) {
    data.set(fbank[frame], frame * CAMPP_NUM_MEL_BINS);
  }
  const outputs = await camppSession.run({
    feats: new window.ort.Tensor("float32", data, [1, fbank.length, CAMPP_NUM_MEL_BINS]),
  });
  const tensor = outputs.embs || outputs[camppSession.outputNames[0]];
  return l2Normalize(tensor.data.slice(0, tensor.dims[1]));
}

function normalizeRegion(region) {
  if (Array.isArray(region)) {
    return { start: Number(region[0]) || 0, end: Number(region[1]) || 0 };
  }
  return { start: Number(region?.start) || 0, end: Number(region?.end) || 0 };
}

function regionsIntersect(startA, endA, startB, endB) {
  return Math.max(startA, startB) < Math.min(endA, endB);
}

function intersectsAnyRegion(start, end, regions) {
  return regions.some((region) => regionsIntersect(start, end, region.start, region.end));
}

function extractAudioRange(samples, startSeconds, endSeconds) {
  const start = Math.max(0, Math.floor(startSeconds * VAD_SAMPLE_RATE));
  const end = Math.min(samples.length, Math.ceil(endSeconds * VAD_SAMPLE_RATE));
  if (end <= start) return new Float32Array(0);
  return samples.slice(start, end);
}

function maxAbs(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

function scaleToMixturePeak(stream, mixturePeak) {
  const peak = maxAbs(stream);
  if (mixturePeak < 1e-6 || peak < 1e-9) {
    throw new Error("Overlap separation cannot scale a silent separated stream.");
  }
  const scale = (mixturePeak * 0.9) / peak;
  const output = new Float32Array(stream.length);
  for (let i = 0; i < stream.length; i += 1) {
    output[i] = stream[i] * scale;
  }
  return output;
}

function participantsInOverlapRegion(region, segments) {
  const participants = new Set();
  for (const segment of segments || []) {
    const start = Number(segment.start) || 0;
    const end = Number(segment.end) || start;
    const speaker = Number(segment.speaker);
    if (!Number.isInteger(speaker) || speaker < 0) continue;
    if (regionsIntersect(region.start, region.end, start, end)) {
      participants.add(speaker);
    }
  }
  return [...participants].sort((a, b) => a - b);
}

async function computeOverlapCentroids(samples, segments, overlapRegions, options = {}) {
  const bySpeaker = new Map();
  const cleanSegments = (segments || []).filter((segment) => {
    const start = Number(segment.start) || 0;
    const end = Number(segment.end) || start;
    return end - start >= OVERLAP_MIN_REF_SECONDS && !intersectsAnyRegion(start, end, overlapRegions);
  });

  for (let i = 0; i < cleanSegments.length; i += 1) {
    const segment = cleanSegments[i];
    const speaker = Number(segment.speaker);
    if (!Number.isInteger(speaker) || speaker < 0) continue;
    const audio = extractAudioRange(samples, segment.start, segment.end);
    const embedding = await computeCamppEmbeddingForAudio(audio);
    if (embedding) {
      if (!bySpeaker.has(speaker)) bySpeaker.set(speaker, []);
      bySpeaker.get(speaker).push(embedding);
    }
    if (options.progress) options.progress(i + 1, cleanSegments.length);
  }

  const centroids = new Map();
  for (const [speaker, embeddings] of bySpeaker) {
    const sum = new Float32Array(embeddings[0].length);
    for (const embedding of embeddings) {
      for (let i = 0; i < embedding.length; i += 1) {
        sum[i] += embedding[i];
      }
    }
    centroids.set(speaker, l2Normalize(sum));
  }
  return centroids;
}

async function separateOverlapRegion(samples, region, participants, centroids) {
  if (participants.length !== 2) {
    throw new Error(`Overlap region ${region.start.toFixed(2)}-${region.end.toFixed(2)}s has ${participants.length} participant(s), expected 2.`);
  }
  if (!centroids.has(participants[0]) || !centroids.has(participants[1])) {
    throw new Error(`Overlap region ${region.start.toFixed(2)}-${region.end.toFixed(2)}s is missing clean CAM++ speaker centroids.`);
  }
  if (region.end - region.start < OVERLAP_MIN_REGION_SECONDS) {
    throw new Error(`Overlap region ${region.start.toFixed(2)}-${region.end.toFixed(2)}s is shorter than ${OVERLAP_MIN_REGION_SECONDS}s.`);
  }

  await ensureOverlapReady();
  const regionAudio = extractAudioRange(samples, region.start, region.end);
  if (regionAudio.length < Math.floor(OVERLAP_MIN_REGION_SECONDS * VAD_SAMPLE_RATE)) {
    throw new Error(`Overlap region ${region.start.toFixed(2)}-${region.end.toFixed(2)}s has too few samples.`);
  }

  const outputs = await overlapSession.run({
    mixture: new window.ort.Tensor("float32", regionAudio, [1, regionAudio.length]),
  });
  const tensor = outputs.sources || outputs[overlapSession.outputNames[0]];
  if (!tensor || tensor.dims.length !== 3 || tensor.dims[0] !== 1 || tensor.dims[1] < 2) {
    throw new Error("Conv-TasNet output shape is not [1, 2, time].");
  }

  const outputSamples = Math.min(regionAudio.length, tensor.dims[2]);
  const mixturePeak = maxAbs(regionAudio);
  const stream0 = scaleToMixturePeak(tensor.data.slice(0, outputSamples), mixturePeak);
  const stream1 = scaleToMixturePeak(tensor.data.slice(tensor.dims[2], tensor.dims[2] + outputSamples), mixturePeak);

  const emb0 = await computeCamppEmbeddingForAudio(stream0);
  const emb1 = await computeCamppEmbeddingForAudio(stream1);
  if (!emb0 || !emb1) {
    throw new Error(`CAM++ could not embed separated streams for overlap ${region.start.toFixed(2)}-${region.end.toFixed(2)}s.`);
  }

  const first = participants[0];
  const second = participants[1];
  const scoreIdentity = cosine(emb0, centroids.get(first)) + cosine(emb1, centroids.get(second));
  const scoreSwap = cosine(emb0, centroids.get(second)) + cosine(emb1, centroids.get(first));
  const streams = new Map();
  if (scoreSwap > scoreIdentity) {
    streams.set(second, stream0);
    streams.set(first, stream1);
  } else {
    streams.set(first, stream0);
    streams.set(second, stream1);
  }
  return {
    streams,
    scoreIdentity,
    scoreSwap,
  };
}

async function decodeShortAsrText(samples, modelConfig, cpuThreads, options = {}) {
  if (samples.length < Math.floor(0.2 * VAD_SAMPLE_RATE)) return "";
  await ensureAsrReady(modelConfig, {
    cpuThreads,
    hotwordsText: options.hotwordsText,
    hotwordsScore: options.hotwordsScore,
  });

  if (modelConfig.type === "rover") {
    const childIds = modelConfig.modelIds || [];
    if (childIds.length !== 2) {
      throw new Error("ROVER requires exactly two ASR models for overlap decoding.");
    }
    const samplesA = new Float32Array(samples);
    const responseA = await callAsrWorker("decode", { modelId: childIds[0], samples: samplesA }, [samplesA.buffer]);
    const samplesB = new Float32Array(samples);
    const responseB = await callAsrWorker("decode", { modelId: childIds[1], samples: samplesB }, [samplesB.buffer]);
    const merged = roverMergeTexts(
      normalizeAsrText(responseA.result?.text || ""),
      normalizeAsrText(responseB.result?.text || "")
    );
    return merged.text;
  }

  const chunk = new Float32Array(samples);
  const response = await callAsrWorker("decode", { modelId: modelConfig.id, samples: chunk }, [chunk.buffer]);
  return normalizeAsrText(response.result?.text || "");
}

async function runOverlapSeparation(samples, diarization, options = {}) {
  const started = performance.now();
  const rawRegions = (diarization.overlapRegions || [])
    .map(normalizeRegion)
    .filter((region) => region.end > region.start);
  const candidateRegions = rawRegions.filter((region) => region.end - region.start >= OVERLAP_MIN_DECODE_SECONDS);
  const skipped = [];
  const outputSegments = [];

  if (!rawRegions.length) {
    renderOverlapSegments([], "[no overlap regions detected]");
    return {
      segments: [],
      elapsed: 0,
      backend: "convtasnet",
      detectedRegions: 0,
      candidateRegions: 0,
      processedRegions: 0,
      skipped,
    };
  }
  if (candidateRegions.length < rawRegions.length) {
    skipped.push(`${rawRegions.length - candidateRegions.length} short region(s) under ${OVERLAP_MIN_DECODE_SECONDS}s`);
  }
  if (!candidateRegions.length) {
    renderOverlapSegments([], `[${rawRegions.length} overlap region(s), none long enough for separation]`);
    return {
      segments: [],
      elapsed: (performance.now() - started) / 1000,
      backend: "convtasnet",
      detectedRegions: rawRegions.length,
      candidateRegions: 0,
      processedRegions: 0,
      skipped,
    };
  }

  log(`Overlap separation: ${rawRegions.length} detected, ${candidateRegions.length} >= ${OVERLAP_MIN_DECODE_SECONDS}s.`);
  if (options.progress) options.progress(0.05);
  const centroids = await computeOverlapCentroids(samples, diarization.segments || [], candidateRegions, {
    progress: (done, total) => {
      if (options.progress) options.progress(0.05 + (done / Math.max(1, total)) * 0.20);
    },
  });
  if (options.progress) options.progress(0.30);

  let processedRegions = 0;
  for (let i = 0; i < candidateRegions.length; i += 1) {
    const region = candidateRegions[i];
    const participants = participantsInOverlapRegion(region, diarization.segments || []);
    if (participants.length !== 2) {
      skipped.push(`${region.start.toFixed(2)}-${region.end.toFixed(2)}s: ${participants.length} participant(s)`);
      continue;
    }
    if (!participants.every((speaker) => centroids.has(speaker))) {
      skipped.push(`${region.start.toFixed(2)}-${region.end.toFixed(2)}s: missing clean centroid`);
      continue;
    }

    const separated = await separateOverlapRegion(samples, region, participants, centroids);
    processedRegions += 1;
    let streamIndex = 0;
    for (const [speaker, audio] of separated.streams) {
      const text = await decodeShortAsrText(
        audio,
        options.asrModel || getSelectedAsrModel(),
        options.cpuThreads || getRequestedThreads(),
        {
          hotwordsText: options.hotwordsText,
          hotwordsScore: options.hotwordsScore,
        }
      );
      if (text.trim()) {
        outputSegments.push({
          start: region.start,
          end: region.end,
          speaker,
          text,
          overlap: true,
        });
      }
      streamIndex += 1;
      if (options.progress) {
        const done = i + streamIndex / separated.streams.size;
        options.progress(0.30 + (done / Math.max(1, candidateRegions.length)) * 0.65);
      }
    }
    log(
      `Overlap ${i + 1}/${candidateRegions.length}: ${region.start.toFixed(2)}-${region.end.toFixed(2)}s, ` +
      `speakers ${participants.map((speaker) => speaker + 1).join("+")}, ` +
      `match ${separated.scoreIdentity.toFixed(3)}/${separated.scoreSwap.toFixed(3)}.`
    );
  }

  const elapsed = (performance.now() - started) / 1000;
  renderOverlapSegments(outputSegments, `[${rawRegions.length} overlap region(s), ${processedRegions} processed, ${skipped.length} skipped]`);
  if (skipped.length) {
    log(`Overlap skipped: ${skipped.join("; ")}.`);
  }
  log(`Overlap separation finished in ${elapsed.toFixed(2)}s: ${outputSegments.length} separated speaker line(s).`);
  if (options.progress) options.progress(1);
  return {
    segments: outputSegments,
    elapsed,
    backend: "convtasnet",
    detectedRegions: rawRegions.length,
    candidateRegions: candidateRegions.length,
    processedRegions,
    skipped,
  };
}

function pyannoteChunkStarts(totalSamples) {
  const starts = [];
  let start = 0;
  let hasLast = false;
  while (true) {
    if (hasLast) break;
    if ((start + DIAR_CHUNK_SAMPLES) / VAD_SAMPLE_RATE > totalSamples / VAD_SAMPLE_RATE) {
      hasLast = true;
    }
    starts.push(start);
    start += PYANNOTE_STEP_SAMPLES;
  }
  return starts.length ? starts : [0];
}

function roundHalfToEven(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function closestPyannoteFrame(time) {
  return roundHalfToEven((time - PYANNOTE_RF_START - 0.5 * PYANNOTE_RF_DURATION) / PYANNOTE_RF_STEP);
}

async function runPyannoteSegmentationLogits(samples, options = {}) {
  await ensureDiarizationReady();
  const started = performance.now();
  const starts = options.starts || pyannoteChunkStarts(samples.length);
  const chunks = starts.length;
  let logitsData = null;
  let numFrames = 0;
  let numClasses = 0;
  let writeOffset = 0;
  const batchSize = options.batchSize || getDiarBatchSizeForProvider(diarizationExecutionProvider);

  for (let batchStart = 0; batchStart < chunks; batchStart += batchSize) {
    const batchStarts = starts.slice(batchStart, batchStart + batchSize);
    // Buffer reuse: tái dùng diarBatchBuf (shared với getPyannoteSpeechRegions)
    const needed = batchStarts.length * DIAR_CHUNK_SAMPLES;
    diarBatchBuf = ensureBuf(diarBatchBuf, needed);
    const batch = diarBatchBuf.subarray(0, needed);
    batch.fill(0);
    for (let b = 0; b < batchStarts.length; b += 1) {
      const sourceStart = batchStarts[b];
      const sourceEnd = Math.min(samples.length, sourceStart + DIAR_CHUNK_SAMPLES);
      batch.set(samples.subarray(sourceStart, sourceEnd), b * DIAR_CHUNK_SAMPLES);
    }
    const outputs = await diarizationSession.run({
      input_values: new window.ort.Tensor("float32", batch, [batchStarts.length, 1, DIAR_CHUNK_SAMPLES]),
    });
    const logits = outputs.logits || outputs[diarizationSession.outputNames[0]];
    if (!logitsData) {
      numFrames = logits.dims[1];
      numClasses = logits.dims[2];
      logitsData = new Float32Array(chunks * numFrames * numClasses);
    }
    logitsData.set(logits.data, writeOffset);
    writeOffset += logits.data.length;
    if (options.progress) options.progress(Math.min(chunks, batchStart + batchStarts.length), chunks);
    throwIfWebGpuRuntimeSlower(options, started, Math.min(chunks, batchStart + batchStarts.length), chunks);
  }
  return {
    logitsData,
    starts,
    chunks,
    numFrames,
    numClasses,
    provider: diarizationExecutionProvider,
    batchSize,
    batchTuning: options.batchTuning || null,
    algorithm: options.algorithm || "pyannote_community",
  };
}

async function runPyannoteSegmentation(samples, options = {}) {
  return runPyannoteSegmentationLogits(samples, {
    ...options,
    starts: pyannoteChunkStarts(samples.length),
    algorithm: "pyannote_community",
  });
}

async function runPyannoteSegmentationWithOptionalWebGpuAutotune(samples, options = {}, runtime = "wasm", benchmarkContext = {}) {
  if (runtime === "calibrated-webgpu") {
    return runPyannoteSegmentationLogits(samples, {
      ...options,
      starts: pyannoteChunkStarts(samples.length),
      batchSize: calibratedBatchSizeForStage("Pyannote Community-1 segmentation", getDiarBatchSizeForProvider("webgpu")),
      algorithm: "pyannote_community",
    });
  }
  if (runtime !== "webgpu") return runPyannoteSegmentation(samples, options);
  const starts = pyannoteChunkStarts(samples.length);
  const candidates = webgpuBatchCandidates("pyannote_segmentation", getDiarBatchSizeForProvider("webgpu"));
  const tuneStarts = starts.slice(0, autotuneSampleCount(candidates, starts.length));
  const tuning = await autotuneWebGpuBatch("Pyannote Community-1 segmentation", candidates, (batchSize) => (
    runPyannoteSegmentationLogits(samples, {
      starts: tuneStarts,
      batchSize,
      algorithm: "pyannote_community_tune",
    })
  ), webGpuAutotuneGuardOptions(benchmarkContext));
  return runPyannoteSegmentationLogits(samples, {
    ...options,
    starts,
    batchSize: tuning.selectedBatchSize,
    batchTuning: tuning,
    abortStageName: "Pyannote Community-1 segmentation",
    ...webGpuRuntimeAbortOptions(benchmarkContext, tuning),
    algorithm: "pyannote_community",
  });
}

async function getPyannoteSpeechRegionsWithOptionalWebGpuAutotune(samples, options = {}, runtime = "wasm", benchmarkContext = {}) {
  if (runtime === "calibrated-webgpu") {
    return getPyannoteSpeechRegions(samples, {
      ...options,
      batchSize: calibratedBatchSizeForStage("CAM++ speech regions (pyannote segmentation)", getDiarBatchSizeForProvider("webgpu")),
    });
  }
  if (runtime !== "webgpu") return getPyannoteSpeechRegions(samples, options);
  const starts = diarizationStarts(samples.length);
  const candidates = webgpuBatchCandidates("campp_speech_regions", getDiarBatchSizeForProvider("webgpu"));
  const tuneStarts = starts.slice(0, autotuneSampleCount(candidates, starts.length));
  const tuning = await autotuneWebGpuBatch("CAM++ speech regions", candidates, (batchSize) => (
    runPyannoteSegmentationLogits(samples, {
      starts: tuneStarts,
      batchSize,
      algorithm: "campp_speech_regions_tune",
    })
  ), webGpuAutotuneGuardOptions(benchmarkContext));
  return getPyannoteSpeechRegions(samples, {
    ...options,
    batchSize: tuning.selectedBatchSize,
    batchTuning: tuning,
    abortStageName: "CAM++ speech regions",
    ...webGpuRuntimeAbortOptions(benchmarkContext, tuning),
  });
}

function powerSetBinarize(logitsData, chunks, numFrames, numClasses) {
  const binarized = new Uint8Array(chunks * numFrames * PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
  for (let c = 0; c < chunks; c += 1) {
    for (let f = 0; f < numFrames; f += 1) {
      const base = (c * numFrames + f) * numClasses;
      let bestClass = 0;
      let bestValue = -Infinity;
      for (let cls = 0; cls < numClasses; cls += 1) {
        const value = logitsData[base + cls];
        if (value > bestValue) {
          bestValue = value;
          bestClass = cls;
        }
      }
      const active = DIAR_POWERSET[bestClass] || DIAR_POWERSET[0];
      const out = (c * numFrames + f) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
      binarized[out] = active[0];
      binarized[out + 1] = active[1];
      binarized[out + 2] = active[2];
    }
  }
  return binarized;
}

function pyannoteBinarizedAt(binarized, numFrames, chunk, frame, speaker) {
  return binarized[(chunk * numFrames + frame) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + speaker];
}

function extractPyannoteOverlapRegions(binarized, starts, numFrames, duration, minDuration = 0.3) {
  const frameSeconds = DIAR_CHUNK_SECONDS / numFrames;
  const nOut = Math.floor(duration / frameSeconds) + 1;
  const overlapCount = new Float32Array(nOut);
  const totalCount = new Float32Array(nOut);
  for (let c = 0; c < starts.length; c += 1) {
    const chunkStart = starts[c] / VAD_SAMPLE_RATE;
    for (let f = 0; f < numFrames; f += 1) {
      const outFrame = Math.floor((chunkStart + f * frameSeconds) / frameSeconds);
      if (outFrame < 0 || outFrame >= nOut) continue;
      let active = 0;
      for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
        active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      }
      if (active >= 2) overlapCount[outFrame] += 1;
      totalCount[outFrame] += 1;
    }
  }
  const mask = Array.from(totalCount, (count, index) => count > 0 && overlapCount[index] / count > 0.5);
  return frameMaskToRegions(mask, frameSeconds, duration, minDuration, 0.1);
}

function aggregatePyannoteCount(binarized, chunks, numFrames) {
  const outFrames = closestPyannoteFrame(
    PYANNOTE_STEP_SECONDS * (chunks - 1) + DIAR_CHUNK_SECONDS + 0.5 * PYANNOTE_RF_DURATION
  ) + 1;
  const output = new Float32Array(outFrames);
  const weight = new Float32Array(outFrames);
  for (let c = 0; c < chunks; c += 1) {
    const startFrame = closestPyannoteFrame(c * PYANNOTE_STEP_SECONDS + 0.5 * PYANNOTE_RF_DURATION);
    for (let f = 0; f < numFrames; f += 1) {
      const out = startFrame + f;
      if (out < 0 || out >= outFrames) continue;
      let count = 0;
      for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
        count += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      }
      output[out] += count;
      weight[out] += 1;
    }
  }
  for (let i = 0; i < outFrames; i += 1) {
    output[i] = weight[i] > 0 ? roundHalfToEven(output[i] / weight[i]) : 0;
  }
  return output;
}

function buildCleanBinarized(binarized, chunks, numFrames) {
  const clean = new Uint8Array(binarized.length);
  for (let c = 0; c < chunks; c += 1) {
    for (let f = 0; f < numFrames; f += 1) {
      let active = 0;
      for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
        active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      }
      if (active >= 2) continue;
      const base = (c * numFrames + f) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
      clean[base] = binarized[base];
      clean[base + 1] = binarized[base + 1];
      clean[base + 2] = binarized[base + 2];
    }
  }
  return clean;
}

function pyannoteMaskSum(mask, numFrames, chunk, speaker) {
  let sum = 0;
  for (let f = 0; f < numFrames; f += 1) {
    sum += pyannoteBinarizedAt(mask, numFrames, chunk, f, speaker);
  }
  return sum;
}

function projectPyannoteStats(stats, assets) {
  const output = new Float32Array(256);
  const weight = assets.weight;
  const bias = assets.bias;
  for (let out = 0; out < 256; out += 1) {
    let sum = bias[out];
    const base = out * 5120;
    for (let i = 0; i < 5120; i += 1) {
      sum += stats[i] * weight[base + i];
    }
    output[out] = sum;
  }
  return output;
}

function maskedStatsPool(frameFeatures, featureDim, featureFrames, weights) {
  const stats = new Float32Array(featureDim * 2);
  let v1 = 1e-8;
  let v2 = 0;
  for (let t = 0; t < featureFrames; t += 1) {
    const w = weights[t];
    v1 += w;
    v2 += w * w;
  }
  for (let d = 0; d < featureDim; d += 1) {
    let mean = 0;
    const base = d * featureFrames;
    for (let t = 0; t < featureFrames; t += 1) {
      mean += frameFeatures[base + t] * weights[t];
    }
    mean /= v1;
    stats[d] = mean;
    let variance = 0;
    for (let t = 0; t < featureFrames; t += 1) {
      const diff = frameFeatures[base + t] - mean;
      variance += diff * diff * weights[t];
    }
    stats[featureDim + d] = Math.sqrt(Math.max(0, variance / (v1 - v2 / v1 + 1e-8)));
  }
  return stats;
}

function makePaddedChunk(samples, start) {
  const chunk = new Float32Array(DIAR_CHUNK_SAMPLES);
  const end = Math.min(samples.length, start + DIAR_CHUNK_SAMPLES);
  if (end > start) chunk.set(samples.subarray(start, end));
  return chunk;
}

async function extractPyannoteEmbeddings(samples, binarized, cleanBinarized, numFrames, starts, options = {}) {
  await ensurePyannoteCommunityReady();
  const started = performance.now();
  const chunks = starts.length;
  const embeddings = new Float32Array(chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK * 256);
  embeddings.fill(NaN);
  const minSegFrames = Math.ceil(numFrames * PYANNOTE_EMB_MIN_NUM_SAMPLES / DIAR_CHUNK_SAMPLES);
  let featureIndex = null;
  const batchSizeConfig = options.batchSize || WESPEAKER_BATCH_SIZE;

  for (let batchStart = 0; batchStart < chunks; batchStart += batchSizeConfig) {
    const batchEnd = Math.min(chunks, batchStart + batchSizeConfig);
    const batchSize = batchEnd - batchStart;
    const fbankRows = [];
    let maxFrames = 0;
    for (let c = batchStart; c < batchEnd; c += 1) {
      const fbank = computeWespeakerFbank(makePaddedChunk(samples, starts[c]));
      fbankRows.push(fbank);
      maxFrames = Math.max(maxFrames, fbank.length);
    }
    if (!maxFrames) throw new Error("Pyannote Community-1 embedding fbank produced no frames.");
    // Buffer reuse: tái dùng wespeakerInputBuf thay vì alloc mới mỗi batch
    const needed = batchSize * maxFrames * WESPEAKER_NUM_MEL_BINS;
    wespeakerInputBuf = ensureBuf(wespeakerInputBuf, needed);
    const input = wespeakerInputBuf.subarray(0, needed);
    input.fill(0);
    for (let b = 0; b < batchSize; b += 1) {
      const rows = fbankRows[b];
      for (let f = 0; f < rows.length; f += 1) {
        input.set(rows[f], (b * maxFrames + f) * WESPEAKER_NUM_MEL_BINS);
      }
    }

    const outputs = await pyannoteEmbeddingSession.run({
      fbank_features: new window.ort.Tensor("float32", input, [batchSize, maxFrames, WESPEAKER_NUM_MEL_BINS]),
    });
    const tensor = outputs[pyannoteEmbeddingSession.outputNames[0]];
    const featureDim = tensor.dims[1];
    const featureFrames = tensor.dims[2];
    if (featureDim * 2 !== 5120) throw new Error(`Pyannote Community-1 encoder feature dim ${featureDim} is unsupported.`);
    if (!featureIndex || featureIndex.length !== featureFrames) {
      featureIndex = new Int32Array(featureFrames);
      for (let i = 0; i < featureFrames; i += 1) {
        featureIndex[i] = Math.max(0, Math.min(numFrames - 1, Math.floor(i * numFrames / featureFrames)));
      }
    }

    for (let b = 0; b < batchSize; b += 1) {
      const c = batchStart + b;
      const frameFeatures = tensor.data.subarray(b * featureDim * featureFrames, (b + 1) * featureDim * featureFrames);
      for (let speaker = 0; speaker < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; speaker += 1) {
        const cleanSum = pyannoteMaskSum(cleanBinarized, numFrames, c, speaker);
        const fullSum = pyannoteMaskSum(binarized, numFrames, c, speaker);
        if (!fullSum) continue;
        const sourceMask = cleanSum > minSegFrames ? cleanBinarized : binarized;
        const weights = new Float32Array(featureFrames);
        for (let t = 0; t < featureFrames; t += 1) {
          weights[t] = pyannoteBinarizedAt(sourceMask, numFrames, c, featureIndex[t], speaker);
        }
        const projected = projectPyannoteStats(
          maskedStatsPool(frameFeatures, featureDim, featureFrames, weights),
          pyannoteCommunityAssets
        );
        embeddings.set(projected, (c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + speaker) * 256);
      }
    }
    if (options.progress) options.progress(batchEnd, chunks);
    throwIfWebGpuRuntimeSlower(options, started, batchEnd, chunks);
  }
  embeddings.batchSize = batchSizeConfig;
  embeddings.batchTuning = options.batchTuning || null;
  embeddings.provider = pyannoteEmbeddingExecutionProvider;
  return embeddings;
}

function l2NormalizeVector(values) {
  let norm = 0;
  for (let i = 0; i < values.length; i += 1) norm += values[i] * values[i];
  norm = Math.sqrt(norm) + 1e-10;
  const output = new Float64Array(values.length);
  for (let i = 0; i < values.length; i += 1) output[i] = values[i] / norm;
  return output;
}

function extractEmbeddingVector(embeddings, index) {
  const output = new Float64Array(256);
  const base = index * 256;
  for (let i = 0; i < 256; i += 1) output[i] = embeddings[base + i];
  return output;
}

function isFiniteEmbedding(embeddings, index) {
  return Number.isFinite(embeddings[index * 256]);
}

function unionFindRoot(parent, value) {
  let root = value;
  while (parent[root] !== root) root = parent[root];
  while (parent[value] !== value) {
    const next = parent[value];
    parent[value] = root;
    value = next;
  }
  return root;
}

function unionFindMerge(parent, a, b) {
  const ra = unionFindRoot(parent, a);
  const rb = unionFindRoot(parent, b);
  if (ra === rb) return;
  if (ra < rb) parent[rb] = ra;
  else parent[ra] = rb;
}

function hierarchicalCentroidLabels(points, threshold) {
  const n = points.length;
  if (n <= 1) return new Int32Array(n);
  let clusters = points.map((point, index) => ({
    id: index,
    centroid: Float64Array.from(point),
    size: 1,
    members: [index],
  }));
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i += 1) parent[i] = i;
  let nextClusterId = n;

  while (clusters.length > 1) {
    let bestI = -1;
    let bestJ = -1;
    let bestDist = Infinity;
    for (let i = 0; i < clusters.length - 1; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const dist = Math.sqrt(squaredEuclidean(clusters[i].centroid, clusters[j].centroid));
        if (
          dist < bestDist ||
          (dist === bestDist && (
            clusters[i].id < clusters[bestI]?.id ||
            (clusters[i].id === clusters[bestI]?.id && clusters[j].id < clusters[bestJ]?.id)
          ))
        ) {
          bestDist = dist;
          bestI = i;
          bestJ = j;
        }
      }
    }
    if (bestI < 0) break;
    const a = clusters[bestI];
    const b = clusters[bestJ];
    if (bestDist <= threshold) {
      for (const left of a.members) {
        for (const right of b.members) unionFindMerge(parent, left, right);
      }
    }
    const size = a.size + b.size;
    const centroid = new Float64Array(a.centroid.length);
    for (let d = 0; d < centroid.length; d += 1) {
      centroid[d] = (a.centroid[d] * a.size + b.centroid[d] * b.size) / size;
    }
    const merged = { id: nextClusterId, centroid, size, members: a.members.concat(b.members) };
    nextClusterId += 1;
    clusters = clusters.filter((_, index) => index !== bestI && index !== bestJ);
    clusters.push(merged);
  }

  const labels = new Int32Array(n);
  for (let i = 0; i < n; i += 1) labels[i] = unionFindRoot(parent, i);
  return relabelClusters(labels);
}

function xvecTransformBatch(vectors, plda) {
  const out = new Array(vectors.length);
  const sqrtIn = Math.sqrt(256);
  const sqrtOut = Math.sqrt(128);
  for (let row = 0; row < vectors.length; row += 1) {
    const centered = new Float64Array(256);
    for (let i = 0; i < 256; i += 1) centered[i] = vectors[row][i] - plda.mean1[i];
    const normed = l2NormalizeVector(centered);
    const transformed = new Float64Array(128);
    for (let j = 0; j < 128; j += 1) {
      let sum = 0;
      for (let i = 0; i < 256; i += 1) sum += normed[i] * sqrtIn * plda.lda[i * 128 + j];
      transformed[j] = sum - plda.mean2[j];
    }
    const normalized = l2NormalizeVector(transformed);
    for (let j = 0; j < 128; j += 1) normalized[j] *= sqrtOut;
    out[row] = normalized;
  }
  return out;
}

function pldaTransformBatch(vectors, plda) {
  return vectors.map((vector) => {
    const output = new Float64Array(128);
    for (let k = 0; k < 128; k += 1) {
      let sum = 0;
      const base = k * 128;
      for (let i = 0; i < 128; i += 1) sum += (vector[i] - plda.mu[i]) * plda.pldaTr[base + i];
      output[k] = sum;
    }
    return output;
  });
}

function vbxCluster(features, pldaPsi, initialLabels, fa = PYANNOTE_FA, fb = PYANNOTE_FB, maxIterations = 20) {
  const frames = features.length;
  const dim = features[0].length;
  const clusters = Math.max(...initialLabels) + 1;
  const gamma = Array.from({ length: frames }, () => new Float64Array(clusters));
  for (let t = 0; t < frames; t += 1) {
    const logits = new Float64Array(clusters);
    logits[initialLabels[t]] = 7;
    let maxLogit = -Infinity;
    for (let k = 0; k < clusters; k += 1) if (logits[k] > maxLogit) maxLogit = logits[k];
    let sum = 0;
    for (let k = 0; k < clusters; k += 1) {
      gamma[t][k] = Math.exp(logits[k] - maxLogit);
      sum += gamma[t][k];
    }
    for (let k = 0; k < clusters; k += 1) gamma[t][k] /= sum;
  }

  let pi = new Float64Array(clusters);
  pi.fill(1 / clusters);
  const g = new Float64Array(frames);
  const rho = Array.from({ length: frames }, () => new Float64Array(dim));
  const sqrtPsi = new Float64Array(dim);
  const logTwoPi = Math.log(2 * Math.PI);
  for (let d = 0; d < dim; d += 1) sqrtPsi[d] = Math.sqrt(Math.max(0, pldaPsi[d]));
  for (let t = 0; t < frames; t += 1) {
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const value = features[t][d];
      norm += value * value;
      rho[t][d] = value * sqrtPsi[d];
    }
    g[t] = -0.5 * (norm + dim * logTwoPi);
  }

  let previousElbo = -Infinity;
  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const gammaSum = new Float64Array(clusters);
    for (let t = 0; t < frames; t += 1) {
      for (let k = 0; k < clusters; k += 1) gammaSum[k] += gamma[t][k];
    }
    const invL = Array.from({ length: clusters }, () => new Float64Array(dim));
    const alpha = Array.from({ length: clusters }, () => new Float64Array(dim));
    for (let k = 0; k < clusters; k += 1) {
      for (let d = 0; d < dim; d += 1) invL[k][d] = 1 / (1 + (fa / fb) * gammaSum[k] * pldaPsi[d]);
      for (let t = 0; t < frames; t += 1) {
        const weight = gamma[t][k];
        if (weight <= 0) continue;
        for (let d = 0; d < dim; d += 1) alpha[k][d] += weight * rho[t][d];
      }
      for (let d = 0; d < dim; d += 1) alpha[k][d] *= (fa / fb) * invL[k][d];
    }

    const logPx = new Float64Array(frames);
    let elbo = 0;
    for (let t = 0; t < frames; t += 1) {
      const logProb = new Float64Array(clusters);
      let maxLog = -Infinity;
      for (let k = 0; k < clusters; k += 1) {
        let dot = 0;
        let penalty = 0;
        for (let d = 0; d < dim; d += 1) {
          dot += rho[t][d] * alpha[k][d];
          penalty += (invL[k][d] + alpha[k][d] * alpha[k][d]) * pldaPsi[d];
        }
        logProb[k] = fa * (dot - 0.5 * penalty + g[t]) + Math.log(pi[k] + 1e-8);
        if (logProb[k] > maxLog) maxLog = logProb[k];
      }
      let sumExp = 0;
      for (let k = 0; k < clusters; k += 1) sumExp += Math.exp(logProb[k] - maxLog);
      logPx[t] = maxLog + Math.log(sumExp);
      elbo += logPx[t];
      for (let k = 0; k < clusters; k += 1) gamma[t][k] = Math.exp(logProb[k] - logPx[t]);
    }

    pi = new Float64Array(clusters);
    for (let t = 0; t < frames; t += 1) {
      for (let k = 0; k < clusters; k += 1) pi[k] += gamma[t][k];
    }
    let piSum = 0;
    for (let k = 0; k < clusters; k += 1) piSum += pi[k];
    for (let k = 0; k < clusters; k += 1) pi[k] /= Math.max(1e-12, piSum);

    for (let k = 0; k < clusters; k += 1) {
      for (let d = 0; d < dim; d += 1) {
        elbo += fb * 0.5 * (Math.log(invL[k][d]) - invL[k][d] - alpha[k][d] * alpha[k][d] + 1);
      }
    }
    if (iteration > 0 && elbo - previousElbo < 1e-4) break;
    previousElbo = elbo;
  }
  return { gamma, pi };
}

function cosineDistanceVector(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return 1 - dot / ((Math.sqrt(na) * Math.sqrt(nb)) + 1e-10);
}

function constrainedArgmax(cost, rows, cols) {
  const output = new Int32Array(rows);
  output.fill(-2);
  if (rows !== PYANNOTE_MAX_SPEAKERS_PER_CHUNK) {
    throw new Error("Pyannote constrained assignment expects three local speaker slots.");
  }
  if (cols <= 0) return output;

  let bestScore = -Infinity;
  let best = Int32Array.from(output);

  function update(score, assignments) {
    if (score <= bestScore) return;
    bestScore = score;
    best = Int32Array.from(assignments);
  }

  if (cols === 1) {
    const assignments = new Int32Array(rows);
    assignments.fill(-2);
    for (let row = 0; row < rows; row += 1) {
      assignments.fill(-2);
      assignments[row] = 0;
      update(cost[row * cols], assignments);
    }
    return best;
  }

  if (cols === 2) {
    const assignments = new Int32Array(rows);
    assignments.fill(-2);
    for (let r0 = 0; r0 < rows; r0 += 1) {
      for (let r1 = 0; r1 < rows; r1 += 1) {
        if (r1 === r0) continue;
        assignments.fill(-2);
        assignments[r0] = 0;
        assignments[r1] = 1;
        update(cost[r0 * cols] + cost[r1 * cols + 1], assignments);
      }
    }
    return best;
  }

  const topForRow = (row) => {
    const top = [];
    for (let col = 0; col < cols; col += 1) {
      const value = cost[row * cols + col];
      let inserted = false;
      for (let i = 0; i < top.length; i += 1) {
        if (value > top[i].value) {
          top.splice(i, 0, { col, value });
          inserted = true;
          break;
        }
      }
      if (!inserted) top.push({ col, value });
      if (top.length > rows) top.length = rows;
    }
    return top;
  };
  const row2Top = topForRow(2);
  const assignments = new Int32Array(rows);
  assignments.fill(-2);
  for (let c0 = 0; c0 < cols; c0 += 1) {
    for (let c1 = 0; c1 < cols; c1 += 1) {
      if (c1 === c0) continue;
      const c2 = row2Top.find((entry) => entry.col !== c0 && entry.col !== c1);
      if (!c2) continue;
      assignments[0] = c0;
      assignments[1] = c1;
      assignments[2] = c2.col;
      update(cost[c0] + cost[cols + c1] + cost[2 * cols + c2.col], assignments);
    }
  }
  return best;
}

function buildPyannoteClusteringCentroids(embeddings, binarized, chunks, numFrames, options = {}) {
  const trainIndices = [];
  for (let c = 0; c < chunks; c += 1) {
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let singleActive = 0;
      for (let f = 0; f < numFrames; f += 1) {
        let count = 0;
        for (let k = 0; k < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; k += 1) {
          count += pyannoteBinarizedAt(binarized, numFrames, c, f, k);
        }
        if (count === 1) singleActive += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      }
      const embIndex = c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s;
      if (singleActive >= 0.2 * numFrames && isFiniteEmbedding(embeddings, embIndex)) {
        trainIndices.push(embIndex);
      }
    }
  }
  if (trainIndices.length < 2) {
    return {
      trainIndices,
      trainEmbeddings: [],
      trainNormed: [],
      centroids: [],
    };
  }

  const trainEmbeddings = trainIndices.map((index) => extractEmbeddingVector(embeddings, index));
  const trainNormed = trainEmbeddings.map(l2NormalizeVector);
  const ahcLabels = hierarchicalCentroidLabels(trainNormed, PYANNOTE_DEFAULT_THRESHOLD);
  const plda = pyannoteCommunityAssets.plda;
  const embPlda = pldaTransformBatch(xvecTransformBatch(trainEmbeddings, plda), plda);
  const { gamma, pi } = vbxCluster(embPlda, plda.pldaPsi, ahcLabels);

  let activeClusters = [];
  for (let k = 0; k < pi.length; k += 1) {
    if (pi[k] > 1e-7) activeClusters.push(k);
  }
  if (!activeClusters.length) activeClusters = [0];

  let centroids = activeClusters.map((cluster) => {
    const centroid = new Float64Array(256);
    let weightSum = 0;
    for (let i = 0; i < trainEmbeddings.length; i += 1) {
      const weight = gamma[i][cluster];
      weightSum += weight;
      for (let d = 0; d < 256; d += 1) centroid[d] += trainEmbeddings[i][d] * weight;
    }
    for (let d = 0; d < 256; d += 1) centroid[d] /= Math.max(1e-8, weightSum);
    return centroid;
  });

  const requestedSpeakers = options.numSpeakers || 0;
  if (requestedSpeakers > 0 && centroids.length > requestedSpeakers) {
    const kmLabels = kmeansEuclidean(trainNormed, requestedSpeakers, 20);
    centroids = Array.from({ length: requestedSpeakers }, () => new Float64Array(256));
    const counts = new Int32Array(requestedSpeakers);
    for (let i = 0; i < trainEmbeddings.length; i += 1) {
      const label = kmLabels[i];
      counts[label] += 1;
      for (let d = 0; d < 256; d += 1) centroids[label][d] += trainEmbeddings[i][d];
    }
    for (let k = 0; k < requestedSpeakers; k += 1) {
      if (!counts[k]) continue;
      for (let d = 0; d < 256; d += 1) centroids[k][d] /= counts[k];
    }
  }
  return {
    trainIndices,
    trainEmbeddings,
    trainNormed,
    centroids,
    gamma,
    pi,
  };
}

function assignPyannoteClustersJs(embeddings, binarized, chunks, numFrames, centroids) {
  const hard = new Int32Array(chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
  hard.fill(-2);
  if (!centroids.length) return hard;
  for (let c = 0; c < chunks; c += 1) {
    const cost = new Float64Array(PYANNOTE_MAX_SPEAKERS_PER_CHUNK * centroids.length);
    let minCost = Infinity;
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      const embIndex = c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s;
      const embedding = extractEmbeddingVector(embeddings, embIndex);
      for (let k = 0; k < centroids.length; k += 1) {
        const value = 2 - cosineDistanceVector(embedding, centroids[k]);
        cost[s * centroids.length + k] = value;
        if (value < minCost) minCost = value;
      }
    }
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let active = 0;
      for (let f = 0; f < numFrames; f += 1) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      if (active) continue;
      for (let k = 0; k < centroids.length; k += 1) cost[s * centroids.length + k] = minCost - 1;
    }
    const assigned = constrainedArgmax(cost, PYANNOTE_MAX_SPEAKERS_PER_CHUNK, centroids.length);
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let active = 0;
      for (let f = 0; f < numFrames; f += 1) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      hard[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] = active ? assigned[s] : -2;
    }
  }
  return hard;
}

function flattenNormalizedPyannoteRowsForWebGpu(embeddings, rows) {
  const dim = 256;
  const data = new Float32Array(rows * dim);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const value = Number(embeddings[base + d]);
      if (Number.isFinite(value)) norm += value * value;
    }
    norm = Math.sqrt(Math.max(norm, 1e-12));
    for (let d = 0; d < dim; d += 1) {
      const value = Number(embeddings[base + d]);
      data[base + d] = Number.isFinite(value) ? value / norm : 0;
    }
  }
  return data;
}

function flattenNormalizedCentroidsForWebGpu(centroids) {
  const dim = 256;
  const data = new Float32Array(centroids.length * dim);
  for (let row = 0; row < centroids.length; row += 1) {
    const vector = centroids[row];
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const value = Number(vector?.[d]);
      if (Number.isFinite(value)) norm += value * value;
    }
    norm = Math.sqrt(Math.max(norm, 1e-12));
    for (let d = 0; d < dim; d += 1) {
      const value = Number(vector?.[d]);
      data[row * dim + d] = Number.isFinite(value) ? value / norm : 0;
    }
  }
  return data;
}

async function computePyannoteAssignmentCostWebGpu(embeddings, centroids, rows) {
  if (!navigator.gpu) throw new Error("WebGPU is not available for Pyannote clustering assignment.");
  const clusters = centroids.length;
  if (!rows || !clusters) return new Float32Array(0);
  const dim = 256;
  const embData = flattenNormalizedPyannoteRowsForWebGpu(embeddings, rows);
  const centroidData = flattenNormalizedCentroidsForWebGpu(centroids);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU adapter is not available for Pyannote clustering assignment.");
  const device = await adapter.requestDevice();
  const total = rows * clusters;
  const paramData = new Uint32Array([rows, clusters, dim, total]);
  const embBuffer = device.createBuffer({
    size: embData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const centroidBuffer = device.createBuffer({
    size: centroidData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBytes = total * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    size: paramData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(embBuffer, 0, embData);
  device.queue.writeBuffer(centroidBuffer, 0, centroidData);
  device.queue.writeBuffer(paramsBuffer, 0, paramData);
  const shader = device.createShaderModule({
    label: "pyannote-clustering-assignment",
    code: `
struct Params {
  rows: u32,
  clusters: u32,
  dim: u32,
  total: u32,
};
@group(0) @binding(0) var<storage, read> emb: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.total) {
    return;
  }
  let row = index / params.clusters;
  let cluster = index - row * params.clusters;
  var dot = 0.0;
  var d = 0u;
  loop {
    if (d >= params.dim) {
      break;
    }
    dot = dot + emb[row * params.dim + d] * centroids[cluster * params.dim + d];
    d = d + 1u;
  }
  out[index] = 1.0 + dot;
}
`,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: embBuffer } },
      { binding: 1, resource: { buffer: centroidBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(total / 256));
  pass.end();
  const readBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  embBuffer.destroy();
  centroidBuffer.destroy();
  outputBuffer.destroy();
  paramsBuffer.destroy();
  readBuffer.destroy();
  if (typeof device.destroy === "function") device.destroy();
  return output;
}

async function assignPyannoteClustersWebGpu(embeddings, binarized, chunks, numFrames, centroids) {
  const hard = new Int32Array(chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
  hard.fill(-2);
  if (!centroids.length) return hard;
  const rows = chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
  const clusters = centroids.length;
  const flatCost = await computePyannoteAssignmentCostWebGpu(embeddings, centroids, rows);
  for (let c = 0; c < chunks; c += 1) {
    const cost = new Float64Array(PYANNOTE_MAX_SPEAKERS_PER_CHUNK * clusters);
    let minCost = Infinity;
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let active = 0;
      for (let f = 0; f < numFrames; f += 1) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      for (let k = 0; k < clusters; k += 1) {
        const value = flatCost[(c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s) * clusters + k];
        cost[s * clusters + k] = value;
        if (active && value < minCost) minCost = value;
      }
    }
    if (!Number.isFinite(minCost)) minCost = 0;
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let active = 0;
      for (let f = 0; f < numFrames; f += 1) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      if (active) continue;
      for (let k = 0; k < clusters; k += 1) cost[s * clusters + k] = minCost - 1;
    }
    const assigned = constrainedArgmax(cost, PYANNOTE_MAX_SPEAKERS_PER_CHUNK, clusters);
    for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
      let active = 0;
      for (let f = 0; f < numFrames; f += 1) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
      hard[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] = active ? assigned[s] : -2;
    }
  }
  return hard;
}

async function clusterPyannoteEmbeddingsWebGpuCandidate(embeddings, binarized, chunks, numFrames, options = {}) {
  const model = buildPyannoteClusteringCentroids(embeddings, binarized, chunks, numFrames, options);
  const hard = await assignPyannoteClustersWebGpu(embeddings, binarized, chunks, numFrames, model.centroids);
  return canonicalizePyannoteClusters(hard, binarized, chunks, numFrames);
}

function clusterPyannoteEmbeddings(embeddings, binarized, chunks, numFrames, options = {}) {
  const model = buildPyannoteClusteringCentroids(embeddings, binarized, chunks, numFrames, options);
  const hard = assignPyannoteClustersJs(embeddings, binarized, chunks, numFrames, model.centroids);
  return canonicalizePyannoteClusters(hard, binarized, chunks, numFrames);
}

function canonicalizePyannoteClusters(hardClusters, binarized, chunks, numFrames) {
  const keys = [];
  const seen = new Set();
  for (let i = 0; i < hardClusters.length; i += 1) {
    const cluster = hardClusters[i];
    if (cluster >= 0) seen.add(cluster);
  }
  for (const cluster of Array.from(seen).sort((a, b) => a - b)) {
    let bestChunk = Number.MAX_SAFE_INTEGER;
    let bestFrame = Number.MAX_SAFE_INTEGER;
    let bestSpeaker = Number.MAX_SAFE_INTEGER;
    for (let c = 0; c < chunks; c += 1) {
      for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
        if (hardClusters[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] !== cluster) continue;
        for (let f = 0; f < numFrames; f += 1) {
          if (!pyannoteBinarizedAt(binarized, numFrames, c, f, s)) continue;
          if (
            c < bestChunk ||
            (c === bestChunk && f < bestFrame) ||
            (c === bestChunk && f === bestFrame && s < bestSpeaker)
          ) {
            bestChunk = c;
            bestFrame = f;
            bestSpeaker = s;
          }
          break;
        }
      }
    }
    keys.push({ cluster, key: [bestChunk, bestFrame, bestSpeaker] });
  }
  keys.sort((a, b) => {
    for (let i = 0; i < 3; i += 1) {
      if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
    }
    return a.cluster - b.cluster;
  });
  const remap = new Map();
  keys.forEach((item, index) => remap.set(item.cluster, index));
  const output = new Int32Array(hardClusters.length);
  output.fill(-2);
  for (let i = 0; i < hardClusters.length; i += 1) {
    const cluster = hardClusters[i];
    output[i] = cluster >= 0 ? remap.get(cluster) : -2;
  }
  return output;
}

function reconstructPyannoteSegments(binarized, hardClusters, countData, chunks, numFrames, duration) {
  const numClusters = Math.max(...hardClusters) + 1;
  if (numClusters <= 0) return [];
  const outFrames = countData.length;
  const activations = new Float32Array(outFrames * numClusters);

  for (let c = 0; c < chunks; c += 1) {
    const startFrame = closestPyannoteFrame(c * PYANNOTE_STEP_SECONDS + 0.5 * PYANNOTE_RF_DURATION);
    for (let f = 0; f < numFrames; f += 1) {
      const out = startFrame + f;
      if (out < 0 || out >= outFrames) continue;
      for (let k = 0; k < numClusters; k += 1) {
        let value = 0;
        for (let s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s += 1) {
          if (hardClusters[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] === k) {
            value = Math.max(value, pyannoteBinarizedAt(binarized, numFrames, c, f, s));
          }
        }
        activations[out * numClusters + k] += value;
      }
    }
  }

  const binary = new Uint8Array(outFrames * numClusters);
  for (let t = 0; t < outFrames; t += 1) {
    if (Math.min(1, Math.max(0, Math.round(countData[t]))) < 1) continue;
    let bestSpeaker = 0;
    let bestValue = -Infinity;
    for (let k = 0; k < numClusters; k += 1) {
      const value = activations[t * numClusters + k];
      if (value > bestValue) {
        bestValue = value;
        bestSpeaker = k;
      }
    }
    binary[t * numClusters + bestSpeaker] = 1;
  }

  const rawSegments = [];
  for (let speaker = 0; speaker < numClusters; speaker += 1) {
    let active = false;
    let start = PYANNOTE_RF_START + 0.5 * PYANNOTE_RF_DURATION;
    let lastTime = start;
    for (let t = 0; t < outFrames; t += 1) {
      const time = PYANNOTE_RF_START + t * PYANNOTE_RF_STEP + 0.5 * PYANNOTE_RF_DURATION;
      lastTime = time;
      const value = binary[t * numClusters + speaker] > 0;
      if (active && !value) {
        rawSegments.push({ start, end: time, speaker });
        active = false;
      } else if (!active && value) {
        start = time;
        active = true;
      }
    }
    if (active) rawSegments.push({ start, end: lastTime, speaker });
  }
  rawSegments.sort((a, b) => a.start - b.start);

  const speakerMap = new Map();
  let next = 0;
  const segments = [];
  for (const segment of rawSegments) {
    if (segment.start >= duration) continue;
    if (!speakerMap.has(segment.speaker)) {
      speakerMap.set(segment.speaker, next);
      next += 1;
    }
    segments.push({
      start: Math.max(0, Number(segment.start.toFixed(4))),
      end: Number(segment.end.toFixed(4)),
      speaker: speakerMap.get(segment.speaker),
    });
  }
  return segments.filter((segment) => segment.end > segment.start);
}

function pyannoteSegmentHash(segments) {
  return hashString((segments || []).map((s) => `${s.speaker}:${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|"));
}

function computePyannoteClusteringSummary(segments, hardClusters = null) {
  return {
    turns: segments.length,
    speakers: new Set(segments.map((segment) => segment.speaker)).size,
    segmentHash: pyannoteSegmentHash(segments),
    hardClusterHash: hardClusters ? hashFloatValues(hardClusters, 1, Number.POSITIVE_INFINITY) : null,
    outputHash: pyannoteSegmentHash(segments),
  };
}

function pyannoteSegmentsFromHardClusters(hardClusters, binarized, countData, chunks, numFrames, duration) {
  const rawSegments = reconstructPyannoteSegments(binarized, hardClusters, countData, chunks, numFrames, duration);
  const segments = rawSegments.map((segment) => cloneDiarSegment(segment));
  return { hardClusters, rawSegments, segments };
}

function computePyannoteClusteringResult(embeddings, binarized, countData, chunks, numFrames, duration, options = {}) {
  const hardClusters = clusterPyannoteEmbeddings(embeddings, binarized, chunks, numFrames, options);
  return pyannoteSegmentsFromHardClusters(hardClusters, binarized, countData, chunks, numFrames, duration);
}

async function computePyannoteClusteringResultWebGpu(embeddings, binarized, countData, chunks, numFrames, duration, options = {}) {
  const hardClusters = await clusterPyannoteEmbeddingsWebGpuCandidate(embeddings, binarized, chunks, numFrames, options);
  return pyannoteSegmentsFromHardClusters(hardClusters, binarized, countData, chunks, numFrames, duration);
}

function updatePyannoteEmbeddingStageAcceptance(options, embeddingStage, downstreamCheck, mismatchDetails, accepted) {
  if (!embeddingStage?.comparison) return;
  embeddingStage.comparison = {
    ...embeddingStage.comparison,
    mismatchDetails,
    numericToleranceAccepted: downstreamCheck.numericToleranceAccepted,
    downstreamSegmentHashEqual: downstreamCheck.downstreamSegmentHashEqual,
    downstreamSegmentDiffWithinTolerance: downstreamCheck.downstreamSegmentDiffWithinTolerance,
    downstreamCheck,
    webgpuAcceptedByTolerance: accepted,
    acceptanceMode: accepted ? "numeric-tolerance+downstream-segment-tolerance" : null,
    webgpuAccepted: accepted || embeddingStage.comparison.webgpuAccepted,
    rejectionReason: accepted ? null : embeddingStage.comparison.rejectionReason,
  };
  if (accepted) {
    embeddingStage.selectedRuntime = "webgpu";
    embeddingStage.selectedProvider = "webgpu";
    options.benchmarkSelectedProviders = options.benchmarkSelectedProviders || {};
    options.benchmarkSelectedProviders["Pyannote Community-1 embedding encoder"] = "webgpu";
  }
}

async function maybeAcceptPyannoteEmbeddingWebGpu(options, referenceResult, binarized, countData, chunks, numFrames, duration) {
  const embeddingBench = options.benchmarkStageResults?.["Pyannote Community-1 embedding encoder"];
  const embeddingStage = embeddingBench?.stage ||
    options.benchmarkStages?.find((stage) => stage.name === "Pyannote Community-1 embedding encoder");
  const webgpuEntry = embeddingBench?.webgpu;
  const wasmEntry = embeddingBench?.wasm;
  if (!embeddingStage?.comparison || !webgpuEntry?.result || !wasmEntry?.result) {
    return { accepted: false, result: referenceResult };
  }
  const downstreamStarted = performance.now();
  try {
    const webgpuCluster = computePyannoteClusteringResult(
      webgpuEntry.result,
      binarized,
      countData,
      chunks,
      numFrames,
      duration,
      options
    );
    const wasmSummary = computePyannoteClusteringSummary(referenceResult.segments, referenceResult.hardClusters);
    const webgpuSummary = computePyannoteClusteringSummary(webgpuCluster.segments, webgpuCluster.hardClusters);
    const segmentDiff = compareRegionArrays(referenceResult.segments, webgpuCluster.segments);
    const downstreamSegmentHashEqual = wasmSummary.segmentHash === webgpuSummary.segmentHash;
    const downstreamSegmentDiffWithinTolerance = downstreamSegmentHashEqual || segmentDiffWithinTolerance(segmentDiff, {
      maxCountDelta: 0,
      maxTimeDiff: 0.05,
    });
    const mismatchDetails = embeddingStage.comparison.mismatchDetails ||
      benchmarkMismatchDetails(
        "Pyannote Community-1 embedding encoder",
        wasmEntry.result,
        webgpuEntry.result,
        wasmEntry.attempt?.summary,
        webgpuEntry.attempt?.summary
      );
    const numericToleranceAccepted = embeddingDiffWithinTolerance(mismatchDetails, {
      maxAbsDiff: 1e-4,
      meanAbsDiff: 1e-5,
      rmsDiff: 1e-5,
      minCosine: 0.9999,
      maxOneNonFinite: 0,
    });
    const speedup = embeddingStage.comparison.speedupWebgpuOverWasm;
    const accepted = Boolean(
      webgpuEntry.attempt?.provider === "webgpu" &&
      numericToleranceAccepted &&
      downstreamSegmentDiffWithinTolerance &&
      speedup > 1
    );
    const downstreamCheck = {
      elapsedSeconds: benchmarkSeconds(downstreamStarted),
      wasmSegmentHash: wasmSummary.segmentHash,
      webgpuSegmentHash: webgpuSummary.segmentHash,
      downstreamSegmentHashEqual,
      downstreamSegmentDiffWithinTolerance,
      segmentDiff,
      numericToleranceAccepted,
      webgpuTurns: webgpuCluster.segments.length,
      webgpuSpeakers: webgpuSummary.speakers,
    };
    updatePyannoteEmbeddingStageAcceptance(options, embeddingStage, downstreamCheck, mismatchDetails, accepted);
    if (accepted) {
      log("[Benchmark] Stage Pyannote Community-1 embedding encoder: WebGPU accepted by numeric tolerance and downstream segment tolerance.");
      return { accepted: true, result: webgpuCluster };
    }
    log(
      `[Benchmark] Stage Pyannote Community-1 embedding encoder downstream check: ` +
      `numericTolerance=${numericToleranceAccepted}, segmentHashEqual=${downstreamSegmentHashEqual}, ` +
      `segmentTolerance=${downstreamSegmentDiffWithinTolerance}.`
    );
  } catch (error) {
    const downstreamCheck = {
      elapsedSeconds: benchmarkSeconds(downstreamStarted),
      error: { message: error.message || String(error) },
    };
    updatePyannoteEmbeddingStageAcceptance(options, embeddingStage, downstreamCheck, embeddingStage.comparison.mismatchDetails, false);
    log(`[Benchmark] Stage Pyannote Community-1 embedding encoder downstream check failed: ${error.message || String(error)}`);
  }
  return { accepted: false, result: referenceResult };
}

async function runPyannoteClusteringBenchmark(options, embeddings, binarized, countData, chunks, numFrames, duration) {
  const jsStarted = performance.now();
  const jsResult = computePyannoteClusteringResult(embeddings, binarized, countData, chunks, numFrames, duration, options);
  const jsAttempt = {
    runtime: "js",
    provider: "js",
    elapsedSeconds: benchmarkSeconds(jsStarted),
    summary: computePyannoteClusteringSummary(jsResult.segments, jsResult.hardClusters),
  };

  let selectedResult = jsResult;
  let selectedRuntime = "js";
  let selectedProvider = "js";
  let webgpuAttempt = null;
  let comparison = {
    speedupWebgpuOverWasm: null,
    outputHashEqual: null,
    webgpuAccepted: false,
    rejectionReason: null,
  };

  try {
    const webgpuStarted = performance.now();
    const webgpuResult = await computePyannoteClusteringResultWebGpu(
      embeddings,
      binarized,
      countData,
      chunks,
      numFrames,
      duration,
      options
    );
    webgpuAttempt = {
      runtime: "webgpu",
      provider: "webgpu-js-hybrid",
      elapsedSeconds: benchmarkSeconds(webgpuStarted),
      summary: computePyannoteClusteringSummary(webgpuResult.segments, webgpuResult.hardClusters),
    };
    const outputHashEqual = jsAttempt.summary.outputHash === webgpuAttempt.summary.outputHash;
    const segmentDiff = compareRegionArrays(jsResult.segments, webgpuResult.segments);
    const segmentDiffOk = outputHashEqual || segmentDiffWithinTolerance(segmentDiff, {
      maxCountDelta: 0,
      maxTimeDiff: 0.05,
    });
    const speedup = Number((jsAttempt.elapsedSeconds / Math.max(webgpuAttempt.elapsedSeconds, 1e-9)).toFixed(3));
    const webgpuAccepted = Boolean(segmentDiffOk && speedup > 1);
    comparison = {
      speedupWebgpuOverWasm: speedup,
      outputHashEqual,
      webgpuAccepted,
      downstreamSegmentDiffWithinTolerance: segmentDiffOk,
      segmentDiff: outputHashEqual ? null : segmentDiff,
      rejectionReason: webgpuAccepted
        ? null
        : (!segmentDiffOk ? "webgpu segment output differs beyond tolerance" : "webgpu was not faster than js clustering"),
    };
    if (webgpuAccepted) {
      selectedResult = webgpuResult;
      selectedRuntime = "webgpu";
      selectedProvider = "webgpu-js-hybrid";
    }
    log(
      `[Benchmark] Stage Pyannote Community-1 clustering/VBx: JS ${jsAttempt.elapsedSeconds.toFixed(2)}s, ` +
      `WEBGPU ${webgpuAttempt.elapsedSeconds.toFixed(2)}s, segmentOk=${segmentDiffOk}.`
    );
  } catch (error) {
    webgpuAttempt = {
      runtime: "webgpu",
      error: { message: error.message || String(error) },
    };
    comparison.rejectionReason = error.message || String(error);
    log(`[Benchmark] Stage Pyannote Community-1 clustering/VBx: WEBGPU failed: ${error.message || String(error)}`);
  }

  addBenchmarkStage(options, {
    name: "Pyannote Community-1 clustering/VBx",
    capability: "wasm-webgpu",
    attempts: [jsAttempt, webgpuAttempt].filter(Boolean),
    selectedRuntime,
    selectedProvider,
    comparison,
  });
  return {
    ...selectedResult,
    elapsedSeconds: selectedRuntime === "webgpu" ? webgpuAttempt?.elapsedSeconds : jsAttempt.elapsedSeconds,
  };
}

async function runPyannoteCommunityDiarization(samples, options = {}) {
  const duration = samples.length / VAD_SAMPLE_RATE;
  if (duration < 0.5) {
    renderDiarization([]);
    return {
      segments: [],
      speakers: 0,
      elapsed: 0,
      chunks: 0,
      embeddings: 0,
      overlapRegions: [],
      backend: "pyannote_community1_vbx",
      executionProvider: { segmentation: "wasm", embedding: pyannoteEmbeddingExecutionProvider },
    };
  }
  const started = performance.now();
  const segmentationRunner = (runtime, benchmarkContext) => runPyannoteSegmentationWithOptionalWebGpuAutotune(samples, {
    progress: (done, total) => {
      if (options.progress) options.progress(0.02 + (done / Math.max(1, total)) * 0.20);
    },
  }, runtime, benchmarkContext);
  const resumedSegmentation = decodePyannoteSegmentationCheckpoint(options.resumeCheckpoints?.pyannote_segmentation, samples.length);
  let segmentation = resumedSegmentation?.segmentation || null;
  let binarized = resumedSegmentation?.binarized || null;
  if (segmentation && binarized) {
    log(`[resume_after_kill] Resumed Pyannote segmentation: ${segmentation.chunks} chunk(s).`);
    if (options.progress) options.progress(0.22);
  } else {
    segmentation = shouldBenchmarkWebGpuStage(options, "Pyannote Community-1 segmentation")
      ? await runBenchmarkDualProviderStage(
          options,
          "Pyannote Community-1 segmentation",
          segmentationRunner,
          unloadDiarizationSegmentationSessionOnly,
          summarizePyannoteSegmentationForBenchmark
        )
      : await segmentationRunner(calibratedProviderForStage("Pyannote Community-1 segmentation", "wasm") === "webgpu"
          ? "calibrated-webgpu"
          : "wasm");
    binarized = powerSetBinarize(segmentation.logitsData, segmentation.chunks, segmentation.numFrames, segmentation.numClasses);
    const segmentationCheckpoint = encodePyannoteSegmentationCheckpoint(segmentation, binarized, samples.length);
    if (segmentationCheckpoint) {
      await writeResumeJsonCheckpoint(
        options.resumeContext,
        "pyannote_segmentation",
        segmentationCheckpoint,
        {
          chunks: segmentation.chunks || 0,
          activeRows: segmentationCheckpoint.activeRows?.length || 0,
        }
      );
    }
  }
  const overlapRegions = extractPyannoteOverlapRegions(binarized, segmentation.starts, segmentation.numFrames, duration);
  const countData = aggregatePyannoteCount(binarized, segmentation.chunks, segmentation.numFrames);
  const cleanBinarized = buildCleanBinarized(binarized, segmentation.chunks, segmentation.numFrames);
  const embeddingRunner = async (runtime, benchmarkContext = {}) => {
    const runFull = (batchSize, batchTuning = null, runtimeGuard = {}) => extractPyannoteEmbeddings(
      samples,
      binarized,
      cleanBinarized,
      segmentation.numFrames,
      segmentation.starts,
      {
        batchSize,
        batchTuning,
        abortStageName: "Pyannote Community-1 embedding encoder",
        ...runtimeGuard,
        progress: (done, total) => {
          if (options.progress) options.progress(0.24 + (done / Math.max(1, total)) * 0.48);
        },
      }
    );
    if (runtime === "calibrated-webgpu") {
      return runFull(calibratedBatchSizeForStage("Pyannote Community-1 embedding encoder", WESPEAKER_BATCH_SIZE));
    }
    if (runtime !== "webgpu") return runFull(WESPEAKER_BATCH_SIZE);
    if (benchmarkContext.useCalibratedWebGpuBatch) {
      const batchSize = calibratedBatchSizeForStage(
        "Pyannote Community-1 embedding encoder",
        defaultWebGpuBatchSize("pyannote_embedding", WESPEAKER_BATCH_SIZE)
      );
      return runFull(batchSize, {
        selectedBatchSize: batchSize,
        source: "calibration_profile",
        totalSeconds: 0,
      });
    }
    const candidates = webgpuBatchCandidates("pyannote_embedding", WESPEAKER_BATCH_SIZE);
    const tuneChunks = autotuneSampleCount(candidates, segmentation.starts.length);
    const tuneStarts = segmentation.starts.slice(0, tuneChunks);
    const tuneBinarized = binarized.subarray(0, tuneChunks * segmentation.numFrames * PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
    const tuneClean = cleanBinarized.subarray(0, tuneChunks * segmentation.numFrames * PYANNOTE_MAX_SPEAKERS_PER_CHUNK);
    const tuning = await autotuneWebGpuBatch("Pyannote Community-1 embedding encoder", candidates, (batchSize) => (
      extractPyannoteEmbeddings(
        samples,
        tuneBinarized,
        tuneClean,
        segmentation.numFrames,
        tuneStarts,
        { batchSize }
      )
    ), webGpuAutotuneGuardOptions(benchmarkContext));
    return runFull(tuning.selectedBatchSize, tuning, webGpuRuntimeAbortOptions(benchmarkContext, tuning));
  };
  const embeddings = shouldBenchmarkWebGpuStage(options, "Pyannote Community-1 embedding encoder")
    ? await runBenchmarkDualProviderStage(
        options,
        "Pyannote Community-1 embedding encoder",
        embeddingRunner,
        unloadPyannoteEmbeddingSessionOnly,
        summarizePyannoteEmbeddingForBenchmark
      )
    : await embeddingRunner(calibratedProviderForStage("Pyannote Community-1 embedding encoder", "wasm") === "webgpu"
        ? "calibrated-webgpu"
        : "wasm");
  if (options.progress) options.progress(0.76);
  let clusteringResult = null;
  if (Array.isArray(options.benchmarkStages)) {
    clusteringResult = computePyannoteClusteringResult(
      embeddings,
      binarized,
      countData,
      segmentation.chunks,
      segmentation.numFrames,
      duration,
      options
    );
    const embeddingAcceptance = await maybeAcceptPyannoteEmbeddingWebGpu(
      options,
      clusteringResult,
      binarized,
      countData,
      segmentation.chunks,
      segmentation.numFrames,
      duration
    );
    if (embeddingAcceptance.accepted) clusteringResult = embeddingAcceptance.result;
  } else {
    clusteringResult = computePyannoteClusteringResult(
      embeddings,
      binarized,
      countData,
      segmentation.chunks,
      segmentation.numFrames,
      duration,
      options
    );
  }
  const rawSegments = clusteringResult.segments;
  const segments = desktopPostProcessDiarizationSegments(rawSegments, options.asrWords || null);
  const speakers = new Set(segments.map((segment) => segment.speaker)).size;
  const elapsed = (performance.now() - started) / 1000;
  renderDiarization(segments);
  log(`Pyannote Community-1 VBx diarization finished in ${elapsed.toFixed(2)}s: ${segments.length} turn(s), ${speakers} speaker(s), ${segmentation.chunks} chunk(s).`);
  if (overlapRegions.length) log(`Pyannote Community-1 overlap regions: ${overlapRegions.length}.`);

  if (options.progress) options.progress(1);
  const embeddingProvider = embeddings.benchmarkSelectedProvider ||
    options.benchmarkSelectedProviders?.["Pyannote Community-1 embedding encoder"] ||
    pyannoteEmbeddingExecutionProvider;
  const segmentationProvider = segmentation.benchmarkSelectedProvider ||
    options.benchmarkSelectedProviders?.["Pyannote Community-1 segmentation"] ||
    diarizationExecutionProvider;
  return {
    segments,
    rawSegments,
    speakers,
    elapsed,
    chunks: segmentation.chunks,
    embeddings: segmentation.chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK,
    overlapRegions,
    backend: "pyannote_community1_vbx",
    executionProvider: {
      segmentation: segmentationProvider,
      embedding: embeddingProvider,
    },
  };
}

function clusterCamppEmbeddings(embeddings, duration, options = {}) {
  const count = embeddings.length;
  if (count <= 2) return new Int32Array(count);

  const requestedSpeakers = options.numSpeakers || 0;
  if (requestedSpeakers > 0) {
    return kmeansCosine(embeddings, requestedSpeakers, 12);
  }

  if (duration >= 1200) {
    const longLabels = senkoUmapHdbscanCluster(embeddings);
    filterMinorClusters(longLabels, embeddings, 10);
    mergeCloseClusters(longLabels, embeddings, CAMPP_MERGE_COS);
    return relabelClusters(longLabels);
  }
  if (count > CAMPP_SPECTRAL_MAX_WINDOWS) {
    throw new Error(`CAM++ spectral clustering window limit exceeded: ${count}/${CAMPP_SPECTRAL_MAX_WINDOWS}.`);
  }

  const spectralLabels = senkoSpectralCluster(embeddings, {
    minNumSpeakers: 1,
    maxNumSpeakers: Math.min(15, count),
    pval: 0.012,
    minPnum: 6,
  });
  filterMinorClusters(spectralLabels, embeddings, CAMPP_MIN_CLUSTER_SIZE);
  mergeCloseClusters(spectralLabels, embeddings, CAMPP_MERGE_COS);
  return relabelClusters(spectralLabels);
}

async function clusterCamppEmbeddingsWebGpuCandidate(embeddings, duration, options = {}) {
  const count = embeddings.length;
  if (count <= 2) return new Int32Array(count);
  if (options.numSpeakers > 0) {
    throw new Error("CAM++ WebGPU clustering branch currently benchmarks automatic spectral clustering only; fixed-speaker kmeans remains JS.");
  }
  if (duration >= 1200) {
    throw new Error("CAM++ WebGPU clustering branch currently benchmarks spectral clustering only; long-form UMAP/HDBSCAN remains JS.");
  }
  if (count > CAMPP_SPECTRAL_MAX_WINDOWS) {
    throw new Error(`CAM++ spectral clustering window limit exceeded: ${count}/${CAMPP_SPECTRAL_MAX_WINDOWS}.`);
  }

  const spectralLabels = await senkoSpectralClusterWebGpuAffinity(embeddings, {
    minNumSpeakers: 1,
    maxNumSpeakers: Math.min(15, count),
    pval: 0.012,
    minPnum: 6,
  });
  filterMinorClusters(spectralLabels, embeddings, CAMPP_MIN_CLUSTER_SIZE);
  mergeCloseClusters(spectralLabels, embeddings, CAMPP_MERGE_COS);
  return relabelClusters(spectralLabels);
}

function senkoUmapHdbscanCluster(embeddings) {
  return senkoUmapHdbscanClusterWithDistance(embeddings);
}

function senkoUmapHdbscanClusterWithDistance(embeddings, options = {}) {
  const count = embeddings.length;
  if (count < 10) return new Int32Array(count);
  const bundle = window.LongFormClustering;
  if (!bundle?.UMAP || !bundle?.ClusternovaHDBSCAN || !bundle?.clusternovaEuclidean || !bundle?.cosine) {
    throw new Error("Long-form UMAP/HDBSCAN clustering runtime is not loaded.");
  }

  const rng = createMt19937(0);
  const nComponents = Math.max(2, Math.min(60, count - 2));
  const data = options.data || embeddings.map((vector) => Array.from(vector));
  const umap = new bundle.UMAP({
    nNeighbors: 40,
    minDist: 0,
    nComponents,
    distanceFn: options.distanceFn || bundle.cosine,
    random: () => rng.random(),
  });
  const projected = umap.fit(data);
  const points = projected.map((vector, index) => ({ id: String(index), vector }));
  // Clusternova exposes one density parameter, while Python hdbscan separates
  // min_samples and min_cluster_size. Scale it lightly to avoid tiny long-form
  // side clusters that desktop HDBSCAN treats as part of the main speaker.
  const minPoints = Math.min(60, Math.max(20, Math.ceil(count * 0.05)));
  const hdbscan = new bundle.ClusternovaHDBSCAN(points, minPoints, bundle.clusternovaEuclidean);
  const { clusters } = hdbscan.run();
  const labels = new Int32Array(count);
  labels.fill(-1);
  clusters.forEach((cluster, label) => {
    for (const item of cluster) {
      labels[Number(item.id)] = label;
    }
  });
  return labels;
}

async function senkoUmapHdbscanClusterWebGpuDistance(embeddings) {
  const count = embeddings.length;
  if (count < 10) return new Int32Array(count);
  const bundle = window.LongFormClustering;
  if (!bundle?.UMAP || !bundle?.ClusternovaHDBSCAN || !bundle?.clusternovaEuclidean || !bundle?.cosine) {
    throw new Error("Long-form UMAP/HDBSCAN clustering runtime is not loaded.");
  }
  const similarity = await buildCosineSimilarityMatrixWebGpu(embeddings);
  const data = embeddings.map((vector) => Array.from(vector));
  const indexByVector = new WeakMap();
  data.forEach((vector, index) => indexByVector.set(vector, index));
  const distanceFn = (a, b) => {
    const ia = indexByVector.get(a);
    const ib = indexByVector.get(b);
    if (ia !== undefined && ib !== undefined) {
      return Math.max(0, 1 - similarity[ia * count + ib]);
    }
    return bundle.cosine(a, b);
  };
  return senkoUmapHdbscanClusterWithDistance(embeddings, { data, distanceFn });
}

function senkoSpectralCluster(embeddings, options) {
  const n = embeddings.length;
  if (n <= 1) return new Int32Array(n);

  const minNumSpeakers = options.minNumSpeakers || 1;
  const maxNumSpeakers = Math.max(minNumSpeakers, Math.min(options.maxNumSpeakers || 10, n));
  const pval = options.pval ?? 0.02;
  const minPnum = options.minPnum ?? 6;
  const { affinity, degree } = buildSenkoAffinity(embeddings, pval, minPnum);
  return senkoSpectralClusterFromAffinity(embeddings, affinity, degree, minNumSpeakers, maxNumSpeakers);
}

function senkoSpectralClusterFromAffinity(embeddings, affinity, degree, minNumSpeakers, maxNumSpeakers) {
  const n = embeddings.length;
  if (n <= 1) return new Int32Array(n);
  const eigenCount = Math.min(maxNumSpeakers + 1, n);
  const eigen = n <= CAMPP_JACOBI_MAX_WINDOWS
    ? jacobiSmallestLaplacianEigen(affinity, degree, n)
    : lanczosSmallestLaplacianEigen(affinity, degree, n, eigenCount);
  const vectorStride = eigen.stride || n;

  const spanStart = minNumSpeakers - 1;
  const spanEnd = Math.min(maxNumSpeakers + 1, eigen.values.length);
  let bestGap = -Infinity;
  let numSpeakers = minNumSpeakers;
  for (let i = spanStart; i < spanEnd - 1; i += 1) {
    const gap = eigen.values[i + 1] - eigen.values[i];
    if (gap > bestGap) {
      bestGap = gap;
      numSpeakers = i - spanStart + minNumSpeakers;
    }
  }
  numSpeakers = Math.max(1, Math.min(numSpeakers, n));

  const rows = new Array(n);
  for (let i = 0; i < n; i += 1) {
    rows[i] = new Float32Array(numSpeakers);
    for (let k = 0; k < numSpeakers; k += 1) {
      rows[i][k] = eigen.vectors[i * vectorStride + k];
    }
  }
  return kmeansEuclidean(rows, numSpeakers, 40);
}

async function senkoSpectralClusterWebGpuAffinity(embeddings, options) {
  const n = embeddings.length;
  if (n <= 1) return new Int32Array(n);
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available for CAM++ clustering affinity.");
  }

  const minNumSpeakers = options.minNumSpeakers || 1;
  const maxNumSpeakers = Math.max(minNumSpeakers, Math.min(options.maxNumSpeakers || 10, n));
  const pval = options.pval ?? 0.02;
  const minPnum = options.minPnum ?? 6;
  const { affinity, degree } = await buildSenkoAffinityWebGpu(embeddings, pval, minPnum);
  return senkoSpectralClusterFromAffinity(embeddings, affinity, degree, minNumSpeakers, maxNumSpeakers);
}

function buildSenkoAffinity(embeddings, pval, minPnum) {
  const n = embeddings.length;
  const affinity = new Array(n);

  for (let i = 0; i < n; i += 1) {
    affinity[i] = new Float64Array(n);
    for (let j = 0; j < n; j += 1) {
      affinity[i][j] = cosine(embeddings[i], embeddings[j]);
    }
  }

  let pruneCount = Math.floor((1 - pval) * n);
  pruneCount = Math.min(pruneCount, n - minPnum);
  pruneCount = Math.max(pruneCount, 0);
  if (pruneCount > 0) {
    for (let i = 0; i < n; i += 1) {
      const order = Array.from({ length: n }, (_, index) => index)
        .sort((a, b) => affinity[i][a] - affinity[i][b]);
      for (let k = 0; k < pruneCount; k += 1) {
        affinity[i][order[k]] = 0;
      }
    }
  }

  const degree = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    affinity[i][i] = 0;
  }
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const value = 0.5 * (affinity[i][j] + affinity[j][i]);
      affinity[i][j] = value;
      affinity[j][i] = value;
      degree[i] += Math.abs(value);
      degree[j] += Math.abs(value);
    }
  }
  return { affinity, degree };
}

function normalizeEmbeddingMatrixForWebGpu(embeddings) {
  const n = embeddings.length;
  const dim = embeddings[0]?.length || 0;
  if (!n || !dim) throw new Error("CAM++ clustering has no embedding data.");
  const data = new Float32Array(n * dim);
  for (let i = 0; i < n; i += 1) {
    const vector = embeddings[i];
    if (!vector || vector.length !== dim) {
      throw new Error("CAM++ clustering embeddings have inconsistent dimensions.");
    }
    let norm = 0;
    for (let d = 0; d < dim; d += 1) norm += vector[d] * vector[d];
    norm = Math.sqrt(Math.max(norm, 1e-12));
    for (let d = 0; d < dim; d += 1) data[i * dim + d] = vector[d] / norm;
  }
  return { data, n, dim };
}

function pruneAndSymmetrizeSenkoAffinity(flatAffinity, n, pval, minPnum) {
  const affinity = new Array(n);
  for (let i = 0; i < n; i += 1) {
    affinity[i] = new Float64Array(n);
    for (let j = 0; j < n; j += 1) {
      affinity[i][j] = flatAffinity[i * n + j];
    }
  }

  let pruneCount = Math.floor((1 - pval) * n);
  pruneCount = Math.min(pruneCount, n - minPnum);
  pruneCount = Math.max(pruneCount, 0);
  if (pruneCount > 0) {
    for (let i = 0; i < n; i += 1) {
      const order = Array.from({ length: n }, (_, index) => index)
        .sort((a, b) => affinity[i][a] - affinity[i][b]);
      for (let k = 0; k < pruneCount; k += 1) {
        affinity[i][order[k]] = 0;
      }
    }
  }

  const degree = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    affinity[i][i] = 0;
  }
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const value = 0.5 * (affinity[i][j] + affinity[j][i]);
      affinity[i][j] = value;
      affinity[j][i] = value;
      degree[i] += Math.abs(value);
      degree[j] += Math.abs(value);
    }
  }
  return { affinity, degree };
}

async function buildSenkoAffinityWebGpu(embeddings, pval, minPnum) {
  const flatAffinity = await buildCosineSimilarityMatrixWebGpu(embeddings);
  return pruneAndSymmetrizeSenkoAffinity(flatAffinity, embeddings.length, pval, minPnum);
}

async function buildCosineSimilarityMatrixWebGpu(embeddings) {
  const { data, n, dim } = normalizeEmbeddingMatrixForWebGpu(embeddings);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU adapter is not available for CAM++ clustering affinity.");
  const device = await adapter.requestDevice();
  const paramData = new Uint32Array([n, dim, n * n, 0]);
  const inputBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: n * n * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const outputBytes = n * n * Float32Array.BYTES_PER_ELEMENT;
  const paramsBuffer = device.createBuffer({
    size: paramData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, data);
  device.queue.writeBuffer(paramsBuffer, 0, paramData);

  const shader = device.createShaderModule({
    label: "campp-clustering-affinity",
    code: `
struct Params {
  n: u32,
  dim: u32,
  total: u32,
  pad: u32,
};
@group(0) @binding(0) var<storage, read> emb: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= params.n || j >= params.n) {
    return;
  }
  var sum = 0.0;
  var d = 0u;
  loop {
    if (d >= params.dim) {
      break;
    }
    sum = sum + emb[i * params.dim + d] * emb[j * params.dim + d];
    d = d + 1u;
  }
  out[i * params.n + j] = sum;
}
`,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8));
  pass.end();

  const readBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const flatAffinity = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  inputBuffer.destroy();
  outputBuffer.destroy();
  paramsBuffer.destroy();
  readBuffer.destroy();
  if (typeof device.destroy === "function") device.destroy();
  return flatAffinity;
}

function jacobiSmallestLaplacianEigen(affinity, degree, n) {
  const laplacian = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    laplacian[i * n + i] = degree[i];
    for (let j = 0; j < n; j += 1) {
      if (i !== j) laplacian[i * n + j] = -affinity[i][j];
    }
  }
  const eigen = jacobiEigenSymmetric(laplacian, n, 24, 1e-9);
  eigen.stride = n;
  return eigen;
}

function laplacianMatVec(affinity, degree, vector, output) {
  const n = vector.length;
  for (let i = 0; i < n; i += 1) {
    const row = affinity[i];
    let sum = degree[i] * vector[i];
    for (let j = 0; j < n; j += 1) {
      sum -= row[j] * vector[j];
    }
    output[i] = sum;
  }
}

function vectorDot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function vectorNorm(a) {
  return Math.sqrt(vectorDot(a, a));
}

function normalizeVectorInPlace(a) {
  const norm = vectorNorm(a) || 1;
  for (let i = 0; i < a.length; i += 1) {
    a[i] /= norm;
  }
  return norm;
}

function seededUnitVector(length) {
  let seed = 0x9e3779b9;
  const vector = new Float64Array(length);
  for (let i = 0; i < length; i += 1) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    vector[i] = seed / 0xffffffff - 0.5;
  }
  normalizeVectorInPlace(vector);
  return vector;
}

function orthogonalizeInPlace(vector, basis) {
  for (const q of basis) {
    const projection = vectorDot(vector, q);
    if (Math.abs(projection) < 1e-14) continue;
    for (let i = 0; i < vector.length; i += 1) {
      vector[i] -= projection * q[i];
    }
  }
}

function lanczosSmallestLaplacianEigen(affinity, degree, n, targetCount) {
  let maxDegree = 0;
  for (let i = 0; i < degree.length; i += 1) {
    if (degree[i] > maxDegree) maxDegree = degree[i];
  }
  const shift = Math.max(1e-6, 2 * maxDegree + 1e-6);
  const maxIterations = Math.min(
    n,
    Math.max(CAMPP_LANCZOS_ITERATIONS, targetCount * 8 + 32)
  );
  const basis = [];
  const alpha = [];
  const beta = [];
  let q = seededUnitVector(n);
  let previousQ = null;
  const lq = new Float64Array(n);
  let z = new Float64Array(n);

  for (let iter = 0; iter < maxIterations; iter += 1) {
    basis.push(q);
    laplacianMatVec(affinity, degree, q, lq);
    for (let i = 0; i < n; i += 1) {
      z[i] = shift * q[i] - lq[i];
      if (previousQ) z[i] -= beta[iter - 1] * previousQ[i];
    }

    const a = vectorDot(q, z);
    alpha.push(a);
    for (let i = 0; i < n; i += 1) {
      z[i] -= a * q[i];
    }
    orthogonalizeInPlace(z, basis);
    const b = normalizeVectorInPlace(z);
    if (b < 1e-10 || iter === maxIterations - 1) break;
    beta.push(b);
    previousQ = q;
    q = z;
    z = new Float64Array(n);
  }

  const m = alpha.length;
  const tridiagonal = new Float64Array(m * m);
  for (let i = 0; i < m; i += 1) {
    tridiagonal[i * m + i] = alpha[i];
    if (i + 1 < m) {
      tridiagonal[i * m + i + 1] = beta[i];
      tridiagonal[(i + 1) * m + i] = beta[i];
    }
  }
  const ritz = jacobiEigenSymmetric(tridiagonal, m, 48, 1e-11);
  const take = Math.min(targetCount, m);
  const values = new Float64Array(take);
  const vectors = new Float64Array(n * take);

  for (let out = 0; out < take; out += 1) {
    const ritzCol = m - 1 - out;
    values[out] = shift - ritz.values[ritzCol];
    for (let row = 0; row < n; row += 1) {
      let value = 0;
      for (let j = 0; j < m; j += 1) {
        value += basis[j][row] * ritz.vectors[j * m + ritzCol];
      }
      vectors[row * take + out] = value;
    }
  }

  return { values, vectors, stride: take };
}

function jacobiEigenSymmetric(input, n, maxSweeps, eps) {
  const a = new Float64Array(input);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    v[i * n + i] = 1;
  }

  for (let sweep = 0; sweep < Math.max(1, maxSweeps); sweep += 1) {
    let maxOffDiag = 0;
    for (let p = 0; p < n - 1; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = a[p * n + q];
        const absApq = Math.abs(apq);
        if (absApq <= eps) continue;
        if (absApq > maxOffDiag) maxOffDiag = absApq;

        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(angle);
        const s = Math.sin(angle);

        for (let k = 0; k < n; k += 1) {
          if (k === p || k === q) continue;
          const aik = a[k * n + p];
          const akq = a[k * n + q];
          const newP = c * aik - s * akq;
          const newQ = s * aik + c * akq;
          a[k * n + p] = newP;
          a[p * n + k] = newP;
          a[k * n + q] = newQ;
          a[q * n + k] = newQ;
        }

        const newApp = c * c * app - 2 * s * c * apq + s * s * aqq;
        const newAqq = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p * n + p] = newApp;
        a[q * n + q] = newAqq;
        a[p * n + q] = 0;
        a[q * n + p] = 0;

        for (let k = 0; k < n; k += 1) {
          const vip = v[k * n + p];
          const viq = v[k * n + q];
          v[k * n + p] = c * vip - s * viq;
          v[k * n + q] = s * vip + c * viq;
        }
      }
    }
    if (maxOffDiag <= eps) break;
  }

  const order = Array.from({ length: n }, (_, index) => index)
    .sort((left, right) => a[left * n + left] - a[right * n + right]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let sorted = 0; sorted < n; sorted += 1) {
    const original = order[sorted];
    values[sorted] = a[original * n + original];
    for (let row = 0; row < n; row += 1) {
      vectors[row * n + sorted] = v[row * n + original];
    }
  }
  return { values, vectors };
}

function nearestCentroid(vector, centroids) {
  let best = -Infinity;
  let bestIndex = 0;
  for (let c = 0; c < centroids.length; c += 1) {
    const sim = cosine(vector, centroids[c]);
    if (sim > best) {
      best = sim;
      bestIndex = c;
    }
  }
  return bestIndex;
}

function computeClusterCentroids(labels, embeddings) {
  const clusters = [...new Set(labels)];
  const centroids = new Map();
  const counts = new Map();
  for (const cluster of clusters) {
    centroids.set(cluster, new Float32Array(embeddings[0].length));
    counts.set(cluster, 0);
  }
  for (let i = 0; i < labels.length; i += 1) {
    const centroid = centroids.get(labels[i]);
    for (let d = 0; d < embeddings[i].length; d += 1) {
      centroid[d] += embeddings[i][d];
    }
    counts.set(labels[i], counts.get(labels[i]) + 1);
  }
  for (const [cluster, centroid] of centroids) {
    const normalized = l2Normalize(centroid);
    centroid.set(normalized);
    counts.set(cluster, counts.get(cluster) || 1);
  }
  return { centroids, counts };
}

function mergeCloseClusters(labels, embeddings, threshold) {
  while (true) {
    const { centroids } = computeClusterCentroids(labels, embeddings);
    const clusters = [...centroids.keys()];
    let best = threshold;
    let mergeFrom = null;
    let mergeTo = null;

    for (let i = 0; i < clusters.length; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const sim = cosine(centroids.get(clusters[i]), centroids.get(clusters[j]));
        if (sim > best) {
          best = sim;
          mergeTo = clusters[i];
          mergeFrom = clusters[j];
        }
      }
    }

    if (mergeFrom === null) return;
    for (let i = 0; i < labels.length; i += 1) {
      if (labels[i] === mergeFrom) labels[i] = mergeTo;
    }
  }
}

function filterMinorClusters(labels, embeddings, minSize) {
  const { centroids, counts } = computeClusterCentroids(labels, embeddings);
  const major = [...counts.entries()].filter(([, size]) => size >= minSize).map(([cluster]) => cluster);
  if (!major.length) {
    labels.fill(0);
    return;
  }

  for (let i = 0; i < labels.length; i += 1) {
    if (counts.get(labels[i]) >= minSize) continue;
    let bestCluster = major[0];
    let best = -Infinity;
    for (const cluster of major) {
      const sim = cosine(embeddings[i], centroids.get(cluster));
      if (sim > best) {
        best = sim;
        bestCluster = cluster;
      }
    }
    labels[i] = bestCluster;
  }
}

function relabelClusters(labels) {
  const remap = new Map();
  let next = 0;
  const output = new Int32Array(labels.length);
  for (let i = 0; i < labels.length; i += 1) {
    if (!remap.has(labels[i])) {
      remap.set(labels[i], next);
      next += 1;
    }
    output[i] = remap.get(labels[i]);
  }
  return output;
}

function stabilizeCamppLabels(labels, embeddings) {
  if (labels.length < 3) return labels;
  const output = Int32Array.from(labels);
  const { centroids } = computeClusterCentroids(output, embeddings);

  let start = 0;
  while (start < output.length) {
    let end = start + 1;
    while (end < output.length && output[end] === output[start]) end += 1;

    if (start > 0) {
      const previousLabel = output[start - 1];
      const currentLabel = output[start];
      let previousRunStart = start - 1;
      while (previousRunStart > 0 && output[previousRunStart - 1] === previousLabel) {
        previousRunStart -= 1;
      }
      const previousRunLength = start - previousRunStart;
      const runLength = end - start;
      const currentCentroid = centroids.get(currentLabel);
      const previousCentroid = centroids.get(previousLabel);
      if (previousCentroid && currentCentroid && previousRunLength >= 3 && runLength <= 3) {
        const currentScore = cosine(embeddings[start], currentCentroid);
        const previousScore = cosine(embeddings[start], previousCentroid);
        if (currentScore - previousScore < 0.15) {
          output[start] = previousLabel;
        }
      }
    }

    start = end;
  }
  return relabelClusters(output);
}

function kmeansCosine(embeddings, k, iterations) {
  const labels = new Int32Array(embeddings.length);
  const centroids = [];
  for (let i = 0; i < k; i += 1) {
    centroids.push(Float32Array.from(embeddings[Math.floor((i * embeddings.length) / k)]));
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 0; i < embeddings.length; i += 1) {
      labels[i] = nearestCentroid(embeddings[i], centroids);
    }

    const sums = Array.from({ length: k }, () => new Float32Array(embeddings[0].length));
    const counts = new Int32Array(k);
    for (let i = 0; i < embeddings.length; i += 1) {
      counts[labels[i]] += 1;
      for (let d = 0; d < embeddings[i].length; d += 1) {
        sums[labels[i]][d] += embeddings[i][d];
      }
    }
    for (let c = 0; c < k; c += 1) {
      if (!counts[c]) continue;
      centroids[c].set(l2Normalize(sums[c]));
    }
  }

  return relabelClusters(labels);
}

function kmeansEuclidean(points, k, iterations) {
  const labels = new Int32Array(points.length);
  const dim = points[0].length;
  const centroids = initKMeansPlusPlus(points, k);
  let previousInertia = Infinity;

  for (let iter = 0; iter < iterations; iter += 1) {
    let changed = false;
    let inertia = 0;
    for (let i = 0; i < points.length; i += 1) {
      let best = Infinity;
      let bestIndex = 0;
      for (let c = 0; c < k; c += 1) {
        let dist = 0;
        for (let d = 0; d < dim; d += 1) {
          const diff = points[i][d] - centroids[c][d];
          dist += diff * diff;
        }
        if (dist < best) {
          best = dist;
          bestIndex = c;
        }
      }
      inertia += best;
      if (labels[i] !== bestIndex) {
        labels[i] = bestIndex;
        changed = true;
      }
    }

    const sums = Array.from({ length: k }, () => new Float64Array(dim));
    const counts = new Int32Array(k);
    for (let i = 0; i < points.length; i += 1) {
      counts[labels[i]] += 1;
      for (let d = 0; d < dim; d += 1) {
        sums[labels[i]][d] += points[i][d];
      }
    }
    for (let c = 0; c < k; c += 1) {
      if (!counts[c]) continue;
      for (let d = 0; d < dim; d += 1) {
        centroids[c][d] = sums[c][d] / counts[c];
      }
    }
    if (!changed || Math.abs(previousInertia - inertia) <= 1e-4 * Math.max(1, previousInertia)) break;
    previousInertia = inertia;
  }

  return relabelClusters(labels);
}

function initKMeansPlusPlus(points, k) {
  const rng = createMt19937(0);
  const n = points.length;
  const centroids = [];
  const closestDistSq = new Float64Array(n);
  closestDistSq.fill(Infinity);
  const firstIndex = rng.randomInt(n);
  centroids.push(Float64Array.from(points[firstIndex]));

  let currentPotential = updateClosestDistances(points, centroids[0], closestDistSq);
  const localTrials = 2 + Math.floor(Math.log(k));
  for (let c = 1; c < k; c += 1) {
    if (currentPotential <= 0) {
      centroids.push(Float64Array.from(points[rng.randomInt(n)]));
      continue;
    }

    let bestCandidate = 0;
    let bestPotential = Infinity;
    for (let trial = 0; trial < localTrials; trial += 1) {
      const threshold = rng.random() * currentPotential;
      let cumulative = 0;
      let candidate = n - 1;
      for (let i = 0; i < n; i += 1) {
        cumulative += closestDistSq[i];
        if (cumulative >= threshold) {
          candidate = i;
          break;
        }
      }

      let potential = 0;
      const center = points[candidate];
      for (let i = 0; i < n; i += 1) {
        potential += Math.min(closestDistSq[i], squaredEuclidean(points[i], center));
      }
      if (potential < bestPotential) {
        bestPotential = potential;
        bestCandidate = candidate;
      }
    }

    const centroid = Float64Array.from(points[bestCandidate]);
    centroids.push(centroid);
    currentPotential = updateClosestDistances(points, centroid, closestDistSq);
  }
  return centroids;
}

function updateClosestDistances(points, centroid, closestDistSq) {
  let potential = 0;
  for (let i = 0; i < points.length; i += 1) {
    const dist = squaredEuclidean(points[i], centroid);
    if (dist < closestDistSq[i]) closestDistSq[i] = dist;
    potential += closestDistSq[i];
  }
  return potential;
}

function squaredEuclidean(a, b) {
  let dist = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

function matAtA(matrix, rows, cols) {
  const output = new Float64Array(cols * cols);
  for (let r = 0; r < rows; r += 1) {
    const base = r * cols;
    for (let i = 0; i < cols; i += 1) {
      const a = matrix[base + i];
      for (let j = 0; j < cols; j += 1) {
        output[i * cols + j] += a * matrix[base + j];
      }
    }
  }
  return output;
}

function matAtPsiInvA(matrix, psi, rows, cols) {
  const output = new Float64Array(cols * cols);
  for (let r = 0; r < rows; r += 1) {
    const base = r * cols;
    const invPsi = 1 / psi[r];
    for (let i = 0; i < cols; i += 1) {
      const a = matrix[base + i] * invPsi;
      for (let j = 0; j < cols; j += 1) {
        output[i * cols + j] += a * matrix[base + j];
      }
    }
  }
  return output;
}

function invertMatrix(input, n) {
  const a = new Float64Array(n * n * 2);
  const width = n * 2;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      a[i * width + j] = input[i * n + j];
    }
    a[i * width + n + i] = 1;
  }

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    let pivotAbs = Math.abs(a[col * width + col]);
    for (let row = col + 1; row < n; row += 1) {
      const value = Math.abs(a[row * width + col]);
      if (value > pivotAbs) {
        pivot = row;
        pivotAbs = value;
      }
    }
    if (pivotAbs < 1e-14) throw new Error("Matrix is singular.");
    if (pivot !== col) {
      for (let j = 0; j < width; j += 1) {
        const tmp = a[col * width + j];
        a[col * width + j] = a[pivot * width + j];
        a[pivot * width + j] = tmp;
      }
    }

    const diag = a[col * width + col];
    for (let j = 0; j < width; j += 1) {
      a[col * width + j] /= diag;
    }
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row * width + col];
      if (Math.abs(factor) < 1e-18) continue;
      for (let j = 0; j < width; j += 1) {
        a[row * width + j] -= factor * a[col * width + j];
      }
    }
  }

  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      inv[i * n + j] = a[i * width + n + j];
    }
  }
  return inv;
}

function choleskyLower(matrix, n) {
  const lower = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let sum = matrix[i * n + j];
      for (let k = 0; k < j; k += 1) {
        sum -= lower[i * n + k] * lower[j * n + k];
      }
      if (i === j) {
        if (sum <= 0) throw new Error("Matrix is not positive definite.");
        lower[i * n + j] = Math.sqrt(sum);
      } else {
        lower[i * n + j] = sum / lower[j * n + j];
      }
    }
  }
  return lower;
}

function invertLowerTriangular(lower, n) {
  const inv = new Float64Array(n * n);
  for (let col = 0; col < n; col += 1) {
    for (let i = 0; i < n; i += 1) {
      let sum = i === col ? 1 : 0;
      for (let k = 0; k < i; k += 1) {
        sum -= lower[i * n + k] * inv[k * n + col];
      }
      inv[i * n + col] = sum / lower[i * n + i];
    }
  }
  return inv;
}

function matMul(a, b, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < n; k += 1) {
      const av = a[i * n + k];
      if (Math.abs(av) < 1e-18) continue;
      for (let j = 0; j < n; j += 1) {
        out[i * n + j] += av * b[k * n + j];
      }
    }
  }
  return out;
}

function matMulTransposedRight(a, b, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      let sum = 0;
      for (let k = 0; k < n; k += 1) {
        sum += a[i * n + k] * b[j * n + k];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

function solveUpperTransposedVector(lower, vector, n) {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i -= 1) {
    let sum = vector[i];
    for (let k = i + 1; k < n; k += 1) {
      sum -= lower[k * n + i] * x[k];
    }
    x[i] = sum / lower[i * n + i];
  }
  return x;
}

function createMt19937(seed) {
  const mt = new Uint32Array(624);
  let index = 624;
  mt[0] = seed >>> 0;
  for (let i = 1; i < 624; i += 1) {
    const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
    mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
  }

  function nextUint32() {
    if (index >= 624) {
      for (let i = 0; i < 624; i += 1) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        mt[i] = (mt[(i + 397) % 624] ^ (y >>> 1)) >>> 0;
        if (y & 1) mt[i] = (mt[i] ^ 0x9908b0df) >>> 0;
      }
      index = 0;
    }
    let y = mt[index];
    index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  return {
    random() {
      const a = nextUint32() >>> 5;
      const b = nextUint32() >>> 6;
      return (a * 67108864 + b) / 9007199254740992;
    },
    randomInt(maxExclusive) {
      if (maxExclusive <= 1) return 0;
      const max = maxExclusive - 1;
      let mask = max;
      mask |= mask >>> 1;
      mask |= mask >>> 2;
      mask |= mask >>> 4;
      mask |= mask >>> 8;
      mask |= mask >>> 16;
      let value = 0;
      do {
        value = nextUint32() & mask;
      } while (value > max);
      return value;
    },
  };
}

function segmentsFromCamppLabels(windowTimes, labels) {
  if (!windowTimes.length) return [];
  const segments = [];
  let currentStart = windowTimes[0].start;
  let currentEnd = windowTimes[0].end;
  let currentLabel = labels[0];

  for (let i = 1; i < windowTimes.length; i += 1) {
    const time = windowTimes[i];
    const label = labels[i];
    if (label === currentLabel && time.start - currentEnd < 0.01) {
      currentEnd = time.end;
    } else {
      segments.push({ start: currentStart, end: currentEnd, speaker: currentLabel });
      currentStart = time.start;
      currentEnd = time.end;
      currentLabel = label;
    }
  }
  segments.push({ start: currentStart, end: currentEnd, speaker: currentLabel });
  return segments;
}

function postProcessCamppSegments(segments) {
  if (segments.length > 1) {
    for (let i = 0; i < segments.length - 1; i += 1) {
      if (segments[i].end > segments[i + 1].start) {
        const mid = (segments[i].end + segments[i + 1].start) / 2;
        segments[i].end = mid;
        segments[i + 1].start = mid;
      }
    }
  }

  let output = segments;
  if (output.length > 1) {
    const merged = [output[0]];
    for (const segment of output.slice(1)) {
      const previous = merged[merged.length - 1];
      if (segment.speaker === previous.speaker && segment.start - previous.end <= 4.0) {
        previous.end = segment.end;
      } else {
        merged.push({ ...segment });
      }
    }
    output = merged;
  }

  if (output.length > 1) {
    const filtered = [];
    for (let i = 0; i < output.length; i += 1) {
      const segment = output[i];
      if (segment.end - segment.start > 0.78) {
        filtered.push(segment);
        continue;
      }
      const previousSpeaker = filtered.length ? filtered[filtered.length - 1].speaker : null;
      const nextSpeaker = i + 1 < output.length ? output[i + 1].speaker : null;
      if (previousSpeaker !== null && previousSpeaker === nextSpeaker) {
        filtered[filtered.length - 1].end = segment.end;
      }
    }
    if (filtered.length) output = filtered;
  }

  if (output.length > 1) {
    const merged = [output[0]];
    for (const segment of output.slice(1)) {
      const previous = merged[merged.length - 1];
      if (segment.speaker === previous.speaker) {
        previous.end = segment.end;
      } else {
        merged.push({ ...segment });
      }
    }
    output = merged;
  }

  const durations = new Map();
  for (const segment of output) {
    durations.set(segment.speaker, (durations.get(segment.speaker) || 0) + segment.end - segment.start);
  }
  const ranked = [...durations.entries()].sort((a, b) => b[1] - a[1]).map(([speaker]) => speaker);
  const remap = new Map(ranked.map((speaker, index) => [speaker, index]));
  return output.map((segment) => ({
    start: segment.start,
    end: segment.end,
    speaker: remap.get(segment.speaker) ?? segment.speaker,
  }));
}

function camppSegmentHash(segments) {
  return hashString((segments || []).map((s) => `${s.speaker}:${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|"));
}

function computeCamppClusterResult(embedding, duration, options = {}) {
  const rawLabels = clusterCamppEmbeddings(embedding.embeddings, duration, options);
  const rawSpeakers = new Set(rawLabels).size;
  const labels = rawLabels;
  const stableSpeakers = new Set(labels).size;
  const modelSegments = postProcessCamppSegments(segmentsFromCamppLabels(embedding.windowTimes, labels));
  const segments = desktopPostProcessDiarizationSegments(modelSegments, options.asrWords || null);
  const speakers = new Set(segments.map((segment) => segment.speaker)).size;
  const segmentHash = camppSegmentHash(segments);
  return {
    rawLabels,
    labels,
    segments,
    rawSpeakers,
    stableSpeakers,
    speakers,
    segmentHash,
  };
}

function computeCamppLongFormClusterResult(embedding) {
  const labels = senkoUmapHdbscanCluster(embedding.embeddings);
  filterMinorClusters(labels, embedding.embeddings, 10);
  mergeCloseClusters(labels, embedding.embeddings, CAMPP_MERGE_COS);
  const relabeled = relabelClusters(labels);
  const modelSegments = postProcessCamppSegments(segmentsFromCamppLabels(embedding.windowTimes, relabeled));
  const segments = desktopPostProcessDiarizationSegments(modelSegments);
  const speakers = new Set(segments.map((segment) => segment.speaker)).size;
  return {
    labels: relabeled,
    segments,
    speakers,
    segmentHash: camppSegmentHash(segments),
  };
}

async function computeCamppLongFormClusterResultWebGpu(embedding) {
  const labels = await senkoUmapHdbscanClusterWebGpuDistance(embedding.embeddings);
  filterMinorClusters(labels, embedding.embeddings, 10);
  mergeCloseClusters(labels, embedding.embeddings, CAMPP_MERGE_COS);
  const relabeled = relabelClusters(labels);
  const segments = postProcessCamppSegments(segmentsFromCamppLabels(embedding.windowTimes, relabeled));
  const speakers = new Set(segments.map((segment) => segment.speaker)).size;
  return {
    labels: relabeled,
    segments,
    speakers,
    segmentHash: camppSegmentHash(segments),
  };
}

async function addCamppLongFormClusteringBenchmark(options, embedding) {
  if (!Array.isArray(options?.benchmarkStages)) return;
  if ((embedding?.embeddings?.length || 0) < 10) return;

  const attempts = [];
  let jsResult = null;
  let webgpuResult = null;
  const jsStarted = performance.now();
  try {
    jsResult = computeCamppLongFormClusterResult(embedding);
    attempts.push({
      runtime: "wasm",
      provider: "js",
      elapsedSeconds: benchmarkSeconds(jsStarted),
      summary: {
        embeddings: embedding.embeddings.length,
        turns: jsResult.segments.length,
        speakers: jsResult.speakers,
        segmentHash: jsResult.segmentHash,
        outputHash: jsResult.segmentHash,
        implementation: "JS UMAP cosine + JS HDBSCAN",
        calibrationInput: "current CAM++ embeddings forced through long-form UMAP/HDBSCAN",
      },
    });
  } catch (error) {
    attempts.push({
      runtime: "wasm",
      provider: "js",
      error: { message: error.message || String(error) },
    });
  }

  const webgpuStarted = performance.now();
  try {
    webgpuResult = await computeCamppLongFormClusterResultWebGpu(embedding);
    attempts.push({
      runtime: "webgpu",
      provider: "webgpu",
      elapsedSeconds: benchmarkSeconds(webgpuStarted),
      summary: {
        embeddings: embedding.embeddings.length,
        turns: webgpuResult.segments.length,
        speakers: webgpuResult.speakers,
        segmentHash: webgpuResult.segmentHash,
        outputHash: webgpuResult.segmentHash,
        implementation: "WebGPU precomputed cosine distance for UMAP + JS HDBSCAN",
        calibrationInput: "current CAM++ embeddings forced through long-form UMAP/HDBSCAN",
      },
    });
  } catch (error) {
    attempts.push({
      runtime: "webgpu",
      provider: "webgpu",
      error: { message: error.message || String(error) },
    });
    log(`[Benchmark] Stage CAM++ long-form clustering: WEBGPU failed: ${error.message || String(error)}`);
  }

  const wasmAttempt = attempts.find((attempt) => attempt.runtime === "wasm" && !attempt.error);
  const webgpuAttempt = attempts.find((attempt) => attempt.runtime === "webgpu" && !attempt.error);
  const outputHashEqual = wasmAttempt && webgpuAttempt
    ? wasmAttempt.summary.outputHash === webgpuAttempt.summary.outputHash
    : null;
  const speedupWebgpuOverWasm = wasmAttempt?.elapsedSeconds && webgpuAttempt?.elapsedSeconds
    ? Number((wasmAttempt.elapsedSeconds / webgpuAttempt.elapsedSeconds).toFixed(3))
    : null;
  const webgpuAccepted = Boolean(webgpuAttempt && outputHashEqual === true && speedupWebgpuOverWasm > 1);
  let rejectionReason = null;
  if (webgpuAttempt && !webgpuAccepted) {
    if (outputHashEqual !== true) rejectionReason = "webgpu output hash did not match js long-form clustering";
    else if (!(speedupWebgpuOverWasm > 1)) rejectionReason = "webgpu was not faster than js long-form clustering";
  } else if (!webgpuAttempt) {
    const webgpuError = attempts.find((attempt) => attempt.runtime === "webgpu" && attempt.error);
    if (webgpuError) rejectionReason = webgpuError.error?.message || "webgpu attempt failed";
  }
  const mismatchDetails = outputHashEqual === false
    ? benchmarkMismatchDetails(
        "CAM++ long-form clustering",
        jsResult,
        webgpuResult,
        wasmAttempt?.summary,
        webgpuAttempt?.summary
      )
    : null;
  addBenchmarkStage(options, {
    name: "CAM++ long-form clustering",
    capability: "wasm-webgpu",
    attempts,
    selectedRuntime: webgpuAccepted ? "webgpu" : "wasm",
    selectedProvider: webgpuAccepted ? "webgpu" : "js",
    comparison: {
      speedupWebgpuOverWasm,
      outputHashEqual,
      webgpuAccepted,
      rejectionReason,
      ...(mismatchDetails ? { mismatchDetails } : {}),
    },
  });
  if (webgpuAttempt && wasmAttempt) {
    log(
      `[Benchmark] Stage CAM++ long-form clustering: JS ${wasmAttempt.elapsedSeconds.toFixed(2)}s, ` +
      `WEBGPU ${webgpuAttempt.elapsedSeconds.toFixed(2)}s, hashEqual=${outputHashEqual}.`
    );
  }
}

function diarSegmentDuration(segment) {
  return Math.max(0, Number(segment.end) - Number(segment.start));
}

function cloneDiarSegment(segment, speaker = segment.speaker) {
  return {
    start: Number(segment.start),
    end: Number(segment.end),
    speaker: normalizeSpeakerId(speaker),
  };
}

function mergeDiarSegmentsWithGap(segments, maxGap = 0.3) {
  const sorted = (segments || [])
    .map((segment) => cloneDiarSegment(segment))
    .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start)
    .sort((a, b) => (a.start - b.start) || (a.speaker - b.speaker));
  if (!sorted.length) return [];

  const merged = [cloneDiarSegment(sorted[0])];
  for (const segment of sorted.slice(1)) {
    const previous = merged[merged.length - 1];
    const gap = segment.start - previous.end;
    if (segment.speaker === previous.speaker && gap <= maxGap) {
      previous.end = Math.max(previous.end, segment.end);
    } else {
      merged.push(cloneDiarSegment(segment));
    }
  }
  return merged;
}

function resolveDiarizationFragmentZones(segments, shortThreshold = 0.5, minZoneSize = 3) {
  if ((segments || []).length < minZoneSize) return segments || [];
  const result = [];
  let i = 0;
  while (i < segments.length) {
    if (diarSegmentDuration(segments[i]) < shortThreshold) {
      let j = i;
      while (j < segments.length && diarSegmentDuration(segments[j]) < shortThreshold) j += 1;
      if (j - i >= minZoneSize) {
        const durations = new Map();
        for (let k = i; k < j; k += 1) {
          durations.set(segments[k].speaker, (durations.get(segments[k].speaker) || 0) + diarSegmentDuration(segments[k]));
        }
        let dominant = segments[i].speaker;
        let best = -Infinity;
        for (const [speaker, duration] of durations.entries()) {
          if (duration > best) {
            best = duration;
            dominant = speaker;
          }
        }
        result.push({ start: segments[i].start, end: segments[j - 1].end, speaker: dominant });
        i = j;
        continue;
      }
    }
    result.push(cloneDiarSegment(segments[i]));
    i += 1;
  }
  return result;
}

function countAsrWordsInRange(words, start, end) {
  if (!Array.isArray(words) || !words.length) return null;
  let count = 0;
  for (const word of words) {
    const mid = (Number(word.start || 0) + Number(word.end || 0)) / 2;
    if (start <= mid && mid <= end) count += 1;
  }
  return count;
}

function naturalTurnMergeDiarization(segments, maxPause = 2.0, asrWords = null) {
  if ((segments || []).length < 3) return segments || [];
  const sorted = segments.map((segment) => cloneDiarSegment(segment)).sort((a, b) => a.start - b.start);
  const speakers = [...new Set(sorted.map((segment) => segment.speaker))];
  const turns = [];

  for (const speaker of speakers) {
    const indices = [];
    for (let i = 0; i < sorted.length; i += 1) {
      if (sorted[i].speaker === speaker) indices.push(i);
    }
    if (!indices.length) continue;
    let turnStart = sorted[indices[0]].start;
    let turnEnd = sorted[indices[0]].end;
    let turnIndices = [indices[0]];
    for (const index of indices.slice(1)) {
      const gap = sorted[index].start - turnEnd;
      if (gap < maxPause) {
        turnEnd = Math.max(turnEnd, sorted[index].end);
        turnIndices.push(index);
      } else {
        turns.push({ start: turnStart, end: turnEnd, speaker, indices: turnIndices });
        turnStart = sorted[index].start;
        turnEnd = sorted[index].end;
        turnIndices = [index];
      }
    }
    turns.push({ start: turnStart, end: turnEnd, speaker, indices: turnIndices });
  }

  turns.sort((a, b) => a.start - b.start);
  const secondary = new Array(turns.length).fill(false);
  const primaryOf = new Array(turns.length).fill(-1);
  for (let i = 0; i < turns.length; i += 1) {
    if (secondary[i]) continue;
    const t1 = turns[i];
    for (let j = i + 1; j < turns.length; j += 1) {
      if (secondary[j]) continue;
      const t2 = turns[j];
      if (t2.start >= t1.end) break;
      if (t2.end <= t1.end && t2.speaker !== t1.speaker) {
        secondary[j] = true;
        primaryOf[j] = i;
      }
    }
  }

  const reassign = new Map();
  for (let i = 0; i < turns.length; i += 1) {
    if (!secondary[i] || primaryOf[i] < 0) continue;
    const turn = turns[i];
    if (turn.end - turn.start >= 2.0) continue;
    const wordCount = countAsrWordsInRange(asrWords, turn.start, turn.end);
    if (wordCount !== null && wordCount > 3) continue;
    const primarySpeaker = turns[primaryOf[i]].speaker;
    for (const index of turn.indices) reassign.set(index, primarySpeaker);
  }

  const reassigned = sorted.map((segment, index) => cloneDiarSegment(segment, reassign.get(index) ?? segment.speaker));
  return mergeDiarSegmentsWithGap(reassigned, 0.5);
}

function desktopPostProcessDiarizationSegments(segments, asrWords = null) {
  if (!segments?.length) return segments || [];
  let processed = mergeDiarSegmentsWithGap(segments, 0.3);
  processed = resolveDiarizationFragmentZones(processed, 0.5, 3);
  processed = naturalTurnMergeDiarization(processed, 2.0, asrWords);
  return mergeDiarSegmentsWithGap(processed, 0.3);
}



async function runCamppDiarization(samples, options = {}) {
  const duration = samples.length / VAD_SAMPLE_RATE;
  if (duration < 0.5) {
    renderDiarization([]);
    return {
      segments: [],
      speakers: 0,
      elapsed: 0,
      chunks: 0,
      embeddings: 0,
      backend: "campp",
      executionProvider: { segmentation: "wasm", embedding: camppExecutionProvider },
    };
  }

  const started = performance.now();
  const speechRegionRunner = (runtime, benchmarkContext) => getPyannoteSpeechRegionsWithOptionalWebGpuAutotune(samples, {
    progress: (done, total) => {
      if (options.progress) options.progress(0.05 + (done / Math.max(1, total)) * 0.15);
    },
  }, runtime, benchmarkContext);
  let vad = decodeCamppSpeechRegionsCheckpoint(options.resumeCheckpoints?.campp_speech_regions);
  if (vad?.regions?.length) {
    log(`[resume_after_kill] Resumed CAM++ speech regions: ${vad.regions.length} region(s).`);
    if (options.progress) options.progress(0.20);
  } else {
    vad = shouldBenchmarkWebGpuStage(options, "CAM++ speech regions (pyannote segmentation)")
      ? await runBenchmarkDualProviderStage(
          options,
          "CAM++ speech regions (pyannote segmentation)",
          speechRegionRunner,
          unloadDiarizationSegmentationSessionOnly,
          summarizePyannoteSpeechRegionsForBenchmark
        )
      : await speechRegionRunner(calibratedProviderForStage("CAM++ speech regions (pyannote segmentation)", "wasm") === "webgpu"
          ? "calibrated-webgpu"
          : "wasm");
    await writeResumeJsonCheckpoint(options.resumeContext, "campp_speech_regions", encodeCamppSpeechRegionsCheckpoint(vad), {
      regions: vad.regions?.length || 0,
      overlapRegions: vad.overlapRegions?.length || 0,
      chunks: vad.chunks || 0,
    });
  }
  const embeddingRunner = async (runtime, benchmarkContext = {}) => {
    const runFull = (batchSize, batchTuning = null, runtimeGuard = {}) => extractCamppEmbeddings(samples, vad.regions, {
      batchSize,
      batchTuning,
      abortStageName: "CAM++ speaker embedding",
      ...runtimeGuard,
      progress: (done, total) => {
        if (options.progress) options.progress(0.20 + (done / Math.max(1, total)) * 0.55);
      },
    });
    if (runtime === "calibrated-webgpu") {
      return runFull(calibratedBatchSizeForStage("CAM++ speaker embedding", CAMPP_BATCH_SIZE));
    }
    if (runtime !== "webgpu") return runFull(CAMPP_BATCH_SIZE);
    if (benchmarkContext.useCalibratedWebGpuBatch) {
      const batchSize = calibratedBatchSizeForStage(
        "CAM++ speaker embedding",
        defaultWebGpuBatchSize("campp_embedding", CAMPP_BATCH_SIZE)
      );
      return runFull(batchSize, {
        selectedBatchSize: batchSize,
        source: "calibration_profile",
        totalSeconds: 0,
      });
    }
    const candidates = webgpuBatchCandidates("campp_embedding", CAMPP_BATCH_SIZE);
    const tuning = await autotuneWebGpuBatch("CAM++ speaker embedding", candidates, (batchSize) => (
      extractCamppEmbeddings(samples, vad.regions, {
        batchSize,
        tuneLimitWindows: autotuneSampleCount(candidates, Math.max(candidates.length, 128)),
      })
    ), webGpuAutotuneGuardOptions(benchmarkContext));
    return runFull(tuning.selectedBatchSize, tuning, webGpuRuntimeAbortOptions(benchmarkContext, tuning));
  };
  const embedding = shouldBenchmarkWebGpuStage(options, "CAM++ speaker embedding")
    ? await runBenchmarkDualProviderStage(
        options,
        "CAM++ speaker embedding",
        embeddingRunner,
        unloadCamppEmbeddingSessionOnly,
        summarizeCamppEmbeddingForBenchmark
      )
    : await embeddingRunner(calibratedProviderForStage("CAM++ speaker embedding", "wasm") === "webgpu"
        ? "calibrated-webgpu"
        : "wasm");

  if (!embedding.embeddings.length) {
    throw new Error("CAM++ produced no speaker embeddings.");
  }

  if (options.progress) options.progress(0.80);
  const clusterStarted = performance.now();
  const clusterResult = computeCamppClusterResult(embedding, duration, options);
  const jsClusterSeconds = benchmarkSeconds(clusterStarted);
  const { rawLabels, labels, segments, rawSpeakers, stableSpeakers, speakers, segmentHash } = clusterResult;
  if (Array.isArray(options.benchmarkStages)) {
    const jsClusterResult = { segments, rawLabels, labels };
    if (shouldBenchmarkWebGpuStage(options, "CAM++ clustering")) {
    let webgpuClusterResult = null;
    const attempts = [{
      runtime: "wasm",
      provider: "js",
      elapsedSeconds: jsClusterSeconds,
      summary: {
        turns: segments.length,
        speakers,
        rawSpeakers,
        stableSpeakers,
        segmentHash,
        outputHash: segmentHash,
      },
    }];
    try {
      const webgpuClusterStarted = performance.now();
      const webgpuRawLabels = await clusterCamppEmbeddingsWebGpuCandidate(embedding.embeddings, duration, options);
      const webgpuRawSpeakers = new Set(webgpuRawLabels).size;
      const webgpuLabels = webgpuRawLabels;
      const webgpuStableSpeakers = new Set(webgpuLabels).size;
      let webgpuSegments = segmentsFromCamppLabels(embedding.windowTimes, webgpuLabels);
      webgpuSegments = postProcessCamppSegments(webgpuSegments);
      webgpuSegments = desktopPostProcessDiarizationSegments(webgpuSegments, options.asrWords || null);
      webgpuClusterResult = { segments: webgpuSegments, rawLabels: webgpuRawLabels, labels: webgpuLabels };
      const webgpuSpeakers = new Set(webgpuSegments.map((segment) => segment.speaker)).size;
      const webgpuSegmentHash = hashString(webgpuSegments.map((s) => `${s.speaker}:${Number(s.start).toFixed(3)}-${Number(s.end).toFixed(3)}`).join("|"));
      attempts.push({
        runtime: "webgpu",
        provider: "webgpu",
        elapsedSeconds: benchmarkSeconds(webgpuClusterStarted),
        summary: {
          turns: webgpuSegments.length,
          speakers: webgpuSpeakers,
          rawSpeakers: webgpuRawSpeakers,
          stableSpeakers: webgpuStableSpeakers,
          segmentHash: webgpuSegmentHash,
          outputHash: webgpuSegmentHash,
          implementation: "WebGPU cosine affinity + JS spectral eigen/kmeans/postprocess",
        },
      });
    } catch (error) {
      attempts.push({
        runtime: "webgpu",
        error: { message: error.message || String(error) },
      });
      log(`[Benchmark] Stage CAM++ clustering: WEBGPU failed: ${error.message || String(error)}`);
    }
    const webgpuAttempt = attempts.find((attempt) => attempt.runtime === "webgpu" && !attempt.error);
    const outputHashEqual = webgpuAttempt
      ? attempts[0].summary.outputHash === webgpuAttempt.summary.outputHash
      : null;
    const speedupWebgpuOverWasm = webgpuAttempt?.elapsedSeconds
      ? Number((attempts[0].elapsedSeconds / webgpuAttempt.elapsedSeconds).toFixed(3))
      : null;
    const webgpuAccepted = Boolean(webgpuAttempt && outputHashEqual === true && speedupWebgpuOverWasm > 1);
    let rejectionReason = null;
    if (webgpuAttempt && !webgpuAccepted) {
      if (outputHashEqual !== true) rejectionReason = "webgpu output hash did not match js";
      else if (!(speedupWebgpuOverWasm > 1)) rejectionReason = "webgpu was not faster than js";
    }
    const mismatchDetails = outputHashEqual === false
      ? benchmarkMismatchDetails(
          "CAM++ clustering",
          jsClusterResult,
          webgpuClusterResult,
          attempts[0].summary,
          webgpuAttempt?.summary
        )
      : null;
    addBenchmarkStage(options, {
      name: "CAM++ clustering",
      capability: "wasm-webgpu",
      attempts,
      selectedRuntime: webgpuAccepted ? "webgpu" : "wasm",
      selectedProvider: webgpuAccepted ? "webgpu" : "js",
      comparison: {
        speedupWebgpuOverWasm,
        outputHashEqual,
        webgpuAccepted,
        rejectionReason,
        ...(mismatchDetails ? { mismatchDetails } : {}),
      },
    });
    if (webgpuAttempt) {
      log(
        `[Benchmark] Stage CAM++ clustering: JS ${attempts[0].elapsedSeconds.toFixed(2)}s, ` +
        `WEBGPU ${webgpuAttempt.elapsedSeconds.toFixed(2)}s, hashEqual=${outputHashEqual}.`
      );
    }
    await addCamppLongFormClusteringBenchmark(options, embedding);
    }

    const embeddingBench = options.benchmarkStageResults?.["CAM++ speaker embedding"];
    const embeddingStage = embeddingBench?.stage ||
      options.benchmarkStages.find((stage) => stage.name === "CAM++ speaker embedding");
    const webgpuEmbeddingEntry = embeddingBench?.webgpu;
    const wasmEmbeddingEntry = embeddingBench?.wasm;
    if (embeddingStage?.comparison && webgpuEmbeddingEntry?.result && wasmEmbeddingEntry?.result) {
      const downstreamStarted = performance.now();
      let downstreamCheck = null;
      try {
        const webgpuEmbeddingCluster = computeCamppClusterResult(webgpuEmbeddingEntry.result, duration, options);
        const downstreamSegmentHashEqual = segmentHash === webgpuEmbeddingCluster.segmentHash;
        const mismatchDetails = embeddingStage.comparison.mismatchDetails ||
          benchmarkMismatchDetails(
            "CAM++ speaker embedding",
            wasmEmbeddingEntry.result,
            webgpuEmbeddingEntry.result,
            wasmEmbeddingEntry.attempt?.summary,
            webgpuEmbeddingEntry.attempt?.summary
          );
        const numericToleranceAccepted = embeddingDiffWithinTolerance(mismatchDetails);
        const webgpuAttemptForEmbedding = embeddingStage.attempts?.find((attempt) => attempt.runtime === "webgpu" && !attempt.error);
        const speedup = embeddingStage.comparison.speedupWebgpuOverWasm;
        const webgpuAcceptedByTolerance = Boolean(
          webgpuAttemptForEmbedding?.provider === "webgpu" &&
          numericToleranceAccepted &&
          downstreamSegmentHashEqual &&
          speedup > 1
        );
        downstreamCheck = {
          elapsedSeconds: benchmarkSeconds(downstreamStarted),
          wasmSegmentHash: segmentHash,
          webgpuSegmentHash: webgpuEmbeddingCluster.segmentHash,
          downstreamSegmentHashEqual,
          webgpuTurns: webgpuEmbeddingCluster.segments.length,
          webgpuSpeakers: webgpuEmbeddingCluster.speakers,
        };
        embeddingStage.comparison = {
          ...embeddingStage.comparison,
          mismatchDetails,
          numericToleranceAccepted,
          downstreamSegmentHashEqual,
          downstreamCheck,
          webgpuAcceptedByTolerance,
          acceptanceMode: webgpuAcceptedByTolerance ? "numeric-tolerance+downstream-segment-hash" : null,
          webgpuAccepted: webgpuAcceptedByTolerance || embeddingStage.comparison.webgpuAccepted,
          rejectionReason: webgpuAcceptedByTolerance ? null : embeddingStage.comparison.rejectionReason,
        };
        if (webgpuAcceptedByTolerance) {
          embeddingStage.selectedRuntime = "webgpu";
          embeddingStage.selectedProvider = "webgpu";
          options.benchmarkSelectedProviders = options.benchmarkSelectedProviders || {};
          options.benchmarkSelectedProviders["CAM++ speaker embedding"] = "webgpu";
          log("[Benchmark] Stage CAM++ speaker embedding: WebGPU accepted by numeric tolerance and downstream segment hash.");
        } else {
          log(
            `[Benchmark] Stage CAM++ speaker embedding downstream check: ` +
            `numericTolerance=${numericToleranceAccepted}, segmentHashEqual=${downstreamSegmentHashEqual}.`
          );
        }
      } catch (error) {
        downstreamCheck = {
          elapsedSeconds: benchmarkSeconds(downstreamStarted),
          error: { message: error.message || String(error) },
        };
        embeddingStage.comparison = {
          ...embeddingStage.comparison,
          downstreamCheck,
        };
        log(`[Benchmark] Stage CAM++ speaker embedding downstream check failed: ${error.message || String(error)}`);
      }
    }
  }
  const elapsed = (performance.now() - started) / 1000;
  renderDiarization(segments);
  log(
    `CAM++ clustering: ${vad.regions.length} speech region(s), ${embedding.embeddings.length} embedding(s), ` +
    `${rawSpeakers} raw cluster(s), ${stableSpeakers} stabilized cluster(s), ${speakers} final speaker(s).`
  );
  log(`CAM++ diarization finished in ${elapsed.toFixed(2)}s: ${segments.length} turn(s), ${speakers} speaker(s), ${embedding.embeddings.length} embedding(s).`);
  if (vad.overlapRegions.length) {
    log(`Pyannote overlap regions: ${vad.overlapRegions.length}.`);
  }
  if (options.progress) options.progress(1);
  const embeddingProvider = options.benchmarkSelectedProviders?.["CAM++ speaker embedding"] ||
    embedding.benchmarkSelectedProvider ||
    camppExecutionProvider;
  const segmentationProvider = vad.benchmarkSelectedProvider ||
    options.benchmarkSelectedProviders?.["CAM++ speech regions (pyannote segmentation)"] ||
    diarizationExecutionProvider;
  return {
    segments,
    speakers,
    elapsed,
    chunks: vad.chunks,
    embeddings: embedding.embeddings.length,
    overlapRegions: vad.overlapRegions,
    backend: "campp",
    executionProvider: {
      segmentation: segmentationProvider,
      embedding: embeddingProvider,
    },
  };
}

async function runDiarization(samples, options = {}) {
  if (options.speakerModel && !SUPPORTED_SPEAKER_MODELS.has(options.speakerModel)) {
    throw new Error(`Speaker model is not supported in offline PWA: ${options.speakerModel}`);
  }
  if (options.speakerModel === "pyannote_community1_vbx") {
    return runPyannoteCommunityDiarization(samples, options);
  }
  return runCamppDiarization(samples, options);
}

async function writeModelFile(file, response, progressEl, progressOffset, progressTotal, onProgress) {
  const dir = await opfsModelDir();
  if (dir.kind === "idb") {
    const name = modelFileName(file);
    if (window.caches) {
      let cacheResponse;
      if (typeof isIOS === "function" && isIOS()) {
        const blob = await response.blob();
        if (file.bytes && blob.size !== file.bytes) {
          throw new Error(`${file.id} size mismatch: got ${blob.size}, expected ${file.bytes}`);
        }
        cacheResponse = new Response(blob, {
          headers: response.headers,
          status: response.status,
          statusText: response.statusText
        });
        if (progressTotal && onProgress) {
          setProgress(progressEl, progressOffset + blob.size, progressTotal);
          onProgress(progressOffset + blob.size, progressTotal);
        }
      } else {
        const reader = response.body.getReader();
        let loaded = 0;
        const stream = new ReadableStream({
          async pull(controller) {
            try {
              const { done, value } = await readStreamChunkWithTimeout(reader, file.id);
              if (done) {
                controller.close();
                return;
              }
              loaded += value.byteLength;
              setProgress(progressEl, progressOffset + loaded, progressTotal);
              if (onProgress) onProgress(progressOffset + loaded, progressTotal);
              controller.enqueue(value);
            } catch (err) {
              controller.error(err);
            }
          },
          cancel() {
            reader.cancel();
          }
        });
        cacheResponse = new Response(stream, {
          headers: response.headers,
          status: response.status,
          statusText: response.statusText
        });
      }
      try {
        await cacheModelFile(name, cacheResponse);
        const stored = await readCachedModelFile(name);
        await verifyModelBlobIntegrity(file, stored);
        await idbDeleteFile(dir.scope, name).catch(() => null);
        return;
      } catch (error) {
        clearModelIntegrity(file);
        await deleteCachedModelFile(name).catch(() => null);
        throw error;
      }
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    try {
      while (true) {
        const { done, value } = await readStreamChunkWithTimeout(reader, file.id);
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        setProgress(progressEl, progressOffset + loaded, progressTotal);
        if (onProgress) onProgress(progressOffset + loaded, progressTotal);
      }
      const blob = new Blob(chunks, { type: "application/octet-stream" });
      await verifyModelBlobIntegrity(file, blob);
      await idbPutFile(dir.scope, name, blob);
      return;
    } catch (error) {
      clearModelIntegrity(file);
      await idbDeleteFile(dir.scope, name).catch(() => null);
      throw error;
    }
  }
  const handle = await dir.getFileHandle(modelFileName(file), { create: true });
  const writable = await handle.createWritable();
  const reader = response.body.getReader();
  let loaded = 0;

  try {
    while (true) {
      const { done, value } = await readStreamChunkWithTimeout(reader, file.id);
      if (done) break;
      await writable.write(value);
      loaded += value.byteLength;
      setProgress(progressEl, progressOffset + loaded, progressTotal);
      if (onProgress) onProgress(progressOffset + loaded, progressTotal);
    }
    await writable.close();

    const stored = await handle.getFile();
    await verifyModelBlobIntegrity(file, stored);
  } catch (error) {
    clearModelIntegrity(file);
    await writable.abort().catch(() => null);
    await removeOpfsFile(dir, modelFileName(file));
    throw error;
  }
}

async function downloadPack(packId, options = {}) {
  const pack = manifest.packs.find((item) => item.id === packId);
  if (!pack) return;
  log(`Downloading pack: ${pack.name}`);

  const progressEl = $(options.progressId || `progress-${pack.id}`);
  const totalBytes = options.progressTotal || packSize(pack);
  let completedBytes = options.progressOffset || 0;
  const onProgress = options.onProgress || null;

  for (const file of pack.files) {
    const status = await fileStatus(file);
    if (status.ready) {
      log(`Skip ${file.id}; already stored.`);
      completedBytes += file.bytes || status.size || 0;
      setProgress(progressEl, completedBytes, totalBytes);
      if (onProgress) onProgress(completedBytes, totalBytes);
      continue;
    }

    log(`Fetching ${file.id}`);
    if (!navigator.onLine) {
      throw new Error(`Cannot download ${file.id} while offline. Reconnect once to finish the offline model pack.`);
    }
    const response = await fetchWithTimeout(
      file.download_url,
      { cache: "no-store" },
      DOWNLOAD_RESPONSE_TIMEOUT_MS
    );
    if (!response.ok || !response.body) {
      throw new Error(`download failed for ${file.id}: ${response.status}`);
    }
    await writeModelFile(file, response, progressEl, completedBytes, totalBytes, onProgress);
    completedBytes += file.bytes || Number(response.headers.get("Content-Length")) || 0;
    log(`Stored ${file.id}`);
    await updateRuntimeStatus();
  }

  log(`Pack ready: ${pack.name}`);
  if (options.refresh !== false) {
    await renderPacks();
    await refreshOfflineBootstrapState();
  }
}

function scheduleBootstrapReload() {
  if (bootstrapReloadScheduled) return;
  bootstrapReloadScheduled = true;
  const message = $("offline-bootstrap-message");
  if (message) {
    message.textContent = "Đã tải xong dữ liệu offline. Ứng dụng sẽ mở lại để Calibrate cấu hình tối ưu cho thiết bị này.";
  }
  setProgress($("offline-bootstrap-progress-bar"), 1, 1);
  window.setTimeout(() => {
    window.location.reload();
  }, 900);
}

async function downloadRequiredOfflinePack() {
  if (offlineBootstrapBusy) return;
  if (!manifest) await loadManifest();
  const packs = requiredOfflinePacks();
  if (packs.length !== REQUIRED_OFFLINE_PACK_IDS.length) {
    throw new Error("Required offline pack is missing from the model manifest.");
  }
  await requestPersistentStorage(true);

  offlineBootstrapBusy = true;
  offlineBootstrapError = "";
  await refreshOfflineBootstrapState();
  let completed = false;
  const showBootstrapProgress = isStandaloneApp() || document.body.classList.contains("standalone-app");
  try {
    await requestScreenWakeLockFor("bootstrap");
    const bootstrapMessage = $("offline-bootstrap-message");
    const runtimeProgress = (ready, total) => {
      const pct = total > 0 ? Math.max(1, Math.min(100, (ready / total) * 100)) : 1;
      if (showBootstrapProgress) {
        setPipelineProgress("Đang tải mã chạy offline", pct);
      }
      if (bootstrapMessage) {
        bootstrapMessage.textContent = `Đang tải mã chạy offline (${ready}/${total}). ${screenWakeLockMessage()}`;
      }
    };
    await ensureRequiredRuntimeAssetsCached({
      progressId: "offline-bootstrap-progress-bar",
      onProgress: runtimeProgress,
    });

    const totalBytes = packs.reduce((sum, pack) => sum + packSize(pack), 0);
    let completedBytes = 0;
    if (bootstrapMessage) {
      bootstrapMessage.textContent = `Đang tải dữ liệu offline lần đầu. ${screenWakeLockMessage()}`;
    }
    if (showBootstrapProgress) setPipelineProgress("Đang tải dữ liệu offline", 1);
    setProgress($("offline-bootstrap-progress-bar"), 0, totalBytes || 1);

    // Callback real-time: cập nhật pipeline progress bar theo từng byte tải về
    const onProgress = (loaded, total) => {
      const pct = total > 0 ? Math.max(1, Math.min(99, (loaded / total) * 100)) : 1;
      if (showBootstrapProgress) {
        setPipelineProgress("Đang tải dữ liệu offline", pct);
      }
      if (bootstrapMessage) {
        bootstrapMessage.textContent = `Đang tải dữ liệu offline lần đầu (${Math.round(pct)}%). ${screenWakeLockMessage()}`;
      }
    };

    for (const pack of packs) {
      await downloadPack(pack.id, {
        progressId: "offline-bootstrap-progress-bar",
        progressOffset: completedBytes,
        progressTotal: totalBytes,
        refresh: false,
        onProgress,
      });
      completedBytes += packSize(pack);
      setProgress($("offline-bootstrap-progress-bar"), completedBytes, totalBytes || 1);
      if (showBootstrapProgress) {
        setPipelineProgress("Đang tải dữ liệu offline", (completedBytes / Math.max(1, totalBytes)) * 100);
      }
    }
    await renderPacks();
    const finalStatus = await refreshOfflineBootstrapState();
    completed = finalStatus.complete;
    await updateRuntimeStatus();
    log("Required PWA offline model pack is ready in browser storage.");
    if (showBootstrapProgress) setPipelineProgress("Dữ liệu offline đã sẵn sàng", 100);
  } finally {
    await releaseScreenWakeLockFor("bootstrap");
    offlineBootstrapBusy = false;
    await refreshOfflineBootstrapState();
    if (completed && showBootstrapProgress) {
      scheduleBootstrapReload();
    }
  }
}

async function autoDownloadAfterInstall(reason = "install") {
  if (offlineBootstrapReady || offlineBootstrapBusy || autoBootstrapAttempted) return;
  autoBootstrapAttempted = true;
  try {
    log(`Preparing offline models after ${reason}.`);
    await downloadRequiredOfflinePack();
  } catch (error) {
    autoBootstrapAttempted = false;
    let errorMsg = error.message;
    let detail = `${error.name || 'Error'}: ${error.message}\n${error.stack || ''}`;
    console.error("Offline bootstrap failed:", detail);
    if (errorMsg === "Load failed" || errorMsg === "Internal error" || errorMsg === "Failed to fetch") {
      errorMsg = `${errorMsg} (Hệ điều hành ngắt do mạng kém hoặc bộ nhớ đầy)`;
    }
    offlineBootstrapError = `Không tải được dữ liệu offline: ${errorMsg}. [Chi tiết: ${error.name}] Vui lòng giữ kết nối, để màn hình luôn sáng rồi mở lại ứng dụng để tải tiếp.`;
    await refreshOfflineBootstrapState();
    if (isStandaloneApp()) setPipelineProgress("Tải dữ liệu offline thất bại", 100);
    log(`Automatic offline model download failed: ${error.message}`);
  }
}

async function deletePack(packId) {
  const pack = manifest.packs.find((item) => item.id === packId);
  if (!pack) return;
  const dir = await opfsModelDir();
  for (const file of pack.files) {
    const name = modelFileName(file);
    clearModelIntegrity(file);
    await removeOpfsFile(dir, name);
    if (dir.kind === "idb") await deleteCachedModelFile(name).catch(() => null);
  }
  log(`Deleted local files for ${pack.name}`);
  await renderPacks();
  await refreshOfflineBootstrapState();
  await updateRuntimeStatus();
}

async function clearModels() {
  const dir = await opfsModelDir();
  if (dir.kind === "idb") {
    await idbDeleteScope(dir.scope).catch(() => null);
    await clearCachedModels().catch(() => null);
    clearAllModelIntegrityRecords();
    log("Deleted all local model files.");
    await renderPacks();
    await refreshOfflineBootstrapState();
    await updateRuntimeStatus();
    return;
  }
  for await (const name of dir.keys()) {
    await removeOpfsFile(dir, name);
  }
  clearAllModelIntegrityRecords();
  log("Deleted all local model files.");
  await renderPacks();
  await refreshOfflineBootstrapState();
  await updateRuntimeStatus();
}

function requiredModelFileNames() {
  const names = new Set();
  for (const pack of requiredOfflinePacks()) {
    for (const file of pack.files || []) {
      names.add(modelFileName(file));
    }
  }
  return names;
}

function clearObsoleteModelIntegrityRecords(allowedNames) {
  const prefix = "asr-vn-model-integrity:";
  try {
    for (let i = localStorage.length - 1; i >= 0; i -= 1) {
      const key = localStorage.key(i);
      if (!key?.startsWith(prefix)) continue;
      const name = key.slice(prefix.length);
      if (!allowedNames.has(name)) localStorage.removeItem(key);
    }
  } catch (_) {
    // ignore
  }
}

async function pruneUnusedModelFiles() {
  const allowedNames = requiredModelFileNames();
  if (!allowedNames.size) return;

  let removed = 0;
  const dir = await opfsModelDir();
  if (dir.kind === "idb") {
    const names = await idbListFileNames(dir.scope).catch(() => []);
    for (const name of names) {
      if (allowedNames.has(name)) continue;
      await idbDeleteFile(dir.scope, name).catch(() => null);
      removed += 1;
    }
  } else {
    for await (const name of dir.keys()) {
      if (allowedNames.has(name)) continue;
      await removeOpfsFile(dir, name);
      removed += 1;
    }
    const fallbackNames = await idbListFileNames("models").catch(() => []);
    for (const name of fallbackNames) {
      if (allowedNames.has(name)) continue;
      await idbDeleteFile("models", name).catch(() => null);
      removed += 1;
    }
  }

  if (window.caches) {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const keys = await cache.keys();
    for (const request of keys) {
      const url = new URL(request.url);
      if (!url.pathname.startsWith("/__asr_vn_model_cache__/")) continue;
      const name = decodeURIComponent(url.pathname.split("/").pop() || "");
      if (allowedNames.has(name)) continue;
      await cache.delete(request).catch(() => null);
      removed += 1;
    }
  }

  clearObsoleteModelIntegrityRecords(allowedNames);
  if (removed > 0) {
    log(`Deleted ${removed} unused model file(s) no longer required by the offline PWA.`);
  }
}

function setupInstallPrompt() {
  updateStandaloneUi();
  window.addEventListener("beforeinstallprompt", (event) => {
    // Let Chrome keep its native address-bar install affordance visible.
    // We still keep the event as a best-effort path for the in-app B2 button.
    installPrompt = event;
    updateStandaloneUi();
    updateInstallButtonState();
  });

  window.addEventListener("appinstalled", () => {
    installPrompt = null;
    // Lưu flag để khi app mở từ standalone lần đầu, tự động tải model
    markStandaloneInstalled();
    try { localStorage.setItem("pwa_just_installed", "1"); } catch(_) {}
    updateStandaloneUi();
    updateInstallButtonState();
  });

  window.matchMedia?.("(display-mode: standalone)")?.addEventListener?.("change", (e) => {
    if (e.matches) {
      markStandaloneInstalled();
      // Khi vừa chuyển sang standalone (ví dụ: cài PWA xong), trigger download ngay
      autoDownloadAfterInstall("display-mode change to standalone")
        .catch((err) => log(`Bootstrap on standalone switch failed: ${err.message}`));
    }
    updateStandaloneUi();
    updateInstallButtonState();
  });

  $("btn-install-cert")?.addEventListener("click", showCertGuide);

  $("btn-install-app")?.addEventListener("click", async () => {
    if (isIOS() && !installPrompt) {
      showIOSInstallGuide();
      return;
    }
    if (!installPrompt?.defaultPrevented) {
      showManualInstallGuide();
      return;
    }
    try {
      const result = await installPrompt.prompt();
      installPrompt = null;
      updateStandaloneUi();
      updateInstallButtonState();
      if (result?.outcome === "accepted") {
        // Đặt flag để khi app mở lần đầu tiên, sẽ tự động tải model ngay
        try { localStorage.setItem("pwa_just_installed", "1"); } catch(_) {}
        const message = $("install-status-msg");
        if (message) {
          message.style.display = "";
          message.textContent = "Đang mở ứng dụng... App sẽ tự động bắt đầu tải model offline ngay khi khởi động.";
        }
      }
    } catch (error) {
      installPrompt = null;
      updateInstallButtonState();
      showManualInstallGuide();
    }
  });
  updateInstallButtonState();
}

function setupEditorEvents() {
  $("btn-save-json")?.addEventListener("click", saveEditorAsrJson);
  $("btn-copy-text")?.addEventListener("click", copyEditorText);
  $("btn-export-transcript")?.addEventListener("click", exportEditorTranscript);
  $("btn-export-debug-log")?.addEventListener("click", exportDebugLog);
  $("btn-attach-audio")?.addEventListener("click", () => $("editor-audio-input")?.click());
  $("editor-audio-input")?.addEventListener("change", async () => {
    const input = $("editor-audio-input");
    await attachEditorAudioFile(input?.files?.[0]);
    if (input) input.value = "";
  });

  $("editor-audio")?.addEventListener("timeupdate", (event) => {
    setEditorActiveByTime(event.currentTarget.currentTime, !event.currentTarget.paused);
  });
  $("editor-audio")?.addEventListener("seeked", (event) => {
    setEditorActiveByTime(event.currentTarget.currentTime, true);
    appendDebugLog("player.seeked", {
      currentTime: debugRound(event.currentTarget.currentTime),
      activeSegmentIndex: editorState?.activeSegmentIndex ?? null,
      activeSegment: editorState?.activeSegmentIndex >= 0
        ? debugSegmentSnapshot(editorState.activeSegmentIndex)
        : null,
      audio: debugAudioSnapshot(),
    });
  });

  $("search-input")?.addEventListener("input", (event) => {
    if (!editorState) return;
    editorState.searchQuery = event.currentTarget.value || "";
    editorState.searchIndex = -1;
    renderEditor();
  });
  $("btn-search-prev")?.addEventListener("click", () => {
    if (!editorState?.searchMatches.length) return;
    editorState.searchIndex = editorState.searchIndex < 0
      ? editorState.searchMatches.length - 1
      : (editorState.searchIndex - 1 + editorState.searchMatches.length) % editorState.searchMatches.length;
    seekEditorSegment(editorState.searchMatches[editorState.searchIndex].segmentIndex);
    renderEditor();
  });
  $("btn-search-next")?.addEventListener("click", () => {
    if (!editorState?.searchMatches.length) return;
    editorState.searchIndex = editorState.searchIndex < 0
      ? 0
      : (editorState.searchIndex + 1) % editorState.searchMatches.length;
    seekEditorSegment(editorState.searchMatches[editorState.searchIndex].segmentIndex);
    renderEditor();
  });
  $("btn-search-clear")?.addEventListener("click", () => {
    if (!editorState) return;
    editorState.searchQuery = "";
    editorState.searchIndex = -1;
    const input = $("search-input");
    if (input) input.value = "";
    renderEditor();
  });
  $("btn-rerun-diarization")?.addEventListener("click", rerunEditorDiarization);

  document.querySelectorAll("[data-result-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.resultTab || "content";
      document.querySelectorAll("[data-result-tab]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      const content = $("result-content");
      const summary = $("result-summary");
      if (content) content.style.display = tab === "content" ? "" : "none";
      if (summary) summary.style.display = tab === "summary" ? "" : "none";
    });
  });

  $("result-content")?.addEventListener("click", (event) => {
    if (!editorState) return;
    const target = event.target instanceof Element ? event.target : event.target?.parentElement;
    const label = target?.closest(".speaker-label[data-spk]");
    if (label) {
      openSpeakerDialog({ mode: "rename", speaker: Number(label.dataset.spk) });
      return;
    }
    const segment = target?.closest("[data-editor-segment-index]");
    if (segment) {
      seekEditorSegment(Number(segment.dataset.editorSegmentIndex));
    }
  });

  document.querySelectorAll("[data-editor-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!editorState) return;
      editorState.activeTab = button.dataset.editorTab;
      renderEditor();
    });
  });

  ["editor-transcript-list", "editor-speaker-blocks"].forEach((id) => {
    $(id)?.addEventListener("click", (event) => {
      if (!editorState) return;
      const target = event.target instanceof Element ? event.target : event.target?.parentElement;
      const action = target?.closest("[data-editor-action]");
      if (action) {
        event.stopPropagation();
        const segmentIndex = Number(action.dataset.index);
        if (action.dataset.editorAction === "rename-speaker") {
          openSpeakerDialog({ mode: "rename", speaker: Number(action.dataset.speakerId) });
        } else if (action.dataset.editorAction === "split-block") {
          openSpeakerDialog({ mode: "split", segmentIndex });
        } else if (action.dataset.editorAction === "merge-prev") {
          mergeEditorSpeakerBlock(segmentIndex, "prev");
        } else if (action.dataset.editorAction === "merge-next") {
          mergeEditorSpeakerBlock(segmentIndex, "next");
        }
        return;
      }
      const row = target?.closest("[data-editor-segment-index]");
      if (row) seekEditorSegment(Number(row.dataset.editorSegmentIndex));
    });
  });

  $("editor-raw-speakers")?.addEventListener("click", (event) => {
    if (!editorState) return;
    const target = event.target instanceof Element ? event.target : event.target?.parentElement;
    const speakerAction = target?.closest("[data-editor-action='rename-speaker']");
    if (speakerAction) {
      event.stopPropagation();
      openSpeakerDialog({ mode: "rename", speaker: Number(speakerAction.dataset.speakerId) });
      return;
    }
    const row = target?.closest("[data-raw-index]");
    if (!row) return;
    const segment = editorState.rawSpeakerSegments[Number(row.dataset.rawIndex)];
    if (segment) seekEditorTo(segment.start, !$("editor-audio")?.paused);
  });

  $("speaker-dialog-cancel")?.addEventListener("click", () => {
    speakerDialogContext = null;
    closeSpeakerDialog();
  });
  $("speaker-dialog-apply")?.addEventListener("click", applySpeakerDialog);
}

async function doProcessSelectedAudioFile() {
  const processStartedAt = performance.now();
  if (!selectedAudioFile) return;
  if (selectedLibraryImportPromise) {
    await selectedLibraryImportPromise;
  }
  if (!offlineBootstrapReady) {
    const message = "Thiếu dữ liệu offline cần thiết. Mở app đã cài và chờ tải đủ model trước khi xử lý.";
    log("Required PWA offline model pack is required before processing.");
    setPipelineProgress(message, 100);
    showToast(message, "error");
    await refreshOfflineBootstrapState();
    return;
  }
  $("btn-process").disabled = true;
  const benchmarkButton = $("btn-benchmark");
  if (benchmarkButton) benchmarkButton.disabled = true;
  setPipelineControlsDisabled(true);
  if (selectedLibraryItemId) {
    await updateLibraryItem(selectedLibraryItemId, { status: "processing" }).catch(() => null);
  }
  try {
    await requestScreenWakeLockFor("processing");
    await runAudioImport(selectedAudioFile, { processStartedAt });
    hidePipelineProgress();
  } catch (error) {
    log(`Audio import failed: ${error.message}`);
    setPipelineProgress(`Lỗi: ${error.message}`, 100);
    showToast(error.message, "error");
    if (selectedLibraryItemId) {
      await updateLibraryItem(selectedLibraryItemId, { status: "source_ready" }).catch(() => null);
    }
  } finally {
    await releaseScreenWakeLockFor("processing");
    setPipelineControlsDisabled(false);
    syncPipelineControls();
    updateProcessButtonState();
  }
}

async function fetchCalibrationSampleFile() {
  let response = null;
  if (window.caches) {
    response = await caches.match(CALIBRATION_SAMPLE_URL).catch(() => null);
  }
  if (!response) {
    response = await fetchWithTimeout(
      CALIBRATION_SAMPLE_URL,
      { cache: "force-cache" },
      STARTUP_FETCH_TIMEOUT_MS
    );
  }
  if (!response.ok) {
    throw new Error(`Calibration sample not available: ${response.status}`);
  }
  const blob = await response.blob();
  return new File([blob], CALIBRATION_SAMPLE_NAME, {
    type: "audio/mpeg",
    lastModified: 1769833675210,
  });
}

function benchmarkReportTemplate(kind, mode, file, baseOptions) {
  return {
    schemaVersion: 1,
    kind,
    benchmarkMode: "per_stage_wasm_webgpu",
    mode,
    note: "Per-device calibration/benchmark runs the pipeline once. ASR stays on the full WASM path. Only selected WebGPU-sensitive stages are measured with WASM then WebGPU on the same intermediate input; WebGPU is selected only when output hash matches or an explicit tolerance/downstream check accepts it.",
    createdAt: new Date().toISOString(),
    file: {
      name: file.name,
      sizeBytes: file.size,
      type: file.type || "",
      lastModified: file.lastModified || null,
    },
    pipelineOptions: summarizePipelineOptionsForBenchmark(baseOptions),
    webgpuCapableSteps: [
      "CAM++ speaker embedding",
      "Pyannote Community-1 embedding encoder",
      "DNSMOS quality",
      "ViBERT punctuation fp32",
    ],
    webgpuNotApplicableSteps: [
      "Audio decode codec stage",
      "Silero VAD exact recurrent runner",
      "CAM++ long-form UMAP/HDBSCAN clustering path",
    ],
    wasmOnlySteps: [
      "Audio decode via FFmpeg WASM when direct WAV path is not available",
      "Silero VAD exact recurrent runner",
      "ASR full WASM backend",
      "CAM++ speech regions (pyannote segmentation)",
      "Pyannote Community-1 segmentation",
      "Speaker VBx core JavaScript",
      "CAM++ / Pyannote clustering JavaScript",
      "Zstandard result compression",
    ],
    environment: null,
    stages: [],
    runs: [],
    comparison: {},
    pipelineLog: [],
  };
}

async function runDeviceCalibrationIfNeeded(reason = "startup") {
  if (calibrationBusy || autoCalibrationAttempted || !offlineBootstrapReady || offlineBootstrapBusy) return;
  if (!isStandaloneApp()) return;
  if (selectedAudioFile || selectedLibraryImportPromise) return;
  autoCalibrationAttempted = true;

  const environment = await collectBenchmarkEnvironment().catch(() => null);
  const signature = await currentCalibrationSignature(environment).catch(() => null);
  const stored = readStoredCalibrationProfile();
  if (stored?.signature && signature && stored.signature === signature) {
    calibrationProfile = stored;
    log("Device calibration profile is current.");
    syncCalibrationSetupUi();
    return;
  }
  if (!signature) {
    log("Device calibration skipped: cannot build calibration signature.");
    syncCalibrationSetupUi();
    return;
  }

  calibrationBusy = true;
  syncCalibrationSetupUi("Đang chuẩn bị", 1);
  const baseOptions = getPipelineOptions();
  const benchmarkButton = $("btn-benchmark");
  const processButton = $("btn-process");
  if (benchmarkButton) benchmarkButton.disabled = true;
  if (processButton) processButton.disabled = true;
  setPipelineControlsDisabled(true);

  const started = performance.now();
  let report = null;
  try {
    await requestScreenWakeLockFor("calibration");
    resetPipelineLog();
    syncCalibrationSetupUi("Đang tính toán cấu hình tối ưu", 1);
    setPipelineProgress("Tối ưu thiết bị", 1);
    log(`[Calibration] Starting device calibration after ${reason}.`);
    log("[Calibration] User library data will not be touched.");
    await unloadModelsAfterStep("all", baseOptions).catch((error) => {
      log(`[Calibration] Pre-run unload failed: ${error.message}`);
    });

    const file = await fetchCalibrationSampleFile();
    report = benchmarkReportTemplate("offline_pwa_device_calibration", "auto-device-calibration", file, baseOptions);
    report.environment = environment || await collectBenchmarkEnvironment();
    setPipelineProgress("Tối ưu thiết bị", 3);
    const runStarted = performance.now();
    const result = await runAudioImport(file, {
      benchmarkLabel: "Device calibration: per-stage WASM/WebGPU",
      resetLog: false,
      clearEditor: false,
      saveLibraryResult: false,
      benchmarkOnly: true,
      benchmarkStages: report.stages,
    });
    const elapsedSeconds = (performance.now() - runStarted) / 1000;
    report.runs.push(summarizeBenchmarkResult("auto-calibration", result, elapsedSeconds));
    report.comparison = summarizeBenchmarkStageComparisons(report.stages);
    report.finishedAt = new Date().toISOString();
    report.totalSeconds = Number(((performance.now() - started) / 1000).toFixed(3));
    report.pipelineLog = pipelineLogLines.slice();
    if (calibrationSkipRequested) {
      log("[Calibration] Measured profile ignored because the user selected the default profile.");
      return;
    }

    const profile = deriveCalibrationProfile(report, report.environment, signature);
    saveCalibrationProfile(profile, report);
    syncCalibrationSetupUi("Đã tối ưu thiết bị", 100);
    setPipelineProgress("Đã tối ưu thiết bị", 100);
    log(
      `[Calibration] Saved provider profile: ASR=wasm, ` +
      `camppEmb=${profile.selectedProviders["CAM++ speaker embedding"] || "wasm"}, ` +
      `pyannoteEmb=${profile.selectedProviders["Pyannote Community-1 embedding encoder"] || "wasm"}, ` +
      `dnsmos=${profile.selectedProviders["DNSMOS quality"] || "wasm"}, ` +
      `punct=${profile.selectedProviders["ViBERT punctuation"] || profile.selectedProviders["ViBERT punctuation fp32"] || "wasm"}.`
    );
  } catch (error) {
    if (report) {
      report.error = { message: error.message || String(error), stack: error.stack || "" };
      report.finishedAt = new Date().toISOString();
      report.totalSeconds = Number(((performance.now() - started) / 1000).toFixed(3));
      report.pipelineLog = pipelineLogLines.slice();
      try {
        window.localStorage?.setItem(CALIBRATION_LAST_REPORT_KEY, JSON.stringify(report));
      } catch (_) {}
    }
    setPipelineProgress("Tối ưu thiết bị thất bại", 100);
    log(`[Calibration] Failed: ${error.message || String(error)}`);
    if (!calibrationSkipRequested && !calibrationProfile) {
      try {
        await saveDefaultCalibrationProfile("auto-calibration-failed");
        log("[Calibration] Saved default profile after calibration failure so the offline app can open.");
      } catch (fallbackError) {
        log(`[Calibration] Default fallback profile failed: ${fallbackError.message || String(fallbackError)}`);
      }
    }
  } finally {
    await releaseScreenWakeLockFor("calibration");
    calibrationBusy = false;
    syncCalibrationSetupUi();
    await unloadModelsAfterStep("all", baseOptions).catch((error) => {
      log(`[Calibration] Final unload failed: ${error.message}`);
    });
    setPipelineControlsDisabled(false);
    syncPipelineControls();
    updateProcessButtonState();
  }
}

async function rerunDeviceCalibrationFromButton() {
  if (calibrationBusy) return;
  const ok = window.confirm(
    "Chạy Re-Calibration sẽ dùng file mẫu 10 phút để đo lại WASM/WebGPU và cập nhật cấu hình tối ưu cho máy này. Dữ liệu tập tin/kết quả đã lưu sẽ không bị xóa. Tiếp tục?"
  );
  if (!ok) return;
  autoCalibrationAttempted = false;
  calibrationProfile = null;
  calibrationSkipRequested = false;
  try {
    window.localStorage?.removeItem(CALIBRATION_PROFILE_KEY);
  } catch (_) {}
  await refreshOfflineBootstrapState();
  await runDeviceCalibrationIfNeeded("manual re-calibration");
}

async function skipDeviceCalibrationFromButton() {
  const skipButton = $("btn-skip-calibration");
  const wasBusy = calibrationBusy;
  calibrationSkipRequested = true;
  if (wasBusy) {
    const ok = window.confirm(
      "Calibration đang chạy. Ứng dụng sẽ lưu cấu hình mặc định và mở lại để dừng bước đo hiện tại. Dữ liệu đã lưu không bị xóa. Tiếp tục?"
    );
    if (!ok) return;
  }
  if (skipButton) skipButton.disabled = true;
  try {
    await saveDefaultCalibrationProfile(wasBusy ? "skip-while-running" : "skip-before-run");
    autoCalibrationAttempted = true;
    calibrationBusy = false;
    setPipelineControlsDisabled(false);
    syncCalibrationSetupUi("Đã chọn cấu hình mặc định", 100);
    setProgress($("offline-bootstrap-progress-bar"), 100, 100);
    showToast("Đã dùng cấu hình mặc định cho thiết bị này.", "success");
    if (wasBusy) {
      window.setTimeout(() => window.location.reload(), 350);
      return;
    }
    await refreshOfflineBootstrapState();
    syncPipelineControls();
    updateProcessButtonState();
  } catch (error) {
    calibrationSkipRequested = false;
    showToast(`Không thể lưu cấu hình mặc định: ${error.message}`, "error");
    log(`[Calibration] Save default profile failed: ${error.message || String(error)}`);
  } finally {
    if (skipButton) skipButton.disabled = false;
  }
}

async function runBenchmarkSelectedAudioFile() {
  if (!selectedAudioFile) return;
  const file = selectedAudioFile;
  const filename = benchmarkLogFilename(file.name);
  let writeBenchmarkLog = null;
  try {
    writeBenchmarkLog = await createTextFileWriterFromUserGesture(
      filename,
      "application/json",
      "PWA benchmark log",
      ".json"
    );
  } catch (error) {
    if (error?.name !== "AbortError") {
      log(`[Benchmark] Cannot prepare log file: ${error.message}`);
      showToast(`Benchmark log file error: ${error.message}`, "error");
    }
    return;
  }

  if (selectedLibraryImportPromise) {
    await selectedLibraryImportPromise;
  }
  if (!offlineBootstrapReady) {
    const message = "Thiếu dữ liệu offline cần thiết. Mở app đã cài và chờ tải đủ model trước khi benchmark.";
    log("Required PWA offline model pack is required before benchmark.");
    setPipelineProgress(message, 100);
    showToast(message, "error");
    await refreshOfflineBootstrapState();
    return;
  }

  const benchmarkButton = $("btn-benchmark");
  const processButton = $("btn-process");
  if (benchmarkButton) benchmarkButton.disabled = true;
  if (processButton) processButton.disabled = true;
  setPipelineControlsDisabled(true);

  const baseOptions = getPipelineOptions();
  const report = {
    schemaVersion: 1,
    kind: "offline_pwa_pipeline_benchmark",
    benchmarkMode: "per_stage_wasm_webgpu",
    note: "Benchmark runs the pipeline once. ASR stays on the full WASM path. Only CAM++ speaker embedding, Pyannote Community-1 embedding encoder, DNSMOS quality, and ViBERT punctuation are measured with WASM then WebGPU. WebGPU is selected when output hash matches and it is faster, or when an explicit numeric-tolerance/downstream segment check accepts the difference.",
    selectionPolicy: {
      exact: "Select WebGPU when output hash equals WASM/JS baseline and WebGPU is faster.",
      tolerance: "For embedding stages such as CAM++ and Pyannote Community-1, raw float hash may differ by tiny GPU arithmetic noise; select WebGPU only if numeric diff/cosine thresholds pass and downstream speaker segments match or stay within tolerance.",
      reject: "If hash differs and tolerance/downstream checks do not pass, benchmark keeps WASM/JS even when WebGPU is faster.",
    },
    createdAt: new Date().toISOString(),
    file: {
      name: file.name,
      sizeBytes: file.size,
      type: file.type || "",
      lastModified: file.lastModified || null,
    },
    pipelineOptions: summarizePipelineOptionsForBenchmark(baseOptions),
    webgpuCapableSteps: [
      "CAM++ speaker embedding",
      "Pyannote Community-1 embedding encoder",
      "DNSMOS quality",
      "ViBERT punctuation fp32",
    ],
    webgpuNotApplicableSteps: [
      "Audio decode codec stage",
      "Silero VAD exact recurrent runner",
      "CAM++ long-form UMAP/HDBSCAN clustering path",
    ],
    wasmOnlySteps: [
      "Audio decode via FFmpeg WASM when direct WAV path is not available",
      "Silero VAD exact recurrent runner",
      "ASR full WASM backend",
      "CAM++ speech regions (pyannote segmentation)",
      "Pyannote Community-1 segmentation",
      "Speaker VBx core JavaScript",
      "CAM++ / Pyannote clustering JavaScript",
      "Zstandard result compression",
    ],
    environment: null,
    stages: [],
    runs: [],
    comparison: {},
    pipelineLog: [],
  };
  let failure = null;
  const started = performance.now();

  try {
    await requestScreenWakeLockFor("benchmark");
    resetPipelineLog();
    setPipelineProgress("Benchmark setup", 1);
    log("[Benchmark] Starting offline PWA benchmark.");
    log("[Benchmark] Per-stage mode: WASM-only stages run once; WebGPU-capable stages run WASM then WebGPU on the same intermediate input.");
    log("[Benchmark] Result will not be opened in the editor or saved to the local library.");
    log("[Benchmark] Browser does not expose physical CPU core count or GPU VRAM; report records available browser-safe hardware data.");
    report.environment = await collectBenchmarkEnvironment();
    log(
      `[Benchmark] WASM: requestedThreads=${report.environment.wasm.requestedThreads}, ` +
      `logicalThreads=${report.environment.wasm.logicalThreads}, maxThreads=${report.environment.wasm.maxThreads}.`
    );
    log(
      `[Benchmark] WebGPU: supported=${report.environment.webgpu.supported}, ` +
      `adapter=${report.environment.webgpu.adapterInfo?.description || report.environment.webgpu.adapterInfo?.device || "n/a"}.`
    );

    await unloadModelsAfterStep("all", baseOptions).catch((error) => {
      log(`[Benchmark] Pre-run unload failed: ${error.message}`);
    });
    const label = "Benchmark run: per-stage WASM/WebGPU";
    log(`[Benchmark] ${label}.`);
    setPipelineProgress("Benchmark", 3);
    const runStarted = performance.now();
    const result = await runAudioImport(file, {
      benchmarkLabel: label,
      resetLog: false,
      clearEditor: false,
      saveLibraryResult: false,
      benchmarkOnly: true,
      benchmarkStages: report.stages,
      useCalibratedWebGpuBenchmark: true,
    });
    const elapsedSeconds = (performance.now() - runStarted) / 1000;
    const summary = summarizeBenchmarkResult("per-stage", result, elapsedSeconds);
    report.runs.push(summary);
    report.comparison = summarizeBenchmarkStageComparisons(report.stages);
    log(
      `[Benchmark] Completed in ${summary.elapsedSeconds.toFixed(2)}s; ` +
      `punct=${summary.timings.punctuationSeconds.toFixed(2)}s (${summary.providers.punctuation}), ` +
      `diar=${summary.timings.diarizationSeconds.toFixed(2)}s, ` +
      `speakerEmbedding=${summary.providers.speakerEmbedding}.`
    );
    log(
      `[Benchmark] Stage comparison: ${report.comparison.dualStageCount} dual stage(s), ` +
      `${report.comparison.webgpuFasterCount} WebGPU faster, ` +
      `${report.comparison.webgpuAcceptedCount} WebGPU selected, ` +
      `${report.comparison.toleratedMismatchCount} tolerated mismatch, ` +
      `${report.comparison.rejectedMismatchCount} rejected mismatch.`
    );

    setPipelineProgress("Benchmark complete", 100);
    log(`[Benchmark] Finished in ${((performance.now() - started) / 1000).toFixed(2)}s.`);
  } catch (error) {
    failure = error;
    report.error = {
      message: error.message || String(error),
      stack: error.stack || "",
    };
    setPipelineProgress("Benchmark failed", 100);
    log(`[Benchmark] Failed: ${error.message || String(error)}`);
  } finally {
    await releaseScreenWakeLockFor("benchmark");
    report.finishedAt = new Date().toISOString();
    report.totalSeconds = Number(((performance.now() - started) / 1000).toFixed(3));
    report.pipelineLog = pipelineLogLines.slice();
    try {
      await writeBenchmarkLog(JSON.stringify(report, null, 2));
      log(`[Benchmark] Exported log: ${filename}`);
      showToast(failure ? "Benchmark failed; partial log exported." : "Benchmark log exported.", failure ? "error" : "info");
    } catch (error) {
      if (error?.name !== "AbortError") {
        log(`[Benchmark] Export log failed: ${error.message}`);
        showToast(`Benchmark log export failed: ${error.message}`, "error");
      }
    }
    await unloadModelsAfterStep("all", baseOptions).catch((error) => {
      log(`[Benchmark] Final unload failed: ${error.message}`);
    });
    setPipelineControlsDisabled(false);
    syncPipelineControls();
    updateProcessButtonState();
  }
}

function updateSelectedFileUi(file) {
  const dropText = document.querySelector(".drop-zone-text");
  const selected = $("file-selected");
  const fileName = $("file-name");
  const fileSize = $("file-size");
  if (file) {
    if (dropText) dropText.style.display = "none";
    if (selected) selected.style.display = "flex";
    if (fileName) fileName.textContent = file.name || "";
    if (fileSize) fileSize.textContent = formatBytes(file.size || 0);
  } else {
    if (dropText) dropText.style.display = "";
    if (selected) selected.style.display = "none";
    if (fileName) fileName.textContent = "";
    if (fileSize) fileSize.textContent = "";
  }
}

function clearSelectedFile() {
  selectedAudioFile = null;
  selectedLibraryItemId = null;
  selectedLibraryImportPromise = null;
  clearEditorResult();
  hidePipelineProgress();
  resetPipelineLog();
  const input = $("file-input");
  if (input) input.value = "";
  updateSelectedFileUi(null);
  updateProcessButtonState();
}

async function handleSelectedInputFile(file) {
  clearEditorResult();
  hidePipelineProgress();
  resetPipelineLog();
  if (file && /\.json$/i.test(file.name || "")) {
    selectedAudioFile = null;
    selectedLibraryItemId = null;
    selectedLibraryImportPromise = null;
    updateSelectedFileUi(null);
    const input = $("file-input");
    if (input) input.value = "";
    updateProcessButtonState();
    showToast("PWA offline không còn hỗ trợ Upload JSON. Vui lòng chọn file âm thanh và xử lý lại.", "error");
    return;
  }

  selectedAudioFile = file || null;
  selectedLibraryItemId = null;
  selectedLibraryImportPromise = null;
  updateSelectedFileUi(selectedAudioFile);
  updateProcessButtonState();
  if (!file) return;
  revealProcessingPanel();
  showSelectedAudioPreview(file);

  selectedLibraryImportPromise = createLibraryItemFromSourceFile(file)
    .then((item) => {
      selectedLibraryItemId = item.id;
      return item;
    })
    .catch((error) => {
      log(`Store source in offline library failed: ${error.message}`);
      selectedAudioFile = null;
      selectedLibraryItemId = null;
      clearEditorResult();
      updateSelectedFileUi(null);
      updateProcessButtonState();
      throw error;
    })
    .finally(() => {
      selectedLibraryImportPromise = null;
      updateProcessButtonState();
    });
  await selectedLibraryImportPromise.catch(() => null);
}

function setupEvents() {
  syncPipelineControls();
  setupScreenWakeLockResume();
  setupEditorEvents();
  window.addEventListener("online", () => {
    updateStandaloneUi();
    updateRuntimeStatus().catch((error) => log(`Runtime status update failed: ${error.message || String(error)}`));
  });
  window.addEventListener("offline", () => {
    updateStandaloneUi();
    updateRuntimeStatus().catch((error) => log(`Runtime status update failed: ${error.message || String(error)}`));
  });
  ["cpu-threads", "punctuation-level", "case-level", "hotwords-score"].forEach((id) => {
    const node = $(id);
    if (node) {
      node.addEventListener("input", () => {
        syncPipelineControls();
        updateHotwordsSummary();
        scheduleUserConfigSave();
      });
    }
  });
  $("ui-text-scale")?.addEventListener("input", () => {
    applyUiTextScale($("ui-text-scale")?.value);
    scheduleUserConfigSave();
  });
  $("cpu-threads")?.addEventListener("input", refreshOrtThreadConfigIfIdle);
  [
    "speaker-diarization",
    "speaker-count",
    "speaker-model",
    "rms-normalize",
    "bypass-vad",
    "overlap-separation",
    "save-ram",
    "hotwords-enabled",
  ].forEach((id) => {
    const node = $(id);
    if (!node) return;
    node.addEventListener("change", () => {
      syncPipelineControls();
      updateHotwordsSummary();
      saveUserConfig();
    });
  });
  $("asr-model")?.addEventListener("change", () => {
    resetAsrWorker("model changed");
    saveUserConfig();
  });
  $("hotwords-enabled")?.addEventListener("change", () => resetAsrWorker("hotwords changed"));
  $("hotwords-score")?.addEventListener("input", () => resetAsrWorker("hotwords changed"));
  $("btn-hotword-manager")?.addEventListener("click", openHotwordDialog);
  $("btn-close-hotword-dialog")?.addEventListener("click", closeHotwordDialog);
  $("hotwords-search")?.addEventListener("input", updateHotwordFilter);
  $("btn-add-hotword")?.addEventListener("click", addHotwordFromControls);
  $("hotword-new-text")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addHotwordFromControls();
    }
  });
  $("btn-export-hotword-txt")?.addEventListener("click", exportHotwordTxt);
  $("btn-import-hotword-txt")?.addEventListener("click", () => $("hotword-file-input")?.click());
  $("hotword-file-input")?.addEventListener("change", async () => {
    const input = $("hotword-file-input");
    await importHotwordTxt(input?.files?.[0]);
    if (input) input.value = "";
  });
  $("btn-reset-hotword-defaults")?.addEventListener("click", async () => {
    if (!window.confirm("Khôi phục hotword.txt mặc định?")) return;
    await resetHotwordsToDefaults();
  });

  $("btn-theme")?.addEventListener("click", () => {
    applyTheme(document.body.dataset.theme === "light" ? "dark" : "light");
    saveUserConfig();
  });
  $("btn-about")?.addEventListener("click", () => window.showAboutDialog?.());
  $("btn-recalibration")?.addEventListener("click", rerunDeviceCalibrationFromButton);
  $("btn-skip-calibration")?.addEventListener("click", skipDeviceCalibrationFromButton);
  $("btn-offline-files")?.addEventListener("click", toggleMeetingsPanel);
  $("btn-close-meetings")?.addEventListener("click", closeMeetingsPanel);
  $("meetings-search")?.addEventListener("input", searchMeetings);
  $("mg-select-all")?.addEventListener("change", toggleSelectAll);
  $("btn-delete-selected-meetings")?.addEventListener("click", () => {
    deleteSelectedMeetings().catch((error) => showToast(`Lỗi xóa: ${error.message}`, "error"));
  });
  $("btn-export-config")?.addEventListener("click", exportUserConfig);
  $("btn-import-config")?.addEventListener("click", () => $("config-input")?.click());
  $("config-input")?.addEventListener("change", async () => {
    const file = $("config-input")?.files?.[0];
    if (!file) return;
    await importUserConfig(file);
    $("config-input").value = "";
  });
  $("btn-reset-config")?.addEventListener("click", async () => {
    if (!window.confirm("Reset PWA config to defaults?")) return;
    const defaults = normalizeUserConfig(DEFAULT_USER_CONFIG);
    const defaultHotwords = await loadDefaultHotwords();
    if (defaultHotwords.length) defaults.hotwords = defaultHotwords;
    applyUserConfig(defaults);
    saveUserConfig();
    resetAsrWorker("config reset");
  });

  $("btn-download-required")?.addEventListener("click", async () => {
    try {
      await downloadRequiredOfflinePack();
    } catch (error) {
      log(`Required PWA offline model pack download failed: ${error.message}`);
    }
  });

  $("btn-refresh-library")?.addEventListener("click", () => {
    renderLibrary().catch((error) => log(`Refresh library failed: ${error.message}`));
  });

  const dropZone = $("drop-zone");
  const fileInput = $("file-input");
  fileInput?.addEventListener("change", async () => {
    await handleSelectedInputFile(fileInput.files?.[0] || null);
  });
  dropZone?.addEventListener("click", (event) => {
    if (event.target.closest?.("#btn-clear-file")) return;
    fileInput?.click();
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone?.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone?.addEventListener(eventName, () => {
      dropZone.classList.remove("drag-over");
    });
  });
  dropZone?.addEventListener("drop", async (event) => {
    event.preventDefault();
    await handleSelectedInputFile(event.dataTransfer?.files?.[0] || null);
  });
  $("btn-clear-file")?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    clearSelectedFile();
  });

  $("btn-process").addEventListener("click", processSelectedAudioFile);
  $("btn-benchmark")?.addEventListener("click", runBenchmarkSelectedAudioFile);
}

function downmixToMono(audioBuffer) {
  const length = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(length);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      mono[i] += data[i] / channels;
    }
  }
  return mono;
}

function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const src = i * ratio;
    const left = Math.floor(src);
    const right = Math.min(left + 1, input.length - 1);
    const frac = src - left;
    output[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return output;
}

function analyzeAudio(samples, sampleRate) {
  let peak = 0;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = samples[i];
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / Math.max(1, samples.length));

  const frameSize = Math.max(1, Math.round(sampleRate * 0.03));
  const speechThreshold = Math.max(0.004, rms * 0.6);
  let speechFrames = 0;
  for (let offset = 0; offset < samples.length; offset += frameSize) {
    let frameSq = 0;
    const end = Math.min(samples.length, offset + frameSize);
    for (let i = offset; i < end; i += 1) frameSq += samples[i] * samples[i];
    const frameRms = Math.sqrt(frameSq / Math.max(1, end - offset));
    if (frameRms >= speechThreshold) speechFrames += 1;
  }
  const totalFrames = Math.ceil(samples.length / frameSize);
  const speechRatio = totalFrames ? speechFrames / totalFrames : 0;
  return { peak, rms, speechRatio };
}

function probabilityStats(probabilities) {
  if (!probabilities.length) return { max: 0, avg: 0 };
  let max = 0;
  let sum = 0;
  for (const value of probabilities) {
    if (value > max) max = value;
    sum += value;
  }
  return { max, avg: sum / probabilities.length };
}

function boostAudioForVad(samples) {
  const boostTarget = 0.071;
  let maxAmp = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const abs = Math.abs(samples[i]);
    if (abs > maxAmp) maxAmp = abs;
  }

  if (maxAmp <= 1e-6 || maxAmp >= boostTarget) {
    return { audio: samples, boosted: false, peak: maxAmp, scale: 1 };
  }

  const scale = boostTarget / maxAmp;
  const audio = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    audio[i] = samples[i] * scale;
  }
  return { audio, boosted: true, peak: maxAmp, scale };
}

function computeRms(samples, start = 0, end = samples.length) {
  const clippedStart = Math.max(0, Math.min(samples.length, start));
  const clippedEnd = Math.max(clippedStart, Math.min(samples.length, end));
  if (clippedEnd <= clippedStart) return 0;
  let sumSq = 0;
  for (let i = clippedStart; i < clippedEnd; i += 1) {
    sumSq += samples[i] * samples[i];
  }
  return Math.sqrt(sumSq / Math.max(1, clippedEnd - clippedStart));
}

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function adaptivePeakLimit(samples, targetPeak = 0.95) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) peak = abs;
  }
  if (peak <= targetPeak) return samples;

  const scale = targetPeak / peak;
  const output = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    output[i] = samples[i] * scale;
  }
  log(`Peak limited: ${peak.toFixed(4)} -> ${targetPeak.toFixed(2)}.`);
  return output;
}

function perSegmentRmsNormalize(samples, segments, sampleRate = VAD_SAMPLE_RATE) {
  if (!segments.length) return { audio: samples, changed: false, targetRms: 0, segmentCount: 0 };

  const minSamples = Math.floor((100 * sampleRate) / 1000);
  const maxGain = 10;
  const crossfadeSamples = Math.floor((5 * sampleRate) / 1000);
  const segmentStats = [];

  for (const segment of segments) {
    const start = Math.max(0, Math.floor(segment.start));
    const end = Math.min(samples.length, Math.ceil(segment.end));
    if (end - start < minSamples) continue;
    const rms = computeRms(samples, start, end);
    if (rms > 1e-8) segmentStats.push({ start, end, rms });
  }

  if (!segmentStats.length) return { audio: samples, changed: false, targetRms: 0, segmentCount: 0 };
  const targetRms = median(segmentStats.map((item) => item.rms));
  if (targetRms < 1e-8) return { audio: samples, changed: false, targetRms, segmentCount: segmentStats.length };

  const gainMap = new Float32Array(samples.length);
  gainMap.fill(1);
  const gainsDb = [];
  for (const item of segmentStats) {
    let gain = targetRms / item.rms;
    gain = Math.max(1 / maxGain, Math.min(maxGain, gain));
    gainsDb.push(20 * Math.log10(gain));
    gainMap.fill(gain, item.start, item.end);
  }

  if (crossfadeSamples > 0) {
    for (const item of segmentStats) {
      const fadeLen = Math.min(crossfadeSamples, Math.floor((item.end - item.start) / 4));
      if (fadeLen <= 0) continue;
      if (item.start > 0) {
        const previousGain = gainMap[Math.max(0, item.start - 1)];
        const segmentGain = gainMap[item.start];
        for (let i = 0; i < fadeLen; i += 1) {
          gainMap[item.start + i] = previousGain + ((segmentGain - previousGain) * i) / Math.max(1, fadeLen - 1);
        }
      }
      if (item.end < samples.length) {
        const segmentGain = gainMap[item.end - 1];
        const nextGain = gainMap[item.end];
        for (let i = 0; i < fadeLen; i += 1) {
          const pos = item.end - fadeLen + i;
          gainMap[pos] = segmentGain + ((nextGain - segmentGain) * i) / Math.max(1, fadeLen - 1);
        }
      }
    }
  }

  const output = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    output[i] = samples[i] * gainMap[i];
  }
  const limited = adaptivePeakLimit(output);
  const minGain = Math.min(...gainsDb);
  const maxGainDb = Math.max(...gainsDb);
  log(`RMS normalized ${segmentStats.length} segment(s), target=${targetRms.toFixed(6)}, gain=${minGain.toFixed(1)}..${maxGainDb.toFixed(1)} dB.`);
  return { audio: limited, changed: true, targetRms, segmentCount: segmentStats.length };
}

async function getVadSession() {
  configureOrt();
  if (vadSession) return vadSession;

  log("Loading Silero VAD ONNX session.");
  const model = await loadModelArrayBuffer("vad.silero");
  const created = await createOrtSession(model, {
    name: "Silero VAD",
    webgpuPreferred: false,
  });
  vadSession = created.session;
  vadExecutionProvider = created.provider;
  log(`Silero VAD session ready (${vadExecutionProvider}).`);
  return vadSession;
}

async function runVadInference(audio, options) {
  const session = await getVadSession();
  const sampleRate = options.sampleRate || VAD_SAMPLE_RATE;
  const threshold = options.threshold ?? 0.2;
  const minSilenceMs = options.minSilenceMs ?? 100;
  const minSpeechMs = options.minSpeechMs ?? 250;
  const progress = options.progress;
  const numWindows = Math.floor(audio.length / VAD_WINDOW_SIZE);

  if (numWindows === 0) {
    return { segments: [], probabilities: [] };
  }

  // Buffer reuse: vadInputBuf và vadStateBuf tái dùng qua các file (không alloc mới mỗi window)
  const VAD_INPUT_SIZE = VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE;
  if (!vadInputBuf || vadInputBuf.length < VAD_INPUT_SIZE) vadInputBuf = new Float32Array(VAD_INPUT_SIZE);
  if (!vadStateBuf || vadStateBuf.length < 256) vadStateBuf = new Float32Array(2 * 1 * 128);
  else vadStateBuf.fill(0);
  let context = new Float32Array(VAD_CONTEXT_SIZE);
  const sampleRateTensor = new window.ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(sampleRate)]),
    []
  );
  const probabilities = [];
  let lastReportedPct = -1;

  for (let i = 0; i < numWindows; i += 1) {
    const start = i * VAD_WINDOW_SIZE;
    const chunk = audio.subarray(start, start + VAD_WINDOW_SIZE);
    vadInputBuf.set(context, 0);
    vadInputBuf.set(chunk, VAD_CONTEXT_SIZE);

    const outputs = await session.run({
      input: new window.ort.Tensor("float32", vadInputBuf, [1, VAD_INPUT_SIZE]),
      state: new window.ort.Tensor("float32", vadStateBuf, [2, 1, 128]),
      sr: sampleRateTensor,
    });
    const probTensor = outputs.output || outputs[session.outputNames[0]];
    const stateTensor = outputs.stateN || outputs[session.outputNames[1]];
    probabilities.push(Number(probTensor.data[0]));
    // Tái dùng vadStateBuf: copy vào buffer sẵn có thay vì new Float32Array
    vadStateBuf.set(stateTensor.data);
    context = chunk.slice(VAD_WINDOW_SIZE - VAD_CONTEXT_SIZE);

    if (progress && numWindows > 100) {
      const pct = Math.floor(((i + 1) * 100) / numWindows);
      if (pct >= lastReportedPct + 2) {
        lastReportedPct = pct;
        progress(pct);
      }
    }
  }

  const minSilenceWindows = Math.floor((minSilenceMs * sampleRate) / 1000 / VAD_WINDOW_SIZE);
  const minSpeechWindows = Math.floor((minSpeechMs * sampleRate) / 1000 / VAD_WINDOW_SIZE);
  const segments = [];
  let isSpeech = false;
  let speechStart = 0;
  let silenceCount = 0;

  for (let i = 0; i < probabilities.length; i += 1) {
    const prob = probabilities[i];
    if (prob >= threshold) {
      if (!isSpeech) {
        speechStart = i;
        isSpeech = true;
      }
      silenceCount = 0;
    } else if (isSpeech) {
      silenceCount += 1;
      if (silenceCount >= minSilenceWindows) {
        const speechEnd = i - silenceCount + 1;
        if (speechEnd - speechStart >= minSpeechWindows) {
          segments.push([speechStart, speechEnd]);
        }
        isSpeech = false;
        silenceCount = 0;
      }
    }
  }

  if (isSpeech) {
    const speechEnd = probabilities.length;
    if (speechEnd - speechStart >= minSpeechWindows) {
      segments.push([speechStart, speechEnd]);
    }
  }

  return { segments, probabilities };
}

async function getVadSegments(samples, options = {}) {
  const sampleRate = options.sampleRate || VAD_SAMPLE_RATE;
  const paddingMs = options.paddingMs ?? 1000;
  const mergeGapMs = options.mergeGapMs ?? 250;
  const progress = options.progress;
  const totalSamples = samples.length;

  if (totalSamples < VAD_WINDOW_SIZE) {
    return {
      segments: [],
      probabilities: [],
      boosted: false,
      peak: 0,
      scale: 1,
    };
  }

  const boosted = boostAudioForVad(samples);
  if (boosted.boosted) {
    log(`VAD boosted low peak ${boosted.peak.toFixed(4)} by ${boosted.scale.toFixed(1)}x.`);
  }

  let pass = await runVadInference(boosted.audio, {
    sampleRate,
    threshold: options.threshold ?? 0.2,
    minSilenceMs: options.minSilenceMs ?? 100,
    minSpeechMs: options.minSpeechMs ?? 250,
    progress,
  });

  if (!pass.segments.length) {
    throw new Error("VAD found no speech segments.");
  }

  const paddingSamples = Math.floor((paddingMs * sampleRate) / 1000);
  let segments = pass.segments.map(([startWindow, endWindow]) => ({
    start: Math.max(0, startWindow * VAD_WINDOW_SIZE - paddingSamples),
    end: Math.min(totalSamples, endWindow * VAD_WINDOW_SIZE + paddingSamples),
  }));

  if (mergeGapMs > 0 && segments.length > 1) {
    const mergeGap = Math.floor((mergeGapMs * sampleRate) / 1000);
    const merged = [segments[0]];
    for (const segment of segments.slice(1)) {
      const previous = merged[merged.length - 1];
      if (segment.start - previous.end < mergeGap) {
        previous.end = segment.end;
      } else {
        merged.push(segment);
      }
    }
    segments = merged;
  }

  return { segments, probabilities: pass.probabilities, ...boosted };
}

function renderAudioSummary(items) {
  const root = $("audio-summary");
  if (!root) return;
  root.style.display = "grid";
  root.textContent = "";
  for (const item of items) {
    const node = document.createElement("div");
    node.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong>`;
    root.appendChild(node);
  }
}

function buildPunctuationPauseHints(words = [], diarizationSegments = []) {
  if (!Array.isArray(words) || words.length < 2) return null;
  const hints = [];
  for (let i = 0; i < words.length; i += 1) {
    const current = words[i];
    const next = words[i + 1];
    const currentInterval = effectiveWordInterval(current);
    let gap = next
      ? Math.max(0, Number(next.start || 0) - currentInterval.end)
      : 1.0;
    if (next && Array.isArray(diarizationSegments) && diarizationSegments.length) {
      const currentSpeaker = speakerForWordByTime(current, diarizationSegments);
      const nextSpeaker = speakerForWordByTime(next, diarizationSegments, currentSpeaker);
      if (currentSpeaker !== nextSpeaker) gap = Math.max(gap, 1.0);
    }
    hints.push(gap);
  }
  return hints.length === words.length ? hints : null;
}

async function runAudioImport(file, options = {}) {
  const previousBenchmarkProviderMode = benchmarkProviderMode;
  const previousResumeContext = activeResumeAfterKillContext;
  benchmarkProviderMode = options.benchmarkProviderMode || null;
  try {
    return await runAudioImportInternal(file, options);
  } finally {
    benchmarkProviderMode = previousBenchmarkProviderMode;
    activeResumeAfterKillContext = previousResumeContext;
  }
}

async function runAudioImportInternal(file, options = {}) {
  const totalStarted = Number.isFinite(options.processStartedAt) ? options.processStartedAt : performance.now();
  const bootstrap = await refreshOfflineBootstrapState();
  if (!bootstrap.complete) {
    throw new Error("Required PWA offline model pack is not ready in this browser.");
  }
  if (options.resetLog !== false) resetPipelineLog();
  if (options.clearEditor !== false) clearEditorResult();
  const pipelineOptions = getPipelineOptions();
  const resumeContext = await createResumeAfterKillContext(file, pipelineOptions, options);
  activeResumeAfterKillContext = resumeContext;
  const resumeCheckpoints = await loadResumeAfterKillCheckpoints(resumeContext);
  setPipelineProgress("Reading input", 2);
  if (options.benchmarkLabel) log(`[Benchmark] ${options.benchmarkLabel}`);
  if (benchmarkProviderMode) {
    log(`[Benchmark] WebGPU-capable ONNX sessions mode: ${benchmarkProviderMode}.`);
  }
  log(`Reading ${file.name}`);
  log(
    `Pipeline config: ASR=${pipelineOptions.asrModel.label}, ` +
    `punctuation=${pipelineOptions.bypassPunctuation ? "off" : `level ${pipelineOptions.punctuationLevel}`}, ` +
    `diarization=${pipelineOptions.speakerDiarization ? "on" : "off"}, ` +
    `overlap=${pipelineOptions.overlapSeparation ? "on" : "off"}, ` +
    `hotwords=${pipelineOptions.hotwordCount}, ` +
    `threads=${pipelineOptions.cpuThreads}/${logicalThreadCount()} logical, ` +
    `diarBatch=${getDiarBatchSize()}, ` +
    `saveRam=${pipelineOptions.saveRam ? "on" : "off"}.`
  );
  try {
    let decoded = null;
    let canonicalAudio = null;
    let stats = null;
    let duration = 0;
    let vad = null;
    let vadElapsed = 0;
    const arrayBuffer = await file.arrayBuffer();
    setPipelineProgress("Decoding input", 5);
    const decodeStarted = performance.now();
    decoded = await decodeAudioFileWithFfmpeg(file, arrayBuffer, {
      progress: (ratio) => setPipelineProgress("FFmpeg decode", 5 + ratio * 5),
    });
    addBenchmarkStage(options, {
      name: "Audio decode",
      capability: "wasm-only",
      attempts: [{
        runtime: "wasm",
        provider: decoded.decoder || "wasm",
        elapsedSeconds: benchmarkSeconds(decodeStarted),
        summary: {
          decoder: decoded.decoder,
          originalSampleRate: decoded.originalSampleRate || null,
          outputSampleRate: decoded.sampleRate,
          channels: decoded.channels || null,
          outputSamples: decoded.samples?.length || 0,
          pcmHash: hashFloatValues(decoded.samples || [], 1e6, Number.POSITIVE_INFINITY),
        },
      }],
      selectedRuntime: "wasm",
      selectedProvider: decoded.decoder || "wasm",
      webgpuNotApplicable: {
        reason: "Browser WebGPU is a compute/render API, not an audio codec decode API. MP3/M4A decode must use FFmpeg WASM, WebCodecs, or browser media codecs before PCM can be processed by WebGPU.",
      },
    });
    log(
      `Decoded with ${decoded.decoder}: ` +
      `${decoded.channels || "?"} channel(s), ` +
      `${decoded.originalSampleRate || "?"} Hz -> ${decoded.sampleRate} Hz mono.`
    );

    canonicalAudio = decoded.samples;
    setPipelineProgress("Analyzing audio", 10);
    stats = analyzeAudio(canonicalAudio, 16000);
    duration = canonicalAudio.length / 16000;
    vad = decodeVadCheckpoint(resumeCheckpoints.vad);
    if (vad?.segments?.length) {
      vadElapsed = Number(resumeCheckpoints.vad?.elapsed || 0);
      setPipelineProgress("VAD resumed", 30);
      log(`[resume_after_kill] Resumed VAD checkpoint: ${vad.segments.length} segment(s).`);
    } else {
      let lastVadLog = 0;
      const vadStart = performance.now();
      const runVadStage = () => {
        lastVadLog = 0;
        return getVadSegments(canonicalAudio, {
            sampleRate: 16000,
            progress: (pct) => {
              setPipelineProgress("Silero VAD", 10 + pct * 0.2);
              if (pct >= lastVadLog + 10 || pct === 100) {
                lastVadLog = pct;
                log(`VAD progress ${pct}%.`);
              }
            },
        });
      };
      vad = pipelineOptions.bypassVad
        ? {
            segments: [{ start: 0, end: canonicalAudio.length }],
            probabilities: [],
            boosted: false,
            peak: 0,
            scale: 1,
            bypassed: true,
          }
        : (Array.isArray(options.benchmarkStages)
            ? await runBenchmarkWasmOnlyStage(
                options,
                "Silero VAD",
                runVadStage,
                summarizeVadForBenchmark,
                {
                  webgpuNotApplicable: {
                    reason: "Silero VAD is a tiny stateful sequential recurrent model; exact WebGPU would dispatch thousands of small dependent runs and is slower than WASM. Batched WebGPU would require changing state semantics and would not be exact.",
                  },
                }
              )
            : await runVadStage());
      const selectedVadStage = options.benchmarkStages?.find((stage) => stage.name === "Silero VAD");
      const selectedVadAttempt = selectedVadStage?.attempts?.find((attempt) => attempt.runtime === selectedVadStage.selectedRuntime);
      vadElapsed = selectedVadAttempt?.elapsedSeconds ?? ((performance.now() - vadStart) / 1000);
      await writeResumeJsonCheckpoint(resumeContext, "vad", encodeVadCheckpoint(vad, vadElapsed, pipelineOptions), {
        segments: vad.segments?.length || 0,
        probabilities: vad.probabilities?.length || 0,
      });
    }
    let asrAudio = canonicalAudio;
    if (pipelineOptions.bypassVad) {
      addBenchmarkStage(options, {
        name: "VAD bypass",
        capability: "off",
        attempts: [{
          runtime: "off",
          provider: "off",
          elapsedSeconds: Number(vadElapsed.toFixed(3)),
          summary: summarizeVadForBenchmark(vad),
        }],
        selectedRuntime: "off",
        selectedProvider: "off",
      });
    }
    if (pipelineOptions.bypassVad) {
      setPipelineProgress("VAD bypassed", 30);
      log("VAD bypassed by user setting; ASR will process the whole file.");
    }
    await unloadModelsAfterStep("vad", pipelineOptions);
    if (pipelineOptions.rmsNormalize) {
      setPipelineProgress("RMS normalize", 30);
      const normalized = perSegmentRmsNormalize(canonicalAudio, vad.segments, 16000);
      asrAudio = normalized.audio;
    }
    lastAudioPcm = canonicalAudio;
    const vadSpeechSeconds = vad.segments.reduce((sum, segment) => sum + segment.end - segment.start, 0) / 16000;
    const vadProb = probabilityStats(vad.probabilities);
    const summaryItems = [
      { label: "Duration", value: `${duration.toFixed(1)} s` },
      { label: "Audio decoder", value: decoded.decoder },
      { label: "Original rate", value: decoded.originalSampleRate ? `${decoded.originalSampleRate} Hz` : "unknown" },
      { label: "Offline rate", value: "16000 Hz mono" },
      { label: "ASR model", value: pipelineOptions.asrModel.label },
      { label: "CPU threads", value: `${pipelineOptions.cpuThreads}` },
      { label: "Peak / RMS", value: `${stats.peak.toFixed(3)} / ${stats.rms.toFixed(3)}` },
      { label: "Energy speech preview", value: `${Math.round(stats.speechRatio * 100)}%` },
      { label: "Silero VAD segments", value: pipelineOptions.bypassVad ? "bypassed" : `${vad.segments.length}` },
      { label: "VAD speech", value: `${vadSpeechSeconds.toFixed(1)} s` },
      { label: "VAD prob max / avg", value: `${vadProb.max.toFixed(3)} / ${vadProb.avg.toFixed(3)}` },
      { label: "VAD runtime", value: `${vadElapsed.toFixed(2)} s` },
      { label: "RMS normalize", value: pipelineOptions.rmsNormalize ? "on" : "off" },
      { label: "PCM size", value: formatBytes(canonicalAudio.byteLength) },
    ];

    renderAudioSummary(summaryItems);

    log(`Canonical PCM: 16 kHz mono, ${formatBytes(canonicalAudio.byteLength)}.`);
    log(`Silero VAD: ${vad.segments.length} segment(s), ${vadSpeechSeconds.toFixed(1)}s speech.`);
    if (vad.segments.length) {
      const preview = vad.segments
        .slice(0, 8)
        .map((segment) => `${(segment.start / 16000).toFixed(2)}-${(segment.end / 16000).toFixed(2)}s`)
        .join(", ");
      log(`VAD preview: ${preview}${vad.segments.length > 8 ? ", ..." : ""}`);
    }
    const preprocessingElapsed = (performance.now() - totalStarted) / 1000;
    try {
      let asr = resumeCheckpoints.asr || null;
      if (asr) {
        setPipelineProgress("ASR resumed", 65);
        renderTranscript(asr.text || "", "ASR resumed from resume_after_kill checkpoint");
        log(`[resume_after_kill] Resumed ASR: ${asr.chunks?.length || 0} chunk(s), ${asr.words?.length || 0} word(s).`);
      } else {
        asr = await runFullAsr(asrAudio, vad.segments, {
          asrModel: pipelineOptions.asrModel,
          cpuThreads: pipelineOptions.cpuThreads,
          hotwordsText: pipelineOptions.hotwordsText,
          hotwordsScore: pipelineOptions.hotwordsScore,
          vadProbabilities: vad.probabilities || [],
          resumeContext,
          resumeAsrChunks: resumeCheckpoints.asr_chunks,
          progress: (done, total) => {
            setPipelineProgress("ASR", 30 + (done / Math.max(1, total)) * 35);
          },
        });
        await writeResumeJsonCheckpoint(resumeContext, "asr", asr, {
          chunks: asr.chunks?.length || 0,
          words: asr.words?.length || 0,
        });
      }
      await addAsrBenchmarkWebGpuBranch(options, asr, asrAudio, vad.segments, pipelineOptions, vad.probabilities || []);
      await unloadModelsAfterStep("asr", pipelineOptions);

      setPipelineProgress("Quality analysis", 66);
      const qualitySpeechSamples = copyTimelineRange(asrAudio, asr.timeline, 0, asr.timeline.totalSamples);
      const qualityRunner = () => computeQualityInfoFromSpeech(qualitySpeechSamples, asr.confidence, {
        progress: (done, total) => {
          setPipelineProgress("Quality analysis", 66 + ((done + 1) / Math.max(1, total)) * 6);
        },
          strictDnsmos: Array.isArray(options.benchmarkStages),
        });
      let qualityInfo = resumeCheckpoints.quality || null;
      if (qualityInfo) {
        log("[resume_after_kill] Resumed quality analysis checkpoint.");
      } else {
        qualityInfo = shouldBenchmarkWebGpuStage(options, "DNSMOS quality")
          ? await runBenchmarkDualProviderStage(
              options,
              "DNSMOS quality",
              qualityRunner,
              unloadQualityModel,
              summarizeQualityForBenchmark
            )
          : await qualityRunner();
        await writeResumeJsonCheckpoint(resumeContext, "quality", qualityInfo || {});
      }
      if (qualityInfo) {
        log(
          `Quality analysis: DNSMOS=${qualityInfo.dnsmos_ovrl ?? "n/a"}, ` +
          `ASR confidence=${qualityInfo.asr_confidence !== undefined ? `${(qualityInfo.asr_confidence * 100).toFixed(1)}%` : "n/a"}.`
        );
      }
      await unloadModelsAfterStep("quality", pipelineOptions);

      let diarization = { segments: [], speakers: 0, elapsed: 0, backend: "off", embeddings: 0, overlapRegions: [] };
      const resumedDiarization = decodeDiarizationCheckpoint(resumeCheckpoints.diarization);
      if (resumedDiarization) {
        diarization = resumedDiarization;
        if (Array.isArray(asr.words) && asr.words.length && Array.isArray(diarization.rawSegments) && diarization.rawSegments.length) {
          const segments = desktopPostProcessDiarizationSegments(diarization.rawSegments, asr.words);
          diarization = {
            ...diarization,
            segments,
            speakers: new Set(segments.map((segment) => segment.speaker)).size,
          };
        }
        renderDiarization(diarization.segments || []);
        setPipelineProgress("Diarization resumed", 88);
        log(`[resume_after_kill] Resumed diarization: ${diarization.segments?.length || 0} turn(s), ${diarization.speakers || 0} speaker(s).`);
      } else if (pipelineOptions.speakerDiarization) {
        setPipelineProgress("Diarization", 72);
        diarization = await runDiarization(canonicalAudio, {
          speakerModel: pipelineOptions.speakerModel,
          numSpeakers: pipelineOptions.numSpeakers,
          asrWords: asr.words,
          benchmarkStages: options.benchmarkStages,
          resumeContext,
          resumeCheckpoints,
          progress: (done, total) => {
            const ratio = Number.isFinite(total) ? done / Math.max(1, total) : done;
            setPipelineProgress("Diarization", 72 + ratio * 16);
          },
        });
        if (!diarization.segments?.length) {
          throw new Error("Speaker diarization was enabled but produced no speaker turns.");
        }
        if (pipelineOptions.numSpeakers > 1 && diarization.speakers < 2) {
          throw new Error(`Speaker diarization was fixed to ${pipelineOptions.numSpeakers} speaker(s) but produced only ${diarization.speakers}.`);
        }
        await writeResumeJsonCheckpoint(resumeContext, "diarization", encodeDiarizationCheckpoint(diarization), {
          turns: diarization.segments?.length || 0,
          speakers: diarization.speakers || 0,
        });
      } else {
        renderDiarization([], "[speaker diarization disabled]");
        log("Speaker diarization skipped by user setting.");
        addBenchmarkStage(options, {
          name: "Speaker diarization",
          capability: "off",
          attempts: [{ runtime: "off", provider: "off", elapsedSeconds: 0, summary: { enabled: false } }],
          selectedRuntime: "off",
          selectedProvider: "off",
        });
      }
      await unloadModelsAfterStep("diarization", pipelineOptions);

      let finalText = asr.text;
      let punctuation = null;
      if (resumeCheckpoints.punctuation) {
        punctuation = resumeCheckpoints.punctuation.punctuation || null;
        finalText = resumeCheckpoints.punctuation.finalText || punctuation?.text || finalText;
        renderTranscript(finalText, "Punctuation resumed from resume_after_kill checkpoint");
        log("[resume_after_kill] Resumed punctuation checkpoint.");
      } else if (pipelineOptions.bypassPunctuation) {
        log("Punctuation skipped by user setting.");
        addBenchmarkStage(options, {
          name: "ViBERT punctuation",
          capability: "off",
          attempts: [{ runtime: "off", provider: "off", elapsedSeconds: 0, summary: { enabled: false } }],
          selectedRuntime: "off",
          selectedProvider: "off",
        });
      } else if (finalText.trim()) {
        const pauseHints = buildPunctuationPauseHints(asr.words, diarization.segments);
        if (pauseHints) {
          const speakerPauses = pauseHints.filter((gap) => gap >= 1.0).length;
          log(`Punctuation pause hints: ${pauseHints.length} word(s), ${speakerPauses} long pause/boundary hint(s).`);
        }
        setPipelineProgress("Punctuation", 88);
        const punctuationInput = finalText;
        const benchmarkPunctuationInput = shouldBenchmarkWebGpuStage(options, "ViBERT punctuation")
          ? syntheticPunctuationBenchmarkText()
          : punctuationInput;
        const punctuationOptions = {
          punctuationConfidence: pipelineOptions.punctuationConfidence,
          caseConfidence: pipelineOptions.caseConfidence,
          pauseHints,
          progress: (iter, done, total) => {
            const iterBase = iter / PUNCT_ITERATIONS;
            const miniBatchPct = (done / Math.max(1, total)) / PUNCT_ITERATIONS;
            setPipelineProgress("Punctuation", 88 + (iterBase + miniBatchPct) * 7);
          },
        };
        const punctuationRunner = () => restorePunctuation(punctuationInput, punctuationOptions);
        if (shouldBenchmarkWebGpuStage(options, "ViBERT punctuation")) {
          const benchmarkPunctuationRunner = () => restorePunctuation(benchmarkPunctuationInput, {
            ...punctuationOptions,
            pauseHints: null,
          });
          log(`[Benchmark] ViBERT punctuation synthetic input: ${benchmarkPunctuationInput.split(/\s+/).length} word(s).`);
          await runBenchmarkDualProviderStage(
              options,
              "ViBERT punctuation",
              benchmarkPunctuationRunner,
              unloadPunctuationModel,
              summarizePunctuationForBenchmark
            );
          await unloadPunctuationModel();
          if (options.benchmarkOnly) {
            punctuation = {
              text: finalText,
              elapsed: 0,
              chunks: 0,
              executionProvider: "benchmark-only-skipped",
            };
            log("[Benchmark] Actual transcript punctuation skipped after synthetic ViBERT benchmark.");
          } else {
            punctuation = await punctuationRunner();
          }
        } else {
          punctuation = await punctuationRunner();
        }
        finalText = punctuation.text;
        renderTranscript(finalText, `ASR + diarization-first punctuation ${punctuation.elapsed.toFixed(2)}s`);
        log(`Punctuation finished in ${punctuation.elapsed.toFixed(2)}s.`);
        await writeResumeJsonCheckpoint(resumeContext, "punctuation", {
          finalText,
          punctuation,
        }, {
          chars: finalText.length,
        });
      } else {
        addBenchmarkStage(options, {
          name: "ViBERT punctuation",
          capability: "off",
          attempts: [{ runtime: "off", provider: "off", elapsedSeconds: 0, summary: { enabled: true, reason: "empty text" } }],
          selectedRuntime: "off",
          selectedProvider: "off",
        });
      }
      await unloadModelsAfterStep("punctuation", pipelineOptions);

      let overlap = {
        segments: [],
        elapsed: 0,
        backend: "off",
        detectedRegions: 0,
        candidateRegions: 0,
        processedRegions: 0,
        skipped: [],
      };
      if (resumeCheckpoints.overlap) {
        overlap = resumeCheckpoints.overlap;
        renderOverlapSegments(overlap.segments || [], "[overlap separation resumed]");
        log(`[resume_after_kill] Resumed overlap separation: ${overlap.segments?.length || 0} line(s).`);
      } else if (pipelineOptions.overlapSeparation) {
        setPipelineProgress("Overlap separation", 96);
        overlap = await runOverlapSeparation(canonicalAudio, diarization, {
          asrModel: pipelineOptions.asrModel,
          cpuThreads: pipelineOptions.cpuThreads,
          hotwordsText: pipelineOptions.hotwordsText,
          hotwordsScore: pipelineOptions.hotwordsScore,
          progress: (ratio) => {
            setPipelineProgress("Overlap separation", 96 + ratio * 3);
          },
        });
        addBenchmarkStage(options, {
          name: "Overlap separation",
          capability: "wasm-only",
          attempts: [{
            runtime: "wasm",
            provider: "wasm",
            elapsedSeconds: Number(overlap.elapsed.toFixed(3)),
            summary: {
              detectedRegions: overlap.detectedRegions,
              candidateRegions: overlap.candidateRegions,
              processedRegions: overlap.processedRegions,
              outputLines: overlap.segments.length,
            },
          }],
          selectedRuntime: "wasm",
          selectedProvider: "wasm",
        });
        await writeResumeJsonCheckpoint(resumeContext, "overlap", overlap, {
          lines: overlap.segments?.length || 0,
        });
      } else {
        renderOverlapSegments([], "[overlap separation disabled]");
        addBenchmarkStage(options, {
          name: "Overlap separation",
          capability: "off",
          attempts: [{ runtime: "off", provider: "off", elapsedSeconds: 0, summary: { enabled: false } }],
          selectedRuntime: "off",
          selectedProvider: "off",
        });
      }
      await unloadModelsAfterStep("overlap", pipelineOptions);

      renderAudioSummary([
        ...summaryItems,
        { label: "ASR chunks", value: `${asr.chunks.length}` },
        { label: "ASR speech", value: `${asr.speechSeconds.toFixed(1)} s` },
        { label: "ASR runtime", value: `${asr.elapsed.toFixed(2)} s` },
        { label: "ASR confidence", value: Number.isFinite(asr.confidence) ? `${(asr.confidence * 100).toFixed(1)}%` : "n/a" },
        { label: "Quality runtime", value: Number.isFinite(qualityInfo?.elapsed) ? `${qualityInfo.elapsed.toFixed(2)} s` : "n/a" },
        { label: "Punctuation", value: pipelineOptions.bypassPunctuation ? "off" : (punctuation ? `${punctuation.elapsed.toFixed(2)} s` : "no text") },
        { label: "Punct chunks", value: punctuation ? `${punctuation.chunks}` : "0" },
        { label: "Speaker turns", value: `${diarization.segments.length}` },
        { label: "Speakers", value: `${diarization.speakers}` },
        { label: "Diar backend", value: `${diarization.backend}` },
        { label: "Speaker embeds", value: diarization?.embeddings ? `${diarization.embeddings}` : "0" },
        { label: "Diar runtime", value: `${diarization.elapsed.toFixed(2)} s` },
        { label: "Overlap regions", value: pipelineOptions.overlapSeparation ? `${overlap.candidateRegions}/${overlap.detectedRegions}` : "off" },
        { label: "Overlap lines", value: `${overlap.segments.length}` },
        { label: "Overlap runtime", value: `${overlap.elapsed.toFixed(2)} s` },
        { label: "Transcript chars", value: `${finalText.length}` },
      ]);
      const timing = {
        preprocessing: Number(preprocessingElapsed.toFixed(3)),
        transcription_detail: Number(asr.elapsed.toFixed(3)),
        diarization: Number((diarization.elapsed || 0).toFixed(3)),
        punctuation: Number((punctuation?.elapsed || 0).toFixed(3)),
        overlap_separation: Number((overlap.elapsed || 0).toFixed(3)),
        total: Number(((performance.now() - totalStarted) / 1000).toFixed(3)),
      };
      const pipelineResult = {
        file,
        samples: canonicalAudio,
        duration,
        text: finalText,
        vad: {
          elapsed: vadElapsed,
          segments: vad.segments.length,
          speechSeconds: vadSpeechSeconds,
          bypassed: pipelineOptions.bypassVad,
          provider: vad.benchmarkSelectedProvider || (pipelineOptions.bypassVad ? "off" : vadExecutionProvider),
        },
        asr,
        punctuation,
        diarization,
        overlap,
        qualityInfo,
        timing,
        pipelineOptions,
        libraryItemId: selectedLibraryItemId,
        pipelineLog: pipelineLogLines.slice(),
        benchmarkStages: options.benchmarkStages || undefined,
      };
      if (options.benchmarkOnly) {
        log("[Benchmark] Benchmark-only run: ASR result was measured but not opened in editor and not saved to library.");
      } else {
        setEditorResult(pipelineResult);
        if (options.saveLibraryResult !== false) {
          await saveCurrentEditorResultToLibrary();
        }
        if (resumeContext) {
          await clearResumeAfterKillForItem(resumeContext.itemId).catch((error) => {
            log(`[resume_after_kill] Cleanup after completion failed: ${error.message || String(error)}`);
          });
        }
        pipelineResult.timing.total = Number(((performance.now() - totalStarted) / 1000).toFixed(3));
        if (editorState?.timing) {
          editorState.timing.total = pipelineResult.timing.total;
          renderResultTiming();
        }
      }
      setPipelineProgress("Done", 100);
      log("Pipeline completed.");
      return pipelineResult;
    } catch (error) {
      setPipelineProgress("Pipeline failed", 100);
      log(`Pipeline failed: ${error.message}`);
      await unloadModelsAfterStep("all", pipelineOptions);
      throw error;
    }
    log("Pipeline completed.");
  } catch (error) {
    setPipelineProgress("Pipeline failed", 100);
    log(`Pipeline failed: ${error.message}`);
    await unloadModelsAfterStep("all", pipelineOptions);
    throw error;
  }
}

async function main() {
  mountAdvancedSettings();
  updateStandaloneUi();
  await initializeUserConfig();
  configureOrt();
  setupInstallPrompt();
  setupEvents();
  await registerServiceWorker().catch((error) => {
    log(`Service worker registration/update failed: ${error.message || String(error)}`);
  });
  await loadManifest();
  await loadCalibrationProfileForCurrentDevice();
  await requestPersistentStorage(false);
  await refreshOfflineBootstrapState();

  // Kiểm tra nếu app vừa được cài (flag từ appinstalled event)
  // hoặc đang chạy ở standalone mode → tự động tải model
  const justInstalled = (() => {
    try { return localStorage.getItem("pwa_just_installed") === "1"; } catch(_) { return false; }
  })();
  if (justInstalled) {
    try { localStorage.removeItem("pwa_just_installed"); } catch(_) {}
    markStandaloneInstalled();
    // Force body classes ngay lập tức để UI hiện bootstrap panel đúng
    // (media query standalone chưa fire kịp khi app vừa mở)
    document.body.classList.add("standalone-app");
    document.body.classList.remove("browser-install-mode");
    document.body.classList.add("bootstrap-needed");
    document.body.classList.remove("offline-ready");
  }
  if (isStandaloneApp() || justInstalled) {
    autoDownloadAfterInstall(justInstalled ? "first install launch" : "standalone launch")
      .catch((error) => log(`Standalone bootstrap failed: ${error.message}`));
  }
  await recoverInterruptedLibraryItems().catch((error) => {
    log(`Interrupted library recovery failed: ${error.message || String(error)}`);
  });
  await renderLibrary();
  await updateRuntimeStatus();
  log("Offline PWA is ready.");
}

main().catch((error) => {
  console.error(error);
  log(`ERROR: ${error.message}`);
  $("runtime-status").textContent = "Error";
});


// =====================================================================
// NEW UI FEATURES: Player Panel, Context Menu, Modals (Meeting, Rename, Split)
// =====================================================================

function processSelectedAudioFile() {
  if (!selectedAudioFile && !selectedLibraryItemId) return;
  const fileName = selectedAudioFile ? selectedAudioFile.name.replace(/\.[^.]+$/, "") : "";
  if ($("meeting-name-input")) {
      $("meeting-name-input").value = "";
      $("meeting-name-input").placeholder = fileName || "Nhập tên cuộc họp...";
      $("meeting-name-modal").style.display = "flex";
      $("meeting-name-input").focus();
  } else {
      doProcessSelectedAudioFile();
  }
}

if ($("btn-confirm-meeting")) {
    $("btn-confirm-meeting").addEventListener("click", async () => {
       $("meeting-name-modal").style.display = "none";
       const meetingName = $("meeting-name-input").value.trim();
       if (meetingName && (selectedLibraryItemId || selectedLibraryImportPromise)) {
          try {
              // Wait for library item if it's still being created
              if (selectedLibraryImportPromise) await selectedLibraryImportPromise;
              const itemId = selectedLibraryItemId;
              const item = itemId ? await libraryGetItem(itemId) : null;
              if (item) {
                 await updateLibraryItem(itemId, { meetingName });
              }
          } catch(e) { console.error(e); }
       }
       await doProcessSelectedAudioFile();
    });
}
if ($("btn-cancel-meeting")) {
    $("btn-cancel-meeting").addEventListener("click", () => {
       $("meeting-name-modal").style.display = "none";
    });
}
if ($("btn-close-meeting-modal")) {
    $("btn-close-meeting-modal").addEventListener("click", () => {
       $("meeting-name-modal").style.display = "none";
    });
}

// Player Panel
function setupPlayerUI() {
  const audio = $("editor-audio");
  if (!audio) return;

  $("btn-play")?.addEventListener("click", () => {
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  });

  audio.addEventListener("play", () => {
    const playIcon = document.querySelector(".play-icon");
    const pauseIcon = document.querySelector(".pause-icon");
    if(playIcon) playIcon.style.display = "none";
    if(pauseIcon) pauseIcon.style.display = "block";
  });
  audio.addEventListener("pause", () => {
    const playIcon = document.querySelector(".play-icon");
    const pauseIcon = document.querySelector(".pause-icon");
    if(playIcon) playIcon.style.display = "block";
    if(pauseIcon) pauseIcon.style.display = "none";
  });

  audio.addEventListener("timeupdate", () => {
    const seek = $("player-seek");
    const time = $("player-time");
    if (seek && !seek.matches(":active") && audio.duration) {
      seek.max = audio.duration;
      seek.value = audio.currentTime;
    }
    if (time) {
      time.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration || 0)}`;
    }
  });

  audio.addEventListener("loadedmetadata", () => {
      const seek = $("player-seek");
      const time = $("player-time");
      if (seek) seek.max = audio.duration || 0;
      if (time) time.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration || 0)}`;
  });

  $("player-seek")?.addEventListener("input", (e) => {
    audio.currentTime = parseFloat(e.target.value);
  });

  // Cho phép click vào thanh seek để nhảy nhanh
  $("player-seek")?.addEventListener("change", (e) => {
    audio.currentTime = parseFloat(e.target.value);
  });

  const scrollTopBtn = $("btn-scroll-top");
  $("btn-scroll-top")?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Hiện nút scroll-to-top khi cuộn xuống đủ xa (> 300px)
  window.addEventListener("scroll", () => {
    if (scrollTopBtn) {
      scrollTopBtn.classList.toggle("visible", window.scrollY > 300);
    }
  }, { passive: true });
}

// Context Menu
let ctxTargetSegmentIndex = -1;

function setupContextMenuUI() {
  const menu = $("context-menu");
  if (!menu) return;

  document.addEventListener("click", (e) => {
    if (!e.target.closest("#context-menu")) {
      menu.style.display = "none";
    }
  });

  $("result-content")?.addEventListener("contextmenu", (e) => {
    const seg = e.target.closest(".editor-segment");
    if (seg && editorState) {
      e.preventDefault();
      ctxTargetSegmentIndex = parseInt(seg.dataset.index, 10);
      menu.style.display = "block";
      menu.style.left = `${e.pageX}px`;
      menu.style.top = `${e.pageY}px`;
    }
  });

  $("ctx-split-speaker")?.addEventListener("click", () => {
    if (ctxTargetSegmentIndex >= 0) {
      const select = $("split-speaker-select");
      select.innerHTML = '<option value="">-- Chọn từ danh sách --</option>';
      for (const [id, meta] of Object.entries(editorState.speakers)) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = meta.name;
        select.appendChild(opt);
      }
      $("split-speaker-input").value = "";
      const seg = editorState.segments[ctxTargetSegmentIndex];
      $("split-current-speaker").textContent = editorState.speakers[seg.speaker]?.name || seg.speaker;

      $("split-speaker-modal").style.display = "flex";
      menu.style.display = "none";
    }
  });

  $("btn-close-split-modal")?.addEventListener("click", () => $("split-speaker-modal").style.display = "none");
  $("btn-cancel-split")?.addEventListener("click", () => $("split-speaker-modal").style.display = "none");

  $("btn-confirm-split")?.addEventListener("click", () => {
     const newIdStr = $("split-speaker-select").value;
     const newName = $("split-speaker-input").value.trim();
     let targetId;
     if (newIdStr) {
       targetId = parseInt(newIdStr, 10);
     } else {
       targetId = nextEditorSpeakerId();
       editorState.speakers[targetId] = {
         name: newName || defaultSpeakerName(targetId),
         color: SPEAKER_COLORS[targetId % SPEAKER_COLORS.length]
       };
     }

     const scope = document.querySelector('input[name="split-scope"]:checked').value;
     if (scope === "to_end") {
       const block = findEditorSpeakerBlock(ctxTargetSegmentIndex);
       for (let i = ctxTargetSegmentIndex; i <= block.end; i++) {
         editorState.segments[i].speaker = targetId;
       }
     } else {
       editorState.segments[ctxTargetSegmentIndex].speaker = targetId;
     }
     $("split-speaker-modal").style.display = "none";
     syncEditorSpeakers(editorState.speakers);
     renderEditor();
     scheduleLibraryResultAutosave();
  });

  $("ctx-merge-up")?.addEventListener("click", () => {
    menu.style.display = "none";
    if (ctxTargetSegmentIndex > 0) {
       mergeEditorSpeakerBlock(ctxTargetSegmentIndex, "prev");
    }
  });

  $("ctx-merge-down")?.addEventListener("click", () => {
    menu.style.display = "none";
    mergeEditorSpeakerBlock(ctxTargetSegmentIndex, "next");
  });

  $("ctx-rename-speaker")?.addEventListener("click", () => {
    menu.style.display = "none";
    if (ctxTargetSegmentIndex >= 0) {
      const seg = editorState.segments[ctxTargetSegmentIndex];
      const speakerId = seg.speaker;
      const meta = editorState.speakers[speakerId];
      $("rename-current").textContent = meta ? meta.name : speakerId;
      $("rename-input").value = meta ? meta.name : "";

      const select = $("rename-select");
      select.innerHTML = '<option value="">-- Chọn --</option>';
      for (const [id, s] of Object.entries(editorState.speakers)) {
        const opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = s.name;
        select.appendChild(opt);
      }

      const colorsContainer = $("rename-colors");
      colorsContainer.innerHTML = "";
      SPEAKER_COLORS.forEach(c => {
         const div = document.createElement("div");
         div.className = "color-swatch";
         div.style.backgroundColor = c;
         if (meta && meta.color === c) div.classList.add("selected");
         div.addEventListener("click", () => {
           colorsContainer.querySelectorAll(".color-swatch").forEach(el => el.classList.remove("selected"));
           div.classList.add("selected");
           colorsContainer.dataset.selectedColor = c;
         });
         colorsContainer.appendChild(div);
      });
      colorsContainer.dataset.selectedColor = meta ? meta.color : SPEAKER_COLORS[0];

      $("rename-modal").style.display = "flex";
    }
  });

  $("btn-close-rename-modal")?.addEventListener("click", () => $("rename-modal").style.display = "none");
  $("btn-cancel-rename")?.addEventListener("click", () => $("rename-modal").style.display = "none");

  const applyRename = (all) => {
      const newName = $("rename-input").value.trim();
      const newColor = $("rename-colors").dataset.selectedColor;
      const seg = editorState.segments[ctxTargetSegmentIndex];
      const speakerId = seg.speaker;

      if (all) {
          if (newName) editorState.speakers[speakerId].name = newName;
          if (newColor) editorState.speakers[speakerId].color = newColor;
      } else {
          const newId = nextEditorSpeakerId();
          editorState.speakers[newId] = {
             name: newName || defaultSpeakerName(newId),
             color: newColor || SPEAKER_COLORS[newId % SPEAKER_COLORS.length]
          };
          const block = findEditorSpeakerBlock(ctxTargetSegmentIndex);
          for (let i = block.start; i <= block.end; i++) {
             editorState.segments[i].speaker = newId;
          }
      }
      $("rename-modal").style.display = "none";
      syncEditorSpeakers(editorState.speakers);
      renderEditor();
      scheduleLibraryResultAutosave();
  };

  $("btn-rename-all")?.addEventListener("click", () => applyRename(true));
  $("btn-rename-single")?.addEventListener("click", () => applyRename(false));

  $("ctx-copy")?.addEventListener("click", () => {
    menu.style.display = "none";
    if (ctxTargetSegmentIndex >= 0) {
      navigator.clipboard.writeText(editorState.segments[ctxTargetSegmentIndex].text);
      showToast("Đã sao chép văn bản");
    }
  });
}

// Call setups
setupPlayerUI();
setupContextMenuUI();

// Make sure player panel shows when there is an audio
const originalLoadAudio = window.loadEditorAudioSource;
// Wait, loadEditorAudioSource isn't global, it is inside app.js
// Actually we can just show player-panel if we have active editorState
// Let's hook into renderEditor to show player-panel
const originalRenderEditor = renderEditor;
renderEditor = function() {
    originalRenderEditor.apply(this, arguments);
    if (editorState) {
        $("player-panel").style.display = "flex";
    }
};
