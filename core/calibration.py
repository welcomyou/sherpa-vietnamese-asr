"""Device calibration helpers for desktop and web server.

Calibration is opt-in: the application starts on CPU, detects whether a GPU
provider is usable, then runs the bundled 10 minute sample only when the user
confirms optimization.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.config import ALLOWED_THREADS, BASE_DIR
from core.hardware_accel import (
    CPU_PROVIDER,
    actual_session_provider,
    auto_batch_size,
    create_ort_session,
    detect_hardware,
    hardware_summary,
    installed_gpu_addons,
    is_gpu_provider,
    ort_provider_request,
    preferred_gpu_provider,
    recommended_gpu_addon,
)


CALIBRATION_SAMPLE_MP3 = Path(BASE_DIR) / "offline_pwa" / "static" / "calibration" / "1hour_qh_10min.mp3"
CALIBRATION_SAMPLE_WAV = Path(BASE_DIR) / "temp" / "1hour_qh_10min_16k.wav"
CALIBRATION_CACHE_WAV = Path(BASE_DIR) / "temp" / "calibration_1hour_qh_10min_16k.wav"
CALIBRATION_REPORT_PATH = Path(BASE_DIR) / "temp" / "device_calibration_last.json"


def _safe_ram_info() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return {
            "total_mb": int(vm.total / (1024 * 1024)),
            "available_mb": int(vm.available / (1024 * 1024)),
        }
    except Exception:
        return {}


def _emit(callback: Optional[Callable[[str, int], None]], message: str, percent: int) -> None:
    if callback:
        callback(message, max(0, min(100, int(percent))))


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _model_specs() -> List[Dict[str, Any]]:
    root = Path(BASE_DIR)
    return [
        {
            "stage": "ASR encoder",
            "path": root / "models" / "sherpa-onnx-zipformer-vi-2025-04-20" / "encoder-epoch-12-avg-8.onnx",
            "batch_default": 16,
        },
        {
            "stage": "ASR decoder",
            "path": root / "models" / "sherpa-onnx-zipformer-vi-2025-04-20" / "decoder-epoch-12-avg-8.onnx",
            "batch_default": 16,
        },
        {
            "stage": "ASR joiner",
            "path": root / "models" / "sherpa-onnx-zipformer-vi-2025-04-20" / "joiner-epoch-12-avg-8.onnx",
            "batch_default": 16,
        },
        {
            "stage": "DNSMOS quality",
            "path": root / "models" / "dnsmos" / "sig_bak_ovr.onnx",
            "batch_default": 16,
        },
        {
            "stage": "CAM++ speaker embedding",
            "path": root / "models" / "campp-3dspeaker" / "campplus_cn_en_common_200k.onnx",
            "batch_default": 32,
        },
        {
            "stage": "ViBERT punctuation fp32",
            "path": root / "models" / "vibert-capu" / "vibert-capu.onnx",
            "batch_default": 8,
        },
        {
            "stage": "Pyannote Community-1 embedding encoder",
            "path": root / "models" / "pyannote-onnx" / "embedding_encoder.onnx",
            "batch_default": 8,
        },
    ]


def run_provider_probe(policy: str = "auto", light: bool = False) -> List[Dict[str, Any]]:
    """Create ONNX sessions and record the actual provider ORT selected."""
    try:
        from core.hardware_accel import configure_gpu_addon_paths
        configure_gpu_addon_paths()
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return [{"stage": "onnxruntime", "error": f"{type(exc).__name__}: {exc}"}]

    specs = _model_specs()
    if light:
        specs = [s for s in specs if s["stage"] == "DNSMOS quality"] or specs[:1]

    results: List[Dict[str, Any]] = []
    for spec in specs:
        path = Path(spec["path"])
        if not path.exists():
            results.append({"stage": spec["stage"], "missing": str(path)})
            continue

        sess_options = ort.SessionOptions()
        try:
            session, info = create_ort_session(
                ort,
                str(path),
                sess_options,
                policy=policy,
                stage=spec["stage"],
            )
            actual = actual_session_provider(session)
            results.append(
                {
                    "stage": spec["stage"],
                    "requested_providers": info.get("requested_providers"),
                    "actual_provider": actual,
                    "session_providers": info.get("session_providers"),
                    "used_gpu": is_gpu_provider(actual),
                    "fallback_reason": info.get("fallback_reason"),
                    "auto_batch": auto_batch_size(spec["stage"], spec["batch_default"], actual),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "stage": spec["stage"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "requested_providers": ort_provider_request(policy, ort),
                }
            )
    return results


def detect_calibration_status() -> Dict[str, Any]:
    hw = detect_hardware()
    provider = preferred_gpu_provider("auto")
    light_probe = run_provider_probe("auto", light=True) if provider else []
    provider_ready = any(item.get("used_gpu") for item in light_probe)
    gpus = hw.get("gpus") or []

    reason = ""
    if not gpus:
        reason = "no_gpu"
    elif not provider:
        reason = "no_supported_ort_provider"
    elif not provider_ready:
        reason = "provider_probe_failed"

    addon = recommended_gpu_addon()
    if addon and gpus and not provider and not addon.get("installed"):
        reason = "gpu_addon_missing"
    elif addon and gpus and addon.get("installed") and not provider_ready:
        reason = "gpu_addon_provider_not_loaded"

    return {
        "hardware": hw,
        "ram": _safe_ram_info(),
        "hardware_summary": hardware_summary(),
        "preferred_provider": provider or CPU_PROVIDER,
        "provider_request": ort_provider_request("auto"),
        "provider_ready": provider_ready,
        "can_optimize": bool(gpus and provider and provider_ready),
        "reason": reason,
        "recommended_addon": addon,
        "installed_addons": installed_gpu_addons(),
        "light_probe": light_probe,
        "sample_file": str(CALIBRATION_SAMPLE_MP3),
        "sample_available": CALIBRATION_SAMPLE_MP3.exists() or CALIBRATION_SAMPLE_WAV.exists(),
        "last_report": str(CALIBRATION_REPORT_PATH),
        "last_report_available": CALIBRATION_REPORT_PATH.exists(),
    }


def _validate_wav_16k_mono(path: Path) -> None:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        if info.samplerate == 16000 and info.channels == 1 and info.frames > 0:
            return
    except Exception:
        pass

    from core.audio_decode import find_ffprobe, ffmpeg_error_tail

    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError(f"cannot validate calibration WAV; ffprobe not found: {path}")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=creationflags)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe validation failed: {ffmpeg_error_tail(result.stderr, result.stdout)}")
    text = result.stdout.decode("utf-8", errors="replace")
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    if fields.get("sample_rate") != "16000" or fields.get("channels") != "1":
        raise RuntimeError(f"calibration WAV validation failed: {fields}")


def ensure_calibration_wav() -> Path:
    if CALIBRATION_SAMPLE_WAV.exists():
        _validate_wav_16k_mono(CALIBRATION_SAMPLE_WAV)
        return CALIBRATION_SAMPLE_WAV
    if CALIBRATION_CACHE_WAV.exists():
        try:
            _validate_wav_16k_mono(CALIBRATION_CACHE_WAV)
            return CALIBRATION_CACHE_WAV
        except Exception:
            try:
                CALIBRATION_CACHE_WAV.unlink()
            except OSError:
                pass
    if not CALIBRATION_SAMPLE_MP3.exists():
        raise FileNotFoundError(f"Calibration sample not found: {CALIBRATION_SAMPLE_MP3}")

    from core.audio_decode import (
        ffmpeg_error_tail,
        ffmpeg_resample_filter_candidates,
        find_ffmpeg,
    )

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg.exe not found in app root folder")

    CALIBRATION_CACHE_WAV.parent.mkdir(parents=True, exist_ok=True)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    last_error = ""
    for filter_expr in ffmpeg_resample_filter_candidates():
        if CALIBRATION_CACHE_WAV.exists():
            try:
                CALIBRATION_CACHE_WAV.unlink()
            except OSError:
                pass
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(CALIBRATION_SAMPLE_MP3),
            "-vn",
        ]
        if filter_expr:
            cmd += ["-af", filter_expr]
        cmd += [
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(CALIBRATION_CACHE_WAV),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=creationflags)
        if result.returncode == 0:
            _validate_wav_16k_mono(CALIBRATION_CACHE_WAV)
            return CALIBRATION_CACHE_WAV
        last_error = ffmpeg_error_tail(result.stderr, result.stdout)
    raise RuntimeError(f"ffmpeg calibration conversion failed: {last_error}")


def _select_model_name(preferred: Optional[str]) -> str:
    candidates = [
        preferred,
        "sherpa-onnx-zipformer-vi-2025-04-20",
        "zipformer-30m-rnnt-6000h",
        "sherpa-onnx-zipformer-vi-30M",
    ]
    for name in candidates:
        if not name or name == "rover-voting":
            continue
        if (Path(BASE_DIR) / "models" / name).is_dir():
            return name
    raise FileNotFoundError("No calibration ASR model directory found")


def _select_speaker_model(preferred: Optional[str]) -> Optional[str]:
    try:
        from core.speaker_diarization import get_available_models

        available = get_available_models()
    except Exception:
        return None

    for name in (
        preferred,
        "community1_pure_ort",
        "senko_campp_optimized",
        "senko_campp",
    ):
        if name and name in available:
            return name
    return None


def _run_pipeline_once(
    wav_path: Path,
    provider: str,
    model_name: str,
    speaker_model: Optional[str],
    cpu_threads: int,
    callback: Optional[Callable[[str, int], None]],
    start_percent: int,
    end_percent: int,
) -> Dict[str, Any]:
    from core.asr_engine import TranscriberPipeline

    model_path = Path(BASE_DIR) / "models" / model_name
    span = max(1, end_percent - start_percent)

    def _pipeline_progress(msg: str) -> None:
        # Keep UI progress coarse; detailed pipeline percentages are stage-local.
        lower = str(msg).lower()
        offset = 0
        if "diar" in lower or "người" in lower:
            offset = int(span * 0.55)
        elif "dấu" in lower or "punct" in lower:
            offset = int(span * 0.80)
        elif "asr" in lower or "nhận" in lower:
            offset = int(span * 0.25)
        _emit(callback, msg, start_percent + offset)

    config = {
        "cpu_threads": max(1, min(int(cpu_threads or 4), ALLOWED_THREADS)),
        "execution_provider": provider,
        "restore_punctuation": True,
        "bypass_restorer": False,
        "punctuation_confidence": 0.5 - (7 - 1) * (1.3 / 9),
        "case_confidence": -1.5 + (6 - 1) * (2.0 / 9),
        "speaker_diarization": bool(speaker_model),
        "speaker_model": speaker_model or "community1_pure_ort",
        "num_speakers": -1,
        "save_ram": True,
        "rover_mode": False,
        "resample_quality": "soxr_hq",
    }

    started = time.monotonic()
    pipeline = TranscriberPipeline(
        file_path=str(wav_path),
        model_path=str(model_path),
        config=config,
        progress_callback=_pipeline_progress,
    )
    result = pipeline.run()
    elapsed = time.monotonic() - started
    duration = float(result.get("duration_sec") or 0)
    segments = result.get("segments") or []
    text = result.get("text") or result.get("full_text") or ""
    speaker_ids = set()
    for seg in segments:
        speaker = seg.get("speaker") if isinstance(seg, dict) else None
        if speaker:
            speaker_ids.add(str(speaker))
    for seg in result.get("speaker_segments_raw") or []:
        speaker = seg.get("speaker") if isinstance(seg, dict) else None
        if speaker:
            speaker_ids.add(str(speaker))
    speaker_count = len(speaker_ids) or len(result.get("speaker_names") or {})

    return {
        "provider": provider,
        "model_name": model_name,
        "speaker_model": speaker_model,
        "elapsed_sec": elapsed,
        "duration_sec": duration,
        "rtf": elapsed / duration if duration > 0 else None,
        "timing": result.get("timing") or {},
        "quality_info": result.get("quality_info") or {},
        "asr_provider_info": result.get("asr_provider_info") or {},
        "execution_provider": result.get("execution_provider"),
        "asr_confidence": result.get("asr_confidence"),
        "text_chars": len(text),
        "segments": len(segments),
        "speaker_turns": len(result.get("speaker_segments_raw") or result.get("speaker_segments") or []),
        "speaker_count": speaker_count,
    }


def _compare_runs(cpu_run: Dict[str, Any], gpu_run: Dict[str, Any]) -> Dict[str, Any]:
    cpu_elapsed = float(cpu_run.get("elapsed_sec") or 0)
    gpu_elapsed = float(gpu_run.get("elapsed_sec") or 0)
    speedup = (cpu_elapsed / gpu_elapsed) if cpu_elapsed > 0 and gpu_elapsed > 0 else None

    stage_speedups: Dict[str, Any] = {}
    cpu_timing = cpu_run.get("timing") or {}
    gpu_timing = gpu_run.get("timing") or {}
    for key in sorted(set(cpu_timing) | set(gpu_timing)):
        c = float(cpu_timing.get(key) or 0)
        g = float(gpu_timing.get(key) or 0)
        stage_speedups[key] = round(c / g, 3) if c > 0 and g > 0 else None

    cpu_text_chars = int(cpu_run.get("text_chars") or 0)
    text_delta = int(gpu_run.get("text_chars") or 0) - cpu_text_chars
    text_delta_ratio = abs(text_delta) / max(1, cpu_text_chars)
    speaker_count_delta = int(gpu_run.get("speaker_count") or 0) - int(cpu_run.get("speaker_count") or 0)
    speaker_turn_delta = int(gpu_run.get("speaker_turns") or 0) - int(cpu_run.get("speaker_turns") or 0)
    confidence_delta = None
    if cpu_run.get("asr_confidence") is not None and gpu_run.get("asr_confidence") is not None:
        confidence_delta = float(gpu_run["asr_confidence"]) - float(cpu_run["asr_confidence"])

    text_ok = abs(text_delta) <= max(20, int(cpu_text_chars * 0.002))
    parity_ok = (
        text_ok
        and speaker_count_delta == 0
        and speaker_turn_delta == 0
        and (confidence_delta is None or abs(confidence_delta) < 1e-4)
    )
    faster = bool(speedup and speedup >= 1.05)

    return {
        "wall_speedup": round(speedup, 3) if speedup else None,
        "stage_speedups": stage_speedups,
        "text_chars_delta": text_delta,
        "text_chars_delta_ratio": text_delta_ratio,
        "text_tolerance_ok": text_ok,
        "speaker_count_delta": speaker_count_delta,
        "speaker_turn_delta": speaker_turn_delta,
        "confidence_delta": confidence_delta,
        "parity_ok": parity_ok,
        "gpu_faster": faster,
        "accepted": bool(parity_ok and faster),
    }


def run_device_calibration(
    model_name: Optional[str] = None,
    speaker_model: Optional[str] = None,
    cpu_threads: int = 4,
    callback: Optional[Callable[[str, int], None]] = None,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Run CPU baseline and GPU candidate on the bundled 10 minute sample."""
    _emit(callback, "Detecting hardware", 1)
    status = detect_calibration_status()
    if not status.get("can_optimize"):
        report = {
            "status": "no_gpu",
            "selected_execution_provider": "cpu",
            "detect": status,
            "message": "Current CPU-only configuration is optimal for this machine.",
        }
        if save_report:
            CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    _emit(callback, "Preparing 10 minute calibration sample", 5)
    wav_path = ensure_calibration_wav()
    selected_model = _select_model_name(model_name)
    selected_speaker = _select_speaker_model(speaker_model)

    _emit(callback, "Running CPU baseline", 10)
    cpu_run = _run_pipeline_once(
        wav_path,
        "cpu",
        selected_model,
        selected_speaker,
        cpu_threads,
        callback,
        10,
        48,
    )

    _emit(callback, "Running GPU candidate", 52)
    gpu_run = _run_pipeline_once(
        wav_path,
        "auto",
        selected_model,
        selected_speaker,
        cpu_threads,
        callback,
        52,
        92,
    )

    _emit(callback, "Verifying calibration result", 94)
    comparison = _compare_runs(cpu_run, gpu_run)
    provider_probe = run_provider_probe("auto", light=False)
    selected_provider = "auto" if comparison["accepted"] else "cpu"

    report = {
        "status": "completed",
        "selected_execution_provider": selected_provider,
        "detect": status,
        "sample_wav": str(wav_path),
        "model_name": selected_model,
        "speaker_model": selected_speaker,
        "runs": {
            "cpu": cpu_run,
            "auto": gpu_run,
        },
        "comparison": comparison,
        "provider_probe": provider_probe,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if save_report:
        CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _emit(callback, "Calibration completed", 100)
    return report
