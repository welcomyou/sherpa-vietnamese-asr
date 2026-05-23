"""
Hardware and ONNX Runtime acceleration helpers.

The desktop pipeline must be conservative: request a GPU provider only when it
is present, verify that ORT actually used it, and fall back to CPU otherwise.
"""
from __future__ import annotations

import csv
import os
import platform
import site
import subprocess
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
OPENVINO_PROVIDER = "OpenVINOExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"
ROCM_PROVIDER = "ROCMExecutionProvider"
_DLL_DIR_HANDLES: List[Any] = []


def _run_text(cmd: Sequence[str], timeout: float = 4.0) -> str:
    try:
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _parse_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


@lru_cache(maxsize=1)
def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    out = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=5.0,
    )
    gpus: List[Dict[str, Any]] = []
    if not out:
        return gpus
    for row in csv.reader(out.splitlines()):
        if len(row) < 4:
            continue
        total_mb = _parse_int(row[1])
        free_mb = _parse_int(row[2])
        gpus.append(
            {
                "vendor": "nvidia",
                "name": row[0].strip(),
                "vram_total_mb": total_mb,
                "vram_free_mb": free_mb,
                "driver_version": row[3].strip(),
            }
        )
    return gpus


@lru_cache(maxsize=1)
def detect_windows_video_controllers() -> List[Dict[str, Any]]:
    if platform.system().lower() != "windows":
        return []
    out = _run_text(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Csv -NoTypeInformation",
        ],
        timeout=6.0,
    )
    if not out:
        return []
    rows = list(csv.DictReader(out.splitlines()))
    controllers: List[Dict[str, Any]] = []
    for row in rows:
        name = (row.get("Name") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if "intel" in lower:
            vendor = "intel"
        elif "nvidia" in lower:
            vendor = "nvidia"
        elif "amd" in lower or "radeon" in lower or "advanced micro devices" in lower:
            vendor = "amd"
        else:
            vendor = "unknown"
        controllers.append(
            {
                "vendor": vendor,
                "name": name,
                "adapter_ram_bytes": _parse_int(row.get("AdapterRAM")),
                "driver_version": (row.get("DriverVersion") or "").strip(),
            }
        )
    return controllers


def ort_available_providers(ort_module=None) -> List[str]:
    try:
        ort = ort_module
        if ort is None:
            import onnxruntime as ort  # type: ignore
        return list(ort.get_available_providers())
    except Exception:
        return []


@lru_cache(maxsize=1)
def preload_ort_cuda_dlls() -> Dict[str, Any]:
    """Make CUDA/cuDNN pip-package DLLs visible to ONNX Runtime on Windows."""
    added: List[str] = []
    errors: List[str] = []

    if platform.system().lower() == "windows" and hasattr(os, "add_dll_directory"):
        for root in site.getsitepackages():
            nvidia_root = Path(root) / "nvidia"
            if not nvidia_root.exists():
                continue
            for bin_dir in sorted(nvidia_root.glob("*/bin")):
                if not bin_dir.is_dir():
                    continue
                text = str(bin_dir)
                try:
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(text))
                    os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
                    added.append(text)
                except Exception as exc:
                    errors.append(f"{text}: {type(exc).__name__}: {exc}")

    try:
        import onnxruntime as ort  # type: ignore
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    except Exception as exc:
        errors.append(f"onnxruntime.preload_dlls: {type(exc).__name__}: {exc}")

    return {"added": added, "errors": errors}


@lru_cache(maxsize=1)
def detect_hardware() -> Dict[str, Any]:
    nvidia = detect_nvidia_gpus()
    controllers = detect_windows_video_controllers()
    gpus = list(nvidia)
    known = {gpu["name"].lower() for gpu in gpus}
    for ctl in controllers:
        if ctl["name"].lower() not in known:
            gpus.append(ctl)
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "gpus": gpus,
        "nvidia_gpus": nvidia,
        "video_controllers": controllers,
        "ort_available_providers": ort_available_providers(),
    }


def best_gpu() -> Optional[Dict[str, Any]]:
    gpus = detect_hardware().get("gpus") or []
    if not gpus:
        return None
    nvidia = [g for g in gpus if g.get("vendor") == "nvidia"]
    if nvidia:
        return max(nvidia, key=lambda g: g.get("vram_total_mb") or 0)
    intel = [g for g in gpus if g.get("vendor") == "intel"]
    if intel:
        return intel[0]
    amd = [g for g in gpus if g.get("vendor") == "amd"]
    if amd:
        return amd[0]
    return gpus[0]


def preferred_gpu_provider(policy: str = "auto", ort_module=None) -> Optional[str]:
    policy = (policy or "cpu").lower()
    if policy in ("cpu", "none", "off"):
        return None

    available = set(ort_available_providers(ort_module))
    gpu = best_gpu()

    if policy in ("cuda", "nvidia"):
        return CUDA_PROVIDER if CUDA_PROVIDER in available else None
    if policy in ("openvino", "intel"):
        return OPENVINO_PROVIDER if OPENVINO_PROVIDER in available else None
    if policy in ("directml", "dml", "amd"):
        if DML_PROVIDER in available:
            return DML_PROVIDER
        if ROCM_PROVIDER in available:
            return ROCM_PROVIDER
        return None
    if policy in ("rocm",):
        return ROCM_PROVIDER if ROCM_PROVIDER in available else None
    if policy not in ("auto", "gpu"):
        return None

    if gpu and gpu.get("vendor") == "nvidia" and CUDA_PROVIDER in available:
        return CUDA_PROVIDER
    if gpu and gpu.get("vendor") == "intel" and OPENVINO_PROVIDER in available:
        return OPENVINO_PROVIDER
    if gpu and gpu.get("vendor") == "amd":
        if DML_PROVIDER in available:
            return DML_PROVIDER
        if ROCM_PROVIDER in available:
            return ROCM_PROVIDER
    if gpu and OPENVINO_PROVIDER in available:
        return OPENVINO_PROVIDER
    if gpu and DML_PROVIDER in available:
        return DML_PROVIDER
    if gpu and ROCM_PROVIDER in available:
        return ROCM_PROVIDER
    return None


def ort_provider_request(policy: str = "cpu", ort_module=None) -> List[str]:
    provider = preferred_gpu_provider(policy, ort_module)
    if provider:
        return [provider, CPU_PROVIDER]
    return [CPU_PROVIDER]


def actual_session_provider(session: Any) -> str:
    try:
        providers = list(session.get_providers())
    except Exception:
        return CPU_PROVIDER
    for provider in (CUDA_PROVIDER, OPENVINO_PROVIDER, DML_PROVIDER, ROCM_PROVIDER):
        if provider in providers:
            return provider
    return providers[0] if providers else CPU_PROVIDER


def is_gpu_provider(provider: Optional[str]) -> bool:
    return provider in {CUDA_PROVIDER, OPENVINO_PROVIDER, DML_PROVIDER, ROCM_PROVIDER}


def create_ort_session(
    ort_module: Any,
    model_path: str,
    sess_options: Any,
    policy: str = "cpu",
    stage: str = "",
) -> Tuple[Any, Dict[str, Any]]:
    """Create an ORT session and verify the requested provider really stuck."""
    requested = ort_provider_request(policy, ort_module)
    tried: List[Dict[str, Any]] = []

    def _create(providers: List[str]):
        return ort_module.InferenceSession(model_path, sess_options, providers=providers)

    if requested != [CPU_PROVIDER]:
        try:
            if requested[0] == CUDA_PROVIDER:
                preload_ort_cuda_dlls()
            session = _create(requested)
            actual = actual_session_provider(session)
            info = {
                "stage": stage,
                "policy": policy,
                "requested_providers": requested,
                "actual_provider": actual,
                "session_providers": list(session.get_providers()),
                "used_gpu": is_gpu_provider(actual),
                "fallback_reason": None,
            }
            if is_gpu_provider(requested[0]) and not is_gpu_provider(actual):
                info["fallback_reason"] = f"requested {requested[0]} but ORT created {actual}"
            return session, info
        except Exception as exc:
            tried.append({"providers": requested, "error": f"{type(exc).__name__}: {exc}"})

    session = _create([CPU_PROVIDER])
    info = {
        "stage": stage,
        "policy": policy,
        "requested_providers": requested,
        "actual_provider": actual_session_provider(session),
        "session_providers": list(session.get_providers()),
        "used_gpu": False,
        "fallback_reason": tried[-1]["error"] if tried else None,
        "tried": tried,
    }
    return session, info


def gpu_vram_mb(free: bool = True) -> Optional[int]:
    gpu = best_gpu()
    if not gpu:
        return None
    key = "vram_free_mb" if free else "vram_total_mb"
    value = gpu.get(key)
    if value is None and not free:
        ram_bytes = gpu.get("adapter_ram_bytes")
        if ram_bytes:
            value = int(ram_bytes / (1024 * 1024))
    return _parse_int(value)


def auto_batch_size(stage: str, default: int, provider: Optional[str] = None) -> int:
    if not is_gpu_provider(provider):
        return int(default)

    free_mb = gpu_vram_mb(free=True)
    total_mb = gpu_vram_mb(free=False)
    budget_mb = free_mb or total_mb or 0
    stage_key = (stage or "").lower()

    if "pyannote" in stage_key and "embedding" in stage_key:
        if budget_mb >= 10000:
            return 32
        if budget_mb >= 6000:
            return 24
        if budget_mb >= 3000:
            return 16
        return 8

    if "punct" in stage_key or "vibert" in stage_key:
        if budget_mb >= 6000:
            return 32
        if budget_mb >= 2500:
            return 16
        return 8

    if "campp" in stage_key or "speaker" in stage_key:
        if budget_mb >= 10000:
            return 128
        if budget_mb >= 7000:
            return 96
        if budget_mb >= 3500:
            return 64
        return 32

    if budget_mb >= 7000:
        return max(default, 64)
    if budget_mb >= 3500:
        return max(default, 32)
    return min(default, 16)


def hardware_summary() -> str:
    hw = detect_hardware()
    gpus = hw.get("gpus") or []
    if not gpus:
        gpu_text = "no GPU detected"
    else:
        parts = []
        for gpu in gpus:
            name = gpu.get("name", "unknown")
            total = gpu.get("vram_total_mb")
            free = gpu.get("vram_free_mb")
            if total:
                parts.append(f"{name} ({free or '?'} / {total} MB free/total)")
            else:
                parts.append(name)
        gpu_text = "; ".join(parts)
    providers = ", ".join(hw.get("ort_available_providers") or [])
    return f"CPU cores={hw.get('cpu_count')}; GPU={gpu_text}; ORT providers=[{providers}]"
