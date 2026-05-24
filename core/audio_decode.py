"""Canonical audio decode/resample helpers shared by ASR and diarization.

All non-direct audio file paths should become mono float32 PCM at 16 kHz via
FFmpeg. SoXR is preferred when the bundled FFmpeg supports it, with a standard
FFmpeg resampler fallback for smaller builds such as gyan.dev essentials.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


TARGET_SAMPLE_RATE = 16000
FFMPEG_RESAMPLE_FILTER = "aresample=resampler=soxr:precision=20"


def ffmpeg_resample_filter_candidates() -> list[str | None]:
    return [FFMPEG_RESAMPLE_FILTER, None]


def ffmpeg_error_tail(stderr: bytes | str | None, stdout: bytes | str | None = None, limit: int = 1200) -> str:
    chunks = []
    for data in (stderr, stdout):
        if not data:
            continue
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        if text.strip():
            chunks.append(text)
    text = "\n".join(chunks).strip()
    if not text:
        return "ffmpeg failed without stderr"
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-14:]) if lines else text
    return tail[-limit:]


def _tool_candidates(name: str) -> list[str]:
    exe_dir = Path(sys.executable).resolve().parent
    root = Path(__file__).resolve().parent.parent
    possible = [
        root / f"{name}.exe",
        root / "ffmpeg" / "bin" / f"{name}.exe",
        exe_dir / f"{name}.exe",
        exe_dir.parent / f"{name}.exe",
        Path(r"C:\ffmpeg\bin") / f"{name}.exe",
    ]
    found_on_path = shutil.which(name)
    if found_on_path:
        possible.append(Path(found_on_path))
    return [str(path) for path in possible]


def find_ffmpeg() -> str | None:
    possible = _tool_candidates("ffmpeg")
    for candidate in possible:
        if os.path.exists(candidate):
            return candidate
    return None


def find_ffprobe() -> str | None:
    possible = _tool_candidates("ffprobe")
    for candidate in possible:
        if os.path.exists(candidate):
            return candidate
    return None


def load_audio_ffmpeg_pipe(file_path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found")

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    last_error = ""
    for filter_expr in ffmpeg_resample_filter_candidates():
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-i", file_path,
            "-vn",
        ]
        if filter_expr:
            cmd += ["-af", filter_expr]
        cmd += [
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "pipe:1",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        raw_bytes, stderr = proc.communicate()
        if proc.returncode == 0:
            return np.frombuffer(raw_bytes, dtype=np.float32)
        last_error = ffmpeg_error_tail(stderr)

    raise RuntimeError(f"ffmpeg error: {last_error}")
