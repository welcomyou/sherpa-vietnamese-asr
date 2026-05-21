"""Canonical audio decode/resample helpers shared by ASR and diarization.

All non-direct audio file paths should become mono float32 PCM at 16 kHz via
the same FFmpeg + SoXR command so MP3/M4A/video decoding does not drift across
Desktop, PWA, and Android implementations.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np


TARGET_SAMPLE_RATE = 16000
FFMPEG_RESAMPLE_FILTER = "aresample=resampler=soxr:precision=20"


def find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path

    possible = [
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in possible:
        if os.path.exists(candidate):
            return candidate
    return None


def load_audio_ffmpeg_pipe(file_path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found")

    cmd = [
        ffmpeg,
        "-i", file_path,
        "-vn",
        "-af", FFMPEG_RESAMPLE_FILTER,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-loglevel", "error",
        "pipe:1",
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    raw_bytes, stderr = proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"ffmpeg error: {message}")
    return np.frombuffer(raw_bytes, dtype=np.float32)
