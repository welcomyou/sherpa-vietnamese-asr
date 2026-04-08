"""
Centralized logging configuration for desktop app and web server.

Creates a log file at logs/asr_app.log (desktop) or logs/asr_web.log (web).
Log file is cleared on each app/server restart.
ALL output (both logging.* and print()) goes to the log file.

Usage:
    from core.log_config import setup_logging
    setup_logging("desktop")  # or "web"
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Log directory relative to project root
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


class _TeeWriter:
    """Write to both original stream (console) and log file.
    Captures all print() output into the log file automatically."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, text):
        if text and text.strip():  # skip empty lines
            try:
                self.log_file.write(text)
                if not text.endswith("\n"):
                    self.log_file.write("\n")
                self.log_file.flush()
            except (OSError, ValueError):
                pass
        # Always write to original console
        try:
            self.original.write(text)
        except (OSError, ValueError):
            pass

    def flush(self):
        try:
            self.log_file.flush()
        except (OSError, ValueError):
            pass
        try:
            self.original.flush()
        except (OSError, ValueError):
            pass

    # Pass through attributes for compatibility
    def fileno(self):
        return self.original.fileno()

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def isatty(self):
        return hasattr(self.original, "isatty") and self.original.isatty()


def setup_logging(mode="desktop", log_level=logging.INFO):
    """Configure file + console logging. Clears log file on startup.
    Redirects print() to log file as well.

    Args:
        mode: "desktop" or "web" — determines log filename
        log_level: logging level (default INFO)
    """
    os.makedirs(_LOG_DIR, exist_ok=True)

    filename = "asr_app.log" if mode == "desktop" else "asr_web.log"
    log_path = os.path.join(_LOG_DIR, filename)

    # Clear log file on startup (fresh log each session)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")
    except OSError:
        pass

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicates on re-init
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — rotate at 50MB, keep 2 backups
    fh = RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=2,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — keep existing console output
    ch = logging.StreamHandler(sys.__stdout__)  # use original stdout
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    for name in ("urllib3", "asyncio", "websockets", "httpcore", "httpx",
                 "multipart", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Redirect print() → tee to both console AND log file
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, log_file)
    sys.stderr = _TeeWriter(sys.__stderr__, log_file)

    logging.info(f"Logging initialized: {log_path} (mode={mode})")
    return log_path
