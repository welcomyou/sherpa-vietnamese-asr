"""
Entry point cho sherpa-vietnamese-asr-services.
Khoi dong FastAPI server voi HTTPS.
"""

import io
import os
import sys
import argparse
import logging

# --- Early crash handler: bat moi exception ke ca ImportError truoc khi log file mo ---
# Viet vao file crash ngay lap tuc de debug tren headless/no-console server.
def _early_crash_handler(exc_type, exc_value, exc_tb):
    import traceback as _tb
    import datetime as _dt
    _base = os.path.dirname(os.path.abspath(__file__))
    _crash_dir = os.path.join(_base, "web_service", "data", "logs")
    try:
        os.makedirs(_crash_dir, exist_ok=True)
        _crash_path = os.path.join(_crash_dir, "server_crash.log")
        with open(_crash_path, "w", encoding="utf-8", errors="replace") as _f:
            _f.write(f"[{_dt.datetime.now()}] SERVER CRASH:\n")
            _tb.print_exception(exc_type, exc_value, exc_tb, file=_f)
    except Exception:
        pass
    # fallback: ghi ra stderr (co the la DEVNULL nhung khong sao)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _early_crash_handler


class _NullWriter:
    """Null writer: thay the sys.stdout/stderr khi chay headless (stdout=DEVNULL).
    Bao ve moi print() call trong toan bo codebase khoi crash ValueError/OSError."""
    def write(self, *a): return 0
    def flush(self): pass
    def isatty(self): return False
    @property
    def encoding(self): return 'utf-8'
    @property
    def errors(self): return 'replace'


def _safe_stdout():
    """Tra ve True neu sys.stdout co the write duoc."""
    if sys.stdout is None:
        return False
    try:
        sys.stdout.write('')
        return True
    except (ValueError, OSError, AttributeError):
        return False


def _safe_stderr():
    if sys.stderr is None:
        return False
    try:
        sys.stderr.write('')
        return True
    except (ValueError, OSError, AttributeError):
        return False


# Neu stdout/stderr dong hoac None (subprocess DEVNULL) -> thay bang NullWriter
# De moi print() call o bat cu dau trong codebase khong bi crash
if not _safe_stdout():
    sys.stdout = _NullWriter()
elif sys.stdout and hasattr(sys.stdout, 'buffer'):
    # Fix Windows cp1252 encoding -> utf-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if not _safe_stderr():
    sys.stderr = _NullWriter()
elif sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# NOTE: Do NOT set WindowsSelectorEventLoopPolicy here.
# It causes PyTorch/torchaudio/pyannote to hang in worker threads (speaker diarization).
# ConnectionResetError from proactor is handled by _SuppressConnectionReset filter below.

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
# Giam log noise tu asyncio va suppress ConnectionResetError
logging.getLogger("asyncio").setLevel(logging.WARNING)


class _SuppressConnectionReset(logging.Filter):
    """Filter bo ConnectionResetError tu Windows proactor (harmless)."""
    def filter(self, record):
        if record.exc_info and record.exc_info[1]:
            if isinstance(record.exc_info[1], ConnectionResetError):
                return False
        msg = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
        if "ConnectionResetError" in msg:
            return False
        return True


_conn_reset_filter = _SuppressConnectionReset()
logging.getLogger("asyncio").addFilter(_conn_reset_filter)
logging.getLogger("uvicorn.error").addFilter(_conn_reset_filter)
logging.getLogger("uvicorn").addFilter(_conn_reset_filter)


def setup_paths():
    """Dam bao core/ va web_service/ co trong sys.path"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)


def start_server(host=None, port=None, no_gui=False):
    """Khoi dong FastAPI server"""
    setup_paths()

    from web_service.config import server_config, LOG_DIR
    from web_service.ssl_utils import ensure_ssl_certs

    # --- File log handler: setup SOM (truoc moi thu co the crash) ---
    from logging.handlers import RotatingFileHandler
    import traceback as _tb
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "server.log")
    crash_file = os.path.join(LOG_DIR, "server_crash.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    # Khi chay tu GUI (--no-gui), stdout/stderr bi redirect sang DEVNULL.
    # Xoa StreamHandler va thay sys.stdout/stderr bang NullWriter de bao ve moi
    # library (uvicorn, logging...) khoi crash khi goi isatty() hay write() tren
    # buffer da dong. Logs da co file handler nen khong mat thong tin.
    if no_gui:
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root_logger.removeHandler(h)
        sys.stdout = _NullWriter()
        sys.stderr = _NullWriter()

    logger = logging.getLogger("asr.launcher")

    try:
        # Override tu tham so
        if host:
            server_config.set_and_save("host", host)
        if port:
            server_config.set_and_save("port", str(port))

        actual_host = server_config.host
        actual_port = server_config.port

        # SSL certs (skip if HTTP mode)
        use_http = server_config.http_mode
        cert_file, key_file = (None, None) if use_http else ensure_ssl_certs()

        protocol = "http" if use_http else "https"
        logger.info(f"Starting server on {protocol}://{actual_host}:{actual_port}")
        if not use_http:
            logger.info(f"SSL cert: {cert_file}")

        # Set CPU threads
        cpu_threads = str(server_config.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = cpu_threads
        os.environ["MKL_NUM_THREADS"] = cpu_threads

        # Force transformers/HuggingFace to use ONLY cached models, no network checks.
        # Tranh truong hop AutoTokenizer.from_pretrained() hang khi kiem tra online.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Khởi tạo DB + tài khoản admin trước khi server chạy
        from web_service.database import db as _db  # noqa: F401 — trigger init
        from web_service.auth import ensure_admin, is_admin_using_default_password
        ensure_admin()
        if is_admin_using_default_password():
            logger.warning(
                "=" * 60
            )
            logger.warning("⚠️  CẢNH BÁO BẢO MẬT: Tài khoản admin đang dùng mật khẩu mặc định 'admin'!")
            logger.warning("    Vui lòng đổi mật khẩu ngay sau khi đăng nhập.")
            logger.warning("    Đăng nhập → Menu Admin → Đổi mật khẩu")
            logger.warning(
                "=" * 60
            )

        import uvicorn
        uvicorn_kwargs = dict(
            app="web_service.server:app",
            host=actual_host,
            port=actual_port,
            log_level="info",
            access_log=True,
            use_colors=False,  # tranh isatty() crash khi stdout=DEVNULL
        )
        if not use_http:
            uvicorn_kwargs["ssl_certfile"] = cert_file
            uvicorn_kwargs["ssl_keyfile"] = key_file
        uvicorn.run(**uvicorn_kwargs)

    except Exception:
        # Ghi crash log truoc khi thoat — rat quan trong khi chay headless (no-gui)
        err = _tb.format_exc()
        logger.critical(f"Server crashed during startup:\n{err}")
        try:
            with open(crash_file, "w", encoding="utf-8") as f:
                import datetime as _dt
                f.write(f"[{_dt.datetime.now()}] Server startup crash:\n{err}\n")
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Sherpa Vietnamese ASR Server")
    parser.add_argument("--host", type=str, help="Bind address (default: from config)")
    parser.add_argument("--port", type=int, help="Port (default: from config)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không có GUI")

    args = parser.parse_args()
    start_server(host=args.host, port=args.port, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
