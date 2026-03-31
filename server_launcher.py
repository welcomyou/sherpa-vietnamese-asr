"""
Entry point cho sherpa-vietnamese-asr-services.
Khoi dong FastAPI server voi HTTPS.
"""

import io
import os
import sys
import argparse
import logging

# Fix Windows cp1252 encoding: đảm bảo print() hỗ trợ Unicode (tiếng Việt)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
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

    from web_service.config import server_config
    from web_service.ssl_utils import ensure_ssl_certs

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

    # Log file (with rotation: 10MB x 5 files)
    from logging.handlers import RotatingFileHandler
    from web_service.config import LOG_DIR
    log_file = os.path.join(LOG_DIR, "server.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    # Khi chay tu GUI (--no-gui), stdout/stderr bi redirect sang DEVNULL hoac pipe.
    # Xoa StreamHandler de tranh block khi pipe buffer day.
    if no_gui:
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root_logger.removeHandler(h)

    logger = logging.getLogger("asr.launcher")
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

    import uvicorn
    uvicorn_kwargs = dict(
        app="web_service.server:app",
        host=actual_host,
        port=actual_port,
        log_level="info",
        access_log=True,
    )
    if not use_http:
        uvicorn_kwargs["ssl_certfile"] = cert_file
        uvicorn_kwargs["ssl_keyfile"] = key_file
    uvicorn.run(**uvicorn_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Sherpa Vietnamese ASR Server")
    parser.add_argument("--host", type=str, help="Bind address (default: from config)")
    parser.add_argument("--port", type=int, help="Port (default: from config)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không có GUI")

    args = parser.parse_args()
    start_server(host=args.host, port=args.port, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
