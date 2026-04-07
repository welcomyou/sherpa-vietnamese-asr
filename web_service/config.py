"""
Web service configuration - doc/ghi config.ini section [ServerSettings]
"""

import os
import configparser

from core.config import BASE_DIR, ALLOWED_THREADS

# Duong dan mac dinh
DATA_DIR = os.path.join(BASE_DIR, "web_service", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CERTS_DIR = os.path.join(BASE_DIR, "web_service", "certs")
LOG_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "asr.db")
CONFIG_FILE = os.path.join(BASE_DIR, "config.ini")

# Dam bao thu muc ton tai
for d in [DATA_DIR, UPLOAD_DIR, CERTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


class ServerConfig:
    """Quan ly cau hinh server, doc/ghi tu config.ini [ServerSettings]"""

    DEFAULTS = {
        "host": "0.0.0.0",
        "port": "8443",
        "cpu_threads": str(min(4, ALLOWED_THREADS)),
        "max_upload_mb": "500",
        "anonymous_timeout_minutes": "120",
        "storage_per_user_gb": "5",
        "max_sessions": "100",
        "default_asr_model": "sherpa-onnx-zipformer-vi-2025-04-20",
        "default_speaker_model": "senko_campp_optimized",
        "default_punctuation_confidence": "7",
        "default_case_confidence": "6",
        "default_diarization_threshold": "70",
        "jwt_expire_minutes": "1440",
        "http_mode": "0",
        "admin_password_hash": "",
        "offline_download_url": "",
        "summarizer_model_path": "",
        "summarizer_ollama_model": "gemma4:e2b",
        "summarizer_threads": "16",
        "summarizer_context_size": "32768",
        "summarizer_enabled": "0",
    }

    def __init__(self):
        self._config = configparser.ConfigParser()
        self.load()

    def load(self):
        """Doc config.ini, tao section ServerSettings neu chua co"""
        if os.path.exists(CONFIG_FILE):
            self._config.read(CONFIG_FILE, encoding="utf-8")

        if not self._config.has_section("ServerSettings"):
            self._config.add_section("ServerSettings")

        # Dien gia tri mac dinh neu thieu
        for key, default in self.DEFAULTS.items():
            if not self._config.has_option("ServerSettings", key):
                self._config.set("ServerSettings", key, default)

        # Migration: update deprecated model IDs
        old_speaker = self._config.get("ServerSettings", "default_speaker_model", fallback="")
        if old_speaker in ("community1_onnx", "titanet_small", "campp_pure_ort", "3dspeaker_campp", "nemo_pipeline"):
            self._config.set("ServerSettings", "default_speaker_model", self.DEFAULTS["default_speaker_model"])

        old_asr = self._config.get("ServerSettings", "default_asr_model", fallback="")
        if old_asr == "zipformer-30m-rnnt-6000h":
            # Upgrade to 68M if model exists
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            if os.path.isdir(os.path.join(models_dir, self.DEFAULTS["default_asr_model"])):
                self._config.set("ServerSettings", "default_asr_model", self.DEFAULTS["default_asr_model"])

    def save(self):
        """Ghi config.ini (giu nguyen cac section khac)"""
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            self._config.write(f)

    def get(self, key: str) -> str:
        return self._config.get("ServerSettings", key, fallback=self.DEFAULTS.get(key, ""))

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def get_float(self, key: str) -> float:
        return float(self.get(key))

    def set(self, key: str, value):
        self._config.set("ServerSettings", key, str(value))

    def set_and_save(self, key: str, value):
        self.set(key, value)
        self.save()

    # --- Convenience properties ---

    @property
    def host(self) -> str:
        return self.get("host")

    @property
    def port(self) -> int:
        return self.get_int("port")

    @property
    def cpu_threads(self) -> int:
        return min(self.get_int("cpu_threads"), ALLOWED_THREADS)

    @property
    def max_upload_bytes(self) -> int:
        return self.get_int("max_upload_mb") * 1024 * 1024

    @property
    def anonymous_timeout_minutes(self) -> int:
        return self.get_int("anonymous_timeout_minutes")

    @property
    def storage_per_user_bytes(self) -> int:
        return int(self.get_float("storage_per_user_gb") * 1024 * 1024 * 1024)

    @property
    def max_sessions(self) -> int:
        return self.get_int("max_sessions")

    @property
    def jwt_expire_minutes(self) -> int:
        return self.get_int("jwt_expire_minutes")

    @property
    def http_mode(self) -> bool:
        return self.get("http_mode") == "1"

    @property
    def admin_password_hash(self) -> str:
        return self.get("admin_password_hash")

    def to_dict(self) -> dict:
        """Tra ve dict cau hinh (cho API admin/config)"""
        return {key: self.get(key) for key in self.DEFAULTS}


# Singleton
server_config = ServerConfig()
