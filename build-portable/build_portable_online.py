#!/usr/bin/env python3
"""
Build script cho ban sherpa-vietnamese-asr-service (web service + admin GUI).
Dua tren build_portable.py nhung them web_service/ va loai tru streaming model.

Usage: python build-portable/build_portable_online.py
"""
import io
import hashlib
import json
import os
import re
import sys
import shutil
import stat
from pathlib import Path

# Fix stdout encoding on Windows (cp1252 -> utf-8) de print Unicode khong bi crash
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Reuse logic tu build_portable.py
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Import cac ham tu build_portable.py
sys.path.insert(0, str(SCRIPT_DIR))
from build_portable import (
    VENV_DIR, BUILD_DIR, PYTHON_EMBED_URL,
    check_venv, download_python_embedded, setup_python,
    copy_pyd_files, copy_dlls, copy_vcredist_dlls, cleanup_pyqt6, _create_stubs,
    get_venv_path, clean_build, calculate_size, ensure_ffmpeg_tools, ensure_offline_models,
)

sys.path.insert(0, str(PROJECT_ROOT))
from core.version import get_version_short
_VERSION = get_version_short()

# Override output directory
DIST_DIR_ONLINE = PROJECT_ROOT / "dist" / f"server-portable-cpu-{_VERSION}"

# Source files cho ban online (KHONG co tab_live, streaming)
ONLINE_SOURCE_FILES = [
    "server_launcher.py",
    "server_gui.py",
    "service_installer.py",
    "config.ini",
    "hotword.txt",           # hotwords cho Sherpa-ONNX
    "verb-form-vocab.txt",
    "speaker_hotkeys.json",
    "ffmpeg.exe",
    "ffprobe.exe",
    "requirements-online.txt",
]

# Service runtime model allowlist. This prevents ad-hoc downloaded models in
# models/ from being shipped in the service bundle.
BUNDLED_MODEL_DIRS_ONLINE = {
    "zipformer-30m-rnnt-6000h",
    "sherpa-onnx-zipformer-vi-2025-04-20",
    "vibert-capu",
    "pyannote",
    "pyannote-onnx",
    "campp-3dspeaker",
    "silero-vad",
    "convtasnet-libri2mix-16k",
    "dnsmos",
}
BUNDLED_MODEL_ROOT_FILES_ONLINE = set()

# Packages khong can cho web service (giam dung luong)
EXCLUDE_PACKAGES_SERVICES = {
    'moonshine_voice',
    'nvidia', 'nvidia_cublas_cu12', 'nvidia_cuda_nvrtc_cu12',
    'nvidia_cuda_runtime_cu12', 'nvidia_cudnn_cu12',
    'nvidia_cufft_cu12', 'nvidia_curand_cu12', 'nvidia_nvjitlink_cu12',
    'onnxruntime_gpu',

    # === PyTorch + ecosystem — app uses ONNX Runtime only ===
    'torch', 'torchaudio', 'torio', 'torchmetrics', 'torchgen',
    'torchcodec', 'torch_audiomentations', 'torch_pitch_shift',
    'functorch', 'ml_dtypes',
    'sympy', 'networkx',
    # jinja2 + markupsafe: GIU LAI — llama_cpp requires jinja2 for chat templates

    # === PyTorch Lightning ===
    'lightning', 'lightning_fabric', 'lightning_utilities',
    'pytorch_lightning', 'pytorch_metric_learning',

    # === Pyannote — pure ORT diarization replaces it ===
    'pyannote', 'pyannote_audio', 'pyannote_core', 'pyannote_pipeline',
    'pyannote_database', 'pyannote_metrics', 'pyannote_onnx_extended',
    'pyannoteai', 'pyannoteai_sdk',
    'asteroid', 'asteroid_filterbanks', 'julius', 'einops',

    # numba/llvmlite: GIỮ LẠI — umap-learn (Senko diarization) cần numba.njit thật
    # 'numba', 'llvmlite',

    # === Large unused packages ===
    'pandas', 'pymupdf', 'fitz', 'onnx', 'ctranslate2', 'av',

    # === Desktop-only packages ===
    'sounddevice', '_sounddevice_data',
    'matplotlib', 'mpl_toolkits', 'kiwisolver', 'contourpy', 'cycler',
    'pyparsing',

    # Auth: auth.py dung hashlib, khong dung passlib
    'passlib', 'bcrypt',
    # Database: web service dung sqlite3 truc tiep, khong can ORM
    'sqlalchemy', 'aiosqlite', 'greenlet', 'alembic', 'mako',
    # HTTP client: web service khong dung aiohttp/httpx
    'aiohttp', 'aiofiles', 'aiohappyeyeballs', 'aiosignal',
    'frozenlist', 'multidict', 'yarl', 'propcache',
    'httpcore', 'httpx',

    # === gRPC ===
    'grpc', 'grpcio', 'grpcio_tools',
    'google', 'googleapis_common_protos', 'protobuf',

    # === Monitoring / Logging ===
    'opentelemetry', 'opentelemetry_api', 'opentelemetry_sdk',
    'opentelemetry_proto', 'opentelemetry_exporter_otlp',
    'opentelemetry_exporter_otlp_proto_http',
    'opentelemetry_exporter_otlp_proto_grpc',
    'opentelemetry_exporter_otlp_proto_common',
    'opentelemetry_semantic_conventions',
    'pygments', 'rich', 'colorlog', 'colorama',

    # === transformers + deps — replaced by minimal stub ===
    'transformers',
    'huggingface_hub', 'safetensors',
    'regex', 'fsspec',  # tqdm: GIỮ LẠI — umap-learn cần tqdm.auto

    # === Text processing / Math / Misc not used ===
    'jiwer', 'langid', 'mosestokenizer', 'wtpsplit', 'textgrid',
    'markdown_it', 'mdurl',
    'optuna', 'primePy', 'mpmath', 'skops', 'prettytable',
    'adapters', 'rapidfuzz', 'chunkformer',
    'wespeakerruntime', 'silero_vad', 'faster_whisper',
    'diarize', 'nara_wpe', 'kaldiio',
    'bottleneck', 'pooch', 'flatbuffers', 'wcwidth', 'docopt',
    'openfile', 'toolwrapper',

    # === Build / packaging tools ===
    'pip', 'setuptools', 'wheel', 'pkg_resources', '_distutils_hack',
    'distutils_precedence',
    'pyinstaller', 'pyinstaller_hooks_contrib', 'altgraph', 'pefile',
}


def should_exclude_services(name):
    """Check if package or its dist-info / .libs should be excluded for service build"""
    lower = name.lower()
    base = lower.replace('.libs', '')
    for pkg in EXCLUDE_PACKAGES_SERVICES:
        pkg_under = pkg.replace('-', '_')
        if lower == pkg or base == pkg or lower == pkg_under or base == pkg_under:
            return True
        if lower.startswith(pkg + '-') or lower.startswith(pkg_under + '-'):
            return True
    return False


def copy_venv_packages_services():
    """Copy packages from virtualenv, excluding desktop-only packages"""
    print("[PKG] Copying packages from venv (services)...")

    venv_site = get_venv_path()
    dst_site = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages"
    dst_site.mkdir(parents=True, exist_ok=True)

    if not venv_site.exists():
        print(f"[ERROR] Venv site-packages not found: {venv_site}")
        return False

    copied = 0
    skipped = 0
    for item in venv_site.iterdir():
        if item.name.startswith('~'):
            continue
        if should_exclude_services(item.name):
            print(f"  [SKIP] {item.name}")
            skipped += 1
            continue
        dst = dst_site / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        try:
            if item.is_dir():
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(item, dst)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Skip {item.name}: {e}")

    # Clean .pth files that cause issues
    for pth in dst_site.glob("*pywin32*.pth"):
        try:
            pth.chmod(0o666)
            pth.unlink()
        except PermissionError:
            pth.write_text("", encoding="utf-8")
    for pth in dst_site.glob("*distutils-precedence*.pth"):
        try:
            pth.chmod(0o666)
            pth.unlink()
        except PermissionError:
            pth.write_text("", encoding="utf-8")

    # Clean CUDA DLLs from llama_cpp (server is CPU only, saves ~300MB)
    llama_lib = dst_site / "llama_cpp" / "lib"
    if llama_lib.exists():
        for cuda_dll in list(llama_lib.glob("cuda*.dll")) + list(llama_lib.glob("cublas*.dll")):
            cuda_dll.unlink()
            print(f"  [CLEAN] Removed CUDA DLL: {cuda_dll.name}")

    # Create stubs (same as desktop build)
    from build_portable import _create_stubs
    _create_stubs(dst_site)

    print(f"[OK] Copied {copied} packages, skipped {skipped}")
    return True


def copy_source_files_online():
    """Copy source files cho ban online"""
    print("[SRC] Copying online source files...")

    for f in ONLINE_SOURCE_FILES:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, DIST_DIR_ONLINE)
        else:
            print(f"  [WARN] Not found: {f}")

    # Portable build: bind 0.0.0.0 thay vì IP cố định của máy dev
    config_dst = DIST_DIR_ONLINE / "config.ini"
    if config_dst.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(config_dst, encoding="utf-8")
        if cfg.has_section("ServerSettings"):
            cfg.set("ServerSettings", "host", "0.0.0.0")
        if not cfg.has_section("OfflinePWA"):
            cfg.add_section("OfflinePWA")
        offline_defaults = {
            "enabled": "true",
            "port": "8444",
            "model_source": "bundled_server",
            "model_proxy_enabled": "false",
            "cache_version": "1",
            "max_model_download_mb": "8192",
        }
        for key, value in offline_defaults.items():
            if not cfg.has_option("OfflinePWA", key):
                cfg.set("OfflinePWA", key, value)
        if cfg.has_section("ServerSettings"):
            with open(config_dst, "w", encoding="utf-8") as f:
                cfg.write(f)
            print("  [OK] config.ini: host set to 0.0.0.0, OfflinePWA defaults added")

    # Copy core/ module
    core_src = PROJECT_ROOT / "core"
    core_dst = DIST_DIR_ONLINE / "core"
    if core_dst.exists():
        shutil.rmtree(core_dst)
    shutil.copytree(core_src, core_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("  [OK] core/ copied")

    # Copy web_service/ module
    ws_src = PROJECT_ROOT / "web_service"
    ws_dst = DIST_DIR_ONLINE / "web_service"
    if ws_dst.exists():
        shutil.rmtree(ws_dst)
    shutil.copytree(ws_src, ws_dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "data", "certs"  # certs tu generate khi chay
    ))
    # Tao thu muc data rong va certs rong
    (ws_dst / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (ws_dst / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (ws_dst / "certs").mkdir(parents=True, exist_ok=True)
    print("  [OK] web_service/ copied")

    # Copy offline_pwa/ module and static shell
    offline_src = PROJECT_ROOT / "offline_pwa"
    offline_dst = DIST_DIR_ONLINE / "offline_pwa"
    if offline_src.exists():
        if offline_dst.exists():
            shutil.rmtree(offline_dst)
        shutil.copytree(offline_src, offline_dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc"
        ))
        print("  [OK] offline_pwa/ copied")
    else:
        print("  [WARN] offline_pwa/ not found")

    # Copy shared UI assets used by both the server PWA and offline PWA.
    shared_src = PROJECT_ROOT / "shared_ui"
    shared_dst = DIST_DIR_ONLINE / "shared_ui"
    if shared_src.exists():
        if shared_dst.exists():
            shutil.rmtree(shared_dst)
        shutil.copytree(shared_src, shared_dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc"
        ))
        print("  [OK] shared_ui/ copied")
    else:
        print("  [WARN] shared_ui/ not found")

    # Copy vocabulary/
    vocab_src = PROJECT_ROOT / "vocabulary"
    if vocab_src.exists():
        vocab_dst = DIST_DIR_ONLINE / "vocabulary"
        if vocab_dst.exists():
            shutil.rmtree(vocab_dst)
        shutil.copytree(vocab_src, vocab_dst)
        print("  [OK] vocabulary/ copied")

    print("[OK] Online source files copied")


def copy_server_extras():
    """Copy ICU shim (cho Windows Server 2016), nssm.exe, bat files"""
    print("[EXTRA] Copying server extras...")

    # ICU shim DLLs (Qt6Core.dll can ICU, Windows Server 2016 khong co)
    icu_shim_dir = SCRIPT_DIR / "icu-shim"
    qt6_bin = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
    qt6_bin.mkdir(parents=True, exist_ok=True)
    icu_copied = 0
    for icu_name in ['icuuc.dll', 'icuuc73.dll', 'icuin73.dll', 'icudt73.dll']:
        src = icu_shim_dir / icu_name
        if src.exists():
            shutil.copy2(src, qt6_bin)
            icu_copied += 1
    if icu_copied >= 4:
        print("  [OK] ICU shim + ICU 73 (Windows Server compatible)")
    else:
        print(f"  [WARN] ICU shim khong day du ({icu_copied}/4) trong {icu_shim_dir}")

    # nssm.exe (cai Windows Service)
    nssm_src = SCRIPT_DIR / "nssm.exe"
    if nssm_src.exists():
        shutil.copy2(nssm_src, DIST_DIR_ONLINE)
        print("  [OK] nssm.exe")
    else:
        print("  [WARN] nssm.exe khong tim thay")

    # Bat files
    for bat_name in ['start-gui.bat', 'start-server.bat', 'install-service.bat']:
        bat_src = SCRIPT_DIR / "server-bats" / bat_name
        if bat_src.exists():
            shutil.copy2(bat_src, DIST_DIR_ONLINE)
            print(f"  [OK] {bat_name}")

    print("[OK] Server extras copied")


def copy_models_online():
    """Copy models, loai tru unused models"""
    print("[MODEL] Copying models...")

    models_src = PROJECT_ROOT / "models"
    models_dst = DIST_DIR_ONLINE / "models"

    if not models_src.exists():
        print("  [WARN] models/ not found")
        return

    models_dst.mkdir(parents=True, exist_ok=True)

    for item in models_src.iterdir():
        if item.is_dir():
            if item.name not in BUNDLED_MODEL_DIRS_ONLINE:
                print(f"  [SKIP] {item.name}")
                continue
            dst_item = models_dst / item.name
            shutil.copytree(item, dst_item, ignore=shutil.ignore_patterns(
                            '.git', '.cache', '.gitattributes', '__pycache__'))
        elif item.is_file():
            if item.name not in BUNDLED_MODEL_ROOT_FILES_ONLINE:
                print(f"  [SKIP] {item.name}")
                continue
            shutil.copy2(item, models_dst / item.name)
        else:
            print(f"  [SKIP] {item.name}")
            continue
        print(f"  [OK] {item.name}")

    # Clean vibert-capu: keep only onnx + vocab + config
    vibert_dst = models_dst / "vibert-capu"
    if vibert_dst.exists():
        VIBERT_KEEP = {'vibert-capu.onnx', 'vocab.txt', 'config.json'}
        removed = 0
        for f in list(vibert_dst.rglob('*')):
            if f.is_file() and f.name not in VIBERT_KEEP:
                removed += f.stat().st_size / 1024 / 1024
                f.unlink()
        for d in sorted(vibert_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] vibert-capu: removed {removed:.0f} MB")

    # Clean pyannote: keep only plda/ (remove pytorch_model.bin)
    pyannote_dst = models_dst / "pyannote" / "speaker-diarization-community-1"
    if pyannote_dst.exists():
        removed = 0
        for f in list(pyannote_dst.rglob('*.bin')):
            removed += f.stat().st_size / 1024 / 1024
            f.unlink()
        for d in sorted(pyannote_dst.rglob('*'), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if removed > 0:
            print(f"  [TRIM] pyannote: removed {removed:.0f} MB pytorch_model.bin")

    # Global cleanup: remove non-ONNX artifacts from ALL model dirs
    JUNK_EXTS = {'.bin', '.pt', '.pth', '.pkl', '.safetensors', '.filepart',
                 '.md', '.gitattributes'}
    removed_total = 0
    for f in list(models_dst.rglob('*')):
        if f.is_file() and f.suffix.lower() in JUNK_EXTS:
            size_mb = f.stat().st_size / 1024 / 1024
            removed_total += size_mb
            print(f"  [DEL] {f.relative_to(models_dst)} ({size_mb:.1f} MB)")
            f.unlink()
    # Remove int8 duplicates when fp32 exists
    for model_dir in models_dst.iterdir():
        if not model_dir.is_dir():
            continue
        for int8f in list(model_dir.glob('*int8*.onnx')):
            fp32_name = int8f.name.replace('.int8', '')
            if (model_dir / fp32_name).exists():
                size_mb = int8f.stat().st_size / 1024 / 1024
                removed_total += size_mb
                print(f"  [DEL] {int8f.relative_to(models_dst)} ({size_mb:.1f} MB) — fp32 exists")
                int8f.unlink()
    for name in ['embedding_model.onnx', 'embedding_model_split.onnx']:
        p = models_dst / 'pyannote-onnx' / name
        if p.exists() and (models_dst / 'pyannote-onnx' / 'embedding_encoder.onnx').exists():
            size_mb = p.stat().st_size / 1024 / 1024
            removed_total += size_mb
            print(f"  [DEL] pyannote-onnx/{name} ({size_mb:.1f} MB)")
            p.unlink()
    for d in sorted(models_dst.rglob('*'), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if removed_total > 0:
        print(f"  [TRIM] Cleaned {removed_total:.0f} MB non-ONNX artifacts")

    count = sum(1 for _ in models_dst.rglob("*") if _.is_file())
    print(f"[OK] Models copied: {count} files")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_bpe_vocab(model_dir: Path) -> None:
    bpe_model = model_dir / "bpe.model"
    bpe_vocab = model_dir / "bpe.vocab"
    if bpe_vocab.exists() or not bpe_model.exists():
        return

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(bpe_model))
    eol = "\r\n"
    lines = [
        f"{processor.IdToPiece(index)}\t{processor.GetScore(index)}"
        for index in range(processor.GetPieceSize())
    ]
    bpe_vocab.write_bytes((eol.join(lines) + eol).encode("utf-8"))
    print(f"  [GEN] {bpe_vocab.relative_to(DIST_DIR_ONLINE)}")


def _generate_plda_prepared(model_dir: Path) -> None:
    plda_dir = model_dir / "plda"
    target = plda_dir / "plda_prepared.npz"
    if target.exists():
        return
    plda_path = plda_dir / "plda.npz"
    xvec_path = plda_dir / "xvec_transform.npz"
    if not plda_path.exists() or not xvec_path.exists():
        return

    import numpy as np
    from scipy.linalg import eigh

    xvec = np.load(xvec_path)
    plda = np.load(plda_path)
    mean1, mean2, lda = xvec["mean1"], xvec["mean2"], xvec["lda"]
    mu, tr, psi = plda["mu"], plda["tr"], plda["psi"]
    w_matrix = np.linalg.inv(tr.T @ tr)
    b_matrix = np.linalg.inv((tr.T / psi) @ tr)
    acvar, wccn = eigh(b_matrix, w_matrix)

    np.savez(
        target,
        mean1=mean1,
        mean2=mean2,
        lda=lda,
        mu=mu,
        plda_tr=wccn.T[::-1],
        plda_psi=acvar[::-1],
    )
    print(f"  [GEN] {target.relative_to(DIST_DIR_ONLINE)}")


def refresh_generated_model_manifest_metadata(file_ids: set[str]) -> None:
    manifest_path = DIST_DIR_ONLINE / "offline_pwa" / "model_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = False
    for pack in manifest.get("packs", []):
        for item in pack.get("files", []):
            if item.get("id") not in file_ids:
                continue
            rel_path = item.get("local_path") or item.get("target_path")
            if not rel_path:
                continue
            path = DIST_DIR_ONLINE / rel_path
            if not path.is_file():
                continue
            item["bytes"] = path.stat().st_size
            item["sha256"] = _sha256_file(path)
            changed = True

    if changed:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("  [OK] Refreshed generated model metadata in offline_pwa/model_manifest.json")


def ensure_offline_pwa_generated_model_assets():
    print("[PWA] Ensuring generated model assets...")
    models_root = DIST_DIR_ONLINE / "models"
    _generate_bpe_vocab(models_root / "zipformer-30m-rnnt-6000h")
    _generate_bpe_vocab(models_root / "sherpa-onnx-zipformer-vi-2025-04-20")
    _generate_plda_prepared(models_root / "pyannote" / "speaker-diarization-community-1")
    refresh_generated_model_manifest_metadata({
        "asr30.bpe_vocab",
        "asr68.bpe_vocab",
        "speaker.pyannote_plda_prepared",
    })


def validate_offline_pwa_model_bundle():
    """Fail the server build if the PWA manifest references missing local models."""
    manifest_path = DIST_DIR_ONLINE / "offline_pwa" / "model_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("offline_pwa/model_manifest.json missing from portable build")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models_root = (DIST_DIR_ONLINE / "models").resolve()
    errors = []
    checked = 0
    for pack in manifest.get("packs", []):
        if pack.get("required") is False or pack.get("optional") is True:
            continue
        for item in pack.get("files", []):
            checked += 1
            rel_path = item.get("local_path") or item.get("target_path")
            file_id = item.get("id", "<unknown>")
            if not rel_path:
                errors.append(f"{file_id}: missing local_path")
                continue
            candidate = (DIST_DIR_ONLINE / rel_path).resolve()
            try:
                candidate.relative_to(models_root)
            except ValueError:
                errors.append(f"{file_id}: local_path outside models/: {rel_path}")
                continue
            if not candidate.is_file():
                errors.append(f"{file_id}: missing bundled file {rel_path}")
                continue
            expected = item.get("bytes")
            size = candidate.stat().st_size
            if expected and size != expected:
                errors.append(f"{file_id}: size mismatch {size} != {expected} ({rel_path})")
            expected_sha = str(item.get("sha256") or "").strip().lower()
            if not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
                errors.append(f"{file_id}: missing sha256 in manifest")
                continue
            sha = hashlib.sha256()
            with open(candidate, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(chunk)
            actual_sha = sha.hexdigest()
            if actual_sha != expected_sha:
                errors.append(f"{file_id}: sha256 mismatch {actual_sha} != {expected_sha} ({rel_path})")

    if errors:
        print("[ERR] Offline PWA server bundle is incomplete:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError("Offline PWA model bundle validation failed")
    print(f"[OK] Offline PWA model bundle validated: {checked} manifest file(s)")


def _js_string_array_values(text: str, const_name: str) -> list[str]:
    match = re.search(
        rf"const\s+{re.escape(const_name)}\s*=\s*(?:Object\.freeze\()?\[([\s\S]*?)\]\)?\s*;",
        text,
    )
    if not match:
        return []
    return re.findall(r"""["']([^"']+)["']""", match.group(1))


def _offline_pwa_url_to_file(url: str) -> Path | None:
    """Map same-origin PWA runtime URLs to files inside the portable bundle."""
    if url in ("", "/"):
        return DIST_DIR_ONLINE / "offline_pwa" / "static" / "index.html"
    if url == "/api/model-manifest":
        return DIST_DIR_ONLINE / "offline_pwa" / "model_manifest.json"
    if url.startswith("/api/"):
        return None
    if url.startswith("/shared/"):
        return DIST_DIR_ONLINE / "shared_ui" / "static" / url[len("/shared/"):]
    return DIST_DIR_ONLINE / "offline_pwa" / "static" / url.lstrip("/")


def validate_offline_pwa_static_bundle():
    """Fail if the server portable build misses any PWA runtime/cache asset."""
    app_js = DIST_DIR_ONLINE / "offline_pwa" / "static" / "js" / "app.js"
    sw_js = DIST_DIR_ONLINE / "offline_pwa" / "static" / "sw.js"
    if not app_js.exists():
        raise RuntimeError("offline_pwa/static/js/app.js missing from portable build")
    if not sw_js.exists():
        raise RuntimeError("offline_pwa/static/sw.js missing from portable build")

    app_text = app_js.read_text(encoding="utf-8")
    sw_text = sw_js.read_text(encoding="utf-8")
    urls = set(_js_string_array_values(app_text, "OFFLINE_RUNTIME_ASSET_URLS"))
    urls.update(_js_string_array_values(sw_text, "CORE_SHELL"))
    urls.update(_js_string_array_values(sw_text, "APP_SHELL"))

    missing = []
    checked = 0
    for url in sorted(urls):
        if not url.startswith("/"):
            continue
        path = _offline_pwa_url_to_file(url)
        if path is None:
            continue
        checked += 1
        if not path.is_file():
            missing.append(f"{url} -> {path.relative_to(DIST_DIR_ONLINE)}")

    if missing:
        print("[ERR] Offline PWA static bundle is incomplete:")
        for item in missing:
            print(f"  - {item}")
        raise RuntimeError("Offline PWA static bundle validation failed")
    print(f"[OK] Offline PWA static bundle validated: {checked} runtime asset(s)")


def create_launcher_online():
    """Tao launcher batch cho ban services"""
    print("[LNCH] Creating service launcher...")

    # Server launcher
    bat_content = (
        '@echo off\n'
        'chcp 65001 >nul\n'
        'setlocal\n'
        '\n'
        'set "BASE_DIR=%~dp0"\n'
        'set "PYTHON_EXE=%BASE_DIR%python\\python.exe"\n'
        '\n'
        'if not exist "%PYTHON_EXE%" (\n'
        '    echo ERROR: Khong tim thay Python embedded\n'
        '    pause\n'
        '    exit /b 1\n'
        ')\n'
        '\n'
        'set "PYTHONHOME=%BASE_DIR%python"\n'
        'set "PYTHONDONTWRITEBYTECODE=1"\n'
        'set "QT6_BIN=%BASE_DIR%python\\Lib\\site-packages\\PyQt6\\Qt6\\bin"\n'
        'set "PATH=%QT6_BIN%;%BASE_DIR%python;%BASE_DIR%python\\Lib\\site-packages;%BASE_DIR%;%PATH%"\n'
        '\n'
        'echo ===================================\n'
        'echo  Sherpa Vietnamese ASR - Service\n'
        'echo ===================================\n'
        'echo.\n'
        '\n'
        'rem Doc host/port dung section tu config.ini\n'
        'set "HOST=0.0.0.0"\n'
        'set "PORT=8443"\n'
        'set "HTTP_MODE=0"\n'
        'set "PWA_ENABLED=1"\n'
        'set "PWA_PORT=8444"\n'
        'for /f "tokens=1,* delims==" %%A in (\'"%PYTHON_EXE%" -c "import configparser, os; c=configparser.ConfigParser(); c.read(os.path.join(os.environ[\'BASE_DIR\'], \'config.ini\'), encoding=\'utf-8\'); s=c[\'ServerSettings\'] if c.has_section(\'ServerSettings\') else {}; p=c[\'OfflinePWA\'] if c.has_section(\'OfflinePWA\') else {}; v=p.get(\'enabled\',\'true\').strip().lower(); print(\'HOST=\'+s.get(\'host\',\'0.0.0.0\')); print(\'PORT=\'+s.get(\'port\',\'8443\')); print(\'HTTP_MODE=\'+s.get(\'http_mode\',\'0\')); print(\'PWA_ENABLED=\'+(\'1\' if v in (\'1\',\'true\',\'yes\',\'on\') else \'0\')); print(\'PWA_PORT=\'+p.get(\'port\',\'8444\'))" 2^>nul\') do set "%%A=%%B"\n'
        'if "%HTTP_MODE%"=="1" (set "PROTO=http") else (set "PROTO=https")\n'
        '\n'
        'if "%1"=="--no-gui" goto :headless\n'
        '\n'
        'rem Thu khoi dong GUI (can Windows 10 1809+ / Server 2019+)\n'
        '"%PYTHON_EXE%" -c "from PyQt6.QtWidgets import QApplication" >nul 2>&1\n'
        'if %errorlevel% equ 0 (\n'
        '    echo Khoi dong Admin GUI...\n'
        '    "%PYTHON_EXE%" "%BASE_DIR%server_gui.py" %*\n'
        '    goto :done\n'
        ')\n'
        '\n'
        'echo [Thong bao] Giao dien GUI can Windows 10 1809+ / Server 2019+\n'
        'echo Tu dong chuyen sang che do headless...\n'
        'echo.\n'
        '\n'
        ':headless\n'
        'echo Server dang chay. Truy cap:\n'
        'echo.\n'
        'if "%HOST%"=="0.0.0.0" (\n'
        '    echo   %PROTO%://localhost:%PORT%\n'
        '    echo   %PROTO%://[IP-may-nay]:%PORT%\n'
        '    if "%PWA_ENABLED%"=="1" (\n'
        '        echo.\n'
        '        echo PWA offline:\n'
        '        echo   %PROTO%://localhost:%PWA_PORT%\n'
        '        echo   %PROTO%://[IP-may-nay]:%PWA_PORT%\n'
        '    )\n'
        ') else (\n'
        '    echo   %PROTO%://%HOST%:%PORT%\n'
        '    if "%PWA_ENABLED%"=="1" (\n'
        '        echo.\n'
        '        echo PWA offline:\n'
        '        echo   %PROTO%://%HOST%:%PWA_PORT%\n'
        '    )\n'
        ')\n'
        'echo.\n'
        'echo Dang nhap admin de quan tri he thong qua web.\n'
        'echo Nhan Ctrl+C de dung server.\n'
        'echo -------------------------------------------------------\n'
        'echo.\n'
        '"%PYTHON_EXE%" "%BASE_DIR%server_launcher.py" --no-gui\n'
        '\n'
        ':done\n'
        'if %errorlevel% neq 0 (\n'
        '    echo.\n'
        '    echo [Loi] Chuong trinh ket thuc voi ma loi %errorlevel%\n'
        '    pause\n'
        ')\n'
        '\n'
        'exit /b %errorlevel%\n'
    )

    (DIST_DIR_ONLINE / "sherpa-vietnamese-asr-service.bat").write_text(bat_content, encoding="utf-8")

    # Copy bat templates từ build-portable/server-bats/
    bats_template_dir = SCRIPT_DIR / "server-bats"
    for bat_name in ("start-gui.bat", "start-server.bat", "install-service.bat"):
        src = bats_template_dir / bat_name
        if src.exists():
            (DIST_DIR_ONLINE / bat_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # README
    readme = '''Sherpa Vietnamese ASR - Service
================================

Cac file khoi dong:
  start-server.bat      -> Chay server headless (moi Windows)
  start-gui.bat         -> Mo GUI quan tri (can Windows 10 1809+ / Server 2019+)
  install-service.bat   -> Cai dat thanh Windows Service (can quyen Admin)

Yeu cau:
- Windows 10+ / Windows Server 2016+ (64-bit)
- Khong can cai Python

Sau khi chay:
- Truy cap https://IP:8443 tren browser
- PWA offline duoc bat kem server neu [OfflinePWA] enabled=true, mac dinh https://IP:8444
- Cho may dien thoai truy cap URL PWA offline lan dau de cai app va tai model tu server
- Admin mac dinh: admin / admin (doi ngay sau khi dang nhap)
- Dang nhap admin -> nut "Quan tri" hien ra -> quan ly phien, queue, user qua web
- Browser se canh bao "Not Secure" vi self-signed cert, bam "Advanced" -> "Proceed"
'''
    (DIST_DIR_ONLINE / "README.txt").write_text(readme, encoding="utf-8")

    print("[OK] Services launcher created")


def main():
    global DIST_DIR_ONLINE

    print("=" * 60)
    print("Sherpa Vietnamese ASR - Service Portable Build")
    print("=" * 60)
    print()

    if not check_venv():
        return 1
    ensure_ffmpeg_tools()
    ensure_offline_models()

    # Clean old build
    print("[CLEAN] Cleaning old service build...")
    if DIST_DIR_ONLINE.exists():
        def on_rm_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(DIST_DIR_ONLINE, onerror=on_rm_error)

    # Also clean old-named directory if exists
    old_dir = PROJECT_ROOT / "dist" / "sherpa-asr-vn-online"
    if old_dir.exists():
        print("[CLEAN] Removing old sherpa-asr-vn-online...")
        shutil.rmtree(old_dir, onerror=lambda f, p, e: (os.chmod(p, stat.S_IWRITE), f(p)))

    DIST_DIR_ONLINE.mkdir(parents=True)

    try:
        # Tam thoi override DIST_DIR trong build_portable module
        import build_portable
        original_dist = build_portable.DIST_DIR
        build_portable.DIST_DIR = DIST_DIR_ONLINE

        # Download and setup Python
        zip_file = download_python_embedded()
        setup_python(zip_file)

        # Copy from venv (with services-specific exclusions)
        if not copy_venv_packages_services():
            return 1
        copy_pyd_files()
        copy_dlls()
        copy_vcredist_dlls()  # Copy VC++ Redist DLLs for PyQt6 on target machines

        # Trim unused Qt6 modules
        cleanup_pyqt6()

        # Restore
        build_portable.DIST_DIR = original_dist

        # Copy online-specific files
        copy_source_files_online()
        copy_server_extras()  # ICU shim, nssm.exe, bat files (SAU cleanup_pyqt6)
        copy_models_online()

        # Write VERSION file (auto from git)
        print("[VER] Writing VERSION file...")
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from core.version import get_version
            version = get_version()
            (DIST_DIR_ONLINE / "VERSION").write_text(version, encoding='utf-8')
            print(f"[OK] VERSION = {version}")
        except Exception as e:
            print(f"[WARN] Cannot determine version: {e}")
            (DIST_DIR_ONLINE / "VERSION").write_text("unknown", encoding='utf-8')

        create_launcher_online()

        # Trim (reuse from desktop build)
        original_dist2 = build_portable.DIST_DIR
        build_portable.DIST_DIR = DIST_DIR_ONLINE
        build_portable.trim_portable()
        build_portable.DIST_DIR = original_dist2

        clean_build()

        # Xóa .opt files — máy target sẽ tự tạo lần đầu chạy (phụ thuộc ORT version + CPU)
        for opt_file in DIST_DIR_ONLINE.rglob("*.opt"):
            opt_file.unlink()
            print(f"  [DEL] {opt_file.relative_to(DIST_DIR_ONLINE)} (ORT cache, auto-generated on target)")

        ensure_offline_pwa_generated_model_assets()
        validate_offline_pwa_model_bundle()
        validate_offline_pwa_static_bundle()

        # Post-build validation
        print()
        print("[CHECK] Validating build...")
        critical = [
            "python/python.exe",
            "python/sitecustomize.py",
            "python/python312._pth",
            "server_launcher.py",
            "server_gui.py",
            "config.ini",
            "ffmpeg.exe",
            "ffprobe.exe",
            "core/__init__.py",
            "core/config.py",
            "core/asr_engine.py",
            "core/speaker_diarization.py",
            "core/speaker_diarization_pure_ort.py",
            "core/speaker_diarization_senko_campp.py",
            "core/speaker_diarization_senko_campp_optimized.py",
            "web_service/__init__.py",
            "web_service/server.py",
            "web_service/config.py",
            "web_service/database.py",
            "web_service/auth.py",
            "web_service/session_manager.py",
            "web_service/queue_manager.py",
            "web_service/ssl_utils.py",
            "web_service/audio_quality.py",
            "web_service/static/index.html",
            "web_service/static/js/app.js",
            "web_service/static/manifest.json",
            "offline_pwa/__init__.py",
            "offline_pwa/server.py",
            "offline_pwa/config.py",
            "offline_pwa/model_manifest.json",
            "offline_pwa/static/index.html",
            "offline_pwa/static/hotword.txt",
            "offline_pwa/static/js/app.js",
            "offline_pwa/static/js/asr-worker.js",
            "offline_pwa/static/js/ffmpeg-decode-worker.js",
            "offline_pwa/static/js/pure-ort-asr-worker.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/classes.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/const.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/errors.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/index.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/types.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/utils.js",
            "offline_pwa/static/vendor/ffmpeg/ffmpeg/worker.js",
            "offline_pwa/static/vendor/ffmpeg/core/ffmpeg-core.js",
            "offline_pwa/static/vendor/ffmpeg/core/ffmpeg-core.wasm",
            "offline_pwa/static/vendor/onnxruntime-web/ort.wasm.min.js",
            "offline_pwa/static/vendor/onnxruntime-web/ort.webgpu.min.js",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm",
            "offline_pwa/static/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs",
            "offline_pwa/static/vendor/longform-clustering/longform-clustering.js",
            "offline_pwa/static/vendor/mpg123-decoder/mpg123-decoder.min.js",
            "offline_pwa/static/vendor/zstd-wasm/zstd-wrapper.js",
            "offline_pwa/static/vendor/zstd-wasm/zstd.js",
            "offline_pwa/static/vendor/zstd-wasm/zstd.wasm",
            "offline_pwa/static/vendor/sherpa-onnx-wasm/sherpa-onnx-asr.js",
            "offline_pwa/static/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.js",
            "offline_pwa/static/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.wasm",
            "offline_pwa/static/css/app.css",
            "offline_pwa/static/icons/icon-192.png",
            "offline_pwa/static/icons/icon-512.png",
            "offline_pwa/static/calibration/1hour_qh_10min.mp3",
            "offline_pwa/static/manifest.json",
            "offline_pwa/static/sw.js",
            "shared_ui/static/css/style.css",
            "shared_ui/static/js/about.js",
            "shared_ui/static/js/status.js",
            "sherpa-vietnamese-asr-service.bat",
        ]
        # Check required packages
        pkg_dir = DIST_DIR_ONLINE / "python" / "Lib" / "site-packages"
        required_pkgs = [
            "fastapi", "uvicorn", "starlette", "sherpa_onnx",
            "numpy", "onnxruntime", "cryptography",
            "jose",
        ]
        missing = []
        for f in critical:
            if not (DIST_DIR_ONLINE / f).exists():
                missing.append(f"  [MISS] {f}")
        for pkg in required_pkgs:
            if not (pkg_dir / pkg).exists():
                missing.append(f"  [MISS] package: {pkg}")

        if missing:
            print("[WARN] Missing files/packages:")
            for m in missing:
                print(m)
        else:
            print("[OK] All critical files and packages present")

        # Report
        total = sum(f.stat().st_size for f in DIST_DIR_ONLINE.rglob("*") if f.is_file())
        size_gb = total / 1024**3

        print()
        print("=" * 60)
        print("BUILD SUCCESS!")
        print("=" * 60)
        print(f"Location: {DIST_DIR_ONLINE.absolute()}")
        print(f"Run:      {DIST_DIR_ONLINE / 'sherpa-vietnamese-asr-service.bat'}")
        print(f"Size:     {size_gb:.2f} GB")

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"BUILD FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
