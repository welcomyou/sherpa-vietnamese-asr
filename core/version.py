"""
Auto-versioning từ git tags (SemVer).

Cách dùng:
    from core.version import get_version
    version = get_version()   # "2.1.0" hoặc "2.1.0+3.a3f8c0e"

Cách đánh số:
    git tag v2.1.0           → build ra "2.1.0"        (release chính thức)
    commit thêm 3 lần        → build ra "2.1.0+3.a3f8c" (dev build)
    git tag v2.2.0           → build ra "2.2.0"        (release mới)

    MAJOR  — thay đổi lớn, breaking change
    MINOR  — tính năng mới
    PATCH  — sửa bug, chỉnh nhỏ
    +BUILD — tự động: số commits sau tag + git hash (chỉ khi dev build)

Portable build:
    Build script ghi version vào VERSION file trong dist/.
    Khi chạy portable (không có .git), đọc từ VERSION file.
"""
import os
import subprocess

# Fallback nếu không có git và không có VERSION file
_FALLBACK_VERSION = "2.1.0"

_cached_version = None


def get_version():
    """Trả về version string. Cache sau lần gọi đầu."""
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    # 1. Thử đọc từ VERSION file (portable build)
    version = _read_version_file()
    if version:
        _cached_version = version
        return version

    # 2. Thử đọc từ git
    version = _read_git_version()
    if version:
        _cached_version = version
        return version

    # 3. Fallback
    _cached_version = _FALLBACK_VERSION
    return _cached_version


def _read_version_file():
    """Đọc VERSION file (tồn tại trong portable build)."""
    # Tìm VERSION ở thư mục gốc project (parent của core/)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    version_file = os.path.join(base, "VERSION")
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                v = f.read().strip()
            if v:
                return v
        except OSError:
            pass
    return None


def _read_git_version():
    """Đọc version từ git describe."""
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ['git', 'describe', '--tags', '--long', '--match', 'v*'],
            capture_output=True, text=True, cwd=base, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )
        if result.returncode != 0:
            return None

        # Parse: "v2.1.0-3-ga3f8c0e" → "2.1.0+3.a3f8c0e"
        desc = result.stdout.strip()
        if not desc.startswith('v'):
            return None

        # v2.1.0-0-g78e03f0 → parts = ["2.1.0", "0", "g78e03f0"]
        desc = desc[1:]  # strip 'v'
        parts = desc.rsplit('-', 2)
        if len(parts) != 3:
            return None

        version_tag = parts[0]      # "2.1.0"
        commits_after = parts[1]    # "0" hoặc "3"
        git_hash = parts[2][1:]     # "78e03f0" (strip 'g')

        if commits_after == '0':
            # Đúng tag → clean version
            return version_tag
        else:
            # Dev build → version+commits.hash
            short_hash = git_hash[:7]
            return f"{version_tag}+{commits_after}.{short_hash}"

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_version_short():
    """Version ngắn gọn, bỏ build info. VD: "2.1.0" (kể cả dev build)."""
    v = get_version()
    return v.split('+')[0]


def get_build_info():
    """Trả về dict chi tiết."""
    v = get_version()
    parts = v.split('+')
    info = {
        'version': parts[0],
        'full': v,
        'is_release': len(parts) == 1,
    }
    if len(parts) > 1:
        build = parts[1].split('.')
        info['commits_after'] = int(build[0]) if build[0].isdigit() else 0
        info['git_hash'] = build[1] if len(build) > 1 else ''
    return info
