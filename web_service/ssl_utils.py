"""
Tu sinh HTTPS self-signed certificate.
"""

import os
import logging
import datetime
import ipaddress
import shutil

from web_service.config import CERTS_DIR, DATA_DIR

logger = logging.getLogger("asr.ssl")

ACTIVE_CERT_ENV = "SHERPA_ASR_ACTIVE_CERT_FILE"
ACTIVE_KEY_ENV = "SHERPA_ASR_ACTIVE_KEY_FILE"
ACTIVE_CERT_SNAPSHOT = os.path.join(DATA_DIR, "active_tls_cert.crt")


def _configured_cert_pair(cert_dir: str = None) -> tuple[str | None, str | None]:
    """Return the cert/key pair selected by runtime priority, without generating."""
    if cert_dir is None:
        cert_dir = CERTS_DIR

    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")
    custom_cert = os.path.join(cert_dir, "custom.crt")
    custom_key = os.path.join(cert_dir, "custom.key")

    if os.path.exists(custom_cert) and os.path.exists(custom_key):
        return custom_cert, custom_key
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file
    return None, None


def publish_active_ssl_cert(cert_file: str | None, key_file: str | None = None) -> str | None:
    """Expose the exact public cert loaded at server startup for /install-cert.

    Uvicorn reads cert/key files only when TLS starts. If the user later imports
    or regenerates cert files before restarting, the files on disk may no longer
    match the live TLS certificate. Keep a public-cert snapshot so downloads
    stay aligned with the currently running server.
    """
    if not cert_file:
        os.environ.pop(ACTIVE_CERT_ENV, None)
        os.environ.pop(ACTIVE_KEY_ENV, None)
        return None

    cert_file = os.path.abspath(cert_file)
    active_cert = cert_file
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copy2(cert_file, ACTIVE_CERT_SNAPSHOT)
        active_cert = os.path.abspath(ACTIVE_CERT_SNAPSHOT)
    except OSError as exc:
        logger.warning("[SSL] Could not snapshot active certificate: %s", exc)

    os.environ[ACTIVE_CERT_ENV] = active_cert
    if key_file:
        os.environ[ACTIVE_KEY_ENV] = os.path.abspath(key_file)
    else:
        os.environ.pop(ACTIVE_KEY_ENV, None)
    return active_cert


def get_install_cert_path(cert_dir: str = None, generate_if_missing: bool = False) -> str | None:
    """Return the certificate that clients should install for the active server."""
    active_cert = os.environ.get(ACTIVE_CERT_ENV)
    if active_cert and os.path.exists(active_cert):
        return active_cert

    cert_file, _ = _configured_cert_pair(cert_dir)
    if cert_file:
        return cert_file

    if generate_if_missing:
        cert_file, key_file = ensure_ssl_certs(cert_dir)
        return publish_active_ssl_cert(cert_file, key_file) or cert_file

    return None


def ensure_ssl_certs(cert_dir: str = None) -> tuple:
    """
    Dam bao co SSL cert. Tu sinh neu chua co.
    Returns: (cert_file_path, key_file_path)
    """
    if cert_dir is None:
        cert_dir = CERTS_DIR

    existing_cert, existing_key = _configured_cert_pair(cert_dir)
    if existing_cert and existing_key:
        if os.path.basename(existing_cert) == "custom.crt":
            logger.info("[SSL] Using custom certificate")
        else:
            logger.info("[SSL] Using existing self-signed certificate")
        return existing_cert, existing_key

    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")

    # Tu sinh
    logger.info("[SSL] Generating self-signed certificate...")
    os.makedirs(cert_dir, exist_ok=True)

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # A02: RSA 3072-bit (NIST recommended after 2030), cert 2 năm thay vì 10 năm
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)

    # Tao self-signed cert (2 nam)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Sherpa Vietnamese ASR"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ASR VN"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=730))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Luu key (restrict permissions trước khi ghi)
    import stat
    try:
        # Tạo file trước, set permissions restrictive (owner-only)
        if os.path.exists(key_file):
            os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows không hỗ trợ đầy đủ Unix permissions

    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    try:
        os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass

    # Luu cert
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"[SSL] Certificate generated: {cert_file}")
    return cert_file, key_file
