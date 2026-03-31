"""
Tu sinh HTTPS self-signed certificate.
"""

import os
import datetime
import ipaddress

from web_service.config import CERTS_DIR


def ensure_ssl_certs(cert_dir: str = None) -> tuple:
    """
    Dam bao co SSL cert. Tu sinh neu chua co.
    Returns: (cert_file_path, key_file_path)
    """
    if cert_dir is None:
        cert_dir = CERTS_DIR

    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")

    # Kiem tra cert custom (admin upload)
    custom_cert = os.path.join(cert_dir, "custom.crt")
    custom_key = os.path.join(cert_dir, "custom.key")
    if os.path.exists(custom_cert) and os.path.exists(custom_key):
        print("[SSL] Using custom certificate")
        return custom_cert, custom_key

    # Da co self-signed cert
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("[SSL] Using existing self-signed certificate")
        return cert_file, key_file

    # Tu sinh
    print("[SSL] Generating self-signed certificate...")
    os.makedirs(cert_dir, exist_ok=True)

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Tao RSA key 2048-bit
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Tao self-signed cert (10 nam)
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
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
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

    print(f"[SSL] Certificate generated: {cert_file}")
    return cert_file, key_file
