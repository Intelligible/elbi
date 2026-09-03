"""Symmetric encryption for secrets at rest (data-source creds, later a secrets vault).

Uses Fernet (AES-128-CBC + HMAC-SHA256, per-message IV, versioned token) keyed off a
single app secret: the simple, correct default on-prem with one trust boundary. The key
comes from ``APP_SECRET_KEY``; a comma-separated list enables rotation via MultiFernet
(newest first; old ciphertext still decrypts, re-save to re-encrypt under the newest).

Encrypts connection credentials off the app secret and stays dependency-light (only
``cryptography``) rather than a heavier JWE stack for what is a single symmetric secret.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, MultiFernet

from .env import env

#: Env var holding the app secret (one key, or comma-separated newest-first to rotate).
SECRET_ENV = "APP_SECRET_KEY"  # noqa: S105 - the name of an env var, not a secret


def is_configured() -> bool:
    """Whether an app secret is set, so a secret can be stored. Guards create routes."""
    return bool((env(SECRET_ENV) or "").strip())


def _fernet() -> MultiFernet:
    raw = (env(SECRET_ENV) or "").strip()
    if not raw:
        raise RuntimeError(f"{SECRET_ENV} must be set to store or read secrets")
    # Derive a urlsafe 32-byte Fernet key from each provided secret (SHA-256), so the
    # operator can use any string (not only a Fernet key). Newest first for rotation.
    keys = [
        Fernet(base64.urlsafe_b64encode(hashlib.sha256(k.strip().encode()).digest()))
        for k in raw.split(",")
        if k.strip()
    ]
    return MultiFernet(keys)


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage; raises if ``APP_SECRET_KEY`` is unset."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a stored secret; tries each configured key (supports rotation)."""
    return _fernet().decrypt(token.encode()).decode()
