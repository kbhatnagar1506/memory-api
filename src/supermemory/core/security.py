"""API key generation, hashing and verification.

Keys are never stored. What is stored is a peppered SHA-256 of the key, and
lookup is by that hash, which makes verification a single indexed read rather
than a scan-and-compare over every key in the table.

Why SHA-256 and not bcrypt/argon2: API keys are 256 bits of CSPRNG output, not
user-chosen passwords. There is no dictionary to attack and no rainbow table
that helps, so a slow KDF buys nothing while adding tens of milliseconds to
every single request. The pepper defends the case that matters — a database
leak without the application secret.

Comparison is constant-time regardless, because the cost of getting that wrong
is a timing oracle and the cost of getting it right is nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from ..domain.models import ApiKey, Scope, utcnow

#: Human-visible prefix. Makes a leaked key greppable in logs and repos, which
#: is how secret-scanning tools find them before an attacker does.
KEY_PREFIX = "sm_"
#: 32 bytes of entropy, urlsafe-base64 encoded.
_KEY_BYTES = 32
#: Characters retained for display, e.g. "sm_a1b2c3d4".
_DISPLAY_CHARS = 11


@dataclass(frozen=True, slots=True)
class GeneratedKey:
    """A freshly minted key. `plaintext` is returned to the caller exactly once."""

    plaintext: str
    key_hash: str
    prefix: str


def generate_key(pepper: str) -> GeneratedKey:
    raw = secrets.token_urlsafe(_KEY_BYTES)
    plaintext = f"{KEY_PREFIX}{raw}"
    return GeneratedKey(
        plaintext=plaintext,
        key_hash=hash_key(plaintext, pepper),
        prefix=plaintext[:_DISPLAY_CHARS],
    )


def hash_key(plaintext: str, pepper: str) -> str:
    """Peppered SHA-256. Deterministic, so lookup is an indexed equality match."""
    return hashlib.sha256(f"{pepper}:{plaintext}".encode()).hexdigest()


def verify_key(plaintext: str, expected_hash: str, pepper: str) -> bool:
    return hmac.compare_digest(hash_key(plaintext, pepper), expected_hash)


def looks_like_key(value: str) -> bool:
    """Cheap shape check so obviously invalid input never reaches the database."""
    if not value.startswith(KEY_PREFIX):
        return False
    body = value[len(KEY_PREFIX) :]
    if not (20 <= len(body) <= 128):
        return False
    return all(c.isalnum() or c in "-_" for c in body)


def parse_authorization(header: str | None) -> str | None:
    """Extract a bearer token. Accepts `Bearer <key>` and a bare key.

    Returns None rather than raising: the caller decides whether a missing
    credential is a 401 or an anonymous request.
    """
    if not header:
        return None
    value = header.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered.startswith("bearer "):
        value = value[7:].strip()
    elif lowered.startswith("token "):
        value = value[6:].strip()
    return value or None


def default_scopes() -> frozenset[Scope]:
    return frozenset(
        {
            Scope.MEMORIES_READ,
            Scope.MEMORIES_WRITE,
            Scope.SPACES_READ,
            Scope.SPACES_WRITE,
            Scope.SEARCH,
        }
    )


def build_api_key(
    *,
    org_id: str,
    name: str,
    pepper: str,
    scopes: frozenset[Scope] | None = None,
    expires_at: object | None = None,
) -> tuple[ApiKey, str]:
    """Create an ApiKey record plus the one-time plaintext."""
    generated = generate_key(pepper)
    record = ApiKey(
        org_id=org_id,
        name=name,
        key_hash=generated.key_hash,
        prefix=generated.prefix,
        scopes=scopes or default_scopes(),
        created_at=utcnow(),
        expires_at=expires_at,  # type: ignore[arg-type]
    )
    return record, generated.plaintext


__all__ = [
    "KEY_PREFIX",
    "GeneratedKey",
    "build_api_key",
    "default_scopes",
    "generate_key",
    "hash_key",
    "looks_like_key",
    "parse_authorization",
    "verify_key",
]
