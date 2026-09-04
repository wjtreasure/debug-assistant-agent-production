from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

_SENSITIVE_KEY_PARTS = (
    'api_key', 'apikey', 'authorization', 'access_token', 'refresh_token',
    'password', 'passwd', 'secret', 'credential', 'private_key',
)
_BEARER_RE = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]{8,}')
_SK_RE = re.compile(r'\bsk-[A-Za-z0-9_-]{12,}\b')


def _sensitive_key(key: Any) -> bool:
    s = str(key).strip().lower().replace('-', '_')
    return any(part in s for part in _SENSITIVE_KEY_PARTS)


def _redact_string(value: str) -> str:
    value = _BEARER_RE.sub('Bearer ***', value)
    value = _SK_RE.sub('***', value)
    return value


def redact_sensitive(value: Any) -> Any:
    """Recursively redact secrets before durable logging/serialization.

    This function is deliberately centralized so a newly added trace event cannot
    accidentally bypass secret masking merely because its caller forgot to sanitize.
    """
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[key] = '***' if _sensitive_key(key) and item not in (None, '') else redact_sensitive(item)
        return out
    if isinstance(value, list):
        return [redact_sensitive(x) for x in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(x) for x in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value
