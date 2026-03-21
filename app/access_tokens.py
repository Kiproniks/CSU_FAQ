from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def create_signed_payload(payload: Dict[str, Any], secret: str, ttl_sec: int) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + max(60, int(ttl_sec))
    payload_part = _b64_encode(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    sig_part = _b64_encode(sig)
    return f"{payload_part}.{sig_part}"


def verify_signed_payload(token: str, secret: str) -> Optional[Dict[str, Any]]:
    if not token or "." not in token:
        return None

    payload_part, sig_part = token.split(".", 1)
    expected_sig = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()

    try:
        actual_sig = _b64_decode(sig_part)
    except Exception:
        return None

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload_raw = _b64_decode(payload_part)
        payload = json.loads(payload_raw.decode("utf-8"))
    except Exception:
        return None

    exp = int(payload.get("exp", 0))
    if exp < int(time.time()):
        return None

    return payload
