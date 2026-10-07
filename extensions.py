# extensions.py
import os
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

cache = Cache()

_VALKEY_URL = os.getenv("REDIS_URL", "memory://")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"],
    storage_uri=_VALKEY_URL,
    storage_options={
        "socket_timeout": 2,
        "socket_connect_timeout": 2,
    },
    headers_enabled=True,
    strategy="fixed-window",
    in_memory_fallback_enabled=True,
    in_memory_fallback=["500 per day", "100 per hour"],
)


# ✅ NEW: expose a per-email key func so /request-reset can rate-limit
# by email as well as by IP.
def get_email_from_request():
    """Rate-limit key that falls back to IP when no email is in the body."""
    from flask import request
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    return email or get_remote_address()