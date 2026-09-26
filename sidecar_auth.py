"""Shared bearer-token check for the sidecars.

The sidecars answer with a transcript of whatever audio they are handed, so anything that can
reach one can both spend its CPU/GPU and read back what was said. That is tolerable while they
listen on loopback; it is not once one of them has to serve another machine. `STT_API_KEY` turns
on the check, and leaving it unset keeps the loopback-only deployments exactly as they were.

Kept in its own module, free of Flask-and-model imports at call time, so the rule itself can be
tested without a model on the machine running the test.
"""
import hmac
import os


def authorized(header, key):
    """True when `header` ("Authorization: ...") carries `key`. No key configured = open."""
    if not key:
        return True
    if not header:
        return False
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return False
    #  Compared without leaking length/prefix through timing.
    return hmac.compare_digest(parts[1].strip(), key)


def install(app, key=None):
    """Reject unauthorized requests to everything but /health, which has to stay probe-able."""
    from flask import jsonify, request
    key = os.environ.get('STT_API_KEY', '') if key is None else key
    if not key:
        return False

    @app.before_request
    def _check():
        if request.path == '/health':
            return None
        if not authorized(request.headers.get('Authorization'), key):
            return jsonify(error='unauthorized'), 401

        return None

    return True
