"""Tracer: neatlogs integration with graceful local buffering.

If NEATLOGS_API_KEY is set, every run is traced in neatlogs (workflow span +
plan/route/llm-call/verify/recover spans). If not, events buffer to a local
JSONL file (local_traces/) so no observability is lost during dev — the buffer
can be flushed/uploaded later. neatlogs tracing was wired from commit #1.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from datetime import datetime, timezone

_NEATLOGS_KEY = os.environ.get("NEATLOGS_API_KEY", "").strip()
_TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_traces")
_neatlogs = None
_workflow_name = "relay-zero"


def _ensure_dir():
    os.makedirs(_TRACE_DIR, exist_ok=True)


def init():
    """Initialize neatlogs if a key is present. Idempotent."""
    global _neatlogs
    if _neatlogs is not None or not _NEATLOGS_KEY:
        return False
    try:
        import neatlogs

        neatlogs.init(
            api_key=_NEATLOGS_KEY,
            workflow_name=_workflow_name,
        )
        _neatlogs = neatlogs
        return True
    except Exception as exc:  # fall back to local buffering
        _append_buffer({"event": "neatlogs_init_failed", "error": str(exc)})
        return False


def _append_buffer(event: dict):
    _ensure_dir()
    path = os.path.join(_TRACE_DIR, f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl")
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    event.setdefault("trace_id", getattr(_append_buffer, "trace_id", str(uuid.uuid4())))
    with open(path, "a") as fh:
        fh.write(json.dumps(event) + "\n")


class Span:
    """Context manager: live neatlogs span when keyed, local-buffer event otherwise."""

    def __init__(self, name: str, kind: str = "SPAN", attrs: dict | None = None):
        self.name = name
        self.kind = kind
        self.attrs = attrs or {}
        self._start = None
        self._cm = None

    def __enter__(self):
        self._start = time.perf_counter()
        if _neatlogs is not None:
            from neatlogs import span as _span_fn

            try:
                self._cm = _span_fn(kind=self.kind, name=self.name, **self.attrs)
                self._cm.__enter__()
                return self
            except Exception:
                self._cm = None
        return self

    def __exit__(self, exc_type, exc, tb):
        latency_ms = (time.perf_counter() - self._start) * 1000
        attrs = dict(self.attrs)
        attrs["latency_ms"] = round(latency_ms, 1)
        if exc_type:
            attrs["error"] = str(exc)
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        _append_buffer(
            {
                "event": "span",
                "span": self.name,
                "kind": self.kind,
                "attrs": attrs,
                "neatlogs_live": self._cm is not None,
            }
        )
        return False


def event(name: str, **attrs):
    """Fire-and-forget event: live trace attribute + local buffer."""
    _append_buffer({"event": name, "attrs": attrs})


def flush():
    if _neatlogs is not None:
        with contextlib.suppress(Exception):
            _neatlogs.flush()


def shutdown():
    if _neatlogs is not None:
        with contextlib.suppress(Exception):
            _neatlogs.shutdown()
