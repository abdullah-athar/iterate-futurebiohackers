"""A key -> JSON value cache in one file, safe across threads and processes.

Every read-modify-write holds a per-path thread lock *and* an fcntl lock on `<file>.lock`, so two
writers (threads of one swarm, or two processes sharing a cache such as compare_runs runs) never lose
each other's updates. Writes go to a unique temp file and are renamed into place, so a reader never
sees a partial file. Callers never hold the lock while making the (slow) LLM call: they `get`, call,
then `put`. Keys come from `cache_key(...)`, a hash of everything that determines the cached value.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def cache_key(**parts: Any) -> str:
    """Stable hash of the parts that determine a cached value (code fingerprint, model, prompt, ...)."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


class JsonCache:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None

    @contextmanager
    def _locked(self):
        key = str(self.path.resolve())
        with _REGISTRY_LOCK:
            tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        with tlock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path.with_suffix(self.path.suffix + ".lock"), "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key: str) -> Any | None:
        if self.path is None:
            return None
        with self._locked():
            return self._read().get(key)

    def put(self, key: str, value: Any) -> None:
        if self.path is None:
            return
        with self._locked():
            data = self._read()
            data[key] = value
            tmp = self.path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=1))
            tmp.replace(self.path)

    def __len__(self) -> int:
        if self.path is None:
            return 0
        with self._locked():
            return len(self._read())
