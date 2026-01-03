from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import redis  # redis-py
except Exception:  # pragma: no cover
    redis = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _default_user_id() -> str:
    # Windows/Linux safe fallback
    return (
        os.getenv("USERNAME")
        or os.getenv("USER")
        or "local_user"
    )


def make_history_key(app_id: str, user_id: str, session_id: str) -> str:
    return f"rag:chat:{app_id}:{user_id}:{session_id}"


@dataclass
class RedisStoreConfig:
    redis_url: str = "redis://localhost:6379"
    app_id: str = "rag_gui"
    user_id: str = "local_user"
    session_id: str = "default"
    history_max_turns: int = 50
    history_ttl_seconds: int = 2592000  # 30 days


class RedisStore:
    """Synchronous Redis store for Tkinter GUI (thread-safe via internal lock)."""

    def __init__(self, cfg: RedisStoreConfig):
        self.cfg = cfg
        self.last_error: Optional[str] = None
        if redis is None:
            self.enabled = False
            self._r = None
            self.last_error = "redis-py is not installed. Run: pip install redis"
            return

        self.enabled = bool(cfg.redis_url)
        self._r = None
        if self.enabled:
            try:
                # decode_responses=True => strings in/out
                self._r = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
            except Exception as e:
                self.enabled = False
                self._r = None
                self.last_error = f"Redis init failed: {e}"


    def ping(self) -> bool:
        if not self.enabled or not self._r:
            return False
        try:
            ok = bool(self._r.ping())
            if ok:
                self.last_error = None
            return ok
        except Exception as e:
            self.last_error = f"Redis ping failed: {e}"
            return False


    # ---------------- logs ----------------
    def add_log(self, line: str, max_items: int = 2000) -> None:
        if not self.enabled or not self._r:
            return
        key = "rag:logs"
        item = {"ts": _now(), "line": line}
        self._r.lpush(key, json.dumps(item, ensure_ascii=False))
        self._r.ltrim(key, 0, max_items - 1)

    def get_logs(self, n: int = 500) -> List[Dict[str, Any]]:
        if not self.enabled or not self._r:
            return []
        raw = self._r.lrange("rag:logs", 0, max(0, n - 1))
        return [json.loads(x) for x in raw]

    # ---------------- results ----------------
    def add_result(self, payload: Dict[str, Any], max_items: int = 1000) -> None:
        if not self.enabled or not self._r:
            return
        key = "rag:results"
        item = {"ts": _now(), "payload": payload}
        self._r.lpush(key, json.dumps(item, ensure_ascii=False))
        self._r.ltrim(key, 0, max_items - 1)

    def get_results(self, n: int = 200) -> List[Dict[str, Any]]:
        if not self.enabled or not self._r:
            return []
        raw = self._r.lrange("rag:results", 0, max(0, n - 1))
        return [json.loads(x) for x in raw]

    # ---------------- history ----------------
    def append_turn(self, question: str, answer: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self._r:
            return
        key = make_history_key(self.cfg.app_id, self.cfg.user_id, self.cfg.session_id)
        item = {"ts": _now(), "q": question, "a": answer, "meta": meta or {}}
        self._r.lpush(key, json.dumps(item, ensure_ascii=False))
        self._r.ltrim(key, 0, max(1, int(self.cfg.history_max_turns)) - 1)
        if self.cfg.history_ttl_seconds and self.cfg.history_ttl_seconds > 0:
            self._r.expire(key, int(self.cfg.history_ttl_seconds))

    def get_last_turns(self, n: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not self._r:
            return []
        key = make_history_key(self.cfg.app_id, self.cfg.user_id, self.cfg.session_id)
        raw = self._r.lrange(key, 0, max(0, n - 1))
        return [json.loads(x) for x in raw]

    def export_all(self, n_history: int = 200, n_logs: int = 2000, n_results: int = 1000) -> Dict[str, Any]:
        return {
            "meta": {
                "app_id": self.cfg.app_id,
                "user_id": self.cfg.user_id,
                "session_id": self.cfg.session_id,
                "redis_url": self.cfg.redis_url,
            },
            "history": list(reversed(self.get_last_turns(n_history))),  # oldest-first
            "logs": list(reversed(self.get_logs(n_logs))),
            "results": list(reversed(self.get_results(n_results))),
        }


def format_history_for_prompt(turns_newest_first: List[Dict[str, Any]]) -> str:
    if not turns_newest_first:
        return ""
    turns = list(reversed(turns_newest_first))  # oldest-first
    blocks = []
    for t in turns:
        ts = (t.get("ts") or "").strip()
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        if not q and not a:
            continue
        blocks.append(f"[{ts}]\nUser: {q}\nAssistant: {a}")
    return "\n\n".join(blocks)


def default_user_id() -> str:
    return _default_user_id()