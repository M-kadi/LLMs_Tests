from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.chat_history_redis import make_history_key

# In-memory fallback (works even if Redis is down)
_LOCK = threading.Lock()
LOGS: List[Dict[str, Any]] = []
RESULTS: List[Dict[str, Any]] = []

# NEW: in-memory history fallback (keyed by (app_id,user_id,session_id))
_HISTORY: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

# Redis-backed store (optional)
_REDIS: Any = None  # redis.Redis OR redis.asyncio.Redis
_REDIS_PREFIX = "rag"

_MAX_LOGS = 2000
_MAX_RESULTS = 1000
_MAX_HISTORY = 2000  # per (app,user,session)


def set_redis_client(redis_client: Any, prefix: str = "rag") -> None:
    """Enable Redis-backed logs/results/history. Safe to call with None."""
    global _REDIS, _REDIS_PREFIX
    _REDIS = redis_client
    _REDIS_PREFIX = prefix or "rag"


def _k(name: str) -> str:
    return f"{_REDIS_PREFIX}:{name}"


def _history_key(app_id: str, user_id: str, session_id: str) -> str:
    # app_id = (app_id or "rag").strip()
    # user_id = (user_id or "anon").strip()
    # session_id = (session_id or "default").strip()
    # return f"{_REDIS_PREFIX}:history:{app_id}:{user_id}:{session_id}"
    return make_history_key(app_id=app_id, user_id=user_id, session_id=session_id)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fire_and_forget(maybe_coro: Any) -> None:
    """
    If using redis.asyncio, methods return coroutines. We can't await here in sync functions,
    so we schedule them on the running loop (best-effort).
    """
    if maybe_coro is None:
        return
    if asyncio.iscoroutine(maybe_coro):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(maybe_coro)
        except Exception:
            # no running loop, ignore
            pass


# -------------------------
# LOGS
# -------------------------
def log(message: str, level: str = "INFO") -> None:
    item = {"ts": _now_ts(), "level": level, "message": message}

    with _LOCK:
        LOGS.append(item)

    r = _REDIS
    if r is not None:
        try:
            _fire_and_forget(r.lpush(_k("logs"), json.dumps(item, ensure_ascii=False)))
            _fire_and_forget(r.ltrim(_k("logs"), 0, _MAX_LOGS - 1))
        except Exception:
            pass


async def get_logs_async(limit: int = 200, offset: int = 0) -> Dict[str, Any]:
    r = _REDIS
    if r is not None:
        try:
            start = int(offset)
            stop = start + int(limit) - 1
            raw = await r.lrange(_k("logs"), start, stop)
            items: List[Dict[str, Any]] = []
            for x in raw or []:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("utf-8", errors="ignore")
                try:
                    items.append(json.loads(x))
                except Exception:
                    items.append({"ts": "", "level": "INFO", "message": str(x)})
            total = int(await r.llen(_k("logs")))
            return {"total": total, "limit": limit, "offset": offset, "items": items}
        except Exception:
            pass

    with _LOCK:
        items = LOGS[offset:offset + limit]
        total = len(LOGS)
    return {"total": total, "limit": limit, "offset": offset, "items": items}


async def clear_logs() -> None:
    with _LOCK:
        LOGS.clear()
    r = _REDIS
    if r is not None:
        try:
            await r.delete(_k("logs"))
        except Exception:
            pass


# -------------------------
# RESULTS
# -------------------------
def add_result(obj: Dict[str, Any]) -> None:
    with _LOCK:
        RESULTS.append(obj)

    r = _REDIS
    if r is not None:
        try:
            _fire_and_forget(r.lpush(_k("results"), json.dumps(obj, ensure_ascii=False)))
            _fire_and_forget(r.ltrim(_k("results"), 0, _MAX_RESULTS - 1))
        except Exception:
            pass


async def get_results_async(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    r = _REDIS
    if r is not None:
        try:
            start = int(offset)
            stop = start + int(limit) - 1
            raw = await r.lrange(_k("results"), start, stop)
            items: List[Dict[str, Any]] = []
            for x in raw or []:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("utf-8", errors="ignore")
                try:
                    items.append(json.loads(x))
                except Exception:
                    items.append({"error": "bad_json", "raw": str(x)})
            total = int(await r.llen(_k("results")))
            return {"total": total, "limit": limit, "offset": offset, "items": items}
        except Exception:
            pass

    with _LOCK:
        items = RESULTS[offset:offset + limit]
        total = len(RESULTS)
    return {"total": total, "limit": limit, "offset": offset, "items": items}


async def clear_results_async() -> None:
    with _LOCK:
        RESULTS.clear()
    r = _REDIS
    if r is not None:
        try:
            await r.delete(_k("results"))
        except Exception:
            pass


# -------------------------
# HISTORY (NEW)
# -------------------------
# async def history_add_turn(
#     *,
#     app_id: str,
#     user_id: str,
#     session_id: str,
#     query: str,
#     answer: str,
#     provider: str = "",
#     meta: Optional[Dict[str, Any]] = None,
#     history_ttl_seconds: int = 0,
# ) -> Dict[str, Any]:
#     item = {
#         "ts": _now_ts(),
#         "app_id": app_id,
#         "user_id": user_id,
#         "session_id": session_id,
#         "query": query,
#         "answer": answer,
#         "provider": provider,
#         "meta": meta or {},
#     }

#     # memory fallback
#     key_mem = ((app_id or "rag"), (user_id or "anon"), (session_id or "default"))
#     with _LOCK:
#         _HISTORY.setdefault(key_mem, []).append(item)
#         # keep bounded
#         if len(_HISTORY[key_mem]) > _MAX_HISTORY:
#             _HISTORY[key_mem] = _HISTORY[key_mem][- _MAX_HISTORY :]

#     # redis
#     r = _REDIS
#     if r is not None:
#         try:
#             key = _history_key(app_id, user_id, session_id)
#             await r.rpush(key, json.dumps(item, ensure_ascii=False))
#             # trim to last _MAX_HISTORY
#             await r.ltrim(key, -_MAX_HISTORY, -1)
#             if history_ttl_seconds and history_ttl_seconds > 0:
#                 await r.expire(key, int(history_ttl_seconds))
#         except Exception:
#             # ignore redis failure (memory still has it)
#             pass

#     return item


async def history_get(
    *,
    app_id: str,
    user_id: str,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    newest_first: bool = False,
) -> Dict[str, Any]:
    limit = max(1, int(limit))
    offset = max(0, int(offset))

    r = _REDIS
    if r is not None:
        try:
            key = _history_key(app_id, user_id, session_id)
            total = int(await r.llen(key))

            if newest_first:
                # newest first: read from right side
                start = total - offset - limit
                end = total - offset - 1
                if end < 0:
                    return {"total": total, "limit": limit, "offset": offset, "items": []}
                start = max(0, start)
                raw = await r.lrange(key, start, end)
                items = []
                for x in raw or []:
                    if isinstance(x, (bytes, bytearray)):
                        x = x.decode("utf-8", errors="ignore")
                    try:
                        items.append(json.loads(x))
                    except Exception:
                        continue
                items.reverse()
                return {"total": total, "limit": limit, "offset": offset, "items": items}

            # oldest first
            start = offset
            stop = offset + limit - 1
            raw = await r.lrange(key, start, stop)
            items = []
            for x in raw or []:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("utf-8", errors="ignore")
                try:
                    items.append(json.loads(x))
                except Exception:
                    continue
            return {"total": total, "limit": limit, "offset": offset, "items": items}
        except Exception:
            pass

    # memory fallback
    key_mem = ((app_id or "rag"), (user_id or "anon"), (session_id or "default"))
    with _LOCK:
        items_all = _HISTORY.get(key_mem, [])
        total = len(items_all)
        if newest_first:
            slice_ = list(reversed(items_all))  # newest->oldest
            items = slice_[offset:offset + limit]
        else:
            items = items_all[offset:offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": items}


async def history_get_last_turns(
    *,
    app_id: str,
    user_id: str,
    session_id: str,
    n: int = 5,
) -> List[Dict[str, Any]]:
    n = max(0, int(n))
    if n == 0:
        return []

    r = _REDIS
    if r is not None:
        try:
            key = _history_key(app_id, user_id, session_id)
            raw = await r.lrange(key, -n, -1)
            items: List[Dict[str, Any]] = []
            for x in raw or []:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("utf-8", errors="ignore")
                try:
                    items.append(json.loads(x))
                except Exception:
                    continue
            return items
        except Exception:
            pass

    # memory fallback
    key_mem = ((app_id or "rag"), (user_id or "anon"), (session_id or "default"))
    with _LOCK:
        items_all = _HISTORY.get(key_mem, [])
        return items_all[-n:]


async def history_clear(*, app_id: str, user_id: str, session_id: str) -> None:
    key_mem = ((app_id or "rag"), (user_id or "anon"), (session_id or "default"))
    with _LOCK:
        _HISTORY.pop(key_mem, None)

    r = _REDIS
    if r is not None:
        try:
            await r.delete(_history_key(app_id, user_id, session_id))
        except Exception:
            pass
