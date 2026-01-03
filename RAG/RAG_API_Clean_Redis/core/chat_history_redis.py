from __future__ import annotations
import json, time
from typing import Any, Dict, List, Optional

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def make_history_key(app_id: str, user_id: str, session_id: str = "default") -> str:
    return f"rag:chat:{app_id}:{user_id}:{session_id}"

async def append_turn(redis, *, app_id: str, user_id: str, session_id: str = "default",
                      question: str, answer: str, meta: Optional[Dict[str, Any]] = None,
                      max_turns: int = 50, ttl_seconds: int = 2592000):
    key = make_history_key(app_id, user_id, session_id)
    item = {"ts": _now(), "q": question, "a": answer, "meta": meta or {}}
    await redis.lpush(key, json.dumps(item, ensure_ascii=False))
    await redis.ltrim(key, 0, max_turns - 1)
    await redis.expire(key, ttl_seconds)

async def get_last_turns(redis, *, app_id: str, user_id: str, session_id: str = "default", n: int = 5):
    key = make_history_key(app_id, user_id, session_id)
    raw = await redis.lrange(key, 0, n - 1)
    return [json.loads(x) for x in raw]