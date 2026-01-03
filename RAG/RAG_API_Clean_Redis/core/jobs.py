from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, List


@dataclass
class JobState:
    id: str
    type: str
    status: str  # queued | running | done | failed
    progress: int
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_LOCK = threading.Lock()
_JOBS: Dict[str, JobState] = {}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def create_job(job_type: str, target: Callable[[], Dict[str, Any]], log_fn: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    st = JobState(
        id=job_id,
        type=job_type,
        status="queued",
        progress=0,
        created_at=_now(),
    )
    with _LOCK:
        _JOBS[job_id] = st

    def runner():
        _set(job_id, status="running", progress=5, started_at=_now())
        if log_fn:
            log_fn(f"Job {job_id} started ({job_type}).", "INFO")
        try:
            res = target()
            _set(job_id, status="done", progress=100, finished_at=_now(), result=res)
            if log_fn:
                log_fn(f"Job {job_id} done ({job_type}).", "INFO")
        except Exception as e:
            _set(job_id, status="failed", progress=100, finished_at=_now(), error=str(e))
            if log_fn:
                log_fn(f"Job {job_id} failed ({job_type}): {e}", "ERROR")

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    return asdict(st)


def _set(job_id: str, **kwargs):
    with _LOCK:
        st = _JOBS.get(job_id)
        if not st:
            return
        for k, v in kwargs.items():
            setattr(st, k, v)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        st = _JOBS.get(job_id)
        return asdict(st) if st else None


def list_jobs(limit: int = 200, offset: int = 0) -> Dict[str, Any]:
    with _LOCK:
        items: List[JobState] = list(_JOBS.values())
    total = len(items)
    sliced = items[offset:offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": [asdict(x) for x in sliced]}
