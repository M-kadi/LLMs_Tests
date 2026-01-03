from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Body, Request
from pydantic import BaseModel, Field
# from requests import request

from core.settings_store import load_settings, save_settings, reset_settings
from core.engine_factory import make_engine
from core.memory_store import log, add_result, get_logs_async, clear_logs, get_results_async, clear_results_async
from core.jobs import create_job, get_job as _get_job, list_jobs as _list_jobs
from core.chat_history_redis import get_last_turns, append_turn
from core.prompt_history import format_history_for_prompt

# Optional model lists (if you keep models_config.py)
from models_config import (
    CHAT_MODELS_OLLAMA, CHAT_MODELS_OPENAI, CHAT_MODELS_GEMINI,
    EMBEDDING_MODELS_OLLAMA, EMBEDDING_MODELS_OPENAI, EMBEDDING_MODELS_GEMINI,
)

router = APIRouter()

# Project root = rag_api_simple/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool(x, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
    return default


class SettingsIn(BaseModel):
    qdrant_url: Optional[str] = None
    # qdrant_api_key: Optional[str] = None
    collection_prefix: Optional[str] = None
    topk_retrieve: Optional[int] = None
    topk_use: Optional[int] = None
    enable_rerank: Optional[bool] = None
    text_group_lines: Optional[int] = None
    embedding_model: Optional[str] = None
    main_model: Optional[str] = None
    docs_dir: Optional[str] = None
    txt_chunk_chars: Optional[int] = None
    txt_overlap: Optional[int] = None
    batch_size: Optional[int] = None
    rerank_prompt_template: Optional[str] = None
    answer_prompt_template: Optional[str] = None
    main_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    env_path: Optional[str] = None
    redis_url: Optional[str] = None
    app_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    history_turns: Optional[int] = None
    history_max_turns: Optional[int] = None
    history_ttl_seconds: Optional[int] = None
    # openai_api_key: Optional[str] = None
    # gemini_api_key: Optional[str] = None


class QueryIn(BaseModel):
    query: str = Field(..., description="User question")

    # Conversation identifiers (optional per-request overrides)
    app_id: Optional[str] = Field(None, description="Application ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")

    # History controls (optional per-request overrides)
    history_turns: Optional[int] = Field(None, ge=0, le=50, description="How many recent turns to include in the prompt")
    history_max_turns: Optional[int] = Field(None, ge=0, le=50000, description="Max turns to keep in Redis per session")
    history_ttl_seconds: Optional[int] = Field(None, ge=0, le=31536000, description="History TTL in seconds")

    # Retrieval controls (optional overrides)
    topk_retrieve: Optional[int] = None
    topk_use: Optional[int] = None
    enable_rerank: Optional[bool] = None


class BuildIndexIn(BaseModel):
    docs_dir: Optional[str] = None
    batch_size: Optional[int] = None
    txt_chunk_chars: Optional[int] = None
    txt_overlap: Optional[int] = None
    group_lines: Optional[int] = None


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/settings", summary="Get settings", operation_id="Get settings")
def get_settings():
    return load_settings(PROJECT_ROOT)


@router.post("/settings")
def update_settings(body: SettingsIn):
    current = load_settings(PROJECT_ROOT)
    upd = {k: v for k, v in body.model_dump().items() if v is not None}
    merged = save_settings(PROJECT_ROOT, {**current, **upd})
    log("Settings saved.")
    return merged


@router.post("/settings/reset")
def reset():
    s = reset_settings(PROJECT_ROOT)
    log("Settings reset to defaults.")
    return s


@router.get("/ping/qdrant")
def ping_qdrant():
    s = load_settings(PROJECT_ROOT)
    try:
        eng = make_engine(PROJECT_ROOT, s)
        cols = eng.client.get_collections()
        log(f"Qdrant OK. collections={len(cols.collections)}")
        return {"ok": True, "collections_count": len(cols.collections)}
    except Exception as e:
        log(f"Qdrant ping failed: {e}", "ERROR")
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {e}")


@router.get("/jobs")
def jobs(limit: int = 200, offset: int = 0):
    return _list_jobs(limit=limit, offset=offset)


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    st = _get_job(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return st


@router.get("/qdrant/collections")
def list_collections():
    s = load_settings(PROJECT_ROOT)
    try:
        eng = make_engine(PROJECT_ROOT, s)
        cols = eng.client.get_collections()
        return {"collections": [c.name for c in cols.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qdrant/collections/latest")
def latest_collection():
    s = load_settings(PROJECT_ROOT)
    try:
        eng = make_engine(PROJECT_ROOT, s)
        cols = eng.client.get_collections()
        names = [c.name for c in cols.collections]
        pref = s.get("collection_prefix", "")
        pref_hits = [n for n in names if n.startswith(pref)]
        return {"latest": (pref_hits[-1] if pref_hits else (names[-1] if names else ""))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/build")
def build_index(body: BuildIndexIn | None = Body(default=None)):
    s = load_settings(PROJECT_ROOT)
    body = body or BuildIndexIn()

    docs_dir = body.docs_dir or s.get("docs_dir")
    if not docs_dir:
        raise HTTPException(status_code=400, detail="docs_dir is required (in request or settings).")

    batch_size = body.batch_size or _safe_int(s.get("batch_size", 64), 64)
    txt_chunk_chars = body.txt_chunk_chars or _safe_int(s.get("txt_chunk_chars", 900), 900)
    txt_overlap = body.txt_overlap or _safe_int(s.get("txt_overlap", 120), 120)
    group_lines = body.group_lines or _safe_int(s.get("text_group_lines", 1), 1)

    try:
        eng = make_engine(PROJECT_ROOT, s)
        total = eng.build_from_folder(
            Path(docs_dir),
            batch_size=batch_size,
            txt_chunk_chars=txt_chunk_chars,
            txt_overlap=txt_overlap,
            group_lines=group_lines,
        )
        log(f"Index build done. upserted={total} collection={eng.collection_name}")
        return {"upserted": total, "collection": eng.collection_name}
    except Exception as e:
        log(f"Index build failed: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/build_async")
def build_index_async(body: BuildIndexIn | None = Body(default=None)):
    s = load_settings(PROJECT_ROOT)
    body = body or BuildIndexIn()

    docs_dir = body.docs_dir or s.get("docs_dir")
    if not docs_dir:
        raise HTTPException(status_code=400, detail="docs_dir is required (in request or settings).")

    batch_size = body.batch_size or _safe_int(s.get("batch_size", 64), 64)
    txt_chunk_chars = body.txt_chunk_chars or _safe_int(s.get("txt_chunk_chars", 900), 900)
    txt_overlap = body.txt_overlap or _safe_int(s.get("txt_overlap", 120), 120)
    group_lines = body.group_lines or _safe_int(s.get("text_group_lines", 1), 1)

    def task():
        eng = make_engine(PROJECT_ROOT, s)
        total = eng.build_from_folder(
            Path(docs_dir),
            batch_size=batch_size,
            txt_chunk_chars=txt_chunk_chars,
            txt_overlap=txt_overlap,
            group_lines=group_lines,
        )
        return {"upserted": total, "collection": eng.collection_name}

    return create_job("build_index", task, log_fn=log)


@router.post("/query")
async def run_query(body: QueryIn, request: Request):
    s = load_settings(PROJECT_ROOT)

    topk_retrieve = body.topk_retrieve if body.topk_retrieve is not None else _safe_int(s.get("topk_retrieve", 10), 10)
    topk_use = body.topk_use if body.topk_use is not None else _safe_int(s.get("topk_use", 3), 3)
    enable_rerank = body.enable_rerank if body.enable_rerank is not None else _safe_bool(s.get("enable_rerank", False), False)

    try:
        eng = make_engine(PROJECT_ROOT, s)

        r = getattr(request.app.state, "redis", None)

        app_id, session_id, user_id, history_turns, history_ttl = _resolve_ids_and_history(
            body=body, settings=s
        )

        history_block = ""
        history_used = 0
        if r is not None and history_turns > 0:
            turns = await get_last_turns(
                r,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                n=history_turns,
            )
            history_block = format_history_for_prompt(turns)
            history_used = len(turns)
            log(f"[RAG] history_used={history_used}")

        original_query = body.query
        result, body.query = eng.answer(
            history_block=history_block,
            query=body.query,
            top_k_retrieve=topk_retrieve,
            top_k_use=topk_use,
            enable_rerank=enable_rerank,
        )

        Ids = {
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id
        }

        queryStr = "".join([
                    f"Query: {body.query}",
                    "" if body.query == original_query else
                    f"\n(modified from original:\n{original_query})"
                ])
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            # "query": body.query,
            "query": queryStr,
            "settings": {"topk_retrieve": topk_retrieve, "topk_use": topk_use, "enable_rerank": enable_rerank},
            "provider": {"main": s.get("main_provider"), "embedding": s.get("embedding_provider")},
            "models": {"main": s.get("main_model"), "embedding": s.get("embedding_model")},
            "Ids": Ids,            
            "result": result,
            "history_used": history_used,
            "history_block": turns
        }
        
        # append to redis history after answer
        if r is not None:
            try:
                await append_turn(
                    r,
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    question=body.query,
                    answer=str(result.get("answer", "")),
                    ttl_seconds=history_ttl,
                )
            except Exception as e:
                log(f"[Redis] append_turn failed: {e}")

        add_result(payload)
        return payload

    except Exception as e:
        log(f"[ERROR] query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# def _resolve_ids_and_history(body: QueryIn, settings: dict):
#     app_id = body.app_id or settings.get("app_id", "rag_api")
#     session_id = body.session_id or settings.get("session_id", "default")
#     user_id = body.user_id or settings.get("user_id", "local_user")

#     hist_default = _safe_int(settings.get("history_max_turns", 5), 5)
#     history_turns = hist_default if body.history_turns is None else max(0, int(body.history_turns))

#     ttl_default = _safe_int(settings.get("history_ttl_seconds", 60 * 60 * 24), 60 * 60 * 24)
#     history_ttl = ttl_default if body.history_ttl_seconds is None else max(60, int(body.history_ttl_seconds))

#     return app_id, session_id, user_id, history_turns, history_ttl


@router.post("/query_async")
def run_query_async(body: QueryIn):
    s = load_settings(PROJECT_ROOT)

    topk_retrieve = body.topk_retrieve if body.topk_retrieve is not None else _safe_int(s.get("topk_retrieve", 10), 10)
    topk_use = body.topk_use if body.topk_use is not None else _safe_int(s.get("topk_use", 3), 3)
    enable_rerank = body.enable_rerank if body.enable_rerank is not None else _safe_bool(s.get("enable_rerank", False), False)

    def task():
        eng = make_engine(PROJECT_ROOT, s)
        result = eng.answer(body.query, top_k_retrieve=topk_retrieve, top_k_use=topk_use, enable_rerank=enable_rerank)
        payload = {
            "query": body.query,
            "settings": {"topk_retrieve": topk_retrieve, "topk_use": topk_use, "enable_rerank": enable_rerank},
            "provider": {"main": s.get("main_provider"), "embedding": s.get("embedding_provider")},
            "models": {"main": s.get("main_model"), "embedding": s.get("embedding_model")},
            "result": result,
        }
        add_result(payload)
        return payload

    return create_job("query", task, log_fn=log)


@router.get("/results")
async def results(limit: int = 100, offset: int = 0):
    return await get_results_async(limit=limit, offset=offset)


@router.delete("/results")
async def results_clear():
    await clear_results_async()
    log("Results cleared.")
    return {"ok": True}


@router.get("/logs")
async def logs(limit: int = 200, offset: int = 0):
    return await get_logs_async(limit=limit, offset=offset)


@router.delete("/logs")
def logs_clear():
    clear_logs()
    log("Logs cleared.")
    return {"ok": True}


@router.get("/models/main/{provider}")
def models_main(provider: str):
    p = (provider or "").strip().lower()
    if p == "ollama":
        return {"provider": p, "models": CHAT_MODELS_OLLAMA}
    if p == "openai":
        return {"provider": p, "models": CHAT_MODELS_OPENAI}
    if p == "gemini":
        return {"provider": p, "models": CHAT_MODELS_GEMINI}
    return {"provider": p, "models": []}


@router.get("/models/embedding/{provider}")
def models_embedding(provider: str):
    p = (provider or "").strip().lower()
    if p == "ollama":
        return {"provider": p, "models": EMBEDDING_MODELS_OLLAMA}
    if p == "openai":
        return {"provider": p, "models": EMBEDDING_MODELS_OPENAI}
    if p == "gemini":
        return {"provider": p, "models": EMBEDDING_MODELS_GEMINI}
    return {"provider": p, "models": []}

# @router.get("/docs1", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return get_swagger_ui_html(
#         openapi_url=router.openapi_url,
#         title=router.title + " - Swagger UI",
#         swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
#         swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
#         swagger_ui_parameters={
#             "dom_id": "#swagger-ui",
#             "layout": "BaseLayout",
#             "deepLinking": True,
#             "filter": True  # Enables the built-in filter input (hides until you type)
#         }
#     )

def _resolve_ids_and_history(body: QueryIn, settings: Dict[str, Any]) -> Dict[str, Any]:
    app_id = (body.app_id or settings.get("app_id") or "default").strip()
    user_id = (body.user_id or settings.get("user_id") or "anonymous").strip()
    session_id = (body.session_id or settings.get("session_id") or "default").strip()

    history_turns = body.history_turns
    if history_turns is None:
        history_turns = int(settings.get("history_turns", 5))
    history_turns = max(0, min(int(history_turns), 50))

    history_max_turns = body.history_max_turns
    if history_max_turns is None:
        history_max_turns = int(settings.get("history_max_turns", 2000))
    history_max_turns = max(0, int(history_max_turns))

    history_ttl_seconds = body.history_ttl_seconds
    if history_ttl_seconds is None:
        history_ttl_seconds = int(settings.get("history_ttl_seconds", 0))
    history_ttl_seconds = max(0, int(history_ttl_seconds))

    # return {
    #     "app_id": app_id,
    #     "user_id": user_id,
    #     "session_id": session_id,
    #     "history_turns": history_turns,
    #     "history_max_turns": history_max_turns,
    #     "history_ttl_seconds": history_ttl_seconds,
    # }
    return app_id, session_id, user_id, history_turns, history_ttl_seconds




