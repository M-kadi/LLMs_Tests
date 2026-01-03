from fastapi import APIRouter, Query
from core import memory_store

router = APIRouter(prefix="/history", tags=["History"])

@router.get("")
async def get_history(
    app_id: str = Query("rag_api"),
    user_id: str = Query("anon"),
    session_id: str = Query("default"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    newest_first: bool = Query(False),
):
    return await memory_store.history_get(
        app_id=app_id, user_id=user_id, session_id=session_id,
        limit=limit, offset=offset, newest_first=newest_first
    )

@router.get("/last")
async def get_history_last(
    app_id: str = Query("rag_api"),
    user_id: str = Query("anon"),
    session_id: str = Query("default"),
    n: int = Query(5, ge=0, le=50),
):
    items = await memory_store.history_get_last_turns(
        app_id=app_id, user_id=user_id, session_id=session_id, n=n
    )
    return {"n": n, "items": items}

@router.post("/clear")
async def clear_history(
    app_id: str = Query("rag_api"),
    user_id: str = Query("anon"),
    session_id: str = Query("default"),
):
    await memory_store.history_clear(app_id=app_id, user_id=user_id, session_id=session_id)
    return {"ok": True}
