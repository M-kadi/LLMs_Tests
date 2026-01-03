from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.settings_store import load_env_from_settings
from core.providers.factory import make_embedding_provider, make_chat_provider
from core.engine_qdrant import RAGEngineQdrant


def make_engine(project_root: Path, s: Dict[str, Any]) -> RAGEngineQdrant:
    load_env_from_settings(project_root, s)

    embedding_client = make_embedding_provider(s)
    chat_client = make_chat_provider(s)

    return RAGEngineQdrant(
        qdrant_url=s["qdrant_url"],
        qdrant_api_key=s.get("qdrant_api_key", ""),
        collection_prefix=s["collection_prefix"],
        embedding_client=embedding_client,
        chat_client=chat_client,
        embedding_model_name=s["embedding_model"],
        main_model_name=s["main_model"],
        rerank_prompt_template=s["rerank_prompt_template"],
        answer_prompt_template=s["answer_prompt_template"],
    )
