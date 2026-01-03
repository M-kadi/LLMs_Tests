from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

from dotenv import load_dotenv
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from models_config import DEFAULT_MODEL, EMBEDDING_MODEL

APP_TITLE = "RAG API (Simple): Qdrant + Providers"

OLLAMA = "ollama"
OPENAI = "openai"
GEMINI = "gemini"
ALL_PROVIDERS = [OLLAMA, OPENAI, GEMINI]

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_API_KEY = ""
DEFAULT_COLLECTION_PREFIX = "csv_rag"

DEFAULT_TOPK_RETRIEVE = 10
DEFAULT_TOPK_USE = 3
DEFAULT_ENABLE_RERANK = False
DEFAULT_TEXT_GROUP_LINES = 1

DEFAULT_TXT_CHUNK_CHARS = 900
DEFAULT_TXT_OVERLAP = 120
DEFAULT_BATCH_SIZE = 64

DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_HISTORY_TURNS = 5

DEFAULT_APP_ID = "rag_api"
DEFAULT_SESSION_ID = "default"
DEFAULT_USER_ID = "local_user"
DEFAULT_HISTORY_MAX_TURNS = 2000          # how many turns to keep in Redis per session
DEFAULT_HISTORY_TTL_SECONDS = 7 * 24 * 3600  # expire history after 7 days


# Default env file path (relative to this folder)
DEFAULT_ENV_PATH = str((SCRIPT_DIR.parent.parent.parent / "keys.env").resolve())


DEFAULT_RERANK_PROMPT_TEMPLATE = (
    "You are a retrieval re-ranker.\n"
    "Given a user question and a list of candidate contexts, select the most relevant items.\n"
    "Rules:\n"
    "- Choose exactly {choose_k} distinct indices.\n"
    "- Prefer contexts that directly contain facts needed to answer.\n"
    "- Avoid redundant/duplicate contexts.\n"
    "- Output ONLY valid JSON, no extra text.\n\n"
    "Return JSON format:\n"
    "{{\n"
    '  "selected_indices": [0, 2, 5],\n'
    '  "reasons": ["short reason 1", "short reason 2", "short reason 3"]\n'
    "}}\n\n"
    "Question:\n{query}\n\n"
    "Candidates:\n{candidates}\n"
)

DEFAULT_ANSWER_PROMPT_TEMPLATE = (
    "You are an assistant that answers using BOTH the retrieved context and the recent conversation history.\n"
    "IMPORTANT RULES:\n"
    "- Prefer the Context for factual answers.\n"
    "- Use History only to keep continuity (follow-ups, pronouns).\n"
    "- If neither History nor Context contains the answer, reply exactly: "
    '"\"I don\'t know based on the provided context.\"\n"\n'
    "- Keep the answer clear and concise.\n\n"
    "History (last turns):\n{history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

SETTINGS_FILE_NAME = "rag_settings.json"


def default_docs_dir(project_root: Path) -> str:
    # Match your GUI style: <project_root>/rag_data/my_csvs/docs
    return str((project_root.parent.parent / "rag_data" / "my_csvs" / "docs").resolve())


def defaults(project_root: Path) -> Dict[str, Any]:
    return {
        "qdrant_url": DEFAULT_QDRANT_URL,
        # "qdrant_api_key": DEFAULT_QDRANT_API_KEY,
        "collection_prefix": DEFAULT_COLLECTION_PREFIX,
        "topk_retrieve": DEFAULT_TOPK_RETRIEVE,
        "topk_use": DEFAULT_TOPK_USE,
        "enable_rerank": DEFAULT_ENABLE_RERANK,
        "text_group_lines": DEFAULT_TEXT_GROUP_LINES,
        "embedding_model": EMBEDDING_MODEL,
        "main_model": DEFAULT_MODEL,
        "docs_dir": default_docs_dir(project_root),
        "txt_chunk_chars": DEFAULT_TXT_CHUNK_CHARS,
        "txt_overlap": DEFAULT_TXT_OVERLAP,
        "batch_size": DEFAULT_BATCH_SIZE,
        "rerank_prompt_template": DEFAULT_RERANK_PROMPT_TEMPLATE,
        "answer_prompt_template": DEFAULT_ANSWER_PROMPT_TEMPLATE,
        # Providers (separate main/embedding)
        "main_provider": OLLAMA,
        "embedding_provider": OLLAMA,
        # Optional: dotenv file path (relative to project root or absolute)
        "env_path": DEFAULT_ENV_PATH,
        "redis_url": DEFAULT_REDIS_URL,
        "history_turns": DEFAULT_HISTORY_TURNS,
        "app_id": DEFAULT_APP_ID,
        "session_id": DEFAULT_SESSION_ID,
        "user_id": DEFAULT_USER_ID,
        "history_max_turns": DEFAULT_HISTORY_MAX_TURNS,
        "history_ttl_seconds": DEFAULT_HISTORY_TTL_SECONDS,
        # Optional: keys also may be passed here, but env vars are recommended.
        # "openai_api_key": "",
        # "gemini_api_key": "",
    }


def settings_path(project_root: Path) -> Path:
    return project_root / SETTINGS_FILE_NAME


def load_settings(project_root: Path) -> Dict[str, Any]:
    path = settings_path(project_root)
    d = defaults(project_root)

    if not path.exists():
        try:
            path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except Exception:
            pass
        return d

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return d
    except Exception:
        # corrupted -> rewrite defaults
        try:
            path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except Exception:
            pass
        return d

    merged = dict(d)
    merged.update(data)
    return merged


def save_settings(project_root: Path, new_settings: Dict[str, Any]) -> Dict[str, Any]:
    d = defaults(project_root)
    merged = dict(d)
    merged.update(new_settings)
    settings_path(project_root).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def reset_settings(project_root: Path) -> Dict[str, Any]:
    d = defaults(project_root)
    settings_path(project_root).write_text(json.dumps(d, indent=2), encoding="utf-8")
    return d


def load_env_from_settings(project_root: Path, s: Dict[str, Any]) -> None:
    env_path = (s.get("env_path") or "").strip()
    if not env_path:
        return

    p = Path(env_path)
    if not p.is_absolute():
        p = (project_root / p).resolve()

    if p.exists():
        load_dotenv(p)
