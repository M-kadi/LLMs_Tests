from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable (for models_config.py).
# Folder layout expectation:
#   <project_root>/
#     models_config.py
#     rag_gui_clean/
#       main.py, rag_gui_app.py, rag_engine_qdrant.py, rag_settings.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional dependency: python-dotenv
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    _load_dotenv = None

# Models config (provided by your project)
sys.path.insert(0, str(Path(__file__).parent))
from models_config import (  # type: ignore
    CHAT_MODELS,
    CHAT_MODELS_GEMINI,
    CHAT_MODELS_OPENAI,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    EMBEDDING_MODELS_GEMINI,
    EMBEDDING_MODELS_OPENAI,
)

# ------------------ App Defaults ------------------

APP_TITLE = "RAG GUI (CSV/TXT) : Qdrant Local + Ollama (Embeddings + Chat)"

SETTINGS_FILE_NAME = "rag_settings.json"

RAG_DATA_0_FOLDER_NAME = "rag_data"
RAG_DATA_1_FOLDER_NAME = "my_csvs"
RAG_DATA_2_FOLDER_NAME = "docs"

OLLAMA_PROVIDER = "ollama"
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
ALL_PROVIDERS = [OLLAMA_PROVIDER, GEMINI_PROVIDER, OPENAI_PROVIDER]

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

# Defaults (env can override AFTER load_env()).
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_APP_ID = os.getenv("RAG_APP_ID", "rag_gui")
DEFAULT_SESSION_ID = os.getenv("RAG_SESSION_ID", "default")
DEFAULT_USER_ID = os.getenv("RAG_USER_ID", "mk_local_user")
DEFAULT_HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "5"))
DEFAULT_HISTORY_MAX_TURNS = int(os.getenv("RAG_HISTORY_MAX_TURNS", "50"))
DEFAULT_HISTORY_TTL_SECONDS = int(os.getenv("RAG_HISTORY_TTL_SECONDS", "2592000"))  # 30 days

DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DEFAULT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "csv_rag")

DEFAULT_TOPK_RETRIEVE = 10
DEFAULT_TOPK_USE = 3
DEFAULT_ENABLE_RERANK = False
DEFAULT_TEXT_GROUP_LINES = 1

DEFAULT_DOCS_DIR = str(
    Path(RAG_DATA_0_FOLDER_NAME) / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME
)

DEFAULT_PROVIDER = OLLAMA_PROVIDER
DEFAULT_ENV_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "keys.env")

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



def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
    return default


def detect_provider_for_model(model_name: str, *, kind: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return OLLAMA_PROVIDER

    if kind == "embedding":
        if name in EMBEDDING_MODELS_GEMINI:
            return GEMINI_PROVIDER
        if name in EMBEDDING_MODELS_OPENAI:
            return OPENAI_PROVIDER
        return OLLAMA_PROVIDER

    if kind == "chat":
        if name in CHAT_MODELS_GEMINI:
            return GEMINI_PROVIDER
        if name in CHAT_MODELS_OPENAI:
            return OPENAI_PROVIDER
        return OLLAMA_PROVIDER

    raise ValueError("kind must be 'embedding' or 'chat'")


def resolve_docs_dir(script_dir: Path, raw_value: str) -> Path:
    p = (raw_value or "").strip()
    if not p:
        return Path(DEFAULT_DOCS_DIR)

    path = Path(p)
    if path.is_absolute():
        return path
    return (script_dir / path).resolve()


def load_env(env_path_value: str, *, script_dir: Path) -> Optional[Path]:
    raw = (env_path_value or "").strip()
    if not raw:
        return None

    env_path = Path(raw)
    if not env_path.is_absolute():
        env_path = (script_dir / env_path).resolve()

    if not env_path.exists():
        return env_path

    if _load_dotenv is not None:
        _load_dotenv(env_path)
        return env_path

    # Minimal .env parser (fallback if python-dotenv isn't installed)
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

    return env_path


def default_settings() -> Dict[str, Any]:
    return {
        "qdrant_url": DEFAULT_QDRANT_URL,
        "collection_prefix": DEFAULT_COLLECTION_PREFIX,
        "topk_retrieve": DEFAULT_TOPK_RETRIEVE,
        "topk_use": DEFAULT_TOPK_USE,
        "enable_rerank": DEFAULT_ENABLE_RERANK,
        "text_group_lines": DEFAULT_TEXT_GROUP_LINES,
        "docs_dir": DEFAULT_DOCS_DIR,
        "rerank_prompt_template": DEFAULT_RERANK_PROMPT_TEMPLATE,
        "answer_prompt_template": DEFAULT_ANSWER_PROMPT_TEMPLATE,
        "embedding_model": EMBEDDING_MODEL,
        "main_model": DEFAULT_MODEL,
        "main_provider": DEFAULT_PROVIDER,
        "embedding_provider": DEFAULT_PROVIDER,
        "env_path": DEFAULT_ENV_PATH,
        "redis_url": DEFAULT_REDIS_URL,
        "app_id": DEFAULT_APP_ID,
        "session_id": DEFAULT_SESSION_ID,
        "user_id": DEFAULT_USER_ID,
        "history_turns": DEFAULT_HISTORY_TURNS,
        "history_max_turns": DEFAULT_HISTORY_MAX_TURNS,
        "history_ttl_seconds": DEFAULT_HISTORY_TTL_SECONDS,

    }


def load_settings(path: Path) -> Dict[str, Any]:
    defaults = default_settings()
    if not path.exists():
        try:
            path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        except Exception:
            pass
        return defaults

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
    except Exception:
        try:
            path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        except Exception:
            pass
        return defaults

    merged = dict(defaults)
    merged.update(data)
    return merged


def save_settings(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
