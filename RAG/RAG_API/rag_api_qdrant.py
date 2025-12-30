import os
import re
import json
import sys
import uuid
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List

from fastapi.params import Body
import numpy as np
import pandas as pd
import ollama
from qdrant_client import QdrantClient, models as qmodels

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

# ------------------------------------------------------------
# Your existing models_config import (keep your same file)
# ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from models_config import DEFAULT_MODEL, EMBEDDING_MODEL, CHAT_MODELS, EMBEDDING_MODELS,\
    EMBEDDING_MODELS_OLLAMA, EMBEDDING_MODELS_GEMINI, EMBEDDING_MODELS_OPENAI, \
    CHAT_MODELS_OLLAMA, CHAT_MODELS_GEMINI, CHAT_MODELS_OPENAI
# ------------------------------------------------------------

'''
pip install fastapi uvicorn pydantic qdrant-client pandas numpy ollama
uvicorn rag_api_qdrant:app --host 0.0.0.0 --port 8000 --reload
'''
# To Run:
# - Start Docker Desktop
# - Start Ollama application
# - Start Qdrant locally: from command line, run:
#  D:\LLM\LLM_Tests\LLMs_Tests\RAG\RAG_API>
    # docker run -p 6333:6333 -p 6334:6334 ^
    # -v %cd%\qdrant_data:/qdrant/storage ^
    # qdrant/qdrant
# - Then run this script:
#     D:\LLM\LLM_Tests\LLMs_Tests\RAG\RAG_API>
    # uvicorn rag_api_qdrant:app --host 0.0.0.0 --port 8000 --reload
    # Open browser to (Swagger): http://localhost:8000/docs
    # Test endpoints
    # GET http://localhost:8000/settings
    # POST http://localhost:8000/settings
    # POST http://localhost:8000/settings/reset
    # POST http://localhost:8000/index/build
    # POST http://localhost:8000/query
# '''
# ------------------ Config ------------------
# Qdrant Dashboard URL :
# http://localhost:6333/dashboard#/collections
# Config File rag_settings.json : 
#    contains: LLM chat models + Embedding models, enable reranking, text group lines
# Disable Reranking : by default enabled
# save settings to file : rag_settings.json
# load settings from file on startup
# Enable reranking checkbox : true : will rerank by get the TOPK_RETRIEVE from Qdrant 
#   then send to LLM to rerank to TOPK_USE as final contexts
#   False : directly use TOPK_USE from Qdrant
# Use Text Group Lines : for TXT files, group N lines per chunk instead of single line chunks (Paragraphs)
# Support CSV + TXT files in the same folder for ingestion
# Use Qdrant for vector storage and retrieval
# Enable change Prompt and Prompt templates for Reranking and Answering from settings

# ------------------ App Defaults ------------------
# ------------------ Default Settings ------------------
APP_TITLE = "RAG Fast API : Qdrant Local + Ollama (Embeddings + Chat)"

# ------------------ Providers ------------------
OLLAMA_PROVIDER = "ollama"
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
ALL_PROVIDERS = [OLLAMA_PROVIDER, GEMINI_PROVIDER, OPENAI_PROVIDER]

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

DEFAULT_MAIN_PROVIDER = OLLAMA_PROVIDER
DEFAULT_EMBEDDING_PROVIDER = OLLAMA_PROVIDER

# ------------------ Env Path ------------------
# DEFAULT_ENV_PATH = "keys.env"   # can be absolute or relative
# GUI uses: Path(__file__).parent.parent.parent.parent / "keys.env"
# Keep same style here for consistency across GUI/API folders.
DEFAULT_ENV_PATH = str(Path(__file__).parent.parent.parent.parent / "keys.env")


# ------------------ Prompt Templates (Settings) ------------------
# NOTE: These templates are stored in rag_settings.json and can be edited at runtime.
# Keep placeholders:
#  - {choose_k}, {query}, {candidates}, {context}

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

# DEFAULT_ANSWER_PROMPT_TEMPLATE = (
#     "You are a hybrid RAG assistant.\n\n"
#     "Decision rules:\n"
#     "1) If the Context contains information that directly answers the Question, answer using ONLY the Context.\n"
#     "2) If the Question is general knowledge and the Context is irrelevant/unrelated, ignore the Context and answer normally.\n"
#     "3) If the Context is related but does NOT contain the answer, reply exactly:\n"
#     "I don't know based on the provided context.\n\n"
#     "IMPORTANT:\n"
#     "- Do NOT mix Context knowledge with general knowledge.\n"
#     "- Use Context ONLY in case (1).\n\n"
#     "Context:\n{context}\n\n"
#     "Question: {query}\n\n"
#     "Answer:"
# )

# DEFAULT_ANSWER_PROMPT_TEMPLATE = (
#     "You are a hybrid RAG assistant.\n\n"
#     "Decision rules:\n"
#     "1) If the Context contains information that directly answers the Question, answer using ONLY the Context.\n"
#     "2) If the Question is general knowledge and the Context is irrelevant/unrelated, ignore the Context and answer normally.\n"
#     "Context:\n{context}\n\n"
#     "Question: {query}\n\n"
#     "Answer:"
# )

'''
You are a hybrid RAG assistant.

Decision rules:
1) If the Context contains information that directly answers the Question, answer using ONLY the Context.
2) If the Question is a general knowledge question and the Context is irrelevant or unrelated, ignore the Context and answer normally.

Context:\n{context}

Question: {query}

Answer:"
'''

DEFAULT_ANSWER_PROMPT_TEMPLATE = (
    "You are an assistant that answers strictly from retrieved context.\n"
    "IMPORTANT RULES:\n"
    "- Use ONLY the information in the Context.\n"
    "- If the Context does not contain the answer, reply exactly: \"I don't know based on the provided context.\"\n"
    "- Keep the answer clear and concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DEFAULT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "csv_rag")

DEFAULT_TOPK_RETRIEVE = 10
DEFAULT_TOPK_USE = 3
DEFAULT_ENABLE_RERANK = False
DEFAULT_TEXT_GROUP_LINES = 1

SETTINGS_FILE_NAME = "rag_settings.json"

# ------------------ In-memory logs/results ------------------

LOGS: List[Dict[str, Any]] = []
RESULTS: List[Dict[str, Any]] = []
LOCK = threading.Lock()

def log(msg: str, level: str = "INFO"):
    item = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "level": level, "message": msg}
    with LOCK:
        LOGS.append(item)

def add_result(obj: Dict[str, Any]):
    with LOCK:
        RESULTS.append(obj)

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

# ------------------ Data Structures ------------------

@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_file: str
    row_index: int

# ------------------ Settings Persistence ------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / SETTINGS_FILE_NAME

def _default_docs_dir() -> str:
    # Match GUI's default: rag_data/my_csvs/docs (store absolute here)
    project_root = SCRIPT_DIR.parent  # usually .../RAG if this file is under .../RAG_API
    default_docs_dir = str((project_root / "rag_data" / "my_csvs" / "docs").resolve())
    print("Default docs dir: ", default_docs_dir)
    return default_docs_dir

def default_settings_dict() -> Dict[str, Any]:
    return {
        "qdrant_url": DEFAULT_QDRANT_URL,
        "collection_prefix": DEFAULT_COLLECTION_PREFIX,
        "topk_retrieve": DEFAULT_TOPK_RETRIEVE,
        "topk_use": DEFAULT_TOPK_USE,
        "enable_rerank": DEFAULT_ENABLE_RERANK,
        "text_group_lines": DEFAULT_TEXT_GROUP_LINES,
        "embedding_model": EMBEDDING_MODEL,
        "main_model": DEFAULT_MODEL,
        # Optional: folder location for docs
        # "docs_dir": str((SCRIPT_DIR / "rag_data" / "my_csvs" / "docs").resolve()),
        "docs_dir": _default_docs_dir(),
        # Optional chunk settings for TXT
        "txt_chunk_chars": 900,
        "txt_overlap": 120,
        "batch_size": 64,
        "rerank_prompt_template": DEFAULT_RERANK_PROMPT_TEMPLATE,
        "answer_prompt_template": DEFAULT_ANSWER_PROMPT_TEMPLATE,
        "main_provider": DEFAULT_MAIN_PROVIDER,
        "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
        "env_path": DEFAULT_ENV_PATH,        
    }

def load_settings() -> Dict[str, Any]:
    defaults = default_settings_dict()
    if not SETTINGS_PATH.exists():
        try:
            SETTINGS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"Failed to create default settings file: {e}", "ERROR")
        return defaults

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
    except Exception as e:
        log(f"Settings file invalid; rewriting defaults. Error: {e}", "ERROR")
        try:
            SETTINGS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        except Exception:
            pass
        return defaults

    merged = dict(defaults)
    merged.update(data)
    return merged

def save_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    # merge with defaults so file always has full keys
    defaults = default_settings_dict()
    merged = dict(defaults)
    merged.update(new_settings)

    SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged

def reset_settings() -> Dict[str, Any]:
    defaults = default_settings_dict()
    SETTINGS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    return defaults

def _resolve_env_path(env_path: str) -> Path:
    p = (env_path or "").strip()
    if not p:
        return Path(DEFAULT_ENV_PATH)

    path = Path(p)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()

def load_env_from_settings(s: Dict[str, Any]) -> None:
    env_path = _resolve_env_path(s.get("env_path", ""))
    if env_path.exists():
        load_dotenv(env_path)
        log(f"[ENV] Loaded: {env_path}")
    else:
        log(f"[ENV] Not found: {env_path}", "WARN")

# ------------------ RAG Engine (Qdrant + Ollama) ------------------

class RAGEngineQdrant:
    def __init__(self, qdrant_url: str, api_key: str, collection_prefix: str,
            rerank_prompt_template: str = DEFAULT_RERANK_PROMPT_TEMPLATE,
            answer_prompt_template: str = DEFAULT_ANSWER_PROMPT_TEMPLATE,):
        self.qdrant_url = qdrant_url
        self.api_key = api_key or None
        self.collection_prefix = collection_prefix

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)

        self.embedding_model = EMBEDDING_MODEL
        self.main_model = DEFAULT_MODEL

        self.collection_name: Optional[str] = None
        self.vector_dim: Optional[int] = None

        self.rerank_prompt_template = rerank_prompt_template
        self.answer_prompt_template = answer_prompt_template      

        # Providers
        self.main_provider = OLLAMA_PROVIDER
        self.embedding_provider = OLLAMA_PROVIDER
        self._gemini_client: Optional[genai.Client] = None
        self._openai_client: Optional[OpenAI] = None          

    @staticmethod
    def _slug(s: str) -> str:
        return s.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")

    def set_provider(self, main_provider: str, embedding_provider: str):
        self.main_provider = (main_provider or OLLAMA_PROVIDER).strip().lower()
        self.embedding_provider = (embedding_provider or OLLAMA_PROVIDER).strip().lower()

        if self.main_provider not in ALL_PROVIDERS:
            raise ValueError(f"Invalid main_provider: {self.main_provider}. Must be one of {ALL_PROVIDERS}")
        if self.embedding_provider not in ALL_PROVIDERS:
            raise ValueError(f"Invalid embedding_provider: {self.embedding_provider}. Must be one of {ALL_PROVIDERS}")

        providers = {self.main_provider, self.embedding_provider}

        if GEMINI_PROVIDER in providers:
            api_key = os.getenv(GEMINI_API_KEY_ENV_VAR, "")
            if not api_key:
                raise ValueError("Gemini provider selected but GEMINI_API_KEY is missing (check env_path).")
            self._gemini_client = genai.Client(api_key=api_key)

        if OPENAI_PROVIDER in providers:
            api_key = os.getenv(OPENAI_API_KEY_ENV_VAR, "")
            if not api_key:
                raise ValueError("OpenAI provider selected but OPENAI_API_KEY is missing (check env_path).")
            self._openai_client = OpenAI(api_key=api_key)

    def set_models(self, embedding_model: str, main_model: str):
        self.embedding_model = embedding_model
        self.main_model = main_model
        self.collection_name = None
        self.vector_dim = None

    def _embed(self, text: str) -> np.ndarray:
        text = text or ""
        if self.embedding_provider == GEMINI_PROVIDER:
            if not self._gemini_client:
                raise RuntimeError("Gemini client not initialized. Check env_path and embedding_provider.")
            resp = self._gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=text,
            )
            return np.array(resp.embeddings[0].values, dtype=np.float32)

        if self.embedding_provider == OPENAI_PROVIDER:
            if not self._openai_client:
                raise RuntimeError("OpenAI client not initialized. Check env_path and embedding_provider.")
            resp = self._openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        # Default: Ollama        
        resp = ollama.embeddings(model=self.embedding_model, prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)

    def _ensure_collection(self, dim: int) -> str:
        collection = f"{self.collection_prefix}__{self._slug(self.embedding_model)}__dim{dim}"
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )
        self.collection_name = collection
        self.vector_dim = dim
        return collection

    @staticmethod
    def _stable_chunk_id(source_file: str, row_index: int, text: str) -> str:
        ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
        key = f"{source_file}::{row_index}::{text}"
        return str(uuid.uuid5(ns, key))

    @staticmethod
    def _row_to_chunk_text(df: pd.DataFrame, row: pd.Series) -> str:
        pairs = []
        for col in df.columns:
            val = row.get(col)
            if pd.notna(val):
                pairs.append(f"{col}={val}")
        return " | ".join(pairs).strip()

    @staticmethod
    def _split_text_to_chunks(
        text: str,
        chunk_chars: int = 900,
        overlap: int = 120,
        group_lines: int = 1,
    ) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if overlap >= chunk_chars:
            overlap = max(0, chunk_chars // 4)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        blocks = []
        if group_lines <= 1:
            blocks = lines
        else:
            for i in range(0, len(lines), group_lines):
                blocks.append("\n".join(lines[i:i + group_lines]))

        chunks: list[str] = []

        def split_long_block(block: str):
            n = len(block)
            start = 0
            while start < n:
                end = min(n, start + chunk_chars)
                part = block[start:end].strip()
                if part:
                    chunks.append(part)
                if end >= n:
                    break
                start = max(0, end - overlap)

        for b in blocks:
            if len(b) <= chunk_chars:
                chunks.append(b)
            else:
                split_long_block(b)

        return chunks

    def build_from_folder(
        self,
        docs_dir: Path,
        batch_size: int = 64,
        txt_chunk_chars: int = 900,
        txt_overlap: int = 120,
        group_lines: int = 1,
    ) -> int:
        docs_dir = Path(docs_dir)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

        records: list[ChunkRecord] = []

        # CSV
        for csv_path in sorted(docs_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            for i, row in df.iterrows():
                text = self._row_to_chunk_text(df, row)
                if not text:
                    continue
                chunk_id = self._stable_chunk_id(csv_path.name, int(i), text)
                records.append(ChunkRecord(chunk_id, text, csv_path.name, int(i)))

        # TXT
        for txt_path in sorted(docs_dir.glob("*.txt")):
            content = txt_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._split_text_to_chunks(
                content, chunk_chars=txt_chunk_chars, overlap=txt_overlap, group_lines=group_lines
            )
            for idx, ch in enumerate(chunks):
                if not ch:
                    continue
                chunk_id = self._stable_chunk_id(txt_path.name, int(idx), ch)
                records.append(ChunkRecord(chunk_id, ch, txt_path.name, int(idx)))

        if not records:
            return 0

        dim = int(self._embed("dimension probe").shape[0])
        self._ensure_collection(dim)

        total = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            points: list[qmodels.PointStruct] = []
            for rec in batch:
                vec = self._embed(rec.text)
                payload = {
                    "text": rec.text,
                    "source_file": rec.source_file,
                    "row_index": rec.row_index,
                    "embedding_model": self.embedding_model,
                }
                points.append(qmodels.PointStruct(id=rec.chunk_id, vector=vec.tolist(), payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)

        return total

    def _ensure_ready_for_query(self):
        if self.collection_name is not None and self.vector_dim is not None:
            return
        dim = int(self._embed("dimension probe").shape[0])
        self._ensure_collection(dim)

    def search(self, query: str, top_k: int) -> list[dict]:
        query = query.strip()
        if not query:
            return []

        self._ensure_ready_for_query()
        qvec = self._embed(query).tolist()

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )

        hits = res.points
        results = []
        for h in hits:
            payload = h.payload or {}
            results.append({
                "score": float(h.score),
                "text": str(payload.get("text", "")),
                "source_file": str(payload.get("source_file", "")),
                "row_index": int(payload.get("row_index", -1)),
            })
        return results

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def rerank_with_llm(self, query: str, hits: list[dict], choose_k: int) -> list[int]:
        if not hits:
            return []

        choose_k = max(1, int(choose_k))
        choose_k = min(choose_k, len(hits))

        candidates_lines = []
        for idx, h in enumerate(hits):
            candidates_lines.append(
                f"{idx}: (score={h['score']:.4f}, file={h['source_file']}, row={h['row_index']}) {h['text']}"
            )
        candidates = "\n".join(candidates_lines)

        # rerank_prompt = (
        #     "You are a retrieval re-ranker.\n"
        #     "Given a user question and a list of candidate contexts, select the most relevant items.\n"
        #     "Rules:\n"
        #     f"- Choose exactly {choose_k} distinct indices.\n"
        #     "- Prefer contexts that directly contain facts needed to answer.\n"
        #     "- Avoid redundant/duplicate contexts.\n"
        #     "- Output ONLY valid JSON, no extra text.\n\n"
        #     "{\n"
        #     '  "selected_indices": [0, 2, 5],\n'
        #     '  "reasons": ["short reason 1", "short reason 2", "short reason 3"]\n'
        #     "}\n\n"
        #     f"Question:\n{query}\n\n"
        #     f"Candidates:\n{candidates}\n"
        # )

        rerank_prompt = self.rerank_prompt_template.format(
            choose_k=choose_k,
            query=query,
            candidates=candidates,
        )

        try:
            # resp = ollama.chat(model=self.main_model, messages=[{"role": "user", "content": rerank_prompt}])
            # out = resp["message"]["content"]
            out = self._call_chat(rerank_prompt)
            obj = self._extract_json_object(out)
            if not obj:
                raise ValueError("No JSON object found in rerank output.")
            
            indices = obj.get("selected_indices", [])
            if not isinstance(indices, list):
                raise ValueError("selected_indices is not a list")

            clean = []
            for x in indices:
                if isinstance(x, int) and 0 <= x < len(hits) and x not in clean:
                    clean.append(x)

            if len(clean) < choose_k:
                for i in range(len(hits)):
                    if i not in clean:
                        clean.append(i)
                    if len(clean) == choose_k:
                        break

            return clean[:choose_k]
        except Exception:
            return list(range(choose_k))

    @staticmethod
    def build_context(hits: list[dict], indices: list[int]) -> str:
        blocks = []
        for rank, i in enumerate(indices, start=1):
            h = hits[i]
            blocks.append(
                f"[#{rank} | score={h['score']:.4f} | file={h['source_file']} | row={h['row_index']}]\n"
                f"{h['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(
        self,
        query: str,
        top_k_retrieve: int,
        top_k_use: int,
        enable_rerank: bool,
    ) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            return {"answer": "", "context": "", "hits": [], "selected": [], "latency": 0.0}

        hits = self.search(query, top_k=int(top_k_retrieve))
        if not hits:
            return {
                "answer": "No retrieved data. Build the index first.",
                "context": "",
                "hits": [],
                "selected": [],
                "latency": 0.0,
            }

        if enable_rerank:
            selected = self.rerank_with_llm(query, hits, choose_k=int(top_k_use))
        else:
            selected = list(range(min(int(top_k_use), len(hits))))

        context = self.build_context(hits, selected)

#         prompt = f"""
# You are a RAG router + answerer.

# You have:
# - INPUT (user message)
# - CONTEXT (retrieved snippets, may be irrelevant)

# Your job:
# 1) If CONTEXT contains an EXACT direct answer to the INPUT, return that answer.
# 2) Else if CONTEXT is clearly RELATED to the INPUT, return the BEST nearest answer by paraphrasing from CONTEXT.
# 3) Else (CONTEXT is unrelated / low relevance), IGNORE CONTEXT and answer the INPUT normally as a general assistant.

# Rules:
# - For (1) and (2): you MUST use the CONTEXT only.
# - For (3): you MUST ignore the CONTEXT completely and answer from general knowledge.
# - Do not mention these rules or the word "context".

# Output format (STRICT):
# Mode: EXACT | NEAREST | GENERAL
# Answer: <your answer>

# INPUT:
# {query}

# CONTEXT:
# {context}
# """.strip()

        # prompt = (
        #     "You are an assistant that answers strictly from retrieved context.\n"
        #     "IMPORTANT RULES:\n"
        #     "- Use ONLY the information in the Context.\n"
        #     "- If the Context does not contain the answer, reply exactly: \"I don't know based on the provided context.\"\n"
        #     "- Keep the answer clear and concise.\n\n"
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     "Answer:"
        # )

        prompt = self.answer_prompt_template.format(
            context=context,
            query=query,
        )

        # start = time.time()
        # resp = ollama.chat(model=self.main_model, messages=[{"role": "user", "content": prompt}])
        # latency = time.time() - start

        # return {
        #     "answer": resp["message"]["content"],
        #     "context": context,
        #     "hits": hits,
        #     "selected": selected,
        #     "latency": latency,
        #     "collection": self.collection_name,
        # }
        start = time.time()
        answer_text = self._call_chat(prompt)
        latency = time.time() - start

        return {
            "answer": answer_text,
            "context": context,
            "hits": hits,
            "selected": selected,
            "latency": latency,
            "collection": self.collection_name,
        }


    def _call_chat(self, prompt: str) -> str:
        if self.main_provider == GEMINI_PROVIDER:
            if not self._gemini_client:
                raise RuntimeError("Gemini client not initialized. Check env_path and main_provider.")
            r = self._gemini_client.models.generate_content(
                model=self.main_model,
                contents=prompt,
            )
            return r.text or ""

        if self.main_provider == OPENAI_PROVIDER:
            if not self._openai_client:
                raise RuntimeError("OpenAI client not initialized. Check env_path and main_provider.")
            try:
                r = self._openai_client.responses.create(
                    model=self.main_model,
                    input=prompt,
                )
                return r.output_text or ""
            except Exception:
                r = self._openai_client.chat.completions.create(
                    model=self.main_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.choices[0].message.content or ""

        resp = ollama.chat(model=self.main_model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]
# ------------------ Engine Factory ------------------

def make_engine_from_settings(s: Dict[str, Any]) -> RAGEngineQdrant:
    load_env_from_settings(s)

    eng = RAGEngineQdrant(
        qdrant_url=s.get("qdrant_url", DEFAULT_QDRANT_URL),
        api_key=DEFAULT_QDRANT_API_KEY,
        collection_prefix=s.get("collection_prefix", DEFAULT_COLLECTION_PREFIX),
        rerank_prompt_template=s.get("rerank_prompt_template", DEFAULT_RERANK_PROMPT_TEMPLATE),
        answer_prompt_template=s.get("answer_prompt_template", DEFAULT_ANSWER_PROMPT_TEMPLATE),        
    )
    emb = s.get("embedding_model", EMBEDDING_MODEL)
    main = s.get("main_model", DEFAULT_MODEL)
    eng.set_models(emb, main)
    eng.set_provider(
        main_provider=s.get("main_provider", DEFAULT_MAIN_PROVIDER),
        embedding_provider=s.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER),
    )

    return eng

# ------------------ FastAPI Models ------------------
class SettingsIn(BaseModel):
    qdrant_url: Optional[str] = None
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

class QueryIn(BaseModel):
    query: str = Field(..., min_length=1)
    topk_retrieve: Optional[int] = None
    topk_use: Optional[int] = None
    enable_rerank: Optional[bool] = None

class BuildIndexIn(BaseModel):
    docs_dir: Optional[str] = None
    batch_size: Optional[int] = None
    txt_chunk_chars: Optional[int] = None
    txt_overlap: Optional[int] = None
    group_lines: Optional[int] = None

# ------------------ FastAPI App ------------------

app = FastAPI(title=APP_TITLE)

@app.get("/health")
def api_health():
    return {"status": "healthy"}

@app.get("/")
def api_root():
    return {"message": "RAG Fast API is running => " + APP_TITLE}

@app.on_event("startup")
def _startup():
    s = load_settings()
    load_env_from_settings(s)
    log(f"API started. Settings loaded from {SETTINGS_PATH}")
    log(f"Qdrant={s.get('qdrant_url')} prefix={s.get('collection_prefix')}")
    log(f"Providers: main={s.get('main_provider')} embedding={s.get('embedding_provider')}")
    log(f"Models: embedding={s.get('embedding_model')} main={s.get('main_model')}")

# ------------------ Settings Endpoints ------------------

@app.get("/settings")
def api_get_settings():
    return load_settings()

@app.post("/settings")
def api_save_settings(body: SettingsIn):
    current = load_settings()
    upd = {k: v for k, v in body.model_dump().items() if v is not None}
    merged = save_settings({**current, **upd})
    load_env_from_settings(merged)
    log("Settings saved.")
    return merged

@app.post("/settings/reset")
def api_reset_settings():
    s = reset_settings()
    load_env_from_settings(s)
    log("Settings reset to defaults.")
    return s

# ------------------ Model Lists (Main/Embedding by Provider) ------------------

def _models_main_by_provider(provider: str) -> List[str]:
    p = (provider or "").strip().lower()
    if p == OLLAMA_PROVIDER:
        return list(CHAT_MODELS_OLLAMA or CHAT_MODELS or [])
    if p == GEMINI_PROVIDER:
        return list(CHAT_MODELS_GEMINI or [])
    if p == OPENAI_PROVIDER:
        return list(CHAT_MODELS_OPENAI or [])
    return []

def _models_embedding_by_provider(provider: str) -> List[str]:
    p = (provider or "").strip().lower()
    if p == OLLAMA_PROVIDER:
        return list(EMBEDDING_MODELS_OLLAMA or EMBEDDING_MODELS or ([EMBEDDING_MODEL] if EMBEDDING_MODEL else []))
    if p == GEMINI_PROVIDER:
        return list(EMBEDDING_MODELS_GEMINI or [])
    if p == OPENAI_PROVIDER:
        return list(EMBEDDING_MODELS_OPENAI or [])
    return []

@app.get("/models/main/ollama")
def api_models_main_ollama():
    return {"provider": OLLAMA_PROVIDER, "models": _models_main_by_provider(OLLAMA_PROVIDER)}

@app.get("/models/main/gemini")
def api_models_main_gemini():
    return {"provider": GEMINI_PROVIDER, "models": _models_main_by_provider(GEMINI_PROVIDER)}

@app.get("/models/main/openai")
def api_models_main_openai():
    return {"provider": OPENAI_PROVIDER, "models": _models_main_by_provider(OPENAI_PROVIDER)}

@app.get("/models/embedding/ollama")
def api_models_embedding_ollama():
    return {"provider": OLLAMA_PROVIDER, "models": _models_embedding_by_provider(OLLAMA_PROVIDER)}

@app.get("/models/embedding/gemini")
def api_models_embedding_gemini():
    return {"provider": GEMINI_PROVIDER, "models": _models_embedding_by_provider(GEMINI_PROVIDER)}

@app.get("/models/embedding/openai")
def api_models_embedding_openai():
    return {"provider": OPENAI_PROVIDER, "models": _models_embedding_by_provider(OPENAI_PROVIDER)}

# ------------------ Ping Qdrant ------------------

@app.get("/ping/qdrant")
def api_ping_qdrant():
    s = load_settings()
    try:
        eng = make_engine_from_settings(s)
        cols = eng.client.get_collections()
        log(f"Qdrant OK. collections={len(cols.collections)}")
        return {"ok": True, "collections_count": len(cols.collections)}
    except Exception as e:
        log(f"Qdrant ping failed: {e}", "ERROR")
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {e}")

# ------------------ Build Index ------------------

# @app.post("/index/build")
# def api_build_index(body: BuildIndexIn):
#     s = load_settings()

#     docs_dir = body.docs_dir or s.get("docs_dir")
#     if not docs_dir:
#         raise HTTPException(status_code=400, detail="docs_dir is required (in request or settings).")

#     batch_size = body.batch_size or _safe_int(s.get("batch_size", 64), 64)
#     txt_chunk_chars = body.txt_chunk_chars or _safe_int(s.get("txt_chunk_chars", 900), 900)
#     txt_overlap = body.txt_overlap or _safe_int(s.get("txt_overlap", 120), 120)
#     group_lines = body.group_lines or _safe_int(s.get("text_group_lines", 1), 1)

#     try:
#         eng = make_engine_from_settings(s)
#         total = eng.build_from_folder(
#             Path(docs_dir),
#             batch_size=batch_size,
#             txt_chunk_chars=txt_chunk_chars,
#             txt_overlap=txt_overlap,
#             group_lines=group_lines,
#         )
#         log(f"Index build done. upserted={total} collection={eng.collection_name}")
#         return {"upserted": total, "collection": eng.collection_name}
#     except Exception as e:
#         log(f"Index build failed: {e}", "ERROR")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/build")
def api_build_index(body: BuildIndexIn | None = Body(default=None)):
    s = load_settings()

    body = body or BuildIndexIn()  # handle None body

    docs_dir = body.docs_dir or s.get("docs_dir")
    if not docs_dir:
        raise HTTPException(status_code=400, detail="docs_dir is required (in request or settings).")

    batch_size = body.batch_size or _safe_int(s.get("batch_size", 64), 64)
    txt_chunk_chars = body.txt_chunk_chars or _safe_int(s.get("txt_chunk_chars", 900), 900)
    txt_overlap = body.txt_overlap or _safe_int(s.get("txt_overlap", 120), 120)
    group_lines = body.group_lines or _safe_int(s.get("text_group_lines", 1), 1)

    try:
        eng = make_engine_from_settings(s)
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

# ------------------ Run Query ------------------

@app.post("/query")
def api_run_query(body: QueryIn):
    s = load_settings()

    topk_retrieve = body.topk_retrieve if body.topk_retrieve is not None else _safe_int(s.get("topk_retrieve", DEFAULT_TOPK_RETRIEVE), DEFAULT_TOPK_RETRIEVE)
    topk_use = body.topk_use if body.topk_use is not None else _safe_int(s.get("topk_use", DEFAULT_TOPK_USE), DEFAULT_TOPK_USE)
    enable_rerank = body.enable_rerank if body.enable_rerank is not None else _safe_bool(s.get("enable_rerank", DEFAULT_ENABLE_RERANK), DEFAULT_ENABLE_RERANK)
    print(f"API Query: topk_retrieve={topk_retrieve} topk_use={topk_use} enable_rerank={enable_rerank}")
    try:
        eng = make_engine_from_settings(s)
        result = eng.answer(
            body.query,
            top_k_retrieve=topk_retrieve,
            top_k_use=topk_use,
            enable_rerank=enable_rerank,
        )
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": body.query,
            "settings": {"topk_retrieve": topk_retrieve, "topk_use": topk_use, "enable_rerank": enable_rerank},
            "provider": {"main": s.get("main_provider"), "embedding": s.get("embedding_provider")},
            "models": {"main": s.get("main_model"), "embedding": s.get("embedding_model")},
            "result": result,
        }
        add_result(payload)
        log(f"Query done. collection={result.get('collection')} latency={result.get('latency'):.3f}s")
        return payload
    except Exception as e:
        log(f"Query failed: {e}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))
# ------------------ Get all results ------------------

@app.get("/results")
def api_get_results(limit: int = 100, offset: int = 0):
    with LOCK:
        items = RESULTS[offset:offset + limit]
        total = len(RESULTS)
    return {"total": total, "limit": limit, "offset": offset, "items": items}

@app.delete("/results")
def api_clear_results():
    with LOCK:
        RESULTS.clear()
    log("Results cleared.")
    return {"ok": True}

# ------------------ Get logs ------------------

@app.get("/logs")
def api_get_logs(limit: int = 200, offset: int = 0):
    with LOCK:
        items = LOGS[offset:offset + limit]
        total = len(LOGS)
    return {"total": total, "limit": limit, "offset": offset, "items": items}

@app.delete("/logs")
def api_clear_logs():
    with LOCK:
        LOGS.clear()
    log("Logs cleared.")
    return {"ok": True}
