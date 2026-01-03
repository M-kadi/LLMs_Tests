import os
from pyexpat.errors import messages
import re
import json
import sys
from tkinter import filedialog
import uuid
import time
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# from flask.cli import 
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict
from google import genai
from openai import OpenAI
import ollama
from qdrant_client import QdrantClient, models as qmodels
sys.path.insert(0, str(Path(__file__).parent.parent))
from models_config import CHAT_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL, EMBEDDING_MODELS, \
    CHAT_MODELS_GEMINI, CHAT_MODELS_OPENAI, EMBEDDING_MODELS_OPENAI, EMBEDDING_MODELS_GEMINI
    # DEFAULT_MODEL_GEMINI, EMBEDDING_MODEL_GEMINI, DEFAULT_MODEL_OPENAI, EMBEDDING_MODEL_OPENAI, \

# '''
# To Run:
# - Start Docker Desktop
# - Start Ollama application
# - Start Qdrant locally: from command line, run:
#  D:\LLM\LLMs_Tests\RAG\RAG_GUI>
    # docker run -p 6333:6333 -p 6334:6334 ^
    # -v %cd%\qdrant_data:/qdrant/storage ^
    # qdrant/qdrant
# - Then run this script:
#     PS D:\LLM\LLMs_Tests\RAG\RAG_GUI>python.exe rag_gui_qdrant.py
# - To Install Qdrant by docker: 
#    C:\Users\Mohammed>docker pull qdrant/qdrant
# '''
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
# Enable change Prompt and Prompt templates for Reranking and Answering from settings GUI
# Use Gemeni / OpenAI / Ollama for embeddings + chat
# Gemeni and OpenAI need API keys in keys.env file
# ------------------ App Defaults ------------------

APP_TITLE = "RAG GUI (CSV/TXT) : Qdrant Local + Ollama (Embeddings + Chat)"

# ------------------ Settings Persistence ------------------
SETTINGS_FILE_NAME = "rag_settings.json"
RAG_DATA_0_FOLDER_NAME = "rag_data"
RAG_DATA_1_FOLDER_NAME = "my_csvs"
RAG_DATA_2_FOLDER_NAME = "docs"
# rag_data\my_csvs\docs

OLLAMA_PROVIDER = "ollama"
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"  # treat OpenAI as Gemini for this RAG engine
ALL_PROVIDERS = [OLLAMA_PROVIDER, GEMINI_PROVIDER, OPENAI_PROVIDER]
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

# ------------------ Default Config ------------------
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DEFAULT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "csv_rag")

DEFAULT_TOPK_RETRIEVE = 10  # retrieve from Qdrant
DEFAULT_TOPK_USE = 3        # final contexts used in answer
DEFAULT_ENABLE_RERANK = False  # disable reranking by default
DEFAULT_TEXT_GROUP_LINES = 1  # lines per chunk for TXT files

DEFAULT_DOCS_DIR = str(Path(RAG_DATA_0_FOLDER_NAME) / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME)

DEFAULT_PROVIDER = OLLAMA_PROVIDER   # "ollama" | "gemini" | "openai"
DEFAULT_ENV_PATH = str(Path(__file__).parent.parent.parent.parent / "keys.env")     # keys file path for Gemini API key 
# ENV_PATH = Path(__file__).parent.parent.parent / "keys.env"
print("DEFAULT_ENV_PATH:", DEFAULT_ENV_PATH)

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
    "You are an assistant that answers strictly from retrieved context.\n"
    "IMPORTANT RULES:\n"
    "- Use ONLY the information in the Context.\n"
    "- If the Context does not contain the answer, reply exactly: \"I don't know based on the provided context.\"\n"
    "- Keep the answer clear and concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

'''
You are a hybrid RAG assistant.

Decision rules:
1) If the Context contains information that directly answers the Question, answer using ONLY the Context.
2) If the Question is a general knowledge question and the Context is irrelevant or unrelated, ignore the Context and answer normally.

Context:\n{context}

Question: {query}

Answer:"
'''

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


# ------------------ RAG Engine (Qdrant + Ollama) ------------------

class RAGEngineQdrant:
    """
    Production-minded RAG core:
    - Qdrant for vectors
    - Ollama for embeddings + chat
    - Retrieve top_k=10, rerank to best 3 using LLM, then answer with selected 3
    """

    def __init__(self, qdrant_url: str, api_key: str, collection_prefix: str):
        self.qdrant_url = qdrant_url
        self.api_key = api_key or None
        self.collection_prefix = collection_prefix

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)

        self.embedding_model = EMBEDDING_MODEL
        self.main_model = DEFAULT_MODEL

        self.collection_name: Optional[str] = None
        self.vector_dim: Optional[int] = None

        self.rerank_prompt_template = DEFAULT_RERANK_PROMPT_TEMPLATE
        self.answer_prompt_template = DEFAULT_ANSWER_PROMPT_TEMPLATE

    def set_provider(
        self,
        main_provider: str,
        embedding_provider: str,
        # cloud_api_key: str = "",
        # gemini_chat_model: str = DEFAULT_GEMINI_CHAT_MODEL,
        # gemini_embed_model: str = DEFAULT_GEMINI_EMBED_MODEL,
    ):
        self.main_provider = (main_provider or OLLAMA_PROVIDER).strip().lower()
        self.embedding_provider = (embedding_provider or OLLAMA_PROVIDER).strip().lower()
        self.providers = [self.main_provider, self.embedding_provider]
        # self.cloud_api_key = (cloud_api_key or self.cloud_api_key or "").strip()
        # self.gemini_chat_model = gemini_chat_model
        # self.gemini_embed_model = gemini_embed_model

        if GEMINI_PROVIDER in self.providers:
            self.cloud_api_key = os.getenv(GEMINI_API_KEY_ENV_VAR, "")
            if not self.cloud_api_key:
                raise ValueError("Gemini provider selected but GEMINI_API_KEY is missing.")
            self._gemini_client = genai.Client(api_key=self.cloud_api_key)

        if OPENAI_PROVIDER in self.providers:
            self.cloud_api_key = os.getenv(OPENAI_API_KEY_ENV_VAR, "")
            if not self.cloud_api_key:
                raise ValueError("OpenAI provider selected but OPENAI_API_KEY is missing.")
            self._openai_client = OpenAI(api_key=self.cloud_api_key)

    def set_prompts(self, rerank_template: str, answer_template: str):
        self.rerank_prompt_template = (rerank_template or "").strip() or DEFAULT_RERANK_PROMPT_TEMPLATE
        self.answer_prompt_template = (answer_template or "").strip() or DEFAULT_ANSWER_PROMPT_TEMPLATE

    @staticmethod
    def _slug(s: str) -> str:
        return (
            s.replace(":", "_")
             .replace("/", "_")
             .replace("\\", "_")
             .replace(" ", "_")
        )

    def set_models(self, embedding_model: str, main_model: str):
        self.embedding_model = embedding_model
        self.main_model = main_model
        # Force re-ensure collection on next operations (embedding model may change dim)
        self.collection_name = None
        self.vector_dim = None

    def _embed(self, text: str) -> np.ndarray:
        text = text or ""
        if self.embedding_provider == GEMINI_PROVIDER:
            # Gemini embeddings
            resp = self._gemini_client.models.embed_content(
                model=self.embedding_model, #gemini_embed_model,
                contents=text,
            )
            vec = np.array(resp.embeddings[0].values, dtype=np.float32)
            return vec    
           
        if self.embedding_provider == OPENAI_PROVIDER:
            # OpenAI embeddings
            resp = self._openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            return vec
        print("OLLAMA embedding model:", self.embedding_provider)                   
        # Ollama embeddings 
        resp = ollama.embeddings(model=self.embedding_model, prompt=text)
        vec = np.array(resp["embedding"], dtype=np.float32)
        return vec

    def _ensure_collection(self, dim: int) -> str:
        """
        Create collection if missing. Uses COSINE distance.
        Collection name includes embedding model + dim to avoid mismatches.
        """
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
        """
        Deterministic UUID so rebuilds don't create duplicates.
        """
        ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
        key = f"{source_file}::{row_index}::{text}"
        return str(uuid.uuid5(ns, key))

    @staticmethod
    def _row_to_chunk_text(df: pd.DataFrame, row: pd.Series) -> str:
        # Keep column names for better faithfulness and search.
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
        """
        Line-aware chunking:
        - Normally: each line becomes one chunk (or group of N lines if group_lines > 1)
        - If a single line (or grouped block) is longer than chunk_chars, split it into sub-chunks
        using overlap.
        """
        text = (text or "").strip()
        if not text:
            return []

        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_chars:
            overlap = max(0, chunk_chars // 4)

        # Keep non-empty lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        # Optionally group lines (e.g., 3 lines per chunk)
        blocks = []
        if group_lines <= 1:
            blocks = lines
        else:
            for i in range(0, len(lines), group_lines):
                blocks.append("\n".join(lines[i:i + group_lines]))
            # strChunk = ""
            # for i in range(0, len(lines), group_lines):
            #     strLine = lines[i:i + group_lines]
            #     if(strLine != "#"):
            #         strChunk = strChunk.join(strLine)
            #     else:
            #         blocks.append("\n".join(strChunk))
            #         strChunk = ""
                

        chunks: list[str] = []

        def split_long_block(block: str):
            # split a long block into overlapping sub-chunks
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

    def _embed_safe(self, text: str) -> np.ndarray:
        """
        Embed with auto-shrink if input exceeds model context.
        Prevents ingestion failures with big TXT chunks.
        """
        t = (text or "").strip()
        if not t:
            return self._embed("")

        try:
            return self._embed(t)
        except Exception as e:
            msg = str(e).lower()
            if "context length" in msg or "exceeds" in msg:
                for factor in (0.7, 0.5, 0.35, 0.25, 0.15):
                    cut = max(200, int(len(t) * factor))
                    try:
                        return self._embed(t[:cut])
                    except Exception:
                        continue
            raise
    
    # def build_from_csv_folder(self, csv_dir: Path, batch_size: int = 64) -> int:
    #     """
    #     Read CSVs -> make chunks -> embed -> upsert into Qdrant
    #     """
    #     csv_dir = Path(csv_dir)
    #     if not csv_dir.exists():
    #         raise FileNotFoundError(f"CSV folder not found: {csv_dir}")

    #     records: list[ChunkRecord] = []
    #     for csv_path in sorted(csv_dir.glob("*.csv")):
    #         try:
    #             df = pd.read_csv(csv_path)
    #         except Exception as e:
    #             raise RuntimeError(f"Failed reading {csv_path.name}: {e}") from e

    #         for i, row in df.iterrows():
    #             text = self._row_to_chunk_text(df, row)
    #             if not text:
    #                 continue
    #             chunk_id = self._stable_chunk_id(csv_path.name, int(i), text)
    #             records.append(ChunkRecord(chunk_id=chunk_id, text=text, source_file=csv_path.name, row_index=int(i)))

    #     if not records:
    #         return 0

    #     # Probe dimension once
    #     dim = int(self._embed("dimension probe").shape[0])
    #     self._ensure_collection(dim)

    #     total = 0
    #     for start in range(0, len(records), batch_size):
    #         batch = records[start:start + batch_size]
    #         points: list[qmodels.PointStruct] = []

    #         for rec in batch:
    #             vec = self._embed(rec.text)
    #             if vec.shape[0] != dim:
    #                 raise RuntimeError(f"Embedding dim mismatch: expected {dim}, got {vec.shape[0]}")

    #             payload = {
    #                 "text": rec.text,
    #                 "source_file": rec.source_file,
    #                 "row_index": rec.row_index,
    #                 "embedding_model": self.embedding_model,
    #             }

    #             points.append(
    #                 qmodels.PointStruct(
    #                     id=rec.chunk_id,
    #                     vector=vec.tolist(),
    #                     payload=payload,
    #                 )
    #             )

    #         self.client.upsert(collection_name=self.collection_name, points=points)
    #         total += len(points)

    #     return total

    def build_from_folder(
        self,
        docs_dir: Path,
        batch_size: int = 64,
        txt_chunk_chars: int = 900,
        txt_overlap: int = 120,
        group_lines: int = 1,
    ) -> int:
        """
        Read CSV + TXT files from a folder -> embed -> upsert into Qdrant
        - CSV: each row = one chunk
        - TXT: split into overlapping chunks
        """
        docs_dir = Path(docs_dir)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

        records: list[ChunkRecord] = []

        # ---------- CSV ingestion ----------
        for csv_path in sorted(docs_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                raise RuntimeError(f"Failed reading {csv_path.name}: {e}") from e

            for i, row in df.iterrows():
                text = self._row_to_chunk_text(df, row)
                if not text:
                    continue
                chunk_id = self._stable_chunk_id(csv_path.name, int(i), text)
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=text,
                        source_file=csv_path.name,
                        row_index=int(i),
                    )
                )

        # ---------- TXT ingestion ----------
        for txt_path in sorted(docs_dir.glob("*.txt")):
            try:
                content = txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                raise RuntimeError(f"Failed reading {txt_path.name}: {e}") from e

            chunks = self._split_text_to_chunks(content, chunk_chars=txt_chunk_chars, overlap=txt_overlap,
                                                group_lines=group_lines)
            for idx, ch in enumerate(chunks):
                if not ch:
                    continue
                chunk_id = self._stable_chunk_id(txt_path.name, int(idx), ch)
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=ch,
                        source_file=txt_path.name,
                        row_index=int(idx),  # for TXT this means chunk index
                    )
                )

        if not records:
            return 0

        # Probe dimension once
        dim = int(self._embed("dimension probe").shape[0])
        self._ensure_collection(dim)

        total = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            points: list[qmodels.PointStruct] = []

            for rec in batch:
                # vec = self._embed_safe(rec.text)   # <-- safe embed _embed
                vec = self._embed(rec.text)   # <-- safe embed _embed
                if vec.shape[0] != dim:
                    raise RuntimeError(f"Embedding dim mismatch: expected {dim}, got {vec.shape[0]}")

                payload = {
                    "text": rec.text,
                    "source_file": rec.source_file,
                    "row_index": rec.row_index,
                    "embedding_model": self.embedding_model,
                }

                points.append(
                    qmodels.PointStruct(
                        id=rec.chunk_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)

        return total

    def _ensure_ready_for_query(self):
        if self.collection_name is not None and self.vector_dim is not None:
            return
        dim = int(self._embed("dimension probe").shape[0])
        self._ensure_collection(dim)

    def search(self, query: str, top_k: int) -> list[dict]:
        """
        Search Qdrant; return hits with payload.
        """
        query = query.strip()
        if not query:
            return []

        self._ensure_ready_for_query()

        qvec = self._embed(query).tolist()
        # hits = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=qvec,
        #     limit=int(top_k),
        #     with_payload=True,
        #     with_vectors=False,
        # )

        # qvec is the embedding vector (list[float])
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,              # <-- IMPORTANT: query= vector
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,      # optional, safe
        )

        hits = res.points  # list of ScoredPoint

        results = []
        for h in hits:
            payload = h.payload or {}
            results.append({
                "score": float(h.score),
                "text": str(payload.get("text", "")),
                "source_file": str(payload.get("source_file", "")),
                "row_index": int(payload.get("row_index", -1)),
            })
        # for p in hits:
        #     payload = p.payload or {}
        #     results.append({
        #         "score": float(p.score),
        #         "text": str(payload.get("text", "")),
        #         "source_name": str(payload.get("source_name", "")),
        #         "chunk_no": int(payload.get("chunk_no", -1)),
        #     })        
        return results

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        """
        Attempts to extract a JSON object from model output.
        Works even if the model wraps JSON in extra text.
        """
        # First try direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Try to find first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None

        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def chat(self, prompt: str) -> str:
        """
        Simple chat wrapper for main model.
        Returns assistant reply text.
        """
        try:
            if self.main_provider == GEMINI_PROVIDER:
                r = self._gemini_client.models.generate_content(
                    model=self.main_model, #gemini_chat_model,
                    contents=prompt,
                )
                out = r.text or ""
            # elif self.provider == OPENAI_PROVIDER:
            #     r = self._openai_client.chat.completions.create(
            #         model=self.main_model, # gpt-4.1 , gpt-4.1-mini
            #         messages=[
            #             {"role": "user", "content": rerank_prompt}
            #         ]
            #     )
            #     out = r.choices[0].message.content or ""
            elif self.main_provider == OPENAI_PROVIDER:
                r = self._openai_client.responses.create(
                    model=self.main_model,   # gpt-4o-mini
                    input=prompt,
                )
                out = r.output_text or ""            
            else:
                resp = ollama.chat(
                    model=self.main_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                out = resp["message"]["content"] 
        except Exception as e:
            out = f"Error during chat: {e}"
        
        return out
        
    def rerank_with_llm(self, query: str, hits: list[dict], choose_k: int) -> list[int]:
        """
        Use LLM to pick best contexts from retrieved candidates.
        Returns indices into `hits` (0-based), length <= choose_k.
        Robust fallback: if parsing fails, return first choose_k by score.
        """
        if not hits:
            return []

        choose_k = max(1, int(choose_k))
        choose_k = min(choose_k, len(hits))

        # Build candidate list compactly
        candidates_lines = []
        for idx, h in enumerate(hits):
            # keep it compact to reduce tokens, but include source metadata
            candidates_lines.append(
                f"{idx}: (score={h['score']:.4f}, file={h['source_file']}, row={h['row_index']}) {h['text']}"
            )
        candidates = "\n".join(candidates_lines)

        # English rerank prompt (your request)
        # rerank_prompt = (
        #     "You are a retrieval re-ranker.\n"
        #     "Given a user question and a list of candidate contexts, select the most relevant items.\n"
        #     "Rules:\n"
        #     f"- Choose exactly {choose_k} distinct indices.\n"
        #     "- Prefer contexts that directly contain facts needed to answer.\n"
        #     "- Avoid redundant/duplicate contexts.\n"
        #     "- Output ONLY valid JSON, no extra text.\n\n"
        #     "Return JSON format:\n"
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

            """
            if self.main_provider == GEMINI_PROVIDER:
                r = self._gemini_client.models.generate_content(
                    model=self.main_model, #gemini_chat_model,
                    contents=rerank_prompt,
                )
                out = r.text or ""
            # elif self.provider == OPENAI_PROVIDER:
            #     r = self._openai_client.chat.completions.create(
            #         model=self.main_model, # gpt-4.1 , gpt-4.1-mini
            #         messages=[
            #             {"role": "user", "content": rerank_prompt}
            #         ]
            #     )
            #     out = r.choices[0].message.content or ""
            elif self.main_provider == OPENAI_PROVIDER:
                r = self._openai_client.responses.create(
                    model=self.main_model,   # gpt-4o-mini
                    input=rerank_prompt,
                )
                out = r.output_text or ""            
            else:
                resp = ollama.chat(
                    model=self.main_model,
                    messages=[{"role": "user", "content": rerank_prompt}],
                )
                out = resp["message"]["content"]            
            """            
            out = self.chat(rerank_prompt)

            # resp = ollama.chat(
            #     model=self.main_model,
            #     messages=[{"role": "user", "content": rerank_prompt}],
            # )
            # out = resp["message"]["content"]
            obj = self._extract_json_object(out)
            if not obj:
                raise ValueError("No JSON object found in rerank output.")

            indices = obj.get("selected_indices", [])
            if not isinstance(indices, list):
                raise ValueError("selected_indices is not a list")

            # sanitize
            clean = []
            for x in indices:
                if isinstance(x, int) and 0 <= x < len(hits) and x not in clean:
                    clean.append(x)

            if len(clean) < choose_k:
                # fill remaining by score order
                for i in range(len(hits)):
                    if i not in clean:
                        clean.append(i)
                    if len(clean) == choose_k:
                        break

            return clean[:choose_k]

        except Exception:
            # fallback: top by score (already sorted by Qdrant)
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
        top_k_retrieve: int = DEFAULT_TOPK_RETRIEVE,
        top_k_use: int = DEFAULT_TOPK_USE,
        enable_rerank: bool = DEFAULT_ENABLE_RERANK,
    ) -> dict[str, Any]:
        """
        Production path:
        - retrieve top_k_retrieve from Qdrant
        - rerank to top_k_use with LLM (optional)
        - answer using only selected contexts
        """
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

        # English answering prompt (your request)
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

        # prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        # prompt = (
        #     f"Context:\n{context}\n\n"
        #     f"Question: {query}\n\n"
        #     f"Answer clearly and concisely based only on the context above."
        # )
        # prompt = f"""
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

        # Decision guide:
        # - Treat CONTEXT as "unrelated" if it does not contain information about the user's question topic.
        # - Example: INPUT "who are you?" and CONTEXT about "Asil" => unrelated => do (3).

        # Output format (STRICT):
        # Mode: EXACT | NEAREST | GENERAL
        # Answer: <your answer>

        # INPUT:
        # {query}

        # CONTEXT:
        # {context}
        # """

        # prompt = f"""
        # You are a hybrid RAG assistant.

        # Step 1:
        # - If the INPUT is a structured record (key=value pairs),
        # then perform record matching:
        # - Select the SINGLE most similar row from CONTEXT.
        # - Return ONLY the selected row exactly as shown.

        # Step 2:
        # - If the INPUT is a natural-language question,
        # then answer the question strictly using the CONTEXT.

        # Rules:
        # - Use ONLY the CONTEXT.
        # - If no suitable match or answer exists, reply exactly:
        # "I don't know based on the provided context."

        # INPUT:
        # {query}

        # CONTEXT:
        # {context}

        # OUTPUT:
        # """

        # prompt = f"""
        # You are a record-matching system.

        # Pick the single most similar row from CONTEXT to the INPUT record.
        # Always pick one row unless there is NO meaningful overlap.
        # Do NOT answer with explanations.

        # Return ONLY the selected row exactly as shown (one row text).
        # If no row shares at least 6 identical fields with the INPUT, return exactly:
        # I don't know based on the provided context.

        # INPUT:
        # {query}

        # CONTEXT:
        # {context}
        # """
        # prompt = (
        #             "You are an assistant that answers user question from the context.\n"
        #             "IMPORTANT RULES:\n"
        #             "- Use ONLY the information in the Context.\n"
        #             "- If the Context does not contain the answer, reply exactly: \"I don't know based on the provided context.\"\n"
        #             "- Keep the answer clear and concise.\n\n"
        #             f"here is the Context:\n{context}\n\n"
        #             f"here is user Question: {query}\n\n"
                
        #         )       

        start = time.time()
        
        # resp = ollama.chat(
        #     model=self.main_model,
        #     messages=[{"role": "user", "content": prompt}],
        # )
        """
        if self.main_provider == GEMINI_PROVIDER:
            r = self._gemini_client.models.generate_content(
                model=self.main_model,
                contents=prompt,
            )
            answer_text = r.text or ""
        elif self.main_provider == OPENAI_PROVIDER:
            r = self._openai_client.responses.create(
                model=self.main_model,   # gpt-4o-mini
                input=prompt,
            )
            answer_text = r.output_text or ""             
        else:
            resp = ollama.chat(
                model=self.main_model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = resp["message"]["content"] 
        """
        answer_text = self.chat(prompt)
  

        latency = time.time() - start

        return {
            "answer": answer_text, #resp["message"]["content"],
            "context": context,
            "hits": hits,
            "selected": selected,
            "latency": latency,
            "collection": self.collection_name,
        }


# ------------------ GUI (Thread-safe) ------------------

class RAGAppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x760")

        # Paths
        self.settings_path = Path(__file__).resolve().parent / SETTINGS_FILE_NAME
        self.script_dir = Path(__file__).resolve().parent.parent
        self.base_dir = self.script_dir / RAG_DATA_0_FOLDER_NAME
        self.csv_dir = self.base_dir / RAG_DATA_1_FOLDER_NAME / RAG_DATA_2_FOLDER_NAME
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data folder: {self.csv_dir}")

        # --------- LOAD SETTINGS (or defaults) BEFORE creating tk variables ----------
        loaded = self._load_settings_file()

        # docs folder path from settings
        self.docs_dir_var = tk.StringVar(value=loaded.get("docs_dir", DEFAULT_DOCS_DIR))

        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data folder: {self.csv_dir}")

        # prompt templates from settings
        self.rerank_prompt_var = tk.StringVar(value=loaded.get("rerank_prompt_template", DEFAULT_RERANK_PROMPT_TEMPLATE))
        self.answer_prompt_var = tk.StringVar(value=loaded.get("answer_prompt_template", DEFAULT_ANSWER_PROMPT_TEMPLATE))
        
        self.main_provider_var = tk.StringVar(value=loaded.get("main_provider", DEFAULT_PROVIDER))
        self.embedding_provider_var = tk.StringVar(value=loaded.get("embedding_provider", DEFAULT_PROVIDER))
        self.env_path_var = tk.StringVar(value=loaded.get("env_path", DEFAULT_ENV_PATH))

        # self._load_env_from_settings()

        # # Models state
        # self.embedding_model = EMBEDDING_MODEL
        # self.main_model = DEFAULT_MODEL

        # Models state (load from file if exists)
        self.embedding_model = loaded.get("embedding_model", EMBEDDING_MODEL)  
        self.main_model = loaded.get("main_model", DEFAULT_MODEL)            

        # # Settings state
        # self.qdrant_url = tk.StringVar(value=DEFAULT_QDRANT_URL)
        # self.collection_prefix = tk.StringVar(value=DEFAULT_COLLECTION_PREFIX)
        # self.topk_retrieve = tk.IntVar(value=DEFAULT_TOPK_RETRIEVE)
        # self.topk_use = tk.IntVar(value=DEFAULT_TOPK_USE)
        # self.enable_rerank = tk.BooleanVar(value=DEFAULT_ENABLE_RERANK)
        # self.text_group_lines = tk.IntVar(value=DEFAULT_TEXT_GROUP_LINES)

        # Settings state (load from file if exists)
        self.qdrant_url = tk.StringVar(value=loaded.get("qdrant_url", DEFAULT_QDRANT_URL))
        self.collection_prefix = tk.StringVar(value=loaded.get("collection_prefix", DEFAULT_COLLECTION_PREFIX))
        self.topk_retrieve = tk.IntVar(value=_safe_int(loaded.get("topk_retrieve", DEFAULT_TOPK_RETRIEVE), DEFAULT_TOPK_RETRIEVE))
        self.topk_use = tk.IntVar(value=_safe_int(loaded.get("topk_use", DEFAULT_TOPK_USE), DEFAULT_TOPK_USE))
        self.enable_rerank = tk.BooleanVar(value=_safe_bool(loaded.get("enable_rerank", DEFAULT_ENABLE_RERANK), DEFAULT_ENABLE_RERANK))
        self.text_group_lines = tk.IntVar(value=_safe_int(loaded.get("text_group_lines", DEFAULT_TEXT_GROUP_LINES), DEFAULT_TEXT_GROUP_LINES))

        # # Engine (created lazily/refreshable)
        # self.engine = self._make_engine()

        # Thread-safe UI logging via queue
        self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        # Build UI
        self._build_ui() # here init logging

        self._load_env_from_settings() # load env after UI is built, for logging

        # Engine (created lazily/refreshable)
        self.engine = self._make_engine() # here after env load        

        # Ensure combos reflect loaded models (and clamp if invalid)
        if self.embedding_model not in EMBEDDING_MODELS:
            self.embedding_model = EMBEDDING_MODEL
        if self.main_model not in CHAT_MODELS:
            self.main_model = DEFAULT_MODEL
        self.embedding_combo.set(self.embedding_model)
        self.main_combo.set(self.main_model)
        self.engine.set_models(self.embedding_model, self.main_model)

        # Start UI queue pump
        self._pump_ui_queue()

    def _load_env_from_settings(self):
        p = (self.env_path_var.get() or "").strip()
        if not p:
            return

        env_path = Path(p)
        if not env_path.is_absolute():
            env_path = self.script_dir / env_path

        if env_path.exists():
            load_dotenv(env_path)
            self._append_log(f"[ENV] Loaded: {env_path}")
        else:
            self._append_log(f"[ENV] Not found: {env_path}")

    def _make_engine(self) -> RAGEngineQdrant:
        eng = RAGEngineQdrant(
            qdrant_url=self.qdrant_url.get(),
            api_key=DEFAULT_QDRANT_API_KEY,
            collection_prefix=self.collection_prefix.get(),
        )
        eng.set_models(self.embedding_model, self.main_model)
        eng.set_prompts(self.rerank_prompt_var.get(), self.answer_prompt_var.get())      
        eng.set_provider(
            main_provider=self.main_provider_var.get(),
            embedding_provider=self.embedding_provider_var.get(),
            # gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        )
        
        return eng

    def _open_docs_dir(self):
        path = self._resolve_docs_dir(self.docs_dir_var.get())
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not path.exists():
            messagebox.showwarning("Warning", f"Folder does not exist:\n{path}")
            return

        try:
            # Windows
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
                return

            # macOS
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
                return

            # Linux
            subprocess.run(["xdg-open", str(path)], check=False)

        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to open folder:\n{path}\n\n{e}")

    def _browse_docs_dir(self):
        # start from current value if possible
        initial = str(self._resolve_docs_dir(self.docs_dir_var.get()))
        if not Path(initial).exists():
            initial = str(self.script_dir)

        selected = filedialog.askdirectory(
            title="Select Docs Folder (CSV/TXT for indexing)",
            initialdir=initial,
            mustexist=True,
        )
        if not selected:
            return

        # Prefer saving relative paths (cleaner + portable)
        try:
            rel = Path(selected).resolve().relative_to(self.script_dir.resolve())
            self.docs_dir_var.set(str(rel))
        except Exception:
            # fallback: store absolute
            self.docs_dir_var.set(str(Path(selected).resolve()))

        # Update live folder target
        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self._append_log(f"[Settings] Docs folder set to: {self.csv_dir}")

    # def _resolve_docs_dir(self, p: str) -> Path:
    #     p = (p or "").strip()
    #     if not p:
    #         p = DEFAULT_DOCS_DIR
    #     path = Path(p)
    #     if not path.is_absolute():
    #         path = self.script_dir / path
    #     return path

    def _resolve_docs_dir(self, p: str) -> Path:
        p = (p or "").strip()

        if not p:
            return Path(DEFAULT_DOCS_DIR)

        path = Path(p)

        # If user provided absolute path → use it as-is
        if path.is_absolute(): # D:\LLM\LLMs_Tests\RAG\rag_data\my_csvs\docs
            return path

        # Otherwise treat it as relative to project root #rag_data\my_csvs\docs
        return (self.script_dir / path).resolve()

    # ---------- UI Layout ----------
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Label(toolbar, text="Embedding Model:").pack(side=tk.LEFT, padx=(0, 6))
        self.embedding_combo = ttk.Combobox(
            toolbar, values=EMBEDDING_MODELS, state="readonly", width=28
        )
        self.embedding_combo.set(self.embedding_model)
        self.embedding_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.embedding_combo.bind("<<ComboboxSelected>>", self._on_embedding_changed)

        tk.Label(toolbar, text="Main Model:").pack(side=tk.LEFT, padx=(0, 6))
        self.main_combo = ttk.Combobox(
            toolbar, values=CHAT_MODELS, state="readonly", width=28
        )
        self.main_combo.set(self.main_model)
        self.main_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.main_combo.bind("<<ComboboxSelected>>", self._on_main_changed)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=8, pady=(0, 6))

        # Notebook tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_query = tk.Frame(self.nb)
        self.tab_results = tk.Frame(self.nb)
        self.tab_logs = tk.Frame(self.nb)
        self.tab_settings = tk.Frame(self.nb)

        self.nb.add(self.tab_query, text="Query")
        self.nb.add(self.tab_results, text="Results")
        self.nb.add(self.tab_logs, text="Logs")
        self.nb.add(self.tab_settings, text="Settings")

        self._build_tab_query()
        self._build_tab_results()
        self._build_tab_logs()  # Build the logs tab
        self._build_tab_settings()

    def _build_tab_query(self):
        frm = self.tab_query
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(2, weight=1)

        btns = tk.Frame(frm)
        btns.grid(row=0, column=0, sticky="ew", pady=(6, 10))
        btns.columnconfigure((0, 1, 2, 3), weight=1)

        tk.Button(btns, text="Create Sample CSVs", command=self._create_sample_csvs_threaded)\
            .grid(row=0, column=0, padx=5, sticky="ew")
        tk.Button(btns, text="Build / Upsert Index (Qdrant)", command=self._confirm_and_build_index)\
            .grid(row=0, column=1, padx=5, sticky="ew")
        tk.Button(btns, text="Ping Qdrant", command=self._ping_qdrant_threaded)\
            .grid(row=0, column=2, padx=5, sticky="ew")
        tk.Button(btns, text="Run Query", command=self._run_query_threaded)\
            .grid(row=0, column=3, padx=5, sticky="ew")

        tk.Label(frm, text="Enter your query (Ctrl+Enter to run):", font=("Arial", 12))\
            .grid(row=1, column=0, sticky="w", padx=6)

        self.query_text = tk.Text(frm, height=6, font=("Arial", 12))
        self.query_text.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self.query_text.bind("<Control-Return>", lambda _e: self._run_query_threaded())

    def _build_tab_results(self):
        frm = self.tab_results
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(frm, font=("Consolas", 11))
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _build_tab_logs(self):
        frm = self.tab_logs
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self.logs_text = scrolledtext.ScrolledText(frm, font=("Consolas", 11))
        self.logs_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _make_scrollable_frame(self, parent: tk.Widget) -> tk.Frame:
        """
        Returns an inner frame inside a Canvas+Scrollbar so we can scroll vertically.
        Put all settings widgets into the returned frame.
        """
        container = tk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Make inner frame width follow canvas width
            canvas.itemconfig(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel support (Windows/macOS/Linux)
        def _on_mousewheel(event):
            # Windows: event.delta is +/-120
            # macOS: event.delta is small
            # Linux: uses Button-4/5
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll_up(_event):
            canvas.yview_scroll(-3, "units")

        def _on_linux_scroll_down(_event):
            canvas.yview_scroll(3, "units")

        # Bind when mouse enters/leaves the scroll area
        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_linux_scroll_up)
            canvas.bind_all("<Button-5>", _on_linux_scroll_down)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Enter>", _bind_wheel)
        inner.bind("<Leave>", _unbind_wheel)

        return inner

    def _build_tab_settings(self):
        # frm = self.tab_settings
        frm = self._make_scrollable_frame(self.tab_settings)
        frm.columnconfigure(1, weight=1)

        row = 0

        tk.Label(frm, text="Qdrant URL:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(frm, textvariable=self.qdrant_url).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        row += 1

        tk.Label(frm, text="Collection Prefix:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(frm, textvariable=self.collection_prefix).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        row += 1

        tk.Label(frm, text="Retrieve top_k (Qdrant):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Spinbox(frm, from_=1, to=200, textvariable=self.topk_retrieve, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        tk.Label(frm, text="Use top_k (final contexts):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Spinbox(frm, from_=1, to=20, textvariable=self.topk_use, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        tk.Checkbutton(frm, text="Enable LLM reranking (top_k=10 → best 3)", variable=self.enable_rerank)\
            .grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        tk.Label(frm, text="Text Group Lines (For Indexing Txt Files):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Spinbox(frm, from_=1, to=20, textvariable=self.text_group_lines, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        # # Docs folder
        # tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        # tk.Entry(frm, textvariable=self.docs_dir_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        # row += 1

        # # Docs folder (Entry + Browse button)
        # tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        # docs_row = tk.Frame(frm)
        # docs_row.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        # docs_row.columnconfigure(0, weight=1)
        # tk.Entry(docs_row, textvariable=self.docs_dir_var).grid(row=0, column=0, sticky="ew")
        # tk.Button(docs_row, text="Browse...", command=self._browse_docs_dir).grid(row=0, column=1, padx=(8, 0))
        # row += 1   

        # Docs folder (Entry + Browse + Open)
        tk.Label(frm, text="Docs Folder (for indexing):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        docs_row = tk.Frame(frm)
        docs_row.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        docs_row.columnconfigure(0, weight=1)
        tk.Entry(docs_row, textvariable=self.docs_dir_var).grid(row=0, column=0, sticky="ew")
        tk.Button(docs_row, text="Browse...", command=self._browse_docs_dir).grid(row=0, column=1, padx=(8, 0))
        tk.Button(docs_row, text="Open Folder", command=self._open_docs_dir).grid(row=0, column=2, padx=(8, 0))
        row += 1

        # Rerank prompt template
        tk.Label(frm, text="Rerank Prompt Template:").grid(row=row, column=0, sticky="nw", padx=6, pady=6)
        self.rerank_prompt_text = scrolledtext.ScrolledText(frm, height=8, font=("Consolas", 10))
        self.rerank_prompt_text.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        self.rerank_prompt_text.insert("1.0", self.rerank_prompt_var.get())
        row += 1

        # Answer prompt template
        tk.Label(frm, text="Answer Prompt Template:").grid(row=row, column=0, sticky="nw", padx=6, pady=6)
        self.answer_prompt_text = scrolledtext.ScrolledText(frm, height=8, font=("Consolas", 10))
        self.answer_prompt_text.grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        self.answer_prompt_text.insert("1.0", self.answer_prompt_var.get())
        row += 1

        # MainProvider
        tk.Label(frm, text="Main LLM Provider:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            frm,
            values= ALL_PROVIDERS, #["ollama", "gemini", "openai"],
            textvariable=self.main_provider_var,
            state="readonly",
            width=15
        ).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        # EmbeddingProvider
        tk.Label(frm, text="Embedding LLM Provider:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            frm,
            values= ALL_PROVIDERS, #["ollama", "gemini", "openai"],
            textvariable=self.embedding_provider_var,
            state="readonly",
            width=15
        ).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        row += 1

        # Env file path
        tk.Label(frm, text=".env File Path:").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(frm, textvariable=self.env_path_var).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        row += 1

        # tk.Button(frm, text="Apply Settings", command=self._apply_settings)\
        #     .grid(row=row, column=1, sticky="w", padx=6)
        
        # tk.Button(frm, text="Reset to Defaults", command=self._reset_to_defaults)\
        #     .grid(row=row, column=2, sticky="w", padx=6)

        # Buttons row (Apply / Reset side-by-side)
        btn_row = tk.Frame(frm)
        btn_row.grid(row=row, column=1, sticky="w", padx=6, pady=(10, 10))

        tk.Button(
            btn_row,
            text="Apply Settings",
            command=self._apply_settings,
            width=16
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_row,
            text="Reset to Defaults",
            command=self._reset_to_defaults,
            width=16
        ).pack(side=tk.LEFT, padx=(10, 0))
        row += 1

    # ---------- Thread safe UI queue ----------
    def _ui_put(self, action: str, payload: Any = None):
        self.ui_queue.put((action, payload))

    def _pump_ui_queue(self):
        try:
            while True:
                action, payload = self.ui_queue.get_nowait()
                if action == "title":
                    self.root.title(str(payload))
                elif action == "log":
                    self._append_log(str(payload))
                elif action == "results":
                    self._set_results(str(payload))
                elif action == "warn":
                    messagebox.showwarning("Warning", str(payload))
                elif action == "info":
                    messagebox.showinfo("Info", str(payload))
        except queue.Empty:
            pass
        self.root.after(50, self._pump_ui_queue)

    def _append_log_for_provider(self):
        env_path_var_text = ""
        if(self.main_provider_var.get() != OLLAMA_PROVIDER):
            env_path_var_text = f", env={self.env_path_var.get()}"
        elif (self.embedding_provider_var.get() != OLLAMA_PROVIDER):
                env_path_var_text = f", env={self.env_path_var.get()}"
        self._append_log(
            f"[Settings] Applied. main_provider={self.main_provider_var.get()}, embedding_provider={self.embedding_provider_var.get()} {env_path_var_text}"
        )

    def _append_log(self, text: str):
        self.logs_text.insert(tk.END, text + "\n")
        self.logs_text.see(tk.END)

    # def _set_results(self, text: str):
    #     self.results_text.delete("1.0", tk.END)
    #     self.results_text.insert(tk.END, text)
    #     self.results_text.see(tk.END)
    #     self.nb.select(self.tab_results)

    def _set_results(self, text: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.results_text.insert(tk.END, f"\n{'='*80}\n[{ts}] New Result\n{'='*80}\n")
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.nb.select(self.tab_results)

    # ---------- Settings helper methods ----------
    def _default_settings_dict(self) -> Dict[str, Any]:
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
        }

    def _load_settings_file(self) -> Dict[str, Any]:
        defaults = self._default_settings_dict()
        print(f"Loading settings from: {self.settings_path} {defaults}")
        if not self.settings_path.exists():
            # Create the file with defaults the first time
            try:
                self.settings_path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
            except Exception:
                pass
            return defaults

        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return defaults
        except Exception:
            # If file is corrupted, fallback to defaults (and optionally rewrite)
            try:
                self.settings_path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
            except Exception:
                pass
            return defaults

        # Merge with defaults so any missing keys get default values
        merged = dict(defaults)
        merged.update(data)
        return merged

    def _save_settings_file(self) -> None:
        data = {
            "qdrant_url": self.qdrant_url.get(),
            "collection_prefix": self.collection_prefix.get(),
            "topk_retrieve": int(self.topk_retrieve.get()),
            "topk_use": int(self.topk_use.get()),
            "enable_rerank": bool(self.enable_rerank.get()),
            "text_group_lines": int(self.text_group_lines.get()),
            "docs_dir": self.docs_dir_var.get(),
            "rerank_prompt_template": self.rerank_prompt_var.get(),
            "answer_prompt_template": self.answer_prompt_var.get(),
            "embedding_model": self.embedding_model,
            "main_model": self.main_model,
            "main_provider": self.main_provider_var.get(),
            "embedding_provider": self.embedding_provider_var.get(),
            "env_path": self.env_path_var.get(),            
        }
        self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------- Settings apply ----------
    def _apply_settings(self):
        # Pull latest templates from text widgets
        if hasattr(self, "rerank_prompt_text"):
            self.rerank_prompt_var.set(self.rerank_prompt_text.get("1.0", tk.END).rstrip())
        if hasattr(self, "answer_prompt_text"):
            self.answer_prompt_var.set(self.answer_prompt_text.get("1.0", tk.END).rstrip())

        # Update docs folder path
        self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"[Docs] Using folder: {self.csv_dir}")

        self._load_env_from_settings()

        # Recreate engine with new settings and prompts
        self.engine = self._make_engine()
        self._append_log(f"[Settings] Applied. Qdrant={self.qdrant_url.get()}, prefix={self.collection_prefix.get()}, docs_dir={self.csv_dir}")
        
        self._append_log_for_provider()

        # env_path_var_text = ""
        # if(self.provider_var.get() != OLLAMA_PROVIDER):
        #     env_path_var_text = f", env={self.env_path_var.get()}"

        # self._append_log(
        #     f"[Settings] Applied. provider={self.provider_var.get()} {env_path_var_text}"
        # )

        # SAVE to rag_settings.json
        try:
            self._save_settings_file()
            self._append_log(f"[Settings] Saved to: {self.settings_path}")
        except Exception as e:
            self._append_log(f"[ERROR] Failed saving settings: {e}")        

    # def _apply_settings(self):
    #     # recreate engine with new qdrant url/prefix; keep models
    #     self.engine = self._make_engine()
    #     self._append_log(f"[Settings] Applied. Qdrant={self.qdrant_url.get()}, prefix={self.collection_prefix.get()}")

    #     # SAVE to rag_settings.json
    #     try:
    #         self._save_settings_file()
    #         self._append_log(f"[Settings] Saved to: {self.settings_path}")
    #     except Exception as e:
    #         self._append_log(f"[ERROR] Failed saving settings: {e}")        

    # ---------- Settings reset ----------
    def _reset_to_defaults(self):
        try:
            defaults = self._default_settings_dict()

            # Reset TK variables (Settings tab)
            self.qdrant_url.set(defaults["qdrant_url"])
            self.collection_prefix.set(defaults["collection_prefix"])
            self.topk_retrieve.set(int(defaults["topk_retrieve"]))
            self.topk_use.set(int(defaults["topk_use"]))
            self.enable_rerank.set(bool(defaults["enable_rerank"]))
            self.text_group_lines.set(int(defaults["text_group_lines"]))

            # Reset models too (optional but consistent with saved file)
            self.embedding_model = defaults["embedding_model"]
            self.main_model = defaults["main_model"]

            # Reflect in UI combos if built
            if hasattr(self, "embedding_combo"):
                self.embedding_combo.set(self.embedding_model)
            if hasattr(self, "main_combo"):
                self.main_combo.set(self.main_model)

            # Reset prompt templates and docs dir
            self.docs_dir_var.set(defaults["docs_dir"])
            self.rerank_prompt_var.set(defaults["rerank_prompt_template"])
            self.answer_prompt_var.set(defaults["answer_prompt_template"])

            # Reflect in text widgets if built
            if hasattr(self, "rerank_prompt_text"):
                self.rerank_prompt_text.delete("1.0", tk.END)
                self.rerank_prompt_text.insert("1.0", self.rerank_prompt_var.get())
            if hasattr(self, "answer_prompt_text"):
                self.answer_prompt_text.delete("1.0", tk.END)
                self.answer_prompt_text.insert("1.0", self.answer_prompt_var.get())

            self.csv_dir = self._resolve_docs_dir(self.docs_dir_var.get())
            self.csv_dir.mkdir(parents=True, exist_ok=True)

            # Apply (recreate engine) + set models
            self.engine = self._make_engine()
            self.engine.set_models(self.embedding_model, self.main_model)

            # Save immediately
            self._save_settings_file()

            self._append_log("[Settings] Reset to defaults and saved.")
            messagebox.showinfo("Info", "Settings reset to defaults.")
        except Exception as e:
            self._append_log(f"[ERROR] reset_to_defaults: {e}")
            messagebox.showwarning("Warning", f"Reset failed: {e}")

    # ---------- Model selection handlers ----------
    def _on_embedding_changed(self, _event):
        self.embedding_model = self.embedding_combo.get()
        self.engine.set_models(self.embedding_model, self.main_model)
        self._append_log(f"[Models] Embedding model set to: {self.embedding_model}")

        self.embedding_provider_var.set(OLLAMA_PROVIDER)
        if(self.embedding_model in EMBEDDING_MODELS_GEMINI):
            self.embedding_provider_var.set(GEMINI_PROVIDER)
        if(self.embedding_model in EMBEDDING_MODELS_OPENAI):
                    self.embedding_provider_var.set(OPENAI_PROVIDER)            
        self._append_log_for_provider()

        try:
            self._save_settings_file()
        except Exception:
            pass

    def _on_main_changed(self, _event):
        self.main_model = self.main_combo.get()
        self.engine.set_models(self.embedding_model, self.main_model)
        self._append_log(f"[Models] Main model set to: {self.main_model}")

        self.main_provider_var.set(OLLAMA_PROVIDER)
        if(self.main_model in CHAT_MODELS_GEMINI):
            self.main_provider_var.set(GEMINI_PROVIDER)
        if(self.main_model in CHAT_MODELS_OPENAI):
            self.main_provider_var.set(OPENAI_PROVIDER)
        self._append_log_for_provider()

        try:
            self._save_settings_file()
        except Exception:
            pass

    # ---------- Worker thread wrappers ----------
    def _create_sample_csvs_threaded(self):
        threading.Thread(target=self._create_sample_csvs, daemon=True).start()

    def _confirm_and_build_index(self):
        proceed = messagebox.askyesno(
            "Confirm Index Build",
            "This will (re)build the index and may overwrite existing data.\n\n"
            "Do you want to continue?"
        )
        if not proceed:
            self._append_log("[Index] Build canceled by user.")
            return

        self._build_index_threaded()

    def _build_index_threaded(self):
        threading.Thread(target=self._build_index, daemon=True).start()

    def _ping_qdrant_threaded(self):
        threading.Thread(target=self._ping_qdrant, daemon=True).start()

    def _run_query_threaded(self):
        threading.Thread(target=self._run_query, daemon=True).start()

    # ---------- Workers ----------
    def _create_sample_csvs(self):
        self._ui_put("title", APP_TITLE + " : create_sample_csvs ...")
        try:
            self.csv_dir.mkdir(parents=True, exist_ok=True)

            customer_data = [
                {"name": "Alice Johnson", "orders": "Laptop, Mouse", "description": "Customer interested in tech gadgets"},
                {"name": "Bob Smith", "orders": "Keyboard, Monitor", "description": "Customer upgrading home office"},
                {"name": "Charlie Davis", "orders": "Smartphone, Tablet", "description": "Customer seeking mobile devices"},
                {"name": "Diana Garcia", "orders": "Headphones, Speakers", "description": "Customer focused on audio equipment"},
            ]
            pd.DataFrame(customer_data).to_csv(self.csv_dir / "customer_data.csv", index=False)

            medical_data = [
                {"ActivityCode": "00100", "AcceptedDiagnosis": "C07, C08.0, C08.1, C08.9, C79.89, C79.9", "Rule": "MedicalNecessity"},
                {"ActivityCode": "00102", "AcceptedDiagnosis": "Q36.0, Q36.1, Q36.9, Q37.0, Q37.1, Q37.2", "Rule": "MedicalNecessity"},
                {"ActivityCode": "00103", "AcceptedDiagnosis": "H02.121, H02.122, H02.123, H02.124", "Rule": "VPSActivityFound"},
                {"ActivityCode": "00104", "AcceptedDiagnosis": "H02.131, H02.132, H02.133, H02.134", "Rule": "VPSActivityFound"},
            ]
            pd.DataFrame(medical_data).to_csv(self.csv_dir / "medical_rules.csv", index=False)

            self._ui_put("log", f"[Data] Sample CSVs created in: {self.csv_dir}")
        except Exception as e:
            self._ui_put("log", f"[ERROR] create_sample_csvs: {e}")
        finally:
            self._ui_put("title", APP_TITLE)

    def _ping_qdrant(self):
        self._ui_put("title", APP_TITLE + " : ping_qdrant ...")
        try:
            # simple ping: list collections
            cols = self.engine.client.get_collections()
            self._ui_put("log", f"[Qdrant] Connected OK. Collections: {len(cols.collections)}")
            self._ui_put("info", "Qdrant is reachable.")
        except Exception as e:
            self._ui_put("log", f"[ERROR] Qdrant ping failed: {e}")
            self._ui_put("warn", f"Qdrant not reachable at {self.qdrant_url.get()}\nStart Qdrant and try again.")
        finally:
            self._ui_put("title", APP_TITLE)

    def _build_index(self):
        proceed = messagebox.askyesno(
            "Confirm Index Build",
            "This will (re)build the index and may overwrite existing data.\n\n"
            "Do you want to continue?"
        )
        if not proceed:
            self._append_log("[Index] Build canceled by user.")
            return        
        
        self._ui_put("title", APP_TITLE + " : build_index (Qdrant) ...")
        try:
            # refresh engine settings (if user changed URL/prefix but didn't apply)
            self.engine = self._make_engine()

            # total = self.engine.build_from_csv_folder(self.csv_dir)
            total = self.engine.build_from_folder(
                self.csv_dir,
                txt_chunk_chars=900,
                txt_overlap=120,
                group_lines=self.text_group_lines.get(),
            )
            if total == 0:
                self._ui_put("log", "[Index] No chunks found in CSV folder.")
            else:
                self._ui_put("log", f"[Index] Upserted {total} chunks into: {self.engine.collection_name}")
        except Exception as e:
            self._ui_put("log", f"[ERROR] build_index: {e}")
            self._ui_put("warn", "Index build failed. Check Qdrant + Ollama are running.")
        finally:
            self._ui_put("title", APP_TITLE)

    def _run_query(self):
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            self._ui_put("warn", "Please enter a query.")
            return

        self._ui_put("title", APP_TITLE + " : run_query ...")
        try:
            # refresh engine settings
            self.engine = self._make_engine()

            topk_retrieve = int(self.topk_retrieve.get())
            topk_use = int(self.topk_use.get())
            enable_rerank = bool(self.enable_rerank.get())

            result = self.engine.answer(
                query,
                top_k_retrieve=topk_retrieve,
                top_k_use=topk_use,
                enable_rerank=enable_rerank,
            )

            hits = result.get("hits", [])
            selected = result.get("selected", [])
            context = result.get("context", "")
            answer = result.get("answer", "")
            latency = float(result.get("latency", 0.0))
            collection = result.get("collection", "")

            # Build a clean, production-style report
            lines = []
            lines.append(f"Query: {query}")
            lines.append(f"Collection: {collection}")
            lines.append(f"Provider: main={self.main_provider_var.get()} | embedding={self.embedding_provider_var.get()}")
            lines.append(f"Models: main={self.main_model} | embedding={self.embedding_model}")
            lines.append(f"Retrieve top_k={topk_retrieve} | Use top_k={topk_use} | Rerank={enable_rerank}")
            lines.append(f"Latency: {latency:.3f} seconds")
            lines.append("")
            lines.append("Selected Context:")
            lines.append(context or "(empty)")
            lines.append("")
            lines.append("Answer:")
            lines.append(answer)

            # Show all hits in logs for debugging
            self._ui_put("log", "\n---\n".join([
                f"[Query] {query}",
                f"[RAG] collection={collection} retrieve={topk_retrieve} use={topk_use} rerank={enable_rerank}",
                f"[RAG] selected_indices={selected}",
                f"[RAG] hits_count={len(hits)} latency={latency:.3f}s",
            ]))

            self._ui_put("results", "\n".join(lines))

        except Exception as e:
            self._ui_put("log", f"[ERROR] run_query: {e}")
            self._ui_put("warn", f"Query failed: {e}")
        finally:
            self._ui_put("title", APP_TITLE)


def main():
    root = tk.Tk()
    app = RAGAppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
