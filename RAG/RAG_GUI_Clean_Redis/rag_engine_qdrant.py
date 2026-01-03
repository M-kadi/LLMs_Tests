from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import ollama
from google import genai
from openai import OpenAI
from qdrant_client import QdrantClient, models as qmodels

from rag_settings import (
    DEFAULT_ANSWER_PROMPT_TEMPLATE,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_MODEL,
    DEFAULT_RERANK_PROMPT_TEMPLATE,
    DEFAULT_TOPK_RETRIEVE,
    DEFAULT_TOPK_USE,
    EMBEDDING_MODEL,
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_PROVIDER,
    OLLAMA_PROVIDER,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_PROVIDER,
)

@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_file: str
    row_index: int


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

    def chat(self, prompt: str) -> str:
        try:
            if self.main_provider == GEMINI_PROVIDER:
                r = self._gemini_client.models.generate_content(
                    model=self.main_model, #gemini_chat_model,
                    contents=prompt,
                )
                out = r.text or ""
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

    @staticmethod
    def clean_translation(s: str) -> str:
        """Remove wrapping quotes / code fences / leading labels."""
        if not s:
            return ""
        s = s.strip()

        # remove code fences
        s = re.sub(r"^```(?:text|json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

        # remove common prefixes
        for prefix in ("Translation:", "English:", "Output:", "Answer:"):
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):].strip()

        # strip surrounding quotes
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        return s.strip()

    @staticmethod
    def looks_english(text: str) -> bool:
        """
        Heuristic: if most letters are basic Latin, assume English.
        (Good enough to avoid translating already-English queries.)
        """
        if not text:
            return True
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return True
        latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
        return (latin / max(1, len(letters))) > 0.8

    def translate_to_english(
        self,
        text: str,
        # provider: Provider,
        # model: str,
        # *,
        # openai_client=None,
        # gemini_client=None,
    ) -> str:
        """
        Translate any language -> English using an LLM.
        - provider: "ollama" | "openai" | "gemini"
        - model: the LLM model name for that provider
        - openai_client: OpenAI client instance (if provider="openai")
        - gemini_client: Gemini client instance (if provider="gemini")
        """
        text = (text or "").strip()
        if not text:
            return ""

        # already English? return as-is
        if self.looks_english(text):
            return text

        prompt = f"""
    You are a translation engine.

    Task:
    - Translate the user text into natural English.
    Rules:
    - Output ONLY the English translation.
    - No explanations, no quotes, no extra text.
    - Keep proper nouns, IDs, emails, URLs, and numbers unchanged.
    - If the text is already English, output it unchanged.

    User text:
    {text}
    """.strip()
        
        try:
            """
            if self.main_provider == GEMINI_PROVIDER:
                r = self._gemini_client.models.generate_content(
                    model=self.main_model,
                    contents=prompt,
                )
                out = r.text or ""
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
            """
            out = self.chat(prompt)

            out = self.clean_translation(out)

            # last fallback if model returns empty
            return out or text

        except Exception:
            # if translation fails, do not break retrieval; return original
            return text        
  
    def answer(
        self,
        query: str,
        history_block: str = "",
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

        query = self.translate_to_english(query)

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
        try:
            prompt = self.answer_prompt_template.format(context=context, query=query, history=history_block)
        except Exception:
            prompt = self.answer_prompt_template.format(context=context, query=query)


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
        }, query
