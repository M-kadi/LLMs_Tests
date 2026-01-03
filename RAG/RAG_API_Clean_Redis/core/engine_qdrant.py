from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models as qmodels

from core.providers.base import EmbeddingProvider, ChatProvider


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_file: str
    row_index: int


class RAGEngineQdrant:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_prefix: str,
        embedding_client: EmbeddingProvider,
        chat_client: ChatProvider,
        embedding_model_name: str,
        main_model_name: str,
        rerank_prompt_template: str,
        answer_prompt_template: str,
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key or None
        self.collection_prefix = collection_prefix

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        self.embedding_client = embedding_client
        self.chat_client = chat_client

        self.embedding_model_name = embedding_model_name
        self.main_model_name = main_model_name

        self.collection_name: Optional[str] = None
        self.vector_dim: Optional[int] = None

        self.rerank_prompt_template = rerank_prompt_template
        self.answer_prompt_template = answer_prompt_template

    @staticmethod
    def _slug(s: str) -> str:
        return s.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _embed(self, text: str) -> np.ndarray:
        return self.embedding_client.embed(text)

    def _chat(self, prompt: str) -> str:
        return self.chat_client.chat(prompt)

    def _ensure_collection(self, dim: int) -> str:
        collection = f"{self.collection_prefix}__{self._slug(self.embedding_model_name)}__dim{dim}"
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
    def split_text_to_chunks(text: str, chunk_chars: int, overlap: int, group_lines: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if overlap >= chunk_chars:
            overlap = max(0, chunk_chars // 4)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        blocks: List[str] = []
        if group_lines <= 1:
            blocks = lines
        else:
            for i in range(0, len(lines), group_lines):
                blocks.append("\n".join(lines[i:i + group_lines]))

        chunks: List[str] = []

        def split_long(block: str):
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
                split_long(b)

        return chunks

    def build_from_folder(
        self,
        docs_dir: Path,
        batch_size: int,
        txt_chunk_chars: int,
        txt_overlap: int,
        group_lines: int,
    ) -> int:
        docs_dir = Path(docs_dir)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

        records: List[ChunkRecord] = []

        for csv_path in sorted(docs_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            for i, row in df.iterrows():
                text = self._row_to_chunk_text(df, row)
                if not text:
                    continue
                chunk_id = self._stable_chunk_id(csv_path.name, int(i), text)
                records.append(ChunkRecord(chunk_id, text, csv_path.name, int(i)))

        for txt_path in sorted(docs_dir.glob("*.txt")):
            content = txt_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self.split_text_to_chunks(content, txt_chunk_chars, txt_overlap, group_lines)
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
            points: List[qmodels.PointStruct] = []
            for rec in batch:
                vec = self._embed(rec.text)
                if vec.shape[0] != dim:
                    raise RuntimeError(f"Embedding dim mismatch: expected {dim}, got {vec.shape[0]}")
                payload = {
                    "text": rec.text,
                    "source_file": rec.source_file,
                    "row_index": rec.row_index,
                    "embedding_model": self.embedding_model_name,
                }
                points.append(qmodels.PointStruct(id=rec.chunk_id, vector=vec.tolist(), payload=payload))
            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)

        return total

    def _ensure_ready_for_query(self):
        if self.collection_name and self.vector_dim:
            return
        dim = int(self._embed("dimension probe").shape[0])
        self._ensure_collection(dim)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query = (query or "").strip()
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
        out: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            out.append({
                "score": float(h.score),
                "text": str(payload.get("text", "")),
                "source_file": str(payload.get("source_file", "")),
                "row_index": int(payload.get("row_index", -1)),
            })
        return out

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
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

    def rerank_with_llm(self, query: str, hits: List[Dict[str, Any]], choose_k: int) -> List[int]:
        if not hits:
            return []
        choose_k = max(1, min(int(choose_k), len(hits)))

        candidates_lines = [
            f"{idx}: (score={h['score']:.4f}, file={h['source_file']}, row={h['row_index']}) {h['text']}"
            for idx, h in enumerate(hits)
        ]
        candidates = "\n".join(candidates_lines)

        rerank_prompt = self.rerank_prompt_template.format(
            choose_k=choose_k,
            query=query,
            candidates=candidates,
        )

        try:
            out = self._chat(rerank_prompt)
            obj = self._extract_json_object(out)
            if not obj:
                raise ValueError("No JSON object found in rerank output.")
            indices = obj.get("selected_indices", [])
            clean: List[int] = []
            if isinstance(indices, list):
                for x in indices:
                    if isinstance(x, int) and 0 <= x < len(hits) and x not in clean:
                        clean.append(x)
            # fill
            for i in range(len(hits)):
                if len(clean) >= choose_k:
                    break
                if i not in clean:
                    clean.append(i)
            return clean[:choose_k]
        except Exception:
            return list(range(choose_k))

    @staticmethod
    def build_context(hits: List[Dict[str, Any]], indices: List[int]) -> str:
        blocks = []
        for rank, i in enumerate(indices, start=1):
            h = hits[i]
            blocks.append(
                f"[#{rank} | score={h['score']:.4f} | file={h['source_file']} | row={h['row_index']}]\n{h['text']}"
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
            out = self._chat(prompt)

            out = self.clean_translation(out)

            # last fallback if model returns empty
            return out or text

        except Exception:
            # if translation fails, do not break retrieval; return original
            return text        

    def answer(self, query: str, top_k_retrieve: int, top_k_use: int, enable_rerank: bool, history_block: str = "") -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"answer": "", "context": "", "hits": [], "selected": [], "latency": 0.0}

        query = self.translate_to_english(query)

        hits = self.search(query, top_k=int(top_k_retrieve))
        if not hits:
            return {"answer": "No retrieved data. Build the index first.", "context": "", "hits": [], "selected": [], "latency": 0.0}

        if enable_rerank:
            selected = self.rerank_with_llm(query, hits, choose_k=int(top_k_use))
        else:
            selected = list(range(min(int(top_k_use), len(hits))))

        context = self.build_context(hits, selected)

        prompt = self.answer_prompt_template.format(context=context, query=query, history=history_block or "")

        import time
        start = time.time()
        answer_text = self._chat(prompt)
        latency = time.time() - start

        return {
            "answer": answer_text,
            "context": context,
            "hits": hits,
            "selected": selected,
            "latency": latency,
            "collection": self.collection_name,
        }, query