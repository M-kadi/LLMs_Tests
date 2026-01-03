from __future__ import annotations
import os
import numpy as np
from openai import OpenAI
from .base import EmbeddingProvider, ChatProvider

class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model: str, api_key: str | None = None):
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=key)
        self.model = model
    def embed(self, text: str) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input=text or "")
        return np.array(r.data[0].embedding, dtype=np.float32)

class OpenAIChat(ChatProvider):
    def __init__(self, model: str, api_key: str | None = None):
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=key)
        self.model = model
    def chat(self, prompt: str) -> str:
        try:
            r = self.client.responses.create(model=self.model, input=prompt)
            return r.output_text or ""
        except Exception:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content or ""
