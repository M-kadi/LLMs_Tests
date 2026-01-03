from __future__ import annotations
import os
import numpy as np
from google import genai
from .base import EmbeddingProvider, ChatProvider

def _key(api_key: str | None) -> str:
    return api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, model: str, api_key: str | None = None):
        k = _key(api_key)
        if not k:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing.")
        self.client = genai.Client(api_key=k)
        self.model = model
    def embed(self, text: str) -> np.ndarray:
        r = self.client.models.embed_content(model=self.model, contents=text or "")
        if hasattr(r, "embeddings") and r.embeddings:
            vec = r.embeddings[0].values
        else:
            vec = r.embedding.values
        return np.array(vec, dtype=np.float32)

class GeminiChat(ChatProvider):
    def __init__(self, model: str, api_key: str | None = None):
        k = _key(api_key)
        if not k:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing.")
        self.client = genai.Client(api_key=k)
        self.model = model
    def chat(self, prompt: str) -> str:
        r = self.client.models.generate_content(model=self.model, contents=prompt)
        return getattr(r, "text", "") or ""
