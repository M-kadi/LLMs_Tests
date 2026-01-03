from __future__ import annotations
import numpy as np
import ollama
from .base import EmbeddingProvider, ChatProvider

class OllamaEmbedding(EmbeddingProvider):
    def __init__(self, model: str):
        self.model = model
    def embed(self, text: str) -> np.ndarray:
        resp = ollama.embeddings(model=self.model, prompt=text or "")
        return np.array(resp["embedding"], dtype=np.float32)

class OllamaChat(ChatProvider):
    def __init__(self, model: str):
        self.model = model
    def chat(self, prompt: str) -> str:
        resp = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]
