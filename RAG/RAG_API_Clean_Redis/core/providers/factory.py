from __future__ import annotations
from typing import Any, Dict

from core.settings_store import OLLAMA, OPENAI, GEMINI
from .base import EmbeddingProvider, ChatProvider
from .ollama import OllamaEmbedding, OllamaChat
from .openai import OpenAIEmbedding, OpenAIChat
from .gemini import GeminiEmbedding, GeminiChat

def make_embedding_provider(s: Dict[str, Any]) -> EmbeddingProvider:
    p = (s.get("embedding_provider") or OLLAMA).strip().lower()
    model = s["embedding_model"]
    if p == OLLAMA:
        return OllamaEmbedding(model=model)
    if p == OPENAI:
        return OpenAIEmbedding(model=model, api_key=(s.get("openai_api_key") or None))
    if p == GEMINI:
        return GeminiEmbedding(model=model, api_key=(s.get("gemini_api_key") or None))
    raise ValueError(f"Invalid embedding_provider: {p}")

def make_chat_provider(s: Dict[str, Any]) -> ChatProvider:
    p = (s.get("main_provider") or OLLAMA).strip().lower()
    model = s["main_model"]
    if p == OLLAMA:
        return OllamaChat(model=model)
    if p == OPENAI:
        return OpenAIChat(model=model, api_key=(s.get("openai_api_key") or None))
    if p == GEMINI:
        return GeminiChat(model=model, api_key=(s.get("gemini_api_key") or None))
    raise ValueError(f"Invalid main_provider: {p}")
