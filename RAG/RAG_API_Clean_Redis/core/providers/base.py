from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

class ChatProvider(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str: ...
