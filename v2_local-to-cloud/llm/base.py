from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResult:
    model: str
    text: str
    latency_ms: int
    error: Optional[str] = None


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> LLMResult:
        """Return LLMResult (text + latency + optional error)."""
        raise NotImplementedError
