from __future__ import annotations

import time
from typing import Optional

from ollama import chat

from llm.base import BaseLLM, LLMResult


class LocalOllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama3.2:3b"):
        super().__init__(model_name)

    def generate(self, prompt: str) -> LLMResult:
        t0 = time.time()
        try:
            resp = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp["message"]["content"].strip()
            latency_ms = int((time.time() - t0) * 1000)
            return LLMResult(model=self.model_name, text=text, latency_ms=latency_ms, error=None)
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            return LLMResult(model=self.model_name, text="", latency_ms=latency_ms, error=str(e))
