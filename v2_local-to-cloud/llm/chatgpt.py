from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from llm.base import BaseLLM, LLMResult


class ChatGPTLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = (api_key or "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate(self, prompt: str) -> LLMResult:
        t0 = time.time()

        if not self.client:
            return LLMResult(
                model=self.model_name,
                text="",
                latency_ms=0,
                error="OPENAI_API_KEY is not set (or empty).",
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            latency_ms = int((time.time() - t0) * 1000)
            return LLMResult(model=self.model_name, text=text, latency_ms=latency_ms, error=None)
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            return LLMResult(model=self.model_name, text="", latency_ms=latency_ms, error=str(e))
