from __future__ import annotations

import time
from typing import List

from google import genai

from llm.base import BaseLLM, LLMResult


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _model_name_variants(model_name: str) -> List[str]:


    # try both (1) model_name (2) prepend 'models/model_name' 
    m = (model_name or "").strip()
    if not m:
        return []

    stripped = m.removeprefix("models/") if m.startswith("models/") else m
    prefixed = f"models/{stripped}"

    # Try: as-is -> stripped -> prefixed
    return _uniq_keep_order([m, stripped, prefixed])


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = (api_key or "").strip()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _looks_like_not_found(self, err: Exception) -> bool:
        s = str(err).lower()
        return (
            "404" in s
            or "not found" in s
            or "not_found" in s
            or "models/" in s and "not found" in s
        )

    def _list_some_base_models(self, limit: int = 20) -> List[str]:
        
        # list base models to help debugging (via client.models.list(...)
    
        if not self.client:
            return []

        names: List[str] = []
        try:
            pager = self.client.models.list(config={"page_size": limit, "query_base": True})
            for i, m in enumerate(pager):
                # model objects vary by SDK version; try common fields
                name = getattr(m, "name", None) or getattr(m, "model", None) or str(m)
                if isinstance(name, str):
                    names.append(name)
                if len(names) >= limit:
                    break
        except Exception:
            return []

        # Normalize output a bit (remove "models/" just for display)
        cleaned = []
        for n in names:
            n = n.strip()
            cleaned.append(n.removeprefix("models/") if n.startswith("models/") else n)
        return _uniq_keep_order(cleaned)

    def generate(self, prompt: str) -> LLMResult:
        t0 = time.time()

        if not self.client:
            return LLMResult(
                model=self.model_name,
                text="",
                latency_ms=0,
                error="GEMINI_API_KEY is not set (or empty).",
            )

        last_err: Exception | None = None
        tried = _model_name_variants(self.model_name)

        # If model name is weirdly empty, fail fast
        if not tried:
            return LLMResult(
                model=self.model_name,
                text="",
                latency_ms=0,
                error="GEMINI_MODEL is not set (or empty).",
            )

        for candidate_model in tried:
            try:
                resp = self.client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                )
                text = (getattr(resp, "text", "") or "").strip()
                latency_ms = int((time.time() - t0) * 1000)
                return LLMResult(
                    model=candidate_model,
                    text=text,
                    latency_ms=latency_ms,
                    error=None,
                )
            except Exception as e:
                last_err = e
                # If it's a not-found style error, try next variant
                if self._looks_like_not_found(e):
                    continue
                # Otherwise, break early (auth/quota/etc.)
                break

        latency_ms = int((time.time() - t0) * 1000)

        # Add a helpful hint: list available models (best effort)
        hint = ""
        if last_err and self._looks_like_not_found(last_err):
            available = self._list_some_base_models(limit=15)
            if available:
                hint = (
                    "\n\nHint: Your configured GEMINI_MODEL may be unavailable for this API key/region/API. "
                    "Here are some available base models for this key (sample):\n- "
                    + "\n- ".join(available[:15])
                    + "\n"
                )
            else:
                hint = (
                    "\n\nHint: Model was not found. Try changing GEMINI_MODEL between "
                    "'gemini-1.5-flash' and 'models/gemini-1.5-flash', or list models via the SDK "
                    "(client.models.list) / REST models endpoint to confirm availability."
                )

        return LLMResult(
            model=self.model_name,
            text="",
            latency_ms=latency_ms,
            error=f"{last_err}{hint}" if last_err else "Unknown error",
        )
