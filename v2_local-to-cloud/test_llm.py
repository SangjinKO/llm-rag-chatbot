from __future__ import annotations

import os
from pathlib import Path

from llm.local_ollama import LocalOllamaLLM
from llm.chatgpt import ChatGPTLLM
from llm.gemini import GeminiLLM
from prompt.templates import build_prompt


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

            ## ONLY for DEBUG ##
            print("LOAD ENV:", k, v)
        


if __name__ == "__main__":
    load_env_file(".env")

    # --------------------------------------------
    # INLINE RAG context (simulated Top-k chunks)
    # --------------------------------------------
    context_block = """
[Chunk 1 | Page 3]
Installation requires Python 3.9 or later.

[Chunk 2 | Page 4]
Docker must be installed before proceeding.
""".strip()

    question = "What are the prerequisites before installation or first use?"

    strategies = ["strict", "generous"]

    local = LocalOllamaLLM(model_name=os.getenv("OLLAMA_MODEL", ""))

    chatgpt = ChatGPTLLM(
        model_name=os.getenv("OPENAI_MODEL", ""),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    gemini = GeminiLLM(
        model_name=os.getenv("GEMINI_MODEL", ""),
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )

    llms = [
        ("Ollama", local),
        ("ChatGPT", chatgpt),
        ("Gemini", gemini),
    ]

    for strategy in strategies:
        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 80)

        prompt = build_prompt(strategy=strategy, question=question, context_block=context_block)

        for label, llm in llms:
            print("\n" + "-" * 80)
            print(f"{label} | model={llm.model_name}")
            print("-" * 80)

            result = llm.generate(prompt)

            if result.error:
                print(f"[ERROR] {result.error}")
            else:
                print(result.text)

            print(f"(latency: {result.latency_ms} ms)")
