from __future__ import annotations


def build_strict_prompt(question: str, context_block: str) -> str:
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "Not specified in the document" and list what is missing.

Return the answer as:
- A short title
- A bullet list of prerequisites/requirements

Question:
{question}

Context:
{context_block}
"""


def build_generous_prompt(question: str, context_block: str) -> str:
    return f"""You are a helpful assistant.

Based ONLY on the provided context:
- Identify any information that can reasonably be interpreted as prerequisites,
  requirements, or preparations needed before installation or first use.
- You may infer prerequisites if they are clearly implied by the context
  (e.g., required registrations, initial setup steps, or mandatory checks).

If the document does not explicitly list prerequisites, summarize them
as "Inferred prerequisites based on the document".

Return the answer as:
- A short title
- A bullet list of prerequisites/requirements

Question:
{question}

Context:
{context_block}
"""


def build_prompt(strategy: str, question: str, context_block: str) -> str:
    s = (strategy or "").strip().lower()
    if s in ("strict", "s"):
        return build_strict_prompt(question, context_block)
    if s in ("generous", "g"):
        return build_generous_prompt(question, context_block)
    raise ValueError("strategy must be 'strict' or 'generous'")
