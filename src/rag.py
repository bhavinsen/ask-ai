"""Retrieve context and generate answers about Bhavin Sen."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sarvamai import SarvamAI

from src.config import (
    SARVAM_MAX_TOKENS,
    SARVAM_MODEL,
    SARVAM_REASONING_EFFORT,
    TOP_K,
)
from src.store import SearchHit, search

SYSTEM_PROMPT = """You are "Ask Bhavin", a helpful assistant that answers questions about Bhavin Sen (software engineer, @bhavinsen on GitHub, @senbhavin on Freelancer).

Rules:
1. Answer ONLY using the provided context about Bhavin. If the context does not contain enough information, say so honestly.
2. Do not invent employers, projects, skills, or metrics not supported by the context.
3. For sensitive topics (negative reviews, limitations), be factual and balanced when the context includes them.
4. Keep answers concise, friendly, and professional — suitable for a recruiter or founder evaluating Bhavin.
5. When relevant, mention which source file the information likely came from (e.g. profile, skills, reviews).
6. Write the final answer directly. Do not show planning steps or chain-of-thought."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    chunks: list[RetrievedChunk]
    used_llm: bool


def expand_query(question: str) -> str:
    """Boost retrieval for short identity / reputation questions."""
    q = question.lower()
    extras: list[str] = []

    if re.search(r"\b(who is|who's|tell me about|about)\b", q) or "bhavinchandra" in q:
        extras.extend(
            [
                "Bhavin Sen",
                "Bhavinchandra",
                "profile identity",
                "software engineer",
                "freelancer",
            ]
        )

    if any(w in q for w in ("reputation", "review", "rating", "freelancer")):
        extras.extend(["Freelancer reputation", "reviews", "rating", "4.9"])

    if not extras:
        return question
    return f"{question} {' '.join(extras)}"


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    hits: list[SearchHit] = search(expand_query(question), top_k=top_k)
    if not hits:
        raise RuntimeError(
            "Knowledge base is empty. Run: python -m src.ingest"
        )
    return [
        RetrievedChunk(text=h.chunk.text, source=h.chunk.source, score=h.score)
        for h in hits
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c.source}, relevance: {c.score:.3f})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _sarvam_api_key() -> str:
    return os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY") or ""


def _message_text(message) -> str:
    content = getattr(message, "content", None) or ""
    return content.strip()


def grounded_summary(question: str, chunks: list[RetrievedChunk]) -> str:
    """Fallback when the LLM returns empty — still useful and accurate."""
    if not chunks:
        return "I don't have enough information in the knowledge base to answer that."

    lines = [
        f"**{question.strip()}**\n",
        "Here's a summary from Bhavin's profile:\n",
    ]
    for c in chunks[:3]:
        body = re.sub(r"^#+\s*", "", c.text.strip(), flags=re.MULTILINE)
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip()


def generate_with_sarvam(question: str, context: str) -> str:
    api_key = _sarvam_api_key()
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set")

    client = SarvamAI(api_subscription_key=api_key)
    response = client.chat.completions(
        model=SARVAM_MODEL,
        temperature=0.2,
        max_tokens=SARVAM_MAX_TOKENS,
        reasoning_effort=SARVAM_REASONING_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Reply with the final answer only (2–6 sentences unless a list is requested)."
                ),
            },
        ],
    )
    return _message_text(response.choices[0].message)


def extractive_fallback(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I don't have any indexed information to answer that."

    lines = [
        "*(Sarvam API key not set — showing retrieved passages instead of a synthesized answer.)*\n",
        f"**Your question:** {question}\n",
        "**Relevant excerpts from Bhavin's knowledge base:**\n",
    ]
    for i, c in enumerate(chunks, 1):
        lines.append(f"**[{i}]** ({c.source}, score {c.score:.3f})\n{c.text}\n")
    lines.append(
        "\n_Set `SARVAM_API_KEY` in `.env` and restart the app for natural-language answers._"
    )
    return "\n".join(lines)


def ask(question: str, top_k: int = TOP_K) -> AnswerResult:
    chunks = retrieve(question, top_k=top_k)
    context = format_context(chunks)

    if _sarvam_api_key():
        try:
            answer = generate_with_sarvam(question, context)
        except Exception:
            answer = ""
        if not answer:
            answer = grounded_summary(question, chunks)
            return AnswerResult(answer=answer, chunks=chunks, used_llm=False)
        return AnswerResult(answer=answer, chunks=chunks, used_llm=True)

    return AnswerResult(
        answer=extractive_fallback(question, chunks),
        chunks=chunks,
        used_llm=False,
    )
