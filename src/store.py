"""Local TF-IDF vector store — no external model downloads."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import INDEX_DIR


@dataclass
class IndexedChunk:
    id: str
    text: str
    source: str
    doc_id: str


@dataclass
class SearchHit:
    chunk: IndexedChunk
    score: float


def _paths():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return (
        INDEX_DIR / "chunks.json",
        INDEX_DIR / "vectorizer.pkl",
        INDEX_DIR / "matrix.pkl",
    )


def save_index(chunks: list[IndexedChunk]) -> int:
    chunks_path, vec_path, mat_path = _paths()
    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )
    matrix = vectorizer.fit_transform(texts)

    chunks_path.write_text(
        json.dumps([c.__dict__ for c in chunks], indent=2),
        encoding="utf-8",
    )
    vec_path.write_bytes(pickle.dumps(vectorizer))
    mat_path.write_bytes(pickle.dumps(matrix))
    return len(chunks)


def load_index() -> tuple[list[IndexedChunk], TfidfVectorizer, object]:
    chunks_path, vec_path, mat_path = _paths()
    if not chunks_path.exists():
        raise FileNotFoundError(
            "Index not found. Run: python -m src.ingest"
        )

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [IndexedChunk(**item) for item in raw]
    vectorizer = pickle.loads(vec_path.read_bytes())
    matrix = pickle.loads(mat_path.read_bytes())
    return chunks, vectorizer, matrix


def search(query: str, top_k: int = 5) -> list[SearchHit]:
    chunks, vectorizer, matrix = load_index()
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    hits: list[SearchHit] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        hits.append(SearchHit(chunk=chunks[idx], score=float(score)))
    return hits
