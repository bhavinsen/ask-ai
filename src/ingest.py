"""Chunk markdown knowledge files and build the TF-IDF index."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from .store import IndexedChunk, save_index


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str, str]]:
    """Return (doc_id, source_filename, text) for each markdown file."""
    docs: list[tuple[str, str, str]] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((path.stem, path.name, text))
    return docs


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_text(text)


def build_index() -> int:
    indexed: list[IndexedChunk] = []
    for doc_id, filename, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            indexed.append(
                IndexedChunk(
                    id=f"{doc_id}__{i}",
                    text=chunk,
                    source=filename,
                    doc_id=doc_id,
                )
            )
    return save_index(indexed)


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks from {DATA_DIR}")


if __name__ == "__main__":
    main()
