"""
Minimal in-memory vector store for the GraphRAG comparison.

This is intentionally lightweight — numpy + cosine similarity, no FAISS
or Chroma. It exists so that the side-by-side "vector RAG vs GraphRAG"
demonstration is a real comparison: both paths see the SAME chunks
embedded the SAME way, and differ only in how context is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


EMBEDDING_MODEL = "text-embedding-3-small"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Word-aware sliding-window chunker. Returns non-empty chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _embed_batch(client, texts: Sequence[str]) -> np.ndarray:
    """Embed a batch via the existing OpenAI client. Returns float32 array."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    vecs = [d.embedding for d in response.data]
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class VectorStore:
    """Holds chunk text + their embeddings as a single numpy matrix."""

    chunks: List[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    @classmethod
    def from_text(cls, text: str, client, chunk_size: int = 500, overlap: int = 100) -> "VectorStore":
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return cls()
        embeddings = _embed_batch(client, chunks)
        return cls(chunks=chunks, embeddings=embeddings)

    def search(self, query: str, client, top_k: int = 4) -> List[Tuple[int, float, str]]:
        """Return top_k (index, similarity, chunk) tuples ranked by cosine."""
        if not self.chunks:
            return []
        q_vec = _embed_batch(client, [query])[0]
        # Normalised cosine.
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-12
        scores = (self.embeddings @ q_vec) / norms
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i]), self.chunks[i]) for i in top]
