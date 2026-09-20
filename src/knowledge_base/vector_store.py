"""In-memory vector store for cosine similarity retrieval over astronomical literature."""

from typing import List, Optional, Tuple
import numpy as np
import os
import json

from src.knowledge_base.chunker import DocumentChunk
from src.knowledge_base.embedder import TextEmbedder


class VectorStore:
    """Stores chunk embeddings and performs top-k semantic cosine similarity searches."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or TextEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> "VectorStore":
        """Add chunks to the vector store and compute their embeddings."""
        if not chunks:
            return self

        self.chunks.extend(chunks)
        texts = [c.text for c in self.chunks]

        self.embedder.fit(texts)
        self.embeddings = self.embedder.embed(texts)
        return self

    def query(self, query_text: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top_k most similar chunks for a given query.
        Returns: list of (chunk, similarity_score) sorted descending.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = self.embedder.embed([query_text])[0]  # (D,)
        # Cosine similarity (vectors are already L2 normalized)
        similarities = np.dot(self.embeddings, q_vec)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def save(self, directory: str) -> None:
        """Persist vector store chunks and embeddings."""
        os.makedirs(directory, exist_ok=True)
        chunks_data = [
            {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
                "text": c.text, "source_type": c.source_type, "filepath": c.filepath,
                "metadata": c.metadata
            }
            for c in self.chunks
        ]
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump(chunks_data, f, indent=2)

        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)

    def load(self, directory: str) -> "VectorStore":
        """Load vector store from disk."""
        chunks_path = os.path.join(directory, "chunks.json")
        embed_path = os.path.join(directory, "embeddings.npy")

        if os.path.exists(chunks_path):
            with open(chunks_path, "r") as f:
                data = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in data]

        if os.path.exists(embed_path):
            self.embeddings = np.load(embed_path)

        return self
