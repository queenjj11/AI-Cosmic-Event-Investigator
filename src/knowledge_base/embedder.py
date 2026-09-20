"""Text embedding module supporting both dense neural embeddings and fast offline TF-IDF."""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEmbedder:
    """Embeds text chunks into dense vectors. Supports sentence-transformers with TF-IDF fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_neural: bool = False):
        self.use_neural = use_neural
        self.model_name = model_name
        self.neural_model = None
        self.tfidf = TfidfVectorizer(max_features=256, stop_words="english", ngram_range=(1, 2))
        self.is_fitted = False

        if use_neural:
            try:
                from sentence_transformers import SentenceTransformer
                self.neural_model = SentenceTransformer(model_name)
            except Exception:
                self.neural_model = None
                self.use_neural = False

    def fit(self, texts: List[str]) -> "TextEmbedder":
        """Fit vocabulary for TF-IDF if using fallback."""
        if not self.use_neural:
            self.tfidf.fit(texts)
            self.is_fitted = True
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of strings into (N, D) normalized vector array.
        """
        if self.use_neural and self.neural_model is not None:
            embeddings = self.neural_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # Fallback to TF-IDF with L2 normalization
        if not self.is_fitted:
            self.tfidf.fit(texts)
            self.is_fitted = True

        sparse_mat = self.tfidf.transform(texts)
        dense = sparse_mat.toarray().astype(np.float32)
        # L2 normalize rows
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return dense / norms
