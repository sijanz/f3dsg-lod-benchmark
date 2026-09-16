"""Text embedding utility shared by all policies.

A single SentenceTransformer-backed embedder, exposed as a process-wide singleton
via get_embedder(). Policies depend on this module, not on a concrete model, so
the backend model is swapped in one place (the _DEFAULT line / constructor arg).

The singleton is deliberate: it loads the model once per process and caches
per-string embeddings, so repeated policy calls reuse both. It is also what makes
cold-start detection meaningful — the first call loads the model (recorded as a
cold start via model_usage_tracker); later calls in the same process are warm.
"""

from typing import List

import numpy as np

def _record_model_usage(*args, **kwargs):
    pass


class TextEmbedder:
    """SentenceTransformer embedder. Lazy-loaded; caches per-string embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model_name(self) -> str:
        """Name of the backend model (readable without loading it)."""
        return self._model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device="cpu")
            _record_model_usage("embedder", self._model_name, cold_start=True)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalised embeddings."""
        # Usage is recorded even on a full cache hit: the model's
        # embeddings decided the result either way.
        _record_model_usage("embedder", self._model_name)
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            model = self._get_model()
            vecs = model.encode(uncached, normalize_embeddings=True,
                                show_progress_bar=False)
            for text, vec in zip(uncached, vecs):
                self._cache[text] = vec.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])

    def similarity(self, a: List[str], b: List[str]) -> np.ndarray:
        """Return cosine similarity matrix of shape (len(a), len(b))."""
        return self.embed(a) @ self.embed(b).T


# Change the model_name here to swap the backend for every policy at once.
_DEFAULT = TextEmbedder("all-MiniLM-L6-v2")


def get_embedder() -> TextEmbedder:
    """Return the shared default embedder instance."""
    return _DEFAULT
