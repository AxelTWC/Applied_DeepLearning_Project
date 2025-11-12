from typing import List, Dict, Tuple, Optional
import numpy as np


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d) b:(m, d) -> (n, m)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)


class InMemoryVectorStore:
    """Simple in-memory vector store using NumPy arrays and cosine similarity.

    Stores vectors and accompanying metadata.
    """

    def __init__(self):
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadatas: List[Dict] = []

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: Optional[List[Dict]] = None):
        if metadatas is None:
            metadatas = [{}] * len(ids)
        if len(ids) != vectors.shape[0] or len(ids) != len(metadatas):
            raise ValueError("ids, vectors and metadatas must have matching lengths")

        if self.vectors is None:
            self.vectors = np.array(vectors, copy=True)
        else:
            self.vectors = np.vstack([self.vectors, np.array(vectors)])

        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Return top_k (id, score, metadata) tuples ordered by descending score (cosine similarity)."""
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        q = np.atleast_2d(query_vector)
        sims = _cosine_sim_matrix(q, self.vectors)[0]
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            results.append((self.ids[i], float(sims[i]), self.metadatas[i]))
        return results
