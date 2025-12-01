from typing import List
import numpy as np
import os
import sys
from typing import Optional
import torch


def _torch_has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class EmbeddingModel:
    """Embedding wrapper with lazy loading and optional device control.

    Args:
        model_name: sentence-transformers model name.
        device: 'auto'|'cpu'|'cuda' - device preference. 'auto' prefers CUDA when available.
        use_fallback: if True, never load sentence-transformers and use deterministic fallback.
        lazy: if True, delay heavy imports/model loading until `embed_texts` is called.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        use_fallback: bool = True,
        lazy: bool = True,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.use_fallback = use_fallback
        self.lazy = lazy
        self.model = None
        self._loaded = False

        # resolve device early for logging; actual model will be created on _load
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[EmbeddingModel] initialized (model={model_name}, device={self.device}, use_fallback={use_fallback}, lazy={lazy})")

        if not self.lazy:
            self._load()

    def _load(self):
        if self._loaded:
            return
        print(f"[EmbeddingModel] loading model (may use significant RAM/VRAM)...")
        # If use_fallback is True, skip heavy model loading and use deterministic embeddings.
        if self.use_fallback:
            print("[EmbeddingModel] use_fallback=True -> using deterministic embeddings (no heavy model loaded)")
            self.model = None
            self._loaded = True
            return

        # Attempt to load a real sentence-transformers model. We add safeguards:
        # - If running on a low-VRAM GPU, automatically pick a smaller model.
        # - If any error occurs, fall back to deterministic embeddings.
        try:
            from sentence_transformers import SentenceTransformer

            # If user requested CUDA but VRAM is small, pick a smaller model automatically
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    prop = torch.cuda.get_device_properties(0)
                    total_gb = prop.total_memory / (1024 ** 3)
                    print(f"[EmbeddingModel] detected GPU: {prop.name}, VRAM={total_gb:.2f} GB")
                    # if VRAM is small (<10GB), prefer a smaller model
                    if total_gb < 10.0:
                        small_model = "paraphrase-MiniLM-L3-v2"
                        print(f"[EmbeddingModel] VRAM < 10GB, switching to smaller model '{small_model}' to reduce memory usage")
                        self.model_name = small_model
            except Exception:
                # couldn't query CUDA props; continue and let SentenceTransformer handle device
                pass

            # Try to construct the model with device argument (newer versions support it)
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except TypeError:
                # Older API: construct then move to device if possible
                self.model = SentenceTransformer(self.model_name)
                try:
                    if self.device == "cuda" and torch.cuda.is_available():
                        self.model.to("cuda")
                    elif self.device == "mps" and torch.backends.mps.is_available():
                        self.model.to("mps")
                    else:
                        self.model.to("cpu")
                except Exception:
                    pass

            print(f"[EmbeddingModel] sentence-transformers model loaded on device={self.device} (model={self.model_name})")
            self._loaded = True
            return
        except Exception as e:
            print(f"[EmbeddingModel] could not load sentence-transformers or model failed to initialize: {e}. Falling back to deterministic embeddings.")

        # deterministic pseudo-embeddings fallback (no heavy deps)
        self.model = None
        self._loaded = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embeddings as a 2D numpy array.

        If sentence-transformers is available and was requested, it will be loaded lazily
        and used. Otherwise, a deterministic pseudo-embedding is returned.
        """
        if not texts:
            return np.zeros((0, 0), dtype=float)

        if not self._loaded:
            self._load()

        if self.model is not None:
            embs = self.model.encode(texts, show_progress_bar=False)
            return np.array(embs)

        # Fallback deterministic pseudo-embeddings using hashing
        dim = 384
        out = np.zeros((len(texts), dim), dtype=float)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            rng = np.random.RandomState(seed=h % (2 ** 32))
            out[i] = rng.normal(size=(dim,))
            # normalize
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
