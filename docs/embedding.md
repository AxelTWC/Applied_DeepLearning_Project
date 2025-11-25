Imagine you have a magical machine that can turn words into numbers so the computer can understand them.
This file is that magic machine.

Core:
This module defines a robust EmbeddingModel class that:

Lazily loads a sentence-transformers embedding model.

Can auto-select CPU/GPU based on system capabilities.

Can fallback to deterministic pseudo-embeddings when:

GPU VRAM is too small

dependencies are missing

loading fails

Provides fault-tolerant, reproducible embeddings for RAG Pipeline

______

Embedding is the backbone of retrieval:

You take text → turn it into a vector representation.

Vectors go into a vector store (FAISS, Milvus, NumPy store in your baseline).

Retrieval later returns the nearest-neighbor vectors for RAG.