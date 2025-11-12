"""Simple RAG pipeline package (baseline fixed-size chunking)

Modules:
- chunking: fixed-size chunking utilities
- embeddings: embedding model wrapper (sentence-transformers or TF-IDF fallback)
- vectorstore: in-memory vector store with cosine search
- retriever: document ingestion and retrieval
- rag: simple RAG orchestration and a default local generator
"""

from .chunking import split_text_fixed_size
from .embeddings import EmbeddingModel
from .vectorstore import InMemoryVectorStore
from .retriever import Retriever
from .rag import RAG

__all__ = [
    "split_text_fixed_size",
    "EmbeddingModel",
    "InMemoryVectorStore",
    "Retriever",
    "RAG",
]
