"""Simple RAG pipeline package (baseline fixed-size chunking)

Modules:
- chunking: fixed-size chunking utilities
- embeddings: embedding model wrapper (sentence-transformers or TF-IDF fallback)
- vectorstore: in-memory vector store with cosine search
- retriever: document ingestion and retrieval
- rag: simple RAG orchestration and a default local generator
"""

from .embeddings import EmbeddingModel
from .vectorstore import InMemoryVectorStore
from .retriever import Retriever, BM25Retriever
from .rag import RAG
from .reranker import BgeReranker

__all__ = [
    "EmbeddingModel",
    "InMemoryVectorStore",
    "Retriever",
    "BM25Retriever",
    "RAG",
    "BgeReranker",
]
