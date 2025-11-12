from typing import List, Dict, Optional
from .embeddings import EmbeddingModel
from .vectorstore import InMemoryVectorStore
from .chunking import split_text_fixed_size
import numpy as np


class Retriever:
    """Ingest documents, chunk them and provide retrieval by query.

    Documents are dicts containing at least 'id' and 'text'. Metadata may be provided.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, chunk_size: int = 500, overlap: int = 50):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = InMemoryVectorStore()

    def add_documents(self, documents: List[Dict]):
        """Ingest a list of documents. Each document must have 'id' and 'text' keys.

        Documents will be split into chunks and embedded before being added to the vector store.
        """
        chunk_texts = []
        chunk_ids = []
        metas = []
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = split_text_fixed_size(text, self.chunk_size, self.overlap)
            for i, c in enumerate(chunks):
                chunk_ids.append(f"{doc_id}::chunk_{i}")
                chunk_texts.append(c)
                m = dict(metadata)
                m.update({"source_id": doc_id, "chunk_index": i, "chunk_text": c[:200]})
                metas.append(m)

        if not chunk_texts:
            return

        embeddings = self.embedding_model.embed_texts(chunk_texts)
        # ensure 2D
        embeddings = np.atleast_2d(embeddings)
        self.store.add(chunk_ids, embeddings, metas)

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embedding_model.embed_texts([query])[0]
        return self.store.search(q_emb, top_k=top_k)
