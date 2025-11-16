from chunking import Chunk, FixedSizeChunk
from typing import List, Dict, Tuple, Optional, Iterable
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
import datasets
import pickle
import math

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

def batch_iterator(corpus: Iterable, batch_size: int):
    batch = []
    for example in corpus:
        batch.append(example['text']) 
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class FassisLocalVectorStore:
    """
    Use fassis vector store from Langchain to build vector storage with FAISS indexing on local disk
    """
    
    def __init__(self,
                 embedding_model: str,
                 device: str = "auto"):
        self.embedding_model_name = embedding_model
        # self.embedding_model = EmbeddingModel(model_name = embedding_model, device=device)
        if device == "auto":
            if torch.backends.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': self.device})
        self.vector_store = None
        self.FAISS_PATH = f"../faiss_store/corpus-{self.embedding_model_name}"
        
    def store(self,
              filepath: str = ".../data/source/psgs_w100.tsv") -> FAISS:
        """
        The retrival source will be Wikipedia corpus dumped from Dec. 20, 2018
        """
        
        if not os.path.exists(filepath):
            return

        streaming_dataset = datasets.load_dataset("csv", 
                                 data_files=filepath, 
                                 delimiter="\t", 
                                 streaming=True)
        wikipedia_corpus = streaming_dataset['train']

        chunk_embeddings = []
        chunks = []
        batch_size = 100000
        batched_texts = batch_iterator(wikipedia_corpus, batch_size)
        total_docs = 21015324
        total_batches = math.ceil(total_docs / batch_size)
        k = 0
        
        # skip if the batch_embeddings.pkl and batch.pkl exist
        if os.path.exists(f"{self.FAISS_PATH}/batch_embeddings.pkl") and os.path.exists(f"{self.FAISS_PATH}/batch.pkl"):
            batch_embeddings = pickle.load(open(f"{self.FAISS_PATH}/batch_embeddings.pkl", "rb"))
            chunks = pickle.load(open(f"{self.FAISS_PATH}/batch.pkl", "rb"))
            k = pickle.load(open(f"{self.FAISS_PATH}/k.pkl", "rb"))
            print(f"Resuming from checkpoint: {k} batches completed.")
        
        if k > 0:
            print(f"Skipping {k} batches...")
            for _ in tqdm(range(k), desc="Fast-forwarding"):
                try:
                    next(batched_texts) 
                except StopIteration:
                    print("Warning: Checkpoint 'k' is larger than dataset size.")
                    break
        
        if os.path.exists(f"{self.FAISS_PATH}/index.faiss") == False:
            os.makedirs(self.FAISS_PATH, exist_ok=True)
            with tqdm(total=total_batches, desc="Embedding text", unit="batches") as progress:
                progress.update(k)
                for batch in batched_texts:
                    batch_embeddings = self.embedding_model.embed_documents(batch)
                    chunks.extend(batch)
                    chunk_embeddings.extend(batch_embeddings)
                    progress.update(1)
                    k += 1
                    if k % 100 == 0:
                        pickle.dump(batch_embeddings, open(f"{self.FAISS_PATH}/batch_embeddings.pkl", "wb"))
                        pickle.dump(chunks, open(f"{self.FAISS_PATH}/batch.pkl", "wb"))
                        pickle.dump(k, open(f"{self.FAISS_PATH}/k.pkl", "wb"))
                        print(f"Saved checkpoint at {k} batches")

            text_embeddings = list(zip(chunks, chunk_embeddings))
            self.vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=self.embedding_model)
            self.vector_store.save_local(self.FAISS_PATH)
        else:
            self.vector_store = FAISS.load_local(self.FAISS_PATH, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        return self.vector_store
    
    def _load(self):
        if os.path.exists(f"{self.FAISS_PATH}/index.faiss"):
            self.vector_store = FAISS.load_local(self.FAISS_PATH, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        else:
            raise ValueError("Vector store is not initialized. Please call store() method first.")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        if self.vector_store is None:
            self._load()
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs_with_scores:
            results.append((doc.page_content, score, doc.metadata))
        return results
    
if __name__ == "__main__":
    # chunk_strategy = FixedSizeChunk()
    # vector_store = FassisLocalVectorStore(embedding_model="BAAI/bge-large-en-v1.5", device="mps")
    # vector_store.store(filepath="../data/source/psgs_w100.tsv")
    # results = vector_store.search(query="What is the capital of France?", top_k=10)
    # for (text, score, metadata) in results:
    #     print(text)
    #     print("*"*50)
    #     print("Score: ", score)
    #     print("Metadata: ", metadata)
    #     print("-"*100)
    
    # Dense Retriever (DPR)
    from pyserini.search.faiss import FaissSearcher
    from pyserini.encode import DprQueryEncoder

    # Load query encoder
    encoder = DprQueryEncoder("facebook/dpr-question_encoder-single-nq-base")
    # Use Wikipedia dump as the retrieval source
    searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-single-nq', encoder)
    # Retrieve documents relevant to the given query
    questions = ['who got the first nobel prize in physics', 'Who was the next British Prime Minister after Arthur Balfour?']
    for question in questions:
        hits = searcher.search(question, k=5)
        # print(hits)
        # Present retrieved document and relevance score
        for hit in hits:
            print(f'doc: {searcher.doc(hit.docid).raw()}\nscore: {hit.score}')
            print("")
        print("-"*100)