from chunking import Chunk, FixedSizeChunk
from typing import List, Dict, Tuple, Optional, Iterable, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
import faiss
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
        batch.append(example) 
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
                 device: str = "auto",
                 param: str = "HNSW64"):
        self.embedding_model_name = embedding_model
        # self.embedding_model = EmbeddingModel(model_name = embedding_model, device=device)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.vector_store: faiss.Index | None = None 
        self.docstore: List[Dict[str, Any]] = []
        self.FAISS_PATH = f"../faiss_store/corpus-{self.embedding_model_name}-{param}"
        os.makedirs(self.FAISS_PATH, exist_ok=True)
        self.INDEX_FILE = os.path.join(self.FAISS_PATH, "index.faiss")
        self.DOCSTORE_FILE = os.path.join(self.FAISS_PATH, "docstore.pkl")
        self.CHECKPOINT_FILE = os.path.join(self.FAISS_PATH, "k.pkl")
        self.INDEX_PARAM = param
        
    def store_wikipedia(self,
              filepath: str = ".../data/source/psgs_w100.tsv",
              total_docs: int = 21015324,
              batch_size: int = 10000,
              threads: int = 1) -> faiss.Index:
        """
        The retrival source will be Wikipedia corpus dumped from Dec. 20, 2018
        """
        
        if not os.path.exists(filepath):
            return
        
        assert batch_size <= total_docs, "batch_size cannot be larger than total_docs"

        streaming_dataset = datasets.load_dataset("csv", 
                                 data_files=filepath, 
                                 delimiter="\t", 
                                 column_names=['id', 'text', 'title'],
                                 streaming=True)
        wikipedia_corpus = streaming_dataset['train']

        batches = batch_iterator(wikipedia_corpus, batch_size)
        total_batches = math.ceil(total_docs / batch_size)
        k = 0
        total_docs_processed = 0
        faiss.omp_set_num_threads(threads)
        
        # skip if checkpoint exists
        if os.path.exists(self.INDEX_FILE):
            try:
                self.vector_store = faiss.read_index(self.INDEX_FILE)
                self.docstore = pickle.load(open(self.DOCSTORE_FILE, "rb"))
                self.k = pickle.load(open(self.CHECKPOINT_FILE, "rb"))
                print(f"Resuming from checkpoint: {self.k} batches completed.")
                total_docs_processed = self.vector_store.ntotal
                assert total_docs_processed == len(self.docstore), "Inconsistent docstore and index sizes"
                if total_docs_processed == total_docs:
                    print("Index already complete. Exiting store process.")
                    return self.vector_store
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                self.k = 0
                total_docs_processed = 0
                self.vector_store = None
                self.docstore = []
        
        if k > 0:
            print(f"Skipping {k} batches...")
            for _ in tqdm(range(k), desc="Fast-forwarding"):
                try:
                    next(batches) 
                except StopIteration:
                    print("Warning: Checkpoint 'k' is larger than dataset size.")
                    break
        
        # initalize FAISS index
        if self.vector_store is None:
            base_vector = faiss.index_factory(self.embedding_model.get_sentence_embedding_dimension(), self.INDEX_PARAM, faiss.METRIC_L2)
            self.vector_store = faiss.IndexIDMap(base_vector)  
        
        with tqdm(total=total_batches, desc="Embedding text", unit="batches") as progress:
            progress.update(k)
            for batch in batches:
                texts_to_embed = [doc['text'] for doc in batch if doc.get('text')]
                batch_embeddings_np = self.embedding_model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
                start_idx = total_docs_processed
                end_idx = start_idx + len(batch_embeddings_np)
                ids = np.arange(start_idx, end_idx).astype(np.int64)
                self.vector_store.add_with_ids(batch_embeddings_np, ids)
                for i, doc in enumerate(batch):
                    metadata = {"id": doc["id"], "title": doc["title"]}
                    self.docstore.append({"content": doc["text"], "metadata": metadata})
                progress.update(1)
                k += 1
                total_docs_processed += len(batch_embeddings_np)
                if total_docs_processed >= total_docs:
                    break
                
                if k % 100 == 0:
                    faiss.write_index(self.vector_store, self.INDEX_FILE)
                    pickle.dump(self.docstore, open(self.DOCSTORE_FILE, "wb"))
                    pickle.dump(k, open(self.CHECKPOINT_FILE, "wb"))
                    print(f"Saved checkpoint at {k} batches")
            progress.close()

        faiss.write_index(self.vector_store, self.INDEX_FILE)
        pickle.dump(self.docstore, open(self.DOCSTORE_FILE, "wb"))
        pickle.dump(k, open(self.CHECKPOINT_FILE, "wb"))
        return self.vector_store
    
    def _load(self):
        if os.path.exists(self.INDEX_FILE):
            self.vector_store = faiss.read_index(self.INDEX_FILE)
            self.docstore = pickle.load(open(self.DOCSTORE_FILE, "rb"))
        else:
            raise ValueError("Vector store is not initialized. Please call store() method first.")
    def batch_search(self, queries: List[str], top_k: int = 10, threads:int = 1) -> List[List[Tuple[str, float, Dict]]]:
        if self.vector_store is None or self.docstore is None:
            self._load()
        query_embeddings_np = self.embedding_model.encode(
            queries, 
            convert_to_numpy=True
        ).astype(np.float32)
        faiss.omp_set_num_threads(threads)
        D, I = self.vector_store.search(query_embeddings_np, top_k)
        results = []
        for q_id in range(len(queries)):
            q_results = []
            for rank in range(top_k):
                doc_id = I[q_id][rank]
                score = D[q_id][rank]
                if doc_id < 0:
                    continue
                try:
                    doc = self.docstore[doc_id]
                    q_results.append((doc["content"], score, doc["metadata"]))
                except IndexError:
                    print(f"Warning: doc_id {doc_id} out of range for docstore of size {len(self.docstore)}")
                    continue
            results.append(q_results)
        
        return results
    
    def search(self, query: str, top_k: int = 10, threads:int = 1) -> List[Tuple[str, float, Dict]]:
        return self.batch_search([query], top_k=top_k)[0]
    
    def build_faiss_index(self, datasetname: str = "Upstash/wikipedia-2024-06-bge-m3", batch_size: int = 10000, dimension: int = 1024, threads: int = 1, embedding_column: str = "embedding") -> faiss.Index:
        dataset = datasets.load_dataset(datasetname, split="train", streaming=True)
        index = faiss.IndexFlatIP(dimension)
        batch_embeddings = []
        count = 0
        faiss.omp_set_num_threads(threads)
        os.makedirs(self.FAISS_PATH, exist_ok=True)
        OUTPUT_FILENAME = f"{self.FAISS_PATH}/{datasetname.split('/')[-1]}.index"
        
        try:
            for i, item in tqdm(enumerate(dataset)):
                # Extract embedding
                emb = item.get(embedding_column)
                
                if emb is None:
                    continue

                batch_embeddings.append(emb)

                # Process batch when full
                if len(batch_embeddings) >= batch_size:
                    # Convert to float32 numpy array required by FAISS
                    batch_matrix = np.array(batch_embeddings).astype('float32')
                    
                    index.add(batch_matrix)
                    
                    count += len(batch_embeddings)
                    batch_embeddings = []  # Clear buffer
                    
                    # Optional: specific print every 100k
                    if count % 100000 == 0:
                        print(f"Indexed {count} documents so far...")

            # Add remaining items
            if batch_embeddings:
                batch_matrix = np.array(batch_embeddings).astype('float32')
                index.add(batch_matrix)
                count += len(batch_embeddings)

            print(f"\nFinished! Total vectors indexed: {count}")
            
            # Save the index to disk
            print(f"Saving index to {OUTPUT_FILENAME}...")
            faiss.write_index(index, OUTPUT_FILENAME)
            print("Done.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Attempt to save what we have so far
            if count > 0:
                print(f"Saving partial index with {count} vectors...")
                faiss.write_index(index, "partial_backup.index")        
    
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
    
    # # # Dense Retriever (DPR)
    # from pyserini.search.faiss import FaissSearcher
    # from pyserini.encode import DprQueryEncoder

    # # # Load query encoder
    # encoder = DprQueryEncoder("facebook/dpr-question_encoder-single-nq-base")
    # # Use Wikipedia dump as the retrieval source
    # searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-single-nq', encoder)
    # # Retrieve documents relevant to the given query
    # questions = ['KHI international airport code Asia']
    # for question in questions:
    #     hits = searcher.search(question, k=5)
    #     # print(hits)
    #     # Present retrieved document and relevance score
    #     for hit in hits:
    #         print(f'doc: {searcher.doc(hit.docid).raw()}\nscore: {hit.score}')
    #         print("")
    #     print("-"*100)
    
    # Build FAISS index for Upstash/wikipedia-2024-06-bge-m3
    vector_store = FassisLocalVectorStore(embedding_model="BAAI/bge-m3", device="auto")
    vector_store.build_faiss_index(datasetname="Upstash/wikipedia-2024-06-bge-m3", batch_size=10000, dimension=1024, threads=1, embedding_column="embedding")