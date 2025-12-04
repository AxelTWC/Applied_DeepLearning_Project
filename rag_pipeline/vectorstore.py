import os
import sys

__package__ = "rag_pipeline"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Tuple, Optional, Iterable
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import os
import torch
import faiss
import datasets
import sqlite3

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

class FaissLocalVectorStore:
    """
    Use faiss vector store from Langchain to build vector storage with FAISS indexing on local disk
    """
    
    def __init__(self,
                 embedding_model: str,
                 device: str = "auto",
                 param: str = "IVF65536,PQ128",
                 n_probe: int = 32,
                 faiss_index_path: str = "../faiss_store"):
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
        self.FAISS_PATH = f"{faiss_index_path}/corpus-{self.embedding_model_name}-{param}"
        os.makedirs(self.FAISS_PATH, exist_ok=True)
        self.INDEX_FILE = os.path.join(self.FAISS_PATH, "wikipedia.index")
        self.CHECKPOINT_FILE = os.path.join(self.FAISS_PATH, "checkpoint.index")
        self.DB_PATH = os.path.join(self.FAISS_PATH, "docstore.db")
        self.INDEX_PARAM = param
        self.n_probe = 32
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def _load(self):
        if os.path.exists(self.INDEX_FILE):
            self.vector_store = faiss.read_index(self.INDEX_FILE)
            self.vector_store.nprobe = self.n_probe
            self.conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
            self.cursor = self.conn.cursor()
        else:
            raise ValueError("Vector store is not initialized. Please call store() method first.")
    
    def batch_search(self, queries: List[str], question_ids: List = None, top_k: int = 10, threads:int = 1) -> List[List[Tuple[str, float, Dict]]]:
        if self.vector_store is None or self.conn is None or self.cursor is None:
            self._load()
        query_embeddings_np = self.embedding_model.encode(
            sentences=queries,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        faiss.omp_set_num_threads(threads)
        D, I = self.vector_store.search(query_embeddings_np, top_k)
        results = {}
        for q_id, idx in zip(range(len(queries)), question_ids if question_ids is not None else map(str, range(len(queries)))):
            q_results = []
            for rank in range(top_k):
                doc_id = I[q_id][rank]
                score = D[q_id][rank]
                if doc_id < 0:
                    continue
                try:
                    doc = self.cursor.execute("SELECT title, text, url FROM documents WHERE id=?", (int(doc_id),))
                    row = self.cursor.fetchone()
                    q_results.append({
                        "id": doc_id,
                        "score": float(score),
                        "title": row[0],
                        "text": row[1],
                        "url": row[2]
                    })
                except IndexError:
                    print(f"Warning: doc_id {doc_id} out of range for docstore of size {len(self.docstore)}")
                    continue
            results[idx] = q_results
        
        return results
    
    def search(self, query: str, top_k: int = 10, threads:int = 1) -> List[Tuple[str, float, Dict]]:
        return self.batch_search([query], top_k=top_k, threads=threads)["0"]
    
    def build_faiss_index(self,
                          datasetname: str = "Upstash/wikipedia-2024-06-bge-m3",
                          batch_size: int = 10000,
                          checkpoint_interval: int = 500000, # vectors between checkpoints
                          ntrain: int = 1000000, # Number of vectors for training
                          dimension: int = 1024,
                          threads: int = 1,
                          embedding_column: str = "embedding") -> faiss.Index:
        
        # init DB for docstore
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                text TEXT,
                url TEXT
            )
        ''')
        conn.commit()
        cursor.execute('PRAGMA journal_mode=WAL;')
        
        os.makedirs(self.FAISS_PATH, exist_ok=True)
        OUTPUT_FILENAME = self.INDEX_FILE
        CHECKPOINT_PATH = self.CHECKPOINT_FILE
        faiss.omp_set_num_threads(threads)
        start_offset = 0
        
        # load from checkpoint if exists
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
            try:
                index = faiss.read_index(CHECKPOINT_PATH)
                start_offset = index.ntotal
                print(f"Resuming from index count: {start_offset}")
            except Exception as e:
                print(f"Checkpoint corrupted ({e}). Starting fresh.")
                index = faiss.index_factory(dimension, self.INDEX_PARAM, faiss.METRIC_L2)
        else:
            print("No checkpoint found. Creating new index.")
            index = faiss.index_factory(dimension, self.INDEX_PARAM, faiss.METRIC_L2)
        
        if not index.is_trained:
            train_stream = datasets.load_dataset(datasetname, "en", split="train", streaming=True)
            train_stream = train_stream.shuffle(buffer_size=10000, seed=1502)
            train_iterable = train_stream.take(ntrain)

            training_vectors = []
            for item in tqdm(train_iterable, total=ntrain, desc="Loading Train Data"):
                emb = item.get(embedding_column)
                if emb is not None:
                    training_vectors.append(emb)

            print("Converting to numpy")
            training_data = np.array(training_vectors).astype('float32')
            
            print("Training Index")
            index.train(training_data)
            print("✅ Training complete.")
            
            # Checkpoint after training
            faiss.write_index(index, CHECKPOINT_PATH)
            
            del training_data
            del training_vectors
            import gc
            gc.collect()

        total_rows = 47018430
        print(f"Detected total rows for 'en': {total_rows}")
        dataset = datasets.load_dataset(datasetname, "en" , split="train", streaming=True)
        batch_embeddings = []
        batch_metadata = []
        vectors_added_since_last_save = 0
        current_id = start_offset
        
        if start_offset > 0:
            print(f"Skipping first {start_offset} rows in dataset stream...")
            dataset = dataset.skip(start_offset)
        
        try:
            for i, item in tqdm(enumerate(dataset), initial=start_offset, desc="Indexing"):
                # Extract embedding
                emb = item.get(embedding_column)
                title = item.get("title", "")
                text = item.get("text", "")
                url = item.get("url", "")
                
                batch_metadata.append((current_id, title, text, url))
                current_id += 1
                
                if emb is None:
                    continue

                batch_embeddings.append(emb)

                # Process batch when full
                if len(batch_embeddings) >= batch_size:
                    # Convert to float32 numpy array required by FAISS
                    batch_matrix = np.array(batch_embeddings).astype('float32')
                    
                    index.add(batch_matrix)
                    cursor.executemany('INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)', batch_metadata)
                    conn.commit()
                    batch_metadata = []  # Clear metadata buffer
                    vectors_added_since_last_save += len(batch_embeddings)
                    batch_embeddings = []  # Clear buffer
                
                if vectors_added_since_last_save >= checkpoint_interval:
                        print(f"\n💾 Saving checkpoint at {index.ntotal} vectors...")
                        temp_ckpt = CHECKPOINT_PATH + ".tmp"
                        faiss.write_index(index, temp_ckpt)
                        if os.path.exists(CHECKPOINT_PATH):
                            os.remove(CHECKPOINT_PATH)
                        os.rename(temp_ckpt, CHECKPOINT_PATH)
                        
                        vectors_added_since_last_save = 0
                        print("✅ Checkpoint saved.")

            # Add remaining items
            if batch_embeddings:
                batch_matrix = np.array(batch_embeddings).astype('float32')
                index.add(batch_matrix)
                cursor.executemany('INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)', batch_metadata)
                conn.commit()
                
            print(f"\nFinished! Total vectors indexed: {index.ntotal}")
            
            # Save the index to disk
            print(f"Saving index to {OUTPUT_FILENAME}...")
            faiss.write_index(index, OUTPUT_FILENAME)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_id ON documents (id)')
            conn.commit()
            conn.close()
            print("Done.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            conn.close()
            # Attempt to save what we have so far
            if index.ntotal > start_offset:
                print(f"Saving partial index with {index.ntotal} vectors...")
                faiss.write_index(index, CHECKPOINT_PATH)        
    
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
    # vector_store.build_faiss_index(datasetname="Upstash/wikipedia-2024-06-bge-m3", batch_size=100000, dimension=1024, threads=1, embedding_column="embedding")
    results = vector_store.search(query="who's the original singer of help me make it through the night?", top_k=10)
    for res in results:
        print(res["text"])
        print("*"*50)
        print("Score: ", res["score"])
        print("Metadata: ", {"title": res["title"], "url": res["url"]})
        print("-"*100)