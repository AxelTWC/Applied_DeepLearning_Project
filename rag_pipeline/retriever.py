from curses import raw
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from .embeddings import EmbeddingModel
from .vectorstore import InMemoryVectorStore
from .chunking import FixedSizeChunk
from .vectorstore import FaissLocalVectorStore
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.encode import DprQueryEncoder
from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
from tqdm import tqdm
import os
import torch

class Retriever:
    """Ingest documents, chunk them and provide retrieval by query.

    Documents are dicts containing at least 'id' and 'text'. Metadata may be provided.
    """
    
    name: str = "BaseRetriever"

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, chunk_size: int = 500, overlap: int = 50):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = InMemoryVectorStore()
        self.chunker = FixedSizeChunk()

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
            chunks = self.chunker.apply(text, self.chunk_size, self.overlap)
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
    
    def retrieve_batch(self, questions: List[str], question_ids: List[int] = None, top_k: int = 5, threads: int = 1):
        answers = []
        for question in questions:
            answers.append(self.retrieve(question, top_k=top_k))
        return answers
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.store.get_document(doc_id)
    
class DPRRetriever(Retriever):
    """
    Use DPR to build retriever with FAISS indexing on local disk
    """
    name: str = "DPR"
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.encoder = DprQueryEncoder("facebook/dpr-question_encoder-single-nq-base", device=self.device)
        self.searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-single-nq', self.encoder)
        
    def add_documents(self, documents: List[Dict]):
        pass
        
    def retrieve(self, query: str, top_k: int = 5):
        raw_contexts = self.searcher.search(query, k=top_k)
        results = []
        scores = []
        for context in raw_contexts:
            results.append(json.loads(self.searcher.doc(context.docid).raw())["contents"])
            scores.append(float(context.score))
        return raw_contexts, results, scores
    
    def retrieve_batch(self, questions: List[str], question_ids: List[int] = None, top_k: int = 5, threads: int = 1) -> Tuple[Dict[str, List], Dict[str, Tuple[List, List]]]:
        """
        Retrieve batch of questions
        Args:
            questions (List[str]): List of questions to retrieve documents for.
            question_ids (List[int], optional): List of question IDs corresponding to the questions. Defaults to None. if None, question IDs will be set to range(len(questions)).
            top_k (int, optional): Number of top documents to retrieve for each question. Defaults to 5.
            threads (int, optional): Number of threads to use for retrieval. Defaults to 1.
        Returns: 
            Dict[str, List[DenseSearchResult]] - a dictionary holding the raw search results, with the query ids as keys and the corresponding lists of search results as the values.
            Dict[str, Tuple[List, List]] - a dictionary holding the corresponding list of retrieved text contexts and list of scores with question id as keys.
        """
        if question_ids is None:
            question_ids = list(range(len(questions)))
        batch_contexts = self.searcher.batch_search(questions, question_ids, k=top_k, threads=threads)
        answers = {}
        for question_id in question_ids:
            question = questions[question_id]
            results = []
            scores = []
            contexts = batch_contexts[question_id]
            for context in contexts:
                results.append(json.loads(self.searcher.doc(context.docid).raw())["contents"])
                scores.append(float(context.score))
            answers[question_id] = (results, scores)
        return batch_contexts, answers
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.searcher.get_document(doc_id)
    
class BM25Retriever(Retriever):
    """
    Use BM25 to build retriever with FAISS indexing on local disk
    """
    
    name: str = "BM25"
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
        
    def add_documents(self, documents: List[Dict]):
        pass
        
    def retrieve(self, query: str, top_k: int = 5):
        contexts = self.searcher.search(query, k=top_k)
        answers = [
            json.loads(self.searcher.doc(context.docid).raw())["contents"] for context in contexts
        ]
        scores = [
            float(context.score) for context in contexts
        ]
        return contexts, answers, scores
    
    def retrieve_batch(self, questions: List[str], question_ids: List[int] = None, top_k: int = 5, threads: int = 1) -> Tuple[Dict[str, List], Dict[str, Tuple[List, List]]]:
        """
        Retrieve batch of questions
        Args:
            questions (List[str]): List of questions to retrieve documents for.
            question_ids (List[int], optional): List of question IDs corresponding to the questions. Defaults to None. if None, question IDs will be set to range(len(questions)).
            top_k (int, optional): Number of top documents to retrieve for each question. Defaults to 5.
            threads (int, optional): Number of threads to use for retrieval. Defaults to 1.
        Returns: 
            Dict[str, List] - a dictionary holding the raw search results, with the query ids as keys and the corresponding lists of search results as the values.
            Dict[str, Tuple[List, List]] - a dictionary holding the corresponding list of retrieved text contexts and list of scores with question id as keys.
        """
        if question_ids is None:
            assert False, "question_ids cannot be None for BM25Retriever" 
        batch_contexts = self.searcher.batch_search(questions, question_ids, k=top_k, threads=threads)
        answers = {}
        for question, question_id in zip(questions, question_ids):
            contexts = batch_contexts[question_id]
            results = [json.loads(context.lucene_document.get('raw')).get("contents") for context in contexts]
            scores = [float(context.score) for context in contexts]
            answers[question_id] = (results, scores)
        return batch_contexts, answers
    
class FaissLocalRetriever(Retriever):
    name: str = "BgeM3Faiss"
    def __init__(self, embedding_model: str = "BAAI/bge-m3", faiss_index_path: str = "../faiss_store"):
        self.store = FaissLocalVectorStore(embedding_model=embedding_model, faiss_index_path=faiss_index_path)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        
    def retrieve(self, query: str, top_k: int = 5):
        context_raw = self.store.search(query, top_k=top_k)
        answers = [res["text"] for res in context_raw]
        scores = [res["score"] for res in context_raw]
        return context_raw, answers, scores
    
    def retrieve_batch(self, questions: List[str], question_ids: List[int] = None, top_k: int = 5, threads: int = 1) -> Tuple[Dict[str, List], Dict[str, Tuple[List, List]]]:
        """
        Retrieve batch of questions
        Args:
            questions (List[str]): List of questions to retrieve documents for.
            question_ids (List[int], optional): List of question IDs corresponding to the questions. Defaults to None. if None, question IDs will be set to range(len(questions)).
            top_k (int, optional): Number of top documents to retrieve for each question. Defaults to 5.
            threads (int, optional): Number of threads to use for retrieval. Defaults to 1.
        Returns: 
            Dict[str, List] - a dictionary holding the raw search results, with the query ids as keys and the corresponding lists of search results as the values.
            Dict[str, Tuple[List, List]] - a dictionary holding the corresponding list of retrieved text contexts and list of scores with question id as keys.
        """
        if question_ids is None:
            question_ids = list(map(str, range(len(questions))))
        batch_contexts = self.store.batch_search(questions, question_ids, top_k=top_k, threads=threads)
        answers = {}
        for question, question_id in zip(questions, question_ids):
            contexts = batch_contexts[question_id]
            texts = [res["text"] for res in contexts]
            scores = [res["score"] for res in contexts]
            answers[question_id] = (texts, scores)
        return batch_contexts, answers
    
if __name__ == "__main__":
    question = "KHI is the international code for which Asian airport?"
    retriever = BM25Retriever()
    answers = retriever.retrieve(question)
    for answer in answers:
        print(f"DocID: {answer.docid}, Score: {answer.score}")
        print(json.loads(answer.lucene_document.get('raw')).get("contents"))
        print("-"*50)
    from reranker import BgeReranker
    reranker = BgeReranker()
    contexts = [json.loads(answer.lucene_document.get('raw')).get("contents") for answer in answers]
    ranked_contexts = reranker.rerank(question, contexts)
    print("Ranked Contexts:")
    for context in ranked_contexts:
        print(context)
        print("-"*50)