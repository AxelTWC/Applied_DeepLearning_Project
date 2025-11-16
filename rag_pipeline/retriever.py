from typing import List, Dict, Optional
from embeddings import EmbeddingModel
from vectorstore import InMemoryVectorStore
from chunking import FixedSizeChunk
from vectorstore import FassisLocalVectorStore
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import DprQueryEncoder
from datasets import load_dataset
from typing import List, Dict, Optional
import numpy as np
import json
from tqdm import tqdm
import os
import torch

class Retriever:
    """Ingest documents, chunk them and provide retrieval by query.

    Documents are dicts containing at least 'id' and 'text'. Metadata may be provided.
    """

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

class FassisLocalRetriever(Retriever):
    """
    Use fassis vector store from Langchain to build retriever with FAISS indexing on local disk
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.store = FassisLocalVectorStore(embedding_model=embedding_model)
        
    def add_documents(self, documents: List[Dict]):
        self.store.store(documents)
        
    def retrieve(self, query: str, top_k: int = 5):
        return self.store.search(query, top_k=top_k)
    
class DPRRetriever(Retriever):
    """
    Use DPR to build retriever with FAISS indexing on local disk
    """
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
        return self.searcher.search(query, k=top_k)
    
    def retrieve_batch(self, questions: List[str], question_ids: List[int] = None, top_k: int = 5, threads: int = 1):
        """
        Retrieve batch of questions
        Args:
            questions (List[str]): List of questions to retrieve documents for.
            question_ids (List[int], optional): List of question IDs corresponding to the questions. Defaults to None. if None, question IDs will be set to range(len(questions)).
            top_k (int, optional): Number of top documents to retrieve for each question. Defaults to 5.
            threads (int, optional): Number of threads to use for retrieval. Defaults to 1.
        Returns: 
            Dict[str, List[DenseSearchResult]]
            returns a dictionary holding the search results, with the query ids as keys and the corresponding lists of search results as the values.
        """
        if question_ids is None:
            question_ids = list(range(len(questions)))
        return self.searcher.batch_search(questions, question_ids, k=top_k, threads=threads)
    
    def preprocess_triviaqa(self, filepath: str, top_k: int = 10, preprocess_length: Optional[int] = None, batch_size: Optional[int]=None, threads : int = 1, split: str = "validation") -> List[Dict]:
        dataset = load_dataset("trivia_qa", "rc", split=split).shuffle(seed=1508)
        os.makedirs(filepath, exist_ok=True)
        if preprocess_length is None:
            preprocess_length = len(dataset)
        if batch_size is None:
            batch_size = preprocess_length
        output_filepath = os.path.join(filepath, f"triviaqa_{split}_{preprocess_length}_DPR.json")
        retrieved_docs = []
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                retrieved_docs = json.load(f)
            print(f"Loaded {len(retrieved_docs)} preprocessed documents from {output_filepath}")
        
        start_idx = len(retrieved_docs)
        remaining_length = min(preprocess_length - start_idx, len(dataset) - start_idx)
        
        if remaining_length <= 0:
            print(f"Already processed {len(retrieved_docs)} documents, target is {preprocess_length}")
            return retrieved_docs
            
        subset = dataset.select(range(start_idx, start_idx + remaining_length))
        with tqdm(desc="Preprocessing TriviaQA", total=preprocess_length) as pbar:
            pbar.update(len(retrieved_docs))
            for i in range(0, remaining_length, batch_size):
                batch_end = min(i + batch_size, remaining_length)
                batch = subset.select(range(i, batch_end))
                question_ids = [doc['question_id'] for doc in batch]
                questions = [doc['question'] for doc in batch]
                answers = [doc['answer']['aliases'] for doc in batch]
                batch_contexts = self.searcher.batch_search(questions, question_ids, k=top_k, threads=threads)
                context_list = []
                context_score_list = []
                for question_id, question, answer in zip(question_ids, questions, answers):
                    retrieved_doc = {}
                    retrieved_doc['question_id'] = question_id
                    retrieved_doc['question'] = question
                    retrieved_doc['answer'] = answer
                    contexts = batch_contexts[question_id]
                    context_score_list = []
                    context_list = []
                    for context in contexts:
                        context_list.append(json.loads(self.searcher.doc(context.docid).raw())["contents"])
                        context_score_list.append(float(context.score))
                    retrieved_doc['contexts'] = context_list
                    retrieved_doc['context_scores'] = context_score_list
                    retrieved_docs.append(retrieved_doc)
                pbar.update(len(batch))
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(retrieved_docs, f, ensure_ascii=False, indent=4)
        return retrieved_docs
    
    def preprocess_nq(self, filepath: str, top_k: int = 10, preprocess_length: Optional[int] = None, threads : int = 1, batch_size: Optional[int]=None, split: str = "validation") -> List[Dict]:
        dataset = load_dataset("google-research-datasets/natural_questions", "default", split=split).shuffle(seed=1508)
        retrieved_docs = []
        idx = 0
        if preprocess_length is None:
            preprocess_length = len(dataset)
        if batch_size is None:
            batch_size = preprocess_length
        os.makedirs(filepath, exist_ok=True)
        output_filepath = os.path.join(filepath, f"nq_{split}_{preprocess_length}_DPR.json")
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                retrieved_docs = json.load(f)
            print(f"Loaded {len(retrieved_docs)} preprocessed documents from {output_filepath}")
        count = len(retrieved_docs)
        idx = retrieved_docs[-1]['question_id'] if len(retrieved_docs) > 0 else 0
        with tqdm(desc="Preprocessing NaturalQuestions", total=preprocess_length) as pbar:
            pbar.update(count)
            while count < preprocess_length and idx < len(dataset):
                question_ids = []
                questions = []
                answers = []
                for _ in range(batch_size):
                    if idx >= len(dataset):
                        break
                    short_answers = dataset[idx]['annotations']['short_answers']
                    retrieved_doc = {}
                    answer_list = []
                    answer_set = set()
                    for answer in short_answers:
                        if answer['text'] is None or len(answer['text']) == 0:
                            continue
                        for txt in answer['text']:
                            if txt in answer_set:
                                continue
                            answer_set.add(txt)
                            answer_list.append(txt)
                    idx += 1
                    if len(answer_list) == 0:
                        if preprocess_length == len(dataset):
                            pbar.update(1)
                        continue
                    question_ids.append(idx)
                    questions.append(dataset[idx]['question']['text'])
                    answers.append(answer_list)
                    count += 1
                    if count >= preprocess_length:
                        break
                batch_contexts = self.searcher.batch_search(questions, question_ids, k=top_k, threads=threads)
                for question_id, question, answer in zip(question_ids, questions, answers):
                    retrieved_doc = {}
                    retrieved_doc['question_id'] = question_id
                    retrieved_doc['question'] = question
                    retrieved_doc['answer'] = answer
                    contexts = batch_contexts[question_id]
                    context_score_list = []
                    context_list = []
                    for context in contexts:
                        context_list.append(json.loads(self.searcher.doc(context.docid).raw())["contents"])
                        context_score_list.append(float(context.score))
                    retrieved_doc['contexts'] = context_list
                    retrieved_doc['context_scores'] = context_score_list
                    retrieved_docs.append(retrieved_doc)
                pbar.update(len(questions))
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(retrieved_docs, f, ensure_ascii=False, indent=4)
            pbar.close()
        return retrieved_docs
    
if __name__ == "__main__":
    retriever = DPRRetriever()
    # retriever.preprocess_triviaqa(filepath="../data/", top_k=20)
    retriever.preprocess_nq(filepath="../data/", top_k=20, preprocess_length=18000 , split="train")