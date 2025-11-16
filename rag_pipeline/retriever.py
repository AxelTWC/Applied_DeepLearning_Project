from typing import List, Dict, Optional
from embeddings import EmbeddingModel
from vectorstore import InMemoryVectorStore
from chunking import FixedSizeChunk
from vectorstore import FassisLocalVectorStore
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import DprQueryEncoder
from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm
import os

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
        self.encoder = DprQueryEncoder("facebook/dpr-question_encoder-single-nq-base", device="mps")
        self.searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-single-nq', self.encoder)
        
    def add_documents(self, documents: List[Dict]):
        pass
        
    def retrieve(self, query: str, top_k: int = 5):
        return self.searcher.search(query, k=top_k)
    
    def preprocess_triviaqa(self, filepath: str, top_k: int = 10, preprocess_length: int = 100, threads : int = 1):
        dataset = load_dataset("trivia_qa", "rc", split="validation").shuffle(seed=1508)
        os.makedirs(filepath, exist_ok=True)
        output_filepath = os.path.join(filepath, f"triviaqa_{preprocess_length}_DPR.json")
        retrieved_docs = []
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                retrieved_docs = json.load(f)
        subset = dataset.select(range(len(retrieved_docs), preprocess_length))
        batch_size = 10
        with tqdm(desc="Preprocessing TriviaQA", total=preprocess_length) as pbar:
            pbar.update(len(retrieved_docs))
            for i in range(0, preprocess_length, batch_size):
                batch = subset.select(range(i, min(i + batch_size, preprocess_length)))
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
    
    def preprocess_nq(self, filepath: str, top_k: int = 10, preprocess_length: int = 100):
        dataset = load_dataset("google-research-datasets/natural_questions", "default", split="validation").shuffle(seed=1508)
        retrieved_docs = []
        idx = 0
        os.makedirs(filepath, exist_ok=True)
        output_filepath = os.path.join(filepath, f"nq_{preprocess_length}_DPR.json")
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                retrieved_docs = json.load(f)
        count = len(retrieved_docs)
        idx = count
        with tqdm(desc="Preprocessing NaturalQuestions", total=preprocess_length) as pbar:
            pbar.update(count)
            while count < preprocess_length:
                short_answers = dataset[idx]['annotations']['short_answers']
                retrieved_doc = {}
                answer_list = []
                answer_set = set()
                for answer in short_answers:
                    if answer['text'] is None or len(answer['text']) == 0:
                        continue
                    if answer['text'][0] in answer_set:
                        continue
                    answer_set.add(answer['text'][0])
                    answer_list.append(answer['text'][0])
                idx += 1
                if len(answer_list) == 0:
                    continue
                print(short_answers)
                retrieved_doc['question_id'] = dataset[idx]['id']
                retrieved_doc['question'] = dataset[idx]['question']['text']
                retrieved_doc['answer'] = answer_list
                # contexts = self.searcher.search(retrieved_doc['question'], k=top_k)
                # context_list = []
                # context_score_list = []
                # for context in contexts:
                #     context_list.append(self.searcher.doc(context.docid).raw()["contents"])
                #     context_score_list.append(float(context.score))
                # retrieved_doc['contexts'] = context_list
                # retrieved_doc['context_scores'] = context_score_list
                retrieved_docs.append(retrieved_doc)
                count += 1
                pbar.update(1)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(retrieved_docs, f, ensure_ascii=False, indent=4)
        return retrieved_docs
    
if __name__ == "__main__":
    retriever = DPRRetriever()
    retriever.preprocess_triviaqa(filepath="../data/", top_k=20, preprocess_length=10)
    # retriever.preprocess_nq(filepath="../data/", top_k=10, preprocess_length=10)