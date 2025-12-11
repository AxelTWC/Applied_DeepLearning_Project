import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline.retriever import BM25Retriever, Retriever, FaissLocalRetriever
from rag_pipeline.reranker import BgeReranker, Reranker
from rag_pipeline.prompt import ADAPTIVE_ROUTER_SYSTEM_PROMPT, ADAPTIVE_ROUTER_INITIAL_PROMPT
import json
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm

def preprocess_triviaqa(filepath: str, retriever:Retriever, reranker:Reranker, top_k: int = 10, preprocess_length: Optional[int] = None, batch_size: Optional[int]=None, threads : int = 1, split: str = "validation") -> List[Dict]:
    dataset = load_dataset("trivia_qa", "rc", split=split).shuffle(seed=1508)
    os.makedirs(filepath, exist_ok=True)
    if preprocess_length is None:
        preprocess_length = len(dataset)
    if batch_size is None:
        batch_size = preprocess_length
    output_filepath = os.path.join(filepath, f"triviaqa_{split}_{preprocess_length}_{retriever.name}{"_" + reranker.name if reranker else ""}.json")
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
            batch_contexts_raw, batch_contexts = retriever.retrieve_batch(questions, question_ids, top_k=top_k, threads=threads)
            for question_id, question, answer in zip(question_ids, questions, answers):
                retrieved_doc = {}
                retrieved_doc['question_id'] = question_id
                retrieved_doc['question'] = question
                retrieved_doc['answer'] = answer
                contexts, context_scores = batch_contexts[question_id]
                if reranker:
                    contexts, context_scores = reranker.rerank(question, contexts)
                retrieved_doc['contexts'] = contexts
                retrieved_doc['context_scores'] = context_scores
                retrieved_docs.append(retrieved_doc)
            pbar.update(len(batch))
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(retrieved_docs, f, ensure_ascii=False, indent=4)
    return retrieved_docs

def preprocess_nq(filepath: str, retriever:Retriever, reranker:Optional[Reranker], top_k: int = 10, preprocess_length: Optional[int] = None, threads : int = 1, batch_size: Optional[int]=None, split: str = "validation") -> List[Dict]:
    dataset = load_dataset("google-research-datasets/natural_questions", "default", split=split).shuffle(seed=1508)
    retrieved_docs = []
    idx = 0
    if preprocess_length is None:
        preprocess_length = len(dataset)
    if batch_size is None:
        batch_size = preprocess_length
    os.makedirs(filepath, exist_ok=True)
    output_filepath = os.path.join(filepath, f"nq_{split}_{preprocess_length}_{retriever.name}{"_" + reranker.name if reranker else ""}.json")
    if os.path.exists(output_filepath):
        with open(output_filepath, 'r', encoding='utf-8') as f:
            retrieved_docs = json.load(f)
        print(f"Loaded {len(retrieved_docs)} preprocessed documents from {output_filepath}")
    idx = retrieved_docs[-1]['question_id'] if len(retrieved_docs) > 0 else 0
    with tqdm(desc="Preprocessing NaturalQuestions", total=preprocess_length) as pbar:
        pbar.update(len(retrieved_docs))
        while len(retrieved_docs) < preprocess_length and idx < len(dataset):
            question_ids = []
            questions = []
            answers = []
            while len(question_ids) < batch_size:
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
                if len(answer_list) == 0:
                    if preprocess_length == len(dataset):
                        pbar.update(1)
                    idx += 1
                    continue
                question_ids.append(str(idx))
                questions.append(dataset[idx]['question']['text'])
                answers.append(answer_list)
                idx += 1
                if len(retrieved_docs) + len(questions) >= preprocess_length:
                    break
            batch_contexts_raw, batch_contexts = retriever.retrieve_batch(questions, question_ids, top_k=top_k, threads=threads)
            for question_id, question, answer in zip(question_ids, questions, answers):
                retrieved_doc = {}
                retrieved_doc['question_id'] = question_id
                retrieved_doc['question'] = question
                retrieved_doc['answer'] = answer
                contexts, context_scores = batch_contexts[question_id]
                if reranker:
                    contexts, context_scores = reranker.rerank(question, contexts)
                retrieved_doc['contexts'] = contexts
                retrieved_doc['context_scores'] = context_scores
                retrieved_docs.append(retrieved_doc)
            pbar.update(len(questions))
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(retrieved_docs, f, ensure_ascii=False, indent=4)
        pbar.close()
    return retrieved_docs

def format_nq_data(output_filepath:str, process_length: Optional[int] = None) -> List[Dict]:
    dataset = load_dataset("google-research-datasets/natural_questions", "default", split="train").shuffle(seed=1508)
    if process_length is None:
        process_length = len(dataset)
    samples = []
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    idx = 0
    with open(output_filepath, 'a', encoding='utf-8') as f:
        with tqdm(desc="Formatting NQ Data", total=process_length) as pbar:
            while len(samples) < process_length:
                if idx >= len(dataset) - 1:
                    break
                sample = dataset[idx]
                short_answers = sample['annotations']['short_answers']
                if not short_answers:
                    idx += 1
                    continue
                answer_list = []
                answer_set = set()
                for answer in short_answers:
                    if answer['text'] is None or len(answer['text']) == 0:
                        idx += 1
                        continue
                    for txt in answer['text']:
                        if txt in answer_set:
                            continue
                        answer_set.add(txt)
                        answer_list.append(txt)
                if len(answer_list) == 0:
                    idx += 1
                    continue
                messages = [
                    {"role": "system", "content": ADAPTIVE_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": ADAPTIVE_ROUTER_INITIAL_PROMPT.format(question=sample['question']['text'])}
                ]
                samples.append({
                    'messages': messages,
                    'ground_truth': answer_list,
                    'question': sample['question']['text']
                })
                f.write(json.dumps(samples[-1], ensure_ascii=False) + '\n')
                idx += 1
                pbar.update(1)
            pbar.close()
    return samples

def format_triviaqa_data(output_filepath:str, process_length: Optional[int] = None) -> List[Dict]:
    dataset = load_dataset("trivia_qa", "rc", split="train").shuffle(seed=1508)
    if process_length is None:
        process_length = len(dataset)
    samples = []
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    idx = 0
    with open(output_filepath, 'a', encoding='utf-8') as f:
        with tqdm(desc="Formatting TriviaQA Data", total=process_length) as pbar:
            while len(samples) < process_length:
                if idx >= len(dataset) - 1:
                    break
                sample = dataset[idx]
                question = sample['question']
                answers = sample['answer']['aliases']
                messages = [
                    {"role": "system", "content": ADAPTIVE_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": ADAPTIVE_ROUTER_INITIAL_PROMPT.format(question=question)}
                ]
                samples.append({
                    'messages': messages,
                    'ground_truth': answers,
                    'question': question,
                })
                f.write(json.dumps(samples[-1], ensure_ascii=False) + '\n')
                idx += 1
                pbar.update(1)
            pbar.close()
    return samples
if __name__ == "__main__":
    retriever = BM25Retriever()
    reranker = None
    # preprocess_triviaqa(
    #     filepath="../data/preprocessed/",
    #     retriever=retriever,
    #     reranker=reranker,
    #     top_k=20,
    #     preprocess_length=1000,
    #     threads=1,
    #     split="validation"
    # )
    # preprocess_nq(
    #     filepath="../data/preprocessed/",
    #     retriever=retriever,
    #     reranker=reranker,
    #     top_k=20,
    #     preprocess_length=1000,
    #     threads=1,
    #     split="validation"
    # )
    dataset_length = 1000
    nq_length = int(dataset_length * 0.6)
    trivia_length = dataset_length - nq_length
    format_nq_data(output_filepath=f"../dataset/RAG_{dataset_length}.jsonl", process_length=nq_length)
    format_triviaqa_data(output_filepath=f"../dataset/RAG_{dataset_length}.jsonl", process_length=trivia_length)
    
    