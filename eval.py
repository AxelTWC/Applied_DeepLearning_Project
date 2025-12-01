import argparse
import requests
import json
import os
import shutil
from typing import List, Dict, Optional
from rag_pipeline.generator import Generator_API
from rag_pipeline.router import AdaptiveRouter
from rag_pipeline.retriever import FaissLocalRetriever
from utils.metrics import exact_match
from tqdm import tqdm
from rag_pipeline.rag import AdaptiveRAG
from rag_pipeline.reranker import BgeReranker

class Config:
    def __init__(self, model: str, filepath: str, context_num: int, context_free: bool, question_limit: Optional[int] = None, base_url: str = "http://localhost:8080", reranker: bool = False):
        self.model = model
        self.filepath = filepath
        self.context_num = context_num
        self.context_free = context_free
        self.question_limit = question_limit
        self.base_url = base_url
        self.reranker = reranker
        
def generate(messages,
             base_url: str = "http://localhost:8080",
             temperature=1.0,
             top_p=0.9,
             top_k=40):
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "enable_thinking": False,
        "stream": False 
    }
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, data=json.dumps(data))
    return response.json()["choices"][0]["message"]["content"]

def get_context_prompt(question:str, contexts:list[str]) -> List[Dict]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions with some reference information. You need to directly give the answer without any other text./no_think"},
        {"role": "user", "content": f"You need to directly answer the question without any other text. You will be provided with some reference information to help you answer the question.\n\nQuestion: {question}\n\nReference Information: {"\n".join(contexts)}"},
    ]
    return messages

def get_prompt(question:str) -> List[Dict]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You need to directly give the answer without any other text./no_think"},
        {"role": "user", "content": f"You need to respond only with the answer without any other text.\n\nQuestion: {question}"},
    ]
    return messages


def eval(config: Config):
    accuracy = 0
    print(f"Evaluating {config.model} on {config.filepath} with {0 if config.context_free else config.context_num} context(s)")
    if config.context_free:
        print("Using context-free mode")
    with open(config.filepath, "r") as f:
        data = json.load(f)
    if config.question_limit and config.question_limit > 0 and config.question_limit < len(data):
        print(f"Limiting to first {config.question_limit} questions")
        data = data[:config.question_limit]
    for item in tqdm(data, desc="Evaluating"):
        if config.context_free:
            messages = get_prompt(item["question"])
        else:
            messages = get_context_prompt(item["question"], item["contexts"][:config.context_num])
        ans = generate(messages, base_url=config.base_url)
        if exact_match(ans, item["answer"]):
            accuracy += 1
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

def eval_RAG(config: Config):
    accuracy = 0
    print(f"Evaluating {config.model} on {config.filepath} {'with reranker' if config.reranker else 'without reranker'} using Adaptive RAG")
    generator = Generator_API(base_url=config.base_url)
    retriever = FaissLocalRetriever(faiss_index_path=f"faiss_store")
    reranker = BgeReranker() if config.reranker else None
    dataset_name = config.filepath.split("/")[-1].split(".")[0].split("_")[0]
    eval_results = []
    base_path = f"data/adaptive/{dataset_name}"
    if config.reranker:
        base_path += f"/reranker"
    else:
        base_path += f"/vanilla"
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)
    with open(config.filepath, "r") as f:
        data = json.load(f)
    if config.question_limit and config.question_limit > 0 and config.question_limit < len(data):
        print(f"Limiting to first {config.question_limit} questions")
        data = data[:config.question_limit]
    try:
        for i in tqdm(range(len(data)), desc="Evaluating"):
            item = data[i]
            file_path = f"{base_path}/rag_history/question_{i}.txt"
            router = AdaptiveRouter(generator=generator)
            rag = AdaptiveRAG(generator=generator, router=router, retriever=retriever, reranker=reranker)
            ans = rag.generate(item["question"], top_k=20, max_step=10, file_path=file_path)
            em_value = exact_match(ans["answer"], item["answer"])
            if em_value:
                accuracy += 1
            eval_results.append({
                "index": i,
                "question": item["question"],
                "eval": "Correct" if em_value else "Incorrect",
                "Steps": ans["steps"],
                "LLM Response": ans["answer"],
                "Ground Truth": item["answer"],
            })
    except Exception as e:
        print(f"Error processing question {i}: {e}")
    with open(f"{base_path}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    with open(f"{base_path}/eval_specs.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy/len(data):.4f}\n")
        f.write(f"Model: {config.model}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Question Limit: {config.question_limit}\n")
        f.write(f"Reranker: {'Yes' if config.reranker else 'No'}\n")
        f.write(f"Base URL: {config.base_url}\n")
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

def eval_context(config: Config):
    accuracy = 0
    print(f"Evaluating Context Retrieval on {config.filepath} with {0 if config.context_free else config.context_num} context(s)")
    with open(config.filepath, "r") as f:
        data = json.load(f)
    if config.question_limit and config.question_limit > 0 and config.question_limit < len(data):
        print(f"Limiting to first {config.question_limit} questions")
        data = data[:config.question_limit]
    for i in range(len(data)):
        item = data[i]
        for context in item["contexts"][:config.context_num]:
            if exact_match(context, item["answer"]):
                accuracy += 1
                break
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

def display(config: Config):
    accuracy = 0
    print(f"Evaluating {config.model} on {config.filepath} with {0 if config.context_free else config.context_num} context(s)")
    if config.context_free:
        print("Using context-free mode")
    with open(config.filepath, "r") as f:
        data = json.load(f)
    if config.question_limit and config.question_limit > 0 and config.question_limit < len(data):
        print(f"Limiting to first {config.question_limit} questions")
        data = data[:config.question_limit]
    for i in range(len(data)):
        item = data[i]
        print(f"Question: {item['question']}")
        # print(f"Answer: {item['answer']}")
        if config.context_free:
            messages = get_prompt(item["question"])
        else:
            messages = get_context_prompt(item["question"], item["contexts"][:config.context_num])
        ans = generate(messages, base_url=config.base_url)
        print(f"LLM Response: {ans}")
        if exact_match(ans, item["answer"]):
            accuracy += 1
        else:
            print(f"Wrong Answer Expected: {"; ".join(item['answer'])}")
        print("-"*100)
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="llama3.1-8b-instant")
    # parser.add_argument("--filepath", type=str, default="data/triviaqa_val_1000_DPR.json", help="The filepath of the dataset")
    # parser.add_argument("--context_num", type=int, default=5, help="The number of context to use")
    # parser.add_argument("--context_free", action="store_true", help="Whether to use context-free mode")
    # args = parser.parse_args()
    # eval(args)
    
    # Example configurations for evaluation
    model = "Qwen-2.5-7B-Instruct"
    base_url = "http://localhost:8081"
    nq_configs = [
        Config(model=model, filepath="data/nq_val_1000_DPR.json", context_num=0, context_free=True, base_url=base_url),
        Config(model=model, filepath="data/nq_val_1000_DPR.json", context_num=1, context_free=False, base_url=base_url),
        Config(model=model, filepath="data/nq_val_1000_DPR.json", context_num=3, context_free=False, base_url=base_url),
        Config(model=model, filepath="data/nq_val_1000_DPR.json", context_num=5, context_free=False, base_url=base_url),
    ]
    triviaqa_configs = [
        Config(model=model, filepath="data/triviaqa_val_1000_DPR.json", context_num=0, context_free=True, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_1000_DPR.json", context_num=1, context_free=False, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_1000_DPR.json", context_num=3, context_free=False, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_1000_DPR.json", context_num=5, context_free=False, question_limit=1000, base_url=base_url),
    ]
    # for config in triviaqa_configs:
        # eval(config)
    
    # Evaluate context retrieval accuracy
    # eval_context(Config(model=model, filepath="data/nq_val_4289_DPR.json", context_num=1, context_free=False, base_url=base_url))
    # eval_context(Config(model=model, filepath="data/preprocessed/triviaqa_validation_1000_BM25_bgeReranker.json", context_num=20, context_free=False, base_url=base_url))
    
    # config with reranker
    test_triviaqa = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BM25_bgeReranker.json", context_num=1, context_free=False, question_limit=1000)
    test_nq_bm25 = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BM25_bgeReranker.json", context_num=1, context_free=False, question_limit=1000)
    test_nq_dpr = Config(model="Qwen2.5-7B-Instruct", filepath="data/nq_val_1000_DPR.json", context_num=1, context_free=False, question_limit=10)
    test_triviqa_bge_reranker = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BgeM3Faiss_bgeReranker.json", context_num=3, context_free=False, question_limit=1000)
    test_nq_bge_reranker = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BgeM3Faiss_bgeReranker.json", context_num=3, context_free=False, question_limit=1000)
    
    # config without reranker
    test_triviqa_bge = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=10)
    test_nq_bge = Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=1000)

    # display(test_nq_bm25)
    # display(test_triviqa_bge)
    # eval(test_triviqa_bge)
    # eval(test_nq_bge)
    # eval(test_triviqa_bge_reranker)
    # eval(test_nq_bge_reranker)

    # eval iterative
    question_limit = 1000
    iterative_configs = [
        Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=False),
        Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=False),
        Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=True),
        Config(model="Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=True),
    ]
    for idx, config in enumerate(iterative_configs):
        eval_RAG(config)