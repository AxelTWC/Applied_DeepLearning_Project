import argparse
import json
import os
import shutil
from typing import List, Dict, Optional
from rag_pipeline.generator import Generator_API, Generator
from rag_pipeline.router import AdaptiveRouter
from rag_pipeline.retriever import FaissLocalRetriever
from rag_pipeline.prompt import ADAPTIVE_GENERATOR_QUERY_PROMPT, ADAPTIVE_GENERATOR_SYSTEM_PROMPT
from utils.metrics import exact_match
from tqdm import tqdm
from typing import Union
from rag_pipeline.rag import AdaptiveRAG
from rag_pipeline.reranker import BgeReranker, Reranker
import time

class Config:
    def __init__(self, model: str, filepath: str, context_num: int, context_free: bool, question_limit: Optional[int] = None, generator: Union[str, Generator] = "http://localhost:8080", reranker: Optional[Reranker] = False, router_model: Optional[str] = "http://localhost:8080"):
        self.model = model
        self.filepath = filepath
        self.context_num = context_num
        self.context_free = context_free
        self.question_limit = question_limit
        self.reranker = reranker
        self.generator = Generator_API(base_url=generator) if isinstance(generator, str) else generator
        self.router_model = Generator_API(base_url=router_model) if isinstance(router_model, str) else router_model

def get_prompt(question:str) -> List[Dict]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You need to directly give the answer without any other text./no_think"},
        {"role": "user", "content": f"You need to respond only with the answer without any other text.\n\nQuestion: {question}"},
    ]
    return messages


def eval(config: Config):
    accuracy = 0
    print(f"Evaluating Standard RAG Baseline with {config.model} on {config.filepath} with {0 if config.context_free else config.context_num} context(s)")
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
            messages = [
                {"role": "system", "content": ADAPTIVE_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": ADAPTIVE_GENERATOR_QUERY_PROMPT.format(query=item["question"], references="\n".join(item["contexts"][:config.context_num]))}
            ]
        ans = config.generator.generate(messages)
        if exact_match(ans, item["answer"]):
            accuracy += 1
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

def eval_RAG(args: argparse.Namespace, config: Config):
    accuracy = 0
    print(f"Evaluating Adaptive RAG with {config.model} on {config.filepath} {'with reranker' if config.reranker else 'without reranker'}")
    generator = config.generator
    retriever = FaissLocalRetriever(faiss_index_path=f"faiss_store")
    reranker = config.reranker
    dataset_name = config.filepath.split("/")[-1].split(".")[0].split("_")[0]
    eval_results = []
    base_path = f"{args.output_path}/{dataset_name}"
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
            router = AdaptiveRouter(generator=config.router_model)
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
        f.write(f"Generator: {config.generator.api_url if isinstance(config.generator, Generator_API) else config.generator.model_name}\n")
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
            messages = [
                {"role": "system", "content": ADAPTIVE_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": ADAPTIVE_GENERATOR_QUERY_PROMPT.format(query=item["question"], references="\n".join(item["contexts"][:config.context_num]))}
            ]
        ans = config.generator.generate(messages)
        print(f"LLM Response: {ans}")
        if exact_match(ans, item["answer"]):
            accuracy += 1
        else:
            print(f"Wrong Answer Expected: {"; ".join(item['answer'])}")
        print("-"*100)
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="out/RAG_lora_adapter", help="The path to the weights")
    parser.add_argument("--no_adaptive", action="store_true", help="Whether to evaluate without Adaptive RAG")
    parser.add_argument("--output_path", type=str, default="data/adaptive-7B-LoRA", help="The path to save the results")
    args = parser.parse_args()
    
    # Example configurations for evaluation
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    router_model = Generator(model_name=base_model, weight_path=args.weights_path)
    generator = router_model

    # Evaluate BGE-M3 Standard RAG baseline
    standard_triviqa_bge = Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/triviaqa_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=1000, generator=generator, router_model=router_model)
    standard_nq_bge = Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/nq_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=1000, generator=generator, router_model=router_model)

    standard_triviqa_bm25 = Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/preprocessed/triviaqa_validation_1000_BM25.json", context_num=1, context_free=False, question_limit=1000, generator=generator, router_model=router_model)
    standard_nq_bm25 = Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/preprocessed/nq_validation_1000_BM25.json", context_num=1, context_free=False, question_limit=1000, generator=generator, router_model=router_model)


    question_limit = 1000
    iterative_configs = [
        Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/triviaqa_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=None, generator=generator, router_model=router_model),
        Config(model="Qwen/Qwen2.5-7B-Instruct", filepath="data/nq_validation_1000_BgeM3Faiss.json", context_num=1, context_free=False, question_limit=question_limit, reranker=None, generator=generator, router_model=router_model),
    ]

    # eval iterative
    if args.no_adaptive:
        print("Adaptive RAG Evaluation is disabled")
    else:
        print("Evaluating Standard RAG Baseline")
        for idx, config in enumerate(iterative_configs):
            start_time = time.time()
            eval_RAG(args, config)
            end_time = time.time()
            # convert time into hours, minutes, and seconds
            hours = int(end_time - start_time) // 3600
            minutes = int(end_time - start_time) % 3600 // 60
            seconds = int(end_time - start_time) % 60
            print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")
            
    eval(standard_triviqa_bm25)
    eval(standard_nq_bm25)