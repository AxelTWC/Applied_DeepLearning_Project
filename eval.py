import argparse
import requests
import json
from typing import List, Dict, Optional
from utils.metrics import exact_match
from tqdm import tqdm


class Config:
    def __init__(self, model: str, filepath: str, context_num: int, context_free: bool, question_limit: Optional[int] = None, base_url: str = "http://localhost:8080"):
        self.model = model
        self.filepath = filepath
        self.context_num = context_num
        self.context_free = context_free
        self.question_limit = question_limit
        self.base_url = base_url

def generate(messages,
             base_url: str = "http://localhost:8080",
             temperature=0.7,
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


def eval(args: Config):
    accuracy = 0
    print(f"Evaluating {args.model} on {args.filepath} with {0 if args.context_free else args.context_num} context(s)")
    if args.context_free:
        print("Using context-free mode")
    with open(args.filepath, "r") as f:
        data = json.load(f)
    if args.question_limit:
        print(f"Limiting to first {args.question_limit} questions")
        data = data[:args.question_limit]
    for item in tqdm(data, desc="Evaluating"):
        if args.context_free:
            messages = get_prompt(item["question"])
        else:
            messages = get_context_prompt(item["question"], item["contexts"][:args.context_num])
        ans = generate(messages, base_url=args.base_url)
        if exact_match(ans, item["answer"]):
            accuracy += 1
    print(f"Accuracy: {accuracy/len(data):.4f}")
    return accuracy/len(data)

def eval_context(config: Config):
    accuracy = 0
    print(f"Evaluating Context Retrieval on {config.filepath} with {0 if config.context_free else config.context_num} context(s)")
    with open(config.filepath, "r") as f:
        data = json.load(f)
    if config.question_limit:
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
    if config.question_limit:
        print(f"Limiting to first {config.question_limit} questions")
        data = data[:config.question_limit]
    for i in range(len(data)):
        item = data[i]
        print(f"Question: {item['question']}")
        print(f"Answer: {item['answer']}")
        if config.context_free:
            messages = get_prompt(item["question"])
        else:
            messages = get_context_prompt(item["question"], item["contexts"][:config.context_num])
        ans = generate(messages, base_url=config.base_url)
        print(f"LLM Response: {ans}")
        if exact_match(ans, item["answer"]):
            accuracy += 1
        else:
            print(f"Wrong Answer: Expected {"; ".join(item['answer'])}")
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
        Config(model=model, filepath="data/triviaqa_val_17944_DPR.json", context_num=0, context_free=True, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_17944_DPR.json", context_num=1, context_free=False, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_17944_DPR.json", context_num=3, context_free=False, question_limit=1000, base_url=base_url),
        Config(model=model, filepath="data/triviaqa_val_17944_DPR.json", context_num=5, context_free=False, question_limit=1000, base_url=base_url),
    ]
    # for config in triviaqa_configs:
    #     eval(config)
    
    # Evaluate context retrieval accuracy
    eval_context(Config(model=model, filepath="data/nq_val_4289_DPR.json", context_num=5, context_free=False, base_url=base_url))
    
    # Display detailed results for manual inspection
    # test = Config(model="Qwen3-8b", filepath="data/triviaqa_val_1000_DPR.json", context_num=3, context_free=True, question_limit=10)
    # display(test)