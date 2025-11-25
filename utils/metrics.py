import re, string, json
from collections import Counter
from typing import List
import numpy as np

def normalize_text(s:str):
    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(response: str, answers: List) -> bool:
    """
    Verify if answer is in the context
    """
    response = normalize_text(response)
    answers = [normalize_text(answer) for answer in answers]
    for answer in answers:
        if answer in response:
            return True
    
    return False

def f1_score(prediction: str, ground_truth: str):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def F1(prediction: str, answers_list: List):
    return max(f1_score(prediction, ans) for ans in answers_list)

def Eval(file_path: str = ''):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cnt = 0
    f1_list = []
    for d in data:
        LLM_answer = d["LLM_answer"]
        reference = d["reference"]
        if exact_match(LLM_answer, reference):
            d["Eval"] = "Correct"
            cnt += 1
        else:
            d["Eval"] = "Wrong"
        d['f1_score'] = F1(LLM_answer, reference)
        f1_list.append(d['f1_score'])

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(file_path.replace('.json', '_accuracy').replace('/details', ''), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {cnt/len(data):.4f}\n")
        f.write(f"EM: {cnt}/{len(data)}\n")
        f.write(f"F1: {np.mean(f1_list):.4f}\n")