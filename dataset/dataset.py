import os
import sys
import json

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict
from rag_pipeline.prompt import ADAPTIVE_ROUTER_SYSTEM_PROMPT, ADAPTIVE_ROUTER_INITIAL_PROMPT

class RAGDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        # prompt = self._create_chat_prompt(sample['messages'])
        ground_truth = sample['ground_truth']
        question = sample['question']
        return {
            'messages': sample['messages'],
            'ground_truth': ground_truth,
            'question': question
        }
    