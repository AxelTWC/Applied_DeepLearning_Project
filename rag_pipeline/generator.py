from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import List, Dict, Union

class Generator:
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = "You are a helpful assistant."

    def generate(self, messages: Union[str, List[Dict]]):
        if isinstance(messages, str):
            text = messages
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        
class Generator_API(Generator):
    """
    Generator that uses an API endpoint for generation
    """
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.api_url = f"{base_url}/v1/chat/completions"
        self.system_prompt = "You are a helpful assistant."
    
    def generate(self, input_data: Union[str, List[Dict]]):
        import requests
        import json
        headers = {"Content-Type": "application/json"}
        
        if isinstance(input_data, list):
            messages = input_data
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_data}
            ]

        data = {
            "messages": messages,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 40,
            "stream": False 
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        return response.json()["choices"][0]["message"]["content"]
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
    
if __name__ == "__main__":
    time_start = time.time()
    generator = Generator()
    question = "when did the nest 3rd generation come out"
    contexts = [
        "None",
    ]
    response = generator.generate(question, contexts)
    print(response)
    # print(f"Question: {question}\n\nReference Information: {"\n".join(contexts)}")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")