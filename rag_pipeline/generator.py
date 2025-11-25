from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class Generator:
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, question: str, contexts: list[str]):
        messages = [
            {"role": "system", "content": """
             You are a helpful assistant. You need to directly give the answer without any other text, and you will be provided with some reference information to help you answer the question. If The answer is not in the reference information, or the reference information is not enough to answer the question, you should use RAG to retrieve more context from konledge base, by <retriever>information you need</retriever> <context_num>number of context you need</context_num>.
             for example, if the question is "What was Grace Darling's father's job?", and the reference information is "None", your response should be <retriever>what was Grace Darling's father's job?</retriever> <context_num>3</context_num>.
             """},
            {"role": "user", "content": f"Question: {question}\n\nReference Information: {"\n".join(contexts)}"}
        ]
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