from typing import List, Dict
import os

def break_condition(response: str) -> bool:
    """Decide whether to continue based on the router response."""
    decision = response.strip().lower()
    if "terminate" in decision or not decision:
        return False
    else:
        # Default to False if unclear
        return True

def print_history(history: List[Dict]):
    for message in history:
        role = message['role']
        content = message['content']
        print(f"\n{role.upper()}\n{content}")
        
def record_step(step: int, model_history: List[Dict], model_response: str, file_path: str):
    with open(file_path, 'a') as f:
        if step >= 0:
            f.write(f"\n{">"*10}  STEP {step}  {"<"*10}\n")
        else:
            f.write(f"\n{">"*10}  FINAL GENERATION  {"<"*10}\n")
        f.write(f"\n----------  MODEL HISTORY. ----------")
        for message in model_history:
            f.write(f"\n{message['role'].upper()}\n {message['content']}\n")
        f.write("\n----------  MODEL RESPONSE. ----------")
        f.write(f"\n{model_response}\n")
        f.write(f"\n{"-"*100}\n")