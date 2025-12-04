import os
import sys

__package__ = "rag_pipeline"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline.generator import Generator
from rag_pipeline.prompt import ADAPTIVE_ROUTER_SYSTEM_PROMPT, ADAPTIVE_ROUTER_INITIAL_PROMPT, ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT
from typing import List, Optional

class Router():
    
    def __init__(self):
        pass
    
    def route(self, question: str) -> str:
        pass
    
    def add_history(self, role: str, content: str):
        pass
    
    
class AdaptiveRouter(Router):
    
    def __init__(self, generator: Generator):
        self.generator = generator
        self.history = [{
            "role": "system",
            "content": ADAPTIVE_ROUTER_SYSTEM_PROMPT.strip()
        }]
        
    def route(self, question: str, references: Optional[str] = None) -> List[str]:
        query = ADAPTIVE_ROUTER_INITIAL_PROMPT.format(question=question).strip()
        if references:
            query = ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT.format(question=question, references=references).strip()
        self.history.append({
            "role": "user",
            "content": query
        })
        response = self.generator.generate(self.history)
        self.history.append({
            "role": "assistant",
            "content": response
        })
        return response
    
    def add_history(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content
        })