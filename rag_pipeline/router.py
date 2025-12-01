from .generator import Generator
from .prompt import ITERATIVE_ROUTER_SYSTEM_PROMPT, ITERATIVE_ROUTER_INITIAL_PROMPT, ITERATIVE_ROUTER_SEQUENTIAL_PROMPT
from typing import List, Optional

class Router():
    
    def __init__(self):
        pass
    
    def route(self, question: str) -> str:
        pass
    
    def add_history(self, role: str, content: str):
        pass
    
    
class InterativeRouter(Router):
    
    def __init__(self, generator: Generator):
        self.generator = generator
        self.history = [{
            "role": "system",
            "content": ITERATIVE_ROUTER_SYSTEM_PROMPT.strip()
        }]
        
    def route(self, question: str, references: Optional[str] = None) -> List[str]:
        query = ITERATIVE_ROUTER_INITIAL_PROMPT.format(question=question).strip()
        if references:
            query = ITERATIVE_ROUTER_SEQUENTIAL_PROMPT.format(question=question, References=references).strip()
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