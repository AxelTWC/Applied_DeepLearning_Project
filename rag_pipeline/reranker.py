
import torch
from FlagEmbedding import FlagReranker
from typing import Tuple, List
import transformers
transformers.logging.set_verbosity_error()

class Reranker:
    name: str = "base_reranker"
    def __init__(self):
        pass
    def rerank(self, question: str, contexts: List[str]) -> Tuple[List[str], List[float]]:
        # Dummy reranker that returns contexts as is
        return contexts, [1.0 for _ in contexts]
    

class BgeReranker(Reranker):
    name: str = "bgeReranker"
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name = model_name
        self.model = FlagReranker(model_name, devices=self.device)
    
    def rerank(self, question: str, contexts: List[str]) -> Tuple[List[str], List[float]]:
        data = [[question, context] for context in contexts]
        scores = self.model.compute_score(data)
        ranked_contexts = [context for _, context in sorted(zip(scores, contexts), reverse=True)]
        return ranked_contexts, sorted(scores, reverse=True)