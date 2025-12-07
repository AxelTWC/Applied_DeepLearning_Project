import torch
import os
import sys

__package__ = "rag_pipeline"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from FlagEmbedding import FlagReranker  # Lazy import only if needed
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
        # Lazy import FlagReranker only if needed
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(model_name, devices=self.device)
        except ImportError:
            # Fallback to base reranker if FlagEmbedding not available
            self.model = None
    
    def rerank(self, question: str, contexts: List[str]) -> Tuple[List[str], List[float]]:
        if self.model is None:
            # Fallback: return contexts as-is
            return contexts, [1.0 for _ in contexts]
        data = [[question, context] for context in contexts]
        scores = self.model.compute_score(data)
        ranked_contexts = [context for _, context in sorted(zip(scores, contexts), reverse=True)]
        return ranked_contexts, sorted(scores, reverse=True)