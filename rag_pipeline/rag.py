import os
import sys

__package__ = "rag_pipeline"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Callable, List, Dict, Optional
from rag_pipeline.retriever import Retriever, BM25Retriever
from rag_pipeline.generator import Generator
from rag_pipeline.reranker import Reranker
from rag_pipeline.router import Router
from rag_pipeline.prompt import ADAPTIVE_GENERATOR_SYSTEM_PROMPT, ADAPTIVE_GENERATOR_QUERY_PROMPT
from rag_pipeline.utils import break_condition, record_step
import os
def default_local_llm(prompt: str) -> str:
    """A very small default 'LLM' that echoes the context and query.

    This is intentionally simple and deterministic; replace with a real LLM
    callable (e.g., an OpenAI wrapper) for production use.
    """
    return "\n---\n".join(["CONTEXT:\n" + prompt, "\nAnswer: (local baseline) This answer is synthesized from the provided context."])

class RAG:
    """Simple Retrieval-Augmented Generation orchestration.

    Args:
        retriever: Retriever instance used to fetch context.
        llm: Optional callable(prompt: str) -> str. If None, uses default_local_llm.
    """

    def __init__(self, retriever: Retriever, llm: Optional[Callable[[str], str]] = None):
        self.retriever = retriever
        self.llm = llm or default_local_llm

    def generate(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve top_k contexts and call the LLM with a combined prompt.

        Returns a dict with the answer and the retrieved contexts.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        contexts = []
        for _id, score, metadata in results:
            # metadata contains short chunk_text preview
            ctx = {
                "id": _id,
                "score": score,
                "source_id": metadata.get("source_id"),
                "chunk_index": metadata.get("chunk_index"),
                "text_preview": metadata.get("chunk_text"),
            }
            contexts.append(ctx)

        # build prompt: include all retrieved chunks
        prompt_parts = [f"Query: {query}"]
        for c in contexts:
            prompt_parts.append(f"Source {c['source_id']} (chunk {c['chunk_index']}):\n{c['text_preview']}")
        prompt = "\n\n".join(prompt_parts)

        answer = self.llm(prompt)
        return {"query": query, "answer": answer, "contexts": contexts}

class AdaptiveRAG(RAG):
    def __init__(self, generator: Generator, router: Router, retriever: BM25Retriever, reranker:Optional[Reranker]):
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker
        self.router = router
    def generate(self, query: str, top_k: int =5, max_step: int = 10, file_path: Optional[str] = None) -> Dict:
        current_step = 0
        answer = ""
        references = ""
        all_contexts = []
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
                os.remove(file_path)
        while (current_step < max_step):
            current_step += 1
            response = self.router.route(query, references=references)
            if file_path:
                record_step(current_step, self.router.history, response, file_path)
            if break_condition(response):
                break
            subquestions = [subq.strip() for subq in response.split(",") if subq.strip()]
            if len(subquestions) == 0:
                subquestions = [query]
            references = ""
            question_ids = [f"{query}_subq_{i}" for i in range(len(subquestions))]
            raw_contexts, retrieval_contexts = self.retriever.retrieve_batch(subquestions, question_ids=question_ids, top_k=top_k)
            for subq in subquestions:
                contexts, scores = retrieval_contexts[f"{query}_subq_{subquestions.index(subq)}"]
                if self.reranker:
                    contexts, scores = self.reranker.rerank(subq, contexts)
                all_contexts.append({
                    "subquestion": subq,
                    "contexts": contexts,
                    "scores": scores
                })
                references += f"\nSub-question: {subq}\nRetrieved Contexts: {contexts[0]}\n"
        references = ""
        for contexts in all_contexts:
            references += f"\nSub-question: {contexts['subquestion']}\nRetrieved Contexts: {contexts['contexts'][0]}\n"
        generator_messages = [
            {"role": "system", "content": ADAPTIVE_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": ADAPTIVE_GENERATOR_QUERY_PROMPT.format(query=query, references=references)}
        ]
        answer = self.generator.generate(generator_messages)
        if file_path:
            record_step(-1, generator_messages, answer, file_path)
        return {"query": query, "answer": answer, "contexts": all_contexts, "steps": current_step}
    