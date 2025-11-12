from typing import Callable, List, Dict, Optional
from .retriever import Retriever


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
