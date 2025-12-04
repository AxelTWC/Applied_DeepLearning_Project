import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.rag import RAG, AdaptiveRAG
from rag_pipeline.retriever import BM25Retriever, FaissLocalRetriever
from rag_pipeline.reranker import BgeReranker
from rag_pipeline.router import AdaptiveRouter
from rag_pipeline.generator import Generator, Generator_API

class TestRAG:
    def test_adaptive_rag(self, query: str, top_k: int =20):
        generator = Generator_API()
        router = AdaptiveRouter(Generator_API())
        retriever = FaissLocalRetriever()
        reranker = None
        rag = AdaptiveRAG(
            generator=generator,
            router=router,
            retriever=retriever,
            reranker=reranker
        )
        if os.path.exists("../data/rag_history/iterative_rag_test.txt"):
            os.remove("../data/rag_history/iterative_rag_test.txt")
        result = rag.generate(query, top_k=top_k, max_step=10, file_path="../data/rag_history/iterative_rag_test.txt")
        assert "query" in result
        assert "answer" in result
        assert "contexts" in result
        print("AdaptiveRAG test passed.")
        return result
    
if __name__ == "__main__":
    test = TestRAG()
    query = "when was the original stephen king it movie made?"
    result = test.test_adaptive_rag(query, top_k=20)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")