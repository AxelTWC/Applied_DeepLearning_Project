import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.retriever import FaissLocalRetriever, BM25Retriever, Retriever

class RetrieverTest:
    def test_retriever(self, retriever: Retriever, questions: list[str], question_ids: list[str], top_k: int = 5):
        batch_contexts_raw, batch_contexts = retriever.retrieve_batch(questions, question_ids, top_k=top_k)
        assert isinstance(batch_contexts_raw, dict)
        assert isinstance(batch_contexts, dict)
        for qid in question_ids:
            assert qid in batch_contexts
            contexts, scores = batch_contexts[qid]
            assert len(contexts) == top_k
            assert len(scores) == top_k
        print("BM25Retriever test passed.")
        return batch_contexts_raw, batch_contexts
    
if __name__ == "__main__":
    test = RetrieverTest()
    retriever = FaissLocalRetriever()
    questions = [
        "What was the Stephen King IT movie?",
        "When was the original Stephen King IT movie released?",
        "Who starred in the original Stephen King IT movie?",
        "What is the plot of the original Stephen King IT movie?"
    ]
    question_ids = [f"q{i}" for i in range(len(questions))]
    batch_contexts_raw, batch_contexts = test.test_retriever(retriever, questions, question_ids, top_k=5)
    for question, qid in zip(questions, question_ids):
        print(f"Question ID: {qid}, Question: {question}")
        contexts, scores = batch_contexts[qid]
        for context, score in zip(contexts, scores):
            print(f"Score: {score:.4f}, Context: {context}")
            break
        print("-"*100)