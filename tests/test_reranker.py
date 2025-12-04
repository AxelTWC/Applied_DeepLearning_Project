import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.reranker import BgeReranker, Reranker

class RerankerTest:
    def test_reranker(self, reranker: Reranker, questions: str, contexts: list[str], top_k: int = 5):
        reranked_contexts, reranked_scores = reranker.rerank(questions, contexts)
        print("BgeReranker test passed.")
        return reranked_contexts, reranked_scores
    
if __name__ == "__main__":
    test = RerankerTest()
    reranker = BgeReranker()
    questions = [
        "What was the Stephen King IT movie?",
        "When was the original Stephen King IT movie released?",
        "Who starred in the original Stephen King IT movie?",
        "What is the plot of the original Stephen King IT movie?"
    ]
    question_ids = [f"q{i}" for i in range(len(questions))]
    contexts = {
        "q0": [
            "The 2017 film It is based on Stephen King's 1986 novel of the same name.",
            "It is a supernatural horror film directed by Andy Muschietti.",
            "The story follows a group of children who are terrorized by a shape-shifting entity.",
            "The film stars Bill Skarsgård as Pennywise the Dancing Clown."
        ],
        "q1": [
            "The original Stephen King IT movie was released in 1990 as a television miniseries.",
            "It was directed by Tommy Lee Wallace.",
            "The miniseries starred Tim Curry as Pennywise the Dancing Clown.",
            "It was based on Stephen King's 1986 novel of the same name."
        ],
        "q2": [
            "The original Stephen King IT movie starred Tim Curry as Pennywise the Dancing Clown.",
            "The miniseries also featured actors such as Richard Thomas, John Ritter, and Annette O'Toole.",
            "It was directed by Tommy Lee Wallace and aired on ABC in 1990.",
            "The story follows a group of children who confront a shape-shifting entity."
        ],
        "q3": [
            "The plot of the original Stephen King IT movie revolves around a group of children in Derry, Maine.",
            "They are terrorized by a shape-shifting entity that often appears as Pennywise the Dancing Clown.",
            "The children band together to confront and defeat the entity.",
            "The story explores themes of friendship, fear, and childhood trauma."
        ]
    }
    for idx, qid in enumerate(question_ids):
        print(f"Question ID: {qid}")
        reranked_contexts, reranked_scores = test.test_reranker(reranker, questions[idx], contexts[qid], top_k=4)
        for context, score in zip(reranked_contexts, reranked_scores):
            print(f"Score: {score:.4f}, Context: {context}")
        print("-"*100)