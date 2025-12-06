import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_pipeline.router import Router, AdaptiveRouter
from rag_pipeline.generator import Generator, Generator_API
from rag_pipeline.utils import extract_subtopics

class RouterTest:
    def test_router(self, router: Router, question: str):
        topics = router.route(question)
        subtopics = extract_subtopics(topics)
        assert isinstance(subtopics, list)
        assert len(subtopics) > 0
        print("AdaptiveRouter test passed.")
        return topics, subtopics
        
if __name__ == "__main__":
    test = RouterTest()
    generator = Generator_API()
    router = AdaptiveRouter(generator=generator)
    topics, subtopics = test.test_router(router, "who's the original singer of help me make it through the night?")
    print(topics)
    print(subtopics)