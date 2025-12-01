import sys 
import os
sys.path.append("..") 
from rag_pipeline.router import Router, AdaptiveRouter
from rag_pipeline.generator import Generator, Generator_API

class RouterTest:
    def test_router(self, router: Router, question: str):
        topics = router.route(question)
        assert isinstance(topics, list)
        assert len(topics) > 0
        print("AdaptiveRouter test passed.")
        return topics
        
if __name__ == "__main__":
    test = RouterTest()
    generator = Generator_API()
    router = AdaptiveRouter(generator=generator)
    topics = test.test_router(router, "who's the original singer of help me make it through the night?")
    print(topics)