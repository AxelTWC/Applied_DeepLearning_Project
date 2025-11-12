import os
import sys
import time

# make repo root importable without setting PYTHONPATH externally
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_pipeline.chunking import split_text_fixed_size
from rag_pipeline.embeddings import EmbeddingModel
import os
import sys

# Super-lightweight chunking check: only import the chunker and run tiny tests.
# This avoids loading any heavy ML libraries or models.

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_pipeline.chunking import split_text_fixed_size


def run_checks():
    print("[run_chunking_check] Starting lightweight checks (no ML libs will be loaded)")

    print("[1/3] empty text")
    assert split_text_fixed_size("", chunk_size=100) == []
    print("  ok")

    print("[2/3] small text")
    t = "Hello world. This is a test."
    chunks = split_text_fixed_size(t, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    print(f"  ok (chunks={len(chunks)})")

    print("[3/3] multiple chunks with overlap")
    t2 = "".join([f"Sentence {i}. " for i in range(30)])
    chunks2 = split_text_fixed_size(t2, chunk_size=60, overlap=10)
    assert len(chunks2) >= 2
    print(f"  ok (chunks={len(chunks2)})")

    print("[run_chunking_check] All lightweight checks passed.")


if __name__ == '__main__':
    run_checks()
