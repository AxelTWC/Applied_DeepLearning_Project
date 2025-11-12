# Enhancing Retrieval-Augmented Generation with Adaptive Chunking 
Contributors: Tanish Upreti, Zihan Gong, Wenlong Zheng, Axel Tang

## An Applied Deep Learning Project
Tech Used: 
```ruby
Concepts:
  - RAG
  - LLM
  - Adaptive Chunking
Tech:
  - Python
  - LLaMA-2, LLama-3, Mixtral
  - FAISS, ColBERT
```

## A Core Summary 
Retrieval-Augmented Generation (RAG) - A framework that combines information retrieval (fetching relevant documents) with large language models (LLMs) by introducing adaptive chunking which is a smarter way of splitting text into pieces for retrieval.

Adaptive chunking makes RAG systems more intelligent by tailoring how text is divided, leading to better retrieval quality, stronger context understanding, and higher fidelity AI responses.

In this project , we will find ways to enhance this process leading to better accuracy for RAG-AC.

## Requirements of the Project
1. Implement a Retrieval-Augmented Generation pipeline using fixed-size chunking strategy as
baseline
2. Implement different adaptive chunking strategies to Retrieval-AugmentedGeneration
3. Use TriviaQA, NaturalQuestions to compare and analyze the impact of different chunking strategies
4. Deliver source code for implementation of different chunking strategies, and results
of each

## Setup
##-----##

## Baseline RAG implementation (added)

Added a small baseline Retrieval-Augmented Generation (RAG) pipeline that uses a fixed-size
chunking strategy to serve as a baseline for later adaptive chunking experiments. The baseline is
minimal and designed to be easy to run locally.

Files added:
- `rag_pipeline/` : package implementing chunking, embeddings (sentence-transformers or fallback),
  an in-memory NumPy vector store, a retriever and a RAG orchestrator.
- `data/sample_doc.txt` : a tiny sample document used by the demo.
- `demo.py` : example script that ingests the sample document and runs a sample query.
- `tests/run_chunking_check.py` : a lightweight script that verifies the chunker (safe to run).
- `docs/RAG_small_gpu_instructions.md` : instructions for attempting small-GPU runs safely.
- `requirements.txt` : dependency suggestions for the baseline.

Quickstart (PowerShell on Windows):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo.py
```

Notes:
- The embedding wrapper will use deterministic pseudo-embeddings by default (safe).
- The vector store uses NumPy-based cosine similarity. This keeps the baseline lightweight
  and avoids platform-specific wheel issues for Faiss.

Small GPU notes
- See `docs/RAG_small_gpu_instructions.md` for safe instructions on attempting a real-model
  run on GPUs with limited VRAM (for example, RTX 3080 mobile with ~8GB). The demo defaults
  to deterministic fallback embeddings to avoid heavy memory use.

