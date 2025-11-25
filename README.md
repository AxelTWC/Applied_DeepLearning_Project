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

## Evaluation Result (Accuracy)

Source: DPR

| Benchmark         | Model                 | vanilla | RAG (Top 1)    |RAG (Top 3)     |RAG (Top 5)     |
| :---------------: | :-------------------: | :-----: | :-----: | :-----: | :-----: |
| TriviaQA_VAL_500  | Llama-3.1-8B-Instruct | 74.2%   | 66.4%       | 70.0%       | 68.4%       |
| TriviaQA_VAL_1000 | Llama-3.1-8B-Instruct | 72.1%   | 61.4%       | 68.0%       | 68.1%       |
| TriviaQA_VAL_2000 | Llama-3.1-8B-Instruct | 72.7%   | 66.5%       | 68.0%       | 67.7%       |
| TriviaQA_VAL_1000 | Qwen-3-8B | 59.2% | 63.5% | 66.9% | 69.5% |
| TriviaQA_VAL_1000 | Qwen-2.5-7B-Instruct |  54.1%  |    57.3%    |     62%     |      D      |
|    NQ_VAL_1000    | Llama-3.1-8B-Instruct |  32.4%  |    39.6%    |    43.9%    |    44.7%    |
| NQ_VAL_1000 | Qwen-3-8B |  25.2%  |    41.7%    |    46.6%    |    50.1%    |
| NQ_VAL_1000 | Qwen-2.5-7B-Instruct  |  21.8%  |    37.9%    |    44.0%    | 44.9% |



Context Confidence

|  Benchmark   |   Method    | Score  |
| :----------: | :---------: | :----: |
| TriviaQA_VAL | DPR (Top 5) | 68.66% |
|    NQ_VAL    | DPR (Top 5) | 66.24% |







Notes:

- The embedding wrapper will use deterministic pseudo-embeddings by default (safe).
- The vector store uses NumPy-based cosine similarity. This keeps the baseline lightweight
  and avoids platform-specific wheel issues for Faiss.

Small GPU notes
- See `docs/RAG_small_gpu_instructions.md` for safe instructions on attempting a real-model
  run on GPUs with limited VRAM (for example, RTX 3080 mobile with ~8GB). The demo defaults
  to deterministic fallback embeddings to avoid heavy memory use.

---
## Benchmark:
<img width="1990" height="556" alt="image" src="https://github.com/user-attachments/assets/dbb93bd8-749a-48e2-9c1c-7adce262fb39" />

