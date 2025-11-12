# ForNow — Project Report (what happened, what I changed, how to run)

This document summarizes the full sequence of work done in this workspace while implementing
a small Retrieval-Augmented Generation (RAG) baseline (fixed-size chunking) and the
practical changes made to keep it runnable on a small GPU (RTX 3080 mobile with ~8GB VRAM).

1) Initial goal
- Implement a RAG baseline using a fixed-size chunking strategy as a starting point.

2) What I added (files)
- `rag_pipeline/` (package)
  - `chunking.py` — fixed-size chunker (character-based), improved for speed and safety.
  - `embeddings.py` — embedding wrapper that supports:
    - deterministic fallback embeddings (safe, default)
    - guarded loading of sentence-transformers models and optional device (cpu|cuda)
    - automatic small-model selection on low VRAM (e.g. picks `paraphrase-MiniLM-L3-v2` when VRAM < 10GB)
  - `vectorstore.py` — simple in-memory NumPy vector store with cosine search.
  - `retriever.py` — document ingestion (chunking -> embed -> index) and retrieval API.
  - `rag.py` — tiny orchestration that fetches top-k contexts and calls a small default local LLM function.
- `demo.py` — demo script that ingests `data/sample_doc.txt`, indexes it, and runs a sample query.
- `data/sample_doc.txt` — a small sample document used by the demo.
- `tests/run_chunking_check.py` — a super-lightweight script that validates the chunker only (no heavy ML libs).
- `docs/RAG_small_gpu_instructions.md` — short guidance for safe GPU runs on low VRAM machines.
- `requirements.txt` — dependency hint (numpy, scikit-learn, sentence-transformers, jupyter, pytest).

3) Problems encountered and why
- The initial approach shipped with `sentence-transformers` / `transformers` in `requirements.txt`.
  Importing these, or constructing a model, triggered heavy memory usage (PyTorch + model weights),
  long imports and sometimes downloads of tens of MBs of model weights.
- Running the demo with default behavior attempted to load a model and then use CUDA, which caused
  heavy RAM/VRAM usage and long stalls while downloading/loading weights.
- The original chunker also had a corner-case that could lead to no progress and an infinite loop
  for certain inputs — this caused the script to appear to hang at "multiple chunks with overlap".

4) How I fixed things (summary of code changes)
- Chunking stability and speed:
  - Rewrote `rag_pipeline/chunking.py` to use rfind-based sentence-end search (faster than large-regex scans).
  - Added a progress guard so the chunking loop always advances and cannot get stuck.
- Embedding safety & GPU friendliness:
  - `rag_pipeline/embeddings.py` now defaults to `use_fallback=True` to avoid loading heavy models by default.
  - Added lazy loading, device handling, and guarded real-model loading.
  - When `use_fallback=False`, the loader will attempt to detect GPU VRAM and automatically choose a
    smaller model when VRAM is limited (example: `paraphrase-MiniLM-L3-v2`). If anything goes wrong,
    we fall back to deterministic pseudo-embeddings.
- Demo and CLI
  - `demo.py` now exposes CLI flags:
    - `--device` (auto|cpu|cuda)
    - `--enable-real-model` (opt-in: attempts to load a real sentence-transformers model)
    - `--small-model` (override the small-model name)
  - By default the demo runs in safe fallback mode (no heavy model loads).
- Tests & lightweight checks
  - `tests/run_chunking_check.py` is a very small script that only exercises the chunker and prints progress.

5) What I ran here (and results)
- I executed the lightweight chunking check to verify the chunker finishes quickly:

  - Command (PowerShell):

    ```powershell
    $env:PYTHONPATH='C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
    C:/Python312/python.exe tests/run_chunking_check.py
    ```

  - Output: the three steps (empty, small, multiple chunks) completed quickly and the script printed status.

- I also tested a guarded real-model GPU run using a small-model auto-selection (because your GPU has ~8GB VRAM):

  - Command (PowerShell):

    ```powershell
    $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:TOKENIZERS_PARALLELISM='false'; \
      $env:PYTHONPATH='C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
    C:/Python312/python.exe demo.py --enable-real-model --device cuda
    ```

  - Behavior: code detected the GPU ("NVIDIA GeForce RTX 3080 Laptop GPU, VRAM=8.00 GB"), automatically
    switched to the smaller model (`paraphrase-MiniLM-L3-v2`), downloaded the model weights, moved them to CUDA
    and completed the demo. The demo printed retrieved contexts and an example generated answer from the
    small local baseline.

6) Files to run (quick list)
- `tests/run_chunking_check.py` — super-lightweight chunker-only check. (safe to run)
- `demo.py` — the main demo. Defaults to fallback embeddings; add `--enable-real-model` to try the guarded
  real-model path (may download weights) and `--device cuda` to attempt GPU.

7) Recommended commands (PowerShell)
- Safe quick run (recommended):

```powershell
$env:PYTHONPATH='C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
C:/Python312/python.exe demo.py
```

- Chunking check (lightweight):

```powershell
$env:PYTHONPATH='C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
C:/Python312/python.exe tests/run_chunking_check.py
```

- Attempt guarded real-model GPU run (auto-small-model selection if VRAM < 10GB):

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:TOKENIZERS_PARALLELISM='false'; $env:PYTHONPATH='C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
C:/Python312/python.exe demo.py --enable-real-model --device cuda
```

8) Notes, limitations and next steps
- Downloads: the first real-model run will download model weights (tens of MB). If you want to avoid downloads,
  keep using the fallback mode.
- Safety: I prioritized making the baseline safe by default for small-memory machines. If you want persistent
  real-model usage, consider pre-downloading the small model, or increasing the swap space / VRAM.
- Next steps I can implement for you:
  - Pre-download small models into a local cache directory to avoid interactive downloads.
  - Add a `--max-chunks` or `--truncate` option to the demo to limit chunking and indexing when memory is limited.
  - Add per-chunk progress logs during ingestion so you can see live indexing progress.

