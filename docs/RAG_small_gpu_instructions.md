# Running the RAG baseline on small GPU (RTX 3080 Mobile, ~8 GB VRAM)

This project includes a minimal Retrieval-Augmented Generation (RAG) baseline. By default
the demo uses deterministic fallback embeddings (no heavy model loaded) so it is safe
to run on small machines.

I added guarded support to attempt a small real embedding model when you explicitly enable
it. The code will automatically switch to a smaller sentence-transformers model when it
detects low VRAM on the GPU.

What I changed
- `rag_pipeline/embeddings.py`:
  - `EmbeddingModel` supports `use_fallback` (defaults to True). When `use_fallback=False` the
    loader will try to load `sentence-transformers` but will:
      - Detect GPU VRAM and, if < 10 GB, switch to a smaller model (`paraphrase-MiniLM-L3-v2`).
      - Fall back to deterministic embeddings if loading fails for any reason.
- `demo.py`:
  - Defaults to `use_fallback=True` (safe). Added a commented example demonstrating how to
    enable the real model / CUDA path. You can also pass `--enable-real-model` (see commands)
    to attempt a real-model run.

Commands to run (PowerShell on Windows)

1) Safe quick run (recommended):

```powershell
$env:PYTHONPATH = 'C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
C:/Python312/python.exe demo.py
```

2) Attempt a real-model GPU run (small-model auto-selection — may download model weights):

```powershell
# limit CPU thread usage to reduce startup overhead
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:TOKENIZERS_PARALLELISM = 'false';
$env:PYTHONPATH = 'C:\Users\axel0\OneDrive\Desktop\Deep Learning\DL Project\Applied_DeepLearning_Project'
C:/Python312/python.exe demo.py --enable-real-model --device cuda
```

Notes & safety
- The demo will detect GPU VRAM and automatically pick a smaller model if your VRAM is low.
- If you see warnings about downloads, the model files are being fetched from Hugging Face.
- If you prefer to avoid all downloads, keep running `demo.py` without `--enable-real-model`.

Files to run
- `demo.py` is the primary script. It ingests `data/sample_doc.txt`, indexes chunks, and runs
  a sample query using the configured embedding behavior.
- `tests/run_chunking_check.py` is a super-lightweight local check for the chunker (safe to run).

If you want, I can update `demo.py` to accept a `--small-model <model-name>` override or to
pre-download the small model to a local directory to avoid time-consuming downloads during runs.
