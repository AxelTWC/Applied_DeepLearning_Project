"""Demo script for the simple RAG baseline using fixed-size chunking."""
from rag_pipeline import Retriever, EmbeddingModel, RAG
import argparse


def main(device: str = "auto", use_fallback: bool = False, model_name: str = None):
    # Create components
    if model_name is None:
        emb = EmbeddingModel(device=device, use_fallback=use_fallback, lazy=False)
    else:
        emb = EmbeddingModel(model_name=model_name, device=device, use_fallback=use_fallback, lazy=False)
    retriever = Retriever(embedding_model=emb, chunk_size=200, overlap=40)

    # Load a small document
    with open("data/sample_doc.txt", "r", encoding="utf-8") as f:
        text = f.read()

    doc = {"id": "sample1", "text": text, "metadata": {"title": "sample doc"}}
    print("[demo] adding documents to retriever (this will embed chunks)...")
    retriever.add_documents([doc])

    rag = RAG(retriever)

    query = "Which pangram is used as example text?"
    out = rag.generate(query, top_k=3)

    print("\nQuery:", out["query"])
    print("Answer:\n", out["answer"])
    print("Retrieved contexts:")
    for c in out["contexts"]:
        print(f"- {c['id']} score={c['score']:.3f} source={c['source_id']} chunk={c['chunk_index']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="device for embeddings: auto|cpu|cuda")
    p.add_argument("--fallback", action="store_true", help="use deterministic fallback embeddings (no heavy models)")
    p.add_argument("--enable-real-model", action="store_true", help="attempt to load a real sentence-transformers model (may download weights)")
    p.add_argument("--small-model", default=None, help="optional: small model name to prefer (e.g. paraphrase-MiniLM-L3-v2)")
    args = p.parse_args()

    # For safety this demo defaults to the deterministic fallback embeddings
    # so it can run on low-VRAM machines. To enable the real model and CUDA,
    # change `use_fallback` below to False and ensure you have an appropriate
    # GPU/VRAM available.
    use_fallback = True

    # If user specifically requests the real model, disable fallback
    if args.enable_real_model:
        use_fallback = False

    # If you want to attempt CUDA with a real model, here's an example (commented):
    # use_fallback = False
    # if args.device == 'cuda' and not use_fallback:
    #     import torch
    #     if not torch.cuda.is_available():
    #         raise SystemExit('CUDA not available')
    #     prop = torch.cuda.get_device_properties(0)
    #     total_gb = prop.total_memory / (1024 ** 3)
    #     print(f"Detected GPU: {prop.name}, VRAM={total_gb:.2f} GB")
    #     if total_gb < 8.0:
    #         raise SystemExit('VRAM < 8GB: aborting')

    main(device=args.device, use_fallback=use_fallback, model_name=args.small_model)
