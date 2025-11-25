rag/chunking: 


In Retrieval-Augmented Generation (RAG), chunking is a critical preprocessing step that directly affects:

- retrieval recall

- retrieval precision

- context window efficiency

- Downstream LLM output quality

Easier:

Imagine you have a huge storybook and the AI can only read small pages at a time.

This file:

- ✂️ Cuts the big story into small pieces (chunks)

- 🔁 Makes the pieces overlap slightly so the AI remembers context

- 😊 Tries NOT to cut sentences halfway

- 📦 Returns a list of these neat little chunks

Core:

Split a long document into smaller chunks with:

- bounded size (≤ chunk_size characters),

- optional overlap (overlap characters),

- sentence-aware boundary detection to avoid degrading semantic coherence.

This prepares text for embedding and vector retrieval while keeping retrieval units semantically meaningful.