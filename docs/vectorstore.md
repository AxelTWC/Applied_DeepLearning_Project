Each toy = a chunk of text
Each toy has:

a name (ID)

a description (metadata)

a special number shape (embedding vector)

This class helps you store all the toys (vectors) neatly
AND pick the ones most similar to a question.

____


InMemoryVectorStore is the toy box for your RAG project.

It:

Stores all chunk embeddings
Keeps their IDs and metadata
Compares your question to all stored vectors
Finds the closest matching chunks
Returns them in order (best match first)

This is how the AI finds “relevant context” before generating an answer.