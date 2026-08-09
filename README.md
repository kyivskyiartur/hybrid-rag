# LLM Zoomcamp — Hybrid Search RAG

A small RAG project built while completing Module 2 of the
[LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

The project extends the RAG and agentic search system from Module 1 by adding
vector search and combining it with keyword search using hybrid retrieval.

## What it does

The system searches through LLM Zoomcamp lesson materials and answers questions
about the course.

The retrieval pipeline combines:

- Keyword search with MiniSearch
- Vector search using `all-MiniLM-L6-v2`
- Reciprocal Rank Fusion (RRF) to combine both search results
- RAG for answering questions using retrieved context
- An agentic loop that allows the LLM to perform multiple searches before answering

## Architecture

```text
User question
     |
     v
Hybrid Search
     |
     +---- Keyword Search (MiniSearch)
     |
     +---- Vector Search (Embeddings)
     |
     v
Reciprocal Rank Fusion
     |
     v
Relevant lesson chunks
     |
     +----> RAG
     |
     +----> Agentic Search