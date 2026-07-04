# gitrag

Ask questions about any public GitHub Python codebase in plain English. Point it at a repo — it clones, parses, and indexes the code with tree-sitter, then answers questions using multi-query hybrid retrieval and Groq's Llama 3.3 70B, with every answer citing the exact file and function it came from.

**Live demo:** https://git-rag.up.railway.app/

**API:** https://git-rag-api.up.railway.app/docs

**Demo walkthrough:** [Watch on Loom](https://www.loom.com/share/7cf2002e224a4ff0b3d6e6b49fc35dd4)

## What it does

1. Give it a GitHub URL → it clones the repo and parses every Python file
2. Each function and class becomes its own chunk, using tree-sitter to split on actual code boundaries instead of arbitrary character counts
3. Chunks are embedded (`all-MiniLM-L6-v2`) and stored in ChromaDB
4. Ask a question → it rewrites your query 3 ways using the LLM, runs hybrid retrieval (BM25 + vector search) for all 4 queries, deduplicates results by chunk ID, and expands retrieval coverage
5. A cross-encoder reranks candidates before the final context window is assembled
6. Conversation history (3-message sliding window) allows follow-up questions without re-explaining context
7. The model answers using only that context, citing source file + function name, and explicitly says when it doesn't know rather than guessing

## Why these design choices

**Tree-sitter over naive chunking.** Splitting by character count cuts functions in half mid-signature. A retrieval system can find the *right file* and still hand the LLM a mangled fragment — chunk boundaries matter as much as embedding quality. Tree-sitter parses the actual syntax tree, so every chunk is a complete, valid function or class.

**Hybrid search over pure vector search.** Vector search ranks on semantic similarity, which creates a specific failure mode: a query like "routing for path parameters" can pull chunks about frontend URL parsing instead of FastAPI's `Path()` parameter handling — same vocabulary, wrong meaning. BM25 matches on actual keywords (`Path(`, `path_params`) rather than embedding-space proximity. Combining both via Reciprocal Rank Fusion gets the benefit of each without needing to tune a blended score weight.

**Multi-query retrieval.** A single query embedding is one point in vector space. Relevant implementation details may be scattered across authentication middleware, JWT helpers, route handlers, and utility functions. Rewriting the question multiple ways and taking the union of retrieval results improves recall and surfaces chunks that any single query formulation would miss.

**CrossEncoder reranking.** The bi-encoder used for vector search embeds query and document independently — fast, but it loses interaction signals between them. A cross-encoder reads `(query, document)` as a single input and scores true relevance. Running it only on the shortlist from retrieval gives the best of both worlds: fast broad retrieval and accurate final ranking.

## Stack

`FastAPI` · `Streamlit` · `tree-sitter` · `sentence-transformers` · `ChromaDB` · `rank-bm25` · `Groq (Llama 3.3 70B)` · `Docker` · `Railway`

## Architecture

```text
User question
     │
     ▼
Multi-Query Rewriter (Groq)
Generates 3 reformulations of the question
     │
     ▼
Hybrid Retrieval × 4 queries (original + 3 rewrites)
  ├── BM25 keyword search (rank_bm25)
  └── Vector similarity search (sentence-transformers + ChromaDB)
  → Reciprocal Rank Fusion merges both ranked lists
     │
     ▼
Deduplicate by chunk ID (union of all retrieval passes)
     │
     ▼
CrossEncoder Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
Scores every (query, chunk) pair — reorders by true relevance
     │
     ▼
Context Assembly
Each chunk tagged with:
source file, chunk type, function name, line range
     │
     ▼
Generation (Groq — llama-3.3-70b-versatile)
Answers strictly from provided context
     │
     ▼
Response with answer + source citations
```

Two Railway services, one repo: `rag-api` (FastAPI backend) and `rag-frontend` (Streamlit UI), communicating over HTTP via an env-configured base URL.

## Bugs worth knowing about

**State silently vanishing mid-test.** Early on, `/ingest` would succeed, then the very next `/ask` would fail as if nothing had been indexed. Turned out `uvicorn --reload`'s file watcher was scoped to the whole project directory — including the cloned repo and ChromaDB folder it was writing into at runtime. Every `/ingest` call triggered a filesystem change, which triggered a reload, which wiped the in-memory app state between requests. Fixed by routing all runtime-generated data (clone target, Chroma DB) to the OS temp directory via a centralized `paths.py`, completely outside the watched directory. Also a good reminder that `--reload` is dev-only — production needs state that survives a process restart regardless.

**A relevant result silently missing the fusion pool.** Hybrid search wasn't finding `Path()` from `param_functions.py` for a path-parameter query, despite it being a strong BM25 match (rank #6). The cause: `candidate_pool=15` meant vector search's top-15 didn't include that chunk at all (it was rank #17 there). RRF can only fuse candidates that made it into the pool — no amount of fusion math helps a document that never entered. Widened to `candidate_pool=25`, which gives both retrieval methods enough room to surface different-but-relevant results before fusion chooses between them.

**Metadata key mismatch.** Retrieval succeeded, but context assembly silently failed because one component wrote `chunk_type` while another expected `type`. Unit tests passed because each component worked independently; the bug only appeared during end-to-end integration testing at the boundary between retrieval and generation.

## Running locally

```bash
pip install -r requirements.txt

# .env file with:
# GROQ_API_KEY=your_key_here

# backend
uvicorn main:app --reload

# frontend (separate terminal)
streamlit run app.py
```

Supports public GitHub repos containing Python code.

## Benchmarks (SQLModel repo, 667 chunks)

| Metric               | Value        |
| -------------------- | ------------ |
| Success rate         | 100% (10/10) |
| Avg latency          | ~3–5s (Groq) |
| Avg answer length    | 226 words    |
| Avg sources returned | 5.0          |
