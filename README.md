# gitrag

Ask questions about any public GitHub Python codebase in plain English. Point it at a repo, it clones, parses, and indexes the code with tree-sitter, then answers questions using hybrid retrieval (BM25 + vector search) and Groq's Llama 3.3 70B — with every answer citing the exact file and function it came from.

**Live demo:** https://gitrag-production.up.railway.app/
**API:** https://secure-freedom-production-57bc.up.railway.app/health

## What it does

1. Give it a GitHub URL → it clones the repo and parses every Python file
2. Each function and class becomes its own chunk, using tree-sitter to split on actual code boundaries instead of arbitrary character counts
3. Chunks are embedded (`all-MiniLM-L6-v2`) and stored in ChromaDB
4. Ask a question → it runs BM25 keyword search and vector similarity search in parallel, fuses the two ranked lists with Reciprocal Rank Fusion, and feeds the top results to the LLM as context
5. The model answers using only that context, citing source file + function name, and explicitly says when it doesn't know rather than guessing

## Why tree-sitter, why hybrid search

Naive chunking (split every N characters) cuts functions in half mid-signature. A retrieval system can find the *right file* and still hand the LLM a mangled fragment — chunk boundaries matter as much as embedding quality. Tree-sitter parses the actual syntax tree, so every chunk is a complete, valid function or class.

Pure vector search also has a specific failure mode: it ranks on semantic similarity, so a query like "routing for path parameters" can pull chunks about frontend URL paths instead of FastAPI's `Path()` parameter handling — same vocabulary, wrong meaning. BM25 catches what vector search misses by matching on actual keywords (`Path(`, `path_params`) rather than embedding-space proximity. Combining both via RRF gets the benefits of each without needing to tune a single blended score.

## Stack

`FastAPI` · `Streamlit` · `tree-sitter` · `sentence-transformers` · `ChromaDB` · `rank-bm25` · `Groq (Llama 3.3 70B)` · `Docker` · `Railway`

## Architecture

1. **Clone** — `clone_repo.py` pulls the target GitHub repo locally
2. **Chunk** — `chunk_treesitter.py` parses each file with tree-sitter and splits it into complete function/class chunks
3. **Embed + store** — `ingest.py` embeds each chunk with `all-MiniLM-L6-v2` and stores it in ChromaDB
4. **Retrieve** — `hybrid_search.py` runs BM25 and vector search in parallel, then fuses the ranked lists with Reciprocal Rank Fusion
5. **Generate** — top fused chunks are passed to Groq's Llama 3.3 70B as context
6. **Answer** — response returned with the exact source file and function it was grounded in

Two Railway services, one repo: `rag-api` (FastAPI backend) and `rag-frontend` (Streamlit UI), communicating over HTTP via an env-configured base URL.

Two services on Railway, one repo: `rag-api` (FastAPI backend) and `rag-frontend` (Streamlit UI), talking to each other over HTTP via an env-configured base URL.

## Two bugs worth mentioning

**State silently vanishing mid-test.** Early on, `/ingest` would succeed, then the very next `/ask` would fail as if nothing had been indexed. Turned out `uvicorn --reload`'s file watcher was scoped to the whole project directory — including the cloned repo and ChromaDB folder it was writing into at runtime. Every `/ingest` call triggered a filesystem change, which triggered a reload, which wiped the in-memory app state between requests. Fixed by routing all runtime-generated data (clone target, Chroma DB) to the OS temp directory via a centralized `paths.py`, completely outside the watched directory. Also a good reminder that `--reload` is dev-only — production needs state that survives a process restart regardless.

**A relevant result silently missing the fusion pool.** Hybrid search wasn't finding `Path()` from `param_functions.py` for a path-parameter query, despite it being a strong BM25 match (rank #6). The cause: RRF only fuses candidates that are *in* both ranked lists, and `candidate_pool=15` meant vector search's top-15 didn't include that chunk at all, in fact it ranked #17 there. No amount of fusion math helps a document that never made it into the pool. Widened `candidate_pool` to 25, which gives both retrieval methods enough room to surface different-but-relevant results before fusion has to choose between them. Distance/rank thresholds being too tight is an easy way to quietly lose good context, and you won't see it unless you check what each method finds *before* fusion.

## Running locally

```bash
pip install -r requirements.txt

# backend
uvicorn main:app --reload

# frontend (separate terminal)
streamlit run app.py
```

Needs a `.env` file with `GROQ_API_KEY=your_key_here`. Currently supports public GitHub repos containing Python code.