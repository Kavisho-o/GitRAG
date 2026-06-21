'''
METHODS:
POST /ingest - index a github repo
POST /ask - query the indexed repo
GET /health - health check
'''

import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

from clone_repo import clone_repo, find_python_files
from chunk_treesitter import chunk_repo_treesitter
from ingest import embed_and_store
from hybrid_search import HybridSearcher
from paths import CLONE_DIR
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# rn we keep global state for simplicity, but in a production
# system we would want to manage this more carefully by keying
# this per user session or per repo

app_state = {"searcher": None, "repo_url": None}
app = FastAPI(title="GitRAG", description="A RAG system for querying any GitHub repo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # to tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="The URL of the GitHub repository to ingest")

class IngestResponse(BaseModel):
    status: str
    repo_url: str
    files_processed: int
    chunks_created: int

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question about the ingested codebase")
    n_results: int = Field(5, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "repo_indexed": app_state["repo_url"]}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):

    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub repositories are supported (https://github.com/username/repo)")
    
    try:
        clone_repo(req.repo_url, CLONE_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clone repo. Check it's public and the URL is correct. ({e})")
    
    files = find_python_files(CLONE_DIR)
    if not files:
        raise HTTPException(status_code=400, detail="No Python files found in this repo. Only Python codebases are supported right now.")
    
    chunks = chunk_repo_treesitter(files)
    if not chunks:
        raise HTTPException(status_code=400, detail="No code chunks could be extracted from this repo. Check that the code is valid Python.")
    
    try:
        embed_and_store(chunks)
        app_state["searcher"] = HybridSearcher()
        app_state["repo_url"] = req.repo_url
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error embedding and storing chunks: {e}")
    
    return IngestResponse(
        status="indexed",
        repo_url=req.repo_url,
        files_processed=len(files),
        chunks_created=len(chunks)
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):

    if app_state["searcher"] is None:
        raise HTTPException(status_code=400, detail="No repo has been indexed yet. Please POST to /ingest first.")

    try:
        results = app_state["searcher"].search(req.question, n_results=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying the codebase: {e}")

    if not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No relevant code chunks found for the question.")
    
    context_parts = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        tag = f"[{meta['source']}  ::  {meta['chunk_type']}  {meta['name']} (lines {meta['start_line']}-{meta['end_line']})]"
        context_parts.append(f"{tag}\n{doc}\n")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a code assistant answering questions about a specific codebase.
Use ONLY the code context below to answer. If the context doesn't contain enough
information to answer confidently, say so explicitly - do not guess or hallucinate.

CODE CONTEXT:
{context}

QUESTION: {req.question}

Answer clearly. Reference specific function/class names from the context."""
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    sources = [
        f"{m['source']} :: {m['chunk_type']} {m['name']} (lines {m['start_line']}-{m['end_line']})"
        for m in results["metadatas"][0]
    ]

    return AskResponse(answer=answer, sources=sources)