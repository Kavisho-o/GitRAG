'''
METHODS:
POST /ingest - index a github repo
POST /ask - query the indexed repo
GET /health - health check
'''

import os
import threading
import traceback
import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

from clone_repo import clone_repo, find_python_files
from chunk_treesitter import chunk_repo_treesitter
from ingest import embed_and_store
from hybrid_search import HybridSearcher
from multi_query import multi_query_search
from agentic_rag import run_agentic_rag
from paths import CLONE_DIR
from fastapi.middleware.cors import CORSMiddleware
from model_cache import get_embed_model, get_cross_encoder

load_dotenv()
_api_ingest_lock = threading.Lock()  # ensure only one ingestion at a time

GROQ_MODEL = "llama-3.3-70b-versatile"
HISTORY_WINDOW = 3  # 3 (user, assistant) exchanges = last 6 messages

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

app_state = {"searcher": None, "repo_url": None, "sessions": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-warming models...")
    get_embed_model()
    if USE_RERANKER:
        get_cross_encoder()
    print("Models ready.")
    yield


app = FastAPI(
    title="GitRAG",
    description="A RAG system for querying any GitHub repo",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]
    session_id: str

@app.get("/health")
def health():
    return {"status": "ok", "repo_indexed": app_state["repo_url"]}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):

    if not req.repo_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only public GitHub repositories are supported (https://github.com/username/repo)")

    with _api_ingest_lock:
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
            app_state["searcher"] = HybridSearcher(use_reranker=USE_RERANKER)
            app_state["repo_url"] = req.repo_url
            app_state["sessions"] = {}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error embedding and storing chunks: {e}")

    return IngestResponse(
        status="indexed",
        repo_url=req.repo_url,
        files_processed=len(files),
        chunks_created=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):

    try:
        if app_state["searcher"] is None:
            raise HTTPException(status_code=400, detail="No repo has been indexed yet. Please POST to /ingest first.")

        sessions = app_state["sessions"]
        if req.session_id not in sessions:
            sessions[req.session_id] = []
        history = sessions[req.session_id]

        # Legacy retrieval + direct completion path (unused after switching to agentic RAG):
        # try:
        #     # results = app_state["searcher"].search(req.question, n_results=req.n_results)
        #     results = multi_query_search(req.question, app_state["searcher"], groq_client, n_results=req.n_results, n_variants=3, candidate_pool=15)
        # except Exception as e:
        #     traceback.print_exc()
        #     raise HTTPException(status_code=500, detail=f"Error querying the codebase: {e}")
        #
        # if not results["documents"][0]:
        #     raise HTTPException(status_code=404, detail="No relevant code chunks found for the question.")
        #
        # context_parts = []
        # for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        #     tag = (
        #         f"[{meta['source']}  ::  {meta['chunk_type']}  {meta['name']} "
        #         f"(lines {meta['start_line']}-{meta['end_line']})]"
        #     )
        #     context_parts.append(f"{tag}\n{doc}\n")
        # context = "\n\n---\n\n".join(context_parts)
        #
        # system_msg = {
        #     "role": "system",
        #     "content": (
        #         "You are a code assistant answering questions about a specific codebase. "
        #         "Use ONLY the code context provided in the user message to answer. "
        #         "If the context doesn't contain enough information, say so explicitly — "
        #         "do not guess or hallucinate. Reference specific function/class names."
        #     ),
        # }
        #
        # user_msg = {
        #     "role": "user",
        #     "content": (
        #         f"CODE CONTEXT:\n{context}\n\n"
        #         f"QUESTION: {req.question}"
        #     ),
        # }
        #
        # windowed_history = history[-(HISTORY_WINDOW * 2):]
        # messages = [system_msg] + windowed_history + [user_msg]
        #
        # max_retries = 3
        # answer = None
        # for attempt in range(max_retries):
        #     try:
        #         response = groq_client.chat.completions.create(
        #             model=GROQ_MODEL,
        #             messages=messages,
        #             temperature=0.1,
        #         )
        #         answer = response.choices[0].message.content
        #         break
        #     except Exception as e:
        #         if attempt < max_retries - 1:
        #             time.sleep(2 ** attempt)
        #         else:
        #             traceback.print_exc()
        #             raise HTTPException(status_code=502, detail=f"LLM unavailable after {max_retries} retries: {e}")
        #
        # history.append(user_msg)
        # history.append({"role": "assistant", "content": answer})
        #
        # sources = [
        #     f"{m['source']} :: {m['chunk_type']} {m['name']} (lines {m['start_line']}-{m['end_line']})"
        #     for m in results["metadatas"][0]
        # ]
        # contexts = results["documents"][0]
        #
        # return AskResponse(answer=answer, sources=sources, contexts=contexts, session_id=req.session_id)

        try:
            result = run_agentic_rag(
                question=req.question,
                searcher=app_state["searcher"],
                groq_client=groq_client,
                history=history,
                n_results=req.n_results,
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agentic RAG pipeline error: {e}")

        if not result["answer"]:
            raise HTTPException(status_code=500, detail="Pipeline completed but produced no answer.")

        history.append({
            "role": "user",
            "content": f"CODE CONTEXT: [retrieved context]\n\nQUESTION: {req.question}"
        })
        history.append({"role": "assistant", "content": result["answer"]})

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            contexts=result["contexts"],
            session_id=req.session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")