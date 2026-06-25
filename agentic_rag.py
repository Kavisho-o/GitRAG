'''
agentic rag using langgraph
retrieve -> evaluate -> generate (if good) -> rewrite -> retrieve (if poor, max 2 retries)

state flows through every node as a typed dict
each node returns a partial update & langgraph merges it into the state for the next node
'''

from typing import TypedDict, Optional
import json
from groq import Groq
from hybrid_search import HybridSearcher
from multi_query import multi_query_search
from langgraph.graph import StateGraph, END

class RAGState(TypedDict):

    # inputs
    original_question: str
    n_results: int

    # working state
    current_question: str       # may be rewritten across retries
    retrieved_docs: list[str]
    retrieved_metas: list[dict]
    retrieved_ids: list[str]
    retrieval_quality: str      # "good" or "poor"
    retry_count: int

    # outputs
    answer: Optional[str]
    sources: list[str]
    contexts: list[str]

    # injected dependencies (not serialized, just passed through)
    searcher: object
    groq_client: object
    history: list[dict]


def retrieve(state: RAGState) -> dict:

    '''
    run multi-query retrieval on current_question.
    updates retrieved_docs, retrieved_metas, retrieved_ids.
    '''

    print(f"[agentic_rag] retrieve | question: {state['current_question'][:80]}")

    results = multi_query_search(
        question=state["current_question"],
        searcher=state["searcher"],
        groq_client=state["groq_client"],
        n_results=state["n_results"],
        n_variants=3,
        candidate_pool=25,
    )

    return {
        "retrieved_docs": results["documents"][0],
        "retrieved_metas": results["metadatas"][0],
        "retrieved_ids": results["ids"][0],
    }


def evaluate_retrieval(state: RAGState) -> dict:

    '''
    ask the LLM: are these chunks actually useful for answering the question?
    returns retrieval_quality = "good" or "poor"

    we judge on:
    - do the chunks contain relevant function/class names?
    - is there substantive code content (not just imports or empty stubs)?
    - would these chunks plausibly answer the question?

    LLM returns JSON: {"quality": "good"} or {"quality": "poor", "reason": "..."}
    '''

    print(f"[agentic_rag] evaluate_retrieval | retry_count: {state['retry_count']}")

    docs = state["retrieved_docs"]

    # hard fallback: if we got nothing back, it's poor
    if not docs:
        print("[agentic_rag] evaluate_retrieval | no docs returned → poor")
        return {"retrieval_quality": "poor"}

    # build a compact preview of retrieved chunks for the judge
    # we don't send full chunks, that would use too many tokens for a quality check
    chunk_previews = []
    for i, (doc, meta) in enumerate(zip(docs, state["retrieved_metas"])):
        preview = doc[:300].replace("\n", " ")
        chunk_previews.append(
            f"Chunk {i+1}: [{meta.get('name', '?')} in {meta.get('source', '?')}]\n{preview}"
        )
    chunks_text = "\n\n".join(chunk_previews)

    prompt = f"""You are evaluating whether retrieved code chunks are useful for answering a question.

Question: {state['original_question']}

Retrieved chunks (truncated previews):
{chunks_text}

Are these chunks likely to help answer the question above?
Consider: Do they contain relevant functions, classes, or logic? Is there substantive content?

Respond with ONLY valid JSON, no explanation:
{{"quality": "good"}} if the chunks are relevant and useful
{{"quality": "poor", "reason": "brief reason"}} if they are off-topic, empty, or unlikely to answer the question"""

    try:
        response = state["groq_client"].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        quality = parsed.get("quality", "good")

        if quality not in ("good", "poor"):
            quality = "good"  # safe default

        if quality == "poor":
            reason = parsed.get("reason", "no reason given")
            print(f"[agentic_rag] evaluate_retrieval | quality=poor | reason: {reason}")
        else:
            print(f"[agentic_rag] evaluate_retrieval | quality=good")

        return {"retrieval_quality": quality}

    except Exception as e:
        # if the judge itself fails, assume good and proceed
        # better to answer with whatever we have than to loop forever
        print(f"[agentic_rag] evaluate_retrieval | judge failed ({e}) → defaulting to good")
        return {"retrieval_quality": "good"}
    

def rewrite_query(state: RAGState) -> dict:

    '''
    LLM rewrites current_question into a better search query.
    increments retry_count.
    called only when evaluate_retrieval returns "poor".
    '''

    print(f"[agentic_rag] rewrite_query | attempt {state['retry_count'] + 1}")

    prompt = f"""A search for the following question returned poor results from a code retrieval system.
Rewrite it as a more specific, keyword-rich search query that might surface better code chunks.
Focus on concrete implementation terms: function names, class names, patterns, method calls.

Original question: {state['original_question']}
Previous query used: {state['current_question']}

Respond with ONLY the rewritten query string. No explanation, no quotes."""

    try:
        response = state["groq_client"].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
        print(f"[agentic_rag] rewrite_query | rewritten: {rewritten}")
    except Exception as e:
        print(f"[agentic_rag] rewrite_query | rewrite failed ({e}) → keeping original")
        rewritten = state["current_question"]

    return {
        "current_question": rewritten,
        "retry_count": state["retry_count"] + 1,
    } 


def generate(state: RAGState) -> dict:

    '''
    assemble context from retrieved chunks and call the LLM to answer.
    uses conversation history for follow-up awareness.
    '''

    print(f"[agentic_rag] generate | chunks: {len(state['retrieved_docs'])}")

    context_parts = []
    for doc, meta in zip(state["retrieved_docs"], state["retrieved_metas"]):
        tag = (
            f"[{meta['source']}  ::  {meta['chunk_type']}  {meta['name']} "
            f"(lines {meta['start_line']}-{meta['end_line']})]"
        )
        context_parts.append(f"{tag}\n{doc}\n")
    context = "\n\n---\n\n".join(context_parts)

    system_msg = {
        "role": "system",
        "content": (
            "You are a code assistant answering questions about a specific codebase. "
            "Use ONLY the code context provided in the user message to answer. "
            "If the context doesn't contain enough information, say so explicitly — "
            "do not guess or hallucinate. Reference specific function/class names."
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"CODE CONTEXT:\n{context}\n\nQUESTION: {state['original_question']}",
    }

    HISTORY_WINDOW = 3
    windowed_history = state["history"][-(HISTORY_WINDOW * 2):]
    messages = [system_msg] + windowed_history + [user_msg]

    import time
    max_retries = 3
    answer = None
    for attempt in range(max_retries):
        try:
            response = state["groq_client"].chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
            )
            answer = response.choices[0].message.content
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM unavailable after {max_retries} retries: {e}")

    sources = [
        f"{m['source']} :: {m['chunk_type']} {m['name']} (lines {m['start_line']}-{m['end_line']})"
        for m in state["retrieved_metas"]
    ]
    contexts = state["retrieved_docs"]

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }


MAX_RETRIES = 2

def should_retry(state: RAGState) -> str:

    '''
    called after evaluate_retrieval.
    returns "retry" → rewrite_query node
    returns "generate" → generate node

    caps retries at MAX_RETRIES to prevent infinite loops.
    if retry cap hit, proceed to generate with whatever we have.
    '''

    if state["retrieval_quality"] == "poor" and state["retry_count"] < MAX_RETRIES:
        print(f"[agentic_rag] should_retry | routing to retry ({state['retry_count']}/{MAX_RETRIES})")
        return "retry"

    if state["retrieval_quality"] == "poor" and state["retry_count"] >= MAX_RETRIES:
        print(f"[agentic_rag] should_retry | max retries hit → generating with best available")

    return "generate"


def build_rag_graph() -> object:
    '''
    builds and compiles the LangGraph state machine.
    call once at startup, reuse the compiled graph.

    graph structure:
        retrieve → evaluate_retrieval
        evaluate_retrieval → (conditional)
            "retry"    → rewrite_query → retrieve
            "generate" → generate → END
    '''

    graph = StateGraph(RAGState)

    # add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate_retrieval", evaluate_retrieval)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    # entry point
    graph.set_entry_point("retrieve")

    # fixed edges
    graph.add_edge("retrieve", "evaluate_retrieval")
    graph.add_edge("rewrite_query", "retrieve")   # retry loop
    graph.add_edge("generate", END)

    # wire evaluate_retrieval to the conditional
    graph.add_conditional_edges(
        "evaluate_retrieval",
        should_retry,
        {
            "retry": "rewrite_query",
            "generate": "generate",
        }
    )

    return graph.compile()


# compile once at module import
rag_graph = build_rag_graph()


def run_agentic_rag(
    question: str,
    searcher: HybridSearcher,
    groq_client: Groq,
    history: list[dict],
    n_results: int = 5,
) -> dict:
    
    '''
    entry point for main.py.
    returns {"answer": str, "sources": list[str], "contexts": list[str]}
    '''

    initial_state: RAGState = {
        "original_question": question,
        "current_question": question,
        "n_results": n_results,
        "retrieved_docs": [],
        "retrieved_metas": [],
        "retrieved_ids": [],
        "retrieval_quality": "good",
        "retry_count": 0,
        "answer": None,
        "sources": [],
        "contexts": [],
        "searcher": searcher,
        "groq_client": groq_client,
        "history": history,
    }

    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "contexts": final_state["contexts"],
    }
