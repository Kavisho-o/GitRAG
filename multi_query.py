'''
generate N reformulations of user query via llm
run hybrid search for each reformulation
deduplicate and return unified result set
'''

# import os
import json
from groq import Groq

from llm_utils import _clean_json_response

GROQ_MODEL = "llama-3.1-8b-instant"  

def generate_query_variants(question: str, groq_client: Groq, n: int = 3) -> list[str]:
    '''
    ask llm to rewrite the question in n different ways
    returns list of n strings (not n+1)
    '''

    prompt = f"""You are a search query rewriter for a code retrieval system.

Given a user question about a codebase, generate {n} different search queries that capture 
the same intent from different angles. Each rewrite should use different vocabulary, 
focus on different aspects, or approach the concept from a different level of abstraction.

User question: {question}

Respond with ONLY a JSON array of {n} strings. No explanation, no preamble.
Example format: ["query one", "query two", "query three"]"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,   # higher temp = more diverse rewrites
        max_tokens=200,
    )
    raw = _clean_json_response(response.choices[0].message.content)

    try:
        variants = json.loads(raw)
        if isinstance(variants,list):
            return [str(v) for v in variants[:n]]
    except json.JSONDecodeError:
        pass


    # fallback: return original question if parsing fails
    print(f"[multi_query] JSON parse failed on: {raw[:100]}")
    return []


def multi_query_search(question: str, searcher, groq_client: Groq, n_results: int = 5, n_variants: int = 3, candidate_pool: int = 15) -> dict:

    '''
    generates n_variants rewrites of question
    runs hybrid search for each rewrite + original question
    deduplicates by chunk id (first occurence wins)
    return top n_results from unified chunks in hybrid search return format
    '''

    variants = generate_query_variants(question, groq_client, n=n_variants)
    all_queries = [question] + variants

    print(f"[multi_query] Queries: {all_queries}")

    seen_ids = set()
    merged_docs, merged_metas, merged_ids = [], [], []

    for q in all_queries:
        
        try:
            results = searcher.search(q, n_results=n_results, candidate_pool=candidate_pool, rerank=False)   # skip reranking for each variant, only rerank final top n_results
        except Exception as e:
            print(f"[multi_query] Error searching for query '{q}': {e}")
            continue

        docs, metas, ids = results["documents"][0], results["metadatas"][0], results["ids"][0]

        for doc, meta, chunk_id in zip(docs, metas, ids):
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_ids.append(chunk_id)

    # return the top n_results from the unified results
    # single rerank on merged pool

    if merged_docs:
        reranked_results = searcher.reranker.rerank(question, merged_docs, merged_metas, merged_ids, top_n=n_results)
        final_docs = reranked_results["documents"][0]
        final_metas = reranked_results["metadatas"][0]
        final_ids = reranked_results["ids"][0]
    else:
        final_docs = merged_docs[:n_results]
        final_metas = merged_metas[:n_results]
        final_ids = merged_ids[:n_results]

    print(f"[multi_query] Merged {len(merged_docs)} unique chunks, returning top {len(final_docs)}.")
    return {
        "documents": [final_docs],
        "metadatas": [final_metas],
        "ids": [final_ids]
    }