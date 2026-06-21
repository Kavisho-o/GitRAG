''' 
BM25 + vector hybrid search combined via score fusion
'''


import re
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
from paths import CHROMA_DIR

MODEL_NAME = "all-MiniLM-L6-v2"

def tokenize(text: str) -> list[str]:

    #simple whitespace + punctuation tokenizer
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class HybridSearcher:

    def __init__(self, collection_name: str = "codebase_treesitter", db_path: str = CHROMA_DIR):

        self.embed_model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(collection_name)

        # pull everything out from chroma to build BM25 index
        # bm25 needs full corpus in memory

        print("Loading corpus for BM25...")

        all_data = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_data["ids"]
        self.documents = all_data["documents"]
        self.metadatas = all_data["metadatas"]

        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"BM25 index built on {len(self.documents)} chunks.")

    
    def vector_search(self, query: str, n_results: int = 10) -> dict:

        '''
        returns {doc_id: rank} for top n_results
        rank 0 is best match
        '''

        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        return {doc_id: rank for rank,doc_id in enumerate(results["ids"][0])}
    

    def bm25_search(self, query: str, n_results: int = 10) -> dict: 

        '''
        returns {doc_id: rank} for top n_results
        rank 0 is best match
        '''

        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results]
        return {self.doc_ids[i]: rank for rank, i in enumerate(ranked_indices)}


    def reciprocal_rank_fusion(self, vector_ranks: dict, bm25_ranks: dict, k: int = 60) -> dict:
        '''
        combines 2 ranked lists without needing to normalize scores
        formula: score(doc) = sum(1/(k+rank(doc))) 
        for all lists it appears in, where k is a constant (generally 60)
        '''

        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        fused_scores = {}

        for doc_id in all_doc_ids: 

            score=0.0

            if doc_id in vector_ranks:
                score += 1/(k+vector_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += 1/(k+bm25_ranks[doc_id])
            
            fused_scores[doc_id] = score

        return sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)
    
    def search(self, query: str, n_results: int = 5, candidate_pool: int = 15) -> dict:

        '''
        get candidates from both methods, fuse, return top n_results.
        '''

        vector_ranks = self.vector_search(query, n_results=candidate_pool)
        bm25_ranks = self.bm25_search(query, n_results=candidate_pool)
        fused_order = self.reciprocal_rank_fusion(vector_ranks, bm25_ranks)[:n_results]

        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        documents, metadatas = [], []

        for doc_id in fused_order:
            idx = id_to_idx[doc_id]
            documents.append(self.documents[idx])
            metadatas.append(self.metadatas[idx])

        return {"documents": [documents], "metadatas": [metadatas], "ids": [fused_order]}    
if __name__ == "__main__":

    searcher = HybridSearcher()

    query = "how does routing work for path parameters"
    print(f"\nQUERY: {query}\n")

    print("--- Vector-only top 5 ---")
    v_results = searcher.vector_search(query, n_results=5)
    for doc_id in list(v_results.keys())[:5]:
        idx = searcher.doc_ids.index(doc_id)
        print(f"  {searcher.metadatas[idx]['name']} ({searcher.metadatas[idx]['source']})")

    print("\n--- BM25-only top 5 ---")
    b_results = searcher.bm25_search(query, n_results=5)
    for doc_id in list(b_results.keys())[:5]:
        idx = searcher.doc_ids.index(doc_id)
        print(f"  {searcher.metadatas[idx]['name']} ({searcher.metadatas[idx]['source']})")

    print("\n--- Hybrid (RRF fused) top 5 ---")
    h_results = searcher.search(query, n_results=5)
    for meta in h_results["metadatas"][0]:
        print(f"  {meta['name']} ({meta['source']})")