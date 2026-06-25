'''
retrieval
   |
top 25
   |
cross-encoder
   |
top 5
   |
answer with sources

cross-encoder reads (query, doc) as a single input.
ran on shortlist, never the full corpus.

'''

# from sentence_transformers import CrossEncoder
from model_cache import get_cross_encoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:

    def __init__(self):
        print("Loading CrossEncoder Reranker...")
        self.model = get_cross_encoder()
        print("CrossEncoder Reranker loaded.")


    def rerank(self, query: str, docs: list[str], metadatas: list[dict], ids: list[str], top_n: int = 5) -> dict:

        '''
        score each (query, doc) pair, sort descending, return top_n
        returns {doc_id: rank} for top_n

        '''

        if not docs:
            return {"documents": [docs], "metadatas": [metadatas], "ids": [ids]}
        
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)  # shape  = (len(docs),)

        ranked = sorted(zip(scores, docs, metadatas, ids), key=lambda x: x[0], reverse=True)[:top_n]
        scores_out, docs_out, metadatas_out, ids_out = zip(*ranked) if ranked else ([], [], [], [])

        return {"documents": [list(docs_out)], "metadatas": [list(metadatas_out)], "ids": [list(ids_out)]}
    

    