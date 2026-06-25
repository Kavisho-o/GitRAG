'''
single model instances
loaded once and shared across ingest and retrieval 
prevents double loading and memory bloat
'''

from sentence_transformers import SentenceTransformer, CrossEncoder

_embed_model = None
_cross_encoder = None

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading CrossEncoder Reranker...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        print("CrossEncoder Reranker loaded.")
    return _cross_encoder