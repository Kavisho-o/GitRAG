'''
clone repo
    |
tree-sitter chunk
    |
embed
    |
store in chromaDB

each step is a seperate function.

'''

import threading
import chromadb
from sentence_transformers import SentenceTransformer
from clone_repo import clone_repo as clone_repo_impl, find_python_files, REPO_URL
from chunk_treesitter import chunk_repo_treesitter, CodeChunk
from paths import CLONE_DIR, CHROMA_DIR
from model_cache import get_embed_model

# MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = CHROMA_DIR
COLLECTION_NAME = "codebase_treesitter"

_ingest_lock = threading.Lock() # ensure only one ingestion at a time

def clone_repo(repo_url: str, dest: str = CLONE_DIR) -> str:
    
    clone_repo_impl(repo_url, dest)
    return dest


def find_files(repo_path: str) -> list[str]:

    files = find_python_files(repo_path)
    print(f"Found {len(files)} Python files")
    return files


def chunk_files(file_paths: list[str]) -> list[CodeChunk]:

    chunks = chunk_repo_treesitter(file_paths)
    print(f"Chunked into {len(chunks)} code-aware chunks")
    return chunks


def embed_and_store(chunks: list[CodeChunk]) -> None:

    with _ingest_lock:
        print("Loading embedding model...")
        model = get_embed_model()
        client = chromadb.PersistentClient(path=CHROMA_PATH)  # persists to disk, survives restarts

        try: 
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(COLLECTION_NAME)

        print(f"Embedding {len(chunks)} chunks...")
        batch_size = 100
        for i in range(0,len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # prefix with file + function name which improves
            # retrieval quality as it provides more context
            texts_to_embed = [f"# {c.chunk_type} {c.name} in {c.source_file}\n{c.text}" for c in batch]
            embeddings = model.encode(texts_to_embed).tolist()

            collection.add(
                embeddings=embeddings,
                documents=[c.text for c in batch],  # store RAW text (no prefix)
                                                    # this gets shown to LLM later
                metadatas=[{
                    "source": c.source_file,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                } for c in batch],
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
            )
            print(f"  Stored {min(i+batch_size, len(chunks))}/{len(chunks)}")

        print("Ingestion complete.")

    # return model, collection


def run_full_ingestion_pipeline(repo_url: str) -> None:

    repo_path = clone_repo(repo_url)
    files = find_files(repo_path)
    chunks = chunk_files(files)
    embed_and_store(chunks)

    # return model, collection


if __name__ == "__main__":
    run_full_ingestion_pipeline(REPO_URL)