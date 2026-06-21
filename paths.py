'''
centralized runtime data paths kept outside the project directory
so uvicorn's --reload file watcher doesnt see writes here and doesnt
trigger spurious restarts that wipe in-memory app state
'''

import os
import tempfile

RUNTIME_DIR = os.path.join(tempfile.gettempdir(), "github_rag_assistant")
os.makedirs(RUNTIME_DIR, exist_ok=True)

CLONE_DIR = os.path.join(RUNTIME_DIR, "repo")
CHROMA_DIR = os.path.join(RUNTIME_DIR, "chroma_db")

'''
Found a real bug: uvicorn --reload watches the whole project directory, 
including runtime-generated files like the cloned repo and ChromaDB folder. 
Calling /ingest triggers a file-system change that the reload watcher 
picks up, silently restarting the server and wiping in-memory app state. 
/ask then fails because the index "disappeared" — not because retrieval 
broke, but because the whole process restarted between calls. Fixed with 
--reload-exclude to scope the watcher to source files only. This is also 
why --reload is a dev-only flag — production uses a process manager 
(gunicorn/uvicorn workers) without file watching, and state would need to 
live outside process memory (Redis, a database) for any real deployment anyway.

'''