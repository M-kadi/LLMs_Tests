# from app.api import app

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#     )

"""Root entry point.

Run in two ways:

1) Recommended (dev):
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

2) Python:
   python main.py

Environment variables:
  HOST (default 0.0.0.0)
  PORT (default 8000)
  RELOAD (default true)
"""

# To Run:
# - Start Docker Desktop
# - Start Ollama application
# - Start Qdrant locally: from command line, run:
#  D:\LLM\LLM_Tests\LLMs_Tests\RAG\RAG_API_Clean_Simple>
    # docker run -p 6333:6333 -p 6334:6334 ^
    # -v %cd%\qdrant_data:/qdrant/storage ^
    # qdrant/qdrant
# - Then run this script:
#     D:\LLM\LLM_Tests\LLMs_Tests\RAG\RAG_API_Clean_Simple>
    # python main.py
    # Open browser to (Swagger): http://localhost:8000/docs
    # Test endpoints
    # GET http://localhost:8000/settings
    # POST http://localhost:8000/settings
    # POST http://localhost:8000/settings/reset
    # POST http://localhost:8000/index/build
    # POST http://localhost:8000/query
# '''
# ------------------ Config ------------------
# Qdrant Dashboard URL :
# http://localhost:6333/dashboard#/collections
# Config File rag_settings.json : 
#    contains: LLM chat models + Embedding models, enable reranking, text group lines
# Disable Reranking : by default enabled
# save settings to file : rag_settings.json
# load settings from file on startup
# Enable reranking checkbox : true : will rerank by get the TOPK_RETRIEVE from Qdrant 
#   then send to LLM to rerank to TOPK_USE as final contexts
#   False : directly use TOPK_USE from Qdrant
# Use Text Group Lines : for TXT files, group N lines per chunk instead of single line chunks (Paragraphs)
# Support CSV + TXT files in the same folder for ingestion
# Use Qdrant for vector storage and retrieval
# Enable change Prompt and Prompt templates for Reranking and Answering from settings

from __future__ import annotations

import os
import uvicorn


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        port = 8000

    reload = _env_bool("RELOAD", True)

    # Use import string so reload works correctly
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
