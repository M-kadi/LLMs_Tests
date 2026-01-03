# RAG GUI (Clean)

Entry point:
- `python main.py`
or
- `python -m main`

This folder keeps the original behavior but splits code into:
- `rag_gui_app.py` (Tkinter GUI)
- `rag_engine_qdrant.py` (RAG engine: Qdrant + embeddings + chat)
- `rag_settings.py` (defaults, settings persistence, .env loading)
- `.env` / `.env.example` (environment variables)

Run with:
    python main.py


Redis support:
- Start Redis: docker run -d -p 6379:6379 redis:7
- GUI will store logs/results/history in Redis.
- History tab shows last N turns (default 5). Export button writes JSON (history+logs+results).

Requirements:
- pip install redis


UI:
- History tab: Refresh + Export History + Export Logs + Export Results + Export ALL
- On close: prompts to export ALL (single JSON) before exit.


Exports:
- Exports are written automatically under ./exports/ (project folder) as 3 JSON files: history_*.json, logs_*.json, results_*.json
- On close: prompts to export automatically to ./exports/ (no dialog).
