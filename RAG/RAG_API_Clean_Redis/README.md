# rag_api_simple

Clean + simple FastAPI REST API for your RAG system (Qdrant + Providers).

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger:
- http://localhost:8000/docs

## Settings
- Settings are stored in `rag_settings.json` (created automatically if missing).
- You can choose different providers for chat and embeddings:
  - main_provider: ollama | openai | gemini
  - embedding_provider: ollama | openai | gemini

Keys:
- OpenAI: OPENAI_API_KEY
- Gemini: GEMINI_API_KEY (or GOOGLE_API_KEY)

Optional:
- env_path in settings can point to a .env file (dotenv format).


## Background Jobs (non-blocking)
- POST /index/build_async  -> returns a job
- POST /query_async        -> returns a job
- GET  /jobs              -> list jobs
- GET  /jobs/{job_id}      -> job status + result when done

Notes:
- When a query job finishes, its payload is also stored in /results.


## Run

### Option A (recommended)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option B

```bash
python main.py
```


-----
## Run:
  - python RAG_API_Clean_Redis\main.py

## Debug 
  - (by : RAG_API_Clean_Redis\.vscode\launch.json):
  - RAG_API_Clean_Redis/app/main.py