# RAG History API (Qdrant + Redis)

A **production-ready RAG (Retrieval-Augmented Generation) API** built with **FastAPI**, **Qdrant**, and **Redis**, supporting **conversation history**, **background jobs**, and **multi-LLM providers**.

---

## ✨ Key Features

- **RAG with History**
  - Prompt = Context (Qdrant) + History (Redis) + Query
  - Uses last *N* conversation turns for follow-up questions

- **Auto-Query Translation**
  - Non-English queries (Arabic, Turkish, Chinese, etc.) are automatically translated to English
  - Ensures compatibility with English-only vector data in Qdrant
- **Redis Integration**
  - Conversation history (per app_id / user_id / session_id)
  - Logs, results, background job state
  - Configurable TTL and max history turns

- **Background Jobs**
  - Async index building and heavy tasks
  - Non-blocking API requests (RQ / Redis workers)

- **Multi-Provider LLM Support**
  - Ollama (local)
  - Gemini
  - OpenAI

- **Flexible Data Ingestion**
  - CSV and TXT files
  - Line or paragraph-based chunking

- **Config-Driven**
  - Central `rag_settings.json`
  - Models, reranking, prompts, chunking, providers

- **Swagger API**
  - Interactive API docs at `/docs`

---

## 🔌 Main Endpoints

- `GET /health`
- `GET /settings`
- `POST /settings`
- `POST /settings/reset`
- `POST /index/build`
- `POST /query`
- `GET /history`
- `GET /logs`
- `GET /results`
- `GET /jobs`

---

## 🧱 Tech Stack

- FastAPI
- Qdrant (Vector Database)
- Redis (History, Jobs, Logs, Results)
- Ollama / Gemini / OpenAI

---

## 🚀 Use Cases

- Conversational RAG APIs
- Enterprise knowledge bases
- Multi-user chat systems
- Backend for RAG GUIs

---

## ▶️ How to Run (Command Line)

### Start : Docker Desktop app and Ollama app
---
### Start Qdrant (Vector Database)

```bat
docker run -p 6333:6333 -p 6334:6334 ^
  -v %cd%\qdrant_data:/qdrant/storage ^
  qdrant/qdrant
```

- Qdrant Dashboard: http://localhost:6333/dashboard

---

### Start Redis

```bat
docker run -p 6379:6379 --name redis -d redis:7
```
---
### Start redisinsight

```bat
docker run -d ^
  --name redisinsight ^
  -p 8001:5540 ^
  redis/redisinsight
```
- redisinsight: http://localhost:5540/
- Add Redis Database: Host: `host.docker.internal` 

---
### Start Qdrant & Redis & redisinsight by docker `docker-compose.yml`

Start All
```bat
docker compose up -d
```

Stop All
```bat
docker compose down
```

---

### Run the RAG API (Development – Recommended)
##### From the Path: `RAG_API_Clean_Redis\app\main.py`

```bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
##### Or by Click F5 in VS Code `RAG_API_Clean_Redis\.vscode\launch.json`
---

### Run the RAG API (Python)
##### From the Path: `RAG_API_Clean_Redis\main.py`

```bat
python main.py
```
- Swagger UI: http://localhost:8000/docs

---

## 🌐 URLs

- Swagger UI: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- redisinsight: http://localhost:5540/

---

## 📜 License

MIT License © 2026 Mohammed & Manaf
