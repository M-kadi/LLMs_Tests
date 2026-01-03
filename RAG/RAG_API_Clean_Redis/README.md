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

## 📜 License

MIT License © 2026 Mohammed & Manaf
