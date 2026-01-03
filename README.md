# LLMs_Tests

A collection of **clean, production-ready RAG (Retrieval-Augmented Generation) experiments** built with **Qdrant**, **Redis**, and multiple LLM providers (Ollama, Gemini, OpenAI).
This repository focuses on **RAG with history**, **clean architecture**, and **real-world usability** (API + GUI).

---

## 🔗 RAG History API — Qdrant + Redis
[👉 **RAG_API_Clean_Redis**](https://github.com/M-kadi/LLMs_Tests/tree/main/RAG/RAG_API_Clean_Redis)

A **FastAPI-based RAG API** that supports **conversation history**, **background jobs**, and **Redis-backed state**, designed for **production usage**.

### Key Features
- **RAG with History**
  - Prompt = **Context (Qdrant) + History (Redis) + Query**
- **Auto-Query Translation**
  - Non-English queries (Arabic, Turkish, Chinese, etc.) are automatically translated to English
  - Translation is performed before retrieval to match English-only Qdrant data
- **Redis Integration**
  - Conversation history (app_id / user_id / session_id)
  - Logs, results, background job state
  - Configurable TTL & max turns
- **Background Jobs**
  - Async index build & heavy operations
- **Multi-Provider Support**
  - Ollama (local)
  - Gemini
  - OpenAI
- **Flexible Ingestion**
  - CSV + TXT
  - Paragraph-based chunking
- **Config-Driven**
  - `rag_settings.json` (models, rerank, chunking, prompts)
- **Swagger API**
  - Fully documented `/docs`

### Main Endpoints
- `/health`
- `/settings`
- `/index/build`
- `/query`
- `/history`
- `/logs`
- `/results`
- `/jobs`

---

## 🖥️ RAG History GUI — Qdrant + Redis
[👉 **RAG_GUI_Clean_Redis**](https://github.com/M-kadi/LLMs_Tests/tree/main/RAG/RAG_GUI_Clean_Redis)

A **desktop GUI** for experimenting with **conversation-aware RAG**, built on top of the same clean architecture as the API.

### Key Features
- **Conversation History**
  - Redis-backed
  - Last N turns injected into prompt
- **Auto-Query Translation**
  - Non-English queries (Arabic, Turkish, Chinese, etc.) are automatically translated to English
  - Translation is performed before retrieval to match English-only Qdrant data
- **Rich GUI**
  - Query / Results / Logs / History / Settings tabs
- **History Management**
  - View & export History, Logs, Results
  - Auto-export on close (with confirmation)
- **Redis Controls**
  - Ping Redis
  - Enable/disable history
  - Control history turns
- **Safe Indexing**
  - Confirmation before rebuilding index
- **Multi-Provider Support**
  - Ollama
  - Gemini
  - OpenAI
---

## 🧠 Why This Repo?
- True **RAG with memory**
- Clean architecture (API / Core / Storage / GUI)
- Redis used correctly (history, jobs, logs, results)
- Production-minded design
- Easy to extend (multimodal RAG coming)
- Will use vLLM (coming)

---

## 📜 License
MIT License © 2026 Mohammed & Manaf
