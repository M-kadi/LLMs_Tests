# RAG History GUI (Qdrant + Redis)

A **desktop GUI application** for experimenting with **conversation-aware RAG**, built on top of **Qdrant**, **Redis**, and multiple LLM providers.

---

## ✨ Key Features

- **RAG with Conversation History**
  - Redis-backed history
  - Last *N* turns injected into the prompt

- **Auto-Query Translation**
  - Non-English queries are automatically translated to English before retrieval
  - Enables multilingual interaction with English-only vector stores

- **Interactive GUI**
  - Tabs: Query, Results, Logs, History, Settings

- **History Management**
  - View conversation history
  - Export History / Logs / Results
  - Auto-export on app close (with confirmation)

- **Redis Integration**
  - History, logs, results stored in Redis
  - Ping Redis from UI
  - Control history size from settings

- **Safe Indexing**
  - Confirmation dialog before rebuilding index

- **Multi-Provider Support**
  - Ollama
  - Gemini
  - OpenAI

- **Flexible Ingestion**
  - CSV and TXT files
  - Paragraph-based chunking

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

### Run the RAG GUI (Development – Recommended)

### Run the RAG API (Python)
##### From the Path: `RAG_GUI_Clean_Redis\main.py`

```bat
python main.py
```

---

## 🖥️ GUI Highlights

- Embedding & main model selection
- App ID / Session ID / User ID support
- History-aware answering
- Qdrant collection visibility

---

## 🧱 Tech Stack

- Python (Tkinter)
- Qdrant
- Redis
- Ollama / Gemini / OpenAI

---

## 🎯 Purpose

- Rapid RAG experimentation
- Debugging RAG pipelines
- Visualizing history-aware RAG behavior

---

## 📜 License

MIT License © 2026 Mohammed & Manaf
