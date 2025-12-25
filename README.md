# LLMs_Tests
 
[**RAG Fast API : Qdrant Local + Ollama (Embeddings + Chat):**](https://github.com/M-kadi/LLMs_Tests/RAG/RAG_API)
##### To Run:
- Start Docker Desktop
- Start Ollama application
- Start Qdrant locally: from command line, run:
D:\LLM\LLMs_Tests\RAG\RAG_API>
> docker run -p 6333:6333 -p 6334:6334 ^
> -v %cd%\qdrant_data:/qdrant/storage ^
> qdrant/qdrant

Qdrant Dashboard URL :
http://localhost:6333/dashboard#/collections
- Then run this script:
D:\LLM\LLMs_Tests\RAG\RAG_API>
> uvicorn rag_api_qdrant:app --host 0.0.0.0 --port 8000 --reload

Open browser to (Swagger): http://localhost:8000/docs

##### Properties:
- Config File rag_settings.json :
  - contains: LLM chat models + Embedding models, enable reranking, text group lines
- Disable Reranking : by default enabled
- save settings to file : rag_settings.json
- load settings from file on startup
- Enable reranking : true : will rerank by get the TOPK_RETRIEVE from Qdrant
  - then send to LLM to rerank to TOPK_USE as final contexts
  - False : directly use TOPK_USE from Qdrant
- Use Text Group Lines : for TXT files, group N lines per chunk instead of single line chunks (Paragraphs)
- Support CSV + TXT files in the same folder for ingestion
- Use Qdrant for vector storage and retrieval
  
##### Test endpoints
GET http://localhost:8000/settings
POST http://localhost:8000/settings
POST http://localhost:8000/settings/reset
POST http://localhost:8000/index/build
POST http://localhost:8000/query




[**RAG GUI (CSV/TXT) : Qdrant Local + Ollama (Embeddings + Chat):**](https://github.com/M-kadi/LLMs_Tests/RAG/RAG_GUI)  

##### To Run:
- Start Docker Desktop
- Start Ollama application
- Start Qdrant locally: from command line, run:
D:\LLM\LLMs_Tests\RAG\RAG_GUI>
> docker run -p 6333:6333 -p 6334:6334 ^
> -v %cd%\qdrant_data:/qdrant/storage ^
> qdrant/qdrant

Qdrant Dashboard URL :
http://localhost:6333/dashboard#/collections

- Then run this script:
> PS D:\LLM\LLMs_Tests\RAG\RAG_GUI>python.exe rag_gui_qdrant.py
  

##### Properties:
- Config File rag_settings.json :
  - contains: LLM chat models + Embedding models, enable reranking, text group lines
- Disable Reranking : by default enabled
- save settings to file : rag_settings.json
- load settings from file on startup
- Enable reranking checkbox : true : will rerank by get the TOPK_RETRIEVE from Qdrant
  - then send to LLM to rerank to TOPK_USE as final contexts
  - False : directly use TOPK_USE from Qdrant
- Use Text Group Lines : for TXT files, group N lines per chunk instead of single line chunks (Paragraphs)
- Support CSV + TXT files in the same folder for ingestion
- Use Qdrant for vector storage and retrieval
