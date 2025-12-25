# models_config.py
# ------------------------------------------
# Public shared list of available chat models
# for Ollama-based RAG, SQL RAG, MCP server,
# and Flask UI.
# ------------------------------------------

CHAT_MODELS = [
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "phi3:mini",
    "qwen3:1.7b",
    "gemma3:270m",
    "gemma3:1b-it-qat",
    "qwen3:8b"
]

# Default chat model
DEFAULT_MODEL = CHAT_MODELS[3]  # "qwen3:1.7b"


EmbeddingModel = "all-minilm:latest"

EmbeddingModelList = [
    "all-minilm:latest",
    "mxbai-embed-large:latest"
]