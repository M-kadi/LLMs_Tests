# models_config.py
# ------------------------------------------
# Public shared list of available chat models
# for Ollama-based RAG, SQL RAG, MCP server,
# and Flask UI.
# ------------------------------------------
# DEFAULT_MODEL_GEMINI = "gemini-2.5-flash"
# EMBEDDING_MODEL_GEMINI = "text-embedding-004"

# DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
# EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"

CHAT_MODELS_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

CHAT_MODELS_OPENAI = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
]

CHAT_MODELS_OLLAMA = [
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "phi3:mini",
    "qwen3:1.7b",
    "gemma3:270m",
    "gemma3:1b-it-qat",
    "qwen3:8b",
    # DEFAULT_MODEL_GEMINI, # Gemini 2.5 Flash need key, cloud access
    # DEFAULT_MODEL_OPENAI, # OpenAI gpt-4o-mini need key, cloud access
]

CHAT_MODELS = [
    # "deepseek-r1:1.5b",
    # "deepseek-r1:7b",
    # "phi3:mini",
    # "qwen3:1.7b",
    # "gemma3:270m",
    # "gemma3:1b-it-qat",
    # "qwen3:8b",
    # DEFAULT_MODEL_GEMINI, # Gemini 2.5 Flash need key, cloud access
    # DEFAULT_MODEL_OPENAI, # OpenAI gpt-4o-mini need key, cloud access
]

CHAT_MODELS.extend(CHAT_MODELS_OLLAMA) # Add local models
CHAT_MODELS.extend(CHAT_MODELS_GEMINI) # Add cloud models
CHAT_MODELS.extend(CHAT_MODELS_OPENAI) # Add cloud models

# Default chat model
DEFAULT_MODEL = CHAT_MODELS[3]  # "qwen3:1.7b"


EMBEDDING_MODEL = "all-minilm:latest"

EMBEDDING_MODELS_GEMINI = [
    "text-embedding-004",
    "gemini-embedding-001"
]

EMBEDDING_MODELS_OPENAI = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

EMBEDDING_MODELS_OLLAMA = [
    # "all-minilm:latest",
    # "mxbai-embed-large:latest",
    # EMBEDDING_MODEL_GEMINI,  # Gemini embedding model, need key, cloud access
    # EMBEDDING_MODEL_OPENAI,  # OpenAI embedding model, need key, cloud access
]

EMBEDDING_MODELS = [
    "all-minilm:latest",
    "mxbai-embed-large:latest",
    # EMBEDDING_MODEL_GEMINI,  # Gemini embedding model, need key, cloud access
    # EMBEDDING_MODEL_OPENAI,  # OpenAI embedding model, need key, cloud access
]

EMBEDDING_MODELS.extend(EMBEDDING_MODELS_OLLAMA) # Add local models
EMBEDDING_MODELS.extend(EMBEDDING_MODELS_GEMINI) # Add cloud models
EMBEDDING_MODELS.extend(EMBEDDING_MODELS_OPENAI) # Add cloud models