# rag/config.py
import logging
from dataclasses import dataclass
import os #for groq

@dataclass
class Config:
    data_folder: str = "data"
    chunk_size: int = 300
    overlap: int = 50

    embed_model_name: str = "all-MiniLM-L6-v2"
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    embedding_batch_size: int = 64
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    top_k_retriever: int = 40
    rerank_top_k: int = 5

    chroma_persist_dir: str = "chroma_db"

    # Ollama
   # ollama_url: str = "http://localhost:11434"
   # llm_model: str = "llama2:13b"
    #llm_temperature: float = 0.1
    #llm_max_tokens: int = 1024

     # Groq LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024


    log_level: int = logging.INFO


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )


  

   