from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
import logging
import pickle
import os

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import Config
from rag.query_engine import QueryEngine
from rag.language_utils import (
    process_text,
    process_audio,
    translate_answer,
    check_models_status
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="Multilingual RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Config
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = Config(
    chroma_persist_dir=os.path.join(BASE_DIR, "chroma_db"),
    #ollama_url="http://localhost:11434",
    #llm_model="llama3:8b",
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    llm_model="llama-3.3-70b-versatile",
    llm_temperature=0.1,
    llm_max_tokens=1024
)

# -------------------------------------------------
# Load Vector Store (RETRIEVAL ONLY)
# -------------------------------------------------
def load_vector_store(config: Config):
    logger.info("Loading ChromaDB collection...")

    client = chromadb.PersistentClient(
    path=config.chroma_persist_dir
)

    collections = client.list_collections()
    names = [c.name for c in collections]

    if "hybrid_docs" not in names:
        raise RuntimeError(
            f"Collection 'hybrid_docs' not found.\n"
            f"Found: {names}\n"
            f"➡ Run: python ingest.py"
        )

    collection = client.get_collection("hybrid_docs")

    embed_model = SentenceTransformer(config.embed_model_name)

    metadata_path = os.path.join(
        config.chroma_persist_dir, "metadata.pkl"
    )

    if not os.path.exists(metadata_path):
        raise RuntimeError("metadata.pkl missing")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info("✓ Vector store loaded")

    return {
        "collection": collection,
        "embed_model": embed_model,
        "docs": metadata["docs"],
        "parents": metadata["parents"]
    }
# -------------------------------------------------
# Initialize RAG Engine at Startup
# -------------------------------------------------
logger.info("Initializing retrieval engine...")

models_status = check_models_status()
logger.info(f"Models status: {models_status}")

vector_store = load_vector_store(config)
engine = QueryEngine(vector_store, config)

logger.info("✓ RAG system ready")

# -------------------------------------------------
# Request / Response Models
# -------------------------------------------------
class ChatRequest(BaseModel):
    query: str | None = None
    language: str              # en, kn, hi, ta, te, ml
    input_type: str     # text | voice
    role: str             
    audio_base64: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list
    transcription: str | None = None
    original_language: str
    was_translated: bool

# -------------------------------------------------
# Health Endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": check_models_status()
    }

# -------------------------------------------------
# Chat Endpoint
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    try:
        logger.info(
            f"Incoming request | lang={req.language} | type={req.input_type}"
        )

        transcription = None
        was_translated = False

        # ---------------- TEXT INPUT ----------------
        if req.input_type == "text":
            if not req.query:
                raise HTTPException(
                    status_code=400,
                    detail="Query is required for text input"
                )

            result = process_text(req.query, req.language)
            query_for_retrieval = result["translated_text"]
            was_translated = result["was_translated"]

        # ---------------- VOICE INPUT ----------------
        elif req.input_type == "voice":
            if not req.audio_base64:
                raise HTTPException(
                    status_code=400,
                    detail="audio_base64 is required for voice input"
                )

            audio_bytes = base64.b64decode(req.audio_base64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

            if len(audio_np) < 8000:
                raise HTTPException(
                    status_code=400,
                    detail="Audio too short"
                )

            result = process_audio(audio_np, req.language)
            transcription = result["original_text"]
            query_for_retrieval = result["translated_text"]
            was_translated = req.language != "en"

        else:
            raise HTTPException(
                status_code=400,
                detail="input_type must be 'text' or 'voice'"
            )

        if not query_for_retrieval.strip():
            return ChatResponse(
                answer="I could not understand the query.",
                sources=[],
                transcription=transcription,
                original_language=req.language,
                was_translated=was_translated
            )

        # ---------------- RAG ----------------
        logger.info(f"RAG query: {query_for_retrieval[:80]}")
        rag_result = engine.query(query_for_retrieval, user_role=req.role)

        # ---------------- OUTPUT LANGUAGE ----------------
        final_answer = translate_answer(
            rag_result["answer"],
            req.language
        )

        return ChatResponse(
            answer=final_answer,
            sources=rag_result["sources"],
            transcription=transcription,
            original_language=req.language,
            was_translated=was_translated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("🚀 Server running at http://127.0.0.1:6001")
