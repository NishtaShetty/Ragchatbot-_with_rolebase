from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import pickle
from typing import List

from rag.config import Config
from rag.data_processing import build_vector_store

# -----------------------------
# App Setup
# -----------------------------
app = FastAPI(title="RAG Ingestion Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# -----------------------------
# Ingestion Endpoint (ROLE-BASED)
# -----------------------------
@app.post("/ingest")
async def ingest_document(
    role: str = Form(...),                      # 👈 NEW
    files: List[UploadFile] = File(...)
):
    """
    Upload documents for a specific role
    """

    role = role.lower().strip().replace(" ", "_")

    if role not in {"field_team", "sales", "hr"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid role. Use field_team, sales, or hr"
        )

    role_dir = os.path.join(DATA_DIR, role)
    os.makedirs(role_dir, exist_ok=True)

    saved_files = []

    # Save files under role folder
    for uploaded_file in files:
        file_path = os.path.join(role_dir, uploaded_file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

        saved_files.append(uploaded_file.filename)

    # Build vector store
    config = Config(
        data_folder=DATA_DIR,
        chroma_persist_dir=CHROMA_DIR
    )

    result = build_vector_store(config)

    # Persist metadata
    with open(
        os.path.join(CHROMA_DIR, "metadata.pkl"),
        "wb"
    ) as f:
        pickle.dump(
            {
                "docs": result["docs"],
                "parents": result["parents"]
            },
            f
        )

    return {
        "status": "success",
        "role": role,
        "message": f"Ingested {len(saved_files)} document(s)",
        "files": saved_files,
        "total_chunks": len(result["docs"])
    }

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
