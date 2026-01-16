# rag/data_processing.py
import os, logging
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict, Any

from .config import Config, configure_logging
from .text_utils import build_documents, SectionDetector
from .ocr_loader import extract_text_with_ocr

logger = logging.getLogger(__name__)

def normalize_role(role_name: str) -> str:
    """Normalize role name for consistent storage and retrieval"""
    normalized = role_name.strip().lower().replace(" ", "_")
    logger.debug(f"Normalized role '{role_name}' -> '{normalized}'")
    return normalized


def is_scanned_pdf(pdf_path: str) -> bool:
    """Check if PDF is scanned (image-based) or text-based"""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages[:2]:
            if page.extract_text():
                return False
        return True
    except Exception as e:
        logger.warning(f"Error checking if PDF is scanned: {e}")
        return True


def extract_text_from_pdf_with_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text and sections from text-based PDF"""
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace("\u00A0", " ").strip()
        if not text:
            continue

        sections = SectionDetector.extract_sections(text)
        pages.append({"page": i, "text": text, "sections": sections})

    return pages


def load_all_pdfs(folder_path: str, role: str) -> List[Dict[str, Any]]:
    """
    Load all PDFs from a folder and assign them to a role.
    
    Args:
        folder_path: Path to folder containing PDFs
        role: Role name (will be normalized)
        
    Returns:
        List of page dictionaries with role metadata
    """
    all_pages = []
    pdf_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".pdf")]
    
    logger.info(f"Loading {len(pdf_files)} PDFs for role '{role}' from {folder_path}")

    for fname in pdf_files:
        pdf_path = os.path.join(folder_path, fname)

        try:
            # Determine if scanned or text-based
            if is_scanned_pdf(pdf_path):
                logger.info(f"  Processing scanned PDF: {fname}")
                pdf_pages = extract_text_with_ocr(pdf_path)
            else:
                logger.info(f"  Processing text-based PDF: {fname}")
                pdf_pages = extract_text_from_pdf_with_sections(pdf_path)
                
        except Exception as e:
            logger.warning(f"Failed to process {pdf_path}: {e}")
            continue

        # Add role to each page
        for p in pdf_pages:
            all_pages.append({
                "source": fname,
                "page": int(p.get("page", 0)),
                "text": p.get("text") or "",
                "sections": p.get("sections", []),
                "role": role  # Parent role assigned here
            })

    logger.info(f"Loaded {len(all_pages)} pages for role '{role}'")
    return all_pages


def validate_role_structure(data_folder: str) -> Dict[str, Any]:
    """
    Validate the role-based folder structure.
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating role structure in {data_folder}")
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    role_folders = []
    validation_results = {
        "valid": True,
        "roles_found": [],
        "warnings": [],
        "errors": []
    }
    
    for item in os.listdir(data_folder):
        item_path = os.path.join(data_folder, item)
        
        if os.path.isdir(item_path):
            role_folders.append(item)
            pdf_count = len([f for f in os.listdir(item_path) if f.lower().endswith(".pdf")])
            
            if pdf_count == 0:
                validation_results["warnings"].append(
                    f"Role folder '{item}' contains no PDF files"
                )
            else:
                validation_results["roles_found"].append({
                    "name": item,
                    "normalized": normalize_role(item),
                    "pdf_count": pdf_count
                })
    
    if not role_folders:
        validation_results["valid"] = False
        validation_results["errors"].append("No role folders found in data directory")
    
    logger.info(f"Found {len(role_folders)} role folders: {role_folders}")
    
    return validation_results


def build_vector_store(config: Config):
    """
    Build vector store with role-based access control.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary containing collection, model, docs, and parents
    """
    configure_logging(config.log_level)
    logger.info("="*80)
    logger.info("Building vector store with role-based access control")
    logger.info("="*80)
    
    # Validate structure
    validation = validate_role_structure(config.data_folder)
    
    if not validation["valid"]:
        raise ValueError(f"Invalid data structure: {validation['errors']}")
    
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)
    
    # Log roles found
    logger.info(f"\nRoles detected ({len(validation['roles_found'])}):")
    for role_info in validation["roles_found"]:
        logger.info(
            f"  - {role_info['name']} (normalized: {role_info['normalized']}) "
            f"-> {role_info['pdf_count']} PDFs"
        )
    
    # Load all pages with roles
    all_pages = []
    
    for role_dir in os.listdir(config.data_folder):
        role_path = os.path.join(config.data_folder, role_dir)
        
        if not os.path.isdir(role_path):
            continue
        
        normalized_role = normalize_role(role_dir)
        pages = load_all_pdfs(role_path, normalized_role)
        all_pages.extend(pages)
    
    logger.info(f"\nTotal pages loaded: {len(all_pages)}")
    
    # Build documents and parents
    logger.info("\nChunking documents...")
    docs, parents = build_documents(
        all_pages,
        chunk_size=config.chunk_size,
        overlap=config.overlap
    )
    
    logger.info(f"Created {len(docs)} chunks from {len(parents)} parent documents")
    
    # Validate role assignment
    logger.info("\nValidating role assignments...")
    chunks_without_role = [d for d in docs if not d.get("role")]
    parents_without_role = [p for p in parents.values() if not p.get("role")]
    
    if chunks_without_role:
        logger.error(f"Found {len(chunks_without_role)} chunks without role!")
        for chunk in chunks_without_role[:5]:
            logger.error(f"  Chunk {chunk['id']}: {chunk.get('source', 'unknown')}")
    
    if parents_without_role:
        logger.error(f"Found {len(parents_without_role)} parents without role!")
    
    if not chunks_without_role and not parents_without_role:
        logger.info("✓ All chunks and parents have valid role assignments")
    
    # Count documents per role
    role_counts = {}
    for d in docs:
        role = d.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    logger.info("\nDocument distribution by role:")
    for role, count in sorted(role_counts.items()):
        logger.info(f"  {role}: {count} chunks")
    
    # Initialize embedding model
    logger.info(f"\nLoading embedding model: {config.embed_model_name}")
    embed_model = SentenceTransformer(config.embed_model_name)
    
    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at: {config.chroma_persist_dir}")
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    
    # Create or get collection
    try:
        client.delete_collection("hybrid_docs")
        logger.info("Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection("hybrid_docs")
    logger.info("Created new collection")
    
    # Prepare data for ChromaDB
    ids, texts, metadatas = [], [], []
    
    for d in docs:
        ids.append(d["id"])
        texts.append(d["text"])
        metadatas.append({
            "source": d.get("source", ""),
            "page": int(d.get("page", 0)),
            "parent_id": d.get("parent_id", ""),
            "chunk_index": int(d.get("chunk_index", 0)),
            "section": d.get("section") or "",
            "role": d.get("role") or "unknown"
        })
    
    # Generate embeddings
    logger.info(f"\nGenerating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=config.embedding_batch_size
    )
    
    # Add to collection
    logger.info("Adding to ChromaDB collection...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    
    logger.info("✓ Vector store built successfully!")
    logger.info("="*80)
    
    return {
        "collection": collection,
        "embed_model": embed_model,
        "docs": docs,
        "parents": parents,
        "role_stats": {
            "total_roles": len(role_counts),
            "roles": list(role_counts.keys()),
            "chunks_per_role": role_counts
        }
    }