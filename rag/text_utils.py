# rag/text_utils.py
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------- SECTION DETECTOR ----------
class SectionDetector:
    """Detect and extract sections from document text"""
    
    HEADING_PATTERNS = [
        re.compile(r'^(\d+\.)+\s+([A-Z][^\n]+)', re.MULTILINE),
        re.compile(r'^Section\s+\d+', re.IGNORECASE),
        re.compile(r'^[A-Z][A-Z\s]{2,}$', re.MULTILINE),  # ALL CAPS headings
    ]

    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, Any]]:
        """
        Extract section headings from text.
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries with title and position
        """
        sections = []
        
        for pattern in SectionDetector.HEADING_PATTERNS:
            for match in pattern.finditer(text):
                sections.append({
                    "title": match.group(0).strip(),
                    "start_pos": match.start()
                })
        
        # Remove duplicates and sort by position
        sections = sorted(
            {s["start_pos"]: s for s in sections}.values(),
            key=lambda x: x["start_pos"]
        )
        
        return sections

    @staticmethod
    def assign_section_to_position(sections: List[Dict[str, Any]], pos: int) -> Optional[str]:
        """
        Find which section a text position belongs to.
        
        Args:
            sections: List of section dictionaries
            pos: Character position in text
            
        Returns:
            Section title or None
        """
        current = None
        for s in sections:
            if s["start_pos"] <= pos:
                current = s["title"]
            else:
                break
        return current


_SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')


def chunk_text_with_sections(
    text: str,
    sections: List[Dict[str, Any]],
    chunk_size: int = 300,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk text while preserving section information.
    
    Args:
        text: Text to chunk
        sections: List of section dictionaries
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        
    Returns:
        List of chunk dictionaries with text and section info
    """
    sentences = _SENTENCE_SPLIT_REGEX.split(text)
    chunks = []
    current = []
    size = 0
    current_pos = 0

    for s in sentences:
        w = len(s.split())
        
        # Check if we need to create a new chunk
        if size + w > chunk_size and current:
            chunk_text = " ".join(current)
            section = SectionDetector.assign_section_to_position(sections, current_pos)
            
            chunks.append({
                "text": chunk_text,
                "section": section,
                "start_pos": current_pos
            })
            
            # Keep overlap
            if overlap > 0:
                current = current[-overlap:]
                size = sum(len(x.split()) for x in current)
            else:
                current = []
                size = 0

        current.append(s)
        size += w
        current_pos += len(s) + 1  # +1 for space

    # Add final chunk
    if current:
        chunk_text = " ".join(current)
        section = SectionDetector.assign_section_to_position(sections, current_pos)
        
        chunks.append({
            "text": chunk_text,
            "section": section,
            "start_pos": current_pos
        })

    return chunks


def build_documents(
    pages: List[Dict[str, Any]],
    chunk_size: int = 300,
    overlap: int = 50
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build document chunks and parent documents from pages.
    Preserves role information at both chunk and parent level.
    
    Args:
        pages: List of page dictionaries with text, source, page, role, etc.
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        
    Returns:
        Tuple of (chunks, parents) where:
        - chunks: List of chunk dictionaries
        - parents: Dictionary of parent documents keyed by parent_id
    """
    docs = []
    parents = {}
    doc_id = 0
    
    logger.info(f"Building documents from {len(pages)} pages...")

    for p in pages:
        # Validate page has required fields
        if not p.get("source"):
            logger.warning(f"Page missing source: {p}")
            continue
        
        role = p.get("role")
        if not role:
            logger.warning(f"Page from {p['source']} missing role assignment")
            role = "unknown"
        
        # Create parent document
        parent_id = f"{p['source']}_page_{p.get('page', 0)}"
        
        parents[parent_id] = {
            "text": p.get("text", ""),
            "source": p["source"],
            "page": p.get("page", 0),
            "sections": p.get("sections", []),
            "role": role  # Role stored in parent
        }

        # Create chunks from this page
        text = p.get("text", "")
        if not text.strip():
            logger.debug(f"Skipping empty page: {parent_id}")
            continue
        
        chunks = chunk_text_with_sections(
            text,
            p.get("sections", []),
            chunk_size,
            overlap
        )
        
        for i, chunk_info in enumerate(chunks):
            docs.append({
                "id": f"doc_{doc_id}",
                "text": chunk_info["text"],
                "source": p["source"],
                "page": p.get("page", 0),
                "parent_id": parent_id,
                "chunk_index": i,
                "section": chunk_info.get("section"),
                "role": role  # Role propagated to chunk
            })
            doc_id += 1

    logger.info(f"Created {len(docs)} chunks from {len(parents)} parent documents")
    
    # Validation
    chunks_with_role = [d for d in docs if d.get("role")]
    parents_with_role = [p for p in parents.values() if p.get("role")]
    
    logger.info(f"Role assignment: {len(chunks_with_role)}/{len(docs)} chunks, "
                f"{len(parents_with_role)}/{len(parents)} parents")
    
    if len(chunks_with_role) != len(docs):
        logger.warning(f"{len(docs) - len(chunks_with_role)} chunks missing role!")
    
    if len(parents_with_role) != len(parents):
        logger.warning(f"{len(parents) - len(parents_with_role)} parents missing role!")

    return docs, parents


def get_chunk_statistics(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about chunks.
    
    Args:
        docs: List of chunk documents
        
    Returns:
        Dictionary with statistics
    """
    if not docs:
        return {"error": "No documents provided"}
    
    word_counts = [len(d["text"].split()) for d in docs]
    roles = [d.get("role", "unknown") for d in docs]
    role_counts = {}
    
    for role in roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    
    return {
        "total_chunks": len(docs),
        "avg_chunk_size": sum(word_counts) / len(word_counts),
        "min_chunk_size": min(word_counts),
        "max_chunk_size": max(word_counts),
        "chunks_with_sections": len([d for d in docs if d.get("section")]),
        "role_distribution": role_counts,
        "unique_roles": len(role_counts)
    }