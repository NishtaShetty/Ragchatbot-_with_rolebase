# rag/utils.py
from typing import List, Dict, Any
import logging

def add_parent_context(results: List[Dict[str, Any]], parents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enrich results with parent document context.
    Ensures parent metadata (especially role) is properly attached.
    
    Args:
        results: List of chunk results from retriever
        parents: Dictionary of parent documents keyed by parent_id
        
    Returns:
        Enriched results with parent information
    """
    enriched = []
    
    for r in results:
        parent_id = r.get("parent_id")
        
        if not parent_id:
            logging.warning(f"Result chunk {r.get('id')} missing parent_id")
            continue
        
        parent = parents.get(parent_id)
        
        if not parent:
            logging.warning(f"Parent document {parent_id} not found in parents dictionary")
            continue
        
        # Create enriched chunk with parent information
        r2 = dict(r)
        r2["parent_text"] = parent.get("text", "")
        r2["parent_role"] = parent.get("role", "unknown")
        r2["parent_source"] = parent.get("source", "")
        r2["parent_sections"] = parent.get("sections", [])
        
        # Log if parent role is missing
        if not parent.get("role"):
            logging.warning(
                f"Parent {parent_id} (source: {parent.get('source')}) has no role assigned"
            )
        
        enriched.append(r2)
    
    logging.debug(f"Enriched {len(enriched)}/{len(results)} chunks with parent context")
    
    return enriched


def verify_role_consistency(chunks: List[Dict[str, Any]], expected_role: str) -> bool:
    """
    Verify that all chunks belong to the expected role.
    
    Args:
        chunks: List of enriched chunks (must have parent_role)
        expected_role: The role to verify against (will be normalized)
        
    Returns:
        True if all chunks match expected role, False otherwise
    """
    expected_role_normalized = expected_role.strip().lower().replace(" ", "_")
    
    for chunk in chunks:
        parent_role = chunk.get("parent_role", "").strip().lower().replace(" ", "_")
        
        if parent_role != expected_role_normalized:
            logging.error(
                f"Role mismatch: chunk {chunk.get('id')} has parent_role '{parent_role}' "
                f"but expected '{expected_role_normalized}'"
            )
            return False
    
    return True


def get_role_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get distribution of roles in a set of chunks.
    Useful for debugging and monitoring.
    
    Args:
        chunks: List of chunks (can be enriched or not)
        
    Returns:
        Dictionary mapping role names to counts
    """
    role_counts = {}
    
    for chunk in chunks:
        # Try both chunk role and parent role
        role = chunk.get("parent_role") or chunk.get("role", "unknown")
        role = role.strip().lower().replace(" ", "_")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    return role_counts