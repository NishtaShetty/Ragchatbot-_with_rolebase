# rag/query_engine.py
import logging
from typing import Dict, Any

from sentence_transformers import CrossEncoder
from .config import Config
from .retriever import HybridRetriever
#from .ollama import OllamaGenerator
from .groq_llm import GroqGenerator
from .prompt import build_rag_prompt
from .utils import add_parent_context

class QueryEngine:
    def __init__(self, vector_store: Dict[str, Any], config: Config):
        self.config = config
        self.parents = vector_store["parents"]

        self.retriever = HybridRetriever(
            docs=vector_store["docs"],
            embed_model=vector_store["embed_model"],
            collection=vector_store["collection"],
            config=config
        
        )

        self.reranker = CrossEncoder(config.cross_encoder_name)
        #self.llm = OllamaGenerator(config)
        self.llm = GroqGenerator(config)
        logging.info("QueryEngine initialized with parent role verification")

    def _normalize_role(self, role: str) -> str:
        """Normalize role name for consistent comparison"""
        return role.strip().lower().replace(" ", "_")

    def _verify_chunk_role(self, chunk: Dict[str, Any], user_role: str) -> bool:
        """
        Verify chunk role by checking parent document directly.
        Returns True if chunk belongs to user's role, False otherwise.
        """
        parent_id = chunk.get("parent_id")
        
        if not parent_id:
            logging.warning(f"Chunk {chunk.get('id')} has no parent_id")
            return False
        
        parent = self.parents.get(parent_id)
        
        if not parent:
            logging.warning(f"Parent {parent_id} not found in parents dictionary")
            return False
        
        parent_role = parent.get("role")
        if not parent_role:
            logging.warning(f"Parent {parent_id} has no role assigned")
            return False
        
        # Normalize both roles for comparison
        normalized_parent_role = self._normalize_role(parent_role)
        normalized_user_role = self._normalize_role(user_role)
        
        return normalized_parent_role == normalized_user_role
    
    def _filter_by_role(self, chunks: list[Dict[str, Any]], user_role: str) -> list[Dict[str, Any]]:
        """
        Filter chunks to only include those from documents matching user's role.
        Verifies each chunk against its parent document's role.
        """
        valid_chunks = []
        rejected_chunks = []
        
        for chunk in chunks:
            if self._verify_chunk_role(chunk, user_role):
                valid_chunks.append(chunk)
            else:
                parent_id = chunk.get("parent_id", "unknown")
                parent = self.parents.get(parent_id, {})
                parent_role = parent.get("role", "unknown")
                rejected_chunks.append({
                    "chunk_id": chunk.get("id"),
                    "parent_id": parent_id,
                    "parent_role": parent_role,
                    "source": chunk.get("source")
                })
        
        if rejected_chunks:
            logging.info(
                f"Filtered out {len(rejected_chunks)} chunks not matching role '{user_role}': "
                f"{rejected_chunks}"
            )
        
        return valid_chunks
    

    def query(self, user_query: str, user_role: str, top_k: int = 3):
        """
        Query the RAG system with strict role-based access control.
        
        Args:
            user_query: The user's question
            user_role: The user's role (will be normalized)
            top_k: Number of top chunks to return after reranking
            
        Returns:
            Dictionary containing answer and sources
        """
        # Normalize user role
        normalized_user_role = self._normalize_role(user_role)
        logging.info(f"Processing query for role: {normalized_user_role}")
        
        # Step 1: Initial retrieval (already filtered by role in retriever)
        candidates = self.retriever.search(
            user_query,
            user_role=normalized_user_role,
            top_k=self.config.top_k_retriever
        )
        
        if not candidates:
            logging.info(f"No candidates found for role: {normalized_user_role}")
            return {
                "answer": "This information is not available for your role.",
                "sources": [],
                "role_verified": True
            }
        
        logging.info(f"Retrieved {len(candidates)} candidates from retriever")
        
        # Step 2: Double-check role verification against parent documents
        role_verified_candidates = self._filter_by_role(candidates, normalized_user_role)
        
        if not role_verified_candidates:
            logging.warning(
                f"All {len(candidates)} candidates failed parent role verification "
                f"for role: {normalized_user_role}"
            )
            return {
                "answer": "This information is not available for your role.",
                "sources": [],
                "role_verified": True
            }
        
        logging.info(
            f"{len(role_verified_candidates)}/{len(candidates)} candidates passed "
            f"parent role verification"
        )
        
        # Step 3: Rerank the role-verified candidates
        pairs = [[user_query, c["text"]] for c in role_verified_candidates]
        scores = self.reranker.predict(pairs)
        
        for c, s in zip(role_verified_candidates, scores):
            c["rerank_score"] = float(s)
        
        # Step 4: Get top-k after reranking
        top_chunks = sorted(
            role_verified_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k]
        
        logging.info(f"Selected top {len(top_chunks)} chunks after reranking")
        
        # Step 5: Enrich with parent context
        enriched = add_parent_context(top_chunks, self.parents)
        
        # Step 6: Final verification (redundant but safe)
        for chunk in enriched:
            parent_role = chunk.get("parent_role")
            if parent_role:
                normalized_parent_role = self._normalize_role(parent_role)
                if normalized_parent_role != normalized_user_role:
                    logging.error(
                        f"CRITICAL: Chunk passed filters but parent_role "
                        f"'{normalized_parent_role}' != user_role '{normalized_user_role}'"
                    )
                    return {
                        "answer": "This information is not available for your role.",
                        "sources": [],
                        "role_verified": False
                    }
        
        # Step 7: Generate answer
        prompt = build_rag_prompt(user_query, enriched, normalized_user_role)
        answer = self.llm.generate(prompt)
        
        # Step 8: Return results with role verification info
        return {
            "answer": answer,
            "sources": [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "section": c.get("section", "N/A"),
                    "role": c.get("parent_role", "unknown")
                } for c in enriched
            ],
            "role_verified": True,
            "chunks_retrieved": len(candidates),
            "chunks_after_verification": len(role_verified_candidates),
            "chunks_used": len(enriched)
        }