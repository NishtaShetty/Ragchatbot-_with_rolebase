# rag/retriever.py
import numpy as np
import logging
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _flatten_chroma_ids(ids_list):
    flattened = []

    def _recurse(x):
        if x is None:
            return
        if isinstance(x, str):
            flattened.append(x)
        elif isinstance(x, (int, float)):
            flattened.append(str(x))
        elif isinstance(x, (list, tuple)):
            for el in x:
                _recurse(el)
        else:
            flattened.append(str(x))

    _recurse(ids_list)
    return flattened


def _min_max_normalize(scores: np.ndarray):
    if len(scores) == 0:
        return scores
    min_s, max_s = float(scores.min()), float(scores.max())
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


class HybridRetriever:
    def __init__(self, docs, embed_model, collection, config):
        self.docs = docs
        self.embed_model = embed_model
        self.collection = collection
        self.config = config

        # Build BM25 indexes PER ROLE
        self.role_bm25 = {}
        self.role_doc_ids = {}

        logger.info("Building role-specific BM25 indexes...")
        
        for d in docs:
            role = d.get("role")
            if not role:
                logger.warning(f"Document {d.get('id')} has no role assigned")
                role = "unknown"
            
            if role not in self.role_doc_ids:
                self.role_doc_ids[role] = []
            self.role_doc_ids[role].append(d)

        for role, role_docs in self.role_doc_ids.items():
            tokenized = [d["text"].lower().split() for d in role_docs]
            self.role_bm25[role] = BM25Okapi(tokenized)
            logger.info(f"  Role '{role}': {len(role_docs)} documents indexed")

        self.id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
        logger.info(f"Total documents indexed: {len(docs)}")

    def _normalize_role(self, role: str) -> str:
        """Normalize role name for consistent comparison"""
        return role.strip().lower().replace(" ", "_")

    def search(self, query: str, user_role: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic (ChromaDB) and keyword (BM25) search.
        Only returns documents that match the user's role.
        
        Args:
            query: Search query
            user_role: User's role (will be normalized)
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        # Normalize role
        user_role = self._normalize_role(user_role)
        logger.info(f"Searching for role: '{user_role}'")
        
        # Check if role exists
        if user_role not in self.role_bm25:
            logger.warning(
                f"Role '{user_role}' not found in indexed roles. "
                f"Available roles: {list(self.role_bm25.keys())}"
            )
            return []

        # ---------------- SEMANTIC SEARCH (ROLE FILTERED) ----------------
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True
        )[0].tolist()

        try:
            chroma_res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.config.top_k_retriever, len(self.docs)),
                where={"role": user_role},
                include=["distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            chroma_res = {"ids": [], "distances": []}

        chroma_ids = _flatten_chroma_ids(chroma_res.get("ids", []))
        distances = np.array(
            _flatten_chroma_ids(chroma_res.get("distances", [])),
            dtype=float
        )

        logger.info(f"Semantic search returned {len(chroma_ids)} results")

        semantic_scores = {
            did: 1 / (1 + d)
            for did, d in zip(chroma_ids, distances)
        }

        # ---------------- BM25 SEARCH (ROLE FILTERED) ----------------
        role_docs = self.role_doc_ids[user_role]
        bm25 = self.role_bm25[user_role]

        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_scores = _min_max_normalize(np.array(bm25_scores))

        logger.info(f"BM25 search computed scores for {len(bm25_scores)} documents")

        # ---------------- COMBINE SCORES ----------------
        combined = {}

        # Add semantic scores
        for did, score in semantic_scores.items():
            combined[did] = self.config.semantic_weight * score

        # Add BM25 scores
        for i, score in enumerate(bm25_scores):
            did = role_docs[i]["id"]
            combined[did] = combined.get(did, 0) + self.config.keyword_weight * score

        # Rank by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        logger.info(f"Combined {len(combined)} unique documents, returning top {len(ranked)}")

        # Build results with full document info
        results = []
        for did, score in ranked:
            if did not in self.id_to_idx:
                logger.warning(f"Document ID {did} not found in index")
                continue
            
            doc = self.docs[self.id_to_idx[did]]
            result = {**doc, "score": float(score)}
            results.append(result)

        return results

    def get_role_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed roles"""
        return {
            "total_roles": len(self.role_bm25),
            "roles": list(self.role_bm25.keys()),
            "documents_per_role": {
                role: len(docs) for role, docs in self.role_doc_ids.items()
            }
        }