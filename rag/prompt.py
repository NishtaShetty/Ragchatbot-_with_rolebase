# rag/prompt.py
from typing import List, Dict, Any


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]],  user_role) -> str:
    """
    Build prompt for LLM using retrieved context
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        section_info = f" (Section: {chunk['section']})" if chunk.get("section") else ""
        source_info = f"[Source: {chunk['source']}, Page {chunk['page']}{section_info}]"

        context_parts.append(
            f"Context {i}:\n{source_info}\n{chunk['text']}\n"
        )

    context = "\n".join(context_parts)

    prompt = f"""
Role: You are a company process assistant for role: {user_role}

Context Information:
{context}

Task:
Using ONLY the context above, answer the following query.

Query:
{query}

Rules:
1. Do NOT use outside knowledge.
2.If question is outside your role, say:
  "This information is not available for your role."
3. Do NOT guess
5. If the answer is not in the context, say:
   "I do not have enough information to answer this based on the provided documents."
6. Cite sources using page/section references when available.

Answer:
""".strip()

    return prompt
