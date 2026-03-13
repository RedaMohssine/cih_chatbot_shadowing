"""Retrieval : recherche vectorielle MMR avec déduplication par document."""

from langchain_core.documents import Document

from indexer import load_vectorstore
from config import TOP_K, MMR_FETCH_K, MMR_LAMBDA, MAX_CHUNKS_PER_DOC


def retrieve(query: str, top_k: int = TOP_K) -> list[Document]:
    """Cherche les chunks pertinents via MMR (diversité) + déduplication."""
    vectorstore = load_vectorstore()

    # MMR = Maximum Marginal Relevance : équilibre pertinence et diversité
    results = vectorstore.max_marginal_relevance_search(
        query,
        k=top_k * 2,          # récupérer plus pour dédupliquer ensuite
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )

    if not results:
        return []

    # Déduplication : max N chunks par document source
    seen: dict[str, int] = {}
    deduped = []
    for doc in results:
        doc_id = doc.metadata.get("id", doc.metadata.get("title", ""))
        if seen.get(doc_id, 0) < MAX_CHUNKS_PER_DOC:
            deduped.append(doc)
            seen[doc_id] = seen.get(doc_id, 0) + 1
        if len(deduped) >= top_k:
            break

    return deduped


def format_context(docs: list[Document]) -> str:
    """Formate les documents récupérés en contexte pour le LLM."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Source {i}] {meta.get('title', 'Sans titre')} | Catégorie: {meta.get('category', 'N/A')}"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)
