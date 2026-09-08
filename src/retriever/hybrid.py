"""Combine KG and vector retrieval into a single context string."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import spacy
except ImportError:
    spacy = None

try:
    from src.retriever.faiss_retriever import get_passage_context
    from src.retriever.kg_retriever import get_kg_context
except ImportError:
    from retriever.faiss_retriever import get_passage_context
    from retriever.kg_retriever import get_kg_context

# Load nlp once at module level to avoid reloading on every query
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    if spacy is None:
        return None
    try:
        _nlp = spacy.load("en_core_web_lg")
        from src.kg_builder.entity_ruler import add_entity_ruler
        _nlp = add_entity_ruler(_nlp)
    except OSError:
        try:
            _nlp = spacy.blank("en")
        except Exception:
            return None
    return _nlp


def _entity_fallback(query: str) -> list[str]:
    pattern = r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*\b|\b[A-Z]{2,}\b"
    candidates = re.findall(pattern, query)
    return [c.strip() for c in candidates if c.strip()][:10]


def _query_keywords(query: str) -> list[str]:
    return [word.lower() for word in query.split() if len(word) > 3]


def _extract_entities(query: str) -> list[str]:
    if not query:
        return []

    nlp = _get_nlp()
    if nlp is None:
        return _entity_fallback(query)

    doc = nlp(query)
    entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    if entities:
        return entities[:10]
    return _entity_fallback(query)


def retrieve(query: str, passage_limit: int = 5, max_tokens: int = 4000) -> str:
    """Merge KG and vector-context evidence into a single retrieval string."""
    if not query or not query.strip():
        return ""

    keywords = _query_keywords(query)
    entities = _extract_entities(query)
    entities = [
        entity for entity in entities
        if any(keyword in entity.lower() for keyword in keywords)
    ]
    kg_context = get_kg_context(entities)
    passage_context = get_passage_context(query, top_k=passage_limit)

    combined_parts = []
    if kg_context.strip():
        combined_parts.append(kg_context.strip())
    if passage_context.strip():
        combined_parts.append(passage_context.strip())

    merged = "\n\n".join(combined_parts)
    tokens = merged.split()
    if len(tokens) > max_tokens:
        merged = " ".join(tokens[:max_tokens])
    return merged.strip()


if __name__ == "__main__":
    print(retrieve("What is ISRO?"))