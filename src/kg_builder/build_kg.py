"""Build a domain-specific knowledge graph from preprocessed ISRO chunks."""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import spacy
from spacy.tokens import Doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"
KG_PATH = ROOT / "data" / "kg" / "knowledge_graph.json"
KG_PKL_PATH = ROOT / "data" / "kg" / "knowledge_graph.pkl"


def load_nlp():
    """Load spaCy pipeline with ISRO entity ruler."""
    nlp = spacy.load("en_core_web_lg")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.kg_builder.entity_ruler import add_entity_ruler
    nlp = add_entity_ruler(nlp)
    logger.info("spaCy pipeline loaded with ISRO entity ruler (%d patterns)", 
                len(nlp.get_pipe("entity_ruler").patterns))
    return nlp


def load_chunks() -> list[dict]:
    """Load preprocessed chunks from JSON."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("chunks", [])


def extract_entities(doc: Doc) -> list[tuple[str, str]]:
    """Extract named entities from a spaCy Doc."""
    entities = []
    for ent in doc.ents:
        label = ent.label_
        text = ent.text.strip()
        if len(text) > 2 and text.isascii() and label in {
            "MISSION", "LAUNCH_VEHICLE", "PAYLOAD", "PERSON",
            "ORG", "LOC", "TECH", "GPE", "NORP", "DATE", "TIME",
            "EVENT", "PRODUCT", "FAC", "WORK_OF_ART"
        }:
            entities.append((text, label))
    return entities


def extract_triples(doc: Doc, entities: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples using dependency parsing.
    Only considers entity pairs within 2-hop dependency distance.
    """
    triples = []
    entity_tokens = {}

    # Map entity text to their token spans
    for ent in doc.ents:
        entity_tokens[ent.text.strip()] = ent.root

    entity_texts = [e[0] for e in entities]

    for sent in doc.sents:
        sent_entities = [e for e in entity_texts if e in sent.text]
        if len(sent_entities) < 2:
            continue

        for i, subj_text in enumerate(sent_entities):
            for obj_text in sent_entities[i + 1:]:
                if subj_text == obj_text:
                    continue

                subj_root = entity_tokens.get(subj_text)
                obj_root = entity_tokens.get(obj_text)

                if subj_root is None or obj_root is None:
                    continue

                # Find governing verb or head connecting the two entities
                relation = None

                # Check if they share a common head within 2 hops
                subj_ancestors = {subj_root.head, subj_root.head.head}
                obj_ancestors = {obj_root.head, obj_root.head.head}
                common = subj_ancestors & obj_ancestors

                if common:
                    head = next(iter(common))
                    if head.pos_ in {"VERB", "AUX"}:
                        relation = head.lemma_.lower()
                    else:
                        relation = f"{subj_root.dep_}_{obj_root.dep_}"
                else:
                    # Direct dependency check
                    if subj_root.head == obj_root or obj_root.head == subj_root:
                        relation = subj_root.dep_.lower()

                if relation:
                    triples.append((subj_text, relation, obj_text))

    return triples


def build_graph(chunks: list[dict], nlp) -> nx.MultiDiGraph:
    """Build a NetworkX knowledge graph from all chunks."""
    G = nx.MultiDiGraph()
    total_triples = 0
    batch_size = 50

    texts = []
    metas = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or ""
            source = chunk.get("source_url", "")
        else:
            text = str(chunk)
            source = ""
        if text.strip():
            texts.append(text.strip())
            metas.append(source)

    logger.info("Processing %d chunks in batches of %d...", len(texts), batch_size)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]

        for doc, source in zip(nlp.pipe(batch_texts, batch_size=batch_size), batch_metas):
            entities = extract_entities(doc)
            triples = extract_triples(doc, entities)

            for subj, rel, obj in triples:
                if not G.has_node(subj):
                    G.add_node(subj)
                if not G.has_node(obj):
                    G.add_node(obj)
                G.add_edge(subj, obj, relation=rel, source=source)
                total_triples += 1

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Processed %d/%d chunks, %d triples so far",
                        min(i + batch_size, len(texts)), len(texts), total_triples)

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def save_graph(G: nx.MultiDiGraph):
    """Save graph as both JSON and pickle."""
    KG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle (fast, preserves full graph)
    with open(KG_PKL_PATH, "wb") as f:
        pickle.dump(G, f)
    logger.info("Graph saved to %s", KG_PKL_PATH)

    # Save as JSON (human-readable, for inspection)
    graph_data = {
        "nodes": list(G.nodes()),
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": d.get("relation", "related_to"),
                "doc_source": d.get("source", "")
            }
            for u, v, d in G.edges(data=True)
        ]
    }
    KG_PATH.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Graph JSON saved to %s", KG_PATH)


def main():
    logger.info("Loading chunks from %s", CHUNKS_PATH)
    chunks = load_chunks()
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Loading spaCy pipeline...")
    nlp = load_nlp()

    logger.info("Building knowledge graph...")
    G = build_graph(chunks, nlp)

    logger.info("Saving graph...")
    save_graph(G)

    logger.info("Done. Nodes: %d, Edges: %d", G.number_of_nodes(), G.number_of_edges())
    logger.info("Top 10 most connected nodes:")
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_nodes:
        logger.info("  %s: degree %d", node, degree)


if __name__ == "__main__":
    main()