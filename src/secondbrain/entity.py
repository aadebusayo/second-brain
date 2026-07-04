"""
Entity model for secondbrain.

Typed entity nodes live in the same associative graph as memory chunks.
This means spreading activation naturally flows across entity-chunk-entity
paths — not a separate knowledge graph, but first-class nodes in the
same weighted, learnable topology.

Entities are self-evolving: they can be auto-extracted from text via LLM,
with open-ended types and progressive refinement as more chunks reference
the same entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .embeddings.local import LocalEmbeddingProvider


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """A single entity extracted from text."""
    name: str
    entity_type: str         # open-ended, LLM-assigned
    description: str = ""    # brief context from the source text
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """A relation between two extracted entities."""
    source_name: str
    target_name: str
    relation_type: str       # open-ended, LLM-assigned


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction from a text fragment."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity Node
# ---------------------------------------------------------------------------

class Entity:
    """A typed entity in the associative graph."""

    __slots__ = ("id", "name", "entity_type", "properties")

    def __init__(
        self,
        id: str,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.entity_type = entity_type
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
        }


# ---------------------------------------------------------------------------
# Entity Model
# ---------------------------------------------------------------------------

class EntityModel:
    """
    Manage typed entities within the memory graph.

    Entities are stored as regular graph nodes with metadata
    distinguishing them from memory chunks. Entity-to-entity
    relations use typed edges alongside weighted edges.
    """

    # Well-known entity types
    PERSON = "Person"
    ORGANIZATION = "Organization"
    DOCUMENT = "Document"
    CONCEPT = "Concept"
    EVENT = "Event"
    LOCATION = "Location"
    PRODUCT = "Product"
    REGULATION = "Regulation"

    # Well-known relation types
    WORKS_AT = "works_at"
    OWNS = "owns"
    REGULATES = "regulates"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    ISSUED_BY = "issued_by"
    APPLIES_TO = "applies_to"
    RELATED_TO = "related_to"
    MENTIONS = "mentions"

    def __init__(self, graph, embedding_provider=None) -> None:
        self._graph = graph
        self._embedder = embedding_provider or LocalEmbeddingProvider()
        self._entity_index: Dict[str, str] = {}  # normalized_name → node_id

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """
        Create a new entity node in the graph.

        Returns the Entity. If an entity with the same normalized name
        already exists, returns the existing one (idempotent).
        """
        key = self._normalize(name)
        existing_id = self._entity_index.get(key)
        if existing_id is not None:
            node = self._graph.get_node(existing_id)
            if node is not None:
                return Entity(
                    id=existing_id,
                    name=node.metadata.get("entity_name", name),
                    entity_type=node.metadata.get("entity_type", entity_type),
                    properties=node.metadata.get("entity_properties", {}),
                )

        # Build embedding from type + name for semantic similarity
        embed_text = f"{entity_type}: {name}"
        embedding = self._embedder.embed(embed_text)

        node = self._graph.add_node(
            text=embed_text,
            embedding=embedding,
            metadata={
                "kind": "entity",
                "entity_name": name,
                "entity_type": entity_type,
                "entity_properties": properties or {},
            },
        )
        self._entity_index[key] = node.id

        # Set a reasonable initial base activation so entities
        # participate in propagation from the start.
        node.base_activation = 0.5

        return Entity(id=node.id, name=name, entity_type=entity_type, properties=properties or {})

    def find_entity(self, name: str) -> Optional[Entity]:
        """Look up an entity by name (case-insensitive, whitespace-normalized)."""
        key = self._normalize(name)
        node_id = self._entity_index.get(key)
        if node_id is None:
            return None
        node = self._graph.get_node(node_id)
        if node is None:
            return None
        return Entity(
            id=node_id,
            name=node.metadata.get("entity_name", name),
            entity_type=node.metadata.get("entity_type", "Concept"),
            properties=node.metadata.get("entity_properties", {}),
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its node ID."""
        node = self._graph.get_node(entity_id)
        if node is None or node.metadata.get("kind") != "entity":
            return None
        return Entity(
            id=entity_id,
            name=node.metadata.get("entity_name", ""),
            entity_type=node.metadata.get("entity_type", "Concept"),
            properties=node.metadata.get("entity_properties", {}),
        )

    def list_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """List all entities, optionally filtered by type."""
        results: List[Entity] = []
        for node in self._graph.list_nodes():
            if node.metadata.get("kind") != "entity":
                continue
            if entity_type is not None and node.metadata.get("entity_type") != entity_type:
                continue
            results.append(Entity(
                id=node.id,
                name=node.metadata.get("entity_name", ""),
                entity_type=node.metadata.get("entity_type", "Concept"),
                properties=node.metadata.get("entity_properties", {}),
            ))
        return results

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.8,
    ) -> None:
        """
        Add a typed relation between two entities.

        Creates a weighted edge in the graph with the relation type
        stored in edge metadata. The weight determines how strongly
        activation flows between the entities.
        """
        self._graph.add_edge(source_id, target_id, weight=float(weight))
        # Store relation type on the edge
        if hasattr(self._graph, "_graph"):
            nx_graph = self._graph._graph
            if nx_graph.has_edge(source_id, target_id):
                nx_graph[source_id][target_id]["relation_type"] = relation_type

    def get_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return all typed relations for an entity."""
        relations: List[Dict[str, Any]] = []
        for neighbor_id in self._graph.neighbors(entity_id):
            weight = self._graph.get_edge_weight(entity_id, neighbor_id)
            rel_type = "related_to"
            if hasattr(self._graph, "_graph"):
                edge_data = self._graph._graph.get_edge_data(entity_id, neighbor_id)
                if edge_data:
                    rel_type = edge_data.get("relation_type", "related_to")
            neighbor = self.get_entity(neighbor_id)
            relations.append({
                "target_id": neighbor_id,
                "target_name": neighbor.name if neighbor else "?",
                "relation_type": rel_type,
                "weight": weight,
            })
        return relations

    # ------------------------------------------------------------------
    # Entity ↔ Memory Chunk linking
    # ------------------------------------------------------------------

    def link_to_node(
        self,
        entity_id: str,
        node_id: str,
        relation_type: str = "mentions",
        weight: float = 0.7,
    ) -> None:
        """
        Link an entity to a memory chunk node.

        Creates a weighted, typed edge so spreading activation flows
        between the entity and the chunk.
        """
        self._graph.add_edge(entity_id, node_id, weight=float(weight))
        if hasattr(self._graph, "_graph"):
            nx_graph = self._graph._graph
            if nx_graph.has_edge(entity_id, node_id):
                nx_graph[entity_id][node_id]["relation_type"] = relation_type

    def get_linked_chunks(self, entity_id: str) -> List[str]:
        """
        Return IDs of all memory chunks linked to this entity.
        """
        chunk_ids: List[str] = []
        for neighbor_id in self._graph.neighbors(entity_id):
            neighbor = self._graph.get_node(neighbor_id)
            if neighbor is not None and neighbor.metadata.get("kind") != "entity":
                chunk_ids.append(neighbor_id)
        return chunk_ids

    def get_entity_neighborhood(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Return the full neighborhood of an entity:
          - linked entities (with relation types)
          - linked memory chunks
          - access stats
        """
        node = self._graph.get_node(entity_id)
        if node is None:
            return {"error": "entity not found"}

        entity = self.get_entity(entity_id)
        linked_entities = self.get_relations(entity_id)
        linked_chunks = self.get_linked_chunks(entity_id)

        return {
            "entity": entity.to_dict() if entity else {},
            "linked_entities": linked_entities,
            "linked_chunks": [
                {
                    "node_id": cid,
                    "text": (self._graph.get_node(cid).text if self._graph.get_node(cid) else "")[:200],
                }
                for cid in linked_chunks[:10]
            ],
            "base_activation": node.base_activation,
            "access_count": len(node.access_log),
        }

    # ------------------------------------------------------------------
    # Self-evolving entity extraction
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
        llm_client: Any = None,
    ) -> ExtractionResult:
        """
        Extract entities and relations from *text*.

        If no *llm_client* is provided, logs a warning and returns an
        empty result — no regex fallback.  Entity extraction requires
        an LLM (DeepSeek, OpenAI, or Anthropic).
        """
        if llm_client is None:
            import logging
            logging.getLogger("secondbrain.entity").warning(
                "No LLM client configured — entity auto-extraction skipped. "
            )
            return ExtractionResult()
        return self._llm_extract(text, llm_client)

    def ingest_extraction(
        self,
        chunk_node_id: str,
        result: ExtractionResult,
    ) -> Dict[str, Any]:
        """
        Wire extracted entities and relations into the graph.

        - Creates or updates entity nodes (idempotent by name)
        - Links entities to the source chunk
        - Creates typed relations between co-mentioned entities
        - Returns summary of what was created/updated
        """
        entity_ids: Dict[str, str] = {}  # name → node_id
        created = 0
        updated = 0

        # Step 1: Upsert entities
        for e in result.entities:
            existing = self.find_entity(e.name)
            if existing is not None:
                # Self-evolve: merge new properties into existing entity
                node = self._graph.get_node(existing.id)
                if node is not None:
                    merged = dict(node.metadata.get("entity_properties", {}))
                    merged.update(e.properties)
                    if e.description:
                        merged["description"] = e.description
                    node.metadata["entity_properties"] = merged
                    # If the LLM assigned a more specific type, adopt it
                    if e.entity_type and e.entity_type != existing.entity_type:
                        node.metadata["entity_type"] = e.entity_type
                entity_ids[e.name] = existing.id
                updated += 1
            else:
                entity = self.add_entity(e.name, e.entity_type, properties={
                    **(e.properties),
                    "description": e.description,
                    "mention_count": 1,
                    "first_seen_in": chunk_node_id,
                })
                entity_ids[e.name] = entity.id
                created += 1

            # Link entity to the source chunk
            self.link_to_node(entity_ids[e.name], chunk_node_id, "mentions")

        # Step 2: Create relations between co-mentioned entities
        relations_created = 0
        for rel in result.relations:
            src_id = entity_ids.get(rel.source_name)
            tgt_id = entity_ids.get(rel.target_name)
            if src_id and tgt_id and src_id != tgt_id:
                self.add_relation(src_id, tgt_id, rel.relation_type, weight=0.75)
                relations_created += 1

        return {
            "entities_created": created,
            "entities_updated": updated,
            "relations_created": relations_created,
            "entity_ids": entity_ids,
        }

    # ------------------------------------------------------------------
    # LLM-backed extraction
    # ------------------------------------------------------------------

    EXTRACTION_PROMPT = """Extract named entities and their relationships from the text below.

Return ONLY valid JSON — no commentary, no markdown fences.

Schema:
{
  "entities": [
    {
      "name": "EXACT full entity name as it appears in the text (copy verbatim — do NOT truncate, abbreviate, or shorten. Include ALL words like 'of Kenya', 'of America', 'Group', 'PLC', 'Ltd', 'Inc')",
      "entity_type": "a concise type label (invent freely — e.g. CentralBank, FintechRegulation, PaymentRail, GovernmentAgency, LegalStatute, TechnologyPlatform, IndustryStandard)",
      "description": "one sentence of context from the text",
      "properties": {"key": "value"}
    }
  ],
  "relations": [
    {
      "source_name": "exact entity name as above",
      "target_name": "exact entity name as above",
      "relation_type": "a concise verb phrase (e.g. regulates, supervises, depends_on, implements, issues)"
    }
  ]
}

CRITICAL RULES:
- Entity names MUST be copied VERBATIM from the text — every word, including connectors like 'of', 'the', 'and', 'for'. Do NOT shorten 'Central Bank of Kenya' to 'Central Bank'.
- Extract every named entity: organisations, people, regulations, products, technologies, concepts, jurisdictions, standards.
- Entity types should be specific and descriptive.
- Extract relations ONLY between entities explicitly named in the text.
- If no relations are explicit, return an empty relations array.

Text:
{text}"""

    def _llm_extract(self, text: str, llm_client: Any) -> ExtractionResult:
        """Use an LLM to extract entities and relations."""
        prompt = self.EXTRACTION_PROMPT.replace("{text}", text[:4000])

        try:
            raw: str | None = None

            # Anthropic-style: client.messages.create()
            if hasattr(llm_client, "messages"):
                try:
                    response = llm_client.messages.create(
                        model=getattr(llm_client, "model", "claude-sonnet-5"),
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = response.content[0].text
                except NotImplementedError:
                    pass  # DeepSeek raises NotImplementedError for .messages()

            # OpenAI-style: client.chat.completions.create()
            # (DeepSeek uses this path)
            if raw is None and hasattr(llm_client, "chat"):
                response = llm_client.chat.completions.create(
                    model=getattr(llm_client, "model", "gpt-4o"),
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.choices[0].message.content

            if raw is None:
                return self._fallback_extract(text)
        except Exception:
            return self._fallback_extract(text)

        return self._parse_llm_response(raw)

    def _parse_llm_response(self, raw: str) -> ExtractionResult:
        """Parse LLM JSON output, handling common formatting issues."""
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return ExtractionResult()
            else:
                return ExtractionResult()

        entities = [
            ExtractedEntity(
                name=self._clean_entity_name(e.get("name", "")),
                entity_type=e.get("entity_type", "Concept"),
                description=e.get("description", ""),
                properties=e.get("properties", {}),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]

        relations = [
            ExtractedRelation(
                source_name=r.get("source_name", ""),
                target_name=r.get("target_name", ""),
                relation_type=r.get("relation_type", "related_to"),
            )
            for r in data.get("relations", [])
            if r.get("source_name") and r.get("target_name")
        ]

        return ExtractionResult(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_entity_name(name: str) -> str:
        """Strip leading articles and whitespace from entity names."""
        cleaned = name.strip()
        for prefix in ("the ", "The "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        return cleaned

    @staticmethod
    def _normalize(name: str) -> str:
        return " ".join(name.lower().split())
