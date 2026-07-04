"""Public package interface for the secondbrain memory system."""

from .api import (
    SecondBrain,
    add_entity,
    add_relation,
    consolidate,
    explain,
    find_entity,
    get_entity_neighborhood,
    link_entity_to_node,
    list_entities,
    mark_relevant,
    recall,
    recall_naive,
    reinforce_pair,
    reinforce_result_pairs,
    remember,
)
from .consolidate import ConsolidationReport
from .entity import Entity, EntityModel
from .llm import DeepSeekClient, create_llm_client

__all__ = [
    "SecondBrain",
    "Entity",
    "EntityModel",
    "remember",
    "recall",
    "recall_naive",
    "explain",
    "consolidate",
    "mark_relevant",
    "reinforce_pair",
    "reinforce_result_pairs",
    "add_entity",
    "add_relation",
    "find_entity",
    "list_entities",
    "get_entity_neighborhood",
    "link_entity_to_node",
    "DeepSeekClient",
    "create_llm_client",
    "ConsolidationReport",
]
