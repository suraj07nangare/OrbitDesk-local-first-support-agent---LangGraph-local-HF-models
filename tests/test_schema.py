import json

import pytest
from jsonschema import ValidationError, validate

from orbitdesk_agent import config
from orbitdesk_agent.nodes import clarify_node, safe_failure_node, safe_response_node


def _load_schema():
    return json.loads(config.OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_valid_answerable_response_matches_schema():
    schema = _load_schema()
    response = {
        "classification": "answerable",
        "answer": "Resave the schedule to apply the new timezone. [KB-003]",
        "sources": [{"source_id": "KB-003", "passage": "Changing the Timezone"}],
        "confidence": 0.82,
        "requires_human": False,
        "reason": "Answer generated from KB-003 and passed verification.",
        "clarification_question": None,
        "warnings": [],
    }
    validate(instance=response, schema=schema)


def test_missing_required_field_fails_schema():
    schema = _load_schema()
    response = {
        "classification": "answerable",
        "answer": "Resave the schedule.",
        "sources": [],
        "confidence": 0.5,
        "requires_human": False,
    }
    with pytest.raises(ValidationError):
        validate(instance=response, schema=schema)


def test_clarify_node_output_matches_schema():
    schema = _load_schema()
    response = clarify_node({"trace": []})["final_response"]
    validate(instance=response, schema=schema)


def test_safe_response_node_output_matches_schema():
    schema = _load_schema()
    response = safe_response_node({"trace": []})["final_response"]
    validate(instance=response, schema=schema)


def test_safe_failure_node_output_matches_schema():
    schema = _load_schema()
    state = {"trace": [], "verification": {"issues": ["low_grounding"]}, "retrieved": []}
    response = safe_failure_node(state)["final_response"]
    validate(instance=response, schema=schema)
