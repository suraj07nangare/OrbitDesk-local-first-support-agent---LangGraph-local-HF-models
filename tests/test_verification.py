import numpy as np

from orbitdesk_agent.nodes import verification_node


class _StubEmbeddingModel:
    def __init__(self, similarity: float):
        self._similarity = similarity

    def encode(self, texts):
        base = np.array([1.0, 0.0])
        if self._similarity >= 0.999:
            other = base
        else:
            other = np.array([self._similarity, (1 - self._similarity ** 2) ** 0.5])
        return np.array([base, other])


def _state(answer, sources, retrieved):
    return {
        "draft_answer": answer,
        "draft_sources": sources,
        "retrieved": retrieved,
        "question_id": "TEST",
        "trace": [],
        "timings": {},
    }


def test_verification_passes_for_grounded_answer_with_sources():
    state = _state(
        "Resave the schedule to clear the pending timezone notice. [KB-003]",
        [{"source_id": "KB-003", "passage": "Changing the Timezone"}],
        [{"source_id": "KB-003", "text": "Resave the schedule to clear the pending timezone notice.", "score": 0.7}],
    )
    result = verification_node(state, _StubEmbeddingModel(similarity=0.95))
    assert result["verification"]["passed"] is True
    assert result["verification"]["issues"] == []


def test_verification_fails_on_forbidden_action_claim():
    state = _state(
        "I have issued the refund to your account already.",
        [{"source_id": "KB-010", "passage": "Unsupported Actions"}],
        [{"source_id": "KB-010", "text": "The support assistant cannot issue refunds.", "score": 0.7}],
    )
    result = verification_node(state, _StubEmbeddingModel(similarity=0.95))
    assert result["verification"]["passed"] is False
    assert "unsupported_action_claim" in result["verification"]["issues"]


def test_verification_fails_on_missing_sources():
    state = _state(
        "Some answer text with no citations.",
        [],
        [{"source_id": "KB-001", "text": "context", "score": 0.7}],
    )
    result = verification_node(state, _StubEmbeddingModel(similarity=0.95))
    assert "missing_sources" in result["verification"]["issues"]


def test_verification_fails_on_low_grounding():
    state = _state(
        "Completely unrelated statement about the weather today.",
        [{"source_id": "KB-003", "passage": "Changing the Timezone"}],
        [{"source_id": "KB-003", "text": "Resave the schedule to clear the pending timezone notice.", "score": 0.7}],
    )
    result = verification_node(state, _StubEmbeddingModel(similarity=0.05))
    assert "low_grounding" in result["verification"]["issues"]


def test_verification_fails_on_empty_answer():
    state = _state("   ", [], [])
    result = verification_node(state, _StubEmbeddingModel(similarity=0.95))
    assert "empty_answer" in result["verification"]["issues"]
    assert result["verification"]["passed"] is False
