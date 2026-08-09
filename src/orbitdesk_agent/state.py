from typing import Any, Dict, List, TypedDict


class TriageSignals(TypedDict):
    policy_violation: bool
    matched_keywords: List[str]
    vague: bool
    escalation_signal: bool


class RetrievedPassage(TypedDict):
    source_id: str
    source_type: str
    title: str
    section: str
    status: str
    text: str
    score: float


class AgentState(TypedDict, total=False):
    question_id: str
    question: str
    debug_force_failure: bool
    triage: TriageSignals
    retrieved: List[RetrievedPassage]
    retrieval_confidence: float
    target_classification: str
    draft_answer: str
    draft_sources: List[Dict[str, str]]
    generation_attempts: int
    verification: Dict[str, Any]
    verification_feedback: str
    final_response: Dict[str, Any]
    trace: List[str]
    timings: Dict[str, float]
