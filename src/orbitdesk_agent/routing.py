from . import config


def route_after_triage(state) -> str:
    if state["triage"]["policy_violation"]:
        return "safe_response"
    return "retrieve"


def classify_route(state) -> str:
    confidence = state.get("retrieval_confidence", 0.0)
    triage = state["triage"]

    if confidence < config.LOW_EVIDENCE_THRESHOLD:
        return "out_of_scope"
    if triage["escalation_signal"]:
        return "escalate"
    if triage["vague"]:
        return "clarify"
    if confidence < config.CLARIFY_THRESHOLD:
        return "clarify"
    return "answer"


def route_after_retrieval(state) -> str:
    decision = classify_route(state)
    return {
        "out_of_scope": "safe_response",
        "escalate": "generate",
        "clarify": "clarify",
        "answer": "generate",
    }[decision]


def route_after_verification(state) -> str:
    verification = state.get("verification", {})
    if verification.get("passed"):
        return "finalize"
    if state.get("generation_attempts", 0) < config.MAX_GENERATION_ATTEMPTS:
        return "regenerate"
    return "safe_failure"
