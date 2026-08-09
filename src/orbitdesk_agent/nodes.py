import re
import time

from . import config, routing
from .logging_utils import get_logger

logger = get_logger(__name__)

POLICY_PHRASES = [
    "refund",
    "chargeback",
    "cancel my subscription",
    "cancel the subscription",
    "legal advice",
    "medical advice",
    "financial advice",
    "ignore the supplied documentation",
    "ignore the documentation",
    "ignore previous instructions",
    "ignore the above",
    "ignore all prior instructions",
    "disregard the documentation",
    "disregard your instructions",
]

VAGUE_PATTERN = re.compile(
    r"(not working|isn.?t working|is not working|broken|doesn.?t work|does not work|"
    r"stopped working|how to fix|fix (it|this))",
    re.IGNORECASE,
)

ESCALATION_PATTERN = re.compile(
    r"(already (checked|tried)|two (consecutive|failed|runs)|in a row|"
    r"did not work|didn.?t work|escalate)",
    re.IGNORECASE,
)

ERROR_CODE_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")

FORBIDDEN_PATTERNS = [
    re.compile(r"i (have|has) (already )?(issued|processed|completed|approved) (a |the )?refund", re.IGNORECASE),    re.compile(r"your refund (has been|is) (issued|processed)", re.IGNORECASE),
    re.compile(r"i (have )?(created|generated|revealed) (the |a )?(api )?(credential|secret|token)", re.IGNORECASE),
    re.compile(r"the (secret|password|token) is[:\s]", re.IGNORECASE),
    re.compile(r"i (have )?changed your (role|workspace)", re.IGNORECASE),
    re.compile(r"here is legal advice", re.IGNORECASE),
]

SYSTEM_PROMPT = (
    "You are the OrbitDesk support assistant. Answer only using the evidence passages "
    "provided. Do not invent steps that are not present in the evidence. If the evidence "
    "does not fully cover the question, say what is known and what is missing. Never claim "
    "to have performed an account action such as issuing a refund, creating a credential or "
    "changing a role. Keep the answer under 180 words and reference sources using their "
    "source id in square brackets, for example [KB-004]."
)

CLASSIFICATION_INSTRUCTIONS = {
    "answerable": "Give a direct, step-by-step answer grounded in the evidence.",
    "requires_escalation": (
        "Explain which documented checks are already covered, state that escalation is "
        "warranted, list exactly which details should be collected per the evidence, and "
        "name the appropriate team if the evidence names one. Do not claim the issue is "
        "already escalated on the user's behalf."
    ),
}


def triage_node(state):
    started = time.perf_counter()
    question = state["question"]
    lowered = question.lower()

    matched_policy = [phrase for phrase in POLICY_PHRASES if phrase in lowered]
    has_error_code = bool(ERROR_CODE_PATTERN.search(question))
    looks_vague = bool(VAGUE_PATTERN.search(question)) and not has_error_code
    escalation_signal = bool(ESCALATION_PATTERN.search(question)) and (has_error_code or "escalate" in lowered)

    triage = {
        "policy_violation": bool(matched_policy),
        "matched_keywords": matched_policy,
        "vague": looks_vague,
        "escalation_signal": escalation_signal,
    }

    trace = list(state.get("trace", []))
    trace.append("triage")
    logger.info("Triage result for %s: %s", state.get("question_id", "adhoc"), triage)

    return {
        "triage": triage,
        "trace": trace,
        "timings": {**state.get("timings", {}), "triage": time.perf_counter() - started},
    }


def retrieval_node(state, retrieval_index):
    started = time.perf_counter()
    results = retrieval_index.search(state["question"], top_k=config.TOP_K)
    confidence = results[0]["score"] if results else 0.0

    trace = list(state.get("trace", []))
    trace.append("retrieve")
    logger.info(
        "Retrieved %d passages for %s (top score=%.3f)",
        len(results),
        state.get("question_id", "adhoc"),
        confidence,
    )

    return {
        "retrieved": results,
        "retrieval_confidence": confidence,
        "trace": trace,
        "timings": {**state.get("timings", {}), "retrieve": time.perf_counter() - started},
    }


def _format_evidence(passages):
    lines = []
    for passage in passages[: config.MAX_CONTEXT_PASSAGES]:
        marker = f"[{passage['source_id']}]"
        note = " (superseded, historical only)" if passage["status"] == "superseded" else ""
        lines.append(f"{marker}{note} {passage['section']}: {passage['text']}")
    return "\n\n".join(lines)


def generation_node(state, generation_model):
    started = time.perf_counter()
    passages = state.get("retrieved", [])
    relevant = [p for p in passages if p["score"] >= config.LOW_EVIDENCE_THRESHOLD] or passages[:2]
    evidence_text = _format_evidence(relevant)

    decision = routing.classify_route(state)
    target_classification = "requires_escalation" if decision == "escalate" else "answerable"
    instruction = CLASSIFICATION_INSTRUCTIONS.get(target_classification, CLASSIFICATION_INSTRUCTIONS["answerable"])

    feedback = state.get("verification_feedback")
    feedback_block = ""
    if feedback:
        feedback_block = f"\n\nThe previous attempt was rejected because: {feedback}."
        if "incomplete_answer" in feedback:
            feedback_block += (
                " Your previous answer was cut off before it finished. Keep the new answer to "
                "at most 5 short sentences and make sure the final sentence is complete."
            )
        else:
            feedback_block += " Fix this in the new answer."

    user_prompt = (
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {state['question']}\n\n"
        f"Instruction: {instruction}{feedback_block}"
    )

    result = generation_model.generate(SYSTEM_PROMPT, user_prompt)
    attempts = state.get("generation_attempts", 0) + 1
    text = result["text"]

    if state.get("debug_force_failure") and attempts == 1:
        text = f"{text} I have already issued the refund to your account."

    draft_sources = [{"source_id": p["source_id"], "passage": p["section"]} for p in relevant]

    trace = list(state.get("trace", []))
    trace.append("generate")
    logger.info(
        "Generated draft answer for %s (attempt %d, target=%s, latency=%.2fs)",
        state.get("question_id", "adhoc"),
        attempts,
        target_classification,
        result["latency_seconds"],
    )

    return {
        "draft_answer": text,
        "draft_sources": draft_sources,
        "target_classification": target_classification,
        "generation_attempts": attempts,
        "trace": trace,
        "timings": {**state.get("timings", {}), f"generate_attempt_{attempts}": result["latency_seconds"]},
    }


def verification_node(state, embedding_model):
    started = time.perf_counter()
    answer = state.get("draft_answer", "")
    sources = state.get("draft_sources", [])
    evidence_text = "\n".join(p["text"] for p in state.get("retrieved", [])[: config.MAX_CONTEXT_PASSAGES])

    issues = []
    if not answer.strip():
        issues.append("empty_answer")
    if not sources:
        issues.append("missing_sources")

    if any(pattern.search(answer) for pattern in FORBIDDEN_PATTERNS):
        issues.append("unsupported_action_claim")

    stripped_answer = answer.strip()
    if stripped_answer and stripped_answer[-1] not in ".!?\"'`)]":
        issues.append("incomplete_answer")

    grounding_score = 0.0
    if answer.strip() and evidence_text.strip():
        vectors = embedding_model.encode([answer, evidence_text])
        grounding_score = float(vectors[0] @ vectors[1])
        if grounding_score < config.GROUNDING_THRESHOLD:
            issues.append("low_grounding")
    elif answer.strip():
        issues.append("low_grounding")

    passed = not issues
    verification = {"passed": passed, "issues": issues, "grounding_score": grounding_score}

    trace = list(state.get("trace", []))
    trace.append("verify")
    logger.info("Verification for %s: %s", state.get("question_id", "adhoc"), verification)

    update = {
        "verification": verification,
        "trace": trace,
        "timings": {**state.get("timings", {}), "verify": time.perf_counter() - started},
    }
    if not passed:
        update["verification_feedback"] = ", ".join(issues)
    return update


def clarify_node(state):
    trace = list(state.get("trace", []))
    trace.append("clarify")

    question_text = (
        "Could you share more detail so I can find the right documented steps? Please include "
        "the workspace ID, the exact object involved (schedule, connection or dashboard) and "
        "the exact error code or message you see."
    )

    response = {
        "classification": "requires_clarification",
        "answer": question_text,
        "sources": [],
        "confidence": 0.3,
        "requires_human": False,
        "reason": "The request does not include enough specific detail to match a documented troubleshooting path.",
        "clarification_question": question_text,
        "warnings": [],
    }
    return {"trace": trace, "final_response": response}


def safe_response_node(state):
    trace = list(state.get("trace", []))
    trace.append("safe_response")

    response = {
        "classification": "out_of_scope",
        "answer": (
            "This request is outside the OrbitDesk support knowledge base, or it asks for an "
            "action the support assistant cannot perform, such as issuing a refund or "
            "providing legal advice. Please contact the appropriate team directly for this "
            "request."
        ),
        "sources": [{"source_id": "KB-010", "passage": "Unsupported Actions"}],
        "confidence": 0.95,
        "requires_human": True,
        "reason": "The request matches an out-of-scope or unsupported-action pattern defined in KB-010.",
        "clarification_question": None,
        "warnings": [],
    }
    return {"trace": trace, "final_response": response}


def safe_failure_node(state):
    trace = list(state.get("trace", []))
    trace.append("safe_failure")

    issues = state.get("verification", {}).get("issues", [])
    reason = (
        "Verification failed after the maximum number of revision attempts: " + ", ".join(issues)
        if issues
        else "Verification failed after the maximum number of revision attempts."
    )

    response = {
        "classification": "safe_failure",
        "answer": (
            "I could not produce an answer that is fully supported by the OrbitDesk "
            "documentation for this question. Please rephrase with more detail, or contact "
            "support directly."
        ),
        "sources": [{"source_id": p["source_id"], "passage": p["section"]} for p in state.get("retrieved", [])[:2]],
        "confidence": 0.2,
        "requires_human": True,
        "reason": reason,
        "clarification_question": None,
        "warnings": ["verification_failed"] + issues,
    }
    return {"trace": trace, "final_response": response}


def finalize_node(state):
    trace = list(state.get("trace", []))
    trace.append("finalize")

    if state.get("final_response"):
        return {"trace": trace}

    target_classification = state.get("target_classification", "answerable")
    verification = state.get("verification", {})
    grounding_score = verification.get("grounding_score", 0.0)

    response = {
        "classification": target_classification,
        "answer": state.get("draft_answer", "").strip(),
        "sources": state.get("draft_sources", []),
        "confidence": round(min(0.95, max(0.3, grounding_score)), 2),
        "requires_human": target_classification == "requires_escalation",
        "reason": (
            "Answer generated from retrieved OrbitDesk documentation and passed verification."
            if target_classification == "answerable"
            else "Documented checks are exhausted for this issue; escalation is warranted per KB-008."
        ),
        "clarification_question": None,
        "warnings": [],
    }
    return {"trace": trace, "final_response": response}

