from orbitdesk_agent import routing


def _state(policy_violation=False, vague=False, escalation_signal=False, confidence=0.6):
    return {
        "triage": {
            "policy_violation": policy_violation,
            "matched_keywords": [],
            "vague": vague,
            "escalation_signal": escalation_signal,
        },
        "retrieval_confidence": confidence,
    }


def test_policy_violation_routes_to_safe_response():
    state = _state(policy_violation=True)
    assert routing.route_after_triage(state) == "safe_response"


def test_normal_question_routes_to_retrieve():
    state = _state()
    assert routing.route_after_triage(state) == "retrieve"


def test_very_low_confidence_routes_to_out_of_scope():
    state = _state(confidence=0.05)
    assert routing.classify_route(state) == "out_of_scope"
    assert routing.route_after_retrieval(state) == "safe_response"


def test_vague_question_routes_to_clarify_even_with_ok_confidence():
    state = _state(vague=True, confidence=0.6)
    assert routing.classify_route(state) == "clarify"
    assert routing.route_after_retrieval(state) == "clarify"


def test_low_but_nonzero_confidence_routes_to_clarify():
    state = _state(confidence=0.35)
    assert routing.classify_route(state) == "clarify"
    assert routing.route_after_retrieval(state) == "clarify"


def test_escalation_signal_routes_to_generate_as_escalation():
    state = _state(escalation_signal=True, confidence=0.6)
    assert routing.classify_route(state) == "escalate"
    assert routing.route_after_retrieval(state) == "generate"


def test_escalation_signal_takes_priority_over_vague():
    state = _state(escalation_signal=True, vague=True, confidence=0.6)
    assert routing.classify_route(state) == "escalate"


def test_confident_specific_question_routes_to_generate_as_answerable():
    state = _state(confidence=0.8)
    assert routing.classify_route(state) == "answer"
    assert routing.route_after_retrieval(state) == "generate"


def test_verification_pass_routes_to_finalize():
    state = {"verification": {"passed": True}, "generation_attempts": 1}
    assert routing.route_after_verification(state) == "finalize"


def test_verification_fail_first_attempt_routes_to_regenerate():
    state = {"verification": {"passed": False}, "generation_attempts": 1}
    assert routing.route_after_verification(state) == "regenerate"


def test_verification_fail_after_max_attempts_routes_to_safe_failure():
    state = {"verification": {"passed": False}, "generation_attempts": 2}
    assert routing.route_after_verification(state) == "safe_failure"


def test_loop_cannot_exceed_configured_max_attempts():
    state = {"verification": {"passed": False}, "generation_attempts": 99}
    assert routing.route_after_verification(state) == "safe_failure"
