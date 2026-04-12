"""
DeepEval tests for the LLM adjudication layer of ai-secure-agent.

Goal:
- Evaluate recommendation quality at the _llm_judge_node(state) layer
- Avoid database-dependent user_id end-to-end instability
- Focus on schema, decision correctness, and faithfulness
"""

import json
import sys
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from deepeval import assert_test  # type: ignore

# ---------------------------------------------------------------------------
# 1. Isolate storage dependencies before importing agent_service
# ---------------------------------------------------------------------------

mock_storage = MagicMock()
sys.modules["ceramicraft_ai_secure_agent.rediscli"] = mock_storage
sys.modules["ceramicraft_ai_secure_agent.rediscli.blacklist_storage"] = mock_storage
sys.modules["ceramicraft_ai_secure_agent.rediscli.watchlist_storage"] = mock_storage
sys.modules["ceramicraft_ai_secure_agent.rediscli.whitelist_storage"] = mock_storage
sys.modules["ceramicraft_ai_secure_agent.rediscli.user_last_status_storage"] = (
    mock_storage
)
sys.modules["ceramicraft_ai_secure_agent.mysqlcli.risk_user_review_storage"] = (
    mock_storage
)

import ceramicraft_ai_secure_agent.service.agent_service as agent_service  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Controlled adjudication cases
#    IMPORTANT:
#    These cases should pass need_llm_judgment(...) == True
#    and should_block_directly(...) == False
# ---------------------------------------------------------------------------


class AdjudicationCase(TypedDict):
    name: str
    state: dict[str, Any]
    expected_action: str
    case_type: str


ADJUDICATION_CASES: list[AdjudicationCase] = [
    {
        "name": "watchlist_case",
        "state": {
            "user_id": 1001,
            "features": {
                "last_status": "allow",
                "order_count_last_1h": 3,
                "order_count_last_24h": 8,
                "unique_ip_count": 2,
                "avg_order_amount_today": 120.0,
                "account_age_days": 15,
            },
            "rule_result": {
                "rule_score": 0.40,
                "hits": ["moderate_order_velocity"],
            },
            "ml_result": {
                "fraud_probability": 0.45,
            },
            "score_result": {
                "risk_score": 0.40,
                "risk_level": "low",
                "fraud_probability": 0.45,
                "triggered_rules": ["moderate_order_velocity"],
            },
            "recommendation": "",
        },
        "expected_action": "watchlist",
    },
    {
        "name": "manual_review_case",
        "state": {
            "user_id": 1002,
            "features": {
                "last_status": "watchlist",
                "order_count_last_1h": 6,
                "order_count_last_24h": 20,
                "unique_ip_count": 4,
                "avg_order_amount_today": 260.0,
                "account_age_days": 7,
            },
            "rule_result": {
                "rule_score": 0.65,
                "hits": ["high_order_velocity", "multiple_ips"],
            },
            "ml_result": {
                "fraud_probability": 0.61,
            },
            "score_result": {
                "risk_score": 0.66,
                "risk_level": "medium",
                "fraud_probability": 0.61,
                "triggered_rules": ["high_order_velocity", "multiple_ips"],
            },
            "recommendation": "",
        },
        "expected_action": "manual_review",
    },
    {
        "name": "block_case",
        "state": {
            "user_id": 1003,
            "features": {
                "last_status": "watchlist",
                "order_count_last_1h": 12,
                "order_count_last_24h": 40,
                "unique_ip_count": 6,
                "avg_order_amount_today": 500.0,
                "account_age_days": 2,
            },
            "rule_result": {
                "rule_score": 0.82,
                "hits": [
                    "high_order_velocity",
                    "multiple_ips",
                    "new_account_large_orders",
                ],
            },
            "ml_result": {
                "fraud_probability": 0.79,
            },
            "score_result": {
                "risk_score": 0.81,
                "risk_level": "high",
                "fraud_probability": 0.79,
                "triggered_rules": [
                    "high_order_velocity",
                    "multiple_ips",
                    "new_account_large_orders",
                ],
            },
            "recommendation": "",
        },
        "expected_action": "block",
    },
]

# ---------------------------------------------------------------------------
# 3. Helper to format the structured input for DeepEval judge metrics
# ---------------------------------------------------------------------------


def build_case_input_text(state: dict[str, Any]) -> str:
    score = state["score_result"]
    return f"""
        Risk Score: {score["risk_score"]}
        Risk Level: {score["risk_level"]}
        Triggered Rules: {
        ", ".join(score["triggered_rules"]) if score["triggered_rules"] else "none"
    }
        Fraud Probability: {score["fraud_probability"]}
        Previous Status: {state["features"].get("last_status", "N/A")}
        Feature Snapshot: {
        json.dumps(state["features"], ensure_ascii=False, sort_keys=True)
    }
        """.strip()


def run_llm_judge(case: dict) -> str:
    """
    Call the internal adjudication node directly.

    We patch policy gates so the test always reaches the LLM adjudication path.
    """
    with (
        patch(
            "ceramicraft_ai_secure_agent.service.agent_service.should_block_directly",
            return_value=False,
        ),
        patch(
            "ceramicraft_ai_secure_agent.service.agent_service.need_llm_judgment",
            return_value=True,
        ),
    ):
        result = agent_service._llm_judge_node(case["state"])
        recommendation = result["recommendation"]

        # Normalize in case the service returns either str or Recommendation-like object
        if hasattr(recommendation, "to_json"):
            return recommendation.to_json()
        if isinstance(recommendation, dict):
            return json.dumps(recommendation, ensure_ascii=False)
        return str(recommendation)


def build_expected_output(case: dict) -> str:
    """
    Keep expected output lightweight.
    DeepEval will judge quality with criteria rather than exact-match.
    """
    return json.dumps(
        {
            "recommended_action": case["expected_action"],
            "confidence": "medium",
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 4. Build LLM test cases dynamically
# ---------------------------------------------------------------------------


def build_test_cases():
    test_cases = []
    for case in ADJUDICATION_CASES:
        actual_output = run_llm_judge(case)
        expected_output = build_expected_output(case)

        test_cases.append(
            LLMTestCase(
                input=build_case_input_text(case["state"]),
                actual_output=actual_output,
                expected_output=expected_output,
            )
        )
    return test_cases


TEST_CASES = build_test_cases()


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

schema_metric = GEval(
    name="Recommendation Schema Compliance",
    criteria=(
        "Evaluate whether the actual output is valid JSON and follows the "
        "recommendation schema. It should contain a recommended_action field "
        "with one of these values only: allow, watchlist, manual_review, block. "
        "If present, confidence should be low, medium, or high. Penalize any "
        "non-JSON output or invalid action labels."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.8,
)

decision_correctness_metric = GEval(
    name="Recommendation Decision Correctness",
    criteria=(
        "Evaluate whether the recommended action is justified by the provided "
        "risk signals. For this case family, both 'watchlist' and "
        "'manual_review' are acceptable conservative actions when evidence is "
        "moderate or ambiguous. Penalize 'allow' if the signals indicate "
        "meaningful suspicion, and penalize 'block' if the evidence is not "
        "strong enough for immediate blocking."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

faithfulness_metric = GEval(
    name="Recommendation Faithfulness",
    criteria=(
        "Evaluate whether every factual claim in the actual output is directly "
        "supported by the structured fraud signals in the input. "
        "Only assess factual grounding. Do NOT judge whether the recommended "
        "action is too strict, too lenient, or policy-optimal. "
        "Penalize only if the output invents evidence, events, user history, "
        "risk factors, or signals that are not explicitly present in the input."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.8,
)


# ---------------------------------------------------------------------------
# 6. Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_llm_judge_schema(test_case: LLMTestCase):
    assert_test(test_case, [schema_metric])


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_llm_judge_decision_correctness(test_case: LLMTestCase):
    assert_test(test_case, [decision_correctness_metric])


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_llm_judge_faithfulness(test_case: LLMTestCase):
    assert_test(test_case, [faithfulness_metric])
