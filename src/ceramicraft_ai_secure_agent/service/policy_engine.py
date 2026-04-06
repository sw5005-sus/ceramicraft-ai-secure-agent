"""Policy Engine.

Contains pure logic for risk assessment thresholds and decision-making policies.
Separating these from the integration layer allows for easier unit testing and
faster policy adjustments.
"""

from ceramicraft_ai_secure_agent.data.state import AssessmentState as Assessment
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# --- threshold const ---
THRESHOLD_RISK_SCORE_BLOCK = 0.85
THRESHOLD_RISK_SCORE_LLM_REQUIRED = 0.40
THRESHOLD_RISK_SCORE_SAFE = 0.20

THRESHOLD_RULE_ML_BOTH_HIGH_RULE = 0.70
THRESHOLD_RULE_ML_BOTH_HIGH_ML = 0.80

THRESHOLD_CONFLICT_DIFF = 0.35

# --- biz const ---
SCENARIO_ORDER_COUNT_BURST = 12
SCENARIO_NEW_ACCOUNT_DAYS = 7
SCENARIO_NEW_ACCOUNT_ORDER_LIMIT = 10
SCENARIO_ANOMALY_ADDR_COUNT = 5
SCENARIO_ANOMALY_IP_COUNT = 4


def should_block_directly(state: Assessment) -> bool:
    """block directly without LLM"""
    features = state["features"]
    rule_score = state["rule_result"].get("rule_score", 0.0)
    fraud_probability = state["ml_result"].get("fraud_probability", 0.0)
    risk_score = state["score_result"].get("risk_score", 0.0)
    triggered_rules = state["score_result"].get("triggered_rules", [])

    # 1. hish risk score
    if risk_score >= THRESHOLD_RISK_SCORE_BLOCK:
        return True

    # 2. high rule score and high fraud probability
    if (
        rule_score >= THRESHOLD_RULE_ML_BOTH_HIGH_RULE
        and fraud_probability >= THRESHOLD_RULE_ML_BOTH_HIGH_ML
    ):
        return True

    # 3. case: many orders in shorgt time with multiple unique IPs
    order_1h = features.get("order_count_last_1h", 0)
    if (
        "high_order_count_last_1h" in triggered_rules
        and "multiple_unique_ips" in triggered_rules
        and order_1h >= SCENARIO_ORDER_COUNT_BURST
    ):
        return True

    # 4. case: new account with many orders
    account_age = features.get("account_age_days", 0)
    if (
        account_age <= SCENARIO_NEW_ACCOUNT_DAYS
        and order_1h >= SCENARIO_NEW_ACCOUNT_ORDER_LIMIT
    ):
        return True

    # 5. case: abnormal receiving addresses and IPs
    ip_count = features.get("unique_ip_count", 0)
    addr_count = features.get("receive_address_count", 0)
    if (
        addr_count >= SCENARIO_ANOMALY_ADDR_COUNT
        and ip_count >= SCENARIO_ANOMALY_IP_COUNT
    ):
        return True

    return False


def need_llm_judgment(state: Assessment) -> bool:
    """need LLM judgment"""
    features = state["features"]
    rule_score = state["rule_result"].get("rule_score", 0.0)
    fraud_probability = state["ml_result"].get("fraud_probability", 0.0)
    risk_score = features.get("risk_score", 0.0)
    # low risk score, no triggered rules, low fraud probability, can skip LLM
    if (
        risk_score < THRESHOLD_RISK_SCORE_SAFE
        and rule_score == 0
        and fraud_probability < 0.15
    ):
        return False

    # medium to high risk score, LLM judgment required
    if risk_score >= THRESHOLD_RISK_SCORE_LLM_REQUIRED:
        return True

    # rule score and fraud probability are in conflict, LLM judgment required
    if abs(rule_score - fraud_probability) >= THRESHOLD_CONFLICT_DIFF:
        return True

    return False
