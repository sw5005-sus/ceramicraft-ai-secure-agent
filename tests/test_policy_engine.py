import unittest

from ceramicraft_ai_secure_agent.data.state import AssessmentState
from ceramicraft_ai_secure_agent.service.policy_engine import (
    need_llm_judgment,
    should_block_directly,
)


class TestPolicyEngine(unittest.TestCase):
    def test_should_block_directly(self):
        # high score block
        state_block = AssessmentState(
            user_id=1001,
            features={},
            rule_result={"rule_score": 0.1},
            ml_result={"fraud_probability": 0.1},
            score_result={"risk_score": 0.9},
            recommendation="",
        )
        self.assertTrue(should_block_directly(state_block))

        # normal state
        state_safe: AssessmentState = {
            "user_id": 1001,
            "features": {"account_age_days": 100},
            "rule_result": {"rule_score": 0.0, "hits": []},
            "ml_result": {"fraud_probability": 0.05},
            "score_result": {"risk_score": 0.1},
            "recommendation": "",
        }
        self.assertFalse(should_block_directly(state_safe))

        state_burst: AssessmentState = {
            "user_id": 1002,
            "features": {"order_count_last_1h": 12},
            "rule_result": {"rule_score": 0.4},
            "ml_result": {"fraud_probability": 0.4},
            "score_result": {
                "risk_score": 0.5,
                "triggered_rules": ["high_order_count_last_1h", "multiple_unique_ips"],
            },
            "recommendation": "",
        }
        self.assertTrue(should_block_directly(state_burst))  #

        # case 4：new account with high order count in last 1h
        state_new_acc: AssessmentState = {
            "user_id": 1003,
            "features": {"account_age_days": 3, "order_count_last_1h": 10},
            "rule_result": {"rule_score": 0.2},
            "ml_result": {"fraud_probability": 0.2},
            "score_result": {"risk_score": 0.3},
            "recommendation": "",
        }
        self.assertTrue(should_block_directly(state_new_acc))  #

        # case 6：abnormal geo location with multiple unique IPs
        state_geo_anomaly: AssessmentState = {
            "user_id": 1004,
            "features": {"receive_address_count": 5, "unique_ip_count": 4},
            "rule_result": {"rule_score": 0.2},
            "ml_result": {"fraud_probability": 0.2},
            "score_result": {"risk_score": 0.3},
            "recommendation": "",
        }
        self.assertTrue(should_block_directly(state_geo_anomaly))

    def test_need_llm_judgment(self):
        # confliect state with high risk score but low rule/ml score
        state_conflict = AssessmentState(
            user_id=1001,
            features={},
            rule_result={"rule_score": 0.8},
            ml_result={"fraud_probability": 0.1},
            score_result={"risk_score": 0.3},
            recommendation="",
        )
        self.assertTrue(need_llm_judgment(state_conflict))


if __name__ == "__main__":
    unittest.main()
