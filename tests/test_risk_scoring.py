import unittest

from ceramicraft_ai_secure_agent.service.risk_scoring import compute_score


class TestRiskScoring(unittest.TestCase):
    def test_compute_score_high_risk(self):
        rule_result = {"rule_score": 0.8, "hits": ["rule1", "rule2"]}
        ml_result = {"fraud_probability": 0.9}

        result = compute_score(rule_result, ml_result)

        self.assertEqual(result["risk_score"], 0.865)
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["triggered_rules"], ["rule1", "rule2"])
        self.assertEqual(result["fraud_probability"], 0.9)

    def test_compute_score_medium_risk(self):
        rule_result = {"rule_score": 0.5, "hits": []}
        ml_result = {"fraud_probability": 0.4}

        result = compute_score(rule_result, ml_result)

        self.assertEqual(result["risk_score"], 0.435)
        self.assertEqual(result["risk_level"], "MEDIUM")
        self.assertEqual(result["triggered_rules"], [])
        self.assertEqual(result["fraud_probability"], 0.4)

    def test_compute_score_low_risk(self):
        rule_result = {"rule_score": 0.1, "hits": []}
        ml_result = {"fraud_probability": 0.1}

        result = compute_score(rule_result, ml_result)

        self.assertEqual(result["risk_score"], 0.1)
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["triggered_rules"], [])
        self.assertEqual(result["fraud_probability"], 0.1)

    def test_compute_score_no_rules_or_ml(self):
        rule_result = {"rule_score": 0.0, "hits": []}
        ml_result = {"fraud_probability": 0.0}

        result = compute_score(rule_result, ml_result)

        self.assertEqual(result["risk_score"], 0.0)
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["triggered_rules"], [])
        self.assertEqual(result["fraud_probability"], 0.0)
