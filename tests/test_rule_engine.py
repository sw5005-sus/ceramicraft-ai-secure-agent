"""unittest-style tests for rule_engine.

Designed for VS Code unittest discovery.
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

MODULE_UNDER_TEST = "ceramicraft_ai_secure_agent.service.rule_engine"


class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.rule_engine = importlib.import_module(MODULE_UNDER_TEST)

    def test_safe_float(self):
        f = self.rule_engine._safe_float
        self.assertEqual(f("10"), 10.0)
        self.assertEqual(f(None), 0.0)
        self.assertEqual(f("abc"), 0.0)

    def test_no_rules_triggered(self):
        features = {}
        result = self.rule_engine.evaluate_rules(features)

        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["reasons"], [])

    def test_high_order_count_last_1h(self):
        features = {"order_count_last_1h": 12}

        result = self.rule_engine.evaluate_rules(features)

        self.assertIn("high_order_count_last_1h", result["hits"])
        self.assertGreater(result["rule_score"], 0)

    def test_multiple_rules_combined(self):
        features = {
            "order_count_last_1h": 12,
            "order_count_last_24h": 25,
            "unique_ip_count": 6,
        }

        result = self.rule_engine.evaluate_rules(features)

        self.assertIn("high_order_count_last_1h", result["hits"])
        self.assertIn("high_order_count_last_24h", result["hits"])
        self.assertIn("multiple_unique_ips", result["hits"])
        self.assertGreater(result["rule_score"], 0.5)

    def test_amount_drop_vs_history(self):
        features = {
            "avg_order_amount_today": 10,
            "avg_order_amount_global": 100,
            "order_count_last_24h": 10,
        }

        result = self.rule_engine.evaluate_rules(features)

        self.assertIn("amount_drop_vs_history", result["hits"])

    def test_new_account_high_activity(self):
        features = {
            "account_age_days": 10,
            "order_count_last_1h": 7,
        }

        result = self.rule_engine.evaluate_rules(features)

        self.assertIn("new_account_high_activity", result["hits"])

    def test_watchlist_user(self):
        features = {"last_status": "watchlist"}

        result = self.rule_engine.evaluate_rules(features)

        self.assertIn("watchlist_user", result["hits"])

    def test_score_capped_at_one(self):
        features = {
            "order_count_last_1h": 20,
            "order_count_last_24h": 50,
            "unique_ip_count": 10,
            "avg_order_amount_today": 5,
            "avg_order_amount_global": 100,
            "account_age_days": 1,
            "receive_address_count": 10,
            "last_status": "watchlist",
        }

        result = self.rule_engine.evaluate_rules(features)

        self.assertLessEqual(result["rule_score"], 1.0)

    def test_evaluate_rules_tool(self):
        feature_data = {"order_count_last_1h": 12}
        tool_input = {"features": feature_data}

        with patch.object(self.rule_engine, "evaluate_rules") as mock_eval:
            mock_eval.return_value = {"rule_score": 0.3, "hits": [], "reasons": []}

            result = self.rule_engine.evaluate_rules_tool.invoke(tool_input)

        self.assertEqual(result["rule_score"], 0.3)
        mock_eval.assert_called_once_with(feature_data)


if __name__ == "__main__":
    unittest.main()
