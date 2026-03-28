"""Unit tests for the rule engine service."""

from __future__ import annotations

from ceramicraft_ai_secure_agent.service.rule_engine import evaluate_rules


class TestEvaluateRules:
    """Tests for rule_engine.evaluate_rules."""

    def _make_features(
        self,
        order_count_last_1h: float = 0.0,
        order_count_last_24h: float = 0.0,
        unique_ip_count: float = 0.0,
        avg_order_amount: float = 0.0,
        account_age_days: float = 0.0,
        device_count: float = 0.0,
    ) -> dict:

        return {
            "order_count_last_1h": order_count_last_1h,
            "order_count_last_24h": order_count_last_24h,
            "unique_ip_count": unique_ip_count,
            "avg_order_amount": avg_order_amount,
            "account_age_days": account_age_days,
            "device_count": device_count,
        }

    # ------------------------------------------------------------------
    # No rules should trigger for a clean, low-value domestic transaction
    # ------------------------------------------------------------------
    def test_no_rules_triggered_for_clean_transaction(self):
        features = self._make_features(avg_order_amount=100.0)
        result = evaluate_rules(features)

        assert result["rule_score"] == 0.0
        assert result["hits"] == []

    # ------------------------------------------------------------------
    # high_order_count_last_1h rule
    # ------------------------------------------------------------------
    def test_high_order_count_last_1h_rule_triggers(self):
        features = self._make_features(order_count_last_1h=12)
        result = evaluate_rules(features)

        assert result["rule_score"] > 0
        assert "high_order_count_last_1h" in result["hits"]

    # ------------------------------------------------------------------
    # multiple_unique_ips rule
    # ------------------------------------------------------------------
    def test_multiple_unique_ips_rule_triggers(self):
        features = self._make_features(unique_ip_count=10)
        result = evaluate_rules(features)

        assert result["rule_score"] > 0
        assert "multiple_unique_ips" in result["hits"]

    # ------------------------------------------------------------------
    # suspicious_order_pattern rule
    # ------------------------------------------------------------------
    def test_suspicious_order_pattern_rule_triggers(self):
        features = self._make_features(avg_order_amount=10.0, order_count_last_24h=20.0)
        result = evaluate_rules(features)

        assert result["rule_score"] > 0
        assert "suspicious_order_pattern" in result["hits"]

    # ------------------------------------------------------------------
    # account_age_days rule
    # ------------------------------------------------------------------
    def test_account_age_days_rule_country(self):
        features = self._make_features(
            account_age_days=10.0,
            order_count_last_1h=6.0,
        )
        result = evaluate_rules(features)

        assert result["rule_score"] > 0
        assert "new_account_high_activity" in result["hits"]

    # ------------------------------------------------------------------
    # Multiple independent rules can all trigger
    # ------------------------------------------------------------------
    def test_multiple_rules_can_trigger_simultaneously(self):
        features = self._make_features(
            device_count=5,
            unique_ip_count=10,
        )
        result = evaluate_rules(features)

        assert result["rule_score"] > 0
        triggered = result["hits"]
        assert "multiple_devices_and_ips" in triggered
        assert "multiple_unique_ips" in triggered

    # ------------------------------------------------------------------
    # Result always contains the expected keys
    # ------------------------------------------------------------------
    def test_result_contains_expected_keys(self):
        features = self._make_features()
        result = evaluate_rules(features)

        assert "hits" in result
        assert "rule_score" in result
        assert "reasons" in result
        assert isinstance(result["hits"], list)
        assert isinstance(result["rule_score"], float)
        assert isinstance(result["reasons"], list)
