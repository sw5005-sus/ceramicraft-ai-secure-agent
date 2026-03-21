"""Unit tests for the rule engine service."""

from __future__ import annotations

import pytest

from ceramicraft_ai_secure_agent.service.rule_engine import evaluate_rules


class TestEvaluateRules:
    """Tests for rule_engine.evaluate_rules."""

    def _make_features(
        self,
        amount: float = 100.0,
        is_high_risk_country: float = 0.0,
        is_high_risk_category: float = 0.0,
        is_large_amount: float = 0.0,
    ) -> dict:
        import math

        return {
            "amount": amount,
            "is_high_risk_country": is_high_risk_country,
            "is_high_risk_category": is_high_risk_category,
            "amount_log": math.log1p(amount),
            "is_large_amount": is_large_amount,
        }

    # ------------------------------------------------------------------
    # No rules should trigger for a clean, low-value domestic transaction
    # ------------------------------------------------------------------
    def test_no_rules_triggered_for_clean_transaction(self):
        features = self._make_features(amount=100.0)
        result = evaluate_rules(features)

        assert result["rule_risk"] is False
        assert result["triggered_rules"] == []

    # ------------------------------------------------------------------
    # large_amount rule
    # ------------------------------------------------------------------
    def test_large_amount_rule_triggers(self):
        features = self._make_features(amount=6000.0, is_large_amount=1.0)
        result = evaluate_rules(features)

        assert result["rule_risk"] is True
        assert "large_amount" in result["triggered_rules"]

    # ------------------------------------------------------------------
    # high_risk_country rule
    # ------------------------------------------------------------------
    def test_high_risk_country_rule_triggers(self):
        features = self._make_features(amount=200.0, is_high_risk_country=1.0)
        result = evaluate_rules(features)

        assert result["rule_risk"] is True
        assert "high_risk_country" in result["triggered_rules"]

    # ------------------------------------------------------------------
    # high_risk_category rule
    # ------------------------------------------------------------------
    def test_high_risk_category_rule_triggers(self):
        features = self._make_features(amount=300.0, is_high_risk_category=1.0)
        result = evaluate_rules(features)

        assert result["rule_risk"] is True
        assert "high_risk_category" in result["triggered_rules"]

    # ------------------------------------------------------------------
    # Combined rule: large amount AND high-risk country
    # ------------------------------------------------------------------
    def test_combined_rule_triggers_with_large_amount_and_high_risk_country(self):
        features = self._make_features(
            amount=10000.0,
            is_high_risk_country=1.0,
            is_large_amount=1.0,
        )
        result = evaluate_rules(features)

        assert result["rule_risk"] is True
        assert "large_amount" in result["triggered_rules"]
        assert "high_risk_country" in result["triggered_rules"]
        assert "large_amount_in_high_risk_country" in result["triggered_rules"]

    # ------------------------------------------------------------------
    # Multiple independent rules can all trigger
    # ------------------------------------------------------------------
    def test_multiple_rules_can_trigger_simultaneously(self):
        features = self._make_features(
            amount=8000.0,
            is_high_risk_country=1.0,
            is_high_risk_category=1.0,
            is_large_amount=1.0,
        )
        result = evaluate_rules(features)

        assert result["rule_risk"] is True
        triggered = result["triggered_rules"]
        assert "large_amount" in triggered
        assert "high_risk_country" in triggered
        assert "high_risk_category" in triggered
        assert "large_amount_in_high_risk_country" in triggered

    # ------------------------------------------------------------------
    # Result always contains the expected keys
    # ------------------------------------------------------------------
    def test_result_contains_expected_keys(self):
        features = self._make_features()
        result = evaluate_rules(features)

        assert "triggered_rules" in result
        assert "rule_risk" in result
        assert isinstance(result["triggered_rules"], list)
        assert isinstance(result["rule_risk"], bool)
