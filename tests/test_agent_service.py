"""Unit tests for the LangGraph-based agent service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ceramicraft_ai_secure_agent.service import agent_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TRANSACTION: dict = {
    "transaction_id": "txn_test_001",
    "amount": 200.0,
    "merchant_category": "retail",
    "country": "US",
    "user_id": "user_42",
    "merchant": "TestMerchant",
}

_HIGH_RISK_TRANSACTION: dict = {
    "transaction_id": "txn_test_002",
    "amount": 12500.0,
    "merchant_category": "unknown",
    "country": "NG",
    "user_id": "user_99",
    "merchant": "ShadyMerchant",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssessRisk:
    """Tests for agent_service.assess_risk via the LangGraph pipeline."""

    def _run(self, transaction: dict) -> dict:
        """Run assess_risk with the LLM node patched out (no API key needed)."""
        # Patch the LLM so tests run without a real OPENAI_API_KEY
        with patch.dict("os.environ", {}, clear=False):
            # Ensure OPENAI_API_KEY is absent so the fallback path is exercised
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            return agent_service.assess_risk(transaction)

    def test_returns_expected_keys(self):
        result = self._run(_SAMPLE_TRANSACTION)
        expected_keys = {
            "transaction_id",
            "risk_score",
            "risk_level",
            "triggered_rules",
            "fraud_probability",
            "recommendation",
        }
        assert expected_keys == set(result.keys())

    def test_transaction_id_preserved(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert result["transaction_id"] == "txn_test_001"

    def test_risk_score_in_range(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_level_is_valid(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert result["risk_level"] in {"HIGH", "MEDIUM", "LOW"}

    def test_fraud_probability_in_range(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_recommendation_is_string(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_triggered_rules_is_list(self):
        result = self._run(_SAMPLE_TRANSACTION)
        assert isinstance(result["triggered_rules"], list)

    def test_high_risk_transaction_triggers_rules(self):
        result = self._run(_HIGH_RISK_TRANSACTION)
        triggered = result["triggered_rules"]
        assert "large_amount" in triggered
        assert "high_risk_country" in triggered
        assert "large_amount_in_high_risk_country" in triggered

    def test_high_risk_transaction_has_high_score(self):
        result = self._run(_HIGH_RISK_TRANSACTION)
        # High-risk transactions should produce at least a MEDIUM level
        assert result["risk_level"] in {"HIGH", "MEDIUM"}

    def test_unknown_transaction_id_defaults(self):
        txn = {"amount": 50.0, "country": "US", "merchant_category": "retail"}
        result = self._run(txn)
        assert result["transaction_id"] == "unknown"

    def test_llm_recommendation_used_when_api_key_present(self):
        """
        When OPENAI_API_KEY is set in the environment,
        the LLM node should be invoked.
        """
        fake_response = MagicMock()
        fake_response.content = "LLM-generated recommendation."

        with patch(
            "ceramicraft_ai_secure_agent.service.agent_service.ChatOpenAI"
        ) as mock_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = fake_response
            mock_cls.return_value = mock_llm

            # Reset the module-level singleton so _get_llm() constructs a new one
            agent_service._llm = None

            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
                result = agent_service.assess_risk(_SAMPLE_TRANSACTION)

        assert result["recommendation"] == "LLM-generated recommendation."
        mock_llm.invoke.assert_called_once()
