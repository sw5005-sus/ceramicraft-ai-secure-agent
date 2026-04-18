import json
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

from ceramicraft_ai_secure_agent.data.state import AssessmentState

# --- 1. 环境与依赖隔离 (必须在最顶部) ---

# 拦截所有存储层模块，防止导入时尝试连接数据库
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

# --- 2. 导入被测模块 ---
# 注意：一定要在上面的 mock 注入后再导入
import ceramicraft_ai_secure_agent.service.agent_service as agent_service  # noqa: E402
from ceramicraft_ai_secure_agent.service.agent_service import (  # noqa: E402  # noqa: E402
    AllowAction,
    BlockAction,
    ManualReviewAction,
    WatchlistAction,
    _AssessmentState,
)


class TestAgentService(unittest.TestCase):
    def setUp(self):
        """清理单例状态，确保测试隔离"""
        os.environ["ENABLE_MLFLOW_TRACING"] = "false"
        os.environ["OPENAI_API_KEY"] = ""
        agent_service._graph = None
        agent_service._llm = None
        agent_service._loaded_prompt = None
        mock_storage.reset_mock()

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_graph")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.whitelist_storage")
    def test_assess_risk_skip_logic(self, mock_whitelist_storage, mock_get_graph):
        """测试黑白名单跳过逻辑"""
        user_id = 888

        # 模拟用户在白名单中
        mock_whitelist_storage.is_whitelisted.return_value = True

        result = agent_service.assess_risk(user_id)

        # 验证返回空字典且未触发图调用
        self.assertEqual(result, {})
        mock_get_graph.return_value.invoke.assert_not_called()

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_graph")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.whitelist_storage")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.blacklist_storage")
    def test_assess_risk_full_flow(
        self, mock_blacklist_storage, mock_whitelist_storage, mock_get_graph
    ):
        """测试完整的风险评估链路"""
        user_id = 1001

        # 1. 模拟非跳过状态
        mock_whitelist_storage.is_whitelisted.return_value = False
        mock_blacklist_storage.is_blacklisted.return_value = False

        # 2. 模拟 LangGraph 的执行结果
        rec_json = '{"recommended_action": "manual_review", "confidence": "high"}'
        mock_invoke_res = {
            "score_result": {
                "risk_score": 0.45,
                "risk_level": "medium",
                "triggered_rules": ["high_order_count"],
                "fraud_probability": 0.3,
            },
            "ml_result": {"fraud_probability": 0.3, "explanation": []},
            "recommendation": rec_json,
        }
        mock_get_graph.return_value.invoke.return_value = mock_invoke_res

        # 3. 执行测试
        result = agent_service.assess_risk(user_id)
        print("Assess Risk Result:", result)

        # 4. 断言验证返回结构
        self.assertEqual(result["user_id"], user_id)
        self.assertEqual(result["risk_level"], "medium")
        self.assertIn("high_order_count", result["triggered_rules"])

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_llm")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.need_llm_judgment")
    def test_llm_judge_node_no_key_fallback(self, mock_need_llm_judgment, mock_get_llm):
        """测试在无 API KEY 时 LLM 节点是否降级"""
        mock_need_llm_judgment.return_value = True
        # 确保环境变量为空
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            state: _AssessmentState = {
                "user_id": 1,
                "features": {},
                "rule_result": {"hits": [], "rule_score": 0.4},
                "ml_result": {"fraud_probability": 0.4},
                "score_result": {"risk_score": 0.5},
                "recommendation": "pending",
            }

            res = agent_service._llm_judge_node(state)

            # 验证是否返回了 fallback 推荐
            self.assertIn("manual_review", res["recommendation"])
            self.assertIn("LLM unavailable", res["recommendation"])
            mock_get_llm.assert_not_called()

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_llm")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.need_llm_judgment")
    def test_llm_invoke_failure_fallback(self, mock_need_llm_judgment, mock_get_llm):
        """测试 LLM 调用失败时是否正确降级"""
        mock_need_llm_judgment.return_value = True
        os.environ["OPENAI_API_KEY"] = "test_key"
        mock_get_llm.return_value.invoke.side_effect = Exception("LLM failure")

        state: _AssessmentState = {
            "user_id": 1,
            "features": {"last_status": "allow"},
            "rule_result": {"hits": [], "rule_score": 0.4},
            "ml_result": {"fraud_probability": 0.4},
            "score_result": {
                "risk_score": 0.5,
                "risk_level": "MEDIUM",
                "fraud_probability": 0.4,
                "triggered_rules": [],
            },
            "recommendation": "pending",
        }

        res = agent_service._llm_judge_node(state)

        self.assertIn("manual_review", res["recommendation"])
        self.assertIn("LLM unavailable", res["recommendation"])

    @patch("ceramicraft_ai_secure_agent.service.agent_service.should_block_directly")
    def test_llm_node_block_directly(self, mock_should_block_directly):
        """测试 LLM Judge Node 中直接 Block 的逻辑分支"""
        state: _AssessmentState = {
            "user_id": 1,
            "features": {"last_status": "allow", "order_count_last_1h": 15},
            "rule_result": {
                "hits": ["high_order_count_last_1h", "multiple_unique_ips"],
                "rule_score": 0.8,
            },
            "ml_result": {"fraud_probability": 0.85},
            "score_result": {
                "risk_score": 0.9,
                "risk_level": "HIGH",
                "triggered_rules": ["high_order_count_last_1h", "multiple_unique_ips"],
            },
            "recommendation": "pending",
        }
        mock_should_block_directly.return_value = True

        res = agent_service._llm_judge_node(state)

        self.assertIn("block", res["recommendation"])

    @patch("ceramicraft_ai_secure_agent.service.agent_service.need_llm_judgment")
    def test_llm_node_no_need_judgement(self, mock_need_llm_judgment):
        """测试 LLM Judge Node 中不需要 LLM 判断的分支"""
        state: _AssessmentState = {
            "user_id": 1,
            "features": {"last_status": "allow"},
            "rule_result": {"hits": [], "rule_score": 0.1},
            "ml_result": {"fraud_probability": 0.1},
            "score_result": {
                "risk_score": 0.1,
                "risk_level": "LOW",
                "triggered_rules": [],
            },
            "recommendation": "pending",
        }
        mock_need_llm_judgment.return_value = False
        res = agent_service._llm_judge_node(state)

        self.assertIn("allow", res["recommendation"])

    def test_llm_node_watchlist_directly(self):
        """测试 LLM Judge Node 中直接 Watchlist 的逻辑分支"""
        state: _AssessmentState = {
            "user_id": 1001,
            "features": {
                "last_status": "allow",
                "order_count_last_1h": 6,  # 触发 new_account_high_activity
            },
            "rule_result": {
                "hits": ["new_account_high_activity", "ip_address_combination_anomaly"],
                "rule_score": 0.28,  # 0.16 + 0.12
            },
            "ml_result": {
                "fraud_probability": 0.42,  # 中等，不触发 LLM
            },
            "score_result": {
                "risk_score": 0.36,  # ⭐关键：0.25~0.40之间
                "risk_level": "LOW",
                "triggered_rules": [
                    "new_account_high_activity",
                    "ip_address_combination_anomaly",
                ],
            },
            "recommendation": "pending",
        }

        res = agent_service._llm_judge_node(state)

        self.assertIn("watchlist", res["recommendation"])

    @patch("ceramicraft_ai_secure_agent.service.agent_service.user_last_status_storage")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.BlockAction.run")
    def test_action_node_execution(self, mock_block_run, mock_status_storage):
        from ceramicraft_ai_secure_agent.service.agent_service import _action_node

        state = AssessmentState(
            user_id=777,
            features={},
            rule_result={},
            ml_result={},
            score_result={"risk_score": 0.9},
            recommendation=json.dumps(
                {
                    "recommended_action": "block",
                    "reason": "test",
                    "analyst_summary": "s",
                    "confidence": "high",
                }
            ),
        )

        _action_node(state)

        mock_status_storage.set_user_last_status.assert_called_once_with(777, "block")
        mock_block_run.assert_called_once()

    @patch("ceramicraft_ai_secure_agent.service.agent_service.user_last_status_storage")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.ManualReviewAction.run")
    def test_action_node_fallback(self, mock_manual_review_run, mock_status_storage):
        from ceramicraft_ai_secure_agent.service.agent_service import _action_node

        state = AssessmentState(
            user_id=777,
            features={},
            rule_result={},
            ml_result={},
            score_result={"risk_score": 0.9},
            recommendation=json.dumps(
                {
                    "recommended_action": "unknown_action",
                    "reason": "test",
                    "analyst_summary": "s",
                    "confidence": "high",
                }
            ),
        )

        try:
            _action_node(state)
        except Exception as e:
            self.fail(f"_action_node raised an exception on unknown action: {e}")
        mock_status_storage.set_user_last_status.assert_called_once_with(
            777, "manual_review"
        )
        mock_manual_review_run.assert_called_once()

    def test_recommendation_parsing_fallback(self):
        from ceramicraft_ai_secure_agent.service.agent_service import (
            Recommendation,
        )

        bad_json = "{ 'invalid': json }"
        res = Recommendation.from_json(bad_json)
        self.assertEqual(res.recommended_action, "manual_review")
        self.assertEqual(res.reason, "LLM unavailable")

    @patch("ceramicraft_ai_secure_agent.service.agent_service.Path.exists")
    def test_get_loaded_prompt_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        res = agent_service._get_loaded_prompt()
        self.assertIn("Emergency fallback", res)

    @patch("builtins.open", new_callable=mock_open, read_data="Mocked Prompt Content")
    @patch("ceramicraft_ai_secure_agent.service.agent_service.Path.exists")
    def test_get_loaded_prompt_success(self, mock_exists, mock_file):
        """test normal file loading"""
        mock_exists.return_value = True
        res = agent_service._get_loaded_prompt()
        self.assertEqual(res, "Mocked Prompt Content")

    @patch("ceramicraft_ai_secure_agent.service.agent_service.extract_features_tool")
    def test_extract_features_node(self, mock_extract_features_tool):
        """Test feature extraction"""
        user_id = 123
        state: AssessmentState = {
            "user_id": user_id,
            "features": {},
            "rule_result": {},
            "ml_result": {},
            "score_result": {},
            "recommendation": "",
        }

        # setup mock return value for feature extraction
        mock_extract_features_tool.invoke.return_value = {
            "feature1": 1.0,
            "feature2": 2.0,
        }

        result = agent_service._extract_features_node(state)

        # verify features were added to the state
        self.assertIn("features", result)
        self.assertEqual(result["features"], {"feature1": 1.0, "feature2": 2.0})
        mock_extract_features_tool.invoke.assert_called_once_with({"user_id": user_id})

    @patch("ceramicraft_ai_secure_agent.service.agent_service.evaluate_rules_tool")
    def test_evaluate_rules_node(self, mock_evaluate_rules_tool):
        """Test the evaluate rules node functionality."""
        user_id = 123
        state: _AssessmentState = {
            "user_id": user_id,
            "features": {"feature1": 1.0, "feature2": 2.0},
            "rule_result": {},
            "ml_result": {},
            "score_result": {},
            "recommendation": "",
        }

        # Mock the return value of the evaluate_rules_tool
        mock_evaluate_rules_tool.invoke.return_value = {
            "rule_score": 0.7,
            "hits": ["rule1", "rule2"],
        }

        result = agent_service._evaluate_rules_node(state)

        # Verify the result structure
        self.assertIn("rule_result", result)
        self.assertEqual(result["rule_result"]["rule_score"], 0.7)
        self.assertIn("hits", result["rule_result"])
        self.assertEqual(result["rule_result"]["hits"], ["rule1", "rule2"])

        # Ensure the evaluate_rules_tool was called with the correct features
        mock_evaluate_rules_tool.invoke.assert_called_once_with(
            {"features": state["features"]}
        )

    @patch("ceramicraft_ai_secure_agent.service.agent_service.predict_tool")
    def test_predict_node(self, mock_predict_tool):
        """Test the predict node functionality."""
        user_id = 123
        state: _AssessmentState = {
            "user_id": user_id,
            "features": {"feature1": 1.0, "feature2": 2.0},
            "rule_result": {},
            "ml_result": {},
            "score_result": {},
            "recommendation": "",
        }

        # Mock the return value of the predict_tool
        mock_predict_tool.invoke.return_value = {
            "prediction": 1,
            "fraud_probability": 0.2,
        }

        result = agent_service._predict_node(state)

        # Verify the result structure
        self.assertIn("ml_result", result)
        self.assertEqual(result["ml_result"]["prediction"], 1)
        self.assertEqual(result["ml_result"]["fraud_probability"], 0.2)

        # Ensure the predict_tool was called with the correct features
        mock_predict_tool.invoke.assert_called_once_with(
            {"features": state["features"]}
        )

    @patch(
        "ceramicraft_ai_secure_agent.service.agent_service.risk_scoring.compute_score"
    )
    def test_compute_score_node(self, mock_compute_score):
        """Test the compute score node functionality."""
        user_id = 123
        state: _AssessmentState = {
            "user_id": user_id,
            "features": {"feature1": 1.0, "feature2": 2.0},
            "rule_result": {"rule_score": 0.7, "hits": ["rule1", "rule2"]},
            "ml_result": {"risk_score": 0.75, "fraud_probability": 0.2},
            "score_result": {},
            "recommendation": "",
        }

        # Mock the return value of the compute_score function
        mock_compute_score.return_value = {
            "risk_score": 0.65,
            "risk_level": "MEDIUM",
            "triggered_rules": ["rule1", "rule2"],
        }

        result = agent_service._compute_score_node(state)

        # Verify the result structure
        self.assertIn("score_result", result)
        self.assertEqual(result["score_result"]["risk_score"], 0.65)
        self.assertEqual(result["score_result"]["risk_level"], "MEDIUM")
        self.assertEqual(result["score_result"]["triggered_rules"], ["rule1", "rule2"])

        # Ensure the compute_score function was called with the correct parameters
        mock_compute_score.assert_called_once_with(
            state["rule_result"], state["ml_result"]
        )

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_loaded_prompt")
    def test_build_llm_prompt(self, mock_get_loaded_prompt):
        """Test the LLM prompt building functionality."""
        user_id = 123
        state: _AssessmentState = {
            "user_id": user_id,
            "features": {
                "last_status": "allow",
                "feature1": 1.0,
                "feature2": 2.0,
                "order_count_last_1h": 5.0,
                "order_count_last_24h": 10.0,
            },
            "rule_result": {},
            "ml_result": {},
            "score_result": {
                "risk_score": 0.75,
                "risk_level": "high",
                "triggered_rules": ["rule1", "rule2"],
                "fraud_probability": 0.5,
            },
            "recommendation": "",
        }

        mock_get_loaded_prompt.return_value = (
            "Risk Score: {risk_score}, Risk Level: {risk_level}, "
            "Triggered Rules: {triggered_rules}, "
            "Fraud Probability: {fraud_probability}, "
            "Previous Status: {previous_status}, "
            "Features: {feature_snapshot}"
        )

        prompt = agent_service._build_llm_prompt(state)

        expected_prompt = (
            "Risk Score: 0.7500, Risk Level: high, "
            "Triggered Rules: rule1, rule2, Fraud Probability: 0.5000, "
            "Previous Status: allow, Features: "
            "{'order_count_last_1h': 5.0, 'order_count_last_24h': 10.0}"
        )

        self.assertEqual(prompt, expected_prompt)

    @patch("ceramicraft_ai_secure_agent.service.agent_service.create_risk_user_review")
    @patch(
        "ceramicraft_ai_secure_agent.service.agent_service.blacklist_storage.add_blacklist"
    )
    def test_block_action_run(self, mock_add_blacklist, mock_create_risk_user_review):
        """Test the run method of BlockAction."""
        user_id = 777
        state: _AssessmentState = {
            "user_id": user_id,
            "features": {},
            "rule_result": {},
            "ml_result": {},
            "score_result": {},
            "recommendation": "",
        }
        action = BlockAction()
        action.run(state)

        mock_add_blacklist.assert_called_once_with(user_id=777)
        mock_create_risk_user_review.assert_called_once()

    @patch("ceramicraft_ai_secure_agent.service.agent_service.create_risk_user_review")
    def test_manual_review_action_run(self, mock_create_risk_user_review):
        """Test the run method of ManualReviewAction."""
        state = _AssessmentState(
            user_id=888,
            features={},
            rule_result={},
            ml_result={},
            score_result={},
            recommendation="",
        )
        action = ManualReviewAction()
        action.run(state)

        mock_create_risk_user_review.assert_called_once()

    @patch("ceramicraft_ai_secure_agent.service.agent_service.create_risk_user_review")
    @patch(
        "ceramicraft_ai_secure_agent.service.agent_service.watchlist_storage.add_watchlist"
    )
    def test_watchlist_action_run(
        self, mock_add_watchlist, mock_create_risk_user_review
    ):
        """Test the run method of WatchlistAction."""
        state = _AssessmentState(
            user_id=999,
            features={},
            rule_result={},
            ml_result={},
            score_result={},
            recommendation="",
        )
        action = WatchlistAction()
        action.run(state)

        mock_add_watchlist.assert_called_once_with(user_id=999)
        mock_create_risk_user_review.assert_called_once()

    def test_allow_action_run(self):
        """Test the run method of AllowAction."""
        state = _AssessmentState(
            user_id=1000,
            features={},
            rule_result={},
            ml_result={},
            score_result={},
            recommendation="",
        )
        action = AllowAction()
        action.run(state)  # Should not raise any exceptions


if __name__ == "__main__":
    unittest.main()
