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
from ceramicraft_ai_secure_agent.service.agent_service import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
