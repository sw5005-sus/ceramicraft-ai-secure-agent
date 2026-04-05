import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# --- 1. 环境与依赖隔离 (必须在最顶部) ---
# 关闭 MLflow 追踪和防止真实调用 OpenAI
os.environ["ENABLE_MLFLOW_TRACING"] = "false"
os.environ["OPENAI_API_KEY"] = ""

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
import ceramicraft_ai_secure_agent.service.agent_service as agent_service


class TestAgentService(unittest.TestCase):

    def setUp(self):
        """清理单例状态，确保测试隔离"""
        agent_service._graph = None
        agent_service._llm = None
        agent_service._loaded_prompt = None
        mock_storage.reset_mock()

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_graph")
    def test_assess_risk_skip_logic(self, mock_get_graph):
        """测试黑白名单跳过逻辑"""
        user_id = 888

        # 模拟用户在白名单中
        mock_storage.whitelist_storage.is_whitelisted.return_value = True

        result = agent_service.assess_risk(user_id)

        # 验证返回空字典且未触发图调用
        self.assertEqual(result, {})
        mock_get_graph.return_value.invoke.assert_not_called()

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_graph")
    def test_assess_risk_full_flow(self, mock_get_graph):
        """测试完整的风险评估链路"""
        user_id = 1001

        # 1. 模拟非跳过状态
        mock_storage.whitelist_storage.is_whitelisted.return_value = False
        mock_storage.blacklist_storage.is_blacklisted.return_value = False

        # 2. 模拟 LangGraph 的执行结果
        mock_invoke_res = {
            "score_result": {
                "risk_score": 0.45,
                "risk_level": "medium",
                "triggered_rules": ["high_order_count"],
                "fraud_probability": 0.3,
            },
            "recommendation": '{"recommended_action": "manual_review", "confidence": "high"}',
        }
        mock_get_graph.return_value.invoke.return_value = mock_invoke_res

        # 3. 执行测试
        result = agent_service.assess_risk(user_id)

        # 4. 断言验证返回结构
        self.assertEqual(result["user_id"], user_id)
        self.assertEqual(result["risk_level"], "medium")
        self.assertIn("high_order_count", result["triggered_rules"])

    def test_should_block_directly_high_score(self):
        """测试极高风险分数触发直接拦截逻辑"""
        # 构造高风险 state
        state = {
            "features": {},
            "rule_result": {"hits": [], "rule_score": 0.1},
            "ml_result": {"fraud_probability": 0.2},
            "score_result": {"risk_score": 0.9},  # 超过 0.85
        }

        self.assertTrue(agent_service._should_block_directly(state))

    @patch("ceramicraft_ai_secure_agent.service.agent_service._get_llm")
    def test_llm_judge_node_no_key_fallback(self, mock_get_llm):
        """测试在无 API KEY 时 LLM 节点是否降级"""
        # 确保环境变量为空
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            state = {
                "user_id": 1,
                "score_result": {"risk_score": 0.5},
                "rule_result": {"rule_score": 0.4, "hits": []},
                "ml_result": {"fraud_probability": 0.4},
                "features": {},
            }

            res = agent_service._llm_judge_node(state)

            # 验证是否返回了 fallback 推荐
            self.assertIn("manual_review", res["recommendation"])
            self.assertIn("LLM unavailable", res["recommendation"])
            mock_get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
