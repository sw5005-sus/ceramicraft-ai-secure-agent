import sys
from unittest.mock import MagicMock, patch

# --- 1. 核心隔离：在导入 feature_service 之前拦截存储层 ---
# 这样可以防止导入时触发任何真实的 Redis/MySQL 连接逻辑
mock_redis = MagicMock()
sys.modules["ceramicraft_ai_secure_agent.rediscli"] = mock_redis

# --- 2. 导入被测模块 ---
import unittest
from ceramicraft_ai_secure_agent.service.feature_service import (
    extract_features,
    UserRequest,
    validate_and_update_feature_with_request,
)


class TestFeatureService(unittest.TestCase):

    def setUp(self):
        """每个测试用例运行前的初始化"""
        # 重置 mock 状态，避免用例间干扰
        mock_redis.reset_mock()

    def test_validate_and_update_feature_success(self):
        """测试正常更新用户 IP 的流程"""
        # 准备数据
        req = UserRequest(user_id=1001, ip="192.168.1.1", uri="/login", method="POST")

        # 模拟黑名单返回 False (未命中)
        mock_redis.blacklist_storage.is_blacklisted.return_value = False

        # 执行
        result = validate_and_update_feature_with_request(req)

        # 验证
        self.assertTrue(result)
        mock_redis.user_storage.update_user_ip.assert_called_once_with(
            user_id=1001, ip_address="192.168.1.1"
        )

    @patch("ceramicraft_ai_secure_agent.service.feature_service.datetime")
    def test_extract_features_logic(self, mock_datetime):
        """测试特征提取的并发调用逻辑"""
        user_id = 2002

        # 1. 模拟时间，确保 account_age_days 计算可控
        # 假设现在是 1712232000 (2024-04-04)，注册时间是 2天前
        now_ts = 1712232000.0
        mock_datetime.now.return_value.timestamp.return_value = now_ts

        # 2. 模拟各个存储层的方法返回值
        mock_redis.order_storage.count_order_by_time.return_value = 5
        mock_redis.user_storage.count_user_ip.return_value = 2
        mock_redis.order_storage.get_global_avg_order_amount.return_value = (
            15000  # 代表 150.00
        )
        mock_redis.order_storage.get_today_order_avg_amount.return_value = (
            5000  # 代表 50.00
        )
        mock_redis.user_storage.get_user_register_time.return_value = now_ts - (
            2 * 24 * 3600
        )
        mock_redis.order_storage.count_user_receiver_address.return_value = 3
        mock_redis.user_last_status_storage.get_user_last_status.return_value = "allow"

        # 3. 执行
        results = extract_features(user_id)

        # 4. 断言验证各字段计算结果
        self.assertEqual(results["order_count_last_1h"], 5)
        self.assertEqual(results["unique_ip_count"], 2)
        self.assertEqual(results["avg_order_amount_global"], 150.0)  # 15000 / 100
        self.assertEqual(results["account_age_days"], 2)  # math.ceil 计算出的天数
        self.assertEqual(results["last_status"], "allow")

    def test_validate_blacklisted_user(self):
        """测试当用户在黑名单时应跳过更新"""
        req = UserRequest(user_id=999, ip="1.1.1.1", uri="/", method="GET")
        mock_redis.blacklist_storage.is_blacklisted.return_value = True

        result = validate_and_update_feature_with_request(req)

        self.assertFalse(result)
        # 确保 update_user_ip 没被调用
        mock_redis.user_storage.update_user_ip.assert_not_called()
