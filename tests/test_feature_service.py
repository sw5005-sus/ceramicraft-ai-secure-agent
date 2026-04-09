import unittest
from unittest.mock import patch

from ceramicraft_ai_secure_agent.service.feature_service import (  # noqa: E402
    UserRequest,
    extract_features,
    validate_and_update_feature_with_request,
)


class TestFeatureService(unittest.TestCase):
    @patch("ceramicraft_ai_secure_agent.service.feature_service.blacklist_storage")
    @patch("ceramicraft_ai_secure_agent.service.feature_service.user_storage")
    def test_validate_and_update_feature_success(
        self, mock_user_storage, mock_blacklist_storage
    ):
        """测试正常更新用户 IP 的流程"""
        # 准备数据
        req = UserRequest(
            user_id=1001, ip="192.168.1.1", uri="/customer/login", method="POST"
        )

        # 模拟黑名单返回 False (未命中)
        mock_blacklist_storage.is_blacklisted.return_value = False

        # 执行
        result = validate_and_update_feature_with_request(req)

        # 验证
        self.assertTrue(result)
        mock_user_storage.update_user_ip.assert_called_once_with(
            user_id=1001, ip_address="192.168.1.1"
        )

    @patch("ceramicraft_ai_secure_agent.service.feature_service.order_storage")
    @patch("ceramicraft_ai_secure_agent.service.feature_service.user_storage")
    @patch(
        "ceramicraft_ai_secure_agent.service.feature_service.user_last_status_storage"
    )
    @patch("ceramicraft_ai_secure_agent.service.feature_service.datetime")
    def test_extract_features_logic(
        self,
        mock_datetime,
        mock_user_last_status_storage,
        mock_user_storage,
        mock_order_storage,
    ):
        """测试特征提取的并发调用逻辑"""
        user_id = 2002

        # 1. 模拟时间，确保 account_age_days 计算可控
        # 假设现在是 1712232000 (2024-04-04)，注册时间是 2天前
        now_ts = 1712232000.0
        mock_datetime.now.return_value.timestamp.return_value = now_ts

        # 2. 模拟各个存储层的方法返回值
        mock_order_storage.count_order_by_time.return_value = 5
        mock_user_storage.count_user_ip.return_value = 2
        mock_order_storage.get_global_avg_order_amount.return_value = (
            15000  # 代表 150.00
        )
        mock_order_storage.get_today_order_avg_amount.return_value = 5000  # 代表 50.00
        mock_user_storage.get_user_register_time.return_value = now_ts - (2 * 24 * 3600)
        mock_order_storage.count_user_receiver_address.return_value = 3
        mock_user_last_status_storage.get_user_last_status.return_value = "allow"

        # 3. 执行
        results = extract_features(user_id)

        # 4. 断言验证各字段计算结果
        self.assertEqual(results["order_count_last_1h"], 5)
        self.assertEqual(results["unique_ip_count"], 2)
        self.assertEqual(results["avg_order_amount_global"], 150.0)  # 15000 / 100
        self.assertEqual(results["account_age_days"], 2)  # math.ceil 计算出的天数
        self.assertEqual(results["last_status"], "allow")

    @patch("ceramicraft_ai_secure_agent.service.feature_service.blacklist_storage")
    @patch("ceramicraft_ai_secure_agent.service.feature_service.user_storage")
    def test_validate_blacklisted_user(self, mock_user_storage, mock_blacklist_storage):
        """blacklist user should be blocked and IP should still be updated"""
        req = UserRequest(
            user_id=999,
            ip="1.1.1.1",
            uri="/test-ms/v1/customer/test-api",
            method="POST",
        )
        mock_blacklist_storage.is_blacklisted.return_value = True

        result = validate_and_update_feature_with_request(req)

        self.assertFalse(result)
        mock_user_storage.update_user_ip.assert_called_once()

    @patch("ceramicraft_ai_secure_agent.service.feature_service.blacklist_storage")
    @patch("ceramicraft_ai_secure_agent.service.feature_service.user_storage")
    def test_blacklisted_user_withoutblock(
        self, mock_user_storage, mock_blacklist_storage
    ):
        """blacklist user is not blocked but updated when method is GET"""
        req = UserRequest(
            user_id=999,
            ip="1.1.1.1",
            uri="/test-ms/v1/customer/test-api",
            method="GET",
        )
        mock_blacklist_storage.is_blacklisted.return_value = True

        result = validate_and_update_feature_with_request(req)

        self.assertTrue(result)
        mock_user_storage.update_user_ip.assert_called_once()

    def test_non_customer_request(self):
        """skip for non customer request"""
        req = UserRequest(
            user_id=999,
            ip="1.1.1.1",
            uri="/test-ms/v1/merchant/test-api",
            method="POST",
        )

        result = validate_and_update_feature_with_request(req)

        self.assertTrue(result)
