import json


class OrderItemInfo:
    def __init__(self, product_id, product_name, quantity, price):
        self.product_id = product_id
        self.product_name = product_name
        self.quantity = quantity
        self.price = price

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            product_id=data["product_id"],
            product_name=data["product_name"],
            quantity=data["quantity"],
            price=data["price"],
        )


class OrderMessage:
    def __init__(
        self,
        user_id: int,
        order_id: str,
        receiver_zip_code: int,
        total_amount: float = 0.0,
    ):
        self.user_id = user_id  # 下单用户
        self.order_id = order_id  # 订单ID
        self.receiver_zip_code = receiver_zip_code  # 收货人邮政编码
        self.total_amount = total_amount  # 订单总金额

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            user_id=data["user_id"],
            order_id=data["order_id"],
            receiver_zip_code=data["receiver_zip_code"],
            total_amount=data["total_amount"],
        )

    @classmethod
    def from_json(cls, json_data: str):
        data = json.loads(json_data)
        return cls.from_dict(data)


class UserActivatedEvent:
    def __init__(self, user_id: int, activated_time: int) -> None:
        self.user_id = user_id
        self.activated_time = activated_time

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            user_id=data["user_id"],
            activated_time=data["activated_time"],
        )

    @classmethod
    def from_json(cls, json_data: str):
        data = json.loads(json_data)
        return cls.from_dict(data)
