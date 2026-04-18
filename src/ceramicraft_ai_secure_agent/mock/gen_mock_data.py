import argparse
import time
from datetime import datetime

from ceramicraft_ai_secure_agent.rediscli import get_redis_client


def clear_mock_data(user_id: int):
    """Clear mock data for a specific user."""
    keys_to_delete = [
        f"u:{user_id}:rt",
        f"u:{user_id}:o",
        f"u:{user_id}:ip",
        f"u:{user_id}:sa",
        f"u:{user_id}:stat",
        f"u:{user_id}:ls",
        f"u:{user_id}:ra",
    ]
    today = datetime.now().strftime("%Y%m%d")
    keys_to_delete.append(f"u:{user_id}:stat:{today}")
    get_redis_client().delete(*keys_to_delete)
    get_redis_client().zrem("blacklist", str(user_id))
    get_redis_client().zrem("whitelist", str(user_id))
    get_redis_client().zrem("watchlist", str(user_id))
    print(f"Cleared mock data for user {user_id}.")


def gen_mock_normal(user_id: int):
    # register_date
    register_date = int(time.time() - 120 * 24 * 3600)  # Registered 120 days ago
    get_redis_client().set(f"u:{user_id}:rt", register_date)
    # order
    get_redis_client().zadd(
        f"u:{user_id}:o",
        {
            "ord-1001-1": time.time() - 2 * 60,  # 2 min ago
            "ord-1001-2": time.time() - 6 * 60,  # 6 min ago
        },
    )
    # ip
    get_redis_client().zadd(
        f"u:{user_id}:ip",
        {
            "1.1.1.1": time.time() - 10 * 60,  # Used 10 min ago
        },
    )
    # zipcode
    get_redis_client().sadd(f"u:{user_id}:ra", "100001")
    # order amount
    get_redis_client().hset(
        f"u:{user_id}:stat",
        mapping={
            "amount": 22000,
            "count": 2,
        },
    )
    # order amount today
    today = datetime.now().strftime("%Y%m%d")
    today_expire = int(
        datetime.strptime(today, "%Y%m%d").timestamp() + 24 * 3600 - time.time()
    )
    get_redis_client().hset(
        f"u:{user_id}:stat:{today}",
        mapping={
            "amount": 21000,
            "count": 2,
        },
    )
    get_redis_client().expire(f"u:{user_id}:stat:{today}", int(today_expire))
    print(f"Generated mock data for user {user_id}.")


def gen_mock_manual_review(user_id: int):
    now = time.time()
    today = datetime.now().strftime("%Y%m%d")
    today_expire = int(datetime.strptime(today, "%Y%m%d").timestamp() + 24 * 3600 - now)

    # registered 2 days ago
    get_redis_client().set(f"u:{user_id}:rt", int(now - 2 * 24 * 3600))

    # 8 orders within last 1h
    get_redis_client().zadd(
        f"u:{user_id}:o",
        {
            f"ord-{user_id}-1": now - 5 * 60,
            f"ord-{user_id}-2": now - 10 * 60,
            f"ord-{user_id}-3": now - 15 * 60,
            f"ord-{user_id}-4": now - 20 * 60,
            f"ord-{user_id}-5": now - 25 * 60,
            f"ord-{user_id}-6": now - 30 * 60,
            f"ord-{user_id}-7": now - 35 * 60,
            f"ord-{user_id}-8": now - 40 * 60,
        },
    )

    # 4 unique IPs in last 24h
    get_redis_client().zadd(
        f"u:{user_id}:ip",
        {
            "2.2.2.1": now - 5 * 60,
            "2.2.2.2": now - 15 * 60,
            "2.2.2.3": now - 25 * 60,
            "2.2.2.4": now - 35 * 60,
        },
    )

    # 3 receiving addresses
    get_redis_client().sadd(f"u:{user_id}:ra", "200001", "200002", "200003")

    # keep amount normal, avoid low-value/order-drop rules
    get_redis_client().hset(
        f"u:{user_id}:stat",
        mapping={
            "amount": 76000,
            "count": 8,
        },
    )

    get_redis_client().hset(
        f"u:{user_id}:stat:{today}",
        mapping={
            "amount": 78400,
            "count": 8,
        },
    )
    get_redis_client().expire(f"u:{user_id}:stat:{today}", today_expire)

    print(f"Generated manual_review mock data for user {user_id}.")


def gen_mock_block(user_id: int):
    now = time.time()
    today = datetime.now().strftime("%Y%m%d")
    today_expire = int(datetime.strptime(today, "%Y%m%d").timestamp() + 24 * 3600 - now)

    # registered 10 days ago
    get_redis_client().set(f"u:{user_id}:rt", int(now - 10 * 24 * 3600))

    orders = {}
    # 12 orders in last hour
    for i in range(12):
        orders[f"ord-{user_id}-{i + 1}"] = now - (i + 1) * 4 * 60

    # 13 more orders between 1h and 24h
    extra_minutes = [90, 120, 180, 240, 300, 420, 540, 660, 780, 900, 1020, 1140, 1320]
    for idx, minutes in enumerate(extra_minutes, start=13):
        orders[f"ord-{user_id}-{idx}"] = now - minutes * 60

    get_redis_client().zadd(f"u:{user_id}:o", orders)

    # 6 unique IPs
    get_redis_client().zadd(
        f"u:{user_id}:ip",
        {
            "3.3.3.1": now - 5 * 60,
            "3.3.3.2": now - 15 * 60,
            "3.3.3.3": now - 30 * 60,
            "3.3.3.4": now - 60 * 60,
            "3.3.3.5": now - 2 * 3600,
            "3.3.3.6": now - 6 * 3600,
        },
    )

    # 5 receiving addresses
    get_redis_client().sadd(
        f"u:{user_id}:ra",
        "300001",
        "300002",
        "300003",
        "300004",
        "300005",
    )

    # global avg = 100.00
    get_redis_client().hset(
        f"u:{user_id}:stat",
        mapping={
            "amount": 250000,
            "count": 25,
        },
    )

    # today avg = 15.00
    get_redis_client().hset(
        f"u:{user_id}:stat:{today}",
        mapping={
            "amount": 37500,
            "count": 25,
        },
    )
    get_redis_client().expire(f"u:{user_id}:stat:{today}", today_expire)

    print(f"Generated block mock data for user {user_id}.")


def gen_mock_watchlist(user_id: int):
    now = int(time.time())
    today = datetime.now().strftime("%Y%m%d")
    today_expire = int(datetime.strptime(today, "%Y%m%d").timestamp() + 24 * 3600 - now)

    r = get_redis_client()

    # 清掉本次行为相关数据，但不动别的用户
    r.delete(
        f"u:{user_id}:o",
        f"u:{user_id}:ip",
        f"u:{user_id}:ra",
        f"u:{user_id}:stat",
        f"u:{user_id}:stat:{today}",
        f"u:{user_id}:ls",
    )

    # 账号 20 天前注册
    # 命中新号高活跃时只需要 <= 30 天，不一定要特别新
    r.set(f"u:{user_id}:rt", now - 20 * 24 * 3600)

    # 最近 1h 恰好 6 单：刚好命中 new_account_high_activity = 0.16
    # 但远离 high_order_count_last_1h >= 10
    r.zadd(
        f"u:{user_id}:o",
        {
            f"ord-{user_id}-1": now - 5 * 60,
            f"ord-{user_id}-2": now - 12 * 60,
            f"ord-{user_id}-3": now - 20 * 60,
            f"ord-{user_id}-4": now - 28 * 60,
            f"ord-{user_id}-5": now - 36 * 60,
            f"ord-{user_id}-6": now - 50 * 60,
        },
    )

    # 3 个 IP + 3 个地址：只命中 ip_address_combination_anomaly = 0.12
    # 不到 5，不会命中 multiple_unique_ips
    r.zadd(
        f"u:{user_id}:ip",
        {
            "9.9.9.1": now - 5 * 60,
            "9.9.9.2": now - 25 * 60,
            "9.9.9.3": now - 45 * 60,
        },
    )

    r.sadd(
        f"u:{user_id}:ra",
        "900001",
        "900002",
        "900003",
    )

    # 金额尽量正常，避免 amount_drop / low_value
    # global avg = 85.00
    r.hset(
        f"u:{user_id}:stat",
        mapping={
            "amount": 51000,
            "count": 6,
        },
    )

    # today avg = 83.00
    r.hset(
        f"u:{user_id}:stat:{today}",
        mapping={
            "amount": 49800,
            "count": 6,
        },
    )
    r.expire(f"u:{user_id}:stat:{today}", today_expire)

    print(f"Generated watchlist mock data for user {user_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="user aiokafka send JSON data")
    parser.add_argument("--user_id", help="userid to mock")
    parser.add_argument("--op_type", help="clear|normal|manual_review|block|watchlist")
    args = parser.parse_args()
    user_id = int(args.user_id)
    op_type = args.op_type
    if op_type == "clear":
        clear_mock_data(user_id)
    elif op_type == "normal":
        gen_mock_normal(user_id)
