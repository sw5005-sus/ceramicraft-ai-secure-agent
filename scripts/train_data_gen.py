from __future__ import annotations

import random
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_rng = secrets.SystemRandom()


@dataclass
class Sample:
    order_count_last_1h: int
    order_count_last_24h: int
    unique_ip_count: int
    avg_order_amount_global: float
    avg_order_amount_today: float
    account_age_days: int
    receive_address_count: int
    label: int


@dataclass
class Config:
    n_samples: int = 5000
    fraud_ratio: float = 0.16
    random_seed: int = 42
    output_dir: str = "data"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def weighted_choice(prob: float) -> bool:
    return _rng.random() < prob


def positive_int_from_gauss(mu: float, sigma: float, minimum: int = 0) -> int:
    return max(minimum, int(round(random.gauss(mu, sigma))))


def generate_normal_sample() -> Sample:
    """
    正常用户：
    - 1h/24h下单数低且平稳
    - IP 变化少
    - 账户年龄偏大
    - 历史客单价较稳定
    - 收货地址通常固定
    """
    order_count_last_1h = positive_int_from_gauss(0.8, 1.0, 0)
    order_count_last_24h = max(
        order_count_last_1h,
        order_count_last_1h + positive_int_from_gauss(2.8, 2.5, 0),
    )

    unique_ip_count = 1 if weighted_choice(0.82) else _rng.randint(2, 3)

    avg_order_amount_global = round(clamp(random.gauss(115, 38), 18, 480), 2)

    today_shift = random.gauss(0, 16)
    avg_order_amount_today = round(
        clamp(avg_order_amount_global + today_shift, 8, 520),
        2,
    )

    account_age_days = positive_int_from_gauss(240, 150, 1)
    receive_address_count = 1 if weighted_choice(0.72) else _rng.randint(2, 3)

    return Sample(
        order_count_last_1h=order_count_last_1h,
        order_count_last_24h=order_count_last_24h,
        unique_ip_count=unique_ip_count,
        avg_order_amount_global=avg_order_amount_global,
        avg_order_amount_today=avg_order_amount_today,
        account_age_days=account_age_days,
        receive_address_count=receive_address_count,
        label=0,
    )


def generate_fraud_sample() -> Sample:
    """
    刷单/异常用户模式：
    1) burst_small_orders: 短时间高频、小额下单
    2) ip_rotation: 同账号频繁更换 IP
    3) address_farm: 收货地址较多，像代收/刷单团伙
    4) aged_camouflage: 伪装成相对老账号，但行为异常
    """
    mode = _rng.choice(
        [
            "burst_small_orders",
            "ip_rotation",
            "address_farm",
            "aged_camouflage",
        ]
    )

    if mode == "burst_small_orders":
        order_count_last_1h = _rng.randint(7, 20)
        order_count_last_24h = _rng.randint(max(12, order_count_last_1h), 70)
        unique_ip_count = _rng.randint(1, 4)
        avg_order_amount_global = round(_rng.uniform(22, 80), 2)
        avg_order_amount_today = round(_rng.uniform(6, 36), 2)
        account_age_days = _rng.randint(1, 90)
        receive_address_count = _rng.randint(1, 3)

    elif mode == "ip_rotation":
        order_count_last_1h = _rng.randint(4, 14)
        order_count_last_24h = _rng.randint(max(10, order_count_last_1h), 45)
        unique_ip_count = _rng.randint(4, 9)
        avg_order_amount_global = round(_rng.uniform(40, 130), 2)
        avg_order_amount_today = round(_rng.uniform(18, 90), 2)
        account_age_days = _rng.randint(3, 140)
        receive_address_count = _rng.randint(1, 4)

    elif mode == "address_farm":
        order_count_last_1h = _rng.randint(3, 12)
        order_count_last_24h = _rng.randint(max(8, order_count_last_1h), 40)
        unique_ip_count = _rng.randint(1, 4)
        avg_order_amount_global = round(_rng.uniform(35, 110), 2)
        avg_order_amount_today = round(_rng.uniform(10, 70), 2)
        account_age_days = _rng.randint(7, 220)
        receive_address_count = _rng.randint(4, 9)

    else:  # aged_camouflage
        order_count_last_1h = _rng.randint(5, 16)
        order_count_last_24h = _rng.randint(max(12, order_count_last_1h), 55)
        unique_ip_count = _rng.randint(2, 6)
        avg_order_amount_global = round(_rng.uniform(55, 170), 2)
        avg_order_amount_today = round(_rng.uniform(8, 65), 2)
        account_age_days = _rng.randint(120, 520)
        receive_address_count = _rng.randint(2, 6)

    return Sample(
        order_count_last_1h=order_count_last_1h,
        order_count_last_24h=order_count_last_24h,
        unique_ip_count=unique_ip_count,
        avg_order_amount_global=avg_order_amount_global,
        avg_order_amount_today=avg_order_amount_today,
        account_age_days=account_age_days,
        receive_address_count=receive_address_count,
        label=1,
    )


def assign_label_with_noise(sample: Sample) -> int:
    """
    不直接用生成时的原始标签，而是重新打标签：
    - 让数据更像真实世界
    - 保留边界样本，提升模型泛化与解释价值
    """
    risk_score = 0.0

    if sample.order_count_last_1h >= 10:
        risk_score += 1.1
    elif sample.order_count_last_1h >= 6:
        risk_score += 0.6

    if sample.order_count_last_24h >= 30:
        risk_score += 0.9
    elif sample.order_count_last_24h >= 15:
        risk_score += 0.45

    if sample.unique_ip_count >= 5:
        risk_score += 1.0
    elif sample.unique_ip_count >= 3:
        risk_score += 0.45

    if sample.receive_address_count >= 5:
        risk_score += 1.0
    elif sample.receive_address_count >= 3:
        risk_score += 0.4

    if sample.account_age_days <= 14:
        risk_score += 0.9
    elif sample.account_age_days <= 60:
        risk_score += 0.45

    amount_ratio = sample.avg_order_amount_today / max(
        sample.avg_order_amount_global, 1
    )
    if sample.avg_order_amount_today <= 20:
        risk_score += 0.75
    elif sample.avg_order_amount_today <= 45:
        risk_score += 0.35

    if amount_ratio <= 0.45:
        risk_score += 0.9
    elif amount_ratio <= 0.7:
        risk_score += 0.4

    if (
        sample.order_count_last_1h >= 8
        and sample.avg_order_amount_today < sample.avg_order_amount_global * 0.55
    ):
        risk_score += 0.7

    if sample.unique_ip_count >= 4 and sample.receive_address_count >= 4:
        risk_score += 0.6

    risk_score += _rng.uniform(-0.7, 0.7)

    if risk_score >= 3.0:
        return 1
    if risk_score <= 1.35:
        return 0

    prob = (risk_score - 1.35) / (3.0 - 1.35)
    return 1 if _rng.random() < prob else 0


def generate_dataset(
    n_samples: int = 5000,
    fraud_ratio: float = 0.16,
    random_seed: int = 42,
) -> pd.DataFrame:
    random.seed(random_seed)

    rows: list[dict] = []

    for _ in range(n_samples):
        # 候选样本比例略高于目标比例，避免噪声重标后正样本不足
        if weighted_choice(fraud_ratio + 0.06):
            sample = generate_fraud_sample()
        else:
            sample = generate_normal_sample()

        row = asdict(sample)
        row["label"] = assign_label_with_noise(sample)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 轻度回采样，把最终正样本控制在目标附近
    current_ratio = df["label"].mean()
    target_fraud = int(n_samples * fraud_ratio)
    target_normal = n_samples - target_fraud

    if current_ratio < fraud_ratio * 0.75 or current_ratio > fraud_ratio * 1.3:
        fraud_df = df[df["label"] == 1]
        normal_df = df[df["label"] == 0]

        fraud_df = fraud_df.sample(
            n=target_fraud,
            replace=len(fraud_df) < target_fraud,
            random_state=random_seed,
        )
        normal_df = normal_df.sample(
            n=target_normal,
            replace=len(normal_df) < target_normal,
            random_state=random_seed,
        )

        df = pd.concat([fraud_df, normal_df], ignore_index=True)

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # 额外生成便于训练/评估的分桶字段（不建议放入模型训练）
    df["account_age_bucket"] = pd.cut(
        df["account_age_days"],
        bins=[-1, 30, 180, 365, 10000],
        labels=["new", "growing", "mature", "old"],
    )
    df["amount_shift_ratio"] = (
        df["avg_order_amount_today"] / df["avg_order_amount_global"].clip(lower=1)
    ).round(4)

    return df


def train_valid_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    train_parts = []
    valid_parts = []
    test_parts = []

    for _, sub_df in df.groupby("label"):
        sub_df = sub_df.sample(frac=1.0, random_state=random_seed)
        n = len(sub_df)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train_parts.append(sub_df.iloc[:n_train])
        valid_parts.append(sub_df.iloc[n_train : n_train + n_valid])
        test_parts.append(sub_df.iloc[n_train + n_valid :])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=random_seed)
    valid_df = pd.concat(valid_parts).sample(frac=1.0, random_state=random_seed)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=random_seed)

    train_df = train_df.assign(split="train")
    valid_df = valid_df.assign(split="valid")
    test_df = test_df.assign(split="test")

    return pd.concat([train_df, valid_df, test_df], ignore_index=True)


def main() -> None:
    config = Config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        n_samples=config.n_samples,
        fraud_ratio=config.fraud_ratio,
        random_seed=config.random_seed,
    )
    df = train_valid_test_split(df, random_seed=config.random_seed)

    output_path = output_dir / "ceramic_secure_agent_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print("\nOverall label distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nOverall fraud ratio:")
    print(round(df["label"].mean(), 4))
    print("\nFraud ratio by split:")
    print(df.groupby("split")["label"].mean().round(4))
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
