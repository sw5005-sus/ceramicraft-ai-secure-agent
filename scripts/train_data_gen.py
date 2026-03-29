from __future__ import annotations

import random
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Sample:
    order_count_last_1h: int
    order_count_last_24h: int
    unique_ip_count: int
    avg_order_amount: float
    account_age_days: int
    device_count: int
    label: int


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def weighted_choice(prob: float) -> bool:
    return secrets.SystemRandom().random() < prob


def generate_normal_sample() -> Sample:
    """
    Normal user distribution:
    - Lower order frequency
    - Stable IP/device usage
    - Older account age
    - Moderate average order amount
    """
    order_count_last_1h = max(0, int(random.gauss(1.5, 1.2)))
    order_count_last_24h = max(
        order_count_last_1h, int(random.gauss(4, 3)) + order_count_last_1h
    )

    unique_ip_count = (
        1 if weighted_choice(0.8) else secrets.SystemRandom().randint(2, 3)
    )
    device_count = 1 if weighted_choice(0.75) else secrets.SystemRandom().randint(2, 3)

    avg_order_amount = round(clamp(random.gauss(95, 35), 15, 400), 2)
    account_age_days = max(1, int(random.gauss(220, 120)))

    return Sample(
        order_count_last_1h=order_count_last_1h,
        order_count_last_24h=order_count_last_24h,
        unique_ip_count=unique_ip_count,
        avg_order_amount=avg_order_amount,
        account_age_days=account_age_days,
        device_count=device_count,
        label=0,
    )


def generate_fraud_sample() -> Sample:
    """
    Fraudulent/bot user distribution:
    - High order frequency
    - Multiple IPs / multiple devices
    - Relatively new accounts
    - Smaller order amounts are more common
    But still add noise to avoid "rule replicators"
    """
    mode = secrets.SystemRandom().choice(["burst", "multi_ip", "low_amount", "mixed"])

    if mode == "burst":
        order_count_last_1h = secrets.SystemRandom().randint(8, 20)
        order_count_last_24h = secrets.SystemRandom().randint(
            max(15, order_count_last_1h), 60
        )
        unique_ip_count = secrets.SystemRandom().randint(2, 5)
        device_count = secrets.SystemRandom().randint(1, 3)
        avg_order_amount = round(secrets.SystemRandom().uniform(12, 60), 2)
        account_age_days = secrets.SystemRandom().randint(1, 90)

    elif mode == "multi_ip":
        order_count_last_1h = secrets.SystemRandom().randint(4, 12)
        order_count_last_24h = secrets.SystemRandom().randint(
            max(10, order_count_last_1h), 40
        )
        unique_ip_count = secrets.SystemRandom().randint(4, 8)
        device_count = secrets.SystemRandom().randint(2, 5)
        avg_order_amount = round(secrets.SystemRandom().uniform(15, 90), 2)
        account_age_days = secrets.SystemRandom().randint(1, 120)

    elif mode == "low_amount":
        order_count_last_1h = secrets.SystemRandom().randint(5, 15)
        order_count_last_24h = secrets.SystemRandom().randint(
            max(12, order_count_last_1h), 50
        )
        unique_ip_count = secrets.SystemRandom().randint(1, 4)
        device_count = secrets.SystemRandom().randint(1, 3)
        avg_order_amount = round(secrets.SystemRandom().uniform(5, 25), 2)
        account_age_days = secrets.SystemRandom().randint(1, 180)

    else:  # mixed
        order_count_last_1h = secrets.SystemRandom().randint(6, 18)
        order_count_last_24h = secrets.SystemRandom().randint(
            max(15, order_count_last_1h), 70
        )
        unique_ip_count = secrets.SystemRandom().randint(3, 8)
        device_count = secrets.SystemRandom().randint(2, 5)
        avg_order_amount = round(secrets.SystemRandom().uniform(8, 45), 2)
        account_age_days = secrets.SystemRandom().randint(1, 60)

    return Sample(
        order_count_last_1h=order_count_last_1h,
        order_count_last_24h=order_count_last_24h,
        unique_ip_count=unique_ip_count,
        avg_order_amount=avg_order_amount,
        account_age_days=account_age_days,
        device_count=device_count,
        label=1,
    )


def assign_label_with_noise(sample: Sample) -> int:
    """
    Do not directly use the original labels from the generation functions,
    but instead re-label based on the risk score + noise.
    This makes it more like the real world and
    avoids the model simply replicating the rules mechanically.
    """
    risk_score = 0.0

    if sample.order_count_last_1h >= 10:
        risk_score += 1.2
    elif sample.order_count_last_1h >= 6:
        risk_score += 0.6

    if sample.order_count_last_24h >= 30:
        risk_score += 1.0
    elif sample.order_count_last_24h >= 15:
        risk_score += 0.5

    if sample.unique_ip_count >= 5:
        risk_score += 1.2
    elif sample.unique_ip_count >= 3:
        risk_score += 0.6

    if sample.device_count >= 4:
        risk_score += 0.8
    elif sample.device_count >= 2:
        risk_score += 0.3

    if sample.avg_order_amount < 20:
        risk_score += 1.0
    elif sample.avg_order_amount < 50:
        risk_score += 0.4

    if sample.account_age_days <= 30:
        risk_score += 1.0
    elif sample.account_age_days <= 90:
        risk_score += 0.5

    # Add noise: simulate uncertainty in the real world
    risk_score += secrets.SystemRandom().uniform(-0.8, 0.8)

    # Fuzzy interval: create boundary samples
    if risk_score >= 3.0:
        return 1
    if risk_score <= 1.4:
        return 0

    # The intermediate area is determined by probability, not hard thresholds
    prob = (risk_score - 1.4) / (3.0 - 1.4)
    return 1 if secrets.SystemRandom().random() < prob else 0


def generate_dataset(
    n_samples: int = 5000,
    fraud_ratio: float = 0.18,
    random_seed: int = 42,
) -> pd.DataFrame:
    random.seed(random_seed)

    rows: list[dict] = []
    fraud_target = int(n_samples * fraud_ratio)

    for _ in range(n_samples):
        # First control the distribution of "candidate samples"
        if len([r for r in rows if r["label"] == 1]) < fraud_target and weighted_choice(
            fraud_ratio + 0.08
        ):
            sample = generate_fraud_sample()
        else:
            sample = generate_normal_sample()

        # Re-label the sample
        label = assign_label_with_noise(sample)
        row = asdict(sample)
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows)

    # If the final fraud ratio deviates too much, perform a slight resampling correction
    current_ratio = df["label"].mean()
    if current_ratio < fraud_ratio * 0.7 or current_ratio > fraud_ratio * 1.4:
        fraud_df = df[df["label"] == 1]
        normal_df = df[df["label"] == 0]

        target_fraud = int(n_samples * fraud_ratio)
        target_normal = n_samples - target_fraud

        fraud_df = fraud_df.sample(
            n=min(len(fraud_df), target_fraud),
            replace=len(fraud_df) < target_fraud,
            random_state=random_seed,
        )
        normal_df = normal_df.sample(
            n=min(len(normal_df), target_normal),
            replace=len(normal_df) < target_normal,
            random_state=random_seed,
        )

        df = pd.concat([fraud_df, normal_df], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        n_samples=5000,
        fraud_ratio=0.18,
        random_seed=42,
    )

    output_path = output_dir / "fraud_training_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print("\nClass distribution:")
    print(df["label"].value_counts(dropna=False))
    print("\nFraud ratio:")
    print(round(df["label"].mean(), 4))
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
