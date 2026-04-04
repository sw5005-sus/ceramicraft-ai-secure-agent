import json
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ceramicraft_ai_secure_agent.data.feature_columns import FEATURE_COLUMNS


def main() -> None:
    # ---------- 1. 配置加载 ----------
    data_path = os.environ.get("TRAINING_DATA_PATH", "data/fraud_training_data.csv")
    label_column = "label"
    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    max_iter = int(os.environ.get("MAX_ITER", "1000"))

    # MLflow 相关配置
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    use_mlflow = mlflow_uri is not None

    artifact_output_dir = Path("artifacts")
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 2. 数据处理 ----------
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # 严格按照 FEATURE_COLUMNS 提取特征，忽略 CSV 中多余的列（如 account_age_bucket 等）
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[label_column]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[label_column]

    # ---------- 3. 模型训练 ----------
    # 使用 .values 确保训练时不带特征名，彻底消除推理时的 UserWarning
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight="balanced",
    )
    model.fit(X_train.values, y_train)

    # ---------- 4. 模型评估 ----------
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # 控制台打印 (无论哪种模式)
    print("\n" + "=" * 30)
    print("      TRAINING REPORT      ")
    print("=" * 30)
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, zero_division=0))
    # ---------- 5. 持久化与上报 ----------
    if use_mlflow:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
        )

        with mlflow.start_run(
            run_name=os.environ.get("MLFLOW_RUN_NAME", "lr_baseline")
        ) as run:
            # 记录参数与指标
            mlflow.log_params({"max_iter": max_iter, "class_weight": "balanced"})
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # 记录模型 (带 Signature)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=os.environ.get(
                    "MLFLOW_REGISTERED_MODEL_NAME", "fraud-detector"
                ),
                input_example=X_train.head(3),
            )
            print(f"Successfully logged to MLflow: {mlflow_uri}")
    else:
        # 本地模式：导出模型和元数据
        print("MLflow tracking URI not set. Saving model locally...")
        model_path = artifact_output_dir / "model.joblib"
        joblib.dump(model, model_path)

        meta_path = artifact_output_dir / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(
                {"feature_columns": FEATURE_COLUMNS, "metrics": metrics}, f, indent=4
            )

        print(f"Local Mode: Model saved to {model_path}")
        print(f"Metadata saved to {meta_path}")

    # 生成评估图表 (通用)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Confusion Matrix")
    fig.savefig(artifact_output_dir / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
