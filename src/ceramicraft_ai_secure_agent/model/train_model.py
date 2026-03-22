import json
from pathlib import Path

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

FEATURE_COLUMNS: list[str] = [
    "order_count_last_1h",
    "order_count_last_24h",
    "unique_ip_count",
    "avg_order_amount",
    "account_age_days",
    "device_count",
]


def main() -> None:
    # ---------- Config ----------
    experiment_name = "ai-secure-agent-logistic-regression"
    data_path = "data/fraud_training_data.csv"
    model_output_dir = Path("model")
    artifact_output_dir = Path("artifacts")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    label_column = "label"

    test_size = 0.2
    random_state = 42
    max_iter = 1000

    # ---------- MLflow setup ----------
    # 本地默认会把 runs 存到 ./mlruns
    # todo 用 mlflow server + remote storage
    # mlflow.set_tracking_uri("file://" + str(Path("mlruns").absolute()))
    mlflow.set_experiment(experiment_name)

    # sklearn autolog 会自动记录 estimator 参数、常见分类指标和模型
    # 兼容版本需注意官方说明
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    # ---------- Load data ----------
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        # 手动补充一些更贴近作业语义的参数
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("label_column", label_column)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("dataset_path", data_path)

        # 可选：记录 dataset preview / schema
        schema = {
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
        }
        schema_path = artifact_output_dir / "dataset_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(schema_path), artifact_path="dataset")

        # ---------- Train ----------
        model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        # ---------- Evaluate ----------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        # 手动记录，便于你在 UI 里看得更明确
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # ---------- Classification report artifact ----------
        report = classification_report(y_test, y_pred, zero_division=0)
        report_path = artifact_output_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        # ---------- Confusion matrix artifact ----------
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        fig.tight_layout()
        cm_path = artifact_output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        # ---------- Save feature order as artifact ----------
        feature_path = artifact_output_dir / "feature_columns.json"
        feature_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(feature_path), artifact_path="model_metadata")

        # ---------- Also save a plain local copy for your app ----------
        # 虽然 mlflow 已经记录模型了，但你本地服务直接加载一个固定文件更方便
        import joblib

        local_model_path = model_output_dir / "fraud_logistic_regression.pkl"
        local_feature_path = model_output_dir / "feature_columns.pkl"
        joblib.dump(model, local_model_path)
        joblib.dump(FEATURE_COLUMNS, local_feature_path)

        mlflow.log_artifact(str(local_model_path), artifact_path="local_export")
        mlflow.log_artifact(str(local_feature_path), artifact_path="local_export")

        # ---------- Tags ----------
        mlflow.set_tag("project", "ai-secure-agent")
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("use_case", "ecommerce_risk_detection")

        print("Training completed.")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"Local model saved to: {local_model_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    import pandas as pd

    main()
