import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
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


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
        print(f"Registered model already exists: {model_name}")
    except Exception:
        client.create_registered_model(model_name)
        print(f"Created registered model: {model_name}")


def main() -> None:
    # ---------- Config ----------
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        "ai-secure-agent-logistic-regression",
    )
    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "ai-secure-agent-fraud-detector",
    )
    run_name = os.environ.get(
        "MLFLOW_RUN_NAME",
        "logistic_regression_baseline",
    )

    data_path = os.environ.get("TRAINING_DATA_PATH", "data/fraud_training_data.csv")
    label_column = "label"

    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    max_iter = int(os.environ.get("MAX_ITER", "1000"))

    dataset_version = os.environ.get("DATASET_VERSION", "v1")
    git_commit = os.environ.get("GIT_COMMIT", "unknown")
    model_alias = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is not set, example: http://<mlflow-host>:5000"
        )

    artifact_output_dir = Path("artifacts")
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- MLflow setup ----------
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    ensure_registered_model(client, registered_model_name)

    # autolog 可保留，帮你记录 sklearn 参数和部分模型信息
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

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ---------- Minimal params ----------
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("label_column", label_column)
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("dataset_version", dataset_version)

        # ---------- Minimal run tag ----------
        mlflow.set_tag("git_commit", git_commit)

        # ---------- Optional dataset schema artifact ----------
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

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # ---------- Evaluation artifacts ----------
        report = classification_report(y_test, y_pred, zero_division=0)
        report_path = artifact_output_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        fig.tight_layout()
        cm_path = artifact_output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        feature_path = artifact_output_dir / "feature_columns.json"
        feature_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(feature_path), artifact_path="model_metadata")

        # ---------- Log + register model ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=X_train.head(3),
        )

        # ---------- Find current registered version ----------
        versions = client.search_model_versions(f"name = '{registered_model_name}'")
        current_version = None
        for mv in versions:
            if mv.run_id == run_id:
                current_version = mv.version
                break

        if current_version is None:
            raise RuntimeError(
                f"Cannot find registered model version for run_id={run_id}"
            )

        # ---------- Minimal model version tags ----------
        client.set_model_version_tag(
            name=registered_model_name,
            version=current_version,
            key="dataset_version",
            value=dataset_version,
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=current_version,
            key="test_f1",
            value=f"{f1:.6f}",
        )

        # ---------- Alias ----------
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=model_alias,
            version=current_version,
        )

        print("Training completed.")
        print(f"Tracking URI: {mlflow_tracking_uri}")
        print(f"Experiment: {experiment_name}")
        print(f"Registered model: {registered_model_name}")
        print(f"Registered version: {current_version}")
        print(f"Alias '{model_alias}' -> version {current_version}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")


if __name__ == "__main__":
    main()
