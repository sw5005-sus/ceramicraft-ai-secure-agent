import json
import os
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

from ceramicraft_ai_secure_agent.data.feature_columns import FEATURE_COLUMNS


def _build_model_weights(
    model: LogisticRegression,
    metrics: dict[str, float],
) -> dict:
    """Export a lightweight inference payload without sklearn runtime dependency."""
    if len(model.coef_) != 1:
        raise ValueError(
            "Only binary LogisticRegression is supported for lightweight export."
        )

    return {
        "model_type": "logistic_regression",
        "feature_columns": FEATURE_COLUMNS,
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "classes": model.classes_.tolist(),
        "positive_class": int(model.classes_[-1]),
        "threshold": 0.5,
        "metrics": metrics,
    }


def main() -> None:
    # ---------- 1. Config ----------
    data_path = os.environ.get("TRAINING_DATA_PATH", "data/fraud_training_data.csv")
    label_column = "label"
    max_iter = int(os.environ.get("MAX_ITER", "1000"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    use_mlflow = mlflow_uri is not None

    artifact_output_dir = Path("artifacts")
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 2. Load data ----------
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    if "split" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'split' column with train/test values."
        )

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty. Check rows where split == 'train'."
        )
    if test_df.empty:
        raise ValueError("Test dataset is empty. Check rows where split == 'test'.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    # Strictly use FEATURE_COLUMNS and ignore extra columns.
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[label_column]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[label_column]

    # ---------- 3. Train ----------
    # Use .values so the runtime payload is purely coefficient-based and
    # inference can be implemented without sklearn.
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight="balanced",
    )
    model.fit(X_train.values, y_train)

    # ---------- 4. Evaluate ----------
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\n" + "=" * 30)
    print("      TRAINING REPORT      ")
    print("=" * 30)
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---------- 5. Export lightweight model ----------
    model_weights = _build_model_weights(model, metrics)

    weights_path = artifact_output_dir / "model_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(model_weights, f, indent=4)

    metadata_path = artifact_output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "logistic_regression",
                "feature_columns": FEATURE_COLUMNS,
                "label_column": label_column,
                "class_weight": "balanced",
                "max_iter": max_iter,
                "random_state": random_state,
                "metrics": metrics,
            },
            f,
            indent=4,
        )

    print(f"Model weights saved to {weights_path}")
    print(f"Metadata saved to {metadata_path}")

    # ---------- 6. Save chart ----------
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Confusion Matrix")
    confusion_matrix_path = artifact_output_dir / "confusion_matrix.png"
    fig.savefig(confusion_matrix_path)
    plt.close(fig)

    # ---------- 7. MLflow tracking ----------
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
        )

        with mlflow.start_run(
            run_name=os.environ.get("MLFLOW_RUN_NAME", "lr_baseline")
        ):
            mlflow.log_params(
                {
                    "model_type": "logistic_regression",
                    "max_iter": max_iter,
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "runtime_dependency": "no_sklearn_for_inference",
                }
            )
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Only upload lightweight artifacts, not sklearn-flavor models.
            mlflow.log_artifact(str(weights_path), artifact_path="model")
            mlflow.log_artifact(str(metadata_path), artifact_path="model")
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn_model",
                registered_model_name=os.environ.get(
                    "MLFLOW_REGISTERED_MODEL_NAME", "fraud-detector"
                ),
                input_example=X_train.head(3),
            )

        print(f"Successfully logged lightweight artifacts to MLflow: {mlflow_uri}")
    else:
        print("MLflow tracking URI not set. Skipping MLflow logging.")


if __name__ == "__main__":
    main()
