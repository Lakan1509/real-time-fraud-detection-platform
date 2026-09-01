from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.train import generate_training_data, train_model


RESULTS_DIR = Path("results")
METRICS_PATH = RESULTS_DIR / "model_evaluation.json"


def evaluate_model(
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    X, y = generate_training_data(
        n_samples=3000,
        random_seed=random_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    threshold_analysis = {}

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)

        threshold_precision, threshold_recall, threshold_f1, _ = (
            precision_recall_fscore_support(
                y_test,
                threshold_predictions,
                average="binary",
                zero_division=0,
            )
        )

        threshold_analysis[str(threshold)] = {
            "precision": round(float(threshold_precision), 4),
            "recall": round(float(threshold_recall), 4),
            "f1": round(float(threshold_f1), 4),
        }

    metrics = {
        "dataset": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "fraud_rate": round(float(np.mean(y)), 4),
            "random_seed": random_seed,
            "test_size": test_size,
        },
        "model": {
            "type": type(model).__name__,
            "classification_threshold": 0.5,
        },
        "metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        },
        "threshold_analysis": threshold_analysis,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    return metrics


if __name__ == "__main__":
    result = evaluate_model()
    print(json.dumps(result, indent=2))
    print(f"\nSaved evaluation metrics to {METRICS_PATH}")
