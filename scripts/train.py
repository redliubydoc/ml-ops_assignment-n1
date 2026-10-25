import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor, load_dataset, save_pipeline


def evaluate(name: str, estimator: Pipeline, X: pd.DataFrame, y: pd.Series, out: Path) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring)
    metrics = {f"cv_{key}": float(scores[f"test_{key}"].mean()) for key in scoring}
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metrics(metrics)
        fitted = estimator.fit(X, y)
        save_pipeline(fitted, out / f"{name.lower().replace(' ', '_')}.joblib")
        mlflow.sklearn.log_model(fitted, "model")
        mlflow.log_artifact(str(out / f"{name.lower().replace(' ', '_')}.joblib"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="heart-disease")
    args = parser.parse_args()
    X, y = load_dataset("data/heart_disease.csv")
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    mlflow.set_experiment(args.experiment)
    models = {
        "Logistic Regression": Pipeline(
            [("preprocess", build_preprocessor()),
             ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]
        ),
        "Random Forest": Pipeline(
            [("preprocess", build_preprocessor()),
             ("model", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"))]
        ),
    }
    results = {name: evaluate(name, model, X, y, out) for name, model in models.items()}
    best_name = max(results, key=lambda name: results[name]["cv_roc_auc"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best = models[best_name].fit(X_train, y_train)
    predictions = best.predict(X_test)
    probabilities = best.predict_proba(X_test)[:, 1]
    final_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    save_pipeline(best, out / "model.joblib")
    pd.DataFrame(confusion_matrix(y_test, predictions)).to_csv(out / "confusion_matrix.csv", index=False)
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=160)
    with mlflow.start_run(run_name=f"{best_name} - holdout"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({key: float(value) for key, value in final_metrics.items()})
        mlflow.log_artifact(str(out / "model.joblib"))
        mlflow.log_artifact(str(out / "confusion_matrix.csv"))
        mlflow.log_artifact(str(out / "roc_curve.png"))
    print({"selected_model": best_name, "cv_results": results, "holdout": final_metrics})


if __name__ == "__main__":
    main()
