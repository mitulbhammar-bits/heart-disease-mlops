"""Select with training-fold ROC-AUC; evaluate the winner once on held-out data."""
import hashlib
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from heart_risk.data import acquire_data
from heart_risk.models import build_model, candidate_models
from heart_risk.settings import DATA_PATH, FEATURES, MODEL_PATH, ROOT, SEED

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "roc_auc": "roc_auc"}


def train_models():
    data = acquire_data()
    features, target = data[FEATURES], data.heart_disease
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name("heart-risk-model-comparison")
    if experiment is None:
        mlflow.create_experiment("heart-risk-model-comparison",
                                 artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment("heart-risk-model-comparison")
    results, best = [], None
    figure_folder = ROOT / "reports/figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    for name, parameters in candidate_models():
        model = build_model(name, **parameters)
        scores = cross_validate(model, x_train, y_train, cv=folds, scoring=SCORING,
                                error_score="raise", n_jobs=1)
        metrics = {f"cv_{metric}_mean": float(scores[f"test_{metric}"].mean())
                   for metric in SCORING}
        metrics.update({f"cv_{metric}_std": float(scores[f"test_{metric}"].std())
                        for metric in SCORING})
        model.fit(x_train, y_train)
        with mlflow.start_run(run_name=f"{name}-{len(results)+1}") as run:
            mlflow.log_params({"model": name, "random_seed": SEED, "folds": 5,
                               "training_rows": len(x_train), "holdout_rows": len(x_test),
                               "threshold": 0.5, **parameters})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"purpose": "cv_candidate", "python": platform.python_version(),
                             "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()})
            fold_data = {metric: scores[f"test_{metric}"].tolist() for metric in SCORING}
            mlflow.log_dict(fold_data, "fold_metrics.json")
            mlflow.log_artifact(str(ROOT / "data/source_manifest.json"))
            fig, axis = plt.subplots(figsize=(7, 4))
            for metric, values in fold_data.items():
                axis.plot(range(1, 6), values, marker="o", label=metric)
            axis.set(xlabel="Validation fold", ylabel="Score", ylim=(0, 1.05),
                     title=f"{name}: {parameters}")
            axis.legend()
            fig.tight_layout()
            mlflow.log_figure(fig, "plots/cross_validation.png")
            plt.close(fig)
            mlflow.sklearn.log_model(model, name="pipeline",
                                     input_example=x_train.head(2))
            row = {"model": name, "parameters": parameters, "run_id": run.info.run_id, **metrics}
            results.append(row)
        if best is None or metrics["cv_roc_auc_mean"] > best["cv_roc_auc_mean"]:
            best = row
    # No candidate sees the holdout during selection. Keep the winner fitted on training data.
    winner = build_model(best["model"], **best["parameters"]).fit(x_train, y_train)
    probabilities = winner.predict_proba(x_test)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    holdout = {"accuracy": accuracy_score(y_test, predicted),
               "precision": precision_score(y_test, predicted, zero_division=0),
               "recall": recall_score(y_test, predicted, zero_division=0),
               "roc_auc": roc_auc_score(y_test, probabilities)}
    for name, display in [("holdout_roc_curve", RocCurveDisplay.from_predictions),
                          ("holdout_confusion_matrix", ConfusionMatrixDisplay.from_predictions)]:
        display(y_test, probabilities if name.endswith("roc_curve") else predicted)
        plt.title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(figure_folder / f"{name}.png", dpi=160)
        plt.close()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, MODEL_PATH)
    metadata = {"selected_model": best["model"], "selected_parameters": best["parameters"],
                "selection_metric": "mean 5-fold training-only ROC-AUC", "selected_run_id": best["run_id"],
                "cv_roc_auc": best["cv_roc_auc_mean"], "holdout_metrics": holdout,
                "threshold": 0.5, "seed": SEED, "train_rows": len(x_train), "test_rows": len(x_test),
                "features": FEATURES, "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                "python": platform.python_version(), "use": "Educational demonstration only"}
    (ROOT / "models/model_metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(results).to_csv(ROOT / "reports/model_comparison.csv", index=False)
    pd.DataFrame({"source_row": x_test.index, "actual": y_test.values,
                  "probability": probabilities, "predicted": predicted}).to_csv(
                      ROOT / "reports/holdout_predictions.csv", index=False)
    (ROOT / "reports/split_indices.json").write_text(json.dumps(
        {"train": x_train.index.tolist(), "holdout": x_test.index.tolist()}, indent=2))
    with mlflow.start_run(run_name="selected-model-holdout"):
        mlflow.log_params({"selected_model": best["model"], "threshold": 0.5})
        mlflow.log_metrics({f"holdout_{key}": value for key, value in holdout.items()})
        mlflow.set_tag("purpose", "final_holdout_evaluation")
        mlflow.log_artifacts(str(figure_folder), artifact_path="plots")
        mlflow.log_artifact(str(ROOT / "reports/model_comparison.csv"))
        mlflow.log_artifact(str(ROOT / "models/model_metadata.json"))
        mlflow.log_artifact(str(ROOT / "reports/split_indices.json"))
        mlflow.sklearn.log_model(winner, name="selected_pipeline", input_example=x_train.head(2))
    fig, axis = plt.subplots(figsize=(9, 4))
    labels = [f"{row['model'].replace('_', ' ')} {i+1}" for i, row in enumerate(results)]
    axis.barh(labels, [r["cv_roc_auc_mean"] for r in results],
              xerr=[r["cv_roc_auc_std"] for r in results], color="#246a82")
    axis.set(xlabel="Mean CV ROC-AUC (+/- fold standard deviation)", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(figure_folder / "model_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))
    return metadata
