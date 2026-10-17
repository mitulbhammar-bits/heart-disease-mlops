from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart_risk.features import build_preprocessor
from heart_risk.settings import SEED


def build_model(name: str, **parameters) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, random_state=SEED, **parameters)
    elif name == "random_forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                            n_jobs=1, **parameters)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prepare_features", build_preprocessor()), ("classifier", classifier)])


def candidate_models():
    # Six small, intentional experiments keep the search understandable and inexpensive.
    for strength in (0.1, 1.0, 10.0):
        yield "logistic_regression", {"C": strength}
    for depth in (3, 6, None):
        yield "random_forest", {"max_depth": depth, "min_samples_leaf": 2}
