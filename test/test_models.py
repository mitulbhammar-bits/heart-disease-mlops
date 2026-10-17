import joblib
import numpy as np
import pytest

from heart_risk.models import build_model
from heart_risk.settings import FEATURES


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_model_handles_missing_and_unseen_categories(dataset, name):
    model = build_model(name).fit(dataset[FEATURES], dataset.heart_disease)
    example = dataset[FEATURES].head(2).copy()
    example.loc[example.index[0], "major_vessels"] = np.nan
    example.loc[example.index[1], "chest_pain_type"] = 99
    probabilities = model.predict_proba(example)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_serialization_preserves_predictions(tmp_path, fitted_model, dataset):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)
    loaded = joblib.load(path)
    example = dataset[FEATURES].head(4)
    np.testing.assert_allclose(loaded.predict_proba(example), fitted_model.predict_proba(example))


def test_imputer_learns_only_training_values(dataset):
    training = dataset[FEATURES].iloc[:50]
    model = build_model("logistic_regression").fit(training, dataset.heart_disease.iloc[:50])
    numeric = model.named_steps["prepare_features"].named_transformers_["continuous"]
    assert numeric.named_steps["fill_missing"].statistics_[0] == training.age.median()


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        build_model("unsupported")
