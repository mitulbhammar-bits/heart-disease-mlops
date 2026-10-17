import json

import pytest

from heart_risk.data import acquire_data
from heart_risk.models import build_model
from heart_risk.settings import FEATURES, ROOT


@pytest.fixture(scope="session")
def dataset():
    return acquire_data()


@pytest.fixture(scope="session")
def fitted_model(dataset):
    return build_model("logistic_regression", C=1.0).fit(dataset[FEATURES], dataset.heart_disease)


@pytest.fixture
def patient():
    return json.loads((ROOT / "examples/patient.json").read_text())
