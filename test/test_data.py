import hashlib
import json

import pandas as pd
import pytest

from heart_risk.data import clean_data
from heart_risk.settings import FEATURES, RAW_COLUMNS, ROOT


def test_data_contract(dataset):
    assert len(dataset) == 303
    assert list(dataset.columns) == FEATURES + ["heart_disease"]
    assert set(dataset.heart_disease.unique()) == {0, 1}
    assert not dataset.duplicated().any()


def test_source_checksum():
    manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
    raw = ROOT / "data/raw/processed.cleveland.data"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == manifest["source_sha256"]


def test_missing_values_preserved_for_pipeline(dataset):
    assert dataset.major_vessels.isna().sum() == 4
    assert dataset.thalassemia_test.isna().sum() == 2


@pytest.mark.parametrize("diagnosis, expected", [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)])
def test_binary_target_mapping(diagnosis, expected):
    row = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, diagnosis]
    clean = clean_data(pd.DataFrame([row], columns=RAW_COLUMNS))
    assert clean.heart_disease.iloc[0] == expected
    assert pd.isna(clean.major_vessels.iloc[0])


def test_reject_bad_schema():
    with pytest.raises(ValueError, match="columns"):
        clean_data(pd.DataFrame({"incorrect": [1]}))


def test_reject_invalid_target():
    with pytest.raises(ValueError, match="Diagnosis"):
        clean_data(pd.DataFrame([[1]*13 + [7]], columns=RAW_COLUMNS))
