"""Every fitted preprocessing operation lives inside the estimator pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_risk.settings import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
    categories = Pipeline([("fill_missing", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("continuous", continuous, NUMERIC_FEATURES),
                              ("categories", categories, CATEGORICAL_FEATURES)])
