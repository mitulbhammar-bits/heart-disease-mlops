from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SOURCE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "diagnosis"]
FRIENDLY_COLUMNS = ["age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol",
                    "high_fasting_blood_sugar", "resting_ecg", "maximum_heart_rate",
                    "exercise_angina", "st_depression", "st_slope", "major_vessels",
                    "thalassemia_test", "heart_disease"]
NUMERIC_FEATURES = ["age", "resting_blood_pressure", "cholesterol", "maximum_heart_rate",
                    "st_depression"]
CATEGORICAL_FEATURES = [name for name in FRIENDLY_COLUMNS[:-1] if name not in NUMERIC_FEATURES]
FEATURES = FRIENDLY_COLUMNS[:-1]
DATA_PATH = ROOT / "data/processed/heart_disease.csv"
MODEL_PATH = ROOT / "models/heart_risk_pipeline.joblib"
