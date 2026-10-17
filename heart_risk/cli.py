"""One readable entry point for reproducible stages."""
import argparse
import json

from heart_risk.settings import ROOT


def main():
    parser = argparse.ArgumentParser(description="Heart disease learning pipeline")
    parser.add_argument("stage", choices=["download", "eda", "train", "predict", "all"])
    parser.add_argument("--refresh", action="store_true", help="Redownload official raw data")
    parser.add_argument("--input", type=str, default=str(ROOT / "examples/patient.json"))
    args = parser.parse_args()
    if args.stage in ("download", "all"):
        from heart_risk.data import acquire_data
        print(f"Prepared {len(acquire_data(args.refresh))} rows")
    if args.stage in ("eda", "all"):
        from heart_risk.exploration import create_eda
        create_eda()
    if args.stage in ("train", "all"):
        from heart_risk.training import train_models
        train_models()
    if args.stage == "predict":
        import joblib
        from pathlib import Path
        from heart_risk.api import Patient, patient_frame
        from heart_risk.settings import MODEL_PATH
        patient = Patient.model_validate(json.loads(Path(args.input).read_text()))
        probability = float(joblib.load(MODEL_PATH).predict_proba(patient_frame(patient))[0, 1])
        print(json.dumps({"prediction": int(probability >= 0.5), "probability": probability}, indent=2))


if __name__ == "__main__":
    main()
