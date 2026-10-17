"""Validated serving, readiness and low-cardinality request monitoring."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heart_risk.settings import FEATURES, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart_risk.requests")
REQUESTS = Counter("heart_api_requests_total", "Requests by route and status", ["route", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "Request latency", ["route"])
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class", ["prediction"])


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float = Field(ge=1, le=120)
    sex: Literal[0, 1]
    chest_pain_type: Literal[1, 2, 3, 4]
    resting_blood_pressure: float = Field(gt=0, le=350)
    cholesterol: float = Field(gt=0, le=1000)
    high_fasting_blood_sugar: Literal[0, 1]
    resting_ecg: Literal[0, 1, 2]
    maximum_heart_rate: float = Field(gt=0, le=300)
    exercise_angina: Literal[0, 1]
    st_depression: float = Field(ge=0, le=10)
    st_slope: Literal[1, 2, 3]
    major_vessels: Literal[0, 1, 2, 3] | None = None
    thalassemia_test: Literal[3, 6, 7] | None = None


def patient_frame(patient: Patient) -> pd.DataFrame:
    return pd.DataFrame([patient.model_dump()], columns=FEATURES).replace({None: np.nan})


def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = joblib.load(path)
        yield

    app = FastAPI(title="Heart Disease Learning API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            route_name = route.path if route else "unmatched"
            elapsed = time.perf_counter() - start
            REQUESTS.labels(route_name, str(status)).inc()
            LATENCY.labels(route_name).observe(elapsed)
            logger.info(json.dumps({"request_id": request_id, "method": request.method,
                                    "route": route_name, "status": status,
                                    "duration_ms": round(elapsed * 1000, 2)}))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready", "model_loaded": hasattr(app.state, "model")}

    @app.post("/predict")
    def predict(patient: Patient):
        probability = float(app.state.model.predict_proba(patient_frame(patient))[0, 1])
        prediction = int(probability >= 0.5)
        PREDICTIONS.labels(str(prediction)).inc()
        return {"prediction": prediction, "label": "present" if prediction else "absent",
                "heart_disease_probability": probability,
                "confidence": probability if prediction else 1 - probability,
                "threshold": 0.5, "model_version": "1.0.0"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
