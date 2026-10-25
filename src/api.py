import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.metrics import LATENCY, REQUESTS
from src.schema import PatientFeatures, PredictionResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heart-disease-api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Heart Disease Risk API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL_PATH.exists() else "model_missing"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    started = time.perf_counter()
    if not MODEL_PATH.exists():
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model artifact is unavailable")
    try:
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([features.model_dump()])
        prediction = int(model.predict(row)[0])
        confidence = float(max(model.predict_proba(row)[0]))
        REQUESTS.labels(status="success").inc()
        logger.info("prediction=%s confidence=%.4f", prediction, confidence)
        return PredictionResponse(
            prediction=prediction, confidence=confidence, model_version="1.0.0"
        )
    except (ValueError, KeyError, OSError) as exc:
        REQUESTS.labels(status="error").inc()
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        LATENCY.observe(time.perf_counter() - started)

