# API Reference

Base URL: `http://localhost:8000`

## `GET /health`

Returns service readiness.

```json
{"status": "ok"}
```

The status is `model_missing` when `artifacts/model.joblib` is unavailable.

## `POST /predict`

Request body:

```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 0.9,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

Response:

```json
{
  "prediction": 0,
  "confidence": 0.6331310533344844,
  "model_version": "1.0.0"
}
```

`prediction` is `0` for the negative class and `1` for the positive class.
`confidence` is the larger of the two model class probabilities. This value
is not a calibrated medical risk score.

## `GET /metrics`

Exposes Prometheus text-format metrics, including:

* `prediction_requests_total{status="success|error"}`
* `prediction_latency_seconds`

## Error behavior

* `422`: request schema or range validation failed.
* `503`: the model artifact is unavailable.
* `500`: an unexpected model or inference failure occurred.

