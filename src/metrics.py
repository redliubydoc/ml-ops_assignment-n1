from prometheus_client import Counter, Histogram

REQUESTS = Counter("prediction_requests_total", "Prediction API requests", ["status"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction API latency")
