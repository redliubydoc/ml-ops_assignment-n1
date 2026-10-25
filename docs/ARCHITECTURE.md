# System Architecture

## Logical architecture

```mermaid
flowchart LR
    UCI[UCI Cleveland CSV] --> D[download_data.py]
    D --> DATA[data/heart_disease.csv]
    DATA --> EDA[EDA plots]
    DATA --> TRAIN[Training pipeline]
    TRAIN --> PREP[Imputation + scaling + one-hot encoding]
    PREP --> LR[Logistic Regression]
    PREP --> RF[Random Forest]
    LR --> MLF[MLflow runs]
    RF --> MLF
    LR --> ART[artifacts/model.joblib]
    RF --> ART
    ART --> API[FastAPI service]
    API --> DOCKER[Docker image]
    DOCKER --> K8S[Kubernetes Deployment: 2 replicas]
    K8S --> LB[LoadBalancer Service]
    API --> METRICS[/metrics]
    METRICS --> PROM[Prometheus]
    PROM --> GRAF[Grafana dashboard]
    CI[GitHub Actions] --> TEST[Lint + tests + training]
    TEST --> ARTIFACTS[Workflow artifacts]
```

## Runtime request flow

1. A client sends JSON to `POST /predict`.
2. Pydantic validates the 13 fields and their ranges.
3. The serialized scikit-learn pipeline applies the same transformations used
   during training.
4. The model returns the predicted class and class probability.
5. The API emits structured logs and Prometheus request/latency metrics.

## Reliability and security boundaries

* The model artifact is read-only at runtime.
* Kubernetes readiness and liveness probes use `/health`.
* Invalid request values are rejected with HTTP 422.
* Missing model artifacts produce HTTP 503 instead of a success-shaped result.
* Production deployment should add authentication, TLS, rate limits, secrets
  management, network policies, and image scanning.

