# Submission Evidence Checklist

The implementation produces the following evidence locally. Capture the
screenshots below after running the corresponding command and place them in
`screenshots/` before submitting.

| Evidence | How to produce it | Status |
|---|---|---|
| EDA histograms | `python -m scripts.eda`; open `artifacts/eda/feature_histograms.png` | Generated |
| Class balance | Open `artifacts/eda/class_balance.png` | Generated |
| Correlation heatmap | Open `artifacts/eda/correlation_heatmap.png` | Generated |
| MLflow comparison | Run `mlflow ui`; show both model runs and metrics | Local run generated |
| Model artifact | Show `artifacts/model.joblib` and training output | Generated |
| Passing tests | Run `pytest -q` | 3 passed |
| CI/CD | Push repository and capture the green GitHub Actions run | To capture |
| Docker | Capture `docker build` and `/docs` or `/health` | To capture |
| Kubernetes | Capture pods, service, and endpoint response | To capture |
| Monitoring | Capture Prometheus/Grafana request and latency panels | To capture |
| API demo | Capture `/docs` request using `examples/predict.json` | Local smoke test passed |

## Integrity note

Do not submit placeholder screenshots. The cloud/Kubernetes, CI, and Grafana
screenshots must be captured from the student's own environment. The repository
contains commands and manifests to reproduce each item.

## Video outline

Record a 2–4 minute screen capture showing:

1. repository layout and README;
2. test run;
3. MLflow comparison;
4. Docker build and API request;
5. Kubernetes rollout and monitoring endpoint.

