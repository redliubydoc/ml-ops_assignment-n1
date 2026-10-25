# End-to-End MLOps: Heart Disease Risk Classifier

**Course:** Machine Learning Operations (MLOps) AIMLCZG523  
**Assignment:** 01  
**Dataset:** UCI Heart Disease, Cleveland subset  

## 1. Executive summary

This project delivers a reproducible binary classifier for the presence of
heart disease. It separates data acquisition, preprocessing, model training,
serving, deployment, and monitoring. The API returns a class prediction and
the maximum class probability; it is not a medical diagnosis and is intended
only for coursework.

## 2. Architecture

```text
UCI CSV -> download_data.py -> pandas validation
                              |
                 EDA plots + preprocessing pipeline
                              |
       Logistic Regression / Random Forest -> MLflow + joblib
                              |
                 FastAPI /predict + /health + /metrics
                              |
                 Docker -> Kubernetes LoadBalancer
                              |
                    Prometheus -> Grafana dashboard
```

The preprocessing transformer is serialized inside the model pipeline, so
training and inference apply identical imputation, scaling, and one-hot
encoding. Kubernetes runs two replicas and HTTP health probes.

## 3. Data acquisition and EDA

`scripts/download_data.py` downloads the Cleveland data directly from UCI,
converts `?` to missing values, and maps the original `num` field to a binary
`target`: zero means no disease and values one through four mean disease.
The input contains 303 rows and 13 predictors. `scripts/eda.py` creates:

* `artifacts/eda/feature_histograms.png`;
* `artifacts/eda/class_balance.png`;
* `artifacts/eda/correlation_heatmap.png`.

Categorical variables are encoded with one-hot encoding. Numeric values are
median-imputed and standardized. This avoids leakage because transformations
are fitted inside the cross-validation pipeline.

## 4. Feature engineering and modelling

Two candidates are trained:

1. **Logistic Regression**, `max_iter=1000`, balanced class weights.
2. **Random Forest**, 300 trees, maximum depth 6, balanced class weights.

Five-fold stratified cross-validation reports accuracy, precision, recall, and
ROC-AUC. The model with the highest mean validation ROC-AUC is selected, then
refit on the training split and evaluated on a stratified 20% holdout. The
exact scores are printed by `scripts/train.py` and stored in the MLflow run;
this keeps the report reproducible instead of hard-coding a result that can
vary with dependency versions.

## 5. Experiment tracking and packaging

Each candidate has a separate MLflow run. Parameters include model name and
fold count; metrics include all four cross-validation measures. The fitted
pipeline and plots are artifacts. `artifacts/model.joblib` contains the final
preprocessor and estimator, and `requirements.txt` pins runtime versions.

## 6. Testing and CI/CD

`tests/test_pipeline.py` checks missing-value handling and schema validation.
`tests/test_api.py` checks the health endpoint. GitHub Actions installs
dependencies, runs Ruff, downloads data, runs Pytest, generates EDA plots,
trains the models, and uploads `artifacts/` for every run. Any failed command
fails the workflow.

## 7. Container and API

The Dockerfile uses Python 3.12 slim, installs pinned dependencies, downloads
the data, trains at image-build time, and starts Uvicorn on port 8000.
`POST /predict` validates the 13 input features using Pydantic and returns
`prediction`, `confidence`, and `model_version`. `/health` supports probes and
`/metrics` exposes Prometheus counters and latency.

## 8. Deployment and monitoring

`k8s/deployment.yaml` creates two API replicas and a LoadBalancer service.
For Minikube, use `minikube service heart-disease-api --url`; for a cloud
cluster, push the image to a registry and update the manifest. The
`ServiceMonitor` manifest integrates with Prometheus Operator. Dashboard
panels should use `prediction_requests_total` and
`prediction_latency_seconds` to show request volume, errors, and latency.

## 9. Reproduction checklist

```powershell
pip install -r requirements.txt
python scripts\download_data.py
python -m scripts.eda
python -m scripts.train
pytest -q
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

The endpoint can be tested with `examples/predict.json`. Evidence screenshots
to attach to the final submission are: MLflow runs, passing CI workflow,
Docker build and `/docs`, Kubernetes pods/service, and Grafana metrics.

## 10. Limitations and ethics

The Cleveland subset is small and historically collected; its variables may
not represent modern or diverse populations. A probability is not calibrated
clinical risk. This coursework service must not be used for diagnosis or
treatment. Production use would require clinical validation, calibration,
fairness analysis, access control, encryption, drift monitoring, and a
human-in-the-loop review process.

## 11. Operations documentation

The supporting documentation is separated into focused operator documents:

* [Architecture](docs/ARCHITECTURE.md) describes data flow, runtime flow, and
  reliability boundaries.
* [Setup and operations](docs/SETUP.md) gives commands for local execution,
  MLflow, Docker, and Minikube.
* [API reference](docs/API.md) documents request validation, response fields,
  metrics, and error behavior.
* [Evidence checklist](docs/EVIDENCE_CHECKLIST.md) maps each marking
  requirement to the evidence that must be captured.

This separation keeps the report readable while allowing another engineer to
operate the service without reverse-engineering the source code.

## 12. Submission evidence and reproducibility statement

The local implementation has been executed successfully: the UCI file was
downloaded, EDA artifacts were generated, both candidate models were trained,
MLflow runs were created, three tests passed, and the `/health` and `/predict`
endpoints returned valid responses. The generated artifacts are included under
`artifacts/`.

The final PDF includes six genuine screenshots from this workspace: EDA
histograms, class balance, correlation heatmap, MLflow, GitHub Actions, and
the Docker-served Swagger UI. Kubernetes and Grafana screenshots are not
claimed because the local Minikube image transfer failed in WSL; the
deployment manifests and exact recovery instructions remain available in
`docs/SETUP.md`. No placeholder images are included. The exact commands and a
short video outline are in `docs/EVIDENCE_CHECKLIST.md`.
