# Heart Disease MLOps Assignment

This project implements an end-to-end heart-disease classifier using the UCI
Heart Disease dataset. It includes reproducible data acquisition, EDA,
scikit-learn pipelines, MLflow tracking, a FastAPI service, tests, Docker,
Kubernetes, Prometheus metrics, and GitHub Actions.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_data.py
python -m scripts.train --experiment heart-disease
pytest -q
uvicorn src.api:app --reload
```

Open `http://localhost:8000/docs`. The sample request is in
`examples/predict.json`. The API expects the 13 features listed in
`src/schema.py`; `num` is the UCI heart-disease diagnosis target encoding and
is intentionally excluded from the request.

## MLflow

```powershell
mlflow ui --backend-store-uri file:./mlruns
```

Then visit `http://localhost:5000`. Training logs parameters, cross-validation
metrics, the confusion matrix, ROC curve, and the serialized pipeline.

## Docker

```powershell
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

The image trains the model during build and starts the API with the packaged
pipeline. Test it with:

```powershell
curl -Method POST http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body (Get-Content examples\predict.json -Raw)
```

## Kubernetes / Minikube

```powershell
minikube start
kubectl apply -f k8s\
kubectl rollout status deployment/heart-disease-api
minikube service heart-disease-api --url
```

The manifest uses a local image. For a registry deployment, change the image
in `k8s/deployment.yaml` and set `imagePullPolicy: Always`.

## Project layout

* `scripts/download_data.py` downloads and normalizes the UCI CSV.
* `scripts/eda.py` writes professional plots to `artifacts/eda`.
* `scripts/train.py` compares logistic regression and random forest, performs
  stratified cross-validation, and writes `artifacts/model.joblib`.
* `src/` contains reusable preprocessing, training, schema, and API code.
* `tests/` contains unit and API tests.
* `.github/workflows/ci.yml` runs linting, tests, training, and artifact upload.
* `report.md` is the submission report template with reproducibility evidence.
* `docs/` contains architecture, setup, API, and evidence documentation.
* `screenshots/` contains the recommended final-submission evidence layout.
* `report.pdf` is the generated written report with an evidence appendix.

## Submission links

* GitHub repository: https://github.com/redliubydoc/ml-ops_assignment-n1
* GitHub Actions workflow: https://github.com/redliubydoc/ml-ops_assignment-n1/actions
* Local API documentation: http://localhost:8000/docs
* Local MLflow UI: http://localhost:5000

The API is configured for local testing; no public deployment URL is claimed.
Submit the short pipeline video separately with the repository and report.
