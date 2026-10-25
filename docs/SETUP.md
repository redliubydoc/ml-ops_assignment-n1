# Setup and Operations Guide

## Local setup

From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the experiment

```powershell
python scripts\download_data.py
python -m scripts.eda
python -m scripts.train --experiment heart-disease
pytest -q
```

Expected local result from the validated run:

| Candidate | CV accuracy | CV precision | CV recall | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8449 | 0.8565 | 0.7984 | **0.9154** |
| Random Forest | 0.8251 | 0.8377 | 0.7696 | 0.9123 |

The selected model is Logistic Regression. The holdout result was accuracy
0.8689, precision 0.8125, recall 0.9286, and ROC-AUC 0.9578.

## Run the API

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = Get-Content examples\predict.json -Raw
Invoke-RestMethod http://127.0.0.1:8000/predict `
  -Method Post -ContentType "application/json" -Body $body
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI page.

## MLflow

```powershell
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Open `http://127.0.0.1:5000` and compare the two runs by `cv_roc_auc`.

## Docker

```powershell
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

The image downloads the data and trains during the image build so the runtime
container only serves the packaged artifact.

## Minikube

```powershell
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f k8s\deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,service
minikube service heart-disease-api --url
```

`k8s/prometheus.yaml` is intended for a cluster with Prometheus Operator.
Apply it after installing that operator:

```powershell
kubectl apply -f k8s\prometheus.yaml
```

