from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report.pdf"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def build() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=24, leading=30, textColor=colors.HexColor("#17365D"),
        spaceAfter=24,
    )
    heading = ParagraphStyle(
        "Section", parent=styles["Heading1"], fontSize=17, leading=21,
        textColor=colors.HexColor("#17365D"), spaceAfter=14,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
        spaceAfter=10,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=9, leading=12,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=0.7 * inch,
        leftMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="End-to-End MLOps Heart Disease Risk Classifier",
        author="Student",
    )
    story = []

    def page(number: str, heading_text: str, content: list[object]) -> None:
        if story:
            story.append(PageBreak())
        story.append(paragraph(heading_text, heading))
        story.extend(content)
        story.append(Spacer(1, 12))
        story.append(paragraph(f"Assignment 01 | Page {number}", small))

    def evidence_image(filename: str, caption: str) -> list[object]:
        image_path = ROOT / "screenshots" / filename
        if not image_path.exists():
            return [paragraph(f"<b>Evidence unavailable:</b> {filename}", small)]
        width, height = ImageReader(str(image_path)).getSize()
        max_width = 6.8 * inch
        max_height = 4.65 * inch
        scale = min(max_width / width, max_height / height)
        return [
            Image(str(image_path), width=width * scale, height=height * scale),
            Spacer(1, 6),
            paragraph(caption, small),
            Spacer(1, 12),
        ]

    story.extend([
        Spacer(1, 1.2 * inch),
        paragraph("End-to-End MLOps", title),
        paragraph("Heart Disease Risk Classifier", title),
        Spacer(1, 0.35 * inch),
        paragraph("Machine Learning Operations (MLOps) AIMLCZG523", body),
        paragraph("Assignment 01 | UCI Heart Disease Cleveland Dataset", body),
        Spacer(1, 1.1 * inch),
        paragraph(
            "A reproducible machine-learning solution covering data acquisition, "
            "exploration, modelling, experiment tracking, CI/CD, containerisation, "
            "deployment, serving, and monitoring.",
            body,
        ),
        Spacer(1, 2.2 * inch),
        paragraph("Prepared for academic submission", body),
        PageBreak(),
    ])

    page("2", "1. Executive Summary", [
        paragraph(
            "This project implements a binary classifier that predicts whether a "
            "patient record indicates the presence of heart disease. The solution "
            "is deliberately structured as a production-style MLOps project: "
            "data acquisition is reproducible, preprocessing is part of the "
            "serialized model pipeline, experiments are tracked with MLflow, "
            "quality checks run in GitHub Actions, and inference is exposed as "
            "a container-ready FastAPI service.",
            body,
        ),
        paragraph(
            "Two classification candidates are evaluated using stratified "
            "five-fold cross-validation. Logistic Regression is selected by "
            "mean ROC-AUC, then evaluated on a stratified holdout set. The "
            "validated local run selected Logistic Regression with cross-validated "
            "ROC-AUC 0.9154 and holdout ROC-AUC 0.9578.",
            body,
        ),
        paragraph(
            "<b>Important limitation:</b> this is an educational model and not a "
            "medical diagnostic system. Its confidence value is not a calibrated "
            "clinical risk score.",
            body,
        ),
    ])

    page("3", "2. Data Acquisition and Exploratory Analysis", [
        paragraph(
            "scripts/download_data.py downloads the Cleveland subset directly from "
            "the UCI Machine Learning Repository. The original question-mark "
            "values are converted to missing values. The original num field is "
            "mapped to a binary target: zero is the negative class and values "
            "one through four are the positive class.",
            body,
        ),
        paragraph(
            "The resulting dataset contains 303 rows and 13 predictors: age, sex, "
            "chest-pain type, resting blood pressure, cholesterol, fasting blood "
            "sugar, resting ECG, maximum heart rate, exercise-induced angina, "
            "oldpeak, slope, number of major vessels, and thalassemia status.",
            body,
        ),
        paragraph(
            "scripts/eda.py generates feature histograms, a class-balance chart, "
            "and a correlation heatmap under artifacts/eda/. These plots are "
            "generated from the downloaded dataset and can be included as "
            "screenshots in the final submission.",
            body,
        ),
        paragraph(
            "The EDA stage is intentionally separate from model fitting. It is "
            "used for understanding distributions, missingness, class balance, "
            "and relationships; it does not fit transformations using the "
            "holdout test data.",
            body,
        ),
    ])

    page("4", "3. Feature Engineering and Reproducibility", [
        paragraph(
            "The reusable preprocessing transformer is defined in src/pipeline.py. "
            "Numeric variables are median-imputed and standardised with "
            "StandardScaler. Categorical variables are imputed with the most "
            "frequent value and encoded with OneHotEncoder(handle_unknown='ignore').",
            body,
        ),
        paragraph(
            "The transformer is placed inside a scikit-learn Pipeline with the "
            "estimator. This prevents data leakage because each cross-validation "
            "fold fits its preprocessing only on the corresponding training "
            "partition. The complete fitted pipeline is saved as "
            "artifacts/model.joblib.",
            body,
        ),
        paragraph(
            "requirements.txt pins the main runtime packages. The project can be "
            "recreated from a clean Python 3.10 or 3.12 environment using the "
            "commands in docs/SETUP.md. The API validates input ranges with "
            "Pydantic before the model receives a request.",
            body,
        ),
    ])

    results = [
        ["Model", "CV Accuracy", "CV Precision", "CV Recall", "CV ROC-AUC"],
        ["Logistic Regression", "0.8449", "0.8565", "0.7984", "0.9154"],
        ["Random Forest", "0.8251", "0.8377", "0.7696", "0.9123"],
    ]
    table = Table(results, colWidths=[1.7 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 1.1 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EAF2F8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    page("5", "4. Model Development and Evaluation", [
        paragraph(
            "The project compares Logistic Regression and Random Forest. Logistic "
            "Regression uses max_iter=1000 and balanced class weights. Random "
            "Forest uses 300 estimators, maximum depth 6, balanced class weights, "
            "and a fixed random seed.",
            body,
        ),
        table,
        Spacer(1, 12),
        paragraph(
            "The selection criterion is mean cross-validated ROC-AUC because it "
            "evaluates ranking quality across classification thresholds. Accuracy, "
            "precision, and recall are also reported because the cost of missed "
            "positive cases is relevant to this problem.",
            body,
        ),
        paragraph(
            "The selected Logistic Regression model achieved holdout accuracy "
            "0.8689, precision 0.8125, recall 0.9286, and ROC-AUC 0.9578 on the "
            "fixed stratified 20% split.",
            body,
        ),
    ])

    page("6", "5. Experiment Tracking and Model Packaging", [
        paragraph(
            "scripts/train.py creates separate MLflow runs for the two candidate "
            "models. Each candidate run logs the model name, number of folds, "
            "cross-validation accuracy, precision, recall, and ROC-AUC. The "
            "fitted candidate pipeline and serialized artifact are also logged.",
            body,
        ),
        paragraph(
            "A final holdout run records the selected model, holdout metrics, "
            "model.joblib, confusion_matrix.csv, and roc_curve.png. This makes "
            "the model choice auditable and allows an evaluator to compare the "
            "candidates through the MLflow UI.",
            body,
        ),
        paragraph(
            "The packaged artifact includes both feature transformations and the "
            "classifier. Inference therefore does not depend on a separately "
            "maintained preprocessing script, reducing the risk of training and "
            "serving skew.",
            body,
        ),
    ])

    page("7", "6. API and Containerisation", [
        paragraph(
            "The FastAPI service exposes /health, /predict, and /metrics. The "
            "prediction endpoint accepts the 13 feature fields, validates their "
            "types and ranges, loads the serialized pipeline, and returns a "
            "prediction, confidence, and model version.",
            body,
        ),
        paragraph(
            "The Dockerfile is based on python:3.12-slim. It installs pinned "
            "dependencies, downloads the dataset, trains the model during image "
            "build, and starts Uvicorn on port 8000. The example request in "
            "examples/predict.json can be posted to the running container.",
            body,
        ),
        paragraph(
            "The service emits request logs and uses Prometheus counters and a "
            "latency histogram. Missing model artifacts return HTTP 503 rather "
            "than a false success response; invalid input returns HTTP 422.",
            body,
        ),
    ])

    page("8", "7. CI/CD and Automated Testing", [
        paragraph(
            "The GitHub Actions workflow in .github/workflows/ci.yml checks out "
            "the repository, installs Python 3.12 dependencies, runs Ruff, "
            "downloads the dataset, runs Pytest, generates EDA plots, trains the "
            "models, and uploads the artifacts directory. Any failed command "
            "fails the workflow.",
            body,
        ),
        paragraph(
            "The tests cover preprocessing with missing values, schema validation "
            "for incomplete datasets, and the API health endpoint. The validated "
            "local test result is 3 passed. Additional tests can be added for "
            "prediction response shape and model-missing behavior.",
            body,
        ),
        paragraph(
            "The CI artifact upload provides a traceable record of generated model "
            "and EDA files for each successful workflow run.",
            body,
        ),
    ])

    page("9", "8. Kubernetes Deployment and Monitoring", [
        paragraph(
            "k8s/deployment.yaml creates a two-replica Deployment and a "
            "LoadBalancer Service. Readiness and liveness probes call /health. "
            "Resource requests and limits are defined to make scheduling behavior "
            "explicit.",
            body,
        ),
        paragraph(
            "For local deployment, Minikube can load the image, apply the "
            "manifest, wait for rollout completion, and expose the service with "
            "minikube service heart-disease-api --url. For cloud deployment, the "
            "image should be pushed to a registry and the image reference changed "
            "in the manifest.",
            body,
        ),
        paragraph(
            "k8s/prometheus.yaml provides a ServiceMonitor for clusters running "
            "Prometheus Operator. Grafana panels should display request volume, "
            "error count, and latency using prediction_requests_total and "
            "prediction_latency_seconds.",
            body,
        ),
    ])

    page("10", "9. Limitations, Ethics, and Submission Evidence", [
        paragraph(
            "The Cleveland dataset is small and historically collected. It may "
            "not represent current clinical practice or diverse populations. "
            "Model confidence is not calibrated clinical risk, and this API must "
            "not be used for diagnosis or treatment.",
            body,
        ),
        paragraph(
            "A production clinical system would require external validation, "
            "calibration, subgroup fairness analysis, access control, TLS, audit "
            "logging, drift monitoring, incident response, and human clinical "
            "oversight.",
            body,
        ),
        paragraph(
            "Submission evidence should include the generated EDA plots, MLflow "
            "comparison, passing CI run, Docker build and API response, "
            "Kubernetes rollout and service, and Prometheus/Grafana metrics. "
            "docs/EVIDENCE_CHECKLIST.md contains the exact capture list and a "
            "short video outline.",
            body,
        ),
        paragraph(
            "<b>Local access:</b> follow docs/SETUP.md. The API is available at "
            "http://localhost:8000 after starting Uvicorn. A public URL is not "
            "claimed because no cloud account or public cluster was provided.",
            body,
        ),
    ])

    page("11", "Appendix A. EDA and Experiment Evidence", [
        paragraph(
            "The following screenshots are generated from the project workspace "
            "and are included as direct submission evidence.",
            body,
        ),
        *evidence_image(
            "01-eda-histograms.png",
            "Figure A1. Feature histograms generated by the EDA script.",
        ),
        *evidence_image(
            "02-class-balance.png",
            "Figure A2. Target class balance for the Cleveland dataset.",
        ),
    ])

    page("12", "Appendix B. Tracking and Continuous Integration Evidence", [
        *evidence_image(
            "03-correlation-heatmap.png",
            "Figure B1. Correlation heatmap generated from the cleaned dataset.",
        ),
        *evidence_image(
            "04.1-mlflow-runs.png",
            "Figure B2. MLflow experiment tracking interface showing the experiment runs.",
        ),
        *evidence_image(
            "04.2-mlflow-runs.png",
            "Figure B3. MLflow model run details and recorded metrics.",
        ),
    ])

    page("13", "Appendix C. Containerised API Evidence", [
        *evidence_image(
            "05-github-actions.png",
            "Figure C1. Successful GitHub Actions workflow.",
        ),
        *evidence_image(
            "06-docker-api.png",
            "Figure C2. FastAPI Swagger UI served from the Docker container.",
        ),
    ])

    doc.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()
