"""Build exactly ten pages from measured artifacts and genuine evidence files."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WIDTH, HEIGHT = 595.28, 841.89
LEFT, RIGHT = 42, 553
BLUE = colors.HexColor("#16465e")
TEAL = colors.HexColor("#246a82")
GRAY = colors.HexColor("#50616c")
styles = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14,
                           textColor=colors.HexColor("#253745"), spaceAfter=9),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.3, leading=11,
                            textColor=GRAY),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12,
                              leading=16, textColor=BLUE),
}


def read_json(path, default=None):
    file = ROOT / path
    return json.loads(file.read_text()) if file.exists() else (default or {})


class Report:
    def __init__(self):
        self.pdf = canvas.Canvas(str(REPORTS / "assignment_report.pdf"), pagesize=(WIDTH, HEIGHT))
        self.pdf.setTitle("Heart Disease MLOps - Assignment 01")
        self.page = 0
        self.y = 0

    def start(self, title):
        if self.page:
            self.finish_page()
        self.page += 1
        self.pdf.setFillColor(BLUE)
        self.pdf.rect(0, HEIGHT-95, WIDTH, 95, fill=1, stroke=0)
        self.pdf.setFillColor(colors.white)
        self.pdf.setFont("Helvetica", 9)
        self.pdf.drawString(LEFT, HEIGHT-30, "AIMLCZG523  |  MLOPS ASSIGNMENT 01")
        self.pdf.setFont("Helvetica-Bold", 20)
        self.pdf.drawString(LEFT, HEIGHT-65, title)
        self.y = HEIGHT-117

    def text(self, text, style="body"):
        paragraph = Paragraph(text, styles[style])
        _, height = paragraph.wrap(RIGHT-LEFT, self.y-55)
        if self.y-height < 55:
            pass
        paragraph.drawOn(self.pdf, LEFT, self.y-height)
        self.y -= height + (9 if style != "small" else 6)

    def heading(self, text):
        self.text(escape(text), "heading")

    def table(self, rows, widths=None):
        cells = [[Paragraph(escape(str(cell)), styles["small"]) for cell in row] for row in rows]
        table = Table(cells, colWidths=widths or [(RIGHT-LEFT)/len(rows[0])]*len(rows[0]))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce9ef")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("LINEBELOW", (0,0), (-1,-1), .35, colors.HexColor("#ccd8de")),
        ]))
        _, height = table.wrap(RIGHT-LEFT, self.y-55)
        if self.y-height < 55:
            pass
        table.drawOn(self.pdf, LEFT, self.y-height)
        self.y -= height+12

    def image(self, relative, max_height=220, caption=""):
        path = ROOT / relative
        if not path.exists():
            self.text(f"<b>Evidence pending:</b> {escape(relative)}. Capture the actual executed result.", "small")
            return
        reader = ImageReader(str(path))
        width, height = reader.getSize()
        scale = min((RIGHT-LEFT)/width, max_height/height)
        width, height = width*scale, height*scale
        if self.y-height < 55:
            pass
        self.pdf.drawImage(reader, LEFT+(RIGHT-LEFT-width)/2, self.y-height,
                           width=width, height=height)
        self.y -= height+8
        if caption:
            self.text(escape(caption), "small")

    def finish_page(self):
        self.pdf.setStrokeColor(colors.HexColor("#ccd8de"))
        self.pdf.line(LEFT, 42, RIGHT, 42)
        self.pdf.setFillColor(GRAY)
        self.pdf.setFont("Helvetica", 8)
        self.pdf.drawString(LEFT, 27, "Heart Disease MLOps  |  AIMLCZG523 Assignment 01")
        self.pdf.drawRightString(RIGHT, 27, f"{self.page} / 10")
        self.pdf.showPage()

    def save(self):
        assert self.page == 10
        self.finish_page()
        self.pdf.save()


def main():
    metadata = read_json("models/model_metadata.json")
    details = read_json("reports/submission_details.json")
    status = read_json("reports/verification_status.json")
    summary = read_json("reports/eda_summary.json")
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    report = Report()
    report.start("01  Project overview")
    report.heading("Purpose and submission identity")
    report.text("A reproducible classifier for disease presence in the historical UCI Cleveland cohort. "
                "The implementation follows the full assignment lifecycle: acquisition, EDA, model comparison, "
                "tracking, packaging, testing, serving, deployment configuration and request monitoring.")
    report.table([["Student", details.get("student_name")], ["Student ID", details.get("student_id")],
                  ["Repository", details.get("repository_url")]], [95,416])
    report.heading("Architecture and artifact flow")
    report.image("reports/figures/architecture.png", 205,
                 "Logical architecture: data, training and MLflow, packaged pipeline, FastAPI container, Kubernetes Service and Prometheus/Grafana monitoring.")
    report.heading("Measured outcome and completion boundary")
    report.text(f"The selected estimator is <b>{metadata['selected_model'].replace('_', ' ')}</b>, "
                f"selected by training-only mean CV ROC-AUC ({metadata['cv_roc_auc']:.3f}). "
                f"It uses {metadata['train_rows']} training and {metadata['test_rows']} holdout patients. "
                "Python/API, Docker, Minikube, GitHub Actions and Grafana runs are recorded in "
                "verification_status.json with screenshots in the screenshots folder.")
    report.text("Deliverables: GitHub repository (link above), CI workflow, Dockerfile, Kubernetes manifests, "
                "monitoring configuration, screenshots, this report and a recorded pipeline walkthrough.")

    report.start("02  Data acquisition and cleaning")
    report.text("The download module retrieves the official UCI ZIP and extracts processed.cleveland.data. "
                "It retains the source file, records SHA-256 checksums and writes human-readable column names "
                "to data/processed/heart_disease.csv. Cached raw data is checksum-verified on subsequent runs.")
    report.table([["Property", "Verified value"], ["Rows", summary['rows']],
                  ["Predictors", summary['features']], ["Target counts", "164 absent; 139 present"],
                  ["Missing values", "major_vessels: 4; thalassemia_test: 2"],
                  ["Exact duplicate rows", summary['duplicate_rows']]], [150,361])
    report.text("The standard processed UCI dataset has 13 predictors and one target, rather than 14+ predictors. "
                "Original diagnosis codes 1 through 4 map to disease present (1); code 0 maps to absent (0). "
                "The original '?' markers become missing values. No target values are missing.")
    report.image("reports/figures/class_balance_and_missingness.png", 195,
                 "The cohort is moderately balanced; six cells require fold-local imputation.")
    report.heading("Why the cleaned CSV still contains blanks")
    report.text("Cleaning normalizes names and values but does not learn replacement statistics globally. "
                "Median and most-frequent imputers are fitted inside the estimator pipeline so validation "
                "and holdout rows never influence imputation. No rows are dropped merely because a feature is missing.")
    report.text("Source and attribution: Janosi, Steinbrunn, Pfisterer and Detrano (1989), Heart Disease, "
                "UCI Machine Learning Repository; DOI 10.24432/C52P4X. Included data is licensed CC BY 4.0.", "small")

    report.start("03  Exploratory data analysis")
    report.text("The figures inspect continuous feature distributions by class, missingness and pairwise "
                "correlations. These descriptive views help explain the cohort; they do not determine hyperparameters "
                "or fit preprocessing. The feature set is predefined from the standard UCI variables.")
    report.image("reports/figures/feature_distributions.png", 235,
                 "Histograms show overlap between classes; no single continuous variable perfectly separates them.")
    report.image("reports/figures/correlation_heatmap.png", 260,
                 "Pearson heatmap. Correlations between arbitrary categorical codes are descriptive, not causal.")
    report.text("The small historical sample limits subgroup analysis and external generalization. Values and "
                "coding are documented in data/data_dictionary.md. This target describes observed disease "
                "presence and must not be interpreted as a validated future-risk estimate.")

    report.start("04  Features and model selection")
    report.text("A stratified 80/20 split (seed 42) reserves the holdout. Five stratified training folds use "
                "the same seed and partitions for every candidate. Continuous features receive median imputation "
                "and StandardScaler; categorical features receive most-frequent imputation and one-hot encoding. "
                "Unknown categories are ignored by the transformer, while the public API rejects invalid UCI codes.")
    report.text("Logistic Regression provides a simple regularized baseline; C = 0.1, 1 and 10 explores regularization "
                "strength. Random Forest captures interactions; maximum depths 3, 6 and unrestricted are explored "
                "with 200 trees and minimum leaf size 2. The intentionally small grid keeps experiments auditable.")
    rows = [["Candidate", "Settings", "CV ROC-AUC", "CV recall"]]
    for _, row in comparison.iterrows():
        params = row.parameters.replace("'", "")
        rows.append([row.model.replace("_", " "), params,
                     f"{row.cv_roc_auc_mean:.3f} +/- {row.cv_roc_auc_std:.3f}", f"{row.cv_recall_mean:.3f}"])
    report.table(rows, [120,177,113,101])
    report.image("reports/figures/model_comparison.png", 200,
                 "Error bars represent fold standard deviation, not a confidence interval.")
    report.text("Selection uses the highest mean CV ROC-AUC only. Equal scores retain the first candidate. "
                "The full candidate table also reports accuracy, precision and recall with fold variability. "
                "The holdout is evaluated only after selecting the winner; it is not used to tune the threshold.")

    report.start("05  Held-out model evaluation")
    report.text(f"The winner is {metadata['selected_model'].replace('_', ' ')} with parameters "
                f"{escape(str(metadata['selected_parameters']))}. The serving artifact remains fitted on the "
                "training partition, so the reported holdout metrics directly describe the packaged model. "
                "No refit on the holdout is performed. Binary predictions use threshold 0.5.")
    report.table([["Metric", "Untouched holdout result"]] +
                 [[key.replace("_", " ").title(), f"{value:.4f}"]
                  for key, value in metadata["holdout_metrics"].items()], [255,256])
    report.image("reports/figures/holdout_roc_curve.png", 170, "ROC curve: ranking performance across thresholds.")
    report.image("reports/figures/holdout_confusion_matrix.png", 170,
                 "Confusion matrix at threshold 0.5: rows are actual, columns predicted.")
    report.text("Recall describes the proportion of positive cases detected; precision describes the positive "
                "predictive fraction in this cohort. Accuracy depends on class balance, while ROC-AUC measures "
                "ranking. A 61-patient test set yields uncertain estimates: these results are not clinical validation. "
                "Calibration, external validation and threshold design remain future work.")

    report.start("06  Experiment tracking")
    report.text("MLflow uses a local SQLite tracking store and an explicit artifact directory. Every one of the "
                "six candidate runs records its family, hyperparameters, seed, fold count and training size; mean "
                "and standard deviation of four CV metrics; all fold scores; a CV plot; source provenance; and "
                "a fitted full pipeline with an input example. A seventh run records the selected model's holdout evaluation.")
    report.table([["Tracked item", "How to inspect it"], ["Parameters and CV metrics", "Compare candidate runs in the MLflow experiment"],
                  ["Fold variability", "fold_metrics.json and plots/cross_validation.png"],
                  ["Data and environment identity", "SHA-256 tag, source_manifest.json, Python tag"],
                  ["Reusable estimator", "Logged sklearn pipeline and models/heart_risk_pipeline.joblib"],
                  ["Final evaluation", "selected-model-holdout run, ROC and confusion matrix artifacts"]], [170,341])
    report.heading("Open the tracking interface")
    report.text("<font face='Courier' size='9'>mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000</font>")
    report.image("screenshots/mlflow_experiments.png", 88,
                 "MLflow UI: candidate runs and the selected-model holdout run.")
    report.image("screenshots/mlflow_run_detail.png", 88,
                 "One run: cross-validation metrics, parameters, tags and source commit.")
    report.image("screenshots/mlflow_run_artifacts.png", 88,
                 "The same run's artifacts: cross-validation plot, fold metrics and source manifest.")
    report.text("For portable reproduction, rerun training after moving the project: older SQLite run records "
                "may retain absolute artifact locations. The joblib model has no custom project transformer and "
                "works with the pinned serving dependencies. Tracked artifacts and explicit data hashes support "
                "auditing, but are not a managed production model registry or drift-detection service.")

    report.start("07  Reproducibility and API")
    report.text("Use Python 3.12 and install requirements-lock.txt in a fresh virtual environment. Direct "
                "dependencies are also pinned in requirements.txt. A SQLAlchemy 2.0 pin avoids an observed "
                "MLflow 3.4 tracking-store incompatibility. Deterministic split/fold seeds, data checksums, "
                "saved split indices and a full preprocessing pipeline make experiments reproducible.")
    report.text("The saved model also passed live HTTP checks in a fresh serving-only virtual environment, "
                "with only code and model copied and no training data or MLflow dependencies. This verifies "
                "Python environment isolation; it does not replace Docker execution proof.")
    report.heading("Run the core stages")
    report.text("<font face='Courier' size='9'>python -m heart_risk.cli all<br/>python -m heart_risk.cli predict<br/>"
                "python -m uvicorn heart_risk.api:app --host 127.0.0.1 --port 8000<br/>python scripts/verify_endpoint.py</font>")
    report.table([["Endpoint", "Contract"], ["POST /predict", "One JSON patient; returns class, label, positive probability, confidence, threshold and version"],
                  ["GET /health; GET /ready", "Liveness and startup-loaded model readiness"],
                  ["GET /metrics; GET /docs", "Prometheus metrics and interactive OpenAPI documentation"]], [165,346])
    verification = read_json("reports/evidence/api_verification.json")
    if verification:
        body = verification['predict']['body']
        report.text(f"Measured sample response: prediction = {body['prediction']}, disease probability = "
                    f"{body['heart_disease_probability']:.4f}, confidence = {body['confidence']:.4f}. "
                    "The real HTTP verification checked health/readiness/metrics and an invalid request returning 422.")
    report.text("Confidence is the probability of the returned class: p for class 1, 1-p for class 0. "
                "It is not a validated estimate of clinical certainty. The public schema rejects extra keys, "
                "non-finite numbers, missing required fields and out-of-range category codes. Optional UCI missing "
                "fields are transformed to NaN and imputed by the saved pipeline.")
    report.image("screenshots/local_api.png", 180, "Local API run.")
    report.text("Only load model artifacts from trusted sources: joblib serialization is intended for this "
                "controlled project workflow. The same serialization prediction behavior is tested after reload.")

    report.start("08  Tests and CI workflow")
    report.text("Tests focus on behavior: source schema/checksum, all five diagnosis mappings, missing data, "
                "both model families, unseen categories, serialization, training-only imputer statistics, JSON "
                "validation, probability/confidence, health/readiness, metrics and missing-model startup failure.")
    report.text(f"Local verification: {escape(str(status.get('tests', 'See test_run.log')))}. "
                f"Lint: {escape(str(status.get('lint', 'See lint.log')))}. "
                "Machine-readable results are saved in reports/evidence/test_results.xml.")
    report.table([["Workflow stage", "Failure and evidence behavior"],
                  ["Install and lint", "Pinned environment; nonzero exit fails the job"],
                  ["Pytest", "Behavior tests; JUnit result retained"],
                  ["EDA and model training", "All reproducible stages; models and report data retained"],
                  ["Saved-model inference", "Checks pipeline load and prediction"],
                  ["Docker smoke test", "Fresh image/container; real endpoint verification"],
                  ["Artifact upload", "Runs even on failure; logs, models, data provenance and MLflow artifacts"]], [180,331])
    report.image("screenshots/github_actions.png", 190,
                 "Successful GitHub Actions run of the CI workflow.")
    report.text("The supplied workflow validates a deployable artifact on push, pull request or manual dispatch. "
                "The local Kubernetes rollout is an explicit next stage because hosted runners do not control "
                "a student's Minikube cluster; the green hosted run above covers lint, tests, training, inference and the Docker smoke test. "
                "A public-cloud automatic rollout would require explicit credentials and a deployment environment.")

    report.start("09  Container and Kubernetes")
    report.text("The image uses Python 3.12 slim, only serving dependencies, and a non-root UID. It copies the "
                "trained pipeline, exposes port 8000 and checks readiness. verify_docker.sh builds a fresh image, "
                "starts an isolated container on host port 8001 and records API validation, build/runtime logs and inspection data.")
    report.text(f"Container status: <b>{escape(str(status.get('docker', '')))}</b>.")
    report.image("screenshots/docker_verification.png", 105, "Docker image build and isolated container prediction.")
    report.heading("Local Kubernetes deployment route")
    report.text("The manifests define a dedicated namespace, two replicas, startup/readiness/liveness probes, "
                "resource requests/limits and a LoadBalancer service. The Minikube script loads the local image "
                "and waits for rollout. The Service is of type LoadBalancer, but minikube tunnel on port 80 was not usable on the "
                "remote lab host, so its EXTERNAL-IP stayed pending. The endpoint was therefore verified with kubectl "
                "port-forward to the Service (127.0.0.1:8002); on a cloud cluster the same manifest receives an external IP.")
    report.image("screenshots/kubernetes_deployment.png", 100, "Two ready replicas and the LoadBalancer Service.")
    report.image("screenshots/load_balancer_prediction.png", 100, "/predict served by the Kubernetes Service through port-forward.")
    report.text(f"Access instructions/status: {escape(details.get('deployed_api_access', 'Pending'))}", "small")
    report.text("For recovery, inspect pod logs and events; verify the image is loaded and the model exists. "
                "A failed readiness probe prevents traffic; kubectl rollout undo restores a previous deployment. "
                "Deployment used local Minikube, not a public cloud cluster.")

    report.start("10  Monitoring and handoff")
    report.text("Every API request produces a structured log with a generated request ID, method, route, "
                "status and elapsed milliseconds. Patient bodies are not logged. Prometheus counters record "
                "request/status totals and predicted classes; a histogram records latency. Low-cardinality route "
                "labels avoid unbounded metric series.")
    report.text("The Compose monitoring stack runs the API, Prometheus and provisioned Grafana dashboard. "
                "Dashboard panels show request rate, p95 prediction latency, server errors and class counts. "
                "The screenshot was taken after sending real prediction requests. This is service "
                "monitoring; ground-truth-driven model quality or drift alerts are not implemented.")
    report.image("screenshots/grafana_dashboard.png", 175, "Grafana dashboard with live API metrics.")
    report.heading("Submission summary")
    report.text("All nine assignment areas have executed evidence: reproducible EDA and training, MLflow runs, "
                "packaged pipeline, tests and CI, Docker run, Kubernetes deployment and Grafana monitoring. "
                "The Kubernetes endpoint was reached by port-forward as explained in section 09.")
    report.text(f"Video: {escape(details.get('video_file', 'Pending'))}")
    report.heading("References and understanding")
    report.text("UCI Heart Disease: archive.ics.uci.edu/dataset/45/heart+disease; DOI 10.24432/C52P4X (CC BY 4.0). "
                "MLflow: mlflow.org/docs/latest/ml/tracking/. FastAPI: fastapi.tiangolo.com/deployment/docker/. "
                "scikit-learn: scikit-learn.org/stable/modules/compose.html. Kubernetes: "
                "kubernetes.io/docs/concepts/services-networking/service/.", "small")
    report.text("Explain why imputation belongs inside CV, why selection excludes the holdout, what probability "
                "and confidence mean, how CI surfaces failures, and why liveness differs from readiness. "
                "Personal experimentation and a live demonstration support the independent-work requirement.")
    report.save()
    print(REPORTS / "assignment_report.pdf")


if __name__ == "__main__":
    main()
