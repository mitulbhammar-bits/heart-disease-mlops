# Heart Disease MLOps Learning Project

A small, modular end-to-end implementation for **AIMLCZG523 Assignment 01**.
Start with this README, then `reports/submission_checklist.md`.

## What is ready

Official UCI Cleveland data and provenance, EDA figures, two classifier families,
five-fold experiment comparisons, MLflow tracking, a saved preprocessing/model
pipeline, FastAPI, tests, GitHub Actions, Docker, Kubernetes and monitoring configuration.
`reports/verification_status.json` records what was executed (Python/API, Docker, Minikube,
GitHub Actions, Grafana). Screenshots are in `screenshots/`, raw evidence in `reports/evidence/`,
and the final report is `reports/assignment_report.pdf` (10 pages).

## Install and reproduce (Python 3.12)

Run all commands from the repository root. macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m heart_risk.cli all
python -m ruff check .
python -m pytest -q --junitxml=reports/evidence/test_results.xml
python -m heart_risk.cli predict
```

Windows: use `py -3.12 -m venv .venv`, then `.venv\Scripts\activate`.
`requirements.txt` pins direct dependencies; `requirements-lock.txt` pins the full
resolved environment. The venv is not included in the submission.
Raw data is cached and checksum-verified, so `all` works offline after installation.
Run `python -m heart_risk.cli download --refresh` only to deliberately refresh the source.

Individual stages: `download`, `eda`, `train`, `predict`. No notebook is required by
the rubric because reproducible scripts are accepted.

## Serve and inspect locally

```bash
python -m uvicorn heart_risk.api:app --host 127.0.0.1 --port 8000
# In another activated terminal:
python scripts/verify_endpoint.py
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data @examples/patient.json
```

Open `http://127.0.0.1:8000/docs` for interactive API docs and
`http://127.0.0.1:8000/metrics` for Prometheus metrics. `/health` is liveness;
`/ready` confirms the model loaded during startup. Missing model fails startup.
Confidence is the predicted class probability, not a clinical reliability guarantee.
The optional `major_vessels` and `thalassemia_test` fields allow null and are imputed.
All other fields are required. Extra fields and invalid values return HTTP 422.
See `data/data_dictionary.md` for original UCI categorical codes.

## Explore actual tracked experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, choose `heart-risk-model-comparison`, compare CV
metrics, inspect fold plots, logged parameters, data checksum and pipeline artifacts.
SQLite/artifact paths reference the directory in which experiments were run.
After moving this project, regenerate fresh runs with `python -m heart_risk.cli train`;
old runs may retain paths to the preparation machine. The joblib serving model is portable.

## Prove the isolated Docker container

Install/start Docker Desktop, then:

```bash
bash scripts/verify_docker.sh
```

This builds the image, starts a fresh container on port 8001, checks readiness,
prediction, invalid-input handling and metrics, and saves build/runtime logs and
container inspection evidence. It removes its own test container on exit.
The image runs as a non-root user and contains the trained pipeline and serving dependencies.
Build after training; retrain before packaging if model dependencies change.
The image installs `requirements-serving-lock.txt` to pin serving dependencies transitively.

## Deploy to local Kubernetes (rubric-approved route)

Install Docker Desktop, Minikube and kubectl:

```bash
bash scripts/deploy_minikube.sh
# In a second terminal (may request local OS permission):
minikube tunnel
# In your project terminal:
kubectl -n heart-risk get service heart-api
python scripts/verify_endpoint.py --url http://LOAD_BALANCER_IP --output reports/evidence/kubernetes_api_verification.json
```

Replace `LOAD_BALANCER_IP` with the real service address. The service exposes port
80 to two replicas with probes and resource limits.
**Note:** on the remote lab host `minikube tunnel` could not expose port 80, so the
Service stayed `EXTERNAL-IP <pending>`. The deployed API was verified through the Service with
`kubectl -n heart-risk port-forward service/heart-api 8002:80` and
`python scripts/verify_endpoint.py --url http://127.0.0.1:8002`
(see `reports/evidence/kubernetes_access_note.md`).
For a public cluster, push the image to your registry and change the manifest's
image name; public hosting is optional for this assignment.

## Monitoring

```bash
export GRAFANA_ADMIN_PASSWORD='choose-your-own-local-password'
docker compose up --build -d
# Generate real requests from another terminal:
for i in {1..20}; do curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data @examples/patient.json; done
```

Prometheus: `http://localhost:9090`; Grafana: `http://localhost:3000`
(user `admin`, password from the environment). The provisioned dashboard displays
request rate, p95 prediction latency, 5xx errors and predictions by class.
Logs record route/status/request ID/duration without logging patient input.
This Compose stack monitors its own Docker API. Kubernetes annotations alone do
not install a Prometheus scraper; do not claim Kubernetes monitoring without configuring one.
Stop the local stack with `docker compose down`.

## CI workflow and repository

Repository: https://github.com/mitulbhammar-bits/heart-disease-mlops
The workflow (`.github/workflows/ci.yaml`) runs lint, tests, training, saved-model inference and Docker smoke tests;
failures stop the job and artifacts are uploaded even on failure. It validates a
deployable artifact. Kubernetes rollout is deliberately an explicit local step
so CI does not assume access to a student's workstation cluster.
A successful run is captured in `screenshots/github_actions.png`; artifacts are uploaded on every run.

## Submission contents

- Report: `reports/assignment_report.pdf` (10 pages); rebuild with `python scripts/build_report.py`.
- Evidence screenshots: `screenshots/`. Raw verification output: `reports/evidence/`.
- Deployed API access: local Minikube Service, reached with `kubectl port-forward` (see above).
- Pipeline video: `reports/pipeline_walkthrough.mp4`.

## Module map

| File | Responsibility |
|---|---|
| `heart_risk/data.py` | Download, clean, checksum and target mapping |
| `heart_risk/features.py` | Fold-local imputation, scaling and encoding |
| `heart_risk/models.py` | Two model families and six candidates |
| `heart_risk/exploration.py` | EDA figures and summary |
| `heart_risk/training.py` | CV selection, MLflow, holdout evaluation, packaging |
| `heart_risk/api.py` | JSON validation, prediction, request logs and metrics |
| `heart_risk/cli.py` | Human-readable stage commands |
| `test/` | Data, model and API behavior tests |
| `scripts/` | Verification, deployment, report and walkthrough helpers |

## Sources and limitations

Dataset: Janosi, Steinbrunn, Pfisterer & Detrano (1989), Heart Disease,
[UCI](https://archive.ics.uci.edu/dataset/45/heart+disease),
[DOI](https://doi.org/10.24432/C52P4X), CC BY 4.0. Credit applies to the included raw/clean data.
Implementation references: [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/),
[FastAPI containers](https://fastapi.tiangolo.com/deployment/docker/),
[scikit-learn pipeline](https://scikit-learn.org/stable/modules/compose.html),
[Kubernetes services](https://kubernetes.io/docs/concepts/services-networking/service/).

This historical cohort has 303 rows and 13 predictors. The assignment describes
14+ features, while UCI's standard processed dataset has 14 columns including the
target. No artificial predictors are invented to satisfy that wording.
The target is disease presence, not future incidence. The default threshold is
0.5, not clinically optimized. CV fold standard deviations are variability
summaries, not confidence intervals. This is an educational API, not a clinical tool.

## Included local walkthrough

`reports/local_pipeline_walkthrough.mp4` is an 80-second captioned walkthrough of
actual local results and captured API/MLflow screens. It is not the final narrated
Docker/Kubernetes/CI recording. Rebuild it with `python scripts/build_local_video.py`.
Rebuild the report with `python scripts/build_report.py`.
