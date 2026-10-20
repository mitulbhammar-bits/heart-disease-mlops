# Rubric and evidence checklist (50 marks total)

These are coverage targets, not a promised grade. Use actual evidence, not configuration alone.

| Area | Marks | Prepared artifacts | Evidence |
|---|---:|---|---|
| Acquisition and EDA | 5 | raw/clean data, provenance, download module, three EDA figures | Explain target and missing data |
| Features and models | 8 | fold-local preprocessing, 2 families, 6 candidates, CV/holdout metrics | Explain tuning and winner |
| Experiment tracking | 5 | SQLite MLflow, parameter/metric/model/plot artifacts | MLflow UI screenshot |
| Reproducibility | 7 | joblib pipeline, metadata, pins, full lock, sample input | Fresh environment and Docker verification |
| CI/CD and testing | 8 | behavioral tests; lint/test/train/Docker GitHub workflow | Own repo URL; real green Actions run screenshot |
| Container | 5 | non-root Dockerfile, build/run verification script | Successful image build and isolated HTTP proof |
| Deployment | 7 | 2-replica manifests; LoadBalancer; Minikube script | Live rollout, LB response, deployment screenshots |
| Monitoring/logging | 3 | request logs, /metrics, Prometheus, Grafana provisioning | Real dashboard after requests |
| Documentation | 2 | README, architecture, exactly 10-page PDF, video guide | Personal details, repo link, screenshots, final video |

## Final submission gates

- [x] Add student details and repository URL to `submission_details.json`.
- [x] Re-run lint/tests/training from a clean Python 3.12 venv.
- [x] Verify a new Docker container; retain real build and runtime evidence.
- [x] Deploy in Minikube; verified via port-forward because tunnel on port 80 failed (documented).
- [x] Capture actual CI, deployment, MLflow, API and Grafana screenshots.
- [ ] Record a short complete-pipeline video with your own explanation.
- [x] Regenerate the PDF and check it has exactly 10 pages.
- [x] Remove or explain every pending-evidence notice in the report.
- [ ] Understand the code, personalize decisions and follow course AI-use rules.
