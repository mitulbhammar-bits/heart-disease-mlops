# Capture real screenshots

PNG files below are automatically included by `scripts/build_report.py` if present.
Do not substitute fabricated screenshots or drawings for executed evidence.

- `mlflow_experiments.png`: MLflow UI showing candidate comparison and real runs.
- `github_actions.png`: Your repository's successful workflow and steps.
- `docker_verification.png`: Docker build plus actual /predict response.
- `kubernetes_deployment.png`: Real `kubectl get pods,deployments,services` output.
- `load_balancer_prediction.png`: Actual HTTP prediction via the LoadBalancer address.
- `grafana_dashboard.png`: Dashboard with metrics after real requests.

Local API/MLflow screenshots may be supplied during preparation. CI/Kubernetes
screenshots must come from your actual runs. Rendered charts are figures, not
screenshots of deployment. Save any additional screenshots with readable names.
