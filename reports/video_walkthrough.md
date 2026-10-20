# Record a 2-4 minute overall-pipeline video

Use macOS Shift-Command-5 / Windows screen recorder. Record the actual terminal
and browser screens; narrate in your own words. Avoid displaying tokens/passwords.

1. Show your repository and explain the module map (20 seconds).
2. Run `python -m heart_risk.cli all`; show data provenance and EDA (40 seconds).
3. Show MLflow candidates, selection by training-only CV ROC-AUC, and holdout results (30 seconds).
4. Run tests; show a successful GitHub Actions run and stored artifacts (30 seconds).
5. Show Docker verification and explain that the serialized pipeline includes preprocessing (20 seconds).
6. Show two healthy Kubernetes replicas and call /predict via kubectl port-forward to the LoadBalancer Service and mention the tunnel limitation (30 seconds).
7. Send requests and show Grafana request rate/latency; finish with limitations (20 seconds).

Save your final recording as `reports/pipeline_walkthrough.mp4` (or `.mov`).
A local-only recording is useful preparation but does not replace deployment/CI evidence.
