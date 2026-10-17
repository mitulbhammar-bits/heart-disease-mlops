#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v minikube >/dev/null || { echo "Install Minikube, kubectl and Docker Desktop first."; exit 1; }
minikube start --driver=docker
docker build -t heart-risk-api:1.0.0 .
minikube image load heart-risk-api:1.0.0
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl -n heart-risk rollout status deployment/heart-api --timeout=180s
mkdir -p reports/evidence
kubectl -n heart-risk get pods,deployments,services -o wide | tee reports/evidence/kubernetes_status.txt
echo "In another terminal run: minikube tunnel"
echo "Then get the LoadBalancer address: kubectl -n heart-risk get service heart-api"
echo "Verify with: python scripts/verify_endpoint.py --url http://LOAD_BALANCER_IP --output reports/evidence/kubernetes_api_verification.json"
echo "Optional local diagnostic access: kubectl -n heart-risk port-forward service/heart-api 8002:80"
