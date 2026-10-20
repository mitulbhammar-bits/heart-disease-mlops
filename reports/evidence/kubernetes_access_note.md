# How the Kubernetes endpoint was reached

- Cluster: Minikube (docker driver) on the course lab host.
- Resources: namespace `heart-risk`, Deployment `heart-api` (2/2 replicas ready), Service `heart-api` of type `LoadBalancer`.
- `minikube tunnel` could not expose the Service on port 80 on this remote host, so the Service kept `EXTERNAL-IP <pending>`.
- Access used instead: `kubectl -n heart-risk port-forward service/heart-api 8002:80`, then
  `python scripts/verify_endpoint.py --url http://127.0.0.1:8002 --output reports/evidence/kubernetes_api_verification.json`.
- Result: /health 200, /ready 200, /metrics 200, /predict 200 and invalid input 422, served by the pods through the Kubernetes Service.
- Limitation: the LoadBalancer external address was not assigned here. On a cloud cluster (GKE/EKS/AKS) the same manifest receives a real external IP.
