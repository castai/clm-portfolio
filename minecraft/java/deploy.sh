#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=minecraft-java
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/nginx-configmap.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/deployment.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/nginx-deployment.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/service-public.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/service-headless.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/nginx-service.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/ingress.yaml"

echo "Waiting for minecraft-java rollout..."
kubectl rollout status deployment/minecraft-java -n "$NAMESPACE" --timeout=120s

echo "Waiting for minecraft-java-nginx rollout..."
kubectl rollout status deployment/minecraft-java-nginx -n "$NAMESPACE" --timeout=60s

echo ""
echo "Done. Resources in namespace '$NAMESPACE':"
kubectl get all -n "$NAMESPACE"
