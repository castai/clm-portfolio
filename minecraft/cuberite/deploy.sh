#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=minecraft-cubernite
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/configmap.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/deployment.yaml"
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/service.yaml"

echo "Waiting for cuberite rollout..."
kubectl rollout status deployment/cuberite -n "$NAMESPACE" --timeout=120s

echo ""
echo "Done. Resources in namespace '$NAMESPACE':"
kubectl get all -n "$NAMESPACE"
