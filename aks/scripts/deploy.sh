#!/bin/bash
set -euo pipefail

# ============================================================================
# Deploy Risk Manager to Kubernetes
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")/k8s"

echo "============================================"
echo "  Risk Manager - Deploy to Kubernetes"
echo "============================================"

# Check kubectl context
echo ""
echo "Current kubectl context:"
kubectl config current-context
echo ""

read -p "Continue with this context? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo "Aborted."
	exit 1
fi

# Apply manifests in order
echo ""
echo "[1/6] Creating namespace..."
kubectl apply -f "${K8S_DIR}/namespace.yaml"

echo ""
echo "[2/6] Checking secrets..."
if kubectl get secret azure-sql-secret -n risk-manager >/dev/null 2>&1; then
	echo "  Secret 'azure-sql-secret' already exists (created by setup-azure-sql.sh). Skipping."
else
	echo "  No secret found. Applying placeholder from secret.yaml..."
	echo "  WARNING: Edit k8s/secret.yaml with real credentials, or run setup-azure-sql.sh first."
	kubectl apply -f "${K8S_DIR}/secret.yaml"
fi

echo ""
echo "[3/6] Deploying backend..."
kubectl apply -f "${K8S_DIR}/backend-deployment.yaml"
kubectl apply -f "${K8S_DIR}/backend-service.yaml"

echo ""
echo "[4/6] Deploying frontend..."
kubectl apply -f "${K8S_DIR}/frontend-deployment.yaml"
kubectl apply -f "${K8S_DIR}/frontend-service.yaml"

echo ""
echo "[5/6] Applying Pod Disruption Budgets..."
kubectl apply -f "${K8S_DIR}/pdb.yaml"

echo ""
echo "[6/6] Waiting for rollout..."
kubectl rollout status deployment/risk-manager-backend -n risk-manager --timeout=120s
kubectl rollout status deployment/risk-manager-frontend -n risk-manager --timeout=120s

echo ""
echo "============================================"
echo "  Deployment complete!"
echo ""
echo "  Pods:"
kubectl get pods -n risk-manager -o wide
echo ""
echo "  Services:"
kubectl get svc -n risk-manager
echo ""
echo "  Frontend external IP (may take a minute):"
kubectl get svc risk-manager-frontend -n risk-manager -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
echo ""
echo "============================================"
