#!/usr/bin/env bash
#
# setup.sh — Deploy Vault (dev mode) + Webhook Injector + CSI Provider
#             and two test workloads that continuously hash their mounted
#             secrets so you can validate integrity after pod migration.
#
# Usage:
#   ./setup.sh          # deploy everything
#   ./setup.sh teardown  # remove everything
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Configuration ────────────────────────────────────────────────────────────
VAULT_NAMESPACE="vault"
TEST_NAMESPACE="vault-test"
VAULT_DEV_ROOT_TOKEN="root"
VAULT_RELEASE="vault"
CSI_DRIVER_RELEASE="csi-driver"
SECRET_PATH="secret/test-secret"
VAULT_ROLE="test-app-role"
VAULT_POLICY="test-app-policy"
SERVICE_ACCOUNT="vault-test-sa"

# Test secret values — change these to whatever you want
SECRET_USERNAME="vault-test-user"
SECRET_PASSWORD="s3cr3t-p@ssw0rd-12345"
SECRET_API_KEY="ak_7f8e9d0c1b2a3456"

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }
header() {
	echo ""
	echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
	echo -e "${CYAN}  $*${NC}"
	echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
}

# ─── Teardown ─────────────────────────────────────────────────────────────────
teardown() {
	header "Tearing down vault-secret-migration-test"

	info "Deleting test workloads in ${TEST_NAMESPACE}..."
	kubectl delete -f "${SCRIPT_DIR}/deploy-injector-test.yaml" --ignore-not-found 2>/dev/null || true
	kubectl delete -f "${SCRIPT_DIR}/deploy-csi-test.yaml" --ignore-not-found 2>/dev/null || true
	kubectl delete -f "${SCRIPT_DIR}/secret-provider-class.yaml" --ignore-not-found 2>/dev/null || true
	kubectl delete -f "${SCRIPT_DIR}/serviceaccount.yaml" --ignore-not-found 2>/dev/null || true

	info "Uninstalling Vault Helm release..."
	helm uninstall "${VAULT_RELEASE}" -n "${VAULT_NAMESPACE}" 2>/dev/null || true

	info "Uninstalling Secrets Store CSI Driver Helm release..."
	helm uninstall "${CSI_DRIVER_RELEASE}" -n kube-system 2>/dev/null || true

	info "Deleting namespaces..."
	kubectl delete namespace "${TEST_NAMESPACE}" --ignore-not-found 2>/dev/null || true
	kubectl delete namespace "${VAULT_NAMESPACE}" --ignore-not-found 2>/dev/null || true

	ok "Teardown complete."
	exit 0
}

# Handle teardown argument
if [[ "${1:-}" == "teardown" ]]; then
	teardown
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────────
header "Pre-flight checks"

for cmd in kubectl helm; do
	if ! command -v "$cmd" &>/dev/null; then
		err "$cmd is required but not found in PATH"
		exit 1
	fi
done
ok "kubectl and helm found"

# Verify cluster connectivity
if ! kubectl cluster-info &>/dev/null; then
	err "Cannot connect to Kubernetes cluster. Check your kubeconfig / context."
	exit 1
fi

CONTEXT=$(kubectl config current-context)
ok "Connected to cluster (context: ${CONTEXT})"

# ─── Phase 1: Install Infrastructure ─────────────────────────────────────────
header "Phase 1: Install Infrastructure"

# Add Helm repos
info "Adding Helm repositories..."
helm repo add secrets-store-csi-driver \
	https://kubernetes-sigs.github.io/secrets-store-csi-driver/charts 2>/dev/null || true
helm repo add hashicorp \
	https://helm.releases.hashicorp.com 2>/dev/null || true
helm repo update
ok "Helm repos updated"

# Create namespaces
info "Creating namespaces..."
kubectl create namespace "${VAULT_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace "${TEST_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
ok "Namespaces ready: ${VAULT_NAMESPACE}, ${TEST_NAMESPACE}"

# Install Secrets Store CSI Driver
info "Installing Secrets Store CSI Driver..."
helm upgrade --install "${CSI_DRIVER_RELEASE}" \
	secrets-store-csi-driver/secrets-store-csi-driver \
	--namespace kube-system \
	--set syncSecret.enabled=true \
	--wait \
	--timeout 3m
ok "Secrets Store CSI Driver installed"

# Install Vault (dev mode + injector + CSI provider)
info "Installing Vault (dev mode) with injector + CSI provider..."
helm upgrade --install "${VAULT_RELEASE}" hashicorp/vault \
	--namespace "${VAULT_NAMESPACE}" \
	--set "server.dev.enabled=true" \
	--set "server.dev.devRootToken=${VAULT_DEV_ROOT_TOKEN}" \
	--set "injector.enabled=true" \
	--set "csi.enabled=true" \
	--wait \
	--timeout 5m
ok "Vault Helm release installed"

# Wait for vault-0 to be fully ready
info "Waiting for vault-0 pod to be ready..."
kubectl wait --for=condition=ready pod/vault-0 \
	-n "${VAULT_NAMESPACE}" \
	--timeout=120s
ok "vault-0 is ready"

# Wait for injector
info "Waiting for vault-agent-injector to be ready..."
kubectl rollout status deployment/vault-agent-injector \
	-n "${VAULT_NAMESPACE}" \
	--timeout=120s
ok "vault-agent-injector is ready"

# Wait for CSI provider
info "Waiting for vault-csi-provider to be ready..."
kubectl rollout status daemonset/vault-csi-provider \
	-n "${VAULT_NAMESPACE}" \
	--timeout=120s
ok "vault-csi-provider is ready"

# ─── Phase 2: Configure Vault ────────────────────────────────────────────────
header "Phase 2: Configure Vault"

# Helper to run vault commands inside vault-0
vault_exec() {
	kubectl exec vault-0 -n "${VAULT_NAMESPACE}" -- \
		env VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}" \
		vault "$@"
}

# Enable KV v2 (dev mode already has it at secret/, but this is idempotent)
info "Enabling KV v2 secrets engine at secret/..."
vault_exec secrets enable -path=secret -version=2 kv 2>/dev/null || {
	warn "KV v2 already enabled at secret/ (expected in dev mode)"
}

# Write the test secret
info "Writing test secret to ${SECRET_PATH}..."
vault_exec kv put "secret/test-secret" \
	"username=${SECRET_USERNAME}" \
	"password=${SECRET_PASSWORD}" \
	"api_key=${SECRET_API_KEY}"
ok "Secret written"

# Verify the secret is readable
info "Verifying secret..."
vault_exec kv get "secret/test-secret"
ok "Secret verified"

# Enable Kubernetes auth
info "Enabling Kubernetes auth method..."
vault_exec auth enable kubernetes 2>/dev/null || {
	warn "Kubernetes auth already enabled"
}

# Configure Kubernetes auth to use the in-cluster service
# vault-0 is running inside K8s, so it can auto-detect the K8s API
info "Configuring Kubernetes auth..."
vault_exec write auth/kubernetes/config \
	kubernetes_host="https://kubernetes.default.svc.cluster.local:443"
ok "Kubernetes auth configured"

# Create policy
info "Creating Vault policy: ${VAULT_POLICY}..."
kubectl exec vault-0 -n "${VAULT_NAMESPACE}" -- \
	env VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}" \
	sh -c "vault policy write ${VAULT_POLICY} - <<'POLICY'
path \"secret/data/test-secret\" {
  capabilities = [\"read\"]
}
POLICY"
ok "Policy created"

# Create role
info "Creating Vault role: ${VAULT_ROLE}..."
vault_exec write "auth/kubernetes/role/${VAULT_ROLE}" \
	"bound_service_account_names=${SERVICE_ACCOUNT}" \
	"bound_service_account_namespaces=${TEST_NAMESPACE}" \
	"policies=${VAULT_POLICY}" \
	"ttl=24h"
ok "Role created"

# ─── Phase 3: Deploy Test Workloads ──────────────────────────────────────────
header "Phase 3: Deploy Test Workloads"

info "Applying ServiceAccount..."
kubectl apply -f "${SCRIPT_DIR}/serviceaccount.yaml"
ok "ServiceAccount applied"

info "Applying SecretProviderClass..."
kubectl apply -f "${SCRIPT_DIR}/secret-provider-class.yaml"
ok "SecretProviderClass applied"

info "Deploying CSI secret test..."
kubectl apply -f "${SCRIPT_DIR}/deploy-csi-test.yaml"

info "Deploying Injector secret test..."
kubectl apply -f "${SCRIPT_DIR}/deploy-injector-test.yaml"

# Wait for CSI test pod
info "Waiting for CSI test deployment to be ready..."
kubectl rollout status deployment/csi-secret-test \
	-n "${TEST_NAMESPACE}" \
	--timeout=180s
ok "CSI test deployment ready"

# Wait for Injector test pod (takes longer due to sidecar injection)
info "Waiting for Injector test deployment to be ready..."
kubectl rollout status deployment/injector-secret-test \
	-n "${TEST_NAMESPACE}" \
	--timeout=180s
ok "Injector test deployment ready"

# ─── Phase 4: Validate ───────────────────────────────────────────────────────
header "Phase 4: Initial Validation"

# Give pods a moment to compute their first hash
sleep 15

echo ""
info "=== CSI Test Pod Logs ==="
kubectl logs -n "${TEST_NAMESPACE}" -l app=csi-secret-test --tail=20
echo ""

info "=== Injector Test Pod Logs (app container) ==="
kubectl logs -n "${TEST_NAMESPACE}" -l app=injector-secret-test -c app --tail=20
echo ""

# ─── Summary ──────────────────────────────────────────────────────────────────
header "Setup Complete"

cat <<EOF

  Components deployed:
    kube-system    : ${CSI_DRIVER_RELEASE} (Secrets Store CSI Driver)
    ${VAULT_NAMESPACE}          : ${VAULT_RELEASE} (Vault dev server + injector + CSI provider)
    ${TEST_NAMESPACE}     : csi-secret-test (hashes /mnt/secrets-store/*)
    ${TEST_NAMESPACE}     : injector-secret-test (hashes /vault/secrets/*)

  Secret stored at: ${SECRET_PATH}
    username = ${SECRET_USERNAME}
    password = ${SECRET_PASSWORD}
    api_key  = ${SECRET_API_KEY}

  ──────────────────────────────────────────────────────────────
  HOW TO VALIDATE AFTER POD MIGRATION
  ──────────────────────────────────────────────────────────────

  1. Record baseline hashes:
     kubectl logs -n ${TEST_NAMESPACE} -l app=csi-secret-test --tail=3
     kubectl logs -n ${TEST_NAMESPACE} -l app=injector-secret-test -c app --tail=3

  2. Trigger migration (e.g., drain the node, delete the pod, etc.):
     kubectl delete pod -n ${TEST_NAMESPACE} -l app=csi-secret-test
     kubectl delete pod -n ${TEST_NAMESPACE} -l app=injector-secret-test

  3. Wait for new pods, then check logs again:
     kubectl logs -n ${TEST_NAMESPACE} -l app=csi-secret-test --tail=5
     kubectl logs -n ${TEST_NAMESPACE} -l app=injector-secret-test -c app --tail=5

  4. Compare: secret_hash must be identical. Status should be "OK".
     If you see "DRIFT_DETECTED", the secret content changed.

  To tear down everything:
     ./setup.sh teardown

EOF
