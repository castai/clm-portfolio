#!/usr/bin/env bash
#
# setup.sh — Deploy Coder on AWS EKS and create a workspace that builds
#             the Linux kernel from source. The workspace pod carries CAST AI
#             live-migration labels so it can be migrated with zero downtime.
#
# Usage:
#   ./setup.sh               # deploy Coder + PostgreSQL + create workspace
#   ./setup.sh build          # trigger kernel build inside workspace
#   ./setup.sh teardown       # remove everything
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Configuration ────────────────────────────────────────────────────────────
NAMESPACE="coder"
PG_NAMESPACE="coder"
PG_PASSWORD="coder"
PG_USER="coder"
PG_DB="coder"
CODER_TEMPLATE_NAME="kernel-build"
CODER_WORKSPACE_NAME="kernel-build"
CODER_FIRST_USER_EMAIL="admin@coder.local"
CODER_FIRST_USER_USERNAME="admin"
CODER_FIRST_USER_PASSWORD="castailive!"

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

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

wait_for_rollout() {
	local name="$1"
	local ns="$2"
	local timeout="${3:-300}"
	info "Waiting for ${name} to be ready (timeout: ${timeout}s)..."
	if ! kubectl rollout status "${name}" -n "${ns}" --timeout="${timeout}s" 2>/dev/null; then
		err "${name} did not become ready within ${timeout}s"
		kubectl get pods -n "${ns}" -o wide
		exit 1
	fi
	ok "${name} is ready"
}

# ─── Teardown ─────────────────────────────────────────────────────────────────
teardown() {
	header "Tearing down Coder live-migration test"

	# Kill any port-forward on coder port
	lsof -ti:6770 | xargs kill -9 2>/dev/null || true

	# Delete workspace via Coder CLI if available
	if command -v coder &>/dev/null; then
		info "Deleting Coder workspace..."
		coder delete "${CODER_WORKSPACE_NAME}" --yes 2>/dev/null || true
		info "Deleting Coder template..."
		coder templates delete "${CODER_TEMPLATE_NAME}" --yes 2>/dev/null || true
	fi

	# Uninstall Coder Helm release
	info "Uninstalling Coder Helm release..."
	helm uninstall coder -n "${NAMESPACE}" 2>/dev/null || true

	# Uninstall PostgreSQL Helm release
	info "Uninstalling PostgreSQL..."
	helm uninstall postgresql -n "${PG_NAMESPACE}" 2>/dev/null || true

	# Clean up secrets and PVCs
	info "Cleaning up secrets..."
	kubectl delete secret coder-db-url -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true

	info "Cleaning up PVCs..."
	kubectl delete pvc --all -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true

	info "Deleting namespace..."
	kubectl delete namespace "${NAMESPACE}" --ignore-not-found 2>/dev/null || true

	ok "Teardown complete."
	exit 0
}

# ─── Build command ────────────────────────────────────────────────────────────
build() {
	header "Triggering kernel build"

	# Try Coder CLI first, fall back to kubectl
	if command -v coder &>/dev/null; then
		info "Using Coder CLI to run build-kernel..."
		coder ssh "${CODER_WORKSPACE_NAME}" -- build-kernel
	else
		POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
			-l "app.kubernetes.io/name=coder-workspace" \
			-o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

		if [[ -z "${POD_NAME}" ]]; then
			err "No workspace pod found. Is the workspace deployed?"
			exit 1
		fi

		info "Using kubectl exec on pod: ${POD_NAME}"
		kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- build-kernel
	fi

	ok "Build command completed."
}

# ─── Deploy ───────────────────────────────────────────────────────────────────
deploy() {
	# ─── Pre-flight checks ────────────────────────────────────────────────────
	header "Pre-flight checks"

	for tool in kubectl helm; do
		if ! command -v "${tool}" &>/dev/null; then
			err "${tool} is required but not found in PATH"
			exit 1
		fi
		ok "${tool} found"
	done

	if ! kubectl cluster-info &>/dev/null; then
		err "Cannot connect to Kubernetes cluster. Check your kubeconfig."
		exit 1
	fi
	CONTEXT=$(kubectl config current-context)
	ok "Connected to cluster (context: ${CONTEXT})"

	# EBS CSI driver
	info "Checking EBS CSI driver..."
	if ! kubectl get csidriver ebs.csi.aws.com &>/dev/null; then
		err "EBS CSI driver not found. Install it with:"
		err "  aws eks create-addon --addon-name aws-ebs-csi-driver --cluster-name <cluster>"
		exit 1
	fi
	ok "EBS CSI driver installed"

	# StorageClass
	info "Checking StorageClass 'gp2'..."
	if ! kubectl get storageclass gp2 &>/dev/null; then
		err "StorageClass 'gp2' not found."
		kubectl get storageclass -o name
		exit 1
	fi
	ok "StorageClass 'gp2' available"

	# ─── Phase 1: PostgreSQL ──────────────────────────────────────────────────
	header "Phase 1: PostgreSQL"

	info "Creating namespace..."
	kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
	ok "Namespace ready: ${NAMESPACE}"

	# Check if PostgreSQL is already running
	if helm status postgresql -n "${PG_NAMESPACE}" &>/dev/null; then
		ok "PostgreSQL already installed — skipping"
	else
		info "Adding Bitnami Helm repo..."
		helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
		helm repo update bitnami

		info "Installing PostgreSQL..."
		helm install postgresql bitnami/postgresql \
			--namespace "${PG_NAMESPACE}" \
			--set auth.username="${PG_USER}" \
			--set auth.password="${PG_PASSWORD}" \
			--set auth.database="${PG_DB}" \
			--set primary.persistence.size=10Gi \
			--set primary.persistence.storageClass=gp2 \
			--wait --timeout 300s

		ok "PostgreSQL installed"
	fi

	wait_for_rollout "statefulset/postgresql" "${PG_NAMESPACE}" 300

	# Create DB URL secret
	DB_URL="postgres://${PG_USER}:${PG_PASSWORD}@postgresql.${PG_NAMESPACE}.svc.cluster.local:5432/${PG_DB}?sslmode=disable"
	kubectl create secret generic coder-db-url \
		-n "${NAMESPACE}" \
		--from-literal=url="${DB_URL}" \
		--dry-run=client -o yaml | kubectl apply -f -
	ok "Database secret created"

	# ─── Phase 2: Coder control plane ─────────────────────────────────────────
	header "Phase 2: Coder Control Plane"

	if helm status coder -n "${NAMESPACE}" &>/dev/null; then
		ok "Coder already installed — upgrading"
		helm upgrade coder coder-v2/coder \
			--namespace "${NAMESPACE}" \
			--values "${SCRIPT_DIR}/values.yaml" \
			--wait --timeout 300s
	else
		info "Adding Coder Helm repo..."
		helm repo add coder-v2 https://helm.coder.com/v2 2>/dev/null || true
		helm repo update coder-v2

		info "Installing Coder..."
		helm install coder coder-v2/coder \
			--namespace "${NAMESPACE}" \
			--values "${SCRIPT_DIR}/values.yaml" \
			--wait --timeout 300s
	fi
	ok "Coder Helm release ready"

	wait_for_rollout "deployment/coder" "${NAMESPACE}" 300

	# ─── Phase 3: Port-forward & CLI setup ────────────────────────────────────
	header "Phase 3: Port-forward & CLI Setup"

	CODER_PORT=6770
	CODER_URL="http://localhost:${CODER_PORT}"

	# Kill any existing port-forward on this port
	lsof -ti:${CODER_PORT} | xargs kill -9 2>/dev/null || true

	info "Starting kubectl port-forward to Coder on localhost:${CODER_PORT}..."
	nohup kubectl port-forward svc/coder -n "${NAMESPACE}" "${CODER_PORT}:80" &>/dev/null &
	PORT_FORWARD_PID=$!
	disown ${PORT_FORWARD_PID} 2>/dev/null || true

	# Wait for port-forward + API to be ready
	info "Waiting for Coder API..."
	for i in $(seq 1 30); do
		if curl -sf "${CODER_URL}/api/v2/buildinfo" &>/dev/null; then
			break
		fi
		echo -n "."
		sleep 2
	done
	echo ""

	if ! curl -sf "${CODER_URL}/api/v2/buildinfo" &>/dev/null; then
		err "Coder API not reachable at ${CODER_URL}"
		err "Try: kubectl port-forward svc/coder -n ${NAMESPACE} ${CODER_PORT}:80"
		exit 1
	fi
	ok "Coder is accessible at: ${CODER_URL} (via port-forward, PID: ${PORT_FORWARD_PID})"

	# Install Coder CLI if not present
	if ! command -v coder &>/dev/null; then
		info "Installing Coder CLI..."
		curl -fsSL https://coder.com/install.sh | sh -s -- 2>/dev/null
		ok "Coder CLI installed"
	else
		ok "Coder CLI already installed"
	fi

	# Create first user (idempotent — will fail silently if user already exists)
	info "Creating first admin user..."
	if coder login "${CODER_URL}" \
		--first-user-email "${CODER_FIRST_USER_EMAIL}" \
		--first-user-username "${CODER_FIRST_USER_USERNAME}" \
		--first-user-password "${CODER_FIRST_USER_PASSWORD}" \
		--first-user-trial=false 2>/dev/null; then
		ok "First user created and logged in"
	else
		# Already created — just log in with token
		info "First user may already exist. Logging in..."
		CODER_SESSION_TOKEN=$(curl -sf -X POST "${CODER_URL}/api/v2/users/login" \
			-H "Content-Type: application/json" \
			-d "{\"email\":\"${CODER_FIRST_USER_EMAIL}\",\"password\":\"${CODER_FIRST_USER_PASSWORD}\"}" |
			python3 -c "import sys,json; print(json.load(sys.stdin)['session_token'])" 2>/dev/null || echo "")

		if [[ -n "${CODER_SESSION_TOKEN}" ]]; then
			export CODER_SESSION_TOKEN
			export CODER_URL
			ok "Logged in to existing Coder instance"
		else
			err "Could not log in to Coder. Check credentials."
			exit 1
		fi
	fi

	# ─── Phase 4: Push template & create workspace ────────────────────────────
	header "Phase 4: Template & Workspace"

	# Push template
	info "Pushing Coder template '${CODER_TEMPLATE_NAME}'..."
	if coder templates list 2>/dev/null | grep -q "${CODER_TEMPLATE_NAME}"; then
		info "Template exists — updating..."
		coder templates push "${CODER_TEMPLATE_NAME}" \
			--directory "${SCRIPT_DIR}/template" \
			--variable "namespace=${NAMESPACE}" \
			--yes
	else
		coder templates push "${CODER_TEMPLATE_NAME}" \
			--directory "${SCRIPT_DIR}/template" \
			--variable "namespace=${NAMESPACE}" \
			--yes
	fi
	ok "Template '${CODER_TEMPLATE_NAME}' pushed"

	# Create workspace
	info "Creating workspace '${CODER_WORKSPACE_NAME}'..."
	if coder list 2>/dev/null | grep -q "${CODER_WORKSPACE_NAME}"; then
		ok "Workspace already exists — skipping creation"
	else
		coder create "${CODER_WORKSPACE_NAME}" \
			--template "${CODER_TEMPLATE_NAME}" \
			--parameter "cpu=16" \
			--parameter "memory=96" \
			--parameter "memory_request=32" \
			--parameter "home_disk_size=50" \
			--yes
		ok "Workspace '${CODER_WORKSPACE_NAME}' created"
	fi

	# Wait for workspace agent to connect
	info "Waiting for workspace agent to connect..."
	for i in $(seq 1 90); do
		AGENT_STATUS=$(coder list --output json 2>/dev/null |
			python3 -c "
import sys, json
ws = json.load(sys.stdin)
for w in ws:
    if w['name'] == '${CODER_WORKSPACE_NAME}':
        agents = w.get('latest_build', {}).get('resources', [])
        for r in agents:
            for a in r.get('agents', []):
                print(a.get('status', 'unknown'))
                sys.exit(0)
print('unknown')
" 2>/dev/null || echo "unknown")

		if [[ "${AGENT_STATUS}" == "connected" ]]; then
			break
		fi
		echo -n "."
		sleep 10
	done
	echo ""

	if [[ "${AGENT_STATUS}" != "connected" ]]; then
		warn "Agent not yet connected (status: ${AGENT_STATUS}). It may still be starting up."
		warn "Check: coder list"
	else
		ok "Workspace agent is connected"
	fi

	# ─── Phase 5: Validation ──────────────────────────────────────────────────
	header "Phase 5: Validation"

	echo ""
	info "=== Workspace Pods ==="
	kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/name=coder-workspace" -o wide
	echo ""

	POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
		-l "app.kubernetes.io/name=coder-workspace" \
		-o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "unknown")
	POD_NODE=$(kubectl get pods -n "${NAMESPACE}" \
		-l "app.kubernetes.io/name=coder-workspace" \
		-o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo "unknown")

	info "=== Pod Labels ==="
	kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
		-o jsonpath='{.metadata.labels}' 2>/dev/null | python3 -m json.tool 2>/dev/null || true
	echo ""

	# ─── Summary ──────────────────────────────────────────────────────────────
	header "Setup Complete"

	cat <<EOF

  Components deployed:
    Coder control plane   : Helm release 'coder' in namespace '${NAMESPACE}'
    PostgreSQL            : Helm release 'postgresql' in namespace '${PG_NAMESPACE}'
    Template              : '${CODER_TEMPLATE_NAME}' (Kubernetes + CAST AI labels)
    Workspace             : '${CODER_WORKSPACE_NAME}' (16 CPU, 32Gi req / 96Gi limit, 50Gi disk)

  Coder URL  : ${CODER_URL}  (via port-forward, PID: ${PORT_FORWARD_PID})
  Internal   : http://coder.${NAMESPACE}.svc.cluster.local:80  (used by agents)
  Username   : ${CODER_FIRST_USER_USERNAME}
  Password   : ${CODER_FIRST_USER_PASSWORD}

  Pod        : ${POD_NAME}
  Node       : ${POD_NODE}

  ──────────────────────────────────────────────────────────────
  ACCESS
  ──────────────────────────────────────────────────────────────

  Coder is accessed via port-forward. If the port-forward dies,
  restart it with:

    kubectl port-forward svc/coder -n ${NAMESPACE} ${CODER_PORT}:80

  Dashboard: ${CODER_URL}

  ──────────────────────────────────────────────────────────────
  HOW TO TEST LIVE MIGRATION
  ──────────────────────────────────────────────────────────────

  1. Trigger the kernel build (workspace is currently idle):

     ./setup.sh build
       or
     coder ssh ${CODER_WORKSPACE_NAME} -- build-kernel

  2. Note the current node: ${POD_NODE}

  3. Once the build is running, trigger a CAST AI live migration
     via the CAST AI console or API.

  4. After migration, verify:

     a. Pod is on a new node:
        kubectl get pod -n ${NAMESPACE} ${POD_NAME} -o wide

     b. Build process survived:
        coder ssh ${CODER_WORKSPACE_NAME} -- pgrep -fa make

     c. Coder agent reconnected:
        coder list

     d. Build artifacts (after completion):
        coder ssh ${CODER_WORKSPACE_NAME} -- ls -la ~/linux/vmlinux

  To tear down everything:
     ./setup.sh teardown

EOF
}

# ─── Main ─────────────────────────────────────────────────────────────────────
case "${1:-}" in
teardown) teardown ;;
build) build ;;
"") deploy ;;
*)
	err "Unknown command: ${1}"
	err "Usage: ./setup.sh [build|teardown]"
	exit 1
	;;
esac
