#!/bin/bash
set -euo pipefail

# ============================================================================
# Setup Azure SQL Database for Risk Manager
#
# Automatically detects the AKS cluster's resource group and region from
# the current kubectl context, then creates:
#   - Azure SQL Server (with auto-generated secure password)
#   - Azure SQL Database (Basic tier, 5 DTU)
#   - Firewall rule allowing Azure services
#   - Kubernetes secret with the connection string
#
# Prerequisites:
#   - az CLI (logged in)
#   - kubectl (context set to target AKS cluster)
#   - openssl
#
# Usage:
#   ./setup-azure-sql.sh              # auto-detect everything
#   ./setup-azure-sql.sh --cleanup    # tear down SQL server + K8s secret
# ============================================================================

NAMESPACE="risk-manager"
SECRET_NAME="azure-sql-secret"
DB_NAME="RiskManagerDb"
SQL_ADMIN_USER="riskmanageradmin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")/k8s"

# ---- Helpers ----

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() {
	echo -e "\033[1;31m[ERROR]\033[0m $*"
	exit 1
}

generate_password() {
	# 24 chars: mix of upper, lower, digits, and special chars
	# Ensures Azure SQL password complexity requirements are met
	local pw
	pw="$(openssl rand -base64 32 | tr -d '/+=' | head -c 20)"
	# Append guaranteed complexity: uppercase, lowercase, digit, special
	echo "${pw}A1a!"
}

# ---- Detect AKS cluster info from kubectl context ----

detect_aks_info() {
	info "Detecting AKS cluster from kubectl context..."

	local context
	context="$(kubectl config current-context)"
	echo "  kubectl context: ${context}"

	# Try to extract cluster name from context
	# AKS contexts are typically named like: my-cluster or my-cluster-admin
	local cluster_name="${context}"

	# Get all AKS clusters and find the matching one
	info "Querying Azure for AKS clusters..."

	local aks_info
	aks_info="$(az aks list --query "[?name=='${cluster_name}'] | [0]" -o json 2>/dev/null || echo "")"

	if [[ -z "$aks_info" || "$aks_info" == "null" ]]; then
		# Context might not match cluster name directly, try listing all and matching
		# Also try stripping -admin suffix
		local stripped="${cluster_name%-admin}"
		aks_info="$(az aks list --query "[?name=='${stripped}'] | [0]" -o json 2>/dev/null || echo "")"
	fi

	if [[ -z "$aks_info" || "$aks_info" == "null" ]]; then
		# Last resort: get the server URL from kubeconfig and match against AKS clusters
		info "Context name didn't match directly. Searching all AKS clusters..."
		local server_url
		server_url="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null || echo "")"

		if [[ -n "$server_url" ]]; then
			local fqdn
			fqdn="$(echo "$server_url" | sed 's|https://||' | sed 's|:.*||')"
			aks_info="$(az aks list --query "[?fqdn=='${fqdn}'] | [0]" -o json 2>/dev/null || echo "")"
		fi
	fi

	if [[ -z "$aks_info" || "$aks_info" == "null" ]]; then
		error "Could not find AKS cluster matching context '${context}'.
    
  Make sure:
    1. You are logged into az CLI (az login)
    2. The correct subscription is set (az account set -s <sub>)
    3. kubectl context points to an AKS cluster"
	fi

	AKS_CLUSTER_NAME="$(echo "$aks_info" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")"
	AKS_RESOURCE_GROUP="$(echo "$aks_info" | python3 -c "import sys,json; print(json.load(sys.stdin)['resourceGroup'])")"
	AKS_LOCATION="$(echo "$aks_info" | python3 -c "import sys,json; print(json.load(sys.stdin)['location'])")"

	ok "AKS Cluster:      ${AKS_CLUSTER_NAME}"
	ok "Resource Group:    ${AKS_RESOURCE_GROUP}"
	ok "Location:          ${AKS_LOCATION}"
}

# ---- Cleanup mode ----

cleanup() {
	info "Cleaning up Azure SQL resources..."

	detect_aks_info

	# Find SQL servers in the resource group with our naming pattern
	local servers
	servers="$(az sql server list -g "${AKS_RESOURCE_GROUP}" --query "[?starts_with(name, 'risk-manager-sql-')].name" -o tsv 2>/dev/null || echo "")"

	if [[ -z "$servers" ]]; then
		warn "No risk-manager-sql-* servers found in resource group ${AKS_RESOURCE_GROUP}"
	else
		for server in $servers; do
			info "Deleting SQL server: ${server}..."
			az sql server delete --name "$server" -g "${AKS_RESOURCE_GROUP}" --yes
			ok "Deleted ${server}"
		done
	fi

	# Delete K8s secret
	info "Deleting K8s secret ${SECRET_NAME} in namespace ${NAMESPACE}..."
	kubectl delete secret "${SECRET_NAME}" -n "${NAMESPACE}" --ignore-not-found
	ok "Cleanup complete."
	exit 0
}

# ---- Main ----

main() {
	# Check for cleanup flag
	if [[ "${1:-}" == "--cleanup" ]]; then
		cleanup
	fi

	echo "============================================"
	echo "  Azure SQL Setup for Risk Manager"
	echo "============================================"
	echo ""

	# Check prerequisites
	command -v az >/dev/null 2>&1 || error "az CLI not found. Install: https://aka.ms/installazurecli"
	command -v kubectl >/dev/null 2>&1 || error "kubectl not found."
	command -v openssl >/dev/null 2>&1 || error "openssl not found."

	# Detect AKS info
	echo ""
	detect_aks_info

	# Generate unique server name and password
	local suffix
	suffix="$(openssl rand -hex 4)"
	SQL_SERVER_NAME="risk-manager-sql-${suffix}"
	SQL_ADMIN_PASSWORD="$(generate_password)"

	echo ""
	info "Will create:"
	echo "  SQL Server:   ${SQL_SERVER_NAME}.database.windows.net"
	echo "  Database:     ${DB_NAME}"
	echo "  Admin User:   ${SQL_ADMIN_USER}"
	echo "  Location:     ${AKS_LOCATION}"
	echo "  Resource Group: ${AKS_RESOURCE_GROUP}"
	echo "  Edition:      Basic (5 DTU)"
	echo ""

	read -p "Continue? (y/N) " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		echo "Aborted."
		exit 1
	fi

	# ---- Step 1: Create SQL Server ----
	echo ""
	info "[1/5] Creating Azure SQL Server..."
	az sql server create \
		--name "${SQL_SERVER_NAME}" \
		--resource-group "${AKS_RESOURCE_GROUP}" \
		--location "${AKS_LOCATION}" \
		--admin-user "${SQL_ADMIN_USER}" \
		--admin-password "${SQL_ADMIN_PASSWORD}" \
		--output none

	ok "SQL Server created: ${SQL_SERVER_NAME}.database.windows.net"

	# ---- Step 2: Firewall rule - Allow Azure Services ----
	echo ""
	info "[2/5] Adding firewall rule (Allow Azure Services)..."
	az sql server firewall-rule create \
		--server "${SQL_SERVER_NAME}" \
		--resource-group "${AKS_RESOURCE_GROUP}" \
		--name "AllowAzureServices" \
		--start-ip-address 0.0.0.0 \
		--end-ip-address 0.0.0.0 \
		--output none

	ok "Firewall rule added."

	# ---- Step 3: Create Database ----
	echo ""
	info "[3/5] Creating database '${DB_NAME}' (Basic, 5 DTU)..."
	az sql db create \
		--server "${SQL_SERVER_NAME}" \
		--resource-group "${AKS_RESOURCE_GROUP}" \
		--name "${DB_NAME}" \
		--edition "Basic" \
		--capacity 5 \
		--max-size "2GB" \
		--output none

	ok "Database created."

	# ---- Step 4: Build connection string ----
	echo ""
	info "[4/5] Building connection string..."

	CONNECTION_STRING="Server=tcp:${SQL_SERVER_NAME}.database.windows.net,1433;Initial Catalog=${DB_NAME};Persist Security Info=False;User ID=${SQL_ADMIN_USER};Password=${SQL_ADMIN_PASSWORD};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"

	ok "Connection string ready."

	# ---- Step 5: Create/Update K8s secret ----
	echo ""
	info "[5/5] Creating Kubernetes secret in namespace '${NAMESPACE}'..."

	# Ensure namespace exists
	kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

	# Create or replace the secret
	kubectl create secret generic "${SECRET_NAME}" \
		--namespace "${NAMESPACE}" \
		--from-literal=AZURE_SQL_CONNECTION_STRING="${CONNECTION_STRING}" \
		--dry-run=client -o yaml | kubectl apply -f -

	ok "Kubernetes secret '${SECRET_NAME}' created/updated in namespace '${NAMESPACE}'."

	# ---- Summary ----
	echo ""
	echo "============================================"
	echo "  Azure SQL Setup Complete!"
	echo "============================================"
	echo ""
	echo "  Server:       ${SQL_SERVER_NAME}.database.windows.net"
	echo "  Database:     ${DB_NAME}"
	echo "  Admin User:   ${SQL_ADMIN_USER}"
	echo "  Admin Pass:   ${SQL_ADMIN_PASSWORD}"
	echo "  Location:     ${AKS_LOCATION}"
	echo "  Resource Grp: ${AKS_RESOURCE_GROUP}"
	echo ""
	echo "  Connection String:"
	echo "  ${CONNECTION_STRING}"
	echo ""
	echo "  K8s Secret:   ${SECRET_NAME} (namespace: ${NAMESPACE})"
	echo ""
	echo "  Next steps:"
	echo "    1. Deploy the app:  ./scripts/deploy.sh"
	echo "    2. Or restart backend if already deployed:"
	echo "       kubectl rollout restart deployment/risk-manager-backend -n ${NAMESPACE}"
	echo ""
	echo "  To tear down:"
	echo "    ./scripts/setup-azure-sql.sh --cleanup"
	echo "============================================"
}

main "$@"
