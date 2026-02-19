#!/bin/bash
set -euo pipefail

# ============================================================================
# Setup Azure SQL Database for Risk Manager
#
# Automatically detects the AKS cluster's resource group and region from
# the current kubectl context, then either:
#   - Reuses an existing risk-manager-sql-* server and database
#   - Creates a new one if none exists
#
# In both cases, ensures the K8s secret is created/updated with the
# correct connection string.
#
# Prerequisites:
#   - az CLI (logged in)
#   - kubectl (context set to target AKS cluster)
#   - openssl
#
# Usage:
#   ./setup-azure-sql.sh              # auto-detect or create
#   ./setup-azure-sql.sh --cleanup    # tear down SQL server + K8s secret
# ============================================================================

NAMESPACE="risk-manager"
SECRET_NAME="azure-sql-secret"
DB_NAME="RiskManagerDb"
SQL_ADMIN_USER="riskmanageradmin"
SQL_SERVER_PREFIX="risk-manager-sql-"
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
	local pw
	pw="$(openssl rand -base64 32 | tr -d '/+=' | head -c 20)"
	echo "${pw}A1a!"
}

# ---- Detect AKS cluster info from kubectl context ----

detect_aks_info() {
	info "Detecting AKS cluster from kubectl context..."

	local context
	context="$(kubectl config current-context)"
	echo "  kubectl context: ${context}"

	local cluster_name="${context}"

	info "Querying Azure for AKS clusters..."

	local aks_info
	aks_info="$(az aks list --query "[?name=='${cluster_name}'] | [0]" -o json 2>/dev/null || echo "")"

	if [[ -z "$aks_info" || "$aks_info" == "null" ]]; then
		local stripped="${cluster_name%-admin}"
		aks_info="$(az aks list --query "[?name=='${stripped}'] | [0]" -o json 2>/dev/null || echo "")"
	fi

	if [[ -z "$aks_info" || "$aks_info" == "null" ]]; then
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

# ---- Find existing SQL server ----

find_existing_sql_server() {
	info "Checking for existing risk-manager-sql-* server in resource group ${AKS_RESOURCE_GROUP}..."

	EXISTING_SERVER="$(az sql server list \
		-g "${AKS_RESOURCE_GROUP}" \
		--query "[?starts_with(name, '${SQL_SERVER_PREFIX}')].name | [0]" \
		-o tsv 2>/dev/null || echo "")"

	if [[ -n "$EXISTING_SERVER" && "$EXISTING_SERVER" != "None" ]]; then
		return 0 # found
	else
		EXISTING_SERVER=""
		return 1 # not found
	fi
}

# ---- Ensure DB exists on the server ----

ensure_database() {
	local server_name="$1"

	info "Checking if database '${DB_NAME}' exists on ${server_name}..."

	local db_exists
	db_exists="$(az sql db list \
		--server "${server_name}" \
		-g "${AKS_RESOURCE_GROUP}" \
		--query "[?name=='${DB_NAME}'].name | [0]" \
		-o tsv 2>/dev/null || echo "")"

	if [[ -n "$db_exists" && "$db_exists" != "None" ]]; then
		ok "Database '${DB_NAME}' already exists."
	else
		info "Creating database '${DB_NAME}' (Basic, 5 DTU)..."
		az sql db create \
			--server "${server_name}" \
			--resource-group "${AKS_RESOURCE_GROUP}" \
			--name "${DB_NAME}" \
			--edition "Basic" \
			--capacity 5 \
			--max-size "2GB" \
			--output none
		ok "Database created."
	fi
}

# ---- Ensure firewall rule ----

ensure_firewall_rule() {
	local server_name="$1"

	info "Checking firewall rule (AllowAzureServices)..."

	local rule_exists
	rule_exists="$(az sql server firewall-rule list \
		--server "${server_name}" \
		-g "${AKS_RESOURCE_GROUP}" \
		--query "[?name=='AllowAzureServices'].name | [0]" \
		-o tsv 2>/dev/null || echo "")"

	if [[ -n "$rule_exists" && "$rule_exists" != "None" ]]; then
		ok "Firewall rule already exists."
	else
		info "Adding firewall rule (Allow Azure Services)..."
		az sql server firewall-rule create \
			--server "${server_name}" \
			--resource-group "${AKS_RESOURCE_GROUP}" \
			--name "AllowAzureServices" \
			--start-ip-address 0.0.0.0 \
			--end-ip-address 0.0.0.0 \
			--output none
		ok "Firewall rule added."
	fi
}

# ---- Create/update K8s secret ----

apply_k8s_secret() {
	local conn_string="$1"

	info "Creating/updating Kubernetes secret in namespace '${NAMESPACE}'..."

	kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

	kubectl create secret generic "${SECRET_NAME}" \
		--namespace "${NAMESPACE}" \
		--from-literal=AZURE_SQL_CONNECTION_STRING="${conn_string}" \
		--dry-run=client -o yaml | kubectl apply -f -

	ok "Kubernetes secret '${SECRET_NAME}' created/updated in namespace '${NAMESPACE}'."
}

# ---- Cleanup mode ----

cleanup() {
	info "Cleaning up Azure SQL resources..."

	detect_aks_info

	local servers
	servers="$(az sql server list -g "${AKS_RESOURCE_GROUP}" --query "[?starts_with(name, '${SQL_SERVER_PREFIX}')].name" -o tsv 2>/dev/null || echo "")"

	if [[ -z "$servers" ]]; then
		warn "No ${SQL_SERVER_PREFIX}* servers found in resource group ${AKS_RESOURCE_GROUP}"
	else
		for server in $servers; do
			info "Deleting SQL server: ${server}..."
			az sql server delete --name "$server" -g "${AKS_RESOURCE_GROUP}" --yes
			ok "Deleted ${server}"
		done
	fi

	info "Deleting K8s secret ${SECRET_NAME} in namespace ${NAMESPACE}..."
	kubectl delete secret "${SECRET_NAME}" -n "${NAMESPACE}" --ignore-not-found
	ok "Cleanup complete."
	exit 0
}

# ---- Print summary ----

print_summary() {
	local server_name="$1"
	local password="$2"
	local conn_string="$3"
	local reused="$4"

	echo ""
	echo "============================================"
	if [[ "$reused" == "true" ]]; then
		echo "  Azure SQL Setup Complete (reused existing)"
	else
		echo "  Azure SQL Setup Complete (newly created)"
	fi
	echo "============================================"
	echo ""
	echo "  Server:       ${server_name}.database.windows.net"
	echo "  Database:     ${DB_NAME}"
	echo "  Admin User:   ${SQL_ADMIN_USER}"
	if [[ "$reused" == "true" ]]; then
		echo "  Admin Pass:   (using password you provided)"
	else
		echo "  Admin Pass:   ${password}"
	fi
	echo "  Location:     ${AKS_LOCATION}"
	echo "  Resource Grp: ${AKS_RESOURCE_GROUP}"
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

# ---- Main ----

main() {
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
	echo ""

	# ---- Check for existing server ----
	if find_existing_sql_server; then
		SQL_SERVER_NAME="${EXISTING_SERVER}"
		ok "Found existing SQL server: ${SQL_SERVER_NAME}.database.windows.net"
		echo ""

		# Existing server - need the admin password to build connection string
		info "To reuse this server, we need the admin password."
		info "If you don't remember it, use --cleanup and re-run to create a new one."
		echo ""
		read -s -p "  Enter SQL admin password for ${SQL_SERVER_NAME}: " SQL_ADMIN_PASSWORD
		echo ""

		# Ensure the DB and firewall rule exist
		echo ""
		ensure_database "${SQL_SERVER_NAME}"
		ensure_firewall_rule "${SQL_SERVER_NAME}"

		# Build connection string and apply secret
		CONNECTION_STRING="Server=tcp:${SQL_SERVER_NAME}.database.windows.net,1433;Initial Catalog=${DB_NAME};Persist Security Info=False;User ID=${SQL_ADMIN_USER};Password=${SQL_ADMIN_PASSWORD};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"

		echo ""
		apply_k8s_secret "${CONNECTION_STRING}"
		print_summary "${SQL_SERVER_NAME}" "" "${CONNECTION_STRING}" "true"

	else
		# ---- No existing server - create new ----
		info "No existing ${SQL_SERVER_PREFIX}* server found. Will create a new one."
		echo ""

		local suffix
		suffix="$(openssl rand -hex 4)"
		SQL_SERVER_NAME="${SQL_SERVER_PREFIX}${suffix}"
		SQL_ADMIN_PASSWORD="$(generate_password)"

		info "Will create:"
		echo "  SQL Server:     ${SQL_SERVER_NAME}.database.windows.net"
		echo "  Database:       ${DB_NAME}"
		echo "  Admin User:     ${SQL_ADMIN_USER}"
		echo "  Location:       ${AKS_LOCATION}"
		echo "  Resource Group: ${AKS_RESOURCE_GROUP}"
		echo "  Edition:        Basic (5 DTU)"
		echo ""

		read -p "Continue? (y/N) " -n 1 -r
		echo
		if [[ ! $REPLY =~ ^[Yy]$ ]]; then
			echo "Aborted."
			exit 1
		fi

		# Create SQL Server
		echo ""
		info "[1/4] Creating Azure SQL Server..."
		az sql server create \
			--name "${SQL_SERVER_NAME}" \
			--resource-group "${AKS_RESOURCE_GROUP}" \
			--location "${AKS_LOCATION}" \
			--admin-user "${SQL_ADMIN_USER}" \
			--admin-password "${SQL_ADMIN_PASSWORD}" \
			--output none
		ok "SQL Server created: ${SQL_SERVER_NAME}.database.windows.net"

		# Firewall rule
		echo ""
		info "[2/4] Adding firewall rule (Allow Azure Services)..."
		ensure_firewall_rule "${SQL_SERVER_NAME}"

		# Create Database
		echo ""
		info "[3/4] Creating database..."
		ensure_database "${SQL_SERVER_NAME}"

		# Build connection string and apply secret
		CONNECTION_STRING="Server=tcp:${SQL_SERVER_NAME}.database.windows.net,1433;Initial Catalog=${DB_NAME};Persist Security Info=False;User ID=${SQL_ADMIN_USER};Password=${SQL_ADMIN_PASSWORD};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"

		echo ""
		info "[4/4] Creating Kubernetes secret..."
		apply_k8s_secret "${CONNECTION_STRING}"

		print_summary "${SQL_SERVER_NAME}" "${SQL_ADMIN_PASSWORD}" "${CONNECTION_STRING}" "false"
	fi
}

main "$@"
