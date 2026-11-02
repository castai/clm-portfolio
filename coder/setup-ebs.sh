#!/usr/bin/env bash
#
# setup-ebs.sh — Complete EBS setup for EKS: OIDC provider, IAM permissions,
#                 EBS CSI driver addon, and gp3 StorageClass.
#
# This is a one-shot script. Run it once and everything is configured:
#   1. Creates the IAM OIDC provider (required for IRSA)
#   2. Creates IAM policy + role for the EBS CSI driver
#   3. Installs/updates the EBS CSI driver addon with the IAM role
#   4. Creates a gp3 StorageClass (encrypted, WaitForFirstConsumer)
#
# Usage:
#   ./setup-ebs.sh [--cluster <cluster-name>] [--dry-run]
#   ./setup-ebs.sh --cleanup [--cluster <cluster-name>]
#
# If --cluster is omitted, the script auto-detects the EKS cluster name
# from the current kubectl context.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# ─── Configuration ────────────────────────────────────────────────────────────
IAM_POLICY_NAME="AmazonEBSCSIDriverPolicy"
IAM_ROLE_NAME="AmazonEKS_EBS_CSI_DriverRole"
NAMESPACE="kube-system"
SERVICE_ACCOUNT="ebs-csi-controller-sa"
DRY_RUN=false
CLEANUP=false

# ─── Argument parsing ────────────────────────────────────────────────────────
usage() {
	echo "Usage: $0 [options]"
	echo ""
	echo "Complete EBS setup: OIDC provider, IAM policy/role, EBS CSI driver"
	echo "addon (with IRSA), and gp3 StorageClass."
	echo ""
	echo "Options:"
	echo "  --cluster <name>    EKS cluster name (auto-detected from kubectl context if omitted)"
	echo "  --dry-run           Show what would be done without making changes"
	echo "  --cleanup           Remove IAM resources, gp3 StorageClass, and IRSA annotation"
	echo "  --help              Show this help message"
	echo ""
	echo "Examples:"
	echo "  $0                                  # auto-detect cluster from context"
	echo "  $0 --cluster my-cluster             # specify cluster explicitly"
	echo "  $0 --dry-run                        # preview changes"
	echo "  $0 --cleanup                        # tear down everything this script created"
	exit 1
}

CLUSTER_NAME=""

while [[ $# -gt 0 ]]; do
	case $1 in
	--cluster)
		CLUSTER_NAME="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN=true
		shift
		;;
	--cleanup)
		CLEANUP=true
		shift
		;;
	--help)
		usage
		;;
	*)
		err "Unknown option: $1"
		usage
		;;
	esac
done

# ─── Resolve cluster name from kubectl context if not provided ────────────────
resolve_cluster_name() {
	if [[ -n "${CLUSTER_NAME}" ]]; then
		return
	fi

	info "No --cluster specified, detecting from kubectl context..."

	local context
	context=$(kubectl config current-context 2>/dev/null || echo "")
	if [[ -z "${context}" ]]; then
		err "No kubectl context set and --cluster not specified."
		exit 1
	fi

	# EKS contexts from 'aws eks update-kubeconfig' look like:
	#   arn:aws:eks:<region>:<account>:cluster/<cluster-name>
	if [[ "${context}" == arn:aws:eks:* ]]; then
		CLUSTER_NAME="${context##*/}"
	# eksctl contexts look like: <user>@<cluster-name>.<region>.eksctl.io
	elif [[ "${context}" == *".eksctl.io" ]]; then
		CLUSTER_NAME=$(echo "${context}" | sed 's/.*@//' | sed 's/\..*//')
	else
		# Last resort: try the context name as the cluster name
		# Strip any prefix before @ or / (e.g. "user@cluster" -> "cluster")
		CLUSTER_NAME="${context##*@}"
		CLUSTER_NAME="${CLUSTER_NAME##*/}"
	fi

	if [[ -z "${CLUSTER_NAME}" ]]; then
		err "Could not determine cluster name from context: ${context}"
		err "Specify it with: $0 --cluster <cluster-name>"
		exit 1
	fi

	# Validate the detected name against the EKS API
	if ! aws eks describe-cluster --name "${CLUSTER_NAME}" &>/dev/null 2>&1; then
		err "Detected cluster name '${CLUSTER_NAME}' from context '${context}'"
		err "but 'aws eks describe-cluster --name ${CLUSTER_NAME}' failed."
		err "Specify the correct name with: $0 --cluster <cluster-name>"
		exit 1
	fi

	ok "Detected EKS cluster: ${CLUSTER_NAME} (from context: ${context})"
}

# ─── Teardown / Cleanup ──────────────────────────────────────────────────────
cleanup() {
	header "Cleaning up EBS setup"

	AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
	POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${IAM_POLICY_NAME}"

	info "Detaching policy from role..."
	aws iam detach-role-policy \
		--role-name "${IAM_ROLE_NAME}" \
		--policy-arn "${POLICY_ARN}" 2>/dev/null || true

	info "Deleting IAM policy..."
	aws iam delete-policy \
		--policy-arn "${POLICY_ARN}" 2>/dev/null || true

	info "Deleting IAM role..."
	aws iam delete-role \
		--role-name "${IAM_ROLE_NAME}" 2>/dev/null || true

	info "Removing IRSA annotation from service account..."
	kubectl annotate serviceaccount "${SERVICE_ACCOUNT}" \
		-n "${NAMESPACE}" \
		eks.amazonaws.com/role-arn- 2>/dev/null || true

	info "Deleting gp3 StorageClass..."
	kubectl delete storageclass gp3 --ignore-not-found 2>/dev/null || true

	ok "Cleanup complete."
	info "The EBS CSI driver addon and OIDC provider were NOT removed."
	info "To remove the addon: aws eks delete-addon --addon-name aws-ebs-csi-driver --cluster-name <cluster>"
	info "To restart the driver: kubectl rollout restart deployment ebs-csi-controller -n ${NAMESPACE}"
}

if [[ "${CLEANUP}" == "true" ]]; then
	cleanup
	exit 0
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────────
header "Pre-flight checks"

for tool in kubectl aws; do
	if ! command -v "${tool}" &>/dev/null; then
		err "${tool} is required but not found in PATH"
		exit 1
	fi
	ok "${tool} found"
done

if ! kubectl cluster-info &>/dev/null; then
	err "Cannot connect to Kubernetes cluster"
	exit 1
fi
ok "Kubernetes cluster accessible"

resolve_cluster_name

# ─── Gather configuration ────────────────────────────────────────────────────
header "Gathering configuration"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ok "AWS Account ID: ${AWS_ACCOUNT_ID}"
ok "EKS Cluster: ${CLUSTER_NAME}"

OIDC_ISSUER=$(aws eks describe-cluster --name "${CLUSTER_NAME}" \
	--query "cluster.identity.oidc.issuer" --output text)
# Strip https:// prefix for use in IAM trust policies
OIDC_PROVIDER="${OIDC_ISSUER#https://}"
# Extract just the OIDC ID (last path segment) for checking existing providers
OIDC_ID="${OIDC_ISSUER##*/}"
ok "OIDC Issuer: ${OIDC_ISSUER}"

# ─── Phase 1: OIDC Provider ─────────────────────────────────────────────────
header "Phase 1: IAM OIDC Identity Provider"

# IRSA requires an IAM OIDC provider to exist for the cluster.
# Without this, the EBS CSI driver cannot assume its IAM role.
# Reference: https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html

EXISTING_OIDC=$(aws iam list-open-id-connect-providers \
	--query "OpenIDConnectProviderList[?ends_with(Arn, '/${OIDC_ID}')].Arn" \
	--output text 2>/dev/null || echo "")

if [[ -n "${EXISTING_OIDC}" ]]; then
	ok "IAM OIDC provider already exists: ${EXISTING_OIDC}"
else
	info "Creating IAM OIDC identity provider for cluster '${CLUSTER_NAME}'..."

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would create OIDC provider for: ${OIDC_ISSUER}"
	else
		# Get the OIDC thumbprint (required by IAM)
		OIDC_HOST="${OIDC_PROVIDER%%/*}"
		THUMBPRINT=$(echo | openssl s_client -servername "${OIDC_HOST}" \
			-connect "${OIDC_HOST}:443" 2>/dev/null |
			openssl x509 -fingerprint -sha1 -noout 2>/dev/null |
			sed 's/.*=//' | tr -d ':' | tr '[:upper:]' '[:lower:]')

		if [[ -z "${THUMBPRINT}" ]]; then
			# Fallback: AWS accepts a dummy thumbprint for EKS OIDC since
			# they verify the certificate chain themselves
			warn "Could not compute OIDC thumbprint, using AWS default"
			THUMBPRINT="9e99a48a9960b14926bb7f3b02e22da2b0ab7280"
		fi

		aws iam create-open-id-connect-provider \
			--url "${OIDC_ISSUER}" \
			--client-id-list sts.amazonaws.com \
			--thumbprint-list "${THUMBPRINT}" >/dev/null

		ok "IAM OIDC provider created"
	fi
fi

# ─── Phase 2: IAM Policy ────────────────────────────────────────────────────
header "Phase 2: IAM Policy"

EXISTING_POLICY=$(aws iam list-policies \
	--query "Policies[?PolicyName=='${IAM_POLICY_NAME}'].Arn" --output text)

if [[ -n "${EXISTING_POLICY}" ]]; then
	ok "Policy ${IAM_POLICY_NAME} already exists: ${EXISTING_POLICY}"
	POLICY_ARN="${EXISTING_POLICY}"
else
	info "Creating IAM policy ${IAM_POLICY_NAME}..."

	POLICY_DOCUMENT='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateSnapshot",
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:ModifyVolume",
                "ec2:DescribeAvailabilityZones",
                "ec2:DescribeInstances",
                "ec2:DescribeSnapshots",
                "ec2:DescribeTags",
                "ec2:DescribeVolumes",
                "ec2:DescribeVolumesModifications",
                "ec2:DescribeVolumeStatus",
                "ec2:DeleteSnapshot"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateTags"
            ],
            "Resource": [
                "arn:aws:ec2:*:*:volume/*",
                "arn:aws:ec2:*:*:snapshot/*"
            ],
            "Condition": {
                "StringEquals": {
                    "ec2:CreateAction": [
                        "CreateVolume",
                        "CreateSnapshot"
                    ]
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DeleteTags"
            ],
            "Resource": [
                "arn:aws:ec2:*:*:volume/*",
                "arn:aws:ec2:*:*:snapshot/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateVolume"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "aws:RequestTag/ebs.csi.aws.com/cluster": "true"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateVolume"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "aws:RequestTag/CSIVolumeName": "*"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DeleteVolume"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "ec2:ResourceTag/ebs.csi.aws.com/cluster": "true"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DeleteVolume"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "ec2:ResourceTag/CSIVolumeName": "*"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DeleteVolume"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "ec2:ResourceTag/kubernetes.io/created-for/pvc/name": "*"
                }
            }
        }
    ]
}'

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would create IAM policy"
		POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${IAM_POLICY_NAME} (dry-run)"
	else
		POLICY_ARN=$(aws iam create-policy \
			--policy-name "${IAM_POLICY_NAME}" \
			--policy-document "${POLICY_DOCUMENT}" \
			--query "Policy.Arn" --output text)

		ok "IAM policy created: ${POLICY_ARN}"
	fi
fi

# ─── Phase 3: IAM Role ──────────────────────────────────────────────────────
header "Phase 3: IAM Role"

ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}"

if ! aws iam get-role --role-name "${IAM_ROLE_NAME}" &>/dev/null 2>&1; then
	info "Creating IAM role ${IAM_ROLE_NAME}..."

	ASSUME_ROLE_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::'"${AWS_ACCOUNT_ID}"':oidc-provider/'"${OIDC_PROVIDER}"'"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "'"${OIDC_PROVIDER}"':sub": "system:serviceaccount:'"${NAMESPACE}"':'"${SERVICE_ACCOUNT}"'",
                    "'"${OIDC_PROVIDER}"':aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}'

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would create IAM role with OIDC trust policy"
	else
		aws iam create-role \
			--role-name "${IAM_ROLE_NAME}" \
			--assume-role-policy-document "${ASSUME_ROLE_POLICY}" >/dev/null

		ok "IAM role created: ${IAM_ROLE_NAME}"
	fi
else
	ok "IAM role ${IAM_ROLE_NAME} already exists"

	# Verify trust policy points to correct OIDC provider
	CURRENT_OIDC=$(aws iam get-role --role-name "${IAM_ROLE_NAME}" \
		--query "Role.AssumeRolePolicyDocument.Statement[0].Principal.Federated" \
		--output text 2>/dev/null || echo "")
	EXPECTED_OIDC="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"

	if [[ "${CURRENT_OIDC}" != "${EXPECTED_OIDC}" ]]; then
		warn "Trust policy OIDC mismatch — updating..."
		warn "  Current:  ${CURRENT_OIDC}"
		warn "  Expected: ${EXPECTED_OIDC}"

		ASSUME_ROLE_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::'"${AWS_ACCOUNT_ID}"':oidc-provider/'"${OIDC_PROVIDER}"'"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "'"${OIDC_PROVIDER}"':sub": "system:serviceaccount:'"${NAMESPACE}"':'"${SERVICE_ACCOUNT}"'",
                    "'"${OIDC_PROVIDER}"':aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}'

		if [[ "${DRY_RUN}" == "true" ]]; then
			info "[DRY-RUN] Would update trust policy"
		else
			aws iam update-assume-role-policy \
				--role-name "${IAM_ROLE_NAME}" \
				--policy-document "${ASSUME_ROLE_POLICY}" >/dev/null
			ok "Trust policy updated"
		fi
	else
		ok "Trust policy is correct"
	fi
fi

# ─── Phase 4: Attach Policy to Role ─────────────────────────────────────────
header "Phase 4: Attach Policy to Role"

if ! aws iam list-attached-role-policies --role-name "${IAM_ROLE_NAME}" \
	--query "AttachedPolicies[?PolicyName=='${IAM_POLICY_NAME}']" --output text 2>/dev/null | grep -q "${IAM_POLICY_NAME}"; then
	info "Attaching policy to role..."

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would attach ${POLICY_ARN} to ${IAM_ROLE_NAME}"
	else
		aws iam attach-role-policy \
			--policy-arn "${POLICY_ARN}" \
			--role-name "${IAM_ROLE_NAME}" >/dev/null
		ok "Policy attached to role"
	fi
else
	ok "Policy already attached to role"
fi

# ─── Phase 5: EBS CSI Driver Addon ──────────────────────────────────────────
header "Phase 5: EBS CSI Driver Addon"

# Install or update the addon with --service-account-role-arn so EKS
# configures IRSA automatically. This is the AWS-recommended approach.
# Reference: https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html

# Helper: wait for addon to leave transitional states (CREATING, UPDATING, DELETING)
wait_for_addon_ready() {
	local max_wait="${1:-120}"
	local status="UNKNOWN"

	for i in $(seq 1 "${max_wait}"); do
		status=$(aws eks describe-addon \
			--addon-name aws-ebs-csi-driver \
			--cluster-name "${CLUSTER_NAME}" \
			--query "addon.status" --output text 2>/dev/null || echo "NOT_FOUND")

		case "${status}" in
		ACTIVE | DEGRADED | CREATE_FAILED | UPDATE_FAILED)
			# Settled states — safe to proceed
			break
			;;
		NOT_FOUND)
			# Addon doesn't exist
			break
			;;
		*)
			# CREATING, UPDATING, DELETING — wait
			echo -n "."
			sleep 5
			;;
		esac
	done
	echo ""
	echo "${status}"
}

ADDON_EXISTS=false
ADDON_STATUS="NOT_FOUND"

if aws eks describe-addon --addon-name aws-ebs-csi-driver \
	--cluster-name "${CLUSTER_NAME}" &>/dev/null 2>&1; then
	ADDON_EXISTS=true

	ADDON_STATUS=$(aws eks describe-addon --addon-name aws-ebs-csi-driver \
		--cluster-name "${CLUSTER_NAME}" \
		--query "addon.status" --output text 2>/dev/null || echo "UNKNOWN")

	# If addon is in a transitional state, wait for it to settle
	if [[ "${ADDON_STATUS}" == "CREATING" || "${ADDON_STATUS}" == "UPDATING" || "${ADDON_STATUS}" == "DELETING" ]]; then
		info "Addon is currently ${ADDON_STATUS} — waiting for it to settle..."
		ADDON_STATUS=$(wait_for_addon_ready 120)
		info "Addon status: ${ADDON_STATUS}"
	fi
fi

if [[ "${DRY_RUN}" == "true" ]]; then
	if [[ "${ADDON_EXISTS}" == "true" ]]; then
		info "[DRY-RUN] Would update addon with --service-account-role-arn ${ROLE_ARN}"
	else
		info "[DRY-RUN] Would install addon with --service-account-role-arn ${ROLE_ARN}"
	fi
elif [[ "${ADDON_EXISTS}" == "true" ]]; then
	# Check if the addon already has the correct role ARN
	CURRENT_ROLE=$(aws eks describe-addon --addon-name aws-ebs-csi-driver \
		--cluster-name "${CLUSTER_NAME}" \
		--query "addon.serviceAccountRoleArn" --output text 2>/dev/null || echo "None")

	if [[ "${CURRENT_ROLE}" == "${ROLE_ARN}" ]]; then
		ok "EBS CSI driver addon already configured with correct IAM role"
	else
		info "EBS CSI driver addon exists but IAM role is '${CURRENT_ROLE}'"
		info "Updating addon with role ARN: ${ROLE_ARN}"

		aws eks update-addon \
			--addon-name aws-ebs-csi-driver \
			--cluster-name "${CLUSTER_NAME}" \
			--service-account-role-arn "${ROLE_ARN}" \
			--resolve-conflicts OVERWRITE || {
			warn "Addon update via API failed — will fall back to SA annotation"
		}
		ok "Addon update initiated"
	fi
else
	info "Installing EBS CSI driver addon with IAM role..."

	aws eks create-addon \
		--addon-name aws-ebs-csi-driver \
		--cluster-name "${CLUSTER_NAME}" \
		--service-account-role-arn "${ROLE_ARN}" || {
		err "Failed to install EBS CSI driver addon."
		exit 1
	}
	ok "EBS CSI driver addon installation initiated"
fi

# Wait for addon to become active
if [[ "${DRY_RUN}" != "true" ]]; then
	info "Waiting for EBS CSI driver addon to become active..."
	FINAL_STATUS=$(wait_for_addon_ready 120)

	if [[ "${FINAL_STATUS}" == "ACTIVE" ]]; then
		ok "EBS CSI driver addon is ACTIVE"
	else
		warn "EBS CSI addon status: ${FINAL_STATUS} (may still be settling)"
	fi

	# Belt-and-suspenders: also annotate the SA directly and restart if needed.
	# The addon --service-account-role-arn should handle this, but if the addon
	# was in a bad state or the update was skipped, this ensures IRSA works.
	if kubectl get serviceaccount "${SERVICE_ACCOUNT}" -n "${NAMESPACE}" &>/dev/null; then
		CURRENT_ANNOTATION=$(kubectl get serviceaccount "${SERVICE_ACCOUNT}" \
			-n "${NAMESPACE}" \
			-o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")

		if [[ "${CURRENT_ANNOTATION}" != "${ROLE_ARN}" ]]; then
			info "Ensuring SA annotation matches role ARN..."
			kubectl annotate serviceaccount "${SERVICE_ACCOUNT}" \
				-n "${NAMESPACE}" \
				eks.amazonaws.com/role-arn="${ROLE_ARN}" \
				--overwrite >/dev/null
			ok "Service account annotation updated"

			info "Restarting EBS CSI controller to pick up new credentials..."
			kubectl rollout restart deployment ebs-csi-controller -n "${NAMESPACE}" 2>/dev/null || true
			kubectl rollout restart daemonset ebs-csi-node -n "${NAMESPACE}" 2>/dev/null || true

			info "Waiting for EBS CSI controller rollout..."
			kubectl rollout status deployment/ebs-csi-controller -n "${NAMESPACE}" --timeout=120s 2>/dev/null || true
			ok "EBS CSI controller restarted"
		else
			ok "Service account annotation already correct"
		fi
	fi
fi

# ─── Phase 6: gp3 StorageClass ──────────────────────────────────────────────
header "Phase 6: gp3 StorageClass"

# gp3 provides 3,000 baseline IOPS and 125 MiB/s throughput (included free),
# 20% cheaper per GiB than gp2, and supports volume expansion.
# Reference: https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html#gp3-ebs-volume-type
# StorageClass params: https://github.com/kubernetes-sigs/aws-ebs-csi-driver/blob/master/docs/parameters.md

GP3_STORAGECLASS='apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
allowVolumeExpansion: true'

if kubectl get storageclass gp3 &>/dev/null; then
	ok "StorageClass 'gp3' already exists"
else
	info "Creating gp3 StorageClass..."

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would create StorageClass:"
		echo "${GP3_STORAGECLASS}"
	else
		echo "${GP3_STORAGECLASS}" | kubectl apply -f -
		ok "StorageClass 'gp3' created"
	fi
fi

info "Current StorageClasses:"
kubectl get storageclass -o custom-columns='NAME:.metadata.name,PROVISIONER:.provisioner,RECLAIM:.reclaimPolicy,BINDING:.volumeBindingMode' 2>/dev/null || true

# ─── Summary ──────────────────────────────────────────────────────────────────
header "Setup Complete"

if [[ "${DRY_RUN}" == "true" ]]; then
	info "This was a dry run. No changes were made."
	info "Run without --dry-run to apply the changes."
else
	cat <<EOF

  Cluster:         ${CLUSTER_NAME}
  IAM Role ARN:    ${ROLE_ARN}
  Policy ARN:      ${POLICY_ARN}
  StorageClass:    gp3 (ebs.csi.aws.com, encrypted, WaitForFirstConsumer)

  ──────────────────────────────────────────────────────────────
  WHAT WAS DONE
  ──────────────────────────────────────────────────────────────

  1. IAM OIDC identity provider created (required for IRSA)
  2. IAM policy created: ${IAM_POLICY_NAME}
  3. IAM role created: ${IAM_ROLE_NAME} (with OIDC trust policy)
  4. Policy attached to role
  5. EBS CSI driver addon installed/updated with role ARN
  6. gp3 StorageClass created (encrypted, WaitForFirstConsumer)

  ──────────────────────────────────────────────────────────────
  STORAGE CLASSES
  ──────────────────────────────────────────────────────────────

  gp2  — Legacy. 3 IOPS/GiB (burst to 3,000). Pre-installed on EKS.
  gp3  — 3,000 baseline IOPS + 125 MiB/s throughput (free). 20% cheaper.
         Used by the AOSP build workspace (500Gi).

  ──────────────────────────────────────────────────────────────
  VERIFY
  ──────────────────────────────────────────────────────────────

  1. EBS CSI driver is healthy:
     kubectl get pods -n ${NAMESPACE} -l app=ebs-csi-controller

  2. Check gp3 StorageClass:
     kubectl get storageclass gp3

  3. Test PVC creation (optional):
     kubectl apply -f - <<EOFPVC
     apiVersion: v1
     kind: PersistentVolumeClaim
     metadata:
       name: test-gp3
       namespace: default
     spec:
       accessModes: [ReadWriteOnce]
       storageClassName: gp3
       resources:
         requests:
           storage: 1Gi
     EOFPVC
     # PVC stays Pending (WaitForFirstConsumer) until a pod uses it.

  ──────────────────────────────────────────────────────────────
  NEXT: Deploy Coder
  ──────────────────────────────────────────────────────────────

  Run ./setup.sh to deploy Coder with both workspaces:
    - kernel-build  (16 CPU, 96Gi, 50Gi gp2)
    - aosp-build    (32 CPU, 128Gi, 500Gi gp3)

EOF

fi

info "To clean up later, run: $0 --cleanup"
