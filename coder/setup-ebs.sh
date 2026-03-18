#!/usr/bin/env bash
#
# setup-ebs.sh — Fix IAM permissions for AWS EBS CSI Driver on EKS
#
# The EBS CSI driver needs IAM permissions to manage EBS volumes.
# This script creates the necessary IAM policy and role, then configures
# the Kubernetes service account to use it via IRSA.
#
# Usage:
#   ./setup-ebs.sh --cluster <cluster-name> [--dry-run]
#   ./setup-ebs.sh --cleanup --cluster <cluster-name>
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
	echo "Options:"
	echo "  --cluster <name>    EKS cluster name (optional, uses kubectl context if not provided)"
	echo "  --dry-run           Show what would be done without making changes"
	echo "  --cleanup           Remove the IAM policy and role created by this script"
	echo "  --help              Show this help message"
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

if [[ -z "${CLUSTER_NAME}" ]]; then
	err "Cluster name is required (could not determine from kubectl context)"
	err "Specify it with: $0 --cluster <cluster-name>"
	usage
fi

# ─── Teardown / Cleanup ──────────────────────────────────────────────────────
cleanup() {
	header "Cleaning up EBS CSI driver IAM configuration"

	AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
	POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${IAM_POLICY_NAME}"

	info "Detaching policy from role..."
	aws iam detach-role-policy \
		--role-name "${IAM_ROLE_NAME}" \
		--policy-arn "${POLICY_ARN}" 2>/dev/null || true

	info "Deleting policy..."
	aws iam delete-policy \
		--policy-arn "${POLICY_ARN}" 2>/dev/null || true

	info "Deleting role..."
	aws iam delete-role \
		--role-name "${IAM_ROLE_NAME}" 2>/dev/null || true

	info "Removing annotation from Kubernetes service account..."
	kubectl annotate serviceaccount "${SERVICE_ACCOUNT}" \
		-n "${NAMESPACE}" \
		eks.amazonaws.com/role-arn- 2>/dev/null || true

	ok "Cleanup complete."
	info "You may also want to:"
	info "  1. Restart the EBS CSI driver: kubectl rollout restart deployment ebs-csi-controller -n ${NAMESPACE}"
	info "  2. Restart the EBS CSI node: kubectl rollout restart daemonset ebs-csi-node -n ${NAMESPACE}"
}

if [[ "${CLEANUP}" == "true" ]]; then
	cleanup
	exit 0
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────────
header "Pre-flight checks"

if ! command -v kubectl &>/dev/null; then
	err "kubectl is required but not found in PATH"
	exit 1
fi
ok "kubectl found"

if ! command -v aws &>/dev/null; then
	err "aws CLI is required but not found in PATH"
	exit 1
fi
ok "aws CLI found"

if ! kubectl cluster-info &>/dev/null; then
	err "Cannot connect to Kubernetes cluster"
	exit 1
fi
ok "Kubernetes cluster accessible"

if ! kubectl get serviceaccount "${SERVICE_ACCOUNT}" -n "${NAMESPACE}" &>/dev/null; then
	err "Service account ${SERVICE_ACCOUNT} not found in namespace ${NAMESPACE}"
	err "Is the EBS CSI driver installed? Run:"
	err "  aws eks create-addon --addon-name aws-ebs-csi-driver --cluster-name ${CLUSTER_NAME}"
	exit 1
fi
ok "EBS CSI driver service account exists"

# ─── Get configuration ────────────────────────────────────────────────────────
header "Gathering configuration"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ok "AWS Account ID: ${AWS_ACCOUNT_ID}"

# Get cluster from kubectl context if not provided
if [[ -z "${CLUSTER_NAME}" ]]; then
	CLUSTER_NAME=$(kubectl config current-context | sed 's/.*\///' | sed 's/\.eksctl\.io//' | sed 's/\.eks\..*//')
	warn "Cluster name not specified, using from kubectl context: ${CLUSTER_NAME}"
fi

OIDC_PROVIDER=$(aws eks describe-cluster --name "${CLUSTER_NAME}" \
	--query "cluster.identity.oidc.issuer" --output text | sed 's|https://||')
ok "OIDC Provider: ${OIDC_PROVIDER}"

# ─── Create IAM Policy ───────────────────────────────────────────────────────
header "Phase 1: Create IAM Policy"

# Check if policy already exists
EXISTING_POLICY=$(aws iam list-policies --query "Policies[?PolicyName=='${IAM_POLICY_NAME}'].Arn" --output text)
if [[ -n "${EXISTING_POLICY}" ]]; then
	warn "Policy ${IAM_POLICY_NAME} already exists"
	warn "Reusing existing policy: ${EXISTING_POLICY}"
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
                "ec2:DescribeVolumes",
                "ec2:DescribeVolumesModifications",
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
                    "aws:RequestTag/CSIVolumeSnapshotName": "*"
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
                    "aws:RequestTag/kubernetes.io/cluster/*": "owned"
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
                    "aws:ResourceTag/ebs.csi.aws.com/cluster": "true"
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
                    "aws:ResourceTag/CSIVolumeSnapshotName": "*"
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
                    "aws:ResourceTag/kubernetes.io/cluster/*": "owned"
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
                    "aws:ResourceTag/ebs.csi.aws.com/cluster": "true"
                }
            }
        }
    ]
}'

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would create IAM policy with document:"
		info "${POLICY_DOCUMENT}"
		POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${IAM_POLICY_NAME} (dry-run)"
	else
		POLICY_ARN=$(aws iam create-policy \
			--policy-name "${IAM_POLICY_NAME}" \
			--policy-document "${POLICY_DOCUMENT}" \
			--query "Policy.Arn" --output text)

		ok "IAM policy created: ${POLICY_ARN}"
	fi
fi

# ─── Create IAM Role ─────────────────────────────────────────────────────────
header "Phase 2: Create IAM Role"

# Check if role already exists
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
		info "[DRY-RUN] Would create IAM role with assume role policy:"
		info "${ASSUME_ROLE_POLICY}"
	else
		aws iam create-role \
			--role-name "${IAM_ROLE_NAME}" \
			--assume-role-policy-document "${ASSUME_ROLE_POLICY}" >/dev/null

		ok "IAM role created: ${IAM_ROLE_NAME}"
	fi
else
	warn "Role ${IAM_ROLE_NAME} already exists"
	info "Checking if trust policy is up to date..."

	# Get current trust policy OIDC provider
	CURRENT_OIDC=$(aws iam get-role --role-name "${IAM_ROLE_NAME}" \
		--query "Role.AssumeRolePolicyDocument.Statement[0].Principal.Federated" \
		--output text 2>/dev/null || echo "")

	OIDC_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"

	if [[ "${CURRENT_OIDC}" != "${OIDC_ARN}" ]]; then
		warn "Trust policy OIDC provider mismatch:"
		warn "  Current:  ${CURRENT_OIDC}"
		warn "  Expected: ${OIDC_ARN}"
		warn "Updating trust policy..."

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
			info "[DRY-RUN] Would update IAM role trust policy with:"
			info "${ASSUME_ROLE_POLICY}"
		else
			aws iam update-assume-role-policy \
				--role-name "${IAM_ROLE_NAME}" \
				--policy-document "${ASSUME_ROLE_POLICY}" >/dev/null

			ok "IAM role trust policy updated"
		fi
	else
		ok "Trust policy is already correct"
	fi
fi

# ─── Attach Policy to Role ───────────────────────────────────────────────────
header "Phase 3: Attach Policy to Role"

# Check if policy is already attached
if ! aws iam list-attached-policy-roles --role-name "${IAM_ROLE_NAME}" |
	grep -q "${IAM_POLICY_NAME}"; then
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

# ─── Configure Kubernetes Service Account ────────────────────────────────────
header "Phase 4: Configure Kubernetes Service Account"

# Check if annotation already exists
CURRENT_ANNOTATION=$(kubectl get serviceaccount "${SERVICE_ACCOUNT}" \
	-n "${NAMESPACE}" \
	-o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")

if [[ "${CURRENT_ANNOTATION}" != "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}" ]]; then
	info "Annotating service account with IAM role ARN..."

	if [[ "${DRY_RUN}" == "true" ]]; then
		info "[DRY-RUN] Would annotate ${SERVICE_ACCOUNT} with:"
		info "  eks.amazonaws.com/role-arn: arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}"
	else
		kubectl annotate serviceaccount "${SERVICE_ACCOUNT}" \
			-n "${NAMESPACE}" \
			eks.amazonaws.com/role-arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}" \
			--overwrite >/dev/null

		ok "Service account annotated"
	fi
else
	ok "Service account already has correct annotation"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
header "Setup Complete"

if [[ "${DRY_RUN}" == "true" ]]; then
	info "This was a dry run. No changes were made."
	info "Run without --dry-run to apply the changes."
else
	cat <<EOF

  IAM Role ARN:  arn:aws:iam::${AWS_ACCOUNT_ID}:role/${IAM_ROLE_NAME}
  Policy ARN:    ${POLICY_ARN}
  Service:       ${NAMESPACE}/${SERVICE_ACCOUNT}

  ──────────────────────────────────────────────────────────────
  NEXT STEPS
  ──────────────────────────────────────────────────────────────

  1. Verify the annotation was applied:
     kubectl get serviceaccount ${SERVICE_ACCOUNT} -n ${NAMESPACE} -o yaml

  2. Restart the EBS CSI driver pods to pick up the new role:
     kubectl rollout restart deployment ebs-csi-controller -n ${NAMESPACE}
     kubectl rollout restart daemonset ebs-csi-node -n ${NAMESPACE}

  3. Verify the controller pod has the new role:
     kubectl describe pod -n ${NAMESPACE} -l app=ebs-csi-controller

  4. Test EBS PVC creation:
     cat <<EOFPVC | kubectl apply -f -
     apiVersion: v1
     kind: PersistentVolumeClaim
     metadata:
       name: test-ebs
       namespace: ${NAMESPACE}
     spec:
       accessModes:
         - ReadWriteOnce
       storageClassName: gp3
       resources:
         requests:
           storage: 1Gi
     EOFPVC

     # Then check if it binds:
     kubectl get pvc test-ebs -n ${NAMESPACE}

EOF

fi

info "To clean up later, run: $0 --cleanup --cluster ${CLUSTER_NAME}"
