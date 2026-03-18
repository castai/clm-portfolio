# Coder Live Migration Test

Deploys a **real Coder instance** on AWS EKS and creates a workspace running a heavy workload (Linux kernel compilation) to validate that CAST AI live migration works end-to-end with Coder workspaces.

This is not a simulation -- it deploys the full Coder stack (control plane, PostgreSQL, Terraform-provisioned workspaces with Coder agent) so that migration is tested against the real product.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster (EKS)                     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Coder Control Plane (Helm)                                   │ │
│  │  coderd + provisionerd                                        │ │
│  │  Service: ClusterIP (access via port-forward)                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌───────────────────┐       │  Terraform provisioner             │
│  │  PostgreSQL        │◄──────                                    │
│  │  (Bitnami Helm)    │       │                                   │
│  └───────────────────┘       ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Workspace Pod (created by Coder via Terraform)               │ │
│  │  codercom/enterprise-base:ubuntu                              │ │
│  │  16 CPU / 32Gi request / 96Gi limit                           │ │
│  │  50Gi gp2 EBS PVC (/home/coder)                               │ │
│  │  Coder Agent: SSH, port-forward, health checks                │ │
│  │  Workload: make -j16 (Linux kernel build)                     │ │
│  │  Labels: live.cast.ai/migration-enabled=true                  │ │
│  │  Node: scheduling.cast.ai/node-template=live-migration        │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Components

| Namespace | Component | Purpose |
|---|---|---|
| `coder` | Helm release `coder` | Coder control plane (coderd + provisionerd) |
| `coder` | Helm release `postgresql` | PostgreSQL database for Coder state |
| `coder` | Template `kernel-build` | Terraform template defining workspace shape |
| `coder` | Workspace `kernel-build` | Workspace pod with Coder agent + kernel build tools |

## Prerequisites

- `kubectl` connected to an EKS cluster
- `helm` 3.5+
- EBS CSI driver addon installed (`kubectl get csidriver ebs.csi.aws.com`)
- `gp2` StorageClass available (standard on EKS)
- CAST AI node template with label `scheduling.cast.ai/node-template: live-migration`
- Outbound internet for `git.kernel.org`, `apt.ubuntu.com`, `helm.coder.com`, `charts.bitnami.com`
- No LoadBalancer or public internet required -- access is via `kubectl port-forward`

### Install EBS CSI driver (if missing)

```bash
aws eks create-addon --addon-name aws-ebs-csi-driver --cluster-name <your-cluster-name>
```

If the EBS CSI driver is installed but PVC creation fails with `UnauthorizedOperation` errors, the driver's service account likely lacks IAM permissions. Use the included helper script to fix this:

```bash
# Dry run first to see what will be created
./setup-ebs.sh --cluster <your-cluster-name> --dry-run

# Apply the fix
./setup-ebs.sh --cluster <your-cluster-name>

# Clean up later if needed
./setup-ebs.sh --cluster <your-cluster-name> --cleanup
```

See [EBS CSI IAM Setup](#ebs-csi-iam-setup) below for details.

## Usage

```bash
# Deploy everything: PostgreSQL + Coder + template + workspace
./setup.sh

# Trigger kernel build inside the Coder workspace
./setup.sh build

# Tear down everything
./setup.sh teardown
```

## What `./setup.sh` Does

1. **Pre-flight checks** -- validates kubectl, helm, EBS CSI driver, gp2 StorageClass
2. **Phase 1: PostgreSQL** -- installs Bitnami PostgreSQL via Helm, creates DB URL secret
3. **Phase 2: Coder control plane** -- installs Coder via Helm (ClusterIP service, `CODER_ACCESS_URL` set to cluster-internal address)
4. **Phase 3: Port-forward & CLI setup** -- starts `kubectl port-forward` on localhost:6770, installs Coder CLI, creates admin user
5. **Phase 4: Template & Workspace** -- pushes Terraform template, creates workspace with 16 CPU / 96Gi / 50Gi disk
6. **Phase 5: Validation** -- shows pod status, labels, node placement

## Testing Live Migration

1. Deploy the full stack:

```bash
./setup.sh
```

2. Verify workspace is running and agent is connected:

```bash
coder list
```

3. Trigger the kernel build:

```bash
./setup.sh build
```

4. Verify the build is running:

```bash
coder ssh kernel-build -- pgrep -fa make
```

5. Note the current node:

```bash
kubectl get pod -n coder -l app.kubernetes.io/name=coder-workspace -o wide
```

6. Trigger a CAST AI live migration (via console or API).

7. After migration, verify:

```bash
# Pod is on a new node
kubectl get pod -n coder -l app.kubernetes.io/name=coder-workspace -o wide

# Build process survived
coder ssh kernel-build -- pgrep -fa make

# Coder agent reconnected
coder list

# Build artifacts (after completion)
coder ssh kernel-build -- ls -la ~/linux/vmlinux
```

If the `make` process is still running, the Coder agent reconnected, and the build eventually completes -- live migration preserved everything.

## Workspace Configuration

| Setting | Value |
|---|---|
| Image | `codercom/enterprise-base:ubuntu` |
| CPU request / limit | 16 / 16 |
| Memory request / limit | 32Gi / 96Gi |
| Home disk (PVC) | 50Gi gp2 EBS |
| Coder Agent | SSH, port-forwarding, health checks, metrics |
| Node selector | `scheduling.cast.ai/node-template: live-migration` |
| Migration label | `live.cast.ai/migration-enabled: "true"` |
| Tolerations | `scheduling.cast.ai/node-template`, `scheduling.cast.ai/live-migration`, `live.cast.ai` |
| Build trigger | Manual via `./setup.sh build` or `coder ssh kernel-build -- build-kernel` |
| Build | Linux 6.12.y stable, `defconfig`, `make -j16` |

## Files

| File | Description |
|---|---|
| `setup.sh` | Main script: deploys Coder stack, creates workspace, provides `build` and `teardown` |
| `setup-ebs.sh` | Fixes EBS CSI driver IAM permissions via IRSA (see below) |
| `values.yaml` | Helm values for Coder control plane (ClusterIP, internal access URL) |
| `template/main.tf` | Coder Terraform template: workspace pod with CAST AI labels, kernel build |

## Access

Coder is accessed via `kubectl port-forward` (no LoadBalancer, no public internet).

The `setup.sh` script starts a port-forward automatically. If it dies, restart it:

```bash
kubectl port-forward svc/coder -n coder 6770:80
```

Then open http://localhost:6770 in your browser.

## Coder Credentials (PoC)

| Field | Value |
|---|---|
| URL | `http://localhost:6770` (via port-forward) |
| Username | `admin` |
| Password | `castailive!` |

## EBS CSI IAM Setup

The `setup-ebs.sh` script fixes a common EKS issue where the EBS CSI driver lacks IAM permissions to provision volumes. This manifests as `UnauthorizedOperation: You are not authorized to perform this operation` errors (e.g. `ec2:DescribeAvailabilityZones`) when creating PVCs.

The script:

1. Creates an IAM policy (`AmazonEBSCSIDriverPolicy`) with the minimum permissions for EBS volume management (create, attach, detach, delete, snapshot, describe)
2. Creates an IAM role (`AmazonEKS_EBS_CSI_DriverRole`) with a trust policy for the cluster's OIDC provider
3. Attaches the policy to the role
4. Annotates the `ebs-csi-controller-sa` Kubernetes service account with the role ARN (IRSA)

```bash
# Required: --cluster flag with your EKS cluster name
./setup-ebs.sh --cluster <your-cluster-name>

# Preview changes without applying
./setup-ebs.sh --cluster <your-cluster-name> --dry-run

# Remove IAM resources created by this script
./setup-ebs.sh --cluster <your-cluster-name> --cleanup
```

After running the script, restart the EBS CSI controller to pick up the new role:

```bash
kubectl rollout restart deployment ebs-csi-controller -n kube-system
kubectl rollout restart daemonset ebs-csi-node -n kube-system
```

## References

- [Coder Docs](https://coder.com/docs) -- Coder platform documentation
- [Coder Kubernetes Install](https://coder.com/docs/install/kubernetes) -- Helm installation guide
- [Linux Kernel Source](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git) -- Stable kernel repository
- [CAST AI Live Migration](https://docs.cast.ai/docs/live-migration/) -- Zero-downtime container migration
- [EBS CSI Driver](https://github.com/kubernetes-sigs/aws-ebs-csi-driver) -- Kubernetes CSI driver for EBS
- [EBS CSI IAM Permissions](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html) -- AWS docs on EBS CSI driver setup
