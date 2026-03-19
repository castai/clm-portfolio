# Coder Live Migration Test

Deploys a **real Coder instance** on AWS EKS and creates two workspaces running heavy build workloads to validate that CAST AI live migration works end-to-end with Coder workspaces.

| Workspace | Workload | Resources | Storage |
|---|---|---|---|
| `kernel-build` | Linux kernel compilation (`make -j16`) | 16 CPU, 32Gi req / 96Gi limit | 50Gi gp2 |
| `aosp-build` | AOSP build (`repo sync` + `lunch` + `make -j16`) | 32 CPU, 64Gi req / 128Gi limit | 500Gi gp3 |

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
│  │  Workspace: kernel-build                                      │ │
│  │  codercom/enterprise-base:ubuntu                              │ │
│  │  16 CPU / 32Gi request / 96Gi limit                           │ │
│  │  50Gi gp2 EBS PVC (/home/coder)                               │ │
│  │  Workload: make -j16 (Linux kernel build)                     │ │
│  │  Labels: live.cast.ai/migration-enabled=true                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Workspace: aosp-build                                        │ │
│  │  codercom/enterprise-base:ubuntu                              │ │
│  │  32 CPU / 64Gi request / 128Gi limit                          │ │
│  │  500Gi gp3 EBS PVC (/home/coder)                              │ │
│  │  Workload: AOSP repo sync + lunch + make -j16                 │ │
│  │  Labels: live.cast.ai/migration-enabled=true                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Components

| Namespace | Component | Purpose |
|---|---|---|
| `coder` | Helm release `coder` | Coder control plane (coderd + provisionerd) |
| `coder` | Helm release `postgresql` | PostgreSQL database for Coder state |
| `coder` | Template `kernel-build` | Terraform template: kernel build workspace |
| `coder` | Template `aosp-build` | Terraform template: AOSP build workspace |
| `coder` | Workspace `kernel-build` | Workspace pod with Coder agent + kernel build tools |
| `coder` | Workspace `aosp-build` | Workspace pod with Coder agent + AOSP build tools |

## Prerequisites

- `kubectl` connected to an EKS cluster
- `helm` 3.5+
- EBS CSI driver addon installed (`kubectl get csidriver ebs.csi.aws.com`)
- `gp2` StorageClass available (standard on EKS)
- `gp3` StorageClass (created by `setup-ebs.sh` -- required for AOSP workspace)
- CAST AI node template with label `scheduling.cast.ai/node-template: live-migration`
- Nodes large enough for AOSP: at least 32 CPU + 128Gi RAM (e.g. `m5.8xlarge`, `r5.4xlarge`)
- Outbound internet for `git.kernel.org`, `android.googlesource.com`, `apt.ubuntu.com`, `helm.coder.com`, `charts.bitnami.com`
- No LoadBalancer or public internet required -- access is via `kubectl port-forward`

### Install EBS CSI driver (if missing)

```bash
aws eks create-addon --addon-name aws-ebs-csi-driver --cluster-name <your-cluster-name>
```

### Setup EBS (IAM + gp3 StorageClass + CSI driver)

The `setup-ebs.sh` script handles everything the EBS CSI driver needs. It auto-detects the cluster name from your kubectl context (or you can specify `--cluster`):

```bash
# Dry run first to see what will be created
./setup-ebs.sh --dry-run

# Apply everything: install CSI addon, IAM policy + role + IRSA, gp3 StorageClass, restart controller
./setup-ebs.sh

# Or specify cluster explicitly
./setup-ebs.sh --cluster <your-cluster-name>

# Clean up later if needed
./setup-ebs.sh --cleanup
```

The script will:
1. Install the `aws-ebs-csi-driver` EKS addon (if missing)
2. Create IAM policy + role with OIDC trust
3. Annotate the CSI service account (IRSA)
4. Restart the EBS CSI controller (to pick up the new role)
5. Create the `gp3` StorageClass (encrypted, WaitForFirstConsumer)

See [EBS CSI IAM Setup](#ebs-csi-iam-setup) below for details.

## Usage

```bash
# Deploy everything: PostgreSQL + Coder + both templates + both workspaces
./setup.sh

# Trigger kernel build inside the kernel-build workspace
./setup.sh build

# Sync AOSP source inside the aosp-build workspace (takes 2-4 hours)
./setup.sh sync-aosp

# Build AOSP inside the aosp-build workspace (takes 4-8 hours)
./setup.sh build-aosp

# Tear down everything
./setup.sh teardown
```

## What `./setup.sh` Does

1. **Pre-flight checks** -- validates kubectl, helm, EBS CSI driver, gp2/gp3 StorageClasses
2. **Phase 1: PostgreSQL** -- installs Bitnami PostgreSQL via Helm, creates DB URL secret
3. **Phase 2: Coder control plane** -- installs Coder via Helm (ClusterIP service, `CODER_ACCESS_URL` set to cluster-internal address)
4. **Phase 3: Port-forward & CLI setup** -- starts `kubectl port-forward` on localhost:6770, installs Coder CLI, creates admin user
5. **Phase 4: Templates & Workspaces** -- pushes both Terraform templates, creates both workspaces (kernel-build + aosp-build)
6. **Phase 5: Validation** -- shows pod status, labels, node placement for both workspaces

## Testing Live Migration

### Kernel Build

1. Deploy the full stack:

```bash
./setup.sh
```

2. Verify workspaces are running:

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
kubectl get pod -n coder -l com.coder.workspace.name=kernel-build -o wide
```

6. Trigger a CAST AI live migration (via console or API).

7. After migration, verify:

```bash
# Pod is on a new node
kubectl get pod -n coder -l com.coder.workspace.name=kernel-build -o wide

# Build process survived
coder ssh kernel-build -- pgrep -fa make

# Coder agent reconnected
coder list

# Build artifacts (after completion)
coder ssh kernel-build -- ls -la ~/linux/vmlinux
```

### AOSP Build

The AOSP build reproduces the customer's exact environment:

1. `mkdir AOSP && cd AOSP`
2. Install `repo` tool
3. `repo init --partial-clone --no-use-superproject -b android-latest-release -u https://android.googlesource.com/platform/manifest`
4. `repo sync -c -j8`
5. `source build/envsetup.sh && lunch aosp_cf_x86_64_phone-userdebug`
6. `make -j16`

To test:

1. Sync the AOSP source (takes 2-4 hours):

```bash
./setup.sh sync-aosp
```

2. Start the AOSP build (takes 4-8 hours):

```bash
./setup.sh build-aosp
```

3. While the build is running, trigger a CAST AI live migration.

4. After migration, verify:

```bash
# Pod is on a new node
kubectl get pod -n coder -l com.coder.workspace.name=aosp-build -o wide

# Build process survived
coder ssh aosp-build -- pgrep -fa make

# Coder agent reconnected
coder list
```

If the `make` process is still running, the Coder agent reconnected, and the build eventually completes -- live migration preserved everything.

## Workspace Configuration

### kernel-build

| Setting | Value |
|---|---|
| Image | `codercom/enterprise-base:ubuntu` |
| CPU request / limit | 16 / 16 |
| Memory request / limit | 32Gi / 96Gi |
| Home disk (PVC) | 50Gi gp2 EBS |
| Coder Agent | SSH, port-forwarding, health checks, metrics |
| Node selector | `scheduling.cast.ai/node-template: live-migration` |
| Migration label | `live.cast.ai/migration-enabled: "true"` |
| Build trigger | `./setup.sh build` or `coder ssh kernel-build -- build-kernel` |
| Build | Linux 6.12.y stable, `defconfig`, `make -j16` |

### aosp-build

| Setting | Value |
|---|---|
| Image | `codercom/enterprise-base:ubuntu` |
| CPU request / limit | 32 / 32 |
| Memory request / limit | 64Gi / 128Gi |
| Home disk (PVC) | 500Gi gp3 EBS (3,000 IOPS, 125 MiB/s baseline) |
| Coder Agent | SSH, port-forwarding, health checks, metrics |
| Node selector | `scheduling.cast.ai/node-template: live-migration` |
| Migration label | `live.cast.ai/migration-enabled: "true"` |
| AOSP branch | `android-latest-release` |
| Lunch target | `aosp_cf_x86_64_phone-userdebug` |
| Sync trigger | `./setup.sh sync-aosp` or `coder ssh aosp-build -- sync-aosp` |
| Build trigger | `./setup.sh build-aosp` or `coder ssh aosp-build -- build-aosp` |
| ccache | Enabled (`USE_CCACHE=1`, persisted on PVC) |

## Files

| File | Description |
|---|---|
| `setup.sh` | Main script: deploys Coder stack, creates both workspaces, provides `build`, `sync-aosp`, `build-aosp`, `teardown` |
| `setup-ebs.sh` | Fixes EBS CSI driver IAM permissions via IRSA + creates gp3 StorageClass |
| `values.yaml` | Helm values for Coder control plane (ClusterIP, internal access URL) |
| `template/main.tf` | Coder Terraform template: kernel-build workspace with CAST AI labels |
| `template-aosp/main.tf` | Coder Terraform template: AOSP build workspace with CAST AI labels |

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

The `setup-ebs.sh` script is a one-shot setup for everything the EBS CSI driver needs. It auto-detects the cluster name from your current kubectl context (or accepts `--cluster <name>`).

What it does:

1. **Creates the IAM OIDC identity provider** for the cluster (required for IRSA to work)
2. **Creates IAM policy** (`AmazonEBSCSIDriverPolicy`) with minimum permissions for EBS volume management
3. **Creates IAM role** (`AmazonEKS_EBS_CSI_DriverRole`) with OIDC trust policy for the CSI service account
4. **Attaches the policy** to the role
5. **Installs/updates the EBS CSI driver addon** with `--service-account-role-arn` (the AWS-recommended approach -- ensures the addon picks up IAM credentials immediately)
6. **Creates a `gp3` StorageClass** with encryption enabled and `WaitForFirstConsumer` binding mode

The `gp3` StorageClass provides:
- **3,000 baseline IOPS** (free, vs gp2's 3 IOPS/GiB = 150 IOPS for 50Gi)
- **125 MiB/s baseline throughput** (free)
- **20% cheaper per GiB** than gp2
- **Volume expansion** enabled (can grow the 500Gi PVC without recreating)

```bash
# Auto-detect cluster from kubectl context
./setup-ebs.sh

# Or specify cluster explicitly
./setup-ebs.sh --cluster <your-cluster-name>

# Preview changes without applying
./setup-ebs.sh --dry-run

# Remove everything this script created
./setup-ebs.sh --cleanup
```

## References

- [Coder Docs](https://coder.com/docs) -- Coder platform documentation
- [Coder Kubernetes Install](https://coder.com/docs/install/kubernetes) -- Helm installation guide
- [Linux Kernel Source](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git) -- Stable kernel repository
- [AOSP Setup](https://source.android.com/docs/setup/start) -- Android Open Source Project build setup
- [AOSP Download](https://source.android.com/docs/setup/download) -- AOSP source download instructions
- [AOSP Building](https://source.android.com/docs/setup/build/building) -- AOSP build instructions
- [CAST AI Live Migration](https://docs.cast.ai/docs/live-migration/) -- Zero-downtime container migration
- [EBS CSI Driver](https://github.com/kubernetes-sigs/aws-ebs-csi-driver) -- Kubernetes CSI driver for EBS
- [EBS CSI Parameters](https://github.com/kubernetes-sigs/aws-ebs-csi-driver/blob/master/docs/parameters.md) -- StorageClass parameters reference
- [EBS CSI IAM Permissions](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html) -- AWS docs on EBS CSI driver setup
- [EBS gp3 Volumes](https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html#gp3-ebs-volume-type) -- gp3 volume performance specs
