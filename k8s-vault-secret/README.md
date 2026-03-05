# Vault Secret Migration Test

Validates that secrets mounted via HashiCorp Vault remain identical after pod migration (reschedule, node drain, pod deletion).

Deploys Vault in dev mode with **both** secret delivery methods — the **webhook injector** (sidecar) and the **CSI provider** — and runs two test pods that continuously SHA-256 hash their mounted secrets, comparing against a baseline every 10 seconds.

## Components

| Namespace | Component | Purpose |
|---|---|---|
| `kube-system` | Secrets Store CSI Driver | Provides the CSI volume interface for secret mounting |
| `vault` | Vault (dev mode) | KV v2 secret store with Kubernetes auth |
| `vault` | Vault Agent Injector | Mutating webhook that injects secret-rendering sidecars |
| `vault` | Vault CSI Provider | DaemonSet bridging CSI driver to Vault |
| `vault-test` | `csi-secret-test` | Test pod reading secrets from CSI volume at `/mnt/secrets-store/` |
| `vault-test` | `injector-secret-test` | Test pod reading secrets from injector sidecar at `/vault/secrets/` |

## Prerequisites

- `kubectl` connected to a cluster
- `helm` 3.6+
- Kubernetes 1.29+

## Usage

```bash
# Deploy everything
./setup.sh

# Tear down everything
./setup.sh teardown
```

## Testing Secret Integrity After Migration

1. Record baseline hashes from the running pods:

```bash
kubectl logs -n vault-test -l app=csi-secret-test --tail=3
kubectl logs -n vault-test -l app=injector-secret-test -c app --tail=3
```

2. Trigger a migration (delete the pod, drain the node, etc.):

```bash
kubectl delete pod -n vault-test -l app=csi-secret-test
kubectl delete pod -n vault-test -l app=injector-secret-test
```

3. Wait for the new pods to start, then check logs again:

```bash
kubectl logs -n vault-test -l app=csi-secret-test --tail=5
kubectl logs -n vault-test -l app=injector-secret-test -c app --tail=5
```

4. Compare the `secret_hash` values. They must match. Log output looks like:

```
[2026-03-05T12:15:00Z] [CSI] iteration=5 secret_hash=a1b2c3... baseline=a1b2c3... status=OK
  username=vault-test-user
  password=s3cr3t-p@ssw0rd-12345
  api_key=ak_7f8e9d0c1b2a3456
```

`status=OK` means the hash matches the baseline. `status=DRIFT_DETECTED` means the secret content changed.

## Files

| File | Description |
|---|---|
| `setup.sh` | Installs CSI driver, Vault, configures auth/policy/secrets, deploys test workloads |
| `serviceaccount.yaml` | ServiceAccount bound to the Vault Kubernetes auth role |
| `secret-provider-class.yaml` | SecretProviderClass CRD for the CSI volume mount path |
| `deploy-csi-test.yaml` | Deployment that mounts secrets via CSI and hashes them |
| `deploy-injector-test.yaml` | Deployment that receives secrets via webhook injector and hashes them |

## References

- [HashiCorp Vault](https://www.vaultproject.io/) — Secrets management, encryption as a service, and privileged access management
- [Vault Helm Chart](https://github.com/hashicorp/vault-helm) — Official Helm chart for deploying Vault on Kubernetes
- [Vault Agent Injector](https://developer.hashicorp.com/vault/docs/platform/k8s/injector) — Mutating webhook that injects Vault Agent sidecars into pods
- [Vault CSI Provider](https://developer.hashicorp.com/vault/docs/platform/k8s/csi) — DaemonSet that allows Kubernetes to mount Vault secrets via the CSI interface
- [Secrets Store CSI Driver](https://secrets-store-csi-driver.sigs.k8s.io/) — Kubernetes SIG project that integrates secret stores with the Container Storage Interface
- [Vault KV Secrets Engine v2](https://developer.hashicorp.com/vault/docs/secrets/kv/kv-v2) — Versioned key-value secret storage backend
- [Vault Kubernetes Auth Method](https://developer.hashicorp.com/vault/docs/auth/kubernetes) — Authenticates pods using Kubernetes Service Account tokens
- [Kubernetes ServiceAccount](https://kubernetes.io/docs/concepts/security/service-accounts/) — Identity for processes running in pods
- [SecretProviderClass CRD](https://secrets-store-csi-driver.sigs.k8s.io/concepts#secretproviderclass) — Custom resource that defines how secrets are fetched and mounted via CSI
- [Helm](https://helm.sh/) — Package manager for Kubernetes
