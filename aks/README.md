# Installing CLM standalone

At the moment of the writing we do not official support AKS onboading of CLM from Cast AI Console. So if you want to try this example you should execute the command manually.

```bash
 helm repo add castai https://castai.github.io/helm-charts || true
 helm repo update castai
 helm install test castai/castai-live --create-namespace \
  --namespace ${HELM_NAMESPACE} \
  --set castai.development=true \
  --set castai-aws-vpc-cni.enabled=false \
  --set castai.apiKey=${CAST_AI_CONSOLE_API_KEY} \
  --set castai.apiURL=${CAST_AI_CONSOLE_API_URL} \
  --set castai.clusterID=${CLUSTER_ID}
```

# Risk Manager - Container Live Migration Demo

A real-time portfolio analytics platform used to demonstrate **container live migration with zero downtime** on Kubernetes. The application simulates a risk management dashboard (similar to BlackRock Aladdin) backed by Azure SQL, with live-ticking market data.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                     │
│                                                          │
│  ┌──────────────────┐     ┌──────────────────────────┐  │
│  │  Frontend (nginx) │────▶│  Backend (.NET 8 API)    │  │
│  │  2 replicas       │     │  3 replicas              │  │
│  │  Port 80          │     │  Port 8080               │  │
│  │  LoadBalancer      │     │  1 pod per node          │  │
│  └──────────────────┘     │  CAST AI node selector    │  │
│                            │  MarketSimulator (2s tick)│  │
│                            └───────────┬──────────────┘  │
│                                        │                  │
└────────────────────────────────────────┼──────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    Azure SQL         │
                              │    Basic (5 DTU)     │
                              │    RiskManagerDb     │
                              └─────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, Chart.js, Nginx |
| Backend | .NET 8, ASP.NET Core, Entity Framework Core |
| Database | Azure SQL (Basic tier, 5 DTU) |
| Container Registry | GCP Artifact Registry (`europe-central2-docker.pkg.dev/castlocal-filipe/live`) |
| Orchestration | Kubernetes (AKS with Calico CNI) |
| Node Management | CAST AI |

## What the App Does

- Displays a portfolio of **20 stock positions** across 6 sectors
- **MarketSimulator** background service ticks every **2 seconds** using Geometric Brownian Motion to update prices in Azure SQL
- Frontend polls all endpoints every **2 seconds** to show live data
- Dashboard shows: risk metrics (VaR, Beta, Sharpe), P&L chart, sector allocation, positions table
- **Status bar** at the bottom shows the **pod name**, uptime, DB latency, and connection status - this is the visual proof of migration

## Prerequisites

- [Azure CLI](https://aka.ms/installazurecli) (`az`) - logged in
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) - logged in
- `kubectl` - configured to point at your AKS cluster
- `docker` with BuildKit/buildx support (Docker Desktop)
- `openssl`

## E2E Deployment Guide

### Step 1: Create the AKS Cluster (if needed)

```bash
./scripts/aks-calico-cluster.sh
```

This creates an AKS cluster in `westcentralus` with Calico CNI (network policy support). Edit the `RG_NAME` variable in the script to change the resource group name.

### Step 2: Build and Push Docker Images

```bash
./scripts/build-and-push.sh
```

Builds both images for `linux/amd64` (cross-compiled via `docker buildx` for Apple Silicon compatibility) and pushes to GCP Artifact Registry.

To tag a specific version:

```bash
./scripts/build-and-push.sh v1.0.0
```

### Step 3: Provision Azure SQL Database

```bash
./scripts/setup-azure-sql.sh
```

This script automatically:

1. Detects the AKS cluster's **resource group** and **region** from your kubectl context
2. Creates an Azure SQL Server (`risk-manager-sql-<random>`) in the same RG and region
3. Creates the `RiskManagerDb` database (Basic tier, 5 DTU, ~$5/month)
4. Generates a secure random admin password
5. Adds a firewall rule to allow Azure services
6. Creates the Kubernetes secret `azure-sql-secret` with the connection string

The connection string is automatically injected into the backend pods via the K8s secret - no manual editing required.

### Step 4: Deploy to Kubernetes

```bash
./scripts/deploy.sh
```

Deploys all manifests in order:

1. Namespace (`risk-manager`)
2. Secrets (skips if already created by `setup-azure-sql.sh`)
3. Backend deployment + service (3 replicas, 1 per node)
4. Frontend deployment + service (2 replicas, LoadBalancer)
5. PodDisruptionBudgets
6. Waits for rollout completion

The script prints the **frontend external IP** at the end. Open it in a browser to see the dashboard.

### Step 5: Demo Live Migration

```bash
./scripts/demo-migration.sh
```

With the dashboard open in a browser:

1. Shows current pods and which nodes they're on
2. Press ENTER to trigger a **rolling restart**
3. Watch the status bar: **pod name changes** while charts/data keep updating
4. Zero downtime confirmed

## Kubernetes Configuration

### Backend Pods

| Setting | Value |
|---|---|
| Replicas | 3 |
| Memory request/limit | 2Gi |
| CPU request/limit | 100m / 500m |
| Node selector | `live.cast.ai: "true"` |
| Toleration | `scheduling.cast.ai/node-template` (NoSchedule) |
| Topology spread | `maxSkew: 1` per hostname, `DoNotSchedule` |
| Rolling update | `maxUnavailable: 0`, `maxSurge: 1` |
| PDB | `minAvailable: 1` |

### Frontend Pods

| Setting | Value |
|---|---|
| Replicas | 2 |
| Memory request/limit | 64Mi / 128Mi |
| CPU request/limit | 50m / 200m |
| Service type | LoadBalancer |
| PDB | `minAvailable: 1` |

## Resilience

### Backend

| Layer | Timeout | Retry |
|---|---|---|
| SQL connection | 30s (from connection string) | EF Core auto-reconnect |
| SQL commands | 15s per query | 5 retries, up to 10s exponential backoff |
| Health check DB ping | 5s dedicated timeout | Returns "timeout" to UI |
| MarketSimulator tick | 10s per tick | Exponential backoff: 2s, 4s, 8s... cap 30s |

### Frontend

| Layer | Timeout | Retry |
|---|---|---|
| HTTP requests | 8s | 3 retries, exponential backoff (500ms, 1s, 2s) |
| Polling | 2s interval | Skips tick if previous request in-flight |
| Data display | -- | Keeps last known good data on errors |
| Error display | -- | Shows "timeout" / "server error (N)" / "connection failed" |

## Project Structure

```
aks/
├── k8s/
│   ├── namespace.yaml              # risk-manager namespace
│   ├── secret.yaml                 # Placeholder (setup-azure-sql.sh creates the real one)
│   ├── backend-deployment.yaml     # 3 replicas, CAST AI node selector, topology spread
│   ├── backend-service.yaml        # ClusterIP :8080
│   ├── frontend-deployment.yaml    # 2 replicas
│   ├── frontend-service.yaml       # LoadBalancer :80
│   └── pdb.yaml                    # PodDisruptionBudgets (minAvailable: 1)
├── scripts/
│   ├── aks-calico-cluster.sh       # Create AKS cluster with Calico CNI
│   ├── build-and-push.sh           # Build linux/amd64 images, push to GCP AR
│   ├── setup-azure-sql.sh          # Provision Azure SQL + K8s secret
│   ├── deploy.sh                   # Deploy all manifests to K8s
└── src/
    ├── backend/                    # .NET 8 API
    │   └── RiskManager.Api/
    │       ├── Controllers/        # Health, Portfolio, Risk endpoints
    │       ├── Data/               # EF Core DbContext + seed data
    │       ├── Models/             # Position, PnLDataPoint, RiskMetrics, etc.
    │       ├── Services/           # MarketSimulator (GBM price ticks every 2s)
    │       └── Dockerfile
    └── frontend/                   # React SPA
        ├── src/
        │   ├── components/         # Dashboard, Charts, Tables, Status bar
        │   ├── hooks/              # usePolling (2s, non-stacking, retry-aware)
        │   ├── services/           # Axios client with retry interceptor
        │   └── types/
        ├── nginx.conf              # Reverse proxy /api/ to backend service
        └── Dockerfile
```

## Cleanup

```bash
# Remove Azure SQL server and K8s secret
./scripts/setup-azure-sql.sh --cleanup

# Delete the entire namespace
kubectl delete namespace risk-manager

# Delete the AKS resource group (destroys everything)
az group delete --name filipe-19-02-2026 --yes --no-wait
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Pod name, uptime, DB connectivity + latency, version |
| GET | `/api/portfolio/positions` | All 20 positions with live prices |
| GET | `/api/portfolio/summary` | AUM, P&L, sector allocations, top 5 holdings |
| GET | `/api/risk/metrics` | VaR(95%), VaR(99%), Beta, Sharpe, Volatility, MaxDrawdown |
| GET | `/api/risk/pnl` | 30-day P&L history |
