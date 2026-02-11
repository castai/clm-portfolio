# Airflow + Spark on Kubernetes

This setup deploys **Apache Airflow 3.0** with the **Spark Kubernetes Operator** on a Kubernetes cluster. Airflow orchestrates a PySpark pi calculation job that runs natively on Kubernetes using `spark-submit` in client mode.

## Architecture

```
Airflow Scheduler (KubernetesExecutor)
  |
  |-- creates executor pod (airflow-worker SA)
        |
        |-- KubernetesPodOperator
              |
              |-- launches Spark driver pod (spark SA)
                    |-- spark-submit --deploy-mode client
                    |-- creates Spark executor pods via K8s API
                    |-- executor pods connect back via headless Service
```

**Key components:**

- **Airflow** (`apache-airflow/airflow` Helm chart) -- DAG scheduling, UI, task orchestration
- **Spark Operator** (`spark/spark-operator` Helm chart) -- manages Spark workload RBAC and service accounts
- **KubernetesExecutor** -- Airflow creates a pod per task (no persistent workers)
- **KubernetesPodOperator** -- the DAG task launches a Spark driver pod that runs `spark-submit`
- **Headless Service** -- stable DNS for executor-to-driver communication

## What the DAG Does

The `spark_submit_pi` DAG:

1. Airflow scheduler creates an executor pod to run the task
2. The task uses `KubernetesPodOperator` to launch a Spark driver pod (`apache/spark:4.0.0-python3`)
3. The driver pod runs `spark-submit` in **client mode** against the Kubernetes API (`k8s://https://kubernetes.default.svc:443`)
4. Spark creates executor pods that perform a Monte Carlo pi estimation across 1000 partitions
5. Executors connect back to the driver via the `spark-pi-driver-svc` headless Service

## Prerequisites

- A running Kubernetes cluster (`kubectl` configured)
- Helm 3 installed

## Deploy

```bash
./deploy.sh
```

This runs 7 steps:

1. Creates the `airflow-spark-operator` namespace
2. Adds the Spark Operator Helm repo
3. Installs the Spark Kubernetes Operator (creates the `spark` service account)
4. Applies RBAC (grants Airflow SAs permission to create pods and SparkApplications)
5. Applies ConfigMaps (DAG definition) and the driver headless Service
6. Adds the Airflow Helm repo
7. Installs Airflow with KubernetesExecutor

## Access the Airflow UI

```bash
kubectl port-forward svc/airflow-api-server 8080:8080 -n airflow-spark-operator
```

Open http://localhost:8080 and log in with `admin` / `admin`.

## Trigger the DAG

In the Airflow UI, find the `spark_submit_pi` DAG, enable it, and trigger it manually.

## Watch Logs

```bash
# Watch pods spin up (driver + executors)
kubectl get pods -n airflow-spark-operator -w

# Follow the Spark driver logs
kubectl logs -f -l spark-role=driver -n airflow-spark-operator
```

## Teardown

```bash
helm uninstall airflow -n airflow-spark-operator
helm uninstall spark-operator -n airflow-spark-operator
kubectl delete namespace airflow-spark-operator
```

## File Overview

| File | Purpose |
|---|---|
| `deploy.sh` | One-command deployment script |
| `namespace.yaml` | Kubernetes namespace definition |
| `spark-operator-values.yaml` | Helm values for Spark Operator v2.4.0 |
| `airflow-values.yaml` | Helm values for Airflow 3.0.2 (KubernetesExecutor) |
| `rbac.yaml` | Role/RoleBinding granting Airflow SAs pod and SparkApplication permissions |
| `dags-configmap.yaml` | Airflow DAG: PySpark pi calculation via `spark-submit` |
| `spark-app-configmap.yaml` | SparkApplication CRD template (for future use) |
| `driver-headless-service.yaml` | Headless Service for Spark executor-to-driver connectivity |
