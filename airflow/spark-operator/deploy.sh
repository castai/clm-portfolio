#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="airflow-spark-operator"

echo "=== Setup 1: Airflow + Apache Spark Kubernetes Operator ==="
echo ""

# 1. Create namespace
echo "[1/7] Creating namespace ${NAMESPACE}..."
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"

# 2. Install Apache Spark Kubernetes Operator
echo "[2/7] Adding Spark Kubernetes Operator helm repo..."
helm repo add spark https://kubeflow.github.io/spark-operator 2>/dev/null || true
helm repo update spark

echo "[3/7] Installing Apache Spark Kubernetes Operator..."
helm upgrade --install spark-operator spark/spark-operator \
  --namespace "${NAMESPACE}" \
  --values "${SCRIPT_DIR}/spark-operator-values.yaml" \
  --wait --timeout 5m

# 4. Apply RBAC
echo "[4/7] Applying RBAC..."
kubectl apply -f "${SCRIPT_DIR}/rbac.yaml"

# 5. Apply ConfigMaps and Services
echo "[5/7] Applying ConfigMaps and headless Service..."
kubectl apply -f "${SCRIPT_DIR}/dags-configmap.yaml"
kubectl apply -f "${SCRIPT_DIR}/spark-app-configmap.yaml"
kubectl apply -f "${SCRIPT_DIR}/driver-headless-service.yaml"

# 6. Add Airflow helm repo (idempotent)
echo "[6/7] Adding Airflow helm repo..."
helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
helm repo update apache-airflow

# 7. Install Airflow
echo "[7/7] Installing Apache Airflow..."
helm upgrade --install airflow apache-airflow/airflow \
  --namespace "${NAMESPACE}" \
  --values "${SCRIPT_DIR}/airflow-values.yaml" \
  --timeout 10m

echo ""
echo "=== Setup 1 deployed! Waiting for pods to become ready... ==="
kubectl wait --for=condition=ready pod -l component=api-server -n "${NAMESPACE}" --timeout=300s 2>/dev/null || true
echo ""
echo "Airflow UI: kubectl port-forward svc/airflow-api-server 8080:8080 -n ${NAMESPACE}"
echo "Login: admin / admin"
echo "DAG: spark_operator_wordcount"
