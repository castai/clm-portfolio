#! /bin/bash

RG_NAME="filipe-19-02-2026"

az group create --name "$RG_NAME" --location northeurope
az aks create \
	--resource-group "$RG_NAME" \
	--name "$RG_NAME" \
	--location northeurope \
	--pod-cidr 192.168.0.0/16 \
	--network-plugin none

echo "Waiting for Azure to propagate the cluster resource..."
until az aks get-credentials --resource-group "$RG_NAME" --name "$RG_NAME" --overwrite-existing 2>/dev/null; do
	echo "  Cluster not found yet, retrying in 30s..."
	sleep 30
done
echo "Credentials retrieved."

kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.31.3/manifests/operator-crds.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.31.3/manifests/tigera-operator.yaml

kubectl create -f - <<EOF
kind: Installation
apiVersion: operator.tigera.io/v1
metadata:
  name: default
spec:
  kubernetesProvider: AKS
  cni:
    type: Calico
  calicoNetwork:
    bgp: Disabled
    ipPools:
     - cidr: 192.168.0.0/16
       encapsulation: VXLAN
---

# This section configures the Calico API server.
# For more information, see: https://docs.tigera.io/calico/latest/reference/installation/api#operator.tigera.io/v1.APIServer
apiVersion: operator.tigera.io/v1
kind: APIServer
metadata:
   name: default
spec: {}

---

# Configures the Calico Goldmane flow aggregator.
apiVersion: operator.tigera.io/v1
kind: Goldmane
metadata:
  name: default

---

# Configures the Calico Whisker observability UI.
apiVersion: operator.tigera.io/v1
kind: Whisker
metadata:
  name: default
EOF

kubectl get tigerastatus
