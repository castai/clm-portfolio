#!/bin/bash
set -euo pipefail

# ============================================================================
# Build and Push Docker images to GCP Artifact Registry
# Registry: europe-central2-docker.pkg.dev/castlocal-filipe/live
# Uses buildx to cross-compile for amd64 from Apple Silicon
# ============================================================================

REGISTRY="europe-central2-docker.pkg.dev/castlocal-filipe/live"
TAG="${1:-latest}"
PLATFORM="linux/amd64"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  Risk Manager - Build & Push"
echo "  Registry: ${REGISTRY}"
echo "  Tag: ${TAG}"
echo "  Platform: ${PLATFORM}"
echo "============================================"

# Configure Docker for GCP Artifact Registry
echo ""
echo "[1/6] Configuring Docker for GCP Artifact Registry..."
gcloud auth configure-docker europe-central2-docker.pkg.dev --quiet

# Ensure buildx builder exists
echo ""
echo "[2/6] Setting up buildx builder..."
docker buildx create --name amd64builder --use 2>/dev/null || docker buildx use amd64builder 2>/dev/null || true
docker buildx inspect --bootstrap

# Build & push backend
echo ""
echo "[3/6] Building & pushing backend image (${PLATFORM})..."
docker buildx build \
	--platform "${PLATFORM}" \
	--push \
	-t "${REGISTRY}/risk-manager-backend:${TAG}" \
	-f "${PROJECT_ROOT}/src/backend/RiskManager.Api/Dockerfile" \
	"${PROJECT_ROOT}/src/backend/RiskManager.Api"

# Build & push frontend
echo ""
echo "[4/6] Building & pushing frontend image (${PLATFORM})..."
docker buildx build \
	--platform "${PLATFORM}" \
	--push \
	-t "${REGISTRY}/risk-manager-frontend:${TAG}" \
	-f "${PROJECT_ROOT}/src/frontend/Dockerfile" \
	"${PROJECT_ROOT}/src/frontend"

echo ""
echo "============================================"
echo "  Done! Images pushed successfully."
echo ""
echo "  Backend:  ${REGISTRY}/risk-manager-backend:${TAG}"
echo "  Frontend: ${REGISTRY}/risk-manager-frontend:${TAG}"
echo "============================================"
