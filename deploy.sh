#!/usr/bin/env bash
set -euo pipefail

# Deploy the Streamlit UI to Azure Container Apps.
#
# Each deploy uses a UNIQUE image tag (git SHA + UTC timestamp). This matters:
# `az containerapp update` only creates a new revision when the template
# changes, so reusing a mutable tag like ":v1" silently keeps the OLD image
# running. A unique tag guarantees every deploy actually rolls out.

REGISTRY="cr3cinvoice"
REGISTRY_SERVER="cr3cinvoice.azurecr.io"
IMAGE="vetcostcheck-ui"
APP="ca-vetcostcheck-ui"
RESOURCE_GROUP="rg-3c-invoice"

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG="${SHA}-$(date -u +%Y%m%d%H%M%S)"
FULL_IMAGE="${REGISTRY_SERVER}/${IMAGE}:${TAG}"

echo "Building ${FULL_IMAGE} ..."
# Tag this build with both the unique tag and a moving ":latest" for reference.
az acr build --registry "$REGISTRY" \
  --image "${IMAGE}:${TAG}" \
  --image "${IMAGE}:latest" \
  .

echo "Rolling out ${FULL_IMAGE} ..."
az containerapp update \
  --name "$APP" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$FULL_IMAGE"

echo ""
echo "Deployed. Active revisions:"
az containerapp revision list \
  --name "$APP" --resource-group "$RESOURCE_GROUP" \
  --query "[?properties.active].{revision:name, running:properties.runningState, traffic:properties.trafficWeight, image:properties.template.containers[0].image}" \
  -o table

FQDN="$(az containerapp show --name "$APP" --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)"
echo ""
echo "URL: https://${FQDN}"
