#!/usr/bin/env bash
set -euo pipefail

az acr build --registry cr3cinvoice --image vetcostcheck-ui:v1 .
az containerapp update \
  --name ca-vetcostcheck-ui \
  --resource-group rg-3c-invoice \
  --image cr3cinvoice.azurecr.io/vetcostcheck-ui:v1
