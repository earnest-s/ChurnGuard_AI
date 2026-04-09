#!/usr/bin/env bash
set -euo pipefail

RG="churn-rg"
LOCATION="centralindia"
IMAGE_NAME="churn-api"
ACI_NAME="churn-api"
SUFFIX="$(date +%s)$RANDOM"
ACR_NAME="churnacr${SUFFIX}"
ACR_NAME="${ACR_NAME:0:24}"

if ! command -v az >/dev/null 2>&1; then
  echo "Error: Azure CLI (az) is not installed or not in PATH."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or not in PATH."
  exit 1
fi

echo "Checking Azure login/session..."
if ! az account show >/dev/null 2>&1; then
  echo "No active Azure session found. Starting login..."
  az login --use-device-code
fi

SUB_COUNT="$(az account list --query 'length([])' -o tsv)"
if [[ "${SUB_COUNT}" == "0" ]]; then
  echo "Error: No Azure subscriptions found for the current account."
  echo "Run: az account list -o table"
  echo "Then login with the correct account/tenant that has an active subscription."
  exit 1
fi

if [[ -n "${AZ_SUBSCRIPTION_ID:-}" ]]; then
  echo "Using AZ_SUBSCRIPTION_ID=${AZ_SUBSCRIPTION_ID}"
  az account set --subscription "${AZ_SUBSCRIPTION_ID}"
fi

echo "Using ACR name: ${ACR_NAME}"

echo "1) Azure session ready"

echo "1.1) Registering required Azure resource providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.ContainerInstance --wait
az provider register --namespace Microsoft.Network --wait

echo "2) Creating resource group..."
az group create --name "${RG}" --location "${LOCATION}"

echo "3) Creating Azure Container Registry..."
az acr create --resource-group "${RG}" --name "${ACR_NAME}" --sku Basic --admin-enabled true

echo "4) Building Docker image..."
docker build -t "${IMAGE_NAME}:latest" ./backend

ACR_LOGIN_SERVER="$(az acr show --name "${ACR_NAME}" --resource-group "${RG}" --query loginServer -o tsv)"

echo "5) Tagging image..."
docker tag "${IMAGE_NAME}:latest" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"

echo "6) Pushing image to ACR..."
az acr login --name "${ACR_NAME}"
docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest"

ACR_USERNAME="$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show --name "${ACR_NAME}" --query passwords[0].value -o tsv)"

echo "7) Creating Azure Container Instance..."
az container create \
  --resource-group "${RG}" \
  --name "${ACI_NAME}" \
  --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest" \
  --os-type Linux \
  --registry-login-server "${ACR_LOGIN_SERVER}" \
  --registry-username "${ACR_USERNAME}" \
  --registry-password "${ACR_PASSWORD}" \
  --dns-name-label "${ACI_NAME}-${SUFFIX}" \
  --ports 8000 \
  --ip-address Public \
  --cpu 1 \
  --memory 1.5

PUBLIC_URL="http://$(az container show --resource-group "${RG}" --name "${ACI_NAME}" --query ipAddress.fqdn -o tsv):8000"
echo "Deployment complete. API URL: ${PUBLIC_URL}"
