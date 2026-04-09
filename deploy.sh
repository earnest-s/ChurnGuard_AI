#!/usr/bin/env bash
set -euo pipefail

RG="churn-rg"
LOCATION="centralindia"
IMAGE_NAME="churn-api"
ACI_NAME="churn-api"
SUFFIX="$(date +%s)$RANDOM"
ACR_NAME="churnacr${SUFFIX}"
ACR_NAME="${ACR_NAME:0:24}"

echo "Using ACR name: ${ACR_NAME}"

echo "1) Logging into Azure..."
az login

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
