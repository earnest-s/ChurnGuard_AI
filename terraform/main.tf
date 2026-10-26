# ChurnGuard AI - Azure infrastructure (Resource Group + Container Registry)
#
# Usage:
#   cd terraform
#   terraform init
#   terraform plan -out tfplan
#   terraform apply tfplan
#
# The ACR name must be globally unique across Azure. Override it with:
#   terraform apply -var="acr_name=churnguardacr123"
#
# The GitHub Actions OIDC identity needs Contributor on the subscription (or
# on this resource group) to let the workflow push images and create ACI.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "project" {
  description = "Name prefix for ChurnGuard resources."
  type        = string
  default     = "churnguard"
}

variable "location" {
  description = "Azure region for the resource group and registry."
  type        = string
  default     = "centralindia"
}

variable "acr_name" {
  description = "Globally unique Container Registry name (5-50 lowercase alphanumerics)."
  type        = string
  default     = "churnguardacr"
}

locals {
  tags = {
    project    = var.project
    managed_by = "terraform"
    workload   = "churn-prediction"
  }
}

resource "azurerm_resource_group" "main" {
  name     = "${var.project}-rg"
  location = var.location
  tags     = local.tags
}

resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

output "resource_group_name" {
  description = "Resource group that holds the registry."
  value       = azurerm_resource_group.main.name
}

output "acr_name" {
  description = "Registry name to pass to `docker login` / `az acr login`."
  value       = azurerm_container_registry.main.name
}

output "acr_login_server" {
  description = "Registry login server used to tag and push the API image."
  value       = azurerm_container_registry.main.login_server
}
