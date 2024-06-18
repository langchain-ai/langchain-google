locals {
  google_apis = [
    "vision.googleapis.com",
    "discoveryengine.googleapis.com",
    "customsearch.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
  ]
}

resource "google_project_service" "google_apis" {
  for_each                   = toset(local.google_apis)
  service                    = each.value
  disable_dependent_services = true
}

# resource "google_discovery_engine_data_store" "data_store" {
#   project= ""
#   location                    = "global"
#   data_store_id               = "data-store"
#   display_name                = "data-store"
#   industry_vertical           = "GENERIC"
#   content_config              = "NO_CONTENT"
# }

# Quota increased:
# Vertex AI API	Online prediction requests per base model per minute per region per base_model	Quota	region : asia-northeast1 base_model : textembedding-gecko
# 100->500

# terraform
# cloud build trigger
# cloud build sa

# enable api
#     vision.googleapis.com
#     discoveryengine.googleapis.com
#     customsearch.googleapis.com
#     cloudresourcemanager.googleapis.com
#     bigquery.googleapis.com
#     cloudbuild.googleapis.com
#     secretmanager.googleapis.com

# secrets 
#     api key
#     cse id
# 3 models in model garden:
#     FALCON_ENDPOINT_ID
#     falcon-instruct-40b-peft-001-mg-one-click-deploy
#     LLAMA_ENDPOINT_ID
#     llama2-70b-chat-001-mg-one-click-deploy
#     GEMMA_ENDPOINT_ID
#     google_gemma-7b-it-mg-one-click-deploy

# api key:
# restrictions:
#     generative api
#     search api





