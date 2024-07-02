module "cloudbuild" {
  source = "./../../../../../terraform/cloudbuild"
  
  library                    = "vertexai"
  project_id                 = ""
  cloudbuildv2_repository_id = ""
  cloudbuild_env_vars = {
    FALCON_ENDPOINT_ID = "",
    GEMMA_ENDPOINT_ID  = "",
    LLAMA_ENDPOINT_ID  = "",
    IMAGE_GCS_PATH     = "",
  }
  cloudbuild_secret_vars = {
    GOOGLE_API_KEY = ""
  }
}