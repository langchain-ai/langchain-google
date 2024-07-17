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
    VECTOR_SEARCH_STAGING_BUCKET="",
    VECTOR_SEARCH_STREAM_INDEX_ID="",
    VECTOR_SEARCH_STREAM_ENDPOINT_ID="",
    VECTOR_SEARCH_BATCH_INDEX_ID="",
    VECTOR_SEARCH_BATCH_ENDPOINT_ID="",
  }
  cloudbuild_secret_vars = {
    GOOGLE_API_KEY = ""
  }
}