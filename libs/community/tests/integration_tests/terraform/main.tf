module "cloudbuild" {
  source = "./../../../../../terraform/cloudbuild"
  
  library                    = "community"
  project_id                 = ""
  cloudbuildv2_repository_id = ""
  cloudbuild_env_vars = {
    DATA_STORE_ID = "",
    IMAGE_GCS_PATH = "gs://cloud-samples-data/vision/label/wakeupcat.jpg",
    PROCESSOR_NAME = ""
  }
  cloudbuild_secret_vars = {
    GOOGLE_API_KEY = ""
    GOOGLE_CSE_ID = ""
  }
}