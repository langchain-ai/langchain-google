module "cloudbuild" {
  source = "./../../../../../terraform/cloudbuild"
  
  library                    = "community"
  project_id                 = ""
  cloudbuildv2_repository_id = ""
  cloudbuild_env_vars = {
    _DATA_STORE_ID = ""
    _IMAGE_GCS_PATH = "gs://cloud-samples-data/vision/label/wakeupcat.jpg"
  }
  cloudbuild_secret_vars = {
    GOOGLE_API_KEY = ""
    GOOGLE_CSE_ID = ""
  }
}