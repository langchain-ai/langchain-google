module "cloudbuild" {
  source = "./../../../../../terraform/cloudbuild"
  
  library                    = "genai"
  project_id                 = ""
  cloudbuildv2_repository_id = ""
  cloudbuild_env_vars = {
  }
  cloudbuild_secret_vars = {
    GOOGLE_API_KEY = ""
  }
}