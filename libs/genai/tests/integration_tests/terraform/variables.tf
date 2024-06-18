variable "project_id" {
  type        = string
  description = ""
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = ""
}

variable "poetry_version" {
  type        = string
  default     = "1.7.1"
  description = ""
}

variable "python_version" {
  type        = string
  default     = "3.11"
  description = ""
}

variable "prefix" {
  type        = string
  default     = "langchain-google"
  description = ""
}

variable "langchain_github_repo" {
  type        = string
  description = ""
}

variable "library" {
  type        = string
  description = ""
}

variable "google_api_key" {
  type        = string
  description = ""
}

variable "google_cse_id" {
  type        = string
  description = ""
}

variable "github_oauth_token" {
  type        = string
  description = ""
}

#add todo for installing the app
variable "github_app_installation_id" {
  type        = string
  description = "Your installation ID can be found in the URL of your Cloud Build GitHub App. In the following URL, https://github.com/settings/installations/1234567, the installation ID is the numerical value 1234567."
}

variable "cloudbuild_env_vars" {
  type = map(string)
}

variable "cloudbuild_secret_vars" {
  type = list(string)
}