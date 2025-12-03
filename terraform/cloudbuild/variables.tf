variable "project_id" {
  type        = string
  description = ""
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = ""
}

variable "prefix" {
  type        = string
  default     = "langchain-google"
  description = ""
}

variable "cloudbuildv2_repository_id" {
  type        = string
  description = ""
}

variable "python_version" {
  type        = string
  default     = "3.11"
  description = ""
}

variable "library" {
  type        = string
  description = ""
}

variable "cloudbuild_env_vars" {
  type = map(string)
}

variable "cloudbuild_secret_vars" {
  type = map(string)
}
