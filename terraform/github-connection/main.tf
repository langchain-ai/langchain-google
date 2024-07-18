provider "google" {
  project = var.project_id
}

resource "google_cloudbuildv2_connection" "langchain_google_github_connection" {
  location = var.region
  name     = "${var.prefix}-connection"

  github_config {
    app_installation_id = var.github_app_installation_id
    authorizer_credential {
      oauth_token_secret_version = "${var.github_oauth_token_secret_id}/versions/latest"
    }
  }
}

resource "google_cloudbuildv2_repository" "langchain_google_repository" {
  name              = "${var.prefix}-repository"
  parent_connection = google_cloudbuildv2_connection.langchain_google_github_connection.id
  remote_uri        = var.langchain_github_repo
}

output "langchain_google_repository_id" {
  value = google_cloudbuildv2_repository.langchain_google_repository.id
}