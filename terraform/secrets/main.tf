provider "google" {
  project = var.project_id
}

resource "google_secret_manager_secret" "google_api_key_secret" {
  secret_id = "${var.prefix}-google-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "google_api_key_secret_version" {
  secret      = google_secret_manager_secret.google_api_key_secret.id
  secret_data = var.google_api_key
}

output "google_api_key_secret_id" {
  value = google_secret_manager_secret.google_api_key_secret.id
}

resource "google_secret_manager_secret" "google_cse_id_secret" {
  secret_id = "${var.prefix}-google-cse-id"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "google_cse_id_secret_version" {
  secret      = google_secret_manager_secret.google_cse_id_secret.id
  secret_data = var.google_cse_id
}

output "google_cse_id_secret_id" {
  value = google_secret_manager_secret.google_cse_id_secret.id
}

resource "google_secret_manager_secret" "github_oauth_token_secret" {
  secret_id = "${var.prefix}-github-oauth-token"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "github_oauth_token_secret_version" {
  secret      = google_secret_manager_secret.github_oauth_token_secret.id
  secret_data = var.github_oauth_token
}

output "github_oauth_token_secret_id" {
  value = google_secret_manager_secret.github_oauth_token_secret.id
}

#autocreate api key with scope restrictions
#add link to google_cse_id creation