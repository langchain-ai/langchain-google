locals {
  cloudbuild_service_account_roles = [
    "roles/bigquery.user",                     #BigQuery User
    "roles/discoveryengine.editor",            #Discovery Engine Editor
    "roles/logging.logWriter",                 #Logs Writer
    "roles/secretmanager.secretAccessor",      #Secret Manager Secret Accessor
    "roles/serviceusage.serviceUsageConsumer", #Service Usage Consumer
    "roles/aiplatform.user",                   #Vertex AI User
  ]
  cloudbuild_env_vars = merge(
    {
      for key, value in var.cloudbuild_env_vars :
      "_${upper(key)}" => value
    },
    { _LIB = var.library },
    { _POETRY_VERSION = var.poetry_version },
    { _PYTHON_VERSION = var.python_version },
  )
  cloudbuild_config = "python -m pip install -q poetry==$${_POETRY_VERSION} --verbose && cd libs/$${_LIB} && poetry install -q --with test,test_integration && poetry run pytest --extended --release tests/integration_tests/"
}

resource "google_cloudbuildv2_connection" "langchain_google_github_connection" {
  location = var.region
  name     = "${var.prefix}-connection"

  github_config {
    app_installation_id = var.github_app_installation_id
    authorizer_credential {
      oauth_token_secret_version = google_secret_manager_secret_version.github_oauth_token_secret_version.id
    }
  }
}

resource "google_cloudbuildv2_repository" "langchain_google_repository" {
  name              = "${var.prefix}-repository"
  parent_connection = google_cloudbuildv2_connection.langchain_google_github_connection.id
  remote_uri        = var.langchain_github_repo
}

resource "google_cloudbuild_trigger" "cloudbuild_trigger" {
  name            = "${var.prefix}-${var.library}"
  location        = var.region
  service_account = google_service_account.cloudbuild_service_account.id

  included_files = ["libs/${var.library}/**"]

  repository_event_config {
    repository = google_cloudbuildv2_repository.langchain_google_repository.id
    pull_request {
      branch = "^main$"
      #comment_control = "COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY" #Configure builds to run whether a repository owner or collaborator need to comment /gcbrun
    }
  }

  substitutions = local.cloudbuild_env_vars

  include_build_logs = "INCLUDE_BUILD_LOGS_WITH_STATUS"

  build {
    step {
      id         = "integration tests"
      name       = "python:$${_PYTHON_VERSION}"
      args       = ["-c", local.cloudbuild_config]
      entrypoint = "bash"
      env        = concat(["PROJECT_ID=$PROJECT_ID"], [for env_name, env_value in var.cloudbuild_env_vars : "${env_name}=$_${env_name}"])
      secret_env = var.cloudbuild_secret_vars
    }

    options {
      logging = "CLOUD_LOGGING_ONLY"
    }

    available_secrets {
      dynamic "secret_manager" {
        for_each = var.cloudbuild_secret_vars
        content {
          env          = secret_manager.value
          version_name = "projects/$${PROJECT_ID}/secrets/${var.prefix}-${lower(replace(secret_manager.value, "_", "-"))}/versions/latest"
        }
      }
    }
  }
}

resource "google_service_account" "cloudbuild_service_account" {
  account_id = "${var.prefix}-cloudbuild-sa"
}

resource "google_project_iam_member" "cloudbuild_service_account_iam" {
  project  = var.project_id
  for_each = toset(local.cloudbuild_service_account_roles)
  role     = each.value
  member   = "serviceAccount:${google_service_account.cloudbuild_service_account.email}"
}

# │ Error: Error waiting to create Connection: Error waiting for Creating Connection: Error code 9, message: could not access secret "projects/546814358370/secrets/langchain-google-github-oauth-token/versions/1" with service account "service-546814358370@gcp-sa-cloudbuild.iam.gserviceaccount.com": generic::permission_denied: Permission 'secretmanager.versions.access' denied for resource 'projects/546814358370/secrets/langchain-google-github-oauth-token/versions/1' (or it may not exist).

# │ Error: Error creating Repository: googleapi: Error 400: connection must have installation_state COMPLETE (current state: PENDING_INSTALL_APP)