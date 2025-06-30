provider "google" {
  project = var.project_id
}


locals {
  cloudbuild_service_account_roles = [
    "roles/bigquery.user",                     #BigQuery User
    "roles/discoveryengine.editor",            #Discovery Engine Editor
    "roles/logging.logWriter",                 #Logs Writer
    "roles/secretmanager.secretAccessor",      #Secret Manager Secret Accessor
    "roles/serviceusage.serviceUsageConsumer", #Service Usage Consumer
    "roles/aiplatform.user",                   #Vertex AI User
    "roles/modelarmor.admin"                   #Model Armor Admin
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
  #TODO: multiline
  cloudbuild_config = "python -m pip install -q poetry==$${_POETRY_VERSION} --verbose && cd libs/$${_LIB} && poetry install -q --with test,test_integration  --all-extras && poetry run pytest --extended --release tests/integration_tests/"
}

resource "google_project_service" "model_armor_service" {
  project = var.project_id
  service = "modelarmor.googleapis.com"

  disable_on_destroy = false
}

resource "google_cloudbuild_trigger" "cloudbuild_trigger" {
  name            = "${var.prefix}-${var.library}"
  location        = var.region
  service_account = google_service_account.cloudbuild_service_account.id

  included_files = ["libs/${var.library}/**"]

  repository_event_config {
    repository = var.cloudbuildv2_repository_id
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
      secret_env = keys(var.cloudbuild_secret_vars)
    }

    options {
      logging = "CLOUD_LOGGING_ONLY"
    }

    available_secrets {
      dynamic "secret_manager" {
        for_each = var.cloudbuild_secret_vars
        content {
          env          = secret_manager.key
          version_name = "projects/$${PROJECT_ID}/secrets/${secret_manager.value}/versions/latest"
        }
      }
    }
  }
}

resource "google_service_account" "cloudbuild_service_account" {
  account_id = "${var.library}-cb-sa"
}

resource "google_project_iam_member" "cloudbuild_service_account_iam" {
  project  = var.project_id
  for_each = toset(local.cloudbuild_service_account_roles)
  role     = each.value
  member   = "serviceAccount:${google_service_account.cloudbuild_service_account.email}"
}
