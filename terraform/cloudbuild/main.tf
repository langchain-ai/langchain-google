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
    { _PYTHON_VERSION = var.python_version },
  )
  cloudbuild_config = <<-EOT
    set -euo pipefail

    python -m pip install -q uv --verbose
    cd libs/$${_LIB}
    uv sync --group test --group test_integration --all-extras

    common_pytest_args="--retries 3 --retry-delay 1 --maxfail=1 --timeout=120 --timeout-method=thread --extended --release"

    case "$${_LIB}" in
      vertexai)
        uv run pytest $common_pytest_args \
          tests/integration_tests/test_chat_models.py::test_vertexai_single_call \
          tests/integration_tests/test_chat_models.py::test_vertexai_stream \
          tests/integration_tests/test_chat_models.py::test_chat_vertexai_gemini_function_calling

        uv run pytest $common_pytest_args \
          tests/integration_tests/test_anthropic_cache.py \
          tests/integration_tests/test_anthropic_files.py \
          tests/integration_tests/test_anthropic_long_context.py \
          tests/integration_tests/test_maas.py \
          tests/integration_tests/test_model_garden.py

        uv run pytest $common_pytest_args \
          tests/integration_tests/test_callbacks.py \
          tests/integration_tests/test_chains.py \
          tests/integration_tests/test_chat_models.py \
          tests/integration_tests/test_llms.py \
          tests/integration_tests/test_llms_safety.py \
          tests/integration_tests/test_standard.py

        uv run pytest $common_pytest_args tests/integration_tests/ \
          --ignore=tests/integration_tests/test_anthropic_cache.py \
          --ignore=tests/integration_tests/test_anthropic_files.py \
          --ignore=tests/integration_tests/test_anthropic_long_context.py \
          --ignore=tests/integration_tests/test_callbacks.py \
          --ignore=tests/integration_tests/test_chains.py \
          --ignore=tests/integration_tests/test_chat_models.py \
          --ignore=tests/integration_tests/test_llms.py \
          --ignore=tests/integration_tests/test_llms_safety.py \
          --ignore=tests/integration_tests/test_maas.py \
          --ignore=tests/integration_tests/test_model_garden.py \
          --ignore=tests/integration_tests/test_standard.py
        ;;
      genai)
        uv run pytest $common_pytest_args \
          tests/integration_tests/test_chat_models.py::test_chat_google_genai_invoke \
          tests/integration_tests/test_chat_models.py::test_basic_streaming \
          tests/integration_tests/test_chat_models.py::test_chat_google_genai_with_structured_output

        uv run pytest $common_pytest_args tests/integration_tests/
        ;;
      *)
        uv run pytest $common_pytest_args tests/integration_tests/
        ;;
    esac
  EOT
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
