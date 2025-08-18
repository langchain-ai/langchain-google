# Model Armor Runnables for LangChain

This package provides LangChain-compatible Runnables for user prompt and model response sanitization using [Google Cloud Model Armor](https://cloud.google.com/security-command-center/docs/model-armor-overview).

## Prerequisites

Before using Model Armor Runnables, ensure the following steps are completed:

- **Select or create a Google Cloud Platform project.**
  - You can do this at: <https://console.cloud.google.com/project>
- **Enable billing for your project.**
  - Instructions: <https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project>
- **Enable the Model Armor API in your GCP project.**
  - See: <https://cloud.google.com/security-command-center/docs/get-started-model-armor>
- **Grant the `modelarmor.user` IAM role** to any user or service account that will use the Model Armor runnables.
- **Authentication:**
  - Your application or environment must be authenticated and have the necessary permissions to access the Model Armor service.
  - You can authenticate using several methods described [here](https://googleapis.dev/python/google-api-core/latest/auth.html)
- **Model Armor Template:**
  - You must create a Model Armor template for prompt and response sanitization. You may use a single template for both, or separate templates as needed.
  - Refer to the guide: [Create and manage Model Armor templates](https://cloud.google.com/security-command-center/docs/get-started-model-armor)
  - The template resource name must be provided when initializing the runnable.
  - To manage Model Armor templates, the `modelarmor.admin` IAM role is required.

## License

See [LICENSE](../../LICENSE) file.
