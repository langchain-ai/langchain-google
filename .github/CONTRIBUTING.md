# 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview). This document outlines additional details specific to the `langchain-google` repository.

## Repository structure

If you plan on contributing to `langchain-google` code or documentation, it can be useful to understand the high level structure of the repository.

`langchain-google` is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── genai
│   │   ├── tests/unit_tests # Unit tests (present in each package, not shown for brevity)
│   │   ├── tests/integration_tests # Integration tests (present in each package, not shown for brevity)
│   ├── vertexai
│   ├── community
```

The root directory also contains the following files:

- `pyproject.toml`: Dependencies for building and linting the docs and cookbook.
- `Makefile`: A file that contains shortcuts for building and linting the docs and cookbook.

There are other files in the root directory level, but their presence should be self-explanatory.

## Integration tests

### Backend selection

For `libs/genai`, integration tests can run against different backends using the `TEST_VERTEXAI` environment variable:

- **Google AI only (default)**: `make integration_tests`
- **Both Google AI and Vertex AI**: `TEST_VERTEXAI=1 make integration_tests`
- **Vertex AI only**: `TEST_VERTEXAI=only make integration_tests`

Vertex AI tests require the `GOOGLE_CLOUD_PROJECT` environment variable to be set. Tests will automatically skip if not configured.

### Annotating integration tests

We annotate integration tests to separate those tests which heavily rely on GCP infrastructure. Especially for running those tests we have created a separate GCP project with all necessary infrastructure parts provisioned. To run the extended integration tests locally you will need to provision a GCP project and pass its configuration via env variables.

Test annotations:

1. Tests without annotations will be executed on every run of the integration tests pipeline.
2. Tests with release annotation ( `@pytest.mark.release` ) will be run with the release pipeline.
3. Tests with extended annotation ( `@pytest.mark.extended` ) will be run on each PR.

### Prepare

The integration tests use several search engines and databases. The tests aim to verify the correct behavior of the engines and databases according to their specifications and requirements.

To run some integration tests, you will need GCP project configured.

The configuration of the GCP project required for integration testing is stored in the terraform folder within each library.

### Prepare environment variables for local testing

- Copy `tests/integration_tests/.env.example` to `tests/integration_tests/.env`
- Set variables in `tests/integration_tests/.env` file, e.g `GOOGLE_API_KEY`

Additionally, it's important to note that some integration tests may require certain environment variables to be set, such as `PROJECT_ID`. Be sure to set any required environment variables before running the tests to ensure they run correctly.

### Run some tests with coverage

```bash
pytest tests/integration_tests/.py --cov=langchain --cov-report=html
start "" htmlcov/index.html || open htmlcov/index.html

```

## Releases

Releases are automated using [release-please](https://github.com/googleapis/release-please). When commits land on `main`, release-please analyzes them and creates/updates release PRs with changelog entries and version bumps.

### How it works

1. **Make commits using [Conventional Commits](https://www.conventionalcommits.org/) format** - this determines version bumps:
   - `fix(genai): ...` → patch bump (4.2.0 → 4.2.1)
   - `feat(genai): ...` → minor bump (4.2.0 → 4.3.0)
   - `feat(genai)!: ...` or `fix(genai)!: ...` → major bump (4.2.0 → 5.0.0)

2. **Release-please creates a PR** for each package with pending changes, titled `release(genai): X.Y.Z`

3. **Merge the release PR** → triggers the release workflow which runs tests, publishes to PyPI, and creates a GitHub Release

### What if a release fails?

If the release workflow fails after a release PR is merged (e.g., tests fail, PyPI publish fails):

1. **The GitHub Release is NOT created** - it only happens after successful PyPI publish
2. **Fix the issue** in a follow-up commit
3. **Manually retry the release** by triggering `_release.yml` via `workflow_dispatch`:
   - Go to Actions → "🚀 Package Release" → Run workflow
   - Select the package directory (e.g., `libs/genai`)
   - Enter the version from `pyproject.toml`

**Why manual retry?** The version was already bumped in `pyproject.toml` when the release PR merged. If you push a fix and let release-please create a new PR, it will bump to the *next* version (e.g., 4.3.0 → 4.3.1), and 4.3.0 will be skipped. Manual retry preserves the intended version.

### Manual releases

For hotfixes or bypassing release-please, you can still manually trigger `_release.yml` via `workflow_dispatch`.
