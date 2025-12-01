# 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Repository Structure

If you plan on contributing to LangChain-Google code or documentation, it can be useful to understand the high level structure of the repository.

`langchain-google` is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── community
│   │   ├── tests/unit_tests # Unit tests (present in each package, not shown for brevity)
│   │   ├── tests/integration_tests # Integration tests (present in each package, not shown for brevity)
│   ├── genai
│   ├── vertexai
```

The root directory also contains the following files:

- `pyproject.toml`: Dependencies for building and linting the docs and cookbook.
- `Makefile`: A file that contains shortcuts for building and linting the docs and cookbook.

There are other files in the root directory level, but their presence should be self-explanatory.

## Local Development Dependencies

Install development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
uv sync --group lint --group typing --group test --group test_integration
```

Then verify dependency installation:

```bash
make test
```

## Formatting and Linting

### Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:

```bash
make format_diff
```

This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:

```bash
make lint
```

To run linting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

## Working with Optional Dependencies

`community`, `genai`, and `vertexai` rely on optional dependencies to keep these packages lightweight.

You'll notice that `pyproject.toml` and `uv.lock` are **not** touched when you add optional dependencies below.

If you're adding a new dependency to Langchain-Google, assume that it will be an optional dependency, and that most users won't have it installed.

Users who do not have the dependency installed should be able to **import** your code without any side effects (no warnings, no errors, no exceptions).

To introduce the dependency to a library, please do the following:

1. Open `extended_testing_deps.txt` and add the dependency
2. Add a unit test that the very least attempts to import the new code. Ideally, the unit test makes use of lightweight fixtures to test the logic of the code.
3. Please use the `@pytest.mark.requires(package_name)` decorator for any unit tests that require the dependency.

## Testing

All of our packages have unit tests and integration tests, and we favor unit tests over integration tests.

Unit tests run on every pull request, so they should be fast and reliable.

Integration tests run once a day, and they require more setup, so they should be reserved for confirming interface points with external services.

### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs.

If you add new logic, please add a unit test.

In unit tests we check pre/post processing and mocking all external dependencies.

To install dependencies for unit tests:

```bash
uv sync --group test
```

To run unit tests:

```bash
make test
```

To run unit tests in Docker:

```bash
make docker_tests
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

### Integration Tests

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).

If you add support for a new external API, please add a new integration test.

**Warning:** Almost no tests should be integration tests.

  Tests that require making network connections make it difficult for other developers to test the code.

  Instead favor relying on `responses` library and/or `mock.patch` to mock requests using small fixtures.

To install dependencies for integration tests:

```bash
uv sync --group test --group test_integration
```

To run integration tests:

```bash
make integration_tests
```

#### Annotating integration tests

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

### Coverage

Code coverage (i.e. the amount of code that is covered by unit tests) helps identify areas of the code that are potentially more or less brittle.

Coverage requires the dependencies for integration tests:

```bash
uv sync --group test_integration
```

To get a report of current coverage, run the following:

```bash
make coverage
```
