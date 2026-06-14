import re

def update_pyproject():
    paths = [
        "libs/community/pyproject.toml",
        "libs/genai/pyproject.toml",
        "libs/vertexai/pyproject.toml"
    ]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add pytest-xdist to test_integration if it's not there
        if "test_integration = [" in content:
            if "pytest-xdist" not in content.split("test_integration = [")[1].split("]")[0]:
                content = content.replace("test_integration = [", "test_integration = [\n    \"pytest-xdist>=3.8.0,<4.0.0\",")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def update_makefile():
    paths = [
        "libs/community/Makefile",
        "libs/genai/Makefile",
        "libs/vertexai/Makefile"
    ]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Update integration_tests target
        content = re.sub(
            r"(integration_tests:\s+uv run pytest) (tests/integration_tests)",
            r"\1 -n auto \2",
            content
        )
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    update_pyproject()
    update_makefile()
