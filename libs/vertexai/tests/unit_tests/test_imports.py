import subprocess
import sys
from importlib.metadata import requires

from packaging.requirements import Requirement

from langchain_google_vertexai import __all__

EXPECTED_ALL = [
    "ChatVertexAI",
    "create_structured_runnable",
    "DataStoreDocumentStorage",
    "FunctionCallingConfig",
    "FunctionDeclaration",
    "GCSDocumentStorage",
    "HarmBlockThreshold",
    "HarmCategory",
    "Modality",
    "PydanticFunctionsOutputParser",
    "SafetySetting",
    "Schema",
    "ToolConfig",
    "Type",
    "VectorSearchVectorStore",
    "VectorSearchVectorStoreDatastore",
    "VectorSearchVectorStoreGCS",
    "VertexAI",
    "VertexAIEmbeddings",
    "VertexAIImageCaptioning",
    "VertexAIImageCaptioningChat",
    "VertexAIImageEditorChat",
    "VertexAIImageGeneratorChat",
    "VertexAIModelGarden",
    "VertexAIVisualQnAChat",
    "VertexPairWiseStringEvaluator",
    "VertexStringEvaluator",
    "__version__",
    "create_context_cache",
    "get_vertex_maas_model",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_pyarrow_not_required() -> None:
    declared = {
        Requirement(r).name for r in requires("langchain-google-vertexai") or []
    }
    assert "pyarrow" not in declared

    code = (
        "import sys, langchain_google_vertexai\n"
        "from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings\n"
        "assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
