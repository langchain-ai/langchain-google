from langchain_google_vertexai.model_garden_maas._base import (
    _LLAMA_MODELS,
    _MISTRAL_MODELS,
)
from langchain_google_vertexai.model_garden_maas.llama import VertexModelGardenLlama

_MAAS_MODELS = _MISTRAL_MODELS + _LLAMA_MODELS


def get_vertex_maas_model(model_name, **kwargs):
    """Return a corresponding Vertex MaaS instance.

    A factory method based on model's name.
    """
    if model_name not in _MAAS_MODELS:
        raise ValueError(f"model name {model_name} is not supported!")
    if model_name in _MISTRAL_MODELS:
        from langchain_google_vertexai.model_garden_maas.mistral import (  # noqa: F401
            VertexModelGardenMistral,
        )

        return VertexModelGardenMistral(model=model_name, **kwargs)
    return VertexModelGardenLlama(model=model_name, **kwargs)
