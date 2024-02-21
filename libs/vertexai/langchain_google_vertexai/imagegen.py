from typing import Any, Dict, List, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_pydantic_field_names
from vertexai.preview.vision_models import ImageGenerationModel


class VertexAIImageGenerator(BaseLLM):
    """ """

    model_name: str = Field(default="imagegeneration@005")
    """ Name of the model that will be used."""

    negative_prompt: Union[str, None] = Field(default=None)
    """A description of what you want to omit in the generated images"""

    number_of_images: int = Field(default=1)
    """Number of images to generate. Range: 1..8."""

    guidance_scale: Union[float, None] = Field(default=None)
    """ Controls the strength of the prompt.Suggested values are: 
        * 0-9 (low strength) 
        * 10-20 (medium strength) 
        * 21+ (high strength)
    """

    language: Union[str, None] = Field(default=None)
    """ upported values are "en" for English, "hi" for Hindi, "ja" for Japanese, "ko" 
    for Korean, and "auto" for automatic language detectio
    """

    seed: Union[int, None] = Field(default=None)
    """ Image generation random seed
    """

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Extra kwargs to add to `generate_images`. 
    """

    def _generate(
        self,
        prompts: List[str],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """

        Args:
            prompts:
            stop:
            run_manager:
        """

        model = ImageGenerationModel.from_pretrained(self.model_name)

        generations: List[List[Generation]] = []

        for query in prompts:
            candidates: List[Generation] = []

            model_response = model.generate_images(
                query,
                negative_prompt=self.negative_prompt,
                number_of_images=self.number_of_images,
                guidance_scale=self.guidance_scale,
                language=self.language,
                seed=self.seed,
                **self.model_kwargs,
            )

            for image in model_response.images:
                # Using a private method, we shouldn't
                generation = Generation(text=image._as_base64_string())
                candidates.append(generation)

            generations.append(candidates)

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vertexai-vision"

    @root_validator(pre=True)
    def _handle_extra_model_args(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Makes sure that there is no argument defined both in the fields and in the
        model kwargs. If the model arg is a class field, it should be put as a field.

        Args:
            values: Values passed to the class constructor.

        Returns:
            Postprocesses values.

        Raises:
            ValueError: If there are fields included in model_kwargs.
        """

        field_names = get_pydantic_field_names(cls)
        model_kwargs = values.get("model_kwargs", {})

        duplicated_kwargs = field_names.intersection(model_kwargs)

        if len(duplicated_kwargs) > 0:
            error_message = (
                f"Fields {','.join(duplicated_kwargs)} are defined in model_kwargs"
                f" but should be specified as class fields"
            )
            ValueError(error_message)

        return values
