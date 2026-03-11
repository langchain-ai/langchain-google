import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from langchain_google_vertexai.vision_models import (
    VertexAIImageCaptioning,
    VertexAIImageCaptioningChat,
    VertexAIImageEditorChat,
    VertexAIImageGeneratorChat,
    VertexAIVisualQnAChat,
)


@pytest.mark.skip(
    reason=(
        "Image captioning is deprecated: "
        "https://cloud.google.com/vertex-ai/generative-ai/docs/image/image-captioning"
    )
)
def test_vertex_ai_image_captioning_chat(base64_image: str) -> None:
    # This should work
    model = VertexAIImageCaptioningChat()
    response = model.invoke(
        input=[
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": base64_image}}]
            ),
        ]
    )

    assert isinstance(response, AIMessage)

    # Content should be an image
    with pytest.raises(ValueError):
        model = VertexAIImageCaptioningChat()
        response = model.invoke(
            input=[
                HumanMessage(content="Text message"),
            ]
        )

    # Not more than one message allowed
    with pytest.raises(ValueError):
        model = VertexAIImageCaptioningChat()
        response = model.invoke(
            input=[
                HumanMessage(content=base64_image),
                HumanMessage(content="Follow up"),
            ]
        )


@pytest.mark.skip(
    reason=(
        "Image captioning is deprecated: "
        "https://cloud.google.com/vertex-ai/generative-ai/docs/image/image-captioning"
    )
)
def test_vertex_ai_image_captioning(base64_image: str) -> None:
    model = VertexAIImageCaptioning()
    response = model.invoke(base64_image)
    assert isinstance(response, str)

    response = model.invoke(base64_image, language="de")
    assert isinstance(response, str)


@pytest.mark.skip(
    reason=(
        "Visual question answering is deprecated: "
        "https://cloud.google.com/vertex-ai/generative-ai/docs/image/image-captioning"
    )
)
def test_vertex_ai_visual_qna_chat(base64_image: str) -> None:
    model = VertexAIVisualQnAChat()

    # This should work
    response = model.invoke(
        input=[
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    "What color is the image?",
                ]
            )
        ]
    )
    assert isinstance(response, AIMessage)

    response = model.invoke(
        input=[
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": "What color is the image?"},
                ]
            )
        ]
    )
    assert isinstance(response, AIMessage)

    # This should not work, the image must be first

    with pytest.raises(ValueError):
        response = model.invoke(
            input=[
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What color is the image?"},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ]
                )
            ]
        )

    # This should not work, only one message with multiparts allowed
    with pytest.raises(ValueError):
        response = model.invoke(
            input=[
                HumanMessage(content=base64_image),
                HumanMessage(content="What color is the image?"),
            ]
        )

    # This should not work, only one message with multiparts allowed
    with pytest.raises(ValueError):
        response = model.invoke(
            input=[
                HumanMessage(content=base64_image),
                HumanMessage(content="What color is the image?"),
                AIMessage(content="yellow"),
                HumanMessage(content="And the eyes?"),
            ]
        )


@pytest.mark.release
@pytest.mark.xfail(reason="Started raising 500 error on 2026-01-26.")
def test_vertex_ai_image_generation_and_edition() -> None:
    generator = VertexAIImageGeneratorChat()

    messages = [HumanMessage(content=["Generate a dog reading the newspaper"])]
    response = generator.invoke(messages)
    assert isinstance(response, AIMessage)

    generated_image = response.content[0]

    model = VertexAIImageGeneratorChat()

    prompt = PromptTemplate(
        template="I want an image of {img_object} in {img_context}.",
        input_variables=["img_object", "img_context"],
    )

    chain = prompt | model

    response = chain.invoke({"img_object": "cat", "img_context": "beach"})
    assert isinstance(response, AIMessage)

    editor = VertexAIImageEditorChat()

    messages = [HumanMessage(content=[generated_image, "Change the dog for a cat"])]

    response = editor.invoke(messages)
    assert isinstance(response, AIMessage)
