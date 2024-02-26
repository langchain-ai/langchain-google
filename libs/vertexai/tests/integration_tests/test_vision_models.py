import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_vertexai.vision_models import (
    VertexAIImageCaptioning,
    VertexAIImageCaptioningChat,
    VertexAIImageEditorChat,
    VertexAIImageGeneratorChat,
    VertexAIVisualQnAChat,
)


def test_vertex_ai_image_captioning_chat(base64_image: str):
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


def test_vertex_ai_image_captioning(base64_image: str):
    model = VertexAIImageCaptioning()
    response = model.invoke(base64_image)
    assert isinstance(response, str)


def test_vertex_ai_visual_qna_chat(base64_image: str):
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


def test_vertex_ai_image_generation_and_edition():
    generator = VertexAIImageGeneratorChat()

    messages = [HumanMessage(content=["Generate a dog reading the newspaper"])]
    response = generator.invoke(messages)
    assert isinstance(response, AIMessage)

    generated_image = response.content[0]

    editor = VertexAIImageEditorChat()

    messages = [HumanMessage(content=[generated_image, "Change the dog for a cat"])]

    response = editor.invoke(messages)
    assert isinstance(response, AIMessage)


@pytest.fixture
def base64_image() -> str:
    return (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAA"
        "BHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3"
        "d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBap"
        "ySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnx"
        "BwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXr"
        "CDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD"
        "1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQD"
        "ry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPs"
        "gxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96Cu"
        "tRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOM"
        "OVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWqua"
        "ZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYS"
        "Ub3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6E"
        "hOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oW"
        "VeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmH"
        "rwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz"
        "8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66Pf"
        "yuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UN"
        "z8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="
    )
