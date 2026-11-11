"""Test `GoogleGenAIClassifier`."""

import base64
import io

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_genai.classifier import (
    Choice,
    GoogleGenAIClassifier,
    Noul,
    Score,
)

_MODEL = "gemini-decision-1"

_QUESTIONS: dict[str, Noul | Choice | Score] = {
    "damaged": Noul(
        instructions="Did the item arrive damaged?",
        criteria={"true": "The item is broken, cracked, or dented."},
    ),
    "department": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Payment or refund questions.",
            "shipping": "Delivery problems or damaged items.",
            "other": None,
        },
    ),
    "frustration": Score(
        instructions="How frustrated does the customer appear?",
        criteria=["Calm.", "Concerned but civil.", "Very angry."],
    ),
}


def test_invoke() -> None:
    classifier = GoogleGenAIClassifier(model=_MODEL)

    response = classifier.invoke(
        {
            "state": "The blue mug I ordered arrived cracked. Fix this today!",
            "questions": _QUESTIONS,
        }
    )

    assert response.model == _MODEL
    assert response.nouls["damaged"].noul > 0.5
    assert response.choices["department"].choice == "shipping"
    assert set(response.choices["department"].probabilities) == {
        "billing",
        "shipping",
        "other",
    }
    assert set(response.scores["frustration"].legend) == {0, 1, 2}
    assert response.usage is not None
    assert response.usage["input_tokens"]
    assert response.usage["output_tokens"] is not None


async def test_ainvoke_with_conversation() -> None:
    classifier = GoogleGenAIClassifier(model=_MODEL)

    response = await classifier.ainvoke(
        {
            "state": {
                "conversation": [
                    HumanMessage("Where is my order?"),
                    AIMessage("It was delivered yesterday."),
                    HumanMessage("It arrived smashed to pieces."),
                ]
            },
            "questions": {"damaged": _QUESTIONS["damaged"]},
        }
    )

    assert response.nouls["damaged"].noul > 0.5


def test_invoke_with_image() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode()
    classifier = GoogleGenAIClassifier(model=_MODEL)

    response = classifier.invoke(
        {
            "state": {
                "caption": "Product photo",
                "photo": {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/png",
                },
            },
            "questions": {
                "red": Noul(instructions="Is the photo a solid red image?"),
            },
        }
    )

    assert response.nouls["red"].noul > 0.5
