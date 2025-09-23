import warnings

import pytest
from vertexai.vision_models import (
    Image,  # TODO: migrate to google-genai since this is deprecated
)

from langchain_google_vertexai.vision_models import _BaseImageTextModel


def test_get_image_from_message_part(base64_image: str) -> None:
    # TODO: Remove this warning suppression when migrating to google-genai
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*deprecated.*", category=UserWarning
        )
        model = _BaseImageTextModel()

        # Should work with a well formatted dictionary:
        message = {"type": "image_url", "image_url": {"url": base64_image}}
        image = model._get_image_from_message_part(message)
        assert isinstance(image, Image)

        # Should not work with a simple string
        simple_string = base64_image
        image = model._get_image_from_message_part(simple_string)
        assert image is None

        # Should not work with a string message
        message = {"type": "text", "text": "I'm a text message"}
        image = model._get_image_from_message_part(message)
        assert image is None


def test_get_text_from_message_part() -> None:
    dummy_message = "Some message"
    model = _BaseImageTextModel()

    # Should not work with an image
    message = {"type": "image_url", "image_url": {"url": base64_image}}
    text = model._get_text_from_message_part(message)
    assert text is None

    # Should work with a simple string
    simple_message = dummy_message
    text = model._get_text_from_message_part(simple_message)
    assert text == dummy_message

    # Should work with a text message
    message = {"type": "text", "text": dummy_message}
    text = model._get_text_from_message_part(message)
    assert text == dummy_message


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
