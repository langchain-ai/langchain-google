from tempfile import NamedTemporaryFile

import pytest

from langchain_google_vertexai._image_utils import (
    ImageBytesLoader,
    create_image_content_part,
    create_text_content_part,
    get_image_str_from_content_part,
    get_text_str_from_content_part,
    image_bytes_to_b64_string,
)


def test_get_text_str_from_content_part() -> None:
    content_part = "This is a text"
    result = get_text_str_from_content_part(content_part)
    assert result == content_part

    content_part_dict = {"type": "text", "text": "This is a text"}
    result = get_text_str_from_content_part(content_part_dict)
    assert result == content_part_dict["text"]

    content_part_dict = {"type": "image", "text": "This is a text"}
    result = get_text_str_from_content_part(content_part_dict)
    assert result is None

    content_part_dict = {"foo": "image", "bar": "This is a text"}
    result = get_text_str_from_content_part(content_part_dict)
    assert result is None


def test_get_image_str_from_content_part() -> None:
    content_part = "This is a text"
    result = get_image_str_from_content_part(content_part)
    assert result is None

    content_part_dict = {"type": "image_url", "image_url": {"url": "img_url"}}
    result = get_image_str_from_content_part(content_part_dict)
    assert isinstance(content_part_dict["image_url"], dict)
    assert result == content_part_dict["image_url"]["url"]

    content_part_dict = {"type": "image", "text": "This is a text"}
    result = get_image_str_from_content_part(content_part_dict)
    assert result is None

    content_part_dict = {"foo": "image", "bar": "This is a text"}
    result = get_image_str_from_content_part(content_part_dict)
    assert result is None


def test_create_content_parts() -> None:
    message_str = "This is a message str"
    text_content_part = create_text_content_part(message_str)
    result = get_text_str_from_content_part(text_content_part)
    assert message_str == result

    message_str = "This is a image str"
    text_content_part = create_image_content_part(message_str)
    result = get_image_str_from_content_part(text_content_part)
    assert message_str == result


def test_image_bytes_loader() -> None:
    loader = ImageBytesLoader()

    base64_image = (
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

    # Check it loads from b64
    image_bytes = loader.load_bytes(base64_image)
    assert isinstance(image_bytes, bytes)

    # Check doesn't load from local file.
    file = NamedTemporaryFile()
    file.write(image_bytes)
    file.seek(0)
    with pytest.raises(ValueError):
        loader.load_bytes(file.name)
    file.close()

    # Check if fails if nosense string
    with pytest.raises(ValueError):
        loader.load_bytes("No sense string")

    # Checks inverse conversion
    recovered_b64 = image_bytes_to_b64_string(
        image_bytes, encoding="ascii", image_format="png"
    )

    assert recovered_b64 == base64_image
