import pytest
from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from google.cloud.exceptions import NotFound

from langchain_google_vertexai._image_utils import ImageBytesLoader


@pytest.mark.skip("CI testing not set up")
def test_image_utils() -> None:
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

    loader = ImageBytesLoader()

    image_bytes = loader.load_bytes(base64_image)

    assert isinstance(image_bytes, bytes)

    # Check loads image from blob

    bucket_name = "test_image_utils"
    blob_name = "my_image.png"

    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_string(data=image_bytes)
    except NotFound:
        client.create_bucket(bucket)
        blob.upload_from_string(data=image_bytes)

    gcs_uri = f"gs://{bucket.name}/{blob.name}"

    gcs_image_bytes = loader.load_bytes(gcs_uri)

    assert image_bytes == gcs_image_bytes

    # Checks loads image from url
    url = (
        "https://www.google.co.jp/images/branding/"
        "googlelogo/1x/googlelogo_color_272x92dp.png"
    )

    image_bytes = loader.load_bytes(url)
    assert isinstance(image_bytes, bytes)
