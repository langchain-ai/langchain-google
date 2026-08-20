"""Reproduce streamed multi-image aggregation with Nano Banana."""

import base64
import binascii
import sys
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import AIMessageChunk

from langchain_google_genai import ChatGoogleGenerativeAI, Modality

MODEL = "gemini-3.1-flash-lite-image"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def save_image(block: dict, output_dir: Path, number: int) -> Path:
    image_url = block["image_url"]
    url = image_url["url"] if isinstance(image_url, dict) else image_url
    parsed = urlparse(url)
    if parsed.scheme != "data":
        message = f"Expected a data URI, got scheme {parsed.scheme!r}"
        raise ValueError(message)

    header, separator, payload = url.partition(",")
    mime_type = header.removeprefix("data:").split(";", 1)[0]
    if not separator or ";base64" not in header or mime_type not in EXTENSIONS:
        message = f"Unsupported image data URI: {header!r}"
        raise ValueError(message)
    if len(payload) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        message = "Image exceeds the 25 MiB limit"
        raise ValueError(message)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except binascii.Error as error:
        message = "Invalid base64 image payload"
        raise ValueError(message) from error
    if len(image_bytes) > MAX_IMAGE_BYTES:
        message = "Image exceeds the 25 MiB limit"
        raise ValueError(message)

    path = output_dir / f"nano-banana-{number}{EXTENSIONS[mime_type]}"
    path.write_bytes(image_bytes)
    return path


def main() -> None:
    model = ChatGoogleGenerativeAI(
        model=MODEL,
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    )
    prompt = (
        "Generate exactly two separate image outputs, not a collage: "
        "first, a red Toyota Supra in a studio; second, a blue Toyota Tacoma "
        "on a mountain road. Return both as independent images."
    )

    combined: AIMessageChunk | None = None
    for chunk in model.stream(prompt):
        combined = chunk if combined is None else combined + chunk

    if combined is None or not isinstance(combined.content, list):
        message = "The model returned no multimodal content"
        raise RuntimeError(message)

    image_blocks = [
        block
        for block in combined.content
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    indices = [block.get("index") for block in image_blocks]
    sys.stdout.write(f"image indices: {indices}\n")

    output_dir = (Path.home() / "Downloads").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, block in enumerate(image_blocks, start=1):
        path = save_image(block, output_dir, number)
        sys.stdout.write(f"{path}\n")

    if len(image_blocks) != 2:
        message = f"Expected 2 image blocks, got {len(image_blocks)}"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
