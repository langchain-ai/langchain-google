from typing import List

import pytest

from langchain_google_vertexai._utils import GoogleModelFamily


@pytest.mark.parametrize(
    "srcs,exp",
    [
        (
            [
                "chat-bison@001",
                "text-bison@002",
                "medlm-medium",
                "medlm-large",
            ],
            GoogleModelFamily.PALM,
        ),
        (
            [
                "code-bison@002",
                "code-gecko@002",
            ],
            GoogleModelFamily.CODEY,
        ),
        (
            [
                "gemini-1.0-pro-001",
                "gemini-1.0-pro-002",
                "gemini-1.0-pro-vision-001",
                "gemini-1.0-pro-vision",
                "medlm-medium@latest",
            ],
            GoogleModelFamily.GEMINI,
        ),
        (
            [
                "gemini-1.5-flash-preview-0514",
                "gemini-1.5-pro-preview-0514",
                "gemini-1.5-pro-preview-0409",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-001",
            ],
            GoogleModelFamily.GEMINI_ADVANCED,
        ),
    ],
)
def test_google_model_family(srcs: List[str], exp: GoogleModelFamily):
    for src in srcs:
        res = GoogleModelFamily(src)
        assert res == exp
