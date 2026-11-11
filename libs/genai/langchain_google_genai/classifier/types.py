"""Question and response types for `GoogleGenAIClassifier`.

These are the shared `langchain_core.decisions` types. The Gemini Decisions API
requires `instructions` on every question, does not support `Box` questions, and
requires at least two `Score` levels; `GoogleGenAIClassifier` raises `ValueError`
for requests that do not meet these requirements.

Strings are sent as-is. Any other state, such as JSON objects, arrays, or LangChain
messages, is sent as JSON text, with messages converted to role/content objects.
LangChain standard media blocks (`type` of `image`, `audio`, `video`, or `file`,
with `base64` or `url` data) are sent as native media wherever they appear in the
state, at the position where they appear in the JSON text.
"""

from langchain_core.decisions import (
    Answer,
    Box,
    BoxAnswer,
    BoxCoordinates,
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    Content,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
    State,
)

__all__ = [
    "Answer",
    "Box",
    "BoxAnswer",
    "BoxCoordinates",
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "Content",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
]
