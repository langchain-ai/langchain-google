import re
from pathlib import Path

import langchain_google_genai

_CROSSREF = re.compile(r"\]\[langchain_google_genai\.(\w+)\.(\w+)\]")


def _source_files() -> list[Path]:
    return sorted(Path(langchain_google_genai.__file__).parent.glob("*.py"))


def test_crossrefs_target_directly_defined_members() -> None:
    """Docstring cross-references must point at directly-defined members.

    The API reference site resolves ``[label][langchain_google_genai.Class.member]``
    by symbol name and falls back to a *different* class when ``member`` is inherited
    (e.g. from the private ``_BaseGoogleGenerativeAI`` base), so the rendered link
    points at the wrong class. Guard against reintroducing such references.
    """
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for cls_name, member in _CROSSREF.findall(text):
            cls = getattr(langchain_google_genai, cls_name, None)
            if cls is None:
                continue
            directly_defined = member in vars(cls) or member in getattr(
                cls, "__annotations__", {}
            )
            if not directly_defined:
                offenders.append(f"{path.name}: {cls_name}.{member}")
    assert not offenders, (
        "Docstring cross-references must target directly-defined members; "
        "inherited members mis-resolve on the reference site. Offenders: "
        f"{offenders}"
    )
