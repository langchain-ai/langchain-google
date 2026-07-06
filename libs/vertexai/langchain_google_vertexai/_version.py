"""Package version for `langchain-google-vertexai`."""

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
