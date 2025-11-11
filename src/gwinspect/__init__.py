# src/gwinspect/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gwinspect")
except PackageNotFoundError:
    __version__ = "0+unknown"  # e.g., if running from a source tree not installed
