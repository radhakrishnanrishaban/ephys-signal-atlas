"""Configuration and validation helpers for neurosignal."""

from .config import FS_DEFAULT  # noqa: F401
from .validators import validate_array  # noqa: F401

__all__ = ["FS_DEFAULT", "validate_array"]
