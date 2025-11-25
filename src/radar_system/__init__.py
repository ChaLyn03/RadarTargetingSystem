"""Radar Targeting System package entry.

Expose high-level subpackages for convenient imports.
"""

from importlib.metadata import version as _version

__all__ = ["sim", "dsp", "detection", "models", "pipeline"]

try:
    __version__ = _version("radar-targeting-system")
except Exception:
    __version__ = "0.0.0"
