"""
TrustFusion package initialisation.

Exposes high-level factory helpers so downstream applications can import
`build_default_system` and start orchestrating verification pipelines.
"""

from .core.pipeline import build_default_system

__all__ = ["build_default_system"]
