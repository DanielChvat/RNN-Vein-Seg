"""Recurrent vein segmentation and vessel-centre inference.

The pipeline runs as ``python -m veinseg <stage>``; see :mod:`veinseg.cli`.
Every path, class definition and physical constant lives in :mod:`veinseg.config`.
"""

from . import config

__all__ = ["config"]
__version__ = "0.2.0"
