"""
Utilities: time averaging, radiometric helpers, QC summaries, and clear-sky / CEE detection.
工具子包：时间平均、辐射计算、QC 摘要及晴空 / 云增强检测。

Exposes :mod:`averaging`, :mod:`calculations`, :mod:`quality`,
:mod:`clear_sky_detection`, :mod:`cee_detection`, and convenience imports such as
:func:`~bsrn.utils.averaging.pretty_average` and CSD entry points.
"""

from . import averaging, calculations, quality, clear_sky_detection, cee_detection
from .averaging import pretty_average
from .clear_sky_detection import (
    reno_csd,
    ineichen_csd,
    lefevre_csd,
    brightsun_csd,
    detect_clearsky,
)

__all__ = [
    "averaging",
    "pretty_average",
    "calculations",
    "quality",
    "clear_sky_detection",
    "cee_detection",
    "reno_csd",
    "ineichen_csd",
    "lefevre_csd",
    "brightsun_csd",
    "detect_clearsky",
]
