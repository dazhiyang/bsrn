"""
Physics subpackage: solar geometry and related helpers (public API in ``geometry``).
物理子包：太阳几何及相关辅助（对外 API 在 ``geometry``）。

Internal SPA routines live in :mod:`bsrn.physics.spa` and are used by
:mod:`bsrn.physics.geometry`; do not import :mod:`spa` from application code.
内部 SPA 实现在 :mod:`bsrn.physics.spa`，由 :mod:`bsrn.physics.geometry` 调用；
应用层请勿直接导入 :mod:`spa`。
"""

from . import geometry
