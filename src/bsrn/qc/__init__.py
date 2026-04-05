"""
Quality control (QC) tests and high-level runners for BSRN station-to-archive data.
BSRN 站点存档数据的质量控制测试与高层入口。

Submodules implement individual tests (PPL, ERL, closure, diffuse ratio, k-index,
tracker-off); :mod:`wrapper` provides :func:`~bsrn.qc.wrapper.run_qc` and phase helpers.
各子模块实现单项测试（PPL、ERL、闭合、散射比、k-index、跟踪器）；
:mod:`wrapper` 提供 :func:`~bsrn.qc.wrapper.run_qc` 与分阶段接口。
"""

from . import ppl, erl, closure, diff_ratio, k_index, tracker, wrapper
from .wrapper import (
    run_qc,
    test_physically_possible,
    test_extremely_rare,
    test_closure,
    test_diff_ratio,
    test_k_index,
    test_tracker_off,
)
