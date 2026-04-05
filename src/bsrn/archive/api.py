"""
BSRN station-to-archive helpers (R ``1_utils.R``).

Logical records are Pydantic models in ``records_dynamic``; this module keeps
``get_azimuth_elevation`` for LR0004 horizon layout.

BSRN 站点存档辅助。逻辑记录为 ``records_dynamic`` 中的 Pydantic 模型；
本模块保留 LR0004 用的 ``get_azimuth_elevation``。
"""

# =============================================================================
# TRANSLATION OF: 1_utils.R
# =============================================================================


def get_azimuth_elevation(azimuth=None, elevation=None):
    """
    Format horizon azimuth/elevation lists for LR0004.

    Translates from R function ``getAzimuthElevation`` (``1_utils.R``).
    对应 R 函数 ``getAzimuthElevation``（``1_utils.R``）。

    Parameters
    ----------
    azimuth : str or sequence of float, optional
        Comma-separated string ``A1,A2,...`` or sequence of degrees from north.
        逗号分隔字符串 ``A1,A2,...`` 或从正北起算的方位角序列。
    elevation : str or sequence of float, optional
        Comma-separated string ``E1,E2,...`` or sequence of elevation angles.
        逗号分隔字符串 ``E1,E2,...`` 或高度角序列。

    Returns
    -------
    str
        Fixed-width lines of ``az el`` pairs, or ``  -1 -1`` when inputs are absent.
        固定宽度 ``az el`` 行；无输入时为 ``  -1 -1``。

    Raises
    ------
    ValueError
        If ``azimuth`` and ``elevation`` lengths differ.
        ``azimuth`` 与 ``elevation`` 长度不一致时。
    """
    if azimuth is None or elevation is None:
        return "  -1 -1"

    az = [float(x) for x in azimuth.split(",")] if isinstance(azimuth, str) else list(azimuth)
    el = [float(x) for x in elevation.split(",")] if isinstance(elevation, str) else list(elevation)

    if len(az) != len(el):
        raise ValueError("azimuth and elevation must have same size")

    n = len(az)
    pad = 11 - (n % 11) if n % 11 != 0 else 0
    az_padded = az + [-1] * pad
    el_padded = el + [-1] * pad

    rows = []
    for i in range(0, len(az_padded), 11):
        line = " ".join(
            [f"{a:>3.0f} {e:>2.0f}" for a, e in zip(az_padded[i : i + 11], el_padded[i : i + 11])]
        )
        rows.append(f" {line}")
    return "\n".join(rows)
