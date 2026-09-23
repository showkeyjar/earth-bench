"""共享气象物理工具（单一来源，供 data_collectors / verification / cars_serve 复用）。

此前 Stull 湿球公式在 data_collectors（内联）、verification（_wet_bulb）、
cars_serve（向量化 wet_bulb_stull）三处各写一份，系数漂移风险高；统一到本模块。
"""

from __future__ import annotations

import math


def wet_bulb_stull(t_c: float, rh_pct: float) -> float:
    """Stull (2011) 湿球温度近似（°C），标量版。

    Tw = T·atan(0.151977·√(RH+8.313659)) + atan(T+RH)
         − atan(RH−1.676331) + 0.00391838·RH^1.5·atan(0.023101·RH) − 4.686035

    适用 −20~50°C、5~99% RH，误差典型 <1°C。RH 自动夹断到 [1,100]。
    返回未舍入浮点值，展示层自行 round。
    """
    t = float(t_c)
    rh = min(max(float(rh_pct), 1.0), 100.0)
    return (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )


def wet_bulb_inv(rh_pct: float, target: float = 27.0) -> float:
    """给定 RH，求使湿球温度 == target 的干球温度。

    湿球温度关于干球温度单调递增，用二分法求逆（60 次迭代足够收敛）。
    """
    lo, hi = -60.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wet_bulb_stull(mid, rh_pct) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)