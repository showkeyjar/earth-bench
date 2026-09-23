# -*- coding: utf-8 -*-
"""共享湿球公式（earthbench.weather）测试。"""
from __future__ import annotations

from earthbench.weather import wet_bulb_stull, wet_bulb_inv


def test_wet_bulb_reference_points():
    # Stull (2011) 标定参考点（误差 <1°C）
    assert 26.0 <= wet_bulb_stull(30.0, 80.0) <= 28.0
    assert 27.5 <= wet_bulb_stull(35.0, 60.0) <= 29.5
    assert 12.5 <= wet_bulb_stull(20.0, 50.0) <= 14.5
    assert wet_bulb_stull(35.0, 15.0) < 22.0


def test_wet_bulb_rh_clamped():
    # RH 自动夹断到 [1,100]，不抛异常
    assert isinstance(wet_bulb_stull(30.0, 0.0), float)
    assert isinstance(wet_bulb_stull(30.0, 150.0), float)


def test_wet_bulb_inv_consistent():
    # 二分求逆：wet_bulb_inv(rh, target) 代回应近似 target
    for rh in (30.0, 50.0, 80.0):
        t = wet_bulb_inv(rh, target=27.0)
        assert abs(wet_bulb_stull(t, rh) - 27.0) < 0.05