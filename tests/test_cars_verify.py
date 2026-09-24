"""检验闭环（cars_verify）与湿球公式（cars_serve）测试。"""
from __future__ import annotations

import json

import numpy as np

from earthbench.cars_serve import wet_bulb_stull
from earthbench.cars_verify import summary, verify


def test_wet_bulb_stull_reference_points():
    # Stull (2011) 标定参考点（误差 <1°C）
    # 30°C / 80% -> ~26.5-27.5；35°C / 60% -> ~28-29；20°C / 50% -> ~13-14
    assert 26.0 <= wet_bulb_stull(np.array([30.0]), np.array([80.0]))[0] <= 28.0
    assert 27.5 <= wet_bulb_stull(np.array([35.0]), np.array([60.0]))[0] <= 29.5
    assert 12.5 <= wet_bulb_stull(np.array([20.0]), np.array([50.0]))[0] <= 14.5
    # 干空气 -> 湿球显著低于干球
    assert wet_bulb_stull(np.array([35.0]), np.array([15.0]))[0] < 22.0


def test_wet_bulb_monotone_in_rh():
    t = np.full(10, 32.0)
    rh = np.linspace(30.0, 95.0, 10)
    tw = wet_bulb_stull(t, rh)
    assert np.all(np.diff(tw) > 0)


def _make_forecast_archive(tmp_path, valid="2026-07-11"):
    hd = tmp_path / "cars_history"
    hd.mkdir(parents=True)
    doc = {
        "model": "test", "generated_for": valid,
        "records": [
            {"region": "A", "p_harm_heat": 0.9},
            {"region": "B", "p_harm_heat": 0.05},
            {"region": "C", "p_harm_heat": 0.5},
        ],
    }
    (hd / f"forecast_{valid}.json").write_text(
        json.dumps(doc), encoding="utf-8")


def test_verify_and_summary_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    _make_forecast_archive(tmp_path)
    # 无 QWeather key、GEFS 不可得时 verify 抛错——用注入观测绕过网络
    import earthbench.cars_verify as cv

    monkeypatch.setattr(
        cv, "_obs_qweather",
        lambda cities: {"A": 36.0, "B": 28.0, "C": 34.0})
    monkeypatch.setattr(cv, "_obs_gefs_f000",
                        lambda v, c: ({}, {}))
    out = verify(__import__("datetime").date(2026, 7, 11))
    assert out["n"] == 3

    s = summary()
    assert s["n"] == 3
    # A: p=0.9, event=1 -> brier .01, hit;  B: p=.05 event=0 -> brier .0025;
    # C: p=.5 event=0 -> brier .25, false alarm (0.5 >= 0.3)
    assert s["hit"] == 1 and s["false_alarm"] == 1 and s["miss"] == 0
    assert s["mean_brier"] == round((0.01 + 0.0025 + 0.25) / 3, 4)
    assert s["csi"] == round(1 / 2, 3)

    # 重跑同日去重（不翻倍）
    verify(__import__("datetime").date(2026, 7, 11))
    assert summary()["n"] == 3
