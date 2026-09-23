# -*- coding: utf-8 -*-
"""冲击变量服务（cars_serve_impact：暴雨/大风）测试。

模型相关用例依赖 crps_cars（与 t2m 通道同款环境约定：本地 PYTHONPATH 指
CRPS/src，CI 从 CRPS 仓库安装）；缺依赖时自动跳过，纯 JSON/记录用例始终跑。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from earthbench.cars_serve_impact import (
    GRID_LAT,
    GRID_LON,
    RAIN_P_TRIG,
    WIND_P_TRIG,
    _city_records,
    load_impact_config,
)

CITIES = [
    {"region": "TestCity", "lat": 30.0, "lon": 110.0, "name_zh": "测试城"},
    {"region": "NorthCity", "lat": 45.0, "lon": 120.0, "name_zh": "北城"},
]


def test_triggers_use_expected_cost_ratios():
    assert RAIN_P_TRIG == 0.3      # flood 成本比
    assert WIND_P_TRIG == 0.3      # 披露假设（无既有类别）


def test_impact_config_single_source():
    cfg = load_impact_config()
    for var in ("tp", "wind"):
        c = cfg[var]
        assert c["model"] == "SeasonalCarsModel"
        # 诚实选择众数超参（CRPS 滚动回测 10/15 年）
        assert c["config"]["k"] == 30
        assert c["config"]["pca_components"] == 16
        assert c["config"]["alpha_state"] == 0.0
        assert c["config"]["beta_forecast"] == 1.0
        assert c["config"]["day_window"] == 121
        assert "provenance" in c
    # 判据 v2：tp 深破裂上补全，wind 浅越界 raw
    assert cfg["tp"]["tail_completion"]["infl"] == 1.5
    assert cfg["wind"]["tail_completion"] is None
    assert cfg["tp"]["thresholds"]["hard"] == 50.0
    assert cfg["wind"]["thresholds"]["hard"] == 10.8


def test_city_records_member_fractions():
    """成员计数概率与统计摘要（合成成员，无模型依赖）。"""
    iy = int(np.argmin(np.abs(GRID_LAT - 30.0)))
    ix = int(np.argmin(np.abs(GRID_LON - 110.0)))
    rng = np.random.default_rng(0)
    mt = rng.uniform(0, 120, 30)          # mm/day
    mw = rng.uniform(0, 16, 30)           # m/s
    mt[:15] = 60.0                        # 15/30 ≥ 50mm
    mt[15:] = 10.0                        # 其余明确低于阈值（不随机）
    mw[:6] = 12.0                         # 6/30 ≥ 10.8 m/s
    mw[6:] = 5.0
    tp_mem = np.zeros((1, 30, 1, 31, 41)); tp_mem[0, :, 0, iy, ix] = mt
    wd_mem = np.zeros((1, 30, 1, 31, 41)); wd_mem[0, :, 0, iy, ix] = mw

    class _P:  # 最小配置桩
        thresholds = {"hard": 50.0, "intense": 100.0}
    class _W:
        thresholds = {"hard": 10.8, "intense": 13.9}

    recs = _city_records(tp_mem, wd_mem, date(2026, 9, 25), _P(), _W(), CITIES)
    r = recs[0]
    assert r["p_harm_rain"] == 0.5
    assert r["p_harm_wind"] == round(6 / 30, 3)
    assert r["rain_member_max_mm"] == round(float(mt.max()), 1)
    assert r["wind_member_max_ms"] == round(float(mw.max()), 1)
    assert r["valid_date"] == "2026-09-25"
    # 北城格点全 0 → 概率 0，不越界
    assert recs[1]["p_harm_rain"] == 0.0


def test_panel_renders_from_json(tmp_path, monkeypatch):
    from earthbench import publish_pipeline as pp

    doc = {
        "generated_for": "2026-09-25",
        "p_trigger": {"rain": 0.3, "wind": 0.3},
        "records": [{
            "region": "TestCity", "name_zh": "测试城",
            "p_harm_rain": 0.4, "p_harm_rain_intense": 0.1,
            "rain_member_max_mm": 88.0,
            "p_harm_wind": 0.1, "p_harm_wind_intense": 0.0,
            "wind_member_max_ms": 12.3,
        }],
    }
    (tmp_path / "cars_probabilities_impact_daily.json").write_text(
        json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    lines = pp._build_cars_impact_section()
    assert any("48小时暴雨/大风概率" in ln for ln in lines)
    assert any("测试城" in ln for ln in lines)
    assert any("🟡 暴雨" in ln for ln in lines)      # 0.4 ≥ 0.3，未到 100mm 线
    # 大风旗：P(≥10.8)=0.1 < 0.3 不触发 → 该行无 💨
    row = next(ln for ln in lines if "测试城" in ln)
    assert "💨 大风" not in row


def test_panel_degrades_without_data(tmp_path, monkeypatch):
    from earthbench import publish_pipeline as pp

    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    assert pp._build_cars_impact_section() == []


def _write_synthetic_archive(path: str, n: int = 90, scale: float = 1.0):
    """合成档案：90 天（2000-03 起隔日），预报正偏、误差含负值。"""
    rng = np.random.default_rng(42)
    f = (rng.uniform(1.0, 20.0, (n, 1, 31, 41)) * scale).astype(np.float32)
    t = np.maximum(f + rng.normal(0, 0.3 * scale, f.shape), 0
                   ).astype(np.float32)
    d = np.asarray([np.datetime64("2000-03-01", "D") + np.timedelta64(2 * i, "D")
                    for i in range(n)])
    np.savez_compressed(path, x=t, forecast=f, truth=t, dates=d,
                        grid_lat=GRID_LAT, grid_lon=GRID_LON)


def _impact_cars(var, tmp_path, completion):
    pytest.importorskip("crps_cars")
    from earthbench.cars_serve_impact import ImpactCars

    _write_synthetic_archive(str(tmp_path / "syn_archive.npz"))
    cfg = load_impact_config()[var]
    cfg = dict(cfg)
    cfg["archive"] = "syn_archive.npz"
    if completion:
        cfg["tail_completion"] = {"mode": "high", "q": 0.2, "infl": 1.5}
    else:
        cfg["tail_completion"] = None
    cp = tmp_path / f"syn_config_{var}_{completion}.json"
    cp.write_text(json.dumps({var: cfg}), encoding="utf-8")
    return ImpactCars(var, config_path=cp, archive_dir=tmp_path)


def test_impact_cars_fit_predict_and_clip(tmp_path):
    cars = _impact_cars("tp", tmp_path, completion=False)
    rng = np.random.default_rng(7)
    fc = rng.uniform(1.0, 15.0, (1, 1, 31, 41)).astype(np.float32)
    members = cars.predict_for(fc, date(2000, 4, 15))
    assert members.shape == (1, 30, 1, 31, 41)
    assert members.dtype == np.float32
    assert float(members.min()) >= 0.0          # 非负截断（风速/降水口径）


def test_tp_completion_policy_inflates_high_tail(tmp_path):
    """判据 v2 落点：tp 档有高尾补全（max 抬升），wind 档没有。"""
    rng = np.random.default_rng(7)
    fc = rng.uniform(1.0, 15.0, (1, 1, 31, 41)).astype(np.float32)
    raw = _impact_cars("tp", tmp_path, completion=False).predict_for(
        fc, date(2000, 4, 15))
    comp = _impact_cars("tp", tmp_path, completion=True).predict_for(
        fc, date(2000, 4, 15))
    # 补全只抬高分位以上成员：max 严格增大，min 不变（截断地板 0）
    assert float(comp.max()) >= float(raw.max())
    assert float(comp.min()) == pytest.approx(float(raw.min()), abs=1e-6)
    wind = _impact_cars("wind", tmp_path, completion=False)
    assert wind.cfg["tail_completion"] is None   # wind = raw 档


def test_daily_update_impact_end_to_end(tmp_path, monkeypatch):
    """全链：注入抓取器（无网络）→ 双变量 → JSON + 历史归档。"""
    import earthbench.cars_serve_impact as si

    pytest.importorskip("crps_cars")
    _write_synthetic_archive(str(tmp_path / "cars_tp_archive.npz"), scale=8.0)
    _write_synthetic_archive(str(tmp_path / "cars_wind_archive.npz"))
    (tmp_path / "cars_cities.json").write_text(
        json.dumps({"cities": CITIES}), encoding="utf-8")
    # 配置改指 tmp 档案名（配置文件在包内，档案名可覆盖）
    cfg = load_impact_config()
    for var in ("tp", "wind"):
        cfg[var]["archive"] = f"cars_{var}_archive.npz"
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        si, "load_impact_config", lambda: cfg)

    rng = np.random.default_rng(3)
    monkeypatch.setattr(si, "fetch_ops_apcp",
                        lambda init, hour="00": rng.uniform(
                            5.0, 40.0, (31, 41)).astype(np.float32))
    monkeypatch.setattr(si, "fetch_ops_wind",
                        lambda init, hour="00": rng.uniform(
                            3.0, 9.0, (31, 41)).astype(np.float32))

    doc = si.daily_update_impact(date(2026, 9, 25))
    assert doc["generated_for"] == "2026-09-25"
    assert len(doc["records"]) == 2
    r = doc["records"][0]
    for k in ("p_harm_rain", "p_harm_rain_intense",
              "p_harm_wind", "p_harm_wind_intense"):
        assert 0.0 <= r[k] <= 1.0
    assert (tmp_path / "cars_probabilities_impact_daily.json").exists()
    assert (tmp_path / "cars_history" /
            "forecast_impact_2026-09-25.json").exists()
    assert doc["p_trigger"] == {"rain": 0.3, "wind": 0.3}
    assert "tail_policy" in doc and "provenance" in doc
