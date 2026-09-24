"""冲击变量检验闭环（cars_verify_impact：暴雨/大风）测试。

全部离线：观测抓取注入合成场，无网络。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

import earthbench.cars_verify_impact as cvi

CITIES = [
    {"region": "Wet", "lat": 30.0, "lon": 110.0, "name_zh": "雨城"},
    {"region": "Dry", "lat": 40.0, "lon": 120.0, "name_zh": "旱城"},
]


def _make_impact_archive(tmp_path, valid="2026-09-21"):
    hd = tmp_path / "cars_history"
    hd.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_for": valid,
        "records": [
            {"region": "Wet", "lat": 30.0, "lon": 110.0,
             "p_harm_rain": 0.8, "p_harm_wind": 0.05,
             "p_harm_rain_intense": 0.1, "p_harm_wind_intense": 0.0,
             "p_harm_wind_typhoon": 0.0, "p_harm_wind_extreme": 0.0,
             "p_harm_snow": 0.0, "p_harm_snow_intense": 0.0},
            {"region": "Dry", "lat": 40.0, "lon": 120.0,
             "p_harm_rain": 0.0, "p_harm_wind": 0.6,
             "p_harm_rain_intense": 0.0, "p_harm_wind_intense": 0.1,
             "p_harm_wind_typhoon": 0.4, "p_harm_wind_extreme": 0.0,
             "p_harm_snow": 0.0, "p_harm_snow_intense": 0.0},
        ],
    }
    (hd / f"forecast_impact_{valid}.json").write_text(
        json.dumps(doc), encoding="utf-8")


def _inject_obs(monkeypatch, rain_grid, wind_grid, t2m_grid=None):
    """注入合成观测：雨城格点暴雨+无风，旱城无雨+大风；t2m 默认全暖。"""
    iy_wet = int(np.argmin(np.abs(cvi.GRID_LAT - 30.0)))
    ix_wet = int(np.argmin(np.abs(cvi.GRID_LON - 110.0)))
    iy_dry = int(np.argmin(np.abs(cvi.GRID_LAT - 40.0)))
    ix_dry = int(np.argmin(np.abs(cvi.GRID_LON - 120.0)))
    rain = np.zeros((31, 41), dtype=np.float32)
    wind = np.zeros((31, 41), dtype=np.float32)
    rain[iy_wet, ix_wet] = 80.0     # ≥50mm → rain event
    wind[iy_dry, ix_dry] = 12.5     # ≥10.8 → wind event
    t2m = (t2m_grid if t2m_grid is not None
           else np.full((31, 41), 10.0, dtype=np.float32))  # 暖 → 无雪
    monkeypatch.setattr(cvi, "obs_gefs_valid_day",
                        lambda v: (rain, wind))
    monkeypatch.setattr(cvi, "obs_gefs_t2m_valid_day",
                        lambda v: t2m)


def test_verify_impact_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    _make_impact_archive(tmp_path)
    _inject_obs(monkeypatch, None, None)

    out = cvi.verify_impact(date(2026, 9, 21))
    assert out["n"] == 2

    s = cvi.summary()
    assert s["n"] == 2
    # 雨城：p=0.8 事件=1 → 命中、brier .04；旱城：p=0 事件=0 → 正确不报
    r = s["rain"]
    assert r["hit"] == 1 and r["miss"] == 0 and r["false_alarm"] == 0
    assert r["mean_brier"] == round((0.04 + 0.0) / 2, 4)
    # 风：旱城 p=0.6 事件=1 → 命中；雨城 p=0.05 事件=0 → 未触发（<30%）正确
    w = s["wind"]
    assert w["hit"] == 1 and w["false_alarm"] == 0
    # 阈值边界：观测恰在阈值上（80≥50、12.5≥10.8）
    assert r["event_rate"] == 0.5 and w["event_rate"] == 0.5
    # Phase D：台风级大风档 —— 旱城 p=0.4 触发但观测 12.5 < 17.2 → 空报
    wty = s["wind_typhoon"]
    assert wty["n"] == 2 and wty["false_alarm"] == 1 and wty["hit"] == 0
    # Phase D：暴雪档 —— 全暖观测（10°C）→ 无雪事件，预报 p=0 → 正确不报
    sn = s["snow"]
    assert sn["n"] == 2 and sn["hit"] == 0 and sn["false_alarm"] == 0
    assert sn["event_rate"] == 0.0
    # 去重（同日重跑不翻倍）
    cvi.verify_impact(date(2026, 9, 21))
    assert cvi.summary()["n"] == 2


def test_verify_impact_miss_and_false_alarm(tmp_path, monkeypatch):
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    _make_impact_archive(tmp_path)
    # 反转观测：雨城无雨（p=0.8 → 空报），旱城暴雨... 风旱城有、雨城大风（p=0.05 → 漏报）
    rain = np.zeros((31, 41), dtype=np.float32)
    wind = np.zeros((31, 41), dtype=np.float32)
    iy = [int(np.argmin(np.abs(cvi.GRID_LAT - c["lat"]))) for c in CITIES]
    ix = [int(np.argmin(np.abs(cvi.GRID_LON - c["lon"]))) for c in CITIES]
    rain[iy[1], ix[1]] = 90.0    # Dry 有暴雨但预报 p=0 → 漏报
    wind[iy[0], ix[0]] = 14.0    # Wet 有大风但预报 p=0.05 → 漏报
    monkeypatch.setattr(cvi, "obs_gefs_valid_day", lambda v: (rain, wind))
    monkeypatch.setattr(cvi, "obs_gefs_t2m_valid_day",
                        lambda v: np.full((31, 41), 10.0, dtype=np.float32))

    cvi.verify_impact(date(2026, 9, 21))
    s = cvi.summary()
    assert s["rain"]["false_alarm"] == 1   # Wet p=0.8 无事件
    assert s["rain"]["miss"] == 1          # Dry p=0 有事件
    assert s["rain"]["csi"] == 0.0
    assert s["wind"]["miss"] == 1


def test_verify_impact_snow_freeze_mask_miss(tmp_path, monkeypatch):
    """Phase D：冻结掩膜检验 —— 同一场 80mm 降水在 -5°C 下是暴雪事件。

    预报侧 p_harm_snow=0（旧归档未启雪档）→ 漏报被新口径捕获。
    """
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    _make_impact_archive(tmp_path)
    # 雨城暴雨 + 冻结（-5°C < 0.5）→ obs 降雪 80mm ≥ 10 → 雪事件
    t2m = np.full((31, 41), 10.0, dtype=np.float32)
    iy_wet = int(np.argmin(np.abs(cvi.GRID_LAT - 30.0)))
    ix_wet = int(np.argmin(np.abs(cvi.GRID_LON - 110.0)))
    t2m[iy_wet, ix_wet] = -5.0
    _inject_obs(monkeypatch, None, None, t2m_grid=t2m)

    out = cvi.verify_impact(date(2026, 9, 21))
    rec_wet = next(r for r in out["records"] if r["region"] == "Wet")
    assert rec_wet["obs_snow_mm_wfe"] == 80.0     # 冻结 → 降水计雪
    assert rec_wet["observed_snow"] is True
    assert rec_wet["p_harm_snow"] == 0.0          # 旧归档未启雪档
    assert rec_wet["brier_snow"] == 1.0           # (0 - 1)^2：漏报重罚
    s = cvi.summary()
    assert s["snow"]["miss"] == 1                 # p=0 有事件 → 漏报
    assert s["snow"]["hit"] == 0


def test_verify_missing_archive_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cvi.verify_impact(date(2026, 1, 1))


def test_latest_verifiable_impact(tmp_path, monkeypatch):
    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    assert cvi.latest_verifiable_impact() is None
    _make_impact_archive(tmp_path, valid="2026-09-20")
    # 归档日不必是"过去"，latest_verifiable 只找存在的归档（今日-1 起回看）
    # → 2026-09-20 在窗口内与否取决于运行日；至少不抛错
    v = cvi.latest_verifiable_impact()
    assert v is None or v.isoformat() == "2026-09-20"


def test_panel_includes_impact_verify_summary(tmp_path, monkeypatch):
    from earthbench import publish_pipeline as pp

    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    _make_impact_archive(tmp_path)
    _inject_obs(monkeypatch, None, None)
    cvi.verify_impact(date(2026, 9, 21))

    doc = {
        "generated_for": "2026-09-21",
        "p_trigger": {"rain": 0.3, "wind": 0.3},
        "records": [{"region": "Wet", "name_zh": "雨城",
                     "p_harm_rain": 0.8, "p_harm_rain_intense": 0.1,
                     "rain_member_max_mm": 90.0,
                     "p_harm_wind": 0.05, "p_harm_wind_intense": 0.0,
                     "wind_member_max_ms": 9.0}],
    }
    (tmp_path / "cars_probabilities_impact_daily.json").write_text(
        json.dumps(doc), encoding="utf-8")
    lines = pp._build_cars_impact_section()
    assert any("冲击变量滚动检验" in ln for ln in lines)
    assert any("暴雨 Brier" in ln for ln in lines)
