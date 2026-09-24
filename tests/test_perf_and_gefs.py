"""性能修复与 gefs_io 单一实现的回归测试。

覆盖：
- gefs_io.prune_gefs_cache 的按 init 日期清理（老文件删、新文件留、异名不动）
- gefs_io.fetch_grib 的路径构造与已存在不重下
- collect_region_weather 区域级 memoize（同区域多灾种只打一次 API）
- publish_pipeline.collect_data 先过滤后采集（included_categories 外不打 API）
- cars_serve.daily_update 全 mock e2e（无 crps_cars 也能跑通主路径）
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

import earthbench.data_collectors as dc
import earthbench.gefs_io as gio
import earthbench.publish_pipeline as pp

# ---------------------------------------------------------------------------
# gefs_io
# ---------------------------------------------------------------------------


def _mk_grib_file(cache, name: str):
    p = cache / name
    p.write_bytes(b"fake-grib")
    return p


def test_prune_gefs_cache_by_init_date(tmp_path, monkeypatch):
    monkeypatch.setattr(gio, "GEFS_CACHE", tmp_path)
    old = _mk_grib_file(tmp_path, "2026010100_gec00.t00z.pgrb2a.0p50.f048")
    new = _mk_grib_file(tmp_path, "2026123100_gec00.t00z.pgrb2a.0p50.f048")
    weird = _mk_grib_file(tmp_path, "not-a-grib.txt")

    removed = gio.prune_gefs_cache(keep_days=14)

    assert removed == 1
    assert not old.exists()
    assert new.exists()
    assert weird.exists()  # 命名不匹配的文件不动


def test_prune_gefs_cache_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gio, "GEFS_CACHE", tmp_path / "nonexistent")
    assert gio.prune_gefs_cache() == 0


def test_fetch_grib_reuses_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(gio, "GEFS_CACHE", tmp_path)
    local = _mk_grib_file(tmp_path, "2026010100_gec00.t00z.pgrb2a.0p50.f048")

    calls = []

    def fake_urlretrieve(url, path):
        calls.append(url)
        from pathlib import Path

        Path(path).write_bytes(b"fake-grib")

    monkeypatch.setattr(gio.urllib.request, "urlretrieve", fake_urlretrieve)

    out = gio.fetch_grib("2026-01-01", "048")
    assert out == local
    assert calls == []  # 已存在不重下

    out2 = gio.fetch_grib("2026-01-02", "048")
    assert calls == [
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260102/00/atmos/"
        "pgrb2ap5/gec00.t00z.pgrb2a.0p50.f048"
    ]
    assert out2.exists()


# ---------------------------------------------------------------------------
# 采集 memoize + 先过滤
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fake_weather(monkeypatch):
    dc._REGION_WEATHER_CACHE.clear()
    calls = {"realtime": 0}

    def fake_realtime(location_id, lon, lat):
        calls["realtime"] += 1
        return {"temp": 25.0, "humidity": 50, "wind_speed_ms": 2.0, "precip_1h": 0.0}

    monkeypatch.setattr(dc, "fetch_realtime_weather", fake_realtime)
    monkeypatch.setattr(dc, "fetch_hourly_forecast", lambda loc, hours=24: [])
    monkeypatch.setattr(dc, "fetch_daily_forecast", lambda loc, days=7: [])
    yield calls
    dc._REGION_WEATHER_CACHE.clear()


def test_collect_region_weather_memoizes_per_region(_fake_weather):
    obs_fire = dc.collect_region_weather("Xiangshan-Beijing", "fire")
    obs_heat = dc.collect_region_weather("Xiangshan-Beijing", "heat")

    assert _fake_weather["realtime"] == 1  # 同区域两灾种只打一次 realtime API
    assert obs_fire and obs_heat


def test_collect_region_weather_caches_failure(_fake_weather, monkeypatch):
    def fail_realtime(location_id, lon, lat):
        _fake_weather["realtime"] += 1
        return None

    monkeypatch.setattr(dc, "fetch_realtime_weather", fail_realtime)
    assert dc.collect_region_weather("Xiangshan-Beijing", "fire") == []
    assert dc.collect_region_weather("Xiangshan-Beijing", "heat") == []
    assert _fake_weather["realtime"] == 1  # 持续失败也只打一次


def test_collect_data_filters_before_fetch(monkeypatch):
    dc._REGION_WEATHER_CACHE.clear()
    calls = []

    def fake_collect(region_key, category):
        calls.append((region_key, category))
        return []  # 全部走 fallback，断言的是调用面

    monkeypatch.setattr(dc, "collect_region_weather", fake_collect)

    suite = pp.collect_data()

    included = set(pp.PUBLISH_CONFIG["included_categories"])
    assert calls, "应至少发生一次采集调用"
    assert all(cat in included for _, cat in calls)  # 过滤后才采集
    assert len(suite) == 40  # 全部场景保留（fallback）
    assert sum(1 for s in suite if s["_data_source"] == "fallback_alertbench") == 40
    dc._REGION_WEATHER_CACHE.clear()


# ---------------------------------------------------------------------------
# cars_serve.daily_update 全 mock e2e
# ---------------------------------------------------------------------------


class _FakeDeepCars:
    """无 crps_cars 环境下的最小 DeepCars 替身。"""

    def __init__(self, artifact=None, climo=None):
        self.lats = gio.GRID_LAT
        self.lons = gio.GRID_LON
        self.climo_mean = np.full((12, 1, 31, 41), 290.0, dtype=np.float32)
        self.climo_std = np.full((12, 1, 31, 41), 3.0, dtype=np.float32)

    def predict(self, forecast):
        t = forecast[0, 0]
        members = t[None, None] + np.linspace(-1.0, 1.0, 30)[:, None, None]
        return members[None]  # (1,30,1,31,41)

    def cdf(self, forecast, y):
        mu = float(forecast[0, 0].mean())
        from math import erf, sqrt

        return np.full((1, 1, 31, 41), 0.5 * (1.0 + erf((float(y) - mu) / (3.0 * sqrt(2)))))


def test_daily_update_e2e_mocked(tmp_path, monkeypatch):
    import earthbench.cars_serve as cs

    monkeypatch.setenv("CARS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "DeepCars", _FakeDeepCars)
    # daily_update 内部是函数级 from .gefs_io import prune_gefs_cache，patch 源模块
    monkeypatch.setattr(gio, "prune_gefs_cache", lambda keep_days=14: 0)

    fields = {
        "t2m": np.full((31, 41), 300.0, dtype=np.float32),
        "r2": np.full((31, 41), 50.0, dtype=np.float32),
        "tmax": np.full((31, 41), 305.0, dtype=np.float32),
        "tmin": np.full((31, 41), 295.0, dtype=np.float32),
    }
    monkeypatch.setattr(cs, "fetch_ops_fields", lambda init, hour="00": fields)

    doc = cs.daily_update(date(2026, 9, 24))

    out = tmp_path / "cars_probabilities_daily.json"
    hist = tmp_path / "cars_history" / "forecast_2026-09-24.json"
    assert out.exists() and hist.exists()
    assert doc["generated_for"] == "2026-09-24"
    assert len(doc["records"]) > 0
    rec = doc["records"][0]
    for key in ("p_exceed_2sigma", "p_harm_heat", "p_harm_wb"):
        assert key in rec and 0.0 <= rec[key] <= 1.0
