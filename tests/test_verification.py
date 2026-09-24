"""闭环验证（fire/drought）关键路径测试。"""
from __future__ import annotations

import earthbench.verification as v


def test_verify_fire_no_key_is_insufficient(monkeypatch):
    monkeypatch.setattr(v, "FIRMS_MAP_KEY", "")
    out = v.verify_fire({}, "Xiangshan-Beijing")
    assert out["verification_status"] == "insufficient_data"
    assert out["actual"] is None


def test_verify_fire_no_hotspot_is_verified_negative(monkeypatch):
    monkeypatch.setattr(v, "FIRMS_MAP_KEY", "dummy")
    monkeypatch.setattr(v, "fetch_firms_fire_data", lambda rid: [])
    out = v.verify_fire({}, "Xiangshan-Beijing")
    assert out["verification_status"] == "verified"
    assert out["actual"] is False


def test_verify_fire_with_hotspot_is_positive(monkeypatch):
    monkeypatch.setattr(v, "FIRMS_MAP_KEY", "dummy")
    monkeypatch.setattr(v, "fetch_firms_fire_data", lambda rid: [{"frp": 1.0}])
    out = v.verify_fire({}, "Xiangshan-Beijing")
    assert out["verification_status"] == "verified"
    assert out["actual"] is True


def test_verify_fire_query_failure_is_insufficient(monkeypatch):
    """FIRMS 查询失败（返回 None）≠ 无火：不得写进验证结论。"""
    monkeypatch.setattr(v, "FIRMS_MAP_KEY", "dummy")
    monkeypatch.setattr(v, "fetch_firms_fire_data", lambda rid: None)
    out = v.verify_fire({}, "Xiangshan-Beijing")
    assert out["verification_status"] == "insufficient_data"
    assert out["actual"] is None
    assert out["fire_count"] is None


def test_fetch_firms_fire_data_unknown_region_returns_none():
    """未知区域返回 None（而非空列表），verify_fire 据此给 insufficient_data。"""
    out = v.fetch_firms_fire_data("NonexistentRegion")
    assert out is None


def test_fetch_firms_hotspots_no_key_returns_none(monkeypatch):
    """无 key 属于「查询不可用」三态中的 None，不是可信的 0 火点。"""
    from earthbench import data_collectors as dc

    monkeypatch.setattr(dc, "FIRMS_MAP_KEY", "")
    assert dc.fetch_firms_hotspots(39.99, 116.16) is None


def test_has_active_fire_handles_unavailable_query(monkeypatch):
    """查询不可用时 has_active_fire 保守返回 False 且不崩溃。"""
    from earthbench import data_collectors as dc

    monkeypatch.setattr(dc, "fetch_firms_hotspots", lambda lat, lon, radius_km=20: None)
    assert dc.has_active_fire(39.99, 116.16) is False


def test_verify_drought_missing_humidity_is_insufficient():
    out = v.verify_drought({}, {"daily": [{"precip": 0}]})
    assert out["verification_status"] == "insufficient_data"
    assert out["actual"] is None


def test_verify_drought_dry_and_low_humidity():
    out = v.verify_drought({}, {"daily": [{"precip": 0, "humidity": 30}]})
    assert out["verification_status"] == "verified"
    assert out["actual"] is True