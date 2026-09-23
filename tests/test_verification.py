# -*- coding: utf-8 -*-
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


def test_verify_drought_missing_humidity_is_insufficient():
    out = v.verify_drought({}, {"daily": [{"precip": 0}]})
    assert out["verification_status"] == "insufficient_data"
    assert out["actual"] is None


def test_verify_drought_dry_and_low_humidity():
    out = v.verify_drought({}, {"daily": [{"precip": 0, "humidity": 30}]})
    assert out["verification_status"] == "verified"
    assert out["actual"] is True