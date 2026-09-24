"""models.ScenarioCategory.from_string 告警行为测试。"""
from __future__ import annotations

import logging

from earthbench.models import ScenarioCategory


def test_from_string_valid():
    assert ScenarioCategory.from_string("fire") is ScenarioCategory.FIRE
    assert ScenarioCategory.from_string("heat") is ScenarioCategory.HEAT


def test_from_string_unknown_warns_and_falls_back(caplog):
    with caplog.at_level(logging.WARNING, logger="earthbench.models"):
        cat = ScenarioCategory.from_string("haet")  # 拼写错误
    assert cat is ScenarioCategory.FIRE  # 兼容旧行为：回退 FIRE
    assert any("haet" in r.message for r in caplog.records)  # 但必须留下痕迹