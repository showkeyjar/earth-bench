# -*- coding: utf-8 -*-
"""阈值自校准（earthbench.calibration）存储测试。"""
from __future__ import annotations

import json

from earthbench.calibration import (
    save_thresholds,
    append_calibration_log,
    load_thresholds,
)


def test_save_thresholds_version_bump(tmp_path):
    save_thresholds(tmp_path, {"fire": 0.42})
    data = json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))
    assert data["thresholds"]["fire"] == 0.42
    assert data["version"] == 1

    save_thresholds(tmp_path, {"fire": 0.45, "heat": 0.44})
    data = json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["thresholds"]["heat"] == 0.44


def test_append_multiple_logs_all_persisted(tmp_path):
    # 修复：多个灾种同日调整时，每个都需落一条日志（此前只落第一条）
    append_calibration_log(
        tmp_path, {"category": "fire", "old_value": 0.4, "new_value": 0.42,
                   "reason": "fp", "adjustment": 0.02}
    )
    append_calibration_log(
        tmp_path, {"category": "heat", "old_value": 0.4, "new_value": 0.44,
                   "reason": "fn", "adjustment": 0.04}
    )

    logs = json.loads((tmp_path / "calibration_log.json").read_text(encoding="utf-8"))
    assert len(logs) == 2
    assert {e["category"] for e in logs} == {"fire", "heat"}


def test_load_thresholds_defaults_when_missing(tmp_path):
    loaded = load_thresholds(tmp_path)
    assert loaded["fire"] == 0.40  # DEFAULT_THRESHOLDS
    assert "heat" in loaded