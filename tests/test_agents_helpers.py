"""agents 辅助函数（连续高温天数推断）测试。"""
from __future__ import annotations

from earthbench.agents import _max_consecutive_days


def test_same_day_dedup():
    # 同一天多条高温读数 → 只算 1 天（修复重复计数）
    assert _max_consecutive_days([
        "2026-07-01T12:00:00", "2026-07-01T18:00:00", "2026-07-01T20:00:00",
    ]) == 1


def test_gap_breaks_streak():
    # 1 号与 3 号高温、2 号缺 → 最长连续 1 天
    assert _max_consecutive_days(["2026-07-01T12", "2026-07-03T12"]) == 1


def test_consecutive_days():
    assert _max_consecutive_days(
        ["2026-07-01T12", "2026-07-02T12", "2026-07-03T12"]
    ) == 3


def test_out_of_order_and_gap():
    # 乱序输入应按日历排序；1/2/4 号 → 最长连续 2
    assert _max_consecutive_days(
        ["2026-07-04T12", "2026-07-01T12", "2026-07-02T12"]
    ) == 2


def test_empty_and_invalid_fallback():
    assert _max_consecutive_days([]) == 1
    assert _max_consecutive_days(["not-a-date"]) == 1