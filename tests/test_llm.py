"""LLM 决策解析（earthbench.llm.parse_yes_no）测试。"""
from __future__ import annotations

from earthbench.llm import domain_hint, parse_yes_no


def test_explicit_decision():
    assert parse_yes_no("决策: YES") == "yes"
    assert parse_yes_no("决策：否") == "no"
    assert parse_yes_no("Decision: NO") == "no"


def test_first_line():
    assert parse_yes_no("YES\n理由略") == "yes"
    assert parse_yes_no("NO\n理由略") == "no"


def test_negation_not_mistaken_for_yes():
    # 修复：否定短语不能被正向短语「建议…预警 / 需要…预警」误判
    assert parse_yes_no("不建议预警") == "no"
    assert parse_yes_no("不需要预警") == "no"
    assert parse_yes_no("不应发出预警") == "no"


def test_positive_action():
    assert parse_yes_no("建议立即预警") == "yes"
    assert parse_yes_no("需要发出预警") == "yes"


def test_unparseable():
    assert parse_yes_no("随便聊聊") is None
    assert parse_yes_no("") is None
    assert parse_yes_no(None) is None


def test_domain_hint():
    assert "FWI" in domain_hint("fire")
    assert "湿球" in domain_hint("heat")
    assert domain_hint("unknown") == ""