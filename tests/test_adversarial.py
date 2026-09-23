# -*- coding: utf-8 -*-
"""判别力扩展套件（adversarial suite）测试。

核心断言：规则 baseline 在基础套件上 100%，但在对抗套件上必然漏判 ——
这正是基准判别力的来源（能区分「超越线性规则」的 Agent）。
"""
from __future__ import annotations

from earthbench.agents import MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.scenarios import get_adversarial_suite


def test_adversarial_suite_shape():
    suite = get_adversarial_suite()
    assert len(suite) >= 8
    cats = {c["category"] for c in suite}
    assert {"fire", "flood", "drought", "heat"} <= cats
    # 每个用例都有独立真值推导函数与 >= 3 条观测（模板最低要求）
    for c in suite:
        assert c.get("_gt_fn") is not None
        assert len(c["observations"]) >= 3


def test_adversarial_ground_truth_deterministic():
    b = AlertBenchEvaluator(suite=get_adversarial_suite())
    # 独立标准推导与文档硬编码值必须一致
    assert b.gt_divergences == []
    # 双向构造：一半「标准不触发」（误报陷阱），一半「标准触发」（漏报陷阱）
    gts = [tc.ground_truth for tc in b.test_cases]
    assert gts.count(False) >= 4  # 误报方向：GT=False
    assert gts.count(True) >= 4   # 漏报方向：GT=True


def test_rule_baseline_fails_adversarial_suite():
    """规则引擎在线性加权下双向失败 —— baseline 不再 100%，基准可判别。"""
    b = AlertBenchEvaluator(suite=get_adversarial_suite())
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc < 1.0
    # 双向失败模式：既有误报（pred=True, gt=False）也有漏报（pred=False, gt=True）
    fps = [r for r in results if r["predicted"] and not r["ground_truth"]]
    fns = [r for r in results if not r["predicted"] and r["ground_truth"]]
    assert len(fps) >= 4
    assert len(fns) >= 4


def test_rule_baseline_still_perfect_on_core_suite():
    """基础套件的既有结论不受对抗套件影响（对抗性只在新用例上体现）。"""
    b = AlertBenchEvaluator()  # 默认基础套件
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc == 1.0