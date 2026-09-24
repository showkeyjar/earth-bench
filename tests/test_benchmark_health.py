"""平凡基线与基准体检段测试（基准卫生批次）。"""
from __future__ import annotations

from earthbench.benchmark import AlertBenchEvaluator
from earthbench.publish_pipeline import _build_benchmark_health_section
from earthbench.scenarios import (
    get_adversarial_suite,
    get_alert_benchmark_suite,
)
from earthbench.trivial_agents import (
    AlwaysAlertAgent,
    NeverAlertAgent,
    RandomAgent,
)


def _ctx():
    """取一个真实场景上下文做接口测试。"""
    ev = AlertBenchEvaluator(suite=list(get_alert_benchmark_suite()[:2]))
    return ev.test_cases[0].to_context()


def test_trivial_agents_interface():
    """三个平凡基线满足 DecisionAgent 接口（decide → DecisionOutput）。"""
    from earthbench.models import DecisionOutput

    ctx = _ctx()
    for agent, expected in (
        (AlwaysAlertAgent(), True),
        (NeverAlertAgent(), False),
    ):
        out = agent.decide(ctx)
        assert isinstance(out, DecisionOutput)
        assert out.decision is expected
        assert out.context is ctx
        assert 0.0 <= out.confidence <= 1.0


def test_random_agent_deterministic():
    """随机基线跨运行可复现（case 派生哈希，无全局状态）。"""
    ctx = _ctx()
    a = RandomAgent(seed=0)
    assert a.decide(ctx).decision == a.decide(ctx).decision
    # 不同 seed 至少可能不同（弱断言：不抛错且稳定）
    b = RandomAgent(seed=1)
    assert b.decide(ctx).decision == b.decide(ctx).decision


def test_always_agent_on_adversarial_one_sided():
    """always 基线在对抗套件单向失败（全 FP 零 FN）——与规则 baseline 的
    双向全败互补，证明对抗套件不是类别不平衡造成的。"""
    ev = AlertBenchEvaluator(suite=list(get_adversarial_suite()))
    res = [r for r in ev.evaluate_agent(AlwaysAlertAgent())
           if "error" not in r]
    n = len(res)
    fp = sum(1 for r in res if r["fp"])
    fn = sum(1 for r in res if r["fn"])
    # 对抗套件 16 例中 8 例真值为 False（FP 陷阱）→ always 全踩
    assert n == 16
    assert fp == 8 and fn == 0
    assert sum(r["accuracy"] for r in res) / n == 0.5


def test_never_agent_on_basic_fails_all_true_cases():
    ev = AlertBenchEvaluator(suite=list(get_alert_benchmark_suite()))
    res = [r for r in ev.evaluate_agent(NeverAlertAgent())
           if "error" not in r]
    n = len(res)
    fn = sum(1 for r in res if r["fn"])
    assert n == 40 and fn == sum(r["ground_truth"] for r in res)


def test_benchmark_health_section_renders():
    """体检段：判别力表格 + 平凡基线参照 + 历史锚定行。"""
    lines = _build_benchmark_health_section()
    text = "\n".join(lines)
    assert "基准体检" in text
    assert "对抗 16" in text
    assert "always" in text and "never" in text and "random" in text
    assert "暴露-动作层" in text
    assert "历史锚定" in text and "8/8" in text
    # 规则 baseline 在对抗套件上的双向全败应可见
    assert "| 对抗 16 | 0% |" in text
    # FP/FN 列：对抗套件规则 8/8
    assert " 8/8 |" in text
