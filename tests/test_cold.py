"""ColdWave（寒潮/冰冻）灾种测试 — Phase B1 扩展（docs/expansion-plan.md）。

核心断言：
1. 查表真值函数与场景标签一致（真值独立性，与 test_adversarial 同源纪律）
2. 规则 baseline 在 5 个基础用例上 100%（维持「基础套件 baseline 满分」结论）
3. 规则 baseline 在 2 个对抗用例上双向失败（误报 + 漏报，判别力来源）
4. MultiAlertAgent 正确路由 cold 类别
5. 「多窗口降幅 OR × 日最低」AND 结构与序列推导降幅的单元行为
"""
from __future__ import annotations

from earthbench.agents import MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.models import (
    DecisionTemplate,
    Observation,
    ScenarioCategory,
    ScenarioContext,
)
from earthbench.scenarios import (
    get_adversarial_suite,
    get_alert_benchmark_suite,
    infer_cold_ground_truth,
)


def _obs(variable: str, value: float, timestamp: str = "2026-01-15T12:00:00+08:00") -> dict:
    return {
        "source": "CMA",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": timestamp,
        "confidence": 0.95,
    }


def _cold_basic_cases() -> list[dict]:
    return [c for c in get_alert_benchmark_suite() if c["category"] == "cold"]


def _cold_adversarial_cases() -> list[dict]:
    return [c for c in get_adversarial_suite() if c["category"] == "cold"]


# ---------------------------------------------------------------------------
# 套件形态
# ---------------------------------------------------------------------------


def test_cold_suite_shape():
    cases = _cold_basic_cases()
    assert len(cases) == 5
    # 难度覆盖 L1-L4，真值双向
    assert {c["difficulty"] for c in cases} == {"L1", "L2", "L3", "L4"}
    assert {c["ground_truth"] for c in cases} == {True, False}
    for c in cases:
        assert c.get("_gt_fn") is infer_cold_ground_truth
        assert len(c["observations"]) >= 3


def test_adversarial_cold_shape():
    cases = _cold_adversarial_cases()
    assert len(cases) == 2
    gts = {c["ground_truth"] for c in cases}
    assert gts == {True, False}  # 误报陷阱 + 漏报陷阱各一


# ---------------------------------------------------------------------------
# 真值独立性与单元行为
# ---------------------------------------------------------------------------


def test_cold_ground_truth_matches_labels():
    for c in _cold_basic_cases() + _cold_adversarial_cases():
        decision, _score, _explanation = c["_gt_fn"](c["observations"])
        assert decision == c["ground_truth"], c["case_id"]


def test_cold_grading_tiers():
    """GB/T 20484-2017 四级体系顶档：多窗口 OR × 日最低 AND（单一寒潮档）。

    2017 版取消 2006 版的寒潮/强寒潮/特强寒潮三档，寒潮为四级冷空气
    体系（弱/较强/强冷空气/寒潮）顶档——本测试锁定该结构修复。
    """
    # 24h 窗口：降幅 9 + 极值 3 → 寒潮（顶档，0.95）
    d3, s3, e3 = infer_cold_ground_truth(
        [_obs("temperature_min", 3.0), _obs("temperature_drop_24h", 9.0)]
    )
    assert d3 is True and s3 == 0.95
    assert "寒潮" in e3["standard"] and "24h 降幅 9.0℃" in e3["detail"]

    # 剧烈过程：降幅 13 + 极值 -1 → 同一顶档（分级已取消，不再分档）
    d1, s1, e1 = infer_cold_ground_truth(
        [_obs("temperature_min", -1.0), _obs("temperature_drop_24h", 13.0)]
    )
    assert d1 is True and s1 == 0.95 and "寒潮" in e1["standard"]


def test_cold_multiday_window_or():
    """缓慢渗透型寒潮：24h 窗口不达线、48h/72h 窗口达线 → 多窗口 OR 触发。

    这正是历史回填发现的口径问题（2016年1月寒潮沿海站 24h 不达线）的
    通道化修复：渐进下滑在更长窗口可被捕捉。
    """
    # 48h 窗口：三日渐进下滑，24h=6.5 < 8、48h=10.5 >= 10 → 触发
    obs = [
        _obs("temperature_min", 12.0, "2026-01-13T12:00:00+08:00"),
        _obs("temperature_min", 8.0, "2026-01-14T12:00:00+08:00"),
        _obs("temperature_min", 1.5, "2026-01-15T12:00:00+08:00"),
    ]
    decision, score, exp = infer_cold_ground_truth(obs)
    assert exp["drop_24h"] == 6.5 and exp["drop_48h"] == 10.5
    assert decision is True and score == 0.95
    assert "48h" in exp["detail"]

    # 72h 窗口：四日渐进下滑，24h=5 / 48h=9 均不达线、72h=13 >= 12 → 触发
    obs72 = [
        _obs("temperature_min", 14.0, "2026-01-12T12:00:00+08:00"),
        _obs("temperature_min", 10.0, "2026-01-13T12:00:00+08:00"),
        _obs("temperature_min", 6.0, "2026-01-14T12:00:00+08:00"),
        _obs("temperature_min", 1.0, "2026-01-15T12:00:00+08:00"),
    ]
    d72, s72, e72 = infer_cold_ground_truth(obs72)
    assert e72["drop_24h"] == 5.0 and e72["drop_48h"] == 9.0
    assert e72["drop_72h"] == 13.0
    assert d72 is True and s72 == 0.95 and "72h" in e72["detail"]

    # 对照：极值低但各窗口降幅均未达线（渐进放缓）→ 不预警（AND 纪律）
    obs_no = [
        _obs("temperature_min", 8.0, "2026-01-13T12:00:00+08:00"),
        _obs("temperature_min", 5.0, "2026-01-14T12:00:00+08:00"),
        _obs("temperature_min", 2.0, "2026-01-15T12:00:00+08:00"),
    ]
    dn, _sn, en = infer_cold_ground_truth(obs_no)
    assert en["drop_24h"] == 3.0 and en["drop_48h"] == 6.0
    assert dn is False and "常态低温" in en["detail"]


def test_cold_and_structure_missing_one():
    """AND 双条件：单条件不成立即不预警。"""
    # 降幅达线但绝对温不低（基础温度高）
    d1, _, e1 = infer_cold_ground_truth(
        [_obs("temperature_min", 8.0), _obs("temperature_drop_24h", 9.0)]
    )
    assert d1 is False and "绝对温度不低" in e1["detail"]

    # 绝对温低但无降幅（北方常态低温）
    d2, _, e2 = infer_cold_ground_truth(
        [_obs("temperature_min", -12.0), _obs("temperature_drop_24h", 3.0)]
    )
    assert d2 is False and "常态低温" in e2["detail"]


def test_cold_drop_derived_from_series():
    """降幅缺失时由 temperature_min 序列最近两日差值推导。"""
    decision, score, exp = infer_cold_ground_truth(
        [
            _obs("temperature_min", 12.0, "2026-01-14T12:00:00+08:00"),
            _obs("temperature_min", 3.0, "2026-01-15T12:00:00+08:00"),
        ]
    )
    assert exp["drop_source"] == "derived"
    assert exp["drop_24h"] == 9.0
    # 推导降幅 9℃ + 极值 3℃ → 寒潮
    assert decision is True and score >= 0.7


def test_cold_warming_clipped():
    """升温序列（负降幅）取 0，不触发。"""
    decision, _, exp = infer_cold_ground_truth(
        [
            _obs("temperature_min", 0.0, "2026-01-14T12:00:00+08:00"),
            _obs("temperature_min", 8.0, "2026-01-15T12:00:00+08:00"),
        ]
    )
    assert decision is False
    assert exp["drop_24h"] == 0.0


def test_cold_no_observation_no_basis():
    """无温况观测：独立标准无依据 → 不预警。"""
    decision, score, exp = infer_cold_ground_truth(
        [_obs("wind_speed", 15.0)]
    )
    assert decision is False
    assert score == 0.0
    assert "reason" in exp


# ---------------------------------------------------------------------------
# 规则 baseline：基础满分 + 对抗双向失败
# ---------------------------------------------------------------------------


def test_rule_baseline_perfect_on_cold_basic():
    """线性加权 baseline 在寒潮基础用例上 100%（既有结论向新灾种延伸）。"""
    b = AlertBenchEvaluator(suite=_cold_basic_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc == 1.0


def test_rule_baseline_fails_adversarial_both_directions():
    """对抗用例上 baseline 双向失败：误报（常态低温无降幅）+ 漏报（双线刚过被稀释）。"""
    b = AlertBenchEvaluator(suite=_cold_adversarial_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    fps = [r for r in results if r["predicted"] and not r["ground_truth"]]
    fns = [r for r in results if not r["predicted"] and r["ground_truth"]]
    assert len(fps) == 1  # adv-cold-chronic-low-no-drop
    assert len(fns) == 1  # adv-cold-both-just-met
    assert sum(r["accuracy"] for r in results) / len(results) == 0.0


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_multi_agent_routes_cold():
    ctx = ScenarioContext(
        category=ScenarioCategory.COLD,
        template=DecisionTemplate.ALERT,
        observations=[
            Observation(
                source="CMA",
                variable="temperature_min",
                value=-6.0,
                unit="°C",
                timestamp="2026-01-15T12:00:00+08:00",
            ),
            Observation(
                source="CMA",
                variable="temperature_drop_24h",
                value=14.0,
                unit="°C",
                timestamp="2026-01-15T12:00:00+08:00",
            ),
            Observation(
                source="Station",
                variable="wind_speed",
                value=10.0,
                unit="m/s",
                timestamp="2026-01-15T12:00:00+08:00",
            ),
        ],
        region="Nanjing-Yangtze",
    )
    out = MultiAlertAgent().decide(ctx)
    assert out.rationale.startswith("ColdWaveAlert")
    assert out.decision is True


def test_scenario_category_enum_cold():
    assert ScenarioCategory.from_string("cold") is ScenarioCategory.COLD
