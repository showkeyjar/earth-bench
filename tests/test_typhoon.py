"""Typhoon（台风/大风）灾种测试 — Phase A2 扩展（docs/expansion-plan.md）。

核心断言：
1. 查表真值函数与场景标签一致（真值独立性，与 test_adversarial 同源纪律）
2. 规则 baseline 在 5 个基础用例上 100%（维持「基础套件 baseline 满分」结论）
3. 规则 baseline 在 2 个对抗用例上双向失败（误报 + 漏报，判别力来源）
4. MultiAlertAgent 正确路由 typhoon 类别
5. 四级风档 / 阵风通道 / 风雨耦合 AND 的单元行为
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
    infer_typhoon_ground_truth,
)


def _obs(variable: str, value: float, timestamp: str = "2026-07-26T12:00:00+08:00") -> dict:
    return {
        "source": "Station",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": timestamp,
        "confidence": 0.95,
    }


def _typhoon_basic_cases() -> list[dict]:
    return [c for c in get_alert_benchmark_suite() if c["category"] == "typhoon"]


def _typhoon_adversarial_cases() -> list[dict]:
    return [c for c in get_adversarial_suite() if c["category"] == "typhoon"]


# ---------------------------------------------------------------------------
# 套件形态
# ---------------------------------------------------------------------------


def test_typhoon_suite_shape():
    cases = _typhoon_basic_cases()
    assert len(cases) == 5
    # 难度覆盖 L1-L4，真值双向
    assert {c["difficulty"] for c in cases} == {"L1", "L2", "L3", "L4"}
    assert {c["ground_truth"] for c in cases} == {True, False}
    for c in cases:
        assert c.get("_gt_fn") is infer_typhoon_ground_truth
        assert len(c["observations"]) >= 3


def test_adversarial_typhoon_shape():
    cases = _typhoon_adversarial_cases()
    assert len(cases) == 2
    gts = {c["ground_truth"] for c in cases}
    assert gts == {True, False}  # 误报陷阱 + 漏报陷阱各一


# ---------------------------------------------------------------------------
# 真值独立性与单元行为
# ---------------------------------------------------------------------------


def test_typhoon_ground_truth_matches_labels():
    for c in _typhoon_basic_cases() + _typhoon_adversarial_cases():
        decision, _score, _explanation = c["_gt_fn"](c["observations"])
        assert decision == c["ground_truth"], c["case_id"]


def test_typhoon_channel_grade12():
    """台风级通道：日均风 >= 32.7（12 级）→ 最高档预警。"""
    decision, score, exp = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 33.0),
            _obs("wind_gust", 41.0),
        ]
    )
    assert decision is True
    assert score >= 0.9
    assert "台风级" in exp["standard"]


def test_typhoon_channel_gust_independent():
    """阵风通道：日均风不高（4 级）但阵风 >= 24.5（10 级）→ 预警。

    雷雨大风/飑线场景：人类安全口径（临设/塔吊线）独立于日均档判定。
    """
    decision, score, exp = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 6.0),
            _obs("wind_gust", 26.0),
        ]
    )
    assert decision is True
    assert score >= 0.8
    assert "阵风" in exp["standard"]


def test_typhoon_channel_grade8_mean():
    """8 级日均通道：>= 17.2（交通停运/户外作业停止线）→ 预警。"""
    decision, score, exp = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 18.0),
            _obs("wind_gust", 23.0),
        ]
    )
    assert decision is True
    assert score >= 0.75
    assert "热带风暴级" in exp["standard"]


def test_typhoon_channel_compound_and_structure():
    """风雨耦合通道：7 级 + 暴雨 AND 结构，缺一不可。"""
    # 两者齐备 → 预警
    decision, score, _ = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 15.0),
            _obs("rainfall_24h", 55.0),
        ]
    )
    assert decision is True
    assert score >= 0.65

    # 降雨差一线（48 < 50）→ 不预警
    decision_no_rain, _, _ = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 15.0),
            _obs("rainfall_24h", 48.0),
        ]
    )
    assert decision_no_rain is False

    # 风差一线（12 < 13.9）→ 不预警
    decision_no_wind, _, _ = infer_typhoon_ground_truth(
        [
            _obs("wind_speed", 12.0),
            _obs("rainfall_24h", 60.0),
        ]
    )
    assert decision_no_wind is False


def test_typhoon_no_wind_no_basis():
    """无风速观测：独立标准无依据 → 不预警。"""
    decision, score, exp = infer_typhoon_ground_truth(
        [
            _obs("rainfall_24h", 80.0),
        ]
    )
    assert decision is False
    assert score == 0.0
    assert "reason" in exp


# ---------------------------------------------------------------------------
# 规则 baseline：基础满分 + 对抗双向失败
# ---------------------------------------------------------------------------


def test_rule_baseline_perfect_on_typhoon_basic():
    """线性加权 baseline 在台风基础用例上 100%（既有结论向新灾种延伸）。"""
    b = AlertBenchEvaluator(suite=_typhoon_basic_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc == 1.0


def test_rule_baseline_fails_adversarial_both_directions():
    """对抗用例上 baseline 双向失败：误报（次阈值混合）+ 漏报（阵风稀释）。"""
    b = AlertBenchEvaluator(suite=_typhoon_adversarial_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    fps = [r for r in results if r["predicted"] and not r["ground_truth"]]
    fns = [r for r in results if not r["predicted"] and r["ground_truth"]]
    assert len(fps) == 1  # adv-typhoon-subthreshold-mix
    assert len(fns) == 1  # adv-typhoon-gust-diluted
    assert sum(r["accuracy"] for r in results) / len(results) == 0.0


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_multi_agent_routes_typhoon():
    ctx = ScenarioContext(
        category=ScenarioCategory.TYPHOON,
        template=DecisionTemplate.ALERT,
        observations=[
            Observation(
                source="Station",
                variable="wind_speed",
                value=35.0,
                unit="m/s",
                timestamp="2026-07-26T12:00:00+08:00",
            ),
            Observation(
                source="Station",
                variable="wind_gust",
                value=45.0,
                unit="m/s",
                timestamp="2026-07-26T12:00:00+08:00",
            ),
            Observation(
                source="CMA",
                variable="rainfall_24h",
                value=100.0,
                unit="mm",
                timestamp="2026-07-26T12:00:00+08:00",
            ),
        ],
        region="Wenzhou-Zhejiang",
    )
    out = MultiAlertAgent().decide(ctx)
    assert out.rationale.startswith("TyphoonAlert")
    assert out.decision is True


def test_scenario_category_enum_typhoon():
    assert ScenarioCategory.from_string("typhoon") is ScenarioCategory.TYPHOON
