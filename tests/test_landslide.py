"""Landslide（滑坡/泥石流）灾种测试 — Phase A1 扩展（docs/expansion-plan.md）。

核心断言：
1. 查表真值函数与场景标签一致（真值独立性，与 test_adversarial 同源纪律）
2. 规则 baseline 在 5 个基础用例上 100%（维持「基础套件 baseline 满分」结论）
3. 规则 baseline 在 2 个对抗用例上双向失败（误报 + 漏报，判别力来源）
4. MultiAlertAgent 正确路由 landslide 类别
5. 三通道 + 退坡抑制的单元行为
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
    infer_landslide_ground_truth,
)


def _obs(variable: str, value: float, timestamp: str = "2026-07-11T12:00:00+08:00") -> dict:
    return {
        "source": "Station",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": timestamp,
        "confidence": 0.95,
    }


def _landslide_basic_cases() -> list[dict]:
    return [c for c in get_alert_benchmark_suite() if c["category"] == "landslide"]


def _landslide_adversarial_cases() -> list[dict]:
    return [c for c in get_adversarial_suite() if c["category"] == "landslide"]


# ---------------------------------------------------------------------------
# 套件形态
# ---------------------------------------------------------------------------


def test_landslide_suite_shape():
    cases = _landslide_basic_cases()
    assert len(cases) == 5
    # 难度覆盖 L1-L4，真值双向
    assert {c["difficulty"] for c in cases} == {"L1", "L2", "L3", "L4"}
    assert {c["ground_truth"] for c in cases} == {True, False}
    for c in cases:
        assert c.get("_gt_fn") is infer_landslide_ground_truth
        assert len(c["observations"]) >= 3


def test_adversarial_landslide_shape():
    cases = _landslide_adversarial_cases()
    assert len(cases) == 2
    gts = {c["ground_truth"] for c in cases}
    assert gts == {True, False}  # 误报陷阱 + 漏报陷阱各一


# ---------------------------------------------------------------------------
# 真值独立性与单元行为
# ---------------------------------------------------------------------------


def test_landslide_ground_truth_matches_labels():
    for c in _landslide_basic_cases() + _landslide_adversarial_cases():
        decision, _score, _explanation = c["_gt_fn"](c["observations"])
        assert decision == c["ground_truth"], c["case_id"]


def test_landslide_channel_intensity():
    """激发雨强通道：高易发 + 短时强降雨 → 预警。"""
    decision, score, exp = infer_landslide_ground_truth(
        [
            _obs("rainfall_1h", 30.0),
            _obs("susceptibility", 3.0),
        ]
    )
    assert decision is True
    assert score >= 0.9
    assert "激发雨强" in exp["standard"]


def test_landslide_channel_antecedent_saturation():
    """累积饱和通道：当日无激发雨亦可触发（雨停 ≠ 风险停）。"""
    decision, score, exp = infer_landslide_ground_truth(
        [
            _obs("rainfall_1h", 2.0),
            _obs("rainfall_24h", 8.0),
            _obs("effective_rainfall_3d", 120.0),
            _obs("soil_moisture", 0.86),
            _obs("susceptibility", 3.0),
        ]
    )
    assert decision is True
    assert score >= 0.8
    assert "累积饱和" in exp["standard"]


def test_landslide_channel_compound():
    """复合通道：暴雨线 + 土壤偏高 + 中易发 → 预警（易发性缺一不可）。"""
    decision, score, _ = infer_landslide_ground_truth(
        [
            _obs("rainfall_1h", 10.0),
            _obs("rainfall_24h", 55.0),
            _obs("soil_moisture", 0.76),
            _obs("effective_rainfall_3d", 70.0),
            _obs("susceptibility", 2.0),
        ]
    )
    assert decision is True
    assert score >= 0.75

    # 易发性 low（1/3）时同一降雨不触发复合通道
    decision_low, _, _ = infer_landslide_ground_truth(
        [
            _obs("rainfall_1h", 10.0),
            _obs("rainfall_24h", 55.0),
            _obs("soil_moisture", 0.76),
            _obs("effective_rainfall_3d", 70.0),
            _obs("susceptibility", 1.0),
        ]
    )
    assert decision_low is False


def test_landslide_suppression_quiet():
    """退坡抑制：雨停（<1mm）且土壤已排水（<0.50）→ 强制不预警。"""
    decision, score, exp = infer_landslide_ground_truth(
        [
            _obs("rainfall_1h", 0.2),
            _obs("rainfall_24h", 0.5),
            _obs("effective_rainfall_3d", 80.0),  # 前期累积仍偏高
            _obs("soil_moisture", 0.40),
            _obs("susceptibility", 3.0),
        ]
    )
    assert decision is False
    assert score <= 0.2
    assert "退坡" in exp["standard"]


def test_landslide_no_rainfall_no_basis():
    """无降雨观测：独立标准无依据 → 不预警。"""
    decision, score, exp = infer_landslide_ground_truth(
        [
            _obs("soil_moisture", 0.9),
            _obs("susceptibility", 3.0),
        ]
    )
    assert decision is False
    assert score == 0.0
    assert "reason" in exp


# ---------------------------------------------------------------------------
# 规则 baseline：基础满分 + 对抗双向失败
# ---------------------------------------------------------------------------


def test_rule_baseline_perfect_on_landslide_basic():
    """线性加权 baseline 在滑坡基础用例上 100%（既有结论向新灾种延伸）。"""
    b = AlertBenchEvaluator(suite=_landslide_basic_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc == 1.0


def test_rule_baseline_fails_adversarial_both_directions():
    """对抗用例上 baseline 双向失败：误报（低易发+雨强达标）+ 漏报（前期饱和被稀释）。"""
    b = AlertBenchEvaluator(suite=_landslide_adversarial_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    fps = [r for r in results if r["predicted"] and not r["ground_truth"]]
    fns = [r for r in results if not r["predicted"] and r["ground_truth"]]
    assert len(fps) == 1  # adv-landslide-intensity-low-suscept
    assert len(fns) == 1  # adv-landslide-antecedent-diluted
    assert sum(r["accuracy"] for r in results) / len(results) == 0.0


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_multi_agent_routes_landslide():
    ctx = ScenarioContext(
        category=ScenarioCategory.LANDSLIDE,
        template=DecisionTemplate.ALERT,
        observations=[
            Observation(
                source="Station",
                variable="rainfall_1h",
                value=30.0,
                unit="mm",
                timestamp="2026-07-11T12:00:00+08:00",
            ),
            Observation(
                source="GeoSurvey",
                variable="susceptibility",
                value=3.0,
                unit="3=high/2=mid/1=low",
                timestamp="2026-07-11T12:00:00+08:00",
            ),
            Observation(
                source="Sensor",
                variable="soil_moisture",
                value=0.6,
                unit="",
                timestamp="2026-07-11T12:00:00+08:00",
            ),
        ],
        region="Wenchuan-Sichuan",
    )
    out = MultiAlertAgent().decide(ctx)
    assert out.rationale.startswith("LandslideAlert")
    assert out.decision is True


def test_scenario_category_enum_landslide():
    assert ScenarioCategory.from_string("landslide") is ScenarioCategory.LANDSLIDE
