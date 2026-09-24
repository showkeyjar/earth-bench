"""Snow（暴雪/道路结冰）灾种测试 — Phase B2 扩展（docs/expansion-plan.md）。

核心断言：
1. 查表真值函数与场景标签一致（真值独立性，与 test_adversarial 同源纪律）
2. 规则 baseline 在 5 个基础用例上 100%（维持「基础套件 baseline 满分」结论）
3. 规则 baseline 在 2 个对抗用例上双向失败（误报 + 漏报，判别力来源）
4. MultiAlertAgent 正确路由 snow 类别
5. 降雪三档分级 / 结冰 AND 复合 / 冻结掩膜代理的单元行为
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
    infer_snow_ground_truth,
)


def _obs(variable: str, value: float, timestamp: str = "2026-01-20T12:00:00+08:00") -> dict:
    return {
        "source": "CMA",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": timestamp,
        "confidence": 0.95,
    }


def _snow_basic_cases() -> list[dict]:
    return [c for c in get_alert_benchmark_suite() if c["category"] == "snow"]


def _snow_adversarial_cases() -> list[dict]:
    return [c for c in get_adversarial_suite() if c["category"] == "snow"]


# ---------------------------------------------------------------------------
# 套件形态
# ---------------------------------------------------------------------------


def test_snow_suite_shape():
    cases = _snow_basic_cases()
    assert len(cases) == 5
    # 难度覆盖 L1-L4，真值双向
    assert {c["difficulty"] for c in cases} == {"L1", "L2", "L3", "L4"}
    assert {c["ground_truth"] for c in cases} == {True, False}
    for c in cases:
        assert c.get("_gt_fn") is infer_snow_ground_truth
        assert len(c["observations"]) >= 3


def test_adversarial_snow_shape():
    cases = _snow_adversarial_cases()
    assert len(cases) == 2
    gts = {c["ground_truth"] for c in cases}
    assert gts == {True, False}  # 误报陷阱 + 漏报陷阱各一


# ---------------------------------------------------------------------------
# 真值独立性与单元行为
# ---------------------------------------------------------------------------


def test_snow_ground_truth_matches_labels():
    for c in _snow_basic_cases() + _snow_adversarial_cases():
        decision, _score, _explanation = c["_gt_fn"](c["observations"])
        assert decision == c["ground_truth"], c["case_id"]


def test_snow_grading_tiers():
    """三档分级：暴雪 / 大暴雪 / 特大暴雪。"""
    d1, s1, e1 = infer_snow_ground_truth(
        [_obs("snowfall_24h", 32.0), _obs("road_surface_temp", -3.0)]
    )
    assert d1 is True and s1 >= 0.9 and "特大暴雪" in e1["standard"]

    d2, s2, e2 = infer_snow_ground_truth(
        [_obs("snowfall_24h", 22.0), _obs("road_surface_temp", -3.0)]
    )
    assert d2 is True and s2 >= 0.85 and "大暴雪" in e2["standard"]

    d3, s3, e3 = infer_snow_ground_truth(
        [_obs("snowfall_24h", 11.0), _obs("road_surface_temp", -1.0)]
    )
    assert d3 is True and s3 >= 0.8 and "暴雪" in e3["standard"]


def test_snow_icing_compound_and_structure():
    """道路结冰复合：中雪档 × 路温 AND 结构，缺一不可。"""
    # 两者齐备（边界值含等号）→ 预警
    d1, s1, e1 = infer_snow_ground_truth(
        [_obs("snowfall_24h", 2.5), _obs("road_surface_temp", 0.0)]
    )
    assert d1 is True and s1 >= 0.65 and "结冰" in e1["standard"]

    # 降雪差一线（2.4 < 2.5）→ 不预警
    d2, _, _ = infer_snow_ground_truth(
        [_obs("snowfall_24h", 2.4), _obs("road_surface_temp", -5.0)]
    )
    assert d2 is False

    # 路温高于冰点 → 不预警
    d3, _, e3 = infer_snow_ground_truth(
        [_obs("snowfall_24h", 4.0), _obs("road_surface_temp", 1.5)]
    )
    assert d3 is False and "路面温度" in e3["detail"]


def test_snow_freeze_mask_proxy():
    """冻结掩膜代理：snowfall 缺失时由 rainfall × 气温推导。"""
    # 气温 >= 0.5℃：降水为雨，降雪量记 0 → 不预警（无论雨量多大）
    d1, _, e1 = infer_snow_ground_truth(
        [
            _obs("rainfall_24h", 40.0),
            _obs("temperature", 3.0),
            _obs("road_surface_temp", 3.0),
        ]
    )
    assert d1 is False
    assert e1["snow_source"] == "proxy"
    assert e1["snow_24h"] == 0.0

    # 气温 < 0.5℃：降水计为雪 → 按降雪量分档判定
    d2, s2, e2 = infer_snow_ground_truth(
        [_obs("rainfall_24h", 30.0), _obs("temperature", 0.2)]
    )
    assert d2 is True and s2 >= 0.9
    assert e2["snow_source"] == "proxy"
    assert e2["snow_24h"] == 30.0


def test_snow_no_observation_no_basis():
    """无降雪/降水观测：独立标准无依据 → 不预警。"""
    decision, score, exp = infer_snow_ground_truth(
        [_obs("road_surface_temp", -5.0), _obs("humidity", 80.0)]
    )
    assert decision is False
    assert score == 0.0
    assert "reason" in exp


# ---------------------------------------------------------------------------
# 规则 baseline：基础满分 + 对抗双向失败
# ---------------------------------------------------------------------------


def test_rule_baseline_perfect_on_snow_basic():
    """线性加权 baseline 在暴雪基础用例上 100%（既有结论向新灾种延伸）。"""
    b = AlertBenchEvaluator(suite=_snow_basic_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    acc = sum(r["accuracy"] for r in results) / len(results)
    assert acc == 1.0


def test_rule_baseline_fails_adversarial_both_directions():
    """对抗用例上 baseline 双向失败：误报（雨非雪）+ 漏报（结冰双线稀释）。"""
    b = AlertBenchEvaluator(suite=_snow_adversarial_cases())
    results = b.evaluate_agent(MultiAlertAgent())
    fps = [r for r in results if r["predicted"] and not r["ground_truth"]]
    fns = [r for r in results if not r["predicted"] and r["ground_truth"]]
    assert len(fps) == 1  # adv-snow-rain-not-snow
    assert len(fns) == 1  # adv-snow-icing-just-met
    assert sum(r["accuracy"] for r in results) / len(results) == 0.0


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_multi_agent_routes_snow():
    ctx = ScenarioContext(
        category=ScenarioCategory.SNOW,
        template=DecisionTemplate.ALERT,
        observations=[
            Observation(
                source="CMA",
                variable="snowfall_24h",
                value=35.0,
                unit="mm",
                timestamp="2026-01-20T12:00:00+08:00",
            ),
            Observation(
                source="Sensor",
                variable="road_surface_temp",
                value=-5.0,
                unit="°C",
                timestamp="2026-01-20T12:00:00+08:00",
            ),
            Observation(
                source="Station",
                variable="humidity",
                value=80.0,
                unit="%",
                timestamp="2026-01-20T12:00:00+08:00",
            ),
        ],
        region="Harbin-Heilongjiang",
    )
    out = MultiAlertAgent().decide(ctx)
    assert out.rationale.startswith("SnowAlert")
    assert out.decision is True


def test_scenario_category_enum_snow():
    assert ScenarioCategory.from_string("snow") is ScenarioCategory.SNOW
