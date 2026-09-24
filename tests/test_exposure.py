"""暴露与脆弱性层测试 — Phase B 扩展（docs/expansion-plan.md 层 2）。

核心断言：
1. 向后兼容：无 exposure 的场景行为完全不变
2. 暴露分级查表（E0-E3）单元行为
3. 动作矩阵不变量：暴露不创造风险（物理 NO × 任意暴露 → monitor）
4. 暴露套件：同一物理观测 × 不同暴露 → 不同动作真值（配对性）
5. 规则 baseline 动作决策：基础套件满分、暴露套件双向出错（判别力）
6. 暴露加权 V：高暴露漏报罚分更重；默认关闭时行为不变
7. SectorValueEvaluator：行业成本比覆盖 + 未知行业回退
"""
from __future__ import annotations

from earthbench.agents import MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.eval import SectorValueEvaluator, ValueEvaluator
from earthbench.models import ExposureProfile
from earthbench.scenarios import (
    get_alert_benchmark_suite,
    get_exposure_suite,
    infer_action_ground_truth,
    infer_exposure_class,
    infer_flood_ground_truth,
)


def _obs(variable: str, value: float, timestamp: str = "2026-07-26T12:00:00+08:00") -> dict:
    return {
        "source": "CMA",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": timestamp,
        "confidence": 0.95,
    }


def _profile(**kwargs) -> ExposureProfile:
    return ExposureProfile(**kwargs)


# ---------------------------------------------------------------------------
# 向后兼容
# ---------------------------------------------------------------------------


def test_scenario_context_exposure_optional():
    """不提供 exposure 的场景（全部既有 40 用例）行为不变。"""
    suite = get_alert_benchmark_suite()
    for item in suite:
        assert item.get("exposure") is None
    b = AlertBenchEvaluator()  # 默认基础套件
    for tc in b.test_cases:
        assert tc.exposure is None
        assert tc.to_context().exposure is None


# ---------------------------------------------------------------------------
# 暴露分级查表
# ---------------------------------------------------------------------------


def test_exposure_class_table():
    assert infer_exposure_class(None)[0] == "E0"

    # 关键基础设施 → E3（无条件升高）
    assert infer_exposure_class(_profile(critical_infrastructure=["hospital"]))[0] == "E3"
    # 人口 high → E3
    assert infer_exposure_class(_profile(population_density_class="high"))[0] == "E3"

    # E2 触发条件
    assert infer_exposure_class(_profile(population_density_class="mid"))[0] == "E2"
    assert infer_exposure_class(_profile(land_use="urban"))[0] == "E2"
    assert infer_exposure_class(_profile(vulnerable_group_ratio=0.4))[0] == "E2"
    assert infer_exposure_class(_profile(outdoor_activity_level="high"))[0] == "E2"

    # 低暴露
    assert infer_exposure_class(
        _profile(population_density_class="low", land_use="rural")
    )[0] == "E1"
    assert infer_exposure_class(
        _profile(population_density_class="none", land_use="forest")
    )[0] == "E1"


# ---------------------------------------------------------------------------
# 动作矩阵单元行为
# ---------------------------------------------------------------------------


def _flood_obs(rain: float, soil: float, water: float) -> list[dict]:
    return [
        _obs("rainfall_24h", rain),
        _obs("soil_moisture", soil),
        _obs("water_level", water),
    ]


# 物理观测档：rain 120 → GT True 0.95（severe）；rain 60+soil 0.8 → True 0.85（moderate）
_SEVERE = _flood_obs(120.0, 0.60, 3.0)
_MODERATE = _flood_obs(60.0, 0.80, 3.0)
_NONE = _flood_obs(0.0, 0.20, 2.0)


def test_action_matrix_invariants():
    """暴露不创造风险：物理 NO × 任意暴露（含 E3）→ monitor。"""
    for exp in [None, _profile(population_density_class="high",
                               critical_infrastructure=["subway"])]:
        action, detail = infer_action_ground_truth(
            infer_flood_ground_truth, _NONE, exp
        )
        assert action == "monitor"
        assert "暴露不创造风险" in detail["matrix"]


def test_action_matrix_escalation():
    """物理 YES 时暴露调制动作：moderate E1→alert / E3→dispatch；severe E2→dispatch。"""
    # moderate + E1 → alert
    a1, _ = infer_action_ground_truth(
        infer_flood_ground_truth, _MODERATE, _profile(population_density_class="none", land_use="forest")
    )
    assert a1 == "alert"
    # moderate + E3 → dispatch
    a2, _ = infer_action_ground_truth(
        infer_flood_ground_truth, _MODERATE, _profile(population_density_class="high")
    )
    assert a2 == "dispatch"
    # severe + E2 → dispatch
    a3, _ = infer_action_ground_truth(
        infer_flood_ground_truth, _SEVERE, _profile(population_density_class="mid")
    )
    assert a3 == "dispatch"
    # severe + E1 → alert（林区极端事件仍只需预警）
    a4, _ = infer_action_ground_truth(
        infer_flood_ground_truth, _SEVERE, _profile(population_density_class="none", land_use="forest")
    )
    assert a4 == "alert"


# ---------------------------------------------------------------------------
# 暴露套件
# ---------------------------------------------------------------------------


def test_exposure_suite_shape():
    suite = get_exposure_suite()
    assert len(suite) == 10
    for c in suite:
        assert c.get("_gt_fn") is not None
        assert c.get("exposure") is not None
        assert c["action_truth"] in ("monitor", "alert", "dispatch")
        assert isinstance(c["ground_truth"], bool)


def test_exposure_suite_paired_same_obs_different_action():
    """配对用例：同一物理观测、不同暴露 → 动作真值不同（暴露调制动作）。"""
    suite = {c["case_id"]: c for c in get_exposure_suite()}
    pairs = [
        ("exp-flood-urban-E3-dispatch", "exp-flood-forest-E1-alert"),
        ("exp-heat-hospital-E3-dispatch", "exp-heat-suburb-E1-alert"),
        ("exp-landslide-school-E3-dispatch", "exp-landslide-village-E2-alert"),
    ]
    for hi, lo in pairs:
        assert suite[hi]["observations"] == suite[lo]["observations"]
        assert suite[hi]["action_truth"] == "dispatch"
        assert suite[lo]["action_truth"] == "alert"
        # 物理真值相同：暴露不改写物理
        assert suite[hi]["ground_truth"] == suite[lo]["ground_truth"]


def test_exposure_suite_action_truth_matches_derivation():
    b = AlertBenchEvaluator(suite=get_exposure_suite())
    for tc in b.test_cases:
        action, _ = infer_action_ground_truth(
            b._gt_fn_of(tc), tc.observations, tc.exposure
        )
        item = next(i for i in b.raw_suite if i["case_id"] == tc.case_id)
        assert action == item["action_truth"], tc.case_id


def test_action_gt_matches_documented_physical_truth():
    """动作推导内的物理判定与文档物理真值一致（真值独立链路）。"""
    b = AlertBenchEvaluator(suite=get_exposure_suite())
    assert b.gt_divergences == []


# ---------------------------------------------------------------------------
# 规则 baseline 动作决策：判别力
# ---------------------------------------------------------------------------


def test_baseline_action_perfect_on_default_suite():
    """无暴露画像（E0）时 baseline 动作与真值一致——既有结论延伸。"""
    b = AlertBenchEvaluator()  # 40 基础用例，全部 E0
    results = b.evaluate_action_agent(MultiAlertAgent())
    acc = [r for r in results if "action_accuracy" in r]
    assert len(acc) == 40
    assert all(r["action_accuracy"] == 1 for r in acc)


def test_baseline_action_discriminative_on_exposure_suite():
    """暴露套件上 baseline 动作双向出错：欠响应 + 过响应各一。"""
    b = AlertBenchEvaluator(suite=get_exposure_suite())
    results = b.evaluate_action_agent(MultiAlertAgent())
    ok = [r for r in results if "action_accuracy" in r]
    assert len(ok) == 10
    wrong = [r for r in ok if r["action_accuracy"] == 0]
    # 恰好两个陷阱出错
    assert len(wrong) == 2
    under = [r for r in wrong if r["action_predicted"] == "alert" and r["action_gt"] == "dispatch"]
    over = [r for r in wrong if r["action_predicted"] == "dispatch" and r["action_gt"] == "alert"]
    assert len(under) == 1  # exp-typhoon-severe-metro-E2-dispatch（欠响应）
    assert len(over) == 1   # exp-typhoon-moderate-urban-E2-overreach（过响应）


# ---------------------------------------------------------------------------
# 暴露加权 V 与分行业 V
# ---------------------------------------------------------------------------


def _v_rows(exposure_class: str | None) -> list[dict]:
    row = {
        "predicted": False,  # 漏报
        "ground_truth": True,
        "category": "flood",
    }
    if exposure_class is not None:
        row["exposure_class"] = exposure_class
    return [dict(row)]


def test_value_evaluator_exposure_weighting():
    """开启暴露加权后：E3 漏报比 E1 漏报更贵（expense 更高 / V 更低）。"""
    ve = ValueEvaluator(exposure_weighted=True)
    low = ve.score(_v_rows("E1"))
    high = ve.score(_v_rows("E3"))
    assert high["expense_agent"] > low["expense_agent"]

    # 未开启加权时：两者完全一致（默认行为不变）
    ve0 = ValueEvaluator()
    assert ve0.score(_v_rows("E1"))["expense_agent"] == ve0.score(_v_rows("E3"))["expense_agent"]
    # 无暴露分级的行：即使开启加权也按 E0（权重 1.0）处理
    assert ve.score(_v_rows(None))["expense_agent"] == low["expense_agent"]


def test_sector_value_evaluator():
    """行业成本比覆盖默认值；未知行业回退；接口兼容。"""
    rows = [{"predicted": True, "ground_truth": True, "category": "heat"}]
    health = SectorValueEvaluator(sector="health").score(rows)
    transport = SectorValueEvaluator(sector="transport").score(rows)
    # health 行业 heat 成本比 0.15 < 默认 0.3 → 触发支出更低
    assert health["cost_ratio"]["heat"] == 0.15
    assert transport["cost_ratio"]["heat"] == 0.3  # transport 未覆盖 heat → 默认
    assert health["expense_agent"] < transport["expense_agent"]

    unknown = SectorValueEvaluator(sector="nonexistent").score(rows)
    assert unknown["sector"] == "nonexistent"
    assert unknown["cost_ratio"]["heat"] == 0.3  # 回退默认
