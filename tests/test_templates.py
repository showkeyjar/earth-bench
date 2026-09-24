"""决策模板闭环测试 — Phase C 扩展（docs/expansion-plan.md §6）。

覆盖 DISPATCH / UPGRADE / CLOSE / RECOVER 四个模板：
1. 持续时间原语 _persist_for 的单元行为（窗口含边界点 / 采样点不足）
2. 各模板真值函数与用例标签一致（真值独立性）
3. 规则 baseline 在模板套件上双向失败（误报 + 漏报），且非陷阱用例通过
4. MultiAlertAgent 按模板路由（decision 由 context.template 决定）
   的符号不变：跑基础套件性能不回归
"""
from __future__ import annotations

from earthbench.agents import MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.models import (
    DecisionTemplate,
    ExposureProfile,
    Observation,
    ScenarioCategory,
    ScenarioContext,
)
from earthbench.scenarios import (
    _persist_for,
    get_close_suite,
    get_dispatch_suite,
    get_recover_suite,
    get_upgrade_suite,
    infer_close_ground_truth,
    infer_dispatch_ground_truth,
    infer_recover_ground_truth,
    infer_upgrade_ground_truth,
)


def _obs(variable: str, value: float, ts: str) -> dict:
    return {
        "source": "CMA",
        "variable": variable,
        "value": value,
        "unit": "",
        "timestamp": ts,
        "confidence": 0.95,
    }


# ---------------------------------------------------------------------------
# _persist_for 持续时间原语
# ---------------------------------------------------------------------------


def test_persist_for_sustained_and_rebound():
    now = "2026-06-20T12:00:00+08:00"
    sustained = [
        _obs("water_level", 4.4, "2026-06-20T06:00:00+08:00"),
        _obs("water_level", 4.3, "2026-06-20T09:00:00+08:00"),
        _obs("water_level", 4.2, now),
    ]
    ok, detail = _persist_for(sustained, "water_level", lambda v: v <= 4.5, 6.0)
    assert ok is True
    assert detail["points"] == 3

    rebound = [
        _obs("water_level", 4.7, "2026-06-20T06:00:00+08:00"),
        _obs("water_level", 4.4, "2026-06-20T09:00:00+08:00"),
        _obs("water_level", 4.3, now),
    ]
    ok, _ = _persist_for(rebound, "water_level", lambda v: v <= 4.5, 6.0)
    assert ok is False  # 6h 前曾有 4.7 反弹 → 不算持续


def test_persist_for_insufficient_points():
    single = [
        _obs("water_level", 4.2, "2026-06-20T11:00:00+08:00"),
    ]
    ok, detail = _persist_for(single, "water_level", lambda v: v <= 4.5, 6.0)
    assert ok is False
    assert "采样点" in detail["reason"]


# ---------------------------------------------------------------------------
# 各模板真值与用例标签一致性
# ---------------------------------------------------------------------------


def _check_suite(get_suite, infer_fn):
    suite = get_suite()
    for item in suite:
        kwargs = {}
        if item.get("exposure") is not None:
            kwargs["exposure"] = ExposureProfile(**item["exposure"])
        decision, _score, _exp = infer_fn(item["observations"], **kwargs)
        assert decision == item["ground_truth"], item["case_id"]
    return suite


def test_dispatch_suite_truth_consistent():
    suite = _check_suite(get_dispatch_suite, infer_dispatch_ground_truth)
    assert len(suite) == 4


def test_dispatch_typhoon_branch_defined_and_correct():
    """锁定 dispatch 台风分支（此前 DISPATCH_TYPHOON_WIND 未定义，一跑即 NameError，套件无台风用例未暴露）。"""
    ts = "2026-06-15T12:00:00+08:00"
    obs = [
        _obs("wind_speed", 26.0, ts),
        _obs("wind_gust", 33.0, ts),
        _obs("rainfall_24h", 60.0, ts),
    ]
    e3 = ExposureProfile(critical_infrastructure=["port_crane"])
    decision, score, exp = infer_dispatch_ground_truth(obs, exposure=e3)
    assert decision is True
    assert score == 0.90
    assert "防风预置" in exp["standard"]

    # 风速够但非 E3 关键设施 → 不预置（AND 缺一）
    e1 = ExposureProfile(population_density_class="none", land_use="forest")
    decision, _, _ = infer_dispatch_ground_truth(obs, exposure=e1)
    assert decision is False

    # 风速不足（< 24.5）即使 E3 → 不预置
    weak = [_obs("wind_speed", 20.0, ts)]
    decision, _, _ = infer_dispatch_ground_truth(weak, exposure=e3)
    assert decision is False


def test_upgrade_suite_truth_consistent():
    suite = _check_suite(get_upgrade_suite, infer_upgrade_ground_truth)
    assert len(suite) == 4


def test_close_suite_truth_consistent():
    suite = _check_suite(get_close_suite, infer_close_ground_truth)
    assert len(suite) == 5


def test_recover_suite_truth_consistent():
    suite = _check_suite(get_recover_suite, infer_recover_ground_truth)
    assert len(suite) == 4


# ---------------------------------------------------------------------------
# 基准层 gt_divergences：硬编码标签与独立推导无分歧
# ---------------------------------------------------------------------------


def _no_divergence(get_suite):
    b = AlertBenchEvaluator(suite=get_suite())
    assert b.gt_divergences == []


def test_template_suites_no_gt_divergence():
    for get_suite in [
        get_dispatch_suite, get_upgrade_suite, get_close_suite, get_recover_suite,
    ]:
        _no_divergence(get_suite)


# ---------------------------------------------------------------------------
# 规则 baseline 判别力：基础性能不回归 + 模板套件双向失败
# ---------------------------------------------------------------------------


def test_baseline_alert_still_perfect():
    """加入模板路由后，40 个 ALERT 基础用例的 baseline 性能不回归（仍满分）。"""
    b = AlertBenchEvaluator()
    results = b.evaluate_agent(MultiAlertAgent())
    assert len(results) == 40
    assert all(r["accuracy"] == 1.0 for r in results)


def _baseline_suite_results(get_suite):
    b = AlertBenchEvaluator(suite=get_suite())
    return b.evaluate_agent(MultiAlertAgent())


def test_baseline_template_suites_discriminative():
    """四个模板套件各自：baseline 非 100% 且误报/漏报双向成立。"""
    all_fp = []
    all_fn = []
    for get_suite in [
        get_dispatch_suite, get_upgrade_suite, get_close_suite, get_recover_suite,
    ]:
        results = _baseline_suite_results(get_suite)
        ok = [r for r in results if "error" not in r]
        acc = sum(r["accuracy"] for r in ok) / len(ok)
        assert acc < 1.0, f"{get_suite.__name__} baseline 应非满分"
        fps = [r for r in ok if r["predicted"] and not r["ground_truth"]]
        fns = [r for r in ok if not r["predicted"] and r["ground_truth"]]
        all_fp.extend(fps)
        all_fn.extend(fns)
    assert len(all_fp) >= 1
    assert len(all_fn) >= 1


def test_recover_premature_trap_specifically():
    """过早恢复陷阱：rec-flood-premature 必须被 baseline 误判为「恢复」（FP）。"""
    results = _baseline_suite_results(get_recover_suite)
    by_id = {r["case_id"]: r for r in results if "error" not in r}
    # 陷阱：水位 6h 前仍 4.7，真值 False；baseline「雨停即恢复」误判 True
    assert by_id["rec-flood-premature"]["predicted"] is True
    assert by_id["rec-flood-premature"]["ground_truth"] is False
    # FN：路温已持续解冻，真值 True；baseline 因残留降雪量仍不解除
    assert by_id["rec-snow-thawed"]["predicted"] is False
    assert by_id["rec-snow-thawed"]["ground_truth"] is True


# ---------------------------------------------------------------------------
# 模板路由
# ---------------------------------------------------------------------------


def test_multi_agent_routes_by_template():
    agent = MultiAlertAgent()

    # DISPATCH 模板：手工构造高危险 + 高暴露（应判 True）
    ctx = ScenarioContext(
        category=ScenarioCategory.FIRE,
        template=DecisionTemplate.DISPATCH,
        region="Xiangshan-Beijing",
        exposure=ExposureProfile(critical_infrastructure=["scenic_area"],
                                 outdoor_activity_level="high"),
        observations=[
            Observation(source="ECMWF", variable="FWI", value=45.0, unit="",
                        timestamp="2026-06-15T12:00:00+08:00"),
            Observation(source="Station", variable="humidity", value=25.0, unit="%",
                        timestamp="2026-06-15T12:00:00+08:00"),
            Observation(source="Station", variable="wind_speed", value=8.0, unit="m/s",
                        timestamp="2026-06-15T12:00:00+08:00"),
            Observation(source="Station", variable="temperature", value=30.0, unit="°C",
                        timestamp="2026-06-15T12:00:00+08:00"),
        ],
    )
    out = agent.decide(ctx)
    assert out.rationale.startswith("Dispatch")
    assert out.decision is True  # conf 0.622 >= 0.55

    # ALERT 模板仍路由到 FireAlertAgent
    ctx_alert = ctx.model_copy(update={"template": DecisionTemplate.ALERT})
    out_alert = agent.decide(ctx_alert)
    assert out_alert.rationale.startswith("FireAlert")
    assert out_alert.decision is True