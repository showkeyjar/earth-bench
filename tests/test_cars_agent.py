# -*- coding: utf-8 -*-
"""CARS 概率决策 Agent 与经济价值评分器测试。"""
from __future__ import annotations

from earthbench.cars_agent import CarsHeatAgent, CarsMultiAgent, CarsProbabilityTable
from earthbench.eval import ValueEvaluator
from earthbench.models import (
    DecisionOutput,
    Observation,
    ScenarioCategory,
    ScenarioContext,
    DecisionTemplate,
)


def _ctx(region: str, temp_max: list[float], dates: list[str]) -> ScenarioContext:
    obs = [
        Observation(source="CMA", variable="temperature_max", value=v,
                    unit="°C", timestamp=t)
        for v, t in zip(temp_max, dates)
    ]
    return ScenarioContext(
        category=ScenarioCategory.HEAT,
        template=DecisionTemplate.ALERT,
        observations=obs,
        region=region,
        horizon_hours=72,
    )


DATES = ["2026-07-09T12:00:00+08:00", "2026-07-10T12:00:00+08:00",
         "2026-07-11T12:00:00+08:00"]


def test_table_loads_and_looks_up():
    table = CarsProbabilityTable()
    rec = table.lookup("Chongqing-HotPotato", "2026-07-11")
    assert rec is not None and rec["covered"]
    assert 0.0 <= rec["p_exceed_2sigma"] <= 1.0
    assert table.lookup("Nowhere", "2026-07-11") is None


def test_agent_fires_on_chongqing_heat():
    agent = CarsHeatAgent()
    out = agent.decide(_ctx("Chongqing-HotPotato", [40.0, 38.0, 39.0], DATES))
    assert isinstance(out, DecisionOutput)
    assert out.decision is True
    assert out.evidence_summary["p_harm"] >= 0.3
    assert "湿球" in out.evidence_summary["harm_basis"]


def test_agent_suppresses_cool_kunming():
    agent = CarsHeatAgent()
    out = agent.decide(_ctx("Kunming-SpringCity", [26.0, 27.0, 25.0], DATES))
    # 昆明湿球危害概率 0：凉爽高原 → 不预警
    assert out.decision is False
    assert out.evidence_summary["p_harm"] == 0.0


def test_agent_fallback_outside_grid():
    agent = CarsHeatAgent()
    # 乌鲁木齐域外：回退到 37°C 阈值（按最新观测判定）
    out = agent.decide(_ctx("Urumqi-Xinjiang", [36.0, 35.0, 34.0], DATES))
    assert out.decision is False
    assert out.confidence <= 0.5
    out2 = agent.decide(_ctx("Urumqi-Xinjiang", [35.0, 36.0, 37.5], DATES))
    assert out2.decision is True


def test_multi_agent_routes_heat_to_cars():
    agent = CarsMultiAgent()
    out = agent.decide(_ctx("Chongqing-HotPotato", [40.0, 38.0, 39.0], DATES))
    assert out.decision is True
    assert "CARS" in out.rationale


def test_value_evaluator_math():
    ve = ValueEvaluator(cost_ratio={"heat": 0.3})
    rows = [
        {"category": "heat", "predicted": True, "ground_truth": True},
        {"category": "heat", "predicted": False, "ground_truth": False},
    ]
    s = ve.score(rows)
    # 完美决策：agent 花费 = perfect 花费 → V=1
    assert s["value_score"] == 1.0

    rows_clim = [
        {"category": "heat", "predicted": False, "ground_truth": True},
        {"category": "heat", "predicted": False, "ground_truth": True},
    ]
    # 全不预警 = 气候学（never）策略 → V=0
    s2 = ve.score(rows_clim)
    assert abs(s2["value_score"]) < 1e-9

    # 最差决策（事件发生却全不预警，而气候学应全预警）→ V<0
    rows_bad = [
        {"category": "heat", "predicted": False, "ground_truth": True},
        {"category": "heat", "predicted": False, "ground_truth": True},
        {"category": "heat", "predicted": False, "ground_truth": False},
        {"category": "heat", "predicted": False, "ground_truth": False},
    ]
    s3 = ve.score(rows_bad)
    # agent=2.0, clim=min(1.2, 2.0)=1.2, perfect=0.6 → V=(1.2-2.0)/0.6 < 0
    assert s3["value_score"] < 0
