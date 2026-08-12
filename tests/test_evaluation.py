"""EarthBench 全面测试套件。

覆盖：模板引擎、所有 Alert Agent、基准评测器、批量评测器、场景构建等。
"""

from __future__ import annotations

import pytest
from pathlib import Path

from earthbench.models import (
    ScenarioContext,
    Observation,
    DecisionTemplate,
    ScenarioCategory,
    DecisionOutput,
)
from earthbench.templates import TemplateEngine
from earthbench.scenarios import (
    ScenarioStore,
    get_alert_benchmark_suite,
    DifficultyLevel,
    infer_fire_ground_truth,
    infer_flood_ground_truth,
    infer_drought_ground_truth,
    infer_heatwave_ground_truth,
)
from earthbench.agents import (
    FireAlertAgent,
    FloodAlertAgent,
    DroughtAlertAgent,
    HeatWaveAlertAgent,
    MultiAlertAgent,
    RuleBasedAgent,
)
from earthbench.eval import BaseEvaluator, BatchEvaluator
from earthbench.benchmark import AlertTestCase, AlertBenchEvaluator


# ============================================================================
# Template Engine
# ============================================================================


class TestTemplates:
    """测试决策模板引擎。"""

    def test_min_observations(self):
        assert TemplateEngine.MIN_OBSERVATIONS[DecisionTemplate.ALERT] == 3
        assert TemplateEngine.MIN_OBSERVATIONS[DecisionTemplate.DISPATCH] == 4
        assert TemplateEngine.MIN_OBSERVATIONS[DecisionTemplate.UPGRADE] == 5
        assert TemplateEngine.MIN_OBSERVATIONS[DecisionTemplate.CLOSE] == 3
        assert TemplateEngine.MIN_OBSERVATIONS[DecisionTemplate.RECOVER] == 4

    def test_validate_context_valid(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=40,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=42,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=12,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="test",
        )
        valid, msg = TemplateEngine.validate_context(ctx)
        assert valid
        assert msg == "验证通过"

    def test_validate_context_insufficient(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=40,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="test",
        )
        valid, msg = TemplateEngine.validate_context(ctx)
        assert not valid

    def test_get_explanation_template(self):
        tmpl = TemplateEngine.get_explanation_template(DecisionTemplate.ALERT)
        assert isinstance(tmpl, str)
        assert len(tmpl) > 0

    def test_all_templates_have_explanations(self):
        for t in DecisionTemplate:
            expl = TemplateEngine.get_explanation_template(t)
            assert isinstance(expl, str)
            assert len(expl) > 0


# ============================================================================
# ScenarioStore
# ============================================================================


class TestScenarioStore:
    """测试场景存储。"""

    def test_register_and_get(self):
        store = ScenarioStore()
        obs = [
            Observation(
                source="s",
                variable="v",
                value=1,
                unit="",
                timestamp="t",
                confidence=1.0,
            )
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="r",
        )
        store.register_scenario("id1", ctx)
        assert store.get("id1") is ctx
        assert store.get("nonexistent") is None

    def test_load_fire_scenario(self):
        store = ScenarioStore()
        ctx = store.load_fire_scenario(
            "s1",
            "region-a",
            horizon_hours=48,
            observations=[
                {
                    "source": "ECMWF",
                    "variable": "FWI",
                    "value": 45.0,
                    "unit": "",
                    "timestamp": "t1",
                    "confidence": 0.9,
                },
            ],
        )
        assert ctx.region == "region-a"
        assert ctx.horizon_hours == 48
        assert ctx.category == ScenarioCategory.FIRE
        assert len(ctx.observations) == 1

    def test_load_scenario_from_dict(self):
        store = ScenarioStore()
        obs = [
            Observation(
                source="s",
                variable="v",
                value=5,
                unit="",
                timestamp="t",
                confidence=1.0,
            )
        ]
        ctx = store.load_scenario_from_dict(
            "s2", "region-b", observations=obs, category=ScenarioCategory.FLOOD
        )
        assert ctx.category == ScenarioCategory.FLOOD

    def test_get_all_ids(self):
        store = ScenarioStore()
        store.register_scenario(
            "a",
            ScenarioContext(
                category=ScenarioCategory.FIRE,
                template=DecisionTemplate.ALERT,
                observations=[],
                region="x",
            ),
        )
        store.register_scenario(
            "b",
            ScenarioContext(
                category=ScenarioCategory.FLOOD,
                template=DecisionTemplate.ALERT,
                observations=[],
                region="y",
            ),
        )
        ids = store.get_all_ids()
        assert set(ids) == {"a", "b"}


# ============================================================================
# Ground Truth Inference
# ============================================================================


class TestGroundTruth:
    """测试 Ground Truth 推导函数。"""

    def test_fire_extreme_risk_gt(self):
        obs = [
            {
                "source": "s",
                "variable": "FWI",
                "value": 55.0,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "humidity",
                "value": 10.0,
                "unit": "%",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "wind_speed",
                "value": 18.0,
                "unit": "m/s",
                "timestamp": "t",
                "confidence": 1.0,
            },
        ]
        decision, score, explanation = infer_fire_ground_truth(obs)
        assert decision is True
        assert score > 0.4

    def test_fire_low_risk_gt(self):
        obs = [
            {
                "source": "s",
                "variable": "FWI",
                "value": 15.0,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "humidity",
                "value": 70.0,
                "unit": "%",
                "timestamp": "t",
                "confidence": 1.0,
            },
        ]
        decision, score, explanation = infer_fire_ground_truth(obs)
        assert decision is False

    def test_flood_extreme_rain_gt(self):
        obs = [
            {
                "source": "s",
                "variable": "rainfall_24h",
                "value": 120.0,
                "unit": "mm",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "soil_moisture",
                "value": 0.90,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
        ]
        decision, score, explanation = infer_flood_ground_truth(obs)
        assert decision is True

    def test_drought_severe_gt(self):
        obs = [
            {
                "source": "s",
                "variable": "SPI",
                "value": -2.3,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "palmer_index",
                "value": -1.0,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "NDVI",
                "value": 0.15,
                "unit": "",
                "timestamp": "t",
                "confidence": 1.0,
            },
        ]
        decision, score, explanation = infer_drought_ground_truth(obs)
        assert decision is True

    def test_heatwave_extreme_gt(self):
        obs = [
            {
                "source": "s",
                "variable": "temperature_max",
                "value": 40.0,
                "unit": "°C",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "wet_bulb_temp",
                "value": 30.0,
                "unit": "°C",
                "timestamp": "t",
                "confidence": 1.0,
            },
            {
                "source": "s",
                "variable": "humidity",
                "value": 80.0,
                "unit": "%",
                "timestamp": "t",
                "confidence": 1.0,
            },
        ]
        decision, score, explanation = infer_heatwave_ground_truth(
            obs, heat_duration_days=3
        )
        assert decision is True


# ============================================================================
# Benchmark Suite Generation
# ============================================================================


class TestBenchmarkSuite:
    """测试基准测试套件生成。"""

    def test_suite_size(self):
        suite = get_alert_benchmark_suite()
        assert len(suite) == 20

    def test_suite_categories(self):
        suite = get_alert_benchmark_suite()
        categories = {item["category"] for item in suite}
        assert categories == {"fire", "flood", "drought", "heat"}

    def test_suite_difficulty_levels(self):
        suite = get_alert_benchmark_suite()
        difficulties = {item["difficulty"] for item in suite}
        expected = {"L1", "L2", "L3", "L4"}
        assert expected.issubset(difficulties)

    def test_suite_has_gt_fn(self):
        suite = get_alert_benchmark_suite()
        for item in suite:
            assert "_gt_fn" in item
            assert callable(item["_gt_fn"])

    def test_suite_has_observations(self):
        suite = get_alert_benchmark_suite()
        for item in suite:
            assert "observations" in item
            assert len(item["observations"]) > 0

    def test_suite_has_ground_truth(self):
        suite = get_alert_benchmark_suite()
        for item in suite:
            assert "ground_truth" in item
            assert isinstance(item["ground_truth"], bool)

    def test_suite_all_difficulties_per_category(self):
        suite = get_alert_benchmark_suite()
        for cat in ["fire", "flood", "drought", "heat"]:
            cat_items = [i for i in suite if i["category"] == cat]
            assert len(cat_items) == 5
            difficulties_found = [i["difficulty"] for i in cat_items]
            # Each category should have at least one case from L1/L2/L3/L4
            # (Some categories have duplicates, e.g., fire has 2x L1 but no L3)
            assert "L1" in difficulties_found
            assert "L2" in difficulties_found
            assert "L4" in difficulties_found


# ============================================================================
# Alert Agent Tests — Fire
# ============================================================================


class TestFireAlertAgent:
    """测试森林火险预警 Agent。"""

    def test_high_risk_fire(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="wind_speed",
                value=15,
                unit="m/s",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="D",
                variable="temperature",
                value=36,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="beijing",
        )
        agent = FireAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True
        assert result.confidence > 0.4

    def test_low_risk_fire(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=15,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=70,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="wind_speed",
                value=3,
                unit="m/s",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="hangzhou",
        )
        agent = FireAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False

    def test_rain_suppression(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=15,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="rainfall",
                value=30,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="changbai",
        )
        agent = FireAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False

    def test_evidence_summary_included(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="test",
        )
        agent = FireAlertAgent()
        result = agent.decide(ctx)
        assert "evidence_summary" in dir(result)
        assert isinstance(result.evidence_summary, dict)


# ============================================================================
# Alert Agent Tests — Flood
# ============================================================================


class TestFloodAlertAgent:
    """测试洪涝预警 Agent。"""

    def test_extreme_rainfall_flood(self):
        obs = [
            Observation(
                source="A",
                variable="rainfall_24h",
                value=120,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="rainfall_6h",
                value=60,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="soil_moisture",
                value=0.90,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="D",
                variable="water_level",
                value=12,
                unit="m",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FLOOD,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="wuhan",
        )
        agent = FloodAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True
        assert result.confidence > 0.45

    def test_no_rain_flood(self):
        obs = [
            Observation(
                source="A",
                variable="rainfall_24h",
                value=0,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="soil_moisture",
                value=0.20,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="water_level",
                value=2,
                unit="m",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FLOOD,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="lhasa",
        )
        agent = FloodAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False

    def test_soil_saturation_flood(self):
        obs = [
            Observation(
                source="A",
                variable="rainfall_24h",
                value=40,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="soil_moisture",
                value=0.88,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="water_level",
                value=8.5,
                unit="m",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FLOOD,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="guangzhou",
        )
        agent = FloodAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True

    def test_water_trend_bonus(self):
        obs = [
            Observation(
                source="A",
                variable="water_level",
                value=9.5,
                unit="m",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="water_level",
                value=10.2,
                unit="m",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="water_level",
                value=10.8,
                unit="m",
                timestamp="t3",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FLOOD,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="nanjing",
        )
        agent = FloodAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True
        assert "water_trend" in result.evidence_summary


# ============================================================================
# Alert Agent Tests — Drought
# ============================================================================


class TestDroughtAlertAgent:
    """测试干旱预警 Agent。"""

    def test_severe_spi_drought(self):
        obs = [
            Observation(
                source="A",
                variable="SPI",
                value=-2.3,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="palmer_index",
                value=-0.8,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="humidity",
                value=15,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="D",
                variable="rainfall_monthly",
                value=5,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="E",
                variable="NDVI",
                value=0.15,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.DROUGHT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="kunming",
        )
        agent = DroughtAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True

    def test_normal_conditions_no_drought(self):
        obs = [
            Observation(
                source="A",
                variable="SPI",
                value=-0.2,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="palmer_index",
                value=-0.1,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="humidity",
                value=60,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="D",
                variable="rainfall_monthly",
                value=150,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="E",
                variable="NDVI",
                value=0.65,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.DROUGHT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="hangzhou",
        )
        agent = DroughtAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False

    def test_palmerspi_conflict_drought(self):
        obs = [
            Observation(
                source="A",
                variable="SPI",
                value=-0.3,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="palmer_index",
                value=-1.2,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="rainfall_monthly",
                value=5,
                unit="mm",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="D",
                variable="NDVI",
                value=0.10,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.DROUGHT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="urumqi",
        )
        agent = DroughtAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True


# ============================================================================
# Alert Agent Tests — HeatWave
# ============================================================================


class TestHeatWaveAlertAgent:
    """测试热浪预警 Agent。"""

    def test_extreme_wetbulb_heat(self):
        obs = [
            Observation(
                source="A",
                variable="temperature_max",
                value=40,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="temperature_max",
                value=38,
                unit="°C",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="temperature_max",
                value=39,
                unit="°C",
                timestamp="t3",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="wet_bulb_temp",
                value=30,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="humidity",
                value=75,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.HEAT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="chongqing",
        )
        agent = HeatWaveAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is True

    def test_comfortable_no_heat(self):
        obs = [
            Observation(
                source="A",
                variable="temperature_max",
                value=26,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="temperature_max",
                value=27,
                unit="°C",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=40,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.HEAT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="kunming",
        )
        agent = HeatWaveAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False

    def test_hot_but_dry(self):
        obs = [
            Observation(
                source="A",
                variable="temperature_max",
                value=36,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=15,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="C",
                variable="wet_bulb_temp",
                value=22,
                unit="°C",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.HEAT,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="urumqi",
        )
        agent = HeatWaveAlertAgent()
        result = agent.decide(ctx)
        assert result.decision is False


# ============================================================================
# MultiAlertAgent Router
# ============================================================================


class TestMultiAlertAgent:
    """测试多场景路由 Agent。"""

    @pytest.fixture
    def agent(self):
        return MultiAlertAgent()

    def test_route_fire(self, agent):
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=[
                Observation(
                    source="s",
                    variable="FWI",
                    value=50,
                    unit="",
                    timestamp="t",
                    confidence=1.0,
                )
            ],
            region="test",
        )
        result = agent.decide(ctx)
        assert result.rationale.startswith("FireAlert:")

    def test_route_flood(self, agent):
        ctx = ScenarioContext(
            category=ScenarioCategory.FLOOD,
            template=DecisionTemplate.ALERT,
            observations=[
                Observation(
                    source="s",
                    variable="rainfall_24h",
                    value=100,
                    unit="mm",
                    timestamp="t",
                    confidence=1.0,
                )
            ],
            region="test",
        )
        result = agent.decide(ctx)
        assert result.rationale.startswith("FloodAlert:")

    def test_route_drought(self, agent):
        ctx = ScenarioContext(
            category=ScenarioCategory.DROUGHT,
            template=DecisionTemplate.ALERT,
            observations=[
                Observation(
                    source="s",
                    variable="SPI",
                    value=-2.0,
                    unit="",
                    timestamp="t",
                    confidence=1.0,
                )
            ],
            region="test",
        )
        result = agent.decide(ctx)
        assert result.rationale.startswith("DroughtAlert:")

    def test_route_heat(self, agent):
        ctx = ScenarioContext(
            category=ScenarioCategory.HEAT,
            template=DecisionTemplate.ALERT,
            observations=[
                Observation(
                    source="s",
                    variable="temperature_max",
                    value=39,
                    unit="°C",
                    timestamp="t",
                    confidence=1.0,
                )
            ],
            region="test",
        )
        result = agent.decide(ctx)
        assert result.rationale.startswith("HeatWaveAlert:")


# ============================================================================
# BaseEvaluator & BatchEvaluator
# ============================================================================


class TestBaseEvaluator:
    """测试单例评测器。"""

    def test_correct_prediction(self):
        evaluator = BaseEvaluator(ground_truth=True)
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=[],
            region="test",
        )
        pred = DecisionOutput(
            context=ctx,
            decision=True,
            confidence=0.8,
            evidence_summary={},
            rationale="test",
        )
        metrics = evaluator.evaluate(pred)
        assert metrics["accuracy"] == 1.0

    def test_wrong_prediction(self):
        evaluator = BaseEvaluator(ground_truth=False)
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=[],
            region="test",
        )
        pred = DecisionOutput(
            context=ctx,
            decision=True,
            confidence=0.7,
            evidence_summary={},
            rationale="test",
        )
        metrics = evaluator.evaluate(pred)
        assert metrics["accuracy"] == 0.0

    def test_confidence_calibration(self):
        evaluator = BaseEvaluator(ground_truth=True)
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=[],
            region="test",
        )
        pred = DecisionOutput(
            context=ctx,
            decision=True,
            confidence=0.95,
            evidence_summary={},
            rationale="test",
        )
        metrics = evaluator.evaluate(pred)
        assert metrics["confidence_calibration"] > 0.9


class TestBatchEvaluatorExtended:
    """测试批量评测器增强指标。"""

    def test_batch_accuracy(self):
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=52,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="test",
        )
        agent = FireAlertAgent()
        evaluator = BatchEvaluator()
        results = evaluator.run(agent, [ctx], [True])
        summary = evaluator.summary()
        assert summary["accuracy"] == 1.0
        assert summary["total_scenarios"] == 1

    def test_batch_precision_recall_f1(self):
        obs_high = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=52,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx_positive = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs_high,
            region="high-risk",
        )
        obs_low = [
            Observation(
                source="A",
                variable="FWI",
                value=15,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=18,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=70,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx_negative = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs_low,
            region="low-risk",
        )

        agent = FireAlertAgent()
        evaluator = BatchEvaluator()
        results = evaluator.run(agent, [ctx_positive, ctx_negative], [True, False])
        summary = evaluator.summary()
        assert summary["accuracy"] == 1.0
        assert summary["precision"] == 1.0
        assert summary["recall"] == 1.0
        assert abs(summary["f1_score"] - 1.0) < 1e-9

    def test_batch_mixed_results(self):
        obs_high = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=52,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs_high,
            region="test",
        )

        class WrongAgent:
            def decide(self, context):
                c = ScenarioContext(
                    category=ScenarioCategory.FIRE,
                    template=DecisionTemplate.ALERT,
                    observations=[],
                    region="stub",
                )
                return DecisionOutput(
                    context=c,
                    decision=False,
                    confidence=0.3,
                    evidence_summary={},
                    rationale="wrong",
                )

        evaluator = BatchEvaluator()
        results = evaluator.run(WrongAgent(), [ctx], [True])
        summary = evaluator.summary()
        assert summary["accuracy"] == 0.0
        assert summary["false_positives"] == 0
        assert summary["false_negatives"] == 1
        assert summary["true_positives"] == 0
        assert summary["true_negatives"] == 0

    def test_batch_error_handling(self):
        agent = FireAlertAgent()
        evaluator = BatchEvaluator()

        # Only 2 obs — below minimum of 3 for ALERT template
        obs = [
            Observation(
                source="s",
                variable="FWI",
                value=10,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="s",
                variable="humidity",
                value=50,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx_low_obs = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="low-obs",
        )
        results = evaluator.run(agent, [ctx_low_obs], [False])
        assert any("error" in r for r in results)


# ============================================================================
# AlertBenchEvaluator (Full Benchmark)
# ============================================================================


class TestAlertBenchEvaluator:
    """测试完整基准评测器。"""

    @pytest.fixture
    def bench(self):
        return AlertBenchEvaluator()

    def test_bench_initialization(self, bench):
        assert len(bench.test_cases) == 20

    def test_bench_agent_evaluation(self, bench):
        agent = MultiAlertAgent()
        results = bench.evaluate_agent(agent)
        assert len(results) == 20
        assert all("accuracy" in r for r in results if "error" not in r)

    def test_bench_category_breakdown(self, bench):
        agent = MultiAlertAgent()
        bench.evaluate_agent(agent)
        breakdown = bench.category_breakdown()
        for cat in ["fire", "flood", "drought", "heat"]:
            assert cat in breakdown
            assert breakdown[cat]["total"] == 5

    def test_bench_difficulty_breakdown(self, bench):
        agent = MultiAlertAgent()
        bench.evaluate_agent(agent)
        breakdown = bench.difficulty_breakdown()
        assert "L1" in breakdown
        assert "L2" in breakdown
        assert "L3" in breakdown
        assert "L4" in breakdown

    def test_bench_summary_report(self, bench):
        agent = MultiAlertAgent()
        bench.evaluate_agent(agent)
        report = bench.summary_report()
        assert report["benchmark"] == "AlertBench"
        assert report["version"] == "0.3.0"
        assert report["total_cases"] == 20
        assert "overall_accuracy" in report
        assert "by_category" in report
        assert "by_difficulty" in report

    def test_perfect_accuracy_rule_agents(self, bench):
        """规则 Agent 使用与 Ground Truth 相同的评分模型，应达到 100% 准确率。"""
        agent = MultiAlertAgent()
        bench.evaluate_agent(agent)
        report = bench.summary_report()
        assert report["overall_accuracy"] == 1.0

    def test_single_category_eval(self, bench):
        """测试单一类别过滤评测。"""
        from earthbench.benchmark import AlertBenchEvaluator

        fb = AlertBenchEvaluator()
        fb.raw_suite = [item for item in fb.raw_suite if item["category"] == "fire"]
        fb.test_cases = []
        fb.results = []
        fb._build_test_cases()

        agent = FireAlertAgent()
        results = fb.evaluate_agent(agent)
        assert len(results) == 5
        for r in results:
            assert r.get("category") == "fire"


# ============================================================================
# AlertTestCase
# ============================================================================


class TestAlertTestCase:
    """测试 AlertTestCase 数据类。"""

    def test_to_context(self):
        tc = AlertTestCase(
            case_id="test-001",
            difficulty="L1",
            category="fire",
            region="beijing",
            ground_truth=True,
            observations=[
                {
                    "source": "s",
                    "variable": "FWI",
                    "value": 50,
                    "unit": "",
                    "timestamp": "t1",
                    "confidence": 1.0,
                },
            ],
        )
        ctx = tc.to_context()
        assert ctx.region == "beijing"
        assert ctx.category == ScenarioCategory.FIRE
        assert len(ctx.observations) == 1

    def test_to_context_flood(self):
        tc = AlertTestCase(
            case_id="flood-001",
            difficulty="L1",
            category="flood",
            region="wuhan",
            observations=[
                {
                    "source": "s",
                    "variable": "rainfall_24h",
                    "value": 80,
                    "unit": "mm",
                    "timestamp": "t1",
                    "confidence": 1.0,
                },
            ],
        )
        ctx = tc.to_context()
        assert ctx.category == ScenarioCategory.FLOOD

    def test_default_ground_truth(self):
        tc = AlertTestCase(case_id="t", difficulty="L1", category="fire", region="r")
        assert tc.ground_truth is True


# ============================================================================
# DifficultyLevel
# ============================================================================


class TestDifficultyLevel:
    """测试难度级别。"""

    def test_difficulty_values(self):
        assert DifficultyLevel.L1_EASY == "L1"
        assert DifficultyLevel.L2_MEDIUM == "L2"
        assert DifficultyLevel.L3_HARD == "L3"
        assert DifficultyLevel.L4_BEYOND == "L4"


# ============================================================================
# Data Collectors
# ============================================================================


class TestDataCollectors:
    """测试数据收集器模块的纯函数接口。"""

    def test_calculate_fwi_from_weather_normal(self):
        from earthbench.data_collectors import calculate_fwi_from_weather

        realtime = {"temp": 35, "humidity": 20, "wind_speed_ms": 15, "precip_1h": 0}
        fwi = calculate_fwi_from_weather(realtime)
        assert isinstance(fwi, float)
        assert fwi > 0

    def test_calculate_fwi_with_water_correction(self):
        from earthbench.data_collectors import (
            calculate_fwi_from_weather,
            WATER_BODY_REGIONS,
        )

        realtime = {"temp": 35, "humidity": 20, "wind_speed_ms": 15, "precip_1h": 0}
        fwi_plain = calculate_fwi_from_weather(realtime, region_key="Xiangshan-Beijing")
        fwi_water = calculate_fwi_from_weather(realtime, region_key="WestLake-Hangzhou")
        # 水体修正区域 FWI 应更低
        assert fwi_water <= fwi_plain

    def test_weather_to_observations_fire(self):
        from earthbench.data_collectors import weather_to_observations

        realtime = {
            "temp": 30,
            "humidity": 25,
            "wind_speed_ms": 12,
            "precip_1h": 0,
            "obsTime": "2026-07-11T12:00:00+08:00",
        }
        obs = weather_to_observations(
            realtime=realtime,
            hourly=[],
            daily=[],
            category="fire",
            region_key="Xiangshan-Beijing",
            region_name="北京",
        )
        assert isinstance(obs, list)
        assert len(obs) > 0
        variables = [o["variable"] for o in obs]
        assert "FWI" in variables
        assert "temperature" in variables

    def test_weather_to_observations_heat(self):
        from earthbench.data_collectors import weather_to_observations

        realtime = {
            "temp": 38,
            "humidity": 70,
            "obsTime": "2026-07-11T12:00:00+08:00",
        }
        obs = weather_to_observations(
            realtime=realtime,
            hourly=[],
            daily=[],
            category="heat",
            region_key="Chongqing-HotPotato",
            region_name="重庆",
        )
        assert len(obs) > 0
        variables = [o["variable"] for o in obs]
        assert "temperature_max" in variables
        assert "humidity" in variables

    def test_fallback_to_scenarios(self):
        from earthbench.data_collectors import fallback_to_scenarios
        from earthbench.scenarios import get_alert_benchmark_suite

        suite = get_alert_benchmark_suite()
        results = fallback_to_scenarios(suite, category_filter=["fire"])
        assert len(results) == 5
        assert all(r["category"] == "fire" for r in results)

    def test_region_location_map_completeness(self):
        from earthbench.data_collectors import REGION_LOCATION_MAP
        from earthbench.scenarios import get_alert_benchmark_suite

        suite = get_alert_benchmark_suite()
        for item in suite:
            region = item["region"]
            assert region in REGION_LOCATION_MAP, (
                f"Region '{region}' not in REGION_LOCATION_MAP"
            )


# ============================================================================
# LLMDecisionAgent
# ============================================================================


class TestLLMDecisionAgent:
    """测试 LLM Decision Agent（含启发式回退）。"""

    def test_heuristic_fallback_fire(self):
        from earthbench.agents import LLMDecisionAgent

        agent = LLMDecisionAgent()
        obs = [
            Observation(
                source="A",
                variable="FWI",
                value=50,
                unit="",
                timestamp="t1",
                confidence=1.0,
            ),
            Observation(
                source="A",
                variable="FWI",
                value=52,
                unit="",
                timestamp="t2",
                confidence=1.0,
            ),
            Observation(
                source="B",
                variable="humidity",
                value=10,
                unit="%",
                timestamp="t1",
                confidence=1.0,
            ),
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="beijing",
        )
        result = agent.decide(ctx)
        assert hasattr(result, "decision")
        assert hasattr(result, "confidence")

    def test_invalid_context_returns_error(self):
        from earthbench.agents import LLMDecisionAgent

        agent = LLMDecisionAgent()
        obs = [
            Observation(
                source="s",
                variable="v",
                value=1,
                unit="",
                timestamp="t",
                confidence=1.0,
            )
        ]
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs,
            region="minimal",
        )
        result = agent.decide(ctx)
        assert result.decision is False
        assert result.confidence == 0.0


class TestRegionPolygons:
    """Test real geographic polygon data for map precision."""

    def test_all_regions_have_polygons(self):
        from earthbench.enhance_data import REGION_COORDS, REGION_POLYGONS

        # Every region in REGION_COORDS should have a matching polygon
        for region_key in REGION_COORDS:
            assert region_key in REGION_POLYGONS, (
                f"Region '{region_key}' missing from REGION_POLYGONS"
            )

    def test_polygon_points_count(self):
        from earthbench.enhance_data import REGION_POLYGONS

        for name, poly in REGION_POLYGONS.items():
            assert len(poly) >= 4, (
                f"Region '{name}' has only {len(poly)} polygon points (need >= 4)"
            )

    def test_polygon_coordinates_valid(self):
        from earthbench.enhance_data import REGION_POLYGONS

        for name, poly in REGION_POLYGONS.items():
            for i, pt in enumerate(poly):
                assert "lat" in pt, f"Region '{name}' point {i} missing 'lat'"
                assert "lng" in pt, f"Region '{name}' point {i} missing 'lng'"
                assert -90 <= pt["lat"] <= 90, (
                    f"Region '{name}' point {i} lat out of range: {pt['lat']}"
                )
                assert -180 <= pt["lng"] <= 180, (
                    f"Region '{name}' point {i} lng out of range: {pt['lng']}"
                )

    def test_enhance_decision_uses_real_polygon(self):
        from earthbench.enhance_data import enhance_decision, REGION_POLYGONS

        decision = {
            "region": "Xiangshan-Beijing",
            "case_id": "test-001",
            "category": "fire",
            "llm_decision": False,
            "confidence": 0.3,
        }
        enhance_decision(decision, None)

        expected_poly = REGION_POLYGONS["Xiangshan-Beijing"]
        assert decision["polygon"] == expected_poly
        assert len(decision["polygon"]) == len(expected_poly)


# ============================================================================
# Verification Module Tests
# ============================================================================


class TestVerificationModule:
    """Test the closed-loop verification system."""

    def test_wet_bulb_calculation(self):
        """Test Stull 2011 wet bulb temperature formula."""
        from earthbench.verification import _wet_bulb

        # 35degC, 50% RH -> approximately 25-27degC wet bulb
        tw = _wet_bulb(35.0, 50.0)
        assert 24.0 < tw < 28.0, f"Expected ~25-27degC, got {tw}"

        # 40degC, 30% RH -> approximately 24-26degC wet bulb (dry heat)
        tw = _wet_bulb(40.0, 30.0)
        assert 23.0 < tw < 27.0, f"Expected ~24-26degC, got {tw}"

        # 30degC, 80% RH -> high humidity, wet bulb close to dry bulb
        tw = _wet_bulb(30.0, 80.0)
        assert 27.0 < tw < 30.0, f"Expected ~27-29degC, got {tw}"

    def test_verify_prediction_structure(self):
        """Test that verify_prediction returns expected structure."""
        from earthbench.verification import verify_prediction

        prediction = {
            "category": "flood",
            "region_id": "Xiangshan-Beijing",
            "region_name": "北京",
            "llm_decision": True,
        }

        # Mock insufficient weather data
        result = verify_prediction(prediction, {"error": "no data"})

        assert result["category"] == "flood"
        assert result["region_id"] == "Xiangshan-Beijing"
        assert result["predicted"] is True
        assert result["verification_status"] == "insufficient_data"
        assert result["hit"] is None

    def test_verify_prediction_unknown_region(self):
        """Test verify_prediction with unknown region for fire."""
        from earthbench.verification import verify_prediction

        prediction = {
            "category": "fire",
            "region_id": "NonexistentRegion",
            "region_name": "Unknown",
            "llm_decision": False,
        }

        result = verify_prediction(prediction)

        assert result["category"] == "fire"
        assert result["verification_status"] == "verified"
        # FIRMS will return empty list for unknown region -> actual=False
        assert result["actual"] is False
        assert result["hit"] is True  # predicted=False, actual=False -> hit

    def test_build_accuracy_trend(self, tmp_path):
        """Test build_accuracy_trend accumulates verification files correctly."""
        from earthbench.verification import build_accuracy_trend

        # Create mock verification files
        import json
        from datetime import datetime, timezone, timedelta

        CST = timezone(timedelta(hours=8))
        now = datetime.now(CST)

        # File 1: T-1, 2 TP, 1 TN
        v1 = {
            "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "generated_at": now.isoformat(),
            "total_predictions": 3,
            "verified": 3,
            "unverified": 0,
            "tp": 2,
            "fp": 0,
            "fn": 0,
            "tn": 1,
            "accuracy": 1.0,
            "verifications": [
                {
                    "category": "fire",
                    "region_id": "A",
                    "predicted": True,
                    "hit": True,
                    "verification_status": "verified",
                    "actual": True,
                },
                {
                    "category": "flood",
                    "region_id": "B",
                    "predicted": True,
                    "hit": True,
                    "verification_status": "verified",
                    "actual": True,
                },
                {
                    "category": "drought",
                    "region_id": "C",
                    "predicted": False,
                    "hit": True,
                    "verification_status": "verified",
                    "actual": False,
                },
            ],
        }

        # File 2: T-2, 1 TP, 1 FP, 1 FN, 1 TN
        v2 = {
            "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
            "generated_at": now.isoformat(),
            "total_predictions": 4,
            "verified": 4,
            "unverified": 0,
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "tn": 1,
            "accuracy": 0.5,
            "verifications": [
                {
                    "category": "fire",
                    "region_id": "A",
                    "predicted": True,
                    "hit": True,
                    "verification_status": "verified",
                    "actual": True,
                },
                {
                    "category": "flood",
                    "region_id": "B",
                    "predicted": True,
                    "hit": False,
                    "verification_status": "verified",
                    "actual": False,
                },
                {
                    "category": "drought",
                    "region_id": "C",
                    "predicted": False,
                    "hit": False,
                    "verification_status": "verified",
                    "actual": True,
                },
                {
                    "category": "heat",
                    "region_id": "D",
                    "predicted": False,
                    "hit": True,
                    "verification_status": "verified",
                    "actual": False,
                },
            ],
        }

        date1_str = (now - timedelta(days=1)).strftime("%Y%m%d")
        date2_str = (now - timedelta(days=2)).strftime("%Y%m%d")

        with open(tmp_path / f"verification_{date1_str}.json", "w") as f:
            json.dump(v1, f)
        with open(tmp_path / f"verification_{date2_str}.json", "w") as f:
            json.dump(v2, f)

        trend = build_accuracy_trend(tmp_path, keep_days=14)

        # Total: tp=3, fp=1, fn=1, tn=2 -> verified=7, correct=5
        assert trend["tp"] == 3
        assert trend["fp"] == 1
        assert trend["fn"] == 1
        assert trend["tn"] == 2
        assert trend["total_verified"] == 7
        assert trend["accuracy"] == round(5 / 7, 4)
        # precision = tp / (tp + fp) = 3/4
        assert trend["precision"] == round(3 / 4, 4)
        # recall = tp / (tp + fn) = 3/4
        assert trend["recall"] == round(3 / 4, 4)

        # Check daily trend has 2 entries
        assert len(trend["daily_trend"]) == 2

        # Check by_category
        assert "fire" in trend["by_category"]
        assert trend["by_category"]["fire"]["tp"] == 2

    def test_run_delayed_verification_no_data(self, tmp_path):
        """Test run_delayed_verification when no decisions file exists."""
        from earthbench.verification import run_delayed_verification

        result = run_delayed_verification(tmp_path, delay_days=2)

        assert result["status"] == "no_data"
        assert result["total"] == 0

    def test_run_delayed_verification_skip_existing(self, tmp_path):
        """Test run_delayed_verification skips when verification file exists."""
        import json
        from earthbench.verification import run_delayed_verification

        from datetime import datetime, timezone, timedelta

        CST = timezone(timedelta(hours=8))
        now = datetime.now(CST)
        target_date = now - timedelta(days=2)
        date_str = target_date.strftime("%Y%m%d")

        # Create existing verification file
        existing = {
            "date": target_date.strftime("%Y-%m-%d"),
            "status": "completed",
            "total": 20,
            "verified": 20,
            "accuracy": 0.95,
        }
        with open(tmp_path / f"verification_{date_str}.json", "w") as f:
            json.dump(existing, f)

        result = run_delayed_verification(tmp_path, delay_days=2)

        # Should return existing data without re-running
        assert result["status"] == "completed"
        assert result["accuracy"] == 0.95


# ============================================================================
# Calibration 模块测试
# ============================================================================


class TestCalibrationModule:
    """Test the threshold self-calibration module."""

    def test_default_thresholds(self):
        """Test DEFAULT_THRESHOLDS has all 4 categories."""
        from earthbench.calibration import DEFAULT_THRESHOLDS

        assert "fire" in DEFAULT_THRESHOLDS
        assert "flood" in DEFAULT_THRESHOLDS
        assert "drought" in DEFAULT_THRESHOLDS
        assert "heat" in DEFAULT_THRESHOLDS

    def test_load_thresholds_default(self, tmp_path):
        """Test load_thresholds returns defaults when no file exists."""
        from earthbench.calibration import load_thresholds, DEFAULT_THRESHOLDS

        result = load_thresholds(tmp_path)
        assert result == DEFAULT_THRESHOLDS

    def test_load_thresholds_from_file(self, tmp_path):
        """Test load_thresholds reads from existing file."""
        import json
        from earthbench.calibration import load_thresholds

        data = {
            "thresholds": {"fire": 0.35, "flood": 0.50},
            "updated_at": "2026-01-01T00:00:00+08:00",
            "version": 1,
        }
        with open(tmp_path / "thresholds.json", "w") as f:
            json.dump(data, f)

        result = load_thresholds(tmp_path)
        assert result["fire"] == 0.35
        assert result["flood"] == 0.50
        # Missing categories should use defaults
        assert result["drought"] == 0.40
        assert result["heat"] == 0.40

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test save then load returns same values."""
        from earthbench.calibration import save_thresholds, load_thresholds

        thresholds = {"fire": 0.38, "flood": 0.47, "drought": 0.42, "heat": 0.39}
        save_thresholds(tmp_path, thresholds)

        loaded = load_thresholds(tmp_path)
        assert loaded["fire"] == 0.38
        assert loaded["flood"] == 0.47
        assert loaded["drought"] == 0.42
        assert loaded["heat"] == 0.39

    def test_compute_adjustment_no_errors(self):
        """Test adjustment is skip when no FP/FN."""
        from earthbench.calibration import compute_adjustment

        result = compute_adjustment("fire", fp=0, fn=0, tn=5, tp=5)
        assert result["action"] == "skip"
        assert result["adjustment"] == 0.0

    def test_compute_adjustment_insufficient_samples(self):
        """Test adjustment is skip with too few samples."""
        from earthbench.calibration import compute_adjustment

        result = compute_adjustment("fire", fp=1, fn=0, tn=0, tp=0)
        assert result["action"] == "skip"
        assert "insufficient" in result["reason"]

    def test_compute_adjustment_fp_dominant(self):
        """Test adjustment goes up when FP > FN."""
        from earthbench.calibration import compute_adjustment

        result = compute_adjustment("fire", fp=3, fn=1, tn=5, tp=5)
        assert result["action"] == "adjust"
        assert result["adjustment"] > 0  # threshold should go up

    def test_compute_adjustment_fn_dominant(self):
        """Test adjustment goes down when FN > FP."""
        from earthbench.calibration import compute_adjustment

        result = compute_adjustment("fire", fp=1, fn=3, tn=5, tp=5)
        assert result["action"] == "adjust"
        assert result["adjustment"] < 0  # threshold should go down

    def test_compute_adjustment_max_step(self):
        """Test adjustment is capped at MAX_STEP."""
        from earthbench.calibration import compute_adjustment, MAX_STEP

        # Extreme FP vs FN ratio
        result = compute_adjustment("fire", fp=10, fn=0, tn=0, tp=0)
        assert result["action"] == "adjust"
        assert abs(result["adjustment"]) <= MAX_STEP

    def test_run_calibration_no_data(self, tmp_path):
        """Test run_calibration with no verification files."""
        from earthbench.calibration import run_calibration

        result = run_calibration(tmp_path)
        assert result["status"] == "no_change"
        assert all(a["adjustment"] == 0.0 for a in result["adjustments"])

    def test_run_calibration_with_fp(self, tmp_path):
        """Test run_calibration adjusts thresholds based on FP-heavy verification."""
        import json
        from earthbench.calibration import run_calibration, load_thresholds
        from datetime import datetime, timezone, timedelta

        CST = timezone(timedelta(hours=8))
        now = datetime.now(CST)

        # Create verification file with FP-heavy results
        verif_data = {
            "date": now.strftime("%Y-%m-%d"),
            "status": "completed",
            "verifications": [
                {
                    "category": "fire",
                    "verification_status": "verified",
                    "predicted": True,
                    "hit": False,  # FP
                },
                {
                    "category": "fire",
                    "verification_status": "verified",
                    "predicted": True,
                    "hit": False,  # FP
                },
                {
                    "category": "fire",
                    "verification_status": "verified",
                    "predicted": True,
                    "hit": True,  # TP
                },
            ],
        }
        with open(tmp_path / f"verification_{now.strftime('%Y%m%d')}.json", "w") as f:
            json.dump(verif_data, f)

        result = run_calibration(tmp_path)
        assert result["status"] == "calibrated"

        # Fire threshold should have increased (FP dominant)
        fire_adj = [a for a in result["adjustments"] if a["category"] == "fire"][0]
        assert fire_adj["adjustment"] > 0
        assert fire_adj["new_value"] > fire_adj["old_value"]

        # Check thresholds.json was saved
        loaded = load_thresholds(tmp_path)
        assert loaded["fire"] > 0.40  # should have gone up from default

    def test_run_calibration_safety_bounds(self, tmp_path):
        """Test that calibration respects MIN/MAX thresholds."""
        import json
        from datetime import datetime, timezone, timedelta
        from earthbench.calibration import (
            run_calibration,
            save_thresholds,
            load_thresholds,
            MAX_THRESHOLD,
        )

        # Set fire threshold near max
        save_thresholds(
            tmp_path,
            {
                "fire": MAX_THRESHOLD - 0.01,
                "flood": 0.45,
                "drought": 0.40,
                "heat": 0.40,
            },
        )

        # Create verification with lots of FP (should push fire up, but bounded)
        CST = timezone(timedelta(hours=8))
        now = datetime.now(CST)
        verif_data = {
            "date": now.strftime("%Y-%m-%d"),
            "status": "completed",
            "verifications": [
                {
                    "category": "fire",
                    "verification_status": "verified",
                    "predicted": True,
                    "hit": False,
                },
            ]
            * 5,
        }
        with open(tmp_path / f"verification_{now.strftime('%Y%m%d')}.json", "w") as f:
            json.dump(verif_data, f)

        run_calibration(tmp_path)

        loaded = load_thresholds(tmp_path)
        assert loaded["fire"] <= MAX_THRESHOLD

    def test_run_calibration_writes_status_file(self, tmp_path):
        """Test that calibration_status.json is written."""
        from earthbench.calibration import run_calibration
        import json

        run_calibration(tmp_path)

        status_file = tmp_path / "calibration_status.json"
        assert status_file.exists()

        with open(status_file) as f:
            status = json.load(f)
        assert "status" in status
        assert "adjustments" in status
        assert "calibrated_at" in status

    def test_run_calibration_writes_log(self, tmp_path):
        """Test that calibration_log.json is written when adjustment occurs."""
        import json
        from earthbench.calibration import run_calibration
        from datetime import datetime, timezone, timedelta

        CST = timezone(timedelta(hours=8))
        now = datetime.now(CST)

        # Create FP-heavy verification
        verif_data = {
            "date": now.strftime("%Y-%m-%d"),
            "status": "completed",
            "verifications": [
                {
                    "category": "fire",
                    "verification_status": "verified",
                    "predicted": True,
                    "hit": False,
                },
            ]
            * 5,
        }
        with open(tmp_path / f"verification_{now.strftime('%Y%m%d')}.json", "w") as f:
            json.dump(verif_data, f)

        run_calibration(tmp_path)

        log_file = tmp_path / "calibration_log.json"
        assert log_file.exists()

        with open(log_file) as f:
            logs = json.load(f)
        assert isinstance(logs, list)
        assert len(logs) > 0
        assert logs[-1]["category"] == "fire"
        assert logs[-1]["old_value"] < logs[-1]["new_value"]


class TestAgentDynamicThreshold:
    """Test that agents properly use dynamic thresholds from calibration."""

    def test_fire_agent_default_threshold(self):
        """Test FireAlertAgent uses default 0.4 when no calibration file."""
        from earthbench.agents import FireAlertAgent

        agent = FireAlertAgent()
        assert agent.decision_threshold == 0.4

    def test_fire_agent_explicit_threshold(self):
        """Test FireAlertAgent respects explicit threshold parameter."""
        from earthbench.agents import FireAlertAgent

        agent = FireAlertAgent(decision_threshold=0.35)
        assert agent.decision_threshold == 0.35

    def test_flood_agent_default_threshold(self):
        """Test FloodAlertAgent uses default 0.45."""
        from earthbench.agents import FloodAlertAgent

        agent = FloodAlertAgent()
        assert agent.decision_threshold == 0.45

    def test_drought_agent_default_threshold(self):
        """Test DroughtAlertAgent uses default 0.4."""
        from earthbench.agents import DroughtAlertAgent

        agent = DroughtAlertAgent()
        assert agent.decision_threshold == 0.4

    def test_heat_agent_default_threshold(self):
        """Test HeatWaveAlertAgent uses default 0.4."""
        from earthbench.agents import HeatWaveAlertAgent

        agent = HeatWaveAlertAgent()
        assert agent.decision_threshold == 0.4

    def test_agents_load_from_env(self, tmp_path):
        """Test agents read calibrated thresholds from EARTHBENCH_THRESHOLDS_JSON."""
        import json
        import os
        from earthbench.agents import FireAlertAgent

        # Write a thresholds.json
        data = {
            "thresholds": {"fire": 0.38},
            "updated_at": "2026-01-01T00:00:00+08:00",
            "version": 1,
        }
        threshold_file = tmp_path / "thresholds.json"
        with open(threshold_file, "w") as f:
            json.dump(data, f)

        # Set env var
        old_val = os.environ.get("EARTHBENCH_THRESHOLDS_JSON")
        os.environ["EARTHBENCH_THRESHOLDS_JSON"] = str(threshold_file)

        try:
            agent = FireAlertAgent()
            assert agent.decision_threshold == 0.38
        finally:
            if old_val is not None:
                os.environ["EARTHBENCH_THRESHOLDS_JSON"] = old_val
            else:
                del os.environ["EARTHBENCH_THRESHOLDS_JSON"]

    def test_multi_alert_agent_passes_thresholds(self):
        """Test MultiAlertAgent can pass thresholds to sub-agents."""
        from earthbench.agents import MultiAlertAgent

        agent = MultiAlertAgent(
            fire_threshold=0.35,
            flood_threshold=0.50,
            drought_threshold=0.42,
            heat_threshold=0.38,
        )

        assert agent.fire_agent.decision_threshold == 0.35
        assert agent.flood_agent.decision_threshold == 0.50
        assert agent.drought_agent.decision_threshold == 0.42
        assert agent.heat_agent.decision_threshold == 0.38
