"""EarthBench Alert 基准测试套件 — 四大高频风险类别。

全新重写：从单一 fire 扩展为 fire/flood/drought/heat 四类 Alert 场景，
每个场景都有可验证的 Ground Truth 推导规则（基于国标/行业标准阈值）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import (
    ScenarioContext,
    DecisionTemplate,
    Observation,
    ScenarioCategory,
)
from .scenarios import ScenarioStore, get_alert_benchmark_suite, DifficultyLevel
from .eval import BatchEvaluator


@dataclass
class AlertTestCase:
    """Alert 测试用例（泛化自原来的 fire-only TestCase）。"""

    case_id: str
    difficulty: str
    category: str  # "fire" | "flood" | "drought" | "heat"
    region: str
    ground_truth: bool = True
    horizon_hours: int = 72
    observations: list[dict[str, Any]] = field(default_factory=list)

    def to_context(self, store: ScenarioStore | None = None) -> ScenarioContext:
        """转换为 ScenarioContext。"""
        if store is None:
            store = ScenarioStore()

        obs_list = [Observation(**o) for o in self.observations]

        cat = ScenarioCategory.from_string(self.category)

        return store.load_scenario_from_dict(
            scenario_id=self.case_id,
            region=self.region,
            horizon_hours=self.horizon_hours,
            observations=obs_list,
            decision_template=DecisionTemplate.ALERT,
            category=cat,
        )


class AlertBenchEvaluator:
    """AlertBench 基准评测引擎（四类场景通用）。"""

    def __init__(self):
        self.raw_suite: list[dict[str, Any]] = get_alert_benchmark_suite()
        self.test_cases: list[AlertTestCase] = []
        self.results: list[dict[str, Any]] = []
        self._build_test_cases()

    def _build_test_cases(self) -> None:
        """从 raw_suite 构建 AlertTestCase 列表，并提取 Ground Truth。"""
        for item in self.raw_suite:
            # 提取 Ground Truth（不会修改原始数据）
            ground_truth = item.get("ground_truth", False)

            tc = AlertTestCase(
                case_id=item["case_id"],
                difficulty=item["difficulty"],
                category=item["category"],
                region=item["region"],
                ground_truth=ground_truth,
                observations=item["observations"],
            )
            self.test_cases.append(tc)

    def evaluate_agent(self, agent) -> list[dict[str, Any]]:
        """
        评测任意决策 Agent。

        Args:
            agent: 必须有 `decide(context: ScenarioContext) -> DecisionOutput` 方法

        Returns:
            评测结果列表
        """
        evaluator = BatchEvaluator()
        contexts = [tc.to_context() for tc in self.test_cases]
        ground_truths = [tc.ground_truth for tc in self.test_cases]

        raw_results = evaluator.run(agent, contexts, ground_truths)

        # 附加难度和类别信息
        for i, result in enumerate(raw_results):
            if "error" not in result:
                tc = self.test_cases[i]
                result["case_id"] = tc.case_id
                result["difficulty"] = tc.difficulty
                result["region"] = tc.region
                result["category"] = tc.category
                result["ground_truth"] = tc.ground_truth

        self.results = raw_results
        return raw_results

    def category_breakdown(self) -> dict[str, dict[str, float]]:
        """按场景类别分组的准确率。"""
        categories: dict[str, dict[str, float]] = {
            "fire": {"correct": 0.0, "total": 0.0},
            "flood": {"correct": 0.0, "total": 0.0},
            "drought": {"correct": 0.0, "total": 0.0},
            "heat": {"correct": 0.0, "total": 0.0},
        }

        for r in self.results:
            if "error" in r:
                continue
            cat = r.get("category", "unknown")
            if cat in categories:
                categories[cat]["total"] += 1
                if r.get("accuracy", 0) == 1.0:
                    categories[cat]["correct"] += 1

        for cat in categories.values():
            if cat["total"] > 0:
                cat["accuracy"] = cat["correct"] / cat["total"]
            else:
                cat["accuracy"] = 0.0

        return categories

    def difficulty_breakdown(self) -> dict[str, dict[str, float]]:
        """按难度分组的准确率。"""
        breakdown: dict[str, dict[str, float]] = {
            dl: {"correct": 0.0, "total": 0.0}
            for dl in [
                DifficultyLevel.L1_EASY,
                DifficultyLevel.L2_MEDIUM,
                DifficultyLevel.L3_HARD,
                DifficultyLevel.L4_BEYOND,
            ]
        }

        for r in self.results:
            if "error" in r:
                continue
            dl = r.get("difficulty", "UNKNOWN")
            if dl in breakdown:
                breakdown[dl]["total"] += 1
                if r.get("accuracy", 0) == 1.0:
                    breakdown[dl]["correct"] += 1

        for dl in breakdown.values():
            if dl["total"] > 0:
                dl["accuracy"] = dl["correct"] / dl["total"]
            else:
                dl["accuracy"] = 0.0

        return breakdown

    def summary_report(self) -> dict[str, Any]:
        """生成完整 AlertBench 评测报告。"""
        accuracy_values = [
            r.get("accuracy", 0) for r in self.results if "accuracy" in r
        ]
        overall_acc = (
            sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
        )

        cat_bd = self.category_breakdown()
        diff_bd = self.difficulty_breakdown()

        return {
            "benchmark": "AlertBench",
            "version": "0.3.0",
            "categories": ["fire", "flood", "drought", "heat"],
            "total_cases": len(self.test_cases),
            "overall_accuracy": round(overall_acc, 4),
            "by_category": cat_bd,
            "by_difficulty": diff_bd,
            "cases": [
                {
                    "id": tc.case_id,
                    "difficulty": tc.difficulty,
                    "category": tc.category,
                    "region": tc.region,
                    "ground_truth": tc.ground_truth,
                }
                for tc in self.test_cases
            ],
        }
