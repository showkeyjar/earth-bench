"""EarthBench Alert 基准测试套件 — 八大高频风险类别。

全新重写：从单一 fire 扩展为 fire/flood/drought/heat 四类 Alert 场景，
Phase A1 再扩展 landslide（滑坡/泥石流）、A2 扩展 typhoon（台风/大风）、
B1 扩展 cold（寒潮/冰冻）、B2 扩展 snow（暴雪/道路结冰），
见 docs/expansion-plan.md。每个场景都有可验证的 Ground Truth
推导规则（基于国标/行业标准阈值）。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from .eval import BatchEvaluator
from .models import (
    DecisionTemplate,
    ExposureProfile,
    Observation,
    ScenarioCategory,
    ScenarioContext,
)
from .scenarios import (
    DifficultyLevel,
    ScenarioStore,
    get_alert_benchmark_suite,
    infer_action_ground_truth,
    infer_exposure_class,
)


@dataclass
class AlertTestCase:
    """Alert 测试用例（泛化自原来的 fire-only TestCase）。"""

    case_id: str
    difficulty: str
    category: str  # "fire" | "flood" | ...
    region: str
    ground_truth: bool = True
    horizon_hours: int = 72
    observations: list[dict[str, Any]] = field(default_factory=list)
    exposure: ExposureProfile | None = None  # Phase B：暴露画像（可选）
    template: str = "alert"  # 决策模板（Phase C：dispatch/upgrade/close/recover）

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
            decision_template=DecisionTemplate(self.template),
            category=cat,
            exposure=self.exposure,
        )


class AlertBenchEvaluator:
    """AlertBench 基准评测引擎（四类场景通用）。"""

    def __init__(self, suite: list[dict[str, Any]] | None = None):
        """初始化评测器。

        Args:
            suite: 可选的自定义场景套件（如 get_adversarial_suite()）；
                   缺省使用基础 AlertBench 套件。
        """
        self.raw_suite: list[dict[str, Any]] = (
            suite if suite is not None else get_alert_benchmark_suite()
        )
        self.test_cases: list[AlertTestCase] = []
        self.results: list[dict[str, Any]] = []
        self.action_results: list[dict[str, Any]] = []
        self._gt_divergences: list[dict[str, Any]] = []
        self._build_test_cases()

    def _build_test_cases(self) -> None:
        """从 raw_suite 构建 AlertTestCase 列表，并提取 Ground Truth。

                Ground Truth 一律通过各场景的 `_gt_fn` **运行时推导**获得（独立物理
                标准查表），不直接读取场景中的硬编码值——硬编码值仅作为文档化参考，
        若与独立标准推导结果不一致，以独立标准为准并记录差异。
        """
        self._gt_divergences = []
        for item in self.raw_suite:
            gt_fn = item.get("_gt_fn")
            obs = item.get("observations", [])
            if gt_fn is not None:
                # 通用真值调用：按函数签名传入可选参数（heat_duration_days /
                # exposure），避免对不接受该参数的旧 gt_fn 抛 TypeError。
                kwargs: dict[str, Any] = {}
                params = inspect.signature(gt_fn).parameters
                if "heat_duration_days" in params:
                    kwargs["heat_duration_days"] = item.get("heat_duration_days", 3)
                if item.get("exposure") is not None and "exposure" in params:
                    kwargs["exposure"] = ExposureProfile(**item["exposure"])
                decision, _score, _explanation = gt_fn(obs, **kwargs)
                ground_truth = decision
                # 硬编码值仅作一致性校验：若与独立标准不一致，记录差异
                doc_truth = item.get("ground_truth", None)
                if doc_truth is not None and doc_truth != ground_truth:
                    self._gt_divergences.append(
                        {
                            "case_id": item["case_id"],
                            "documented": doc_truth,
                            "derived": ground_truth,
                        }
                    )
            else:
                ground_truth = item.get("ground_truth", False)

            tc = AlertTestCase(
                case_id=item["case_id"],
                difficulty=item["difficulty"],
                category=item["category"],
                region=item["region"],
                ground_truth=ground_truth,
                observations=obs,
                exposure=(
                    ExposureProfile(**item["exposure"])
                    if item.get("exposure") is not None
                    else None
                ),
                template=item.get("template", "alert"),
            )
            self.test_cases.append(tc)

    @property
    def gt_divergences(self) -> list[dict[str, Any]]:
        """独立标准推导与文档硬编码不一致的场景（应为空，否则需更新文档真值）。"""
        return self._gt_divergences

    def evaluate_action_agent(self, agent) -> list[dict[str, Any]]:
        """评测动作等级 Agent（Phase B 暴露层）。

        Args:
            agent: 必须有 `decide_action(context) -> str` 方法
                   （str ∈ {"monitor", "alert", "dispatch"}）

        动作真值 = infer_action_ground_truth(物理真值函数, 观测, 暴露画像)；
        适用于暴露套件（get_exposure_suite()），对无暴露画像的基础套件
        也可运行（E0 档动作）。
        """
        results: list[dict[str, Any]] = []
        for tc in self.test_cases:
            ctx = tc.to_context()
            try:
                action_gt, detail = infer_action_ground_truth(
                    self._gt_fn_of(tc), tc.observations, tc.exposure
                )
                action_pred = agent.decide_action(ctx)
                results.append(
                    {
                        "case_id": tc.case_id,
                        "category": tc.category,
                        "region": tc.region,
                        "difficulty": tc.difficulty,
                        "exposure_class": detail.get("exposure_class", "E0"),
                        "action_gt": action_gt,
                        "action_predicted": action_pred,
                        "action_accuracy": 1 if action_gt == action_pred else 0,
                    }
                )
            except Exception as e:  # pragma: no cover - 防御式
                results.append(
                    {"case_id": tc.case_id, "error": str(e)}
                )
        self.action_results = results
        return results

    def _gt_fn_of(self, tc: AlertTestCase) -> Any:
        """取回测试用例对应的物理真值函数（构建时暂存于 case 上）。"""
        fn = getattr(tc, "_gt_fn", None)
        if fn is not None:
            return fn
        for item in self.raw_suite:
            if item.get("case_id") == tc.case_id:
                fn = item.get("_gt_fn")
                if fn is None:
                    raise ValueError(
                        f"case {tc.case_id} has no _gt_fn for action evaluation"
                    )
                tc._gt_fn = fn  # type: ignore[attr-defined]
                return fn
        raise ValueError(f"case {tc.case_id} not found in raw_suite")

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
                if tc.exposure is not None:
                    exp_class, _ = infer_exposure_class(tc.exposure)
                    result["exposure_class"] = exp_class

        self.results = raw_results
        return raw_results

    def category_breakdown(self) -> dict[str, dict[str, float]]:
        """按场景类别分组的准确率。"""
        categories: dict[str, dict[str, float]] = {
            "fire": {"correct": 0.0, "total": 0.0},
            "flood": {"correct": 0.0, "total": 0.0},
            "drought": {"correct": 0.0, "total": 0.0},
            "heat": {"correct": 0.0, "total": 0.0},
            "landslide": {"correct": 0.0, "total": 0.0},
            "typhoon": {"correct": 0.0, "total": 0.0},
            "cold": {"correct": 0.0, "total": 0.0},
            "snow": {"correct": 0.0, "total": 0.0},
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
            "categories": [
                "fire", "flood", "drought", "heat",
                "landslide", "typhoon", "cold", "snow",
            ],
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
