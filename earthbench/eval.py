"""评测引擎。

参考 SIM 项目的实验框架，提供可量化评估的能力。
"""

from __future__ import annotations

from typing import Any, Protocol

from .models import ScenarioContext, DecisionOutput
from .templates import TemplateEngine


class DecisionAgent(Protocol):
    """决策 Agent 接口。

    所有需要评测的 Agent 必须实现此接口。
    """

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """基于场景上下文做出决策。"""
        ...


class BaseEvaluator:
    """评测器基类。"""

    def __init__(self, ground_truth: bool):
        self.ground_truth = ground_truth  # 正确答案（YES/NO）

    def evaluate(self, prediction: DecisionOutput) -> dict[str, Any]:
        """评估一次决策，返回度量指标。"""
        pred_label = 1.0 if prediction.decision else 0.0
        gt_label = 1.0 if self.ground_truth else 0.0

        is_correct = int(pred_label == gt_label)
        # confidence（风险评分，[0,1]）充当预测事件概率：Brier 分数能区分过度/不足
        # 自信，替代旧 1-|pred-conf| 的「软正确率」假校准指标。
        brier = (prediction.confidence - gt_label) ** 2

        return {
            "accuracy": float(is_correct),
            "brier": round(brier, 4),
            "ground_truth": self.ground_truth,
            "predicted": prediction.decision,
            "confidence": prediction.confidence,
        }


class BatchEvaluator:
    """批量评测引擎。

    对一组场景进行评估，汇总统计指标（准确率、精确率、召回率、F1等）。
    """

    def __init__(self):
        self._results: list[dict[str, Any]] = []

    def run(
        self,
        agent: DecisionAgent,
        contexts: list[ScenarioContext],
        ground_truths: list[bool],
    ) -> list[dict[str, Any]]:
        """运行批量评测。"""
        assert len(contexts) == len(ground_truths), "场景数量与真值数量不匹配"

        results: list[dict[str, Any]] = []
        for ctx, gt in zip(contexts, ground_truths):
            # 验证场景上下文
            valid, msg = TemplateEngine.validate_context(ctx)
            if not valid:
                err = {"error": msg, "scenario": str(ctx.region)}
                self._results.append(err)
                results.append(err)
                continue

            # Agent 做出决策
            output = agent.decide(ctx)

            # 评估
            evaluator = BaseEvaluator(ground_truth=gt)
            metrics = evaluator.evaluate(output)
            metrics["region"] = ctx.region
            metrics["template"] = ctx.template.value

            # 计算混淆矩阵指标
            predicted_label = int(metrics["predicted"])
            gt_label = int(metrics["ground_truth"])
            metrics["tp"] = int(predicted_label == 1 and gt_label == 1)
            metrics["fp"] = int(predicted_label == 1 and gt_label == 0)
            metrics["tn"] = int(predicted_label == 0 and gt_label == 0)
            metrics["fn"] = int(predicted_label == 0 and gt_label == 1)

            self._results.append(metrics)
            results.append(metrics)

        return results

    def _ece(self) -> float:
        """期望校准误差 ECE（10 档，按预测概率 confidence 分箱加权）。"""
        rows = [
            r for r in self._results
            if "confidence" in r and "ground_truth" in r
        ]
        n = len(rows)
        if n == 0:
            return 0.0
        bins: dict[int, list[tuple[float, float]]] = {}
        for r in rows:
            p = float(r["confidence"])
            y = 1.0 if r["ground_truth"] else 0.0
            idx = int(min(p, 0.9999) * 10)
            bins.setdefault(idx, []).append((p, y))
        ece = 0.0
        for bucket in bins.values():
            mean_p = sum(p for p, _ in bucket) / len(bucket)
            mean_y = sum(y for _, y in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(mean_p - mean_y)
        return ece

    def summary(self) -> dict[str, float]:
        """返回汇总统计。"""
        if not self._results:
            return {
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "total_scenarios": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "brier_score": 0.0,
                "ece": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            }

        accuracies = [r["accuracy"] for r in self._results if "accuracy" in r]
        confidences = [r["confidence"] for r in self._results if "confidence" in r]
        briers = [r["brier"] for r in self._results if "brier" in r]

        tp = sum(r.get("tp", 0) for r in self._results)
        fp = sum(r.get("fp", 0) for r in self._results)
        tn = sum(r.get("tn", 0) for r in self._results)
        fn = sum(r.get("fn", 0) for r in self._results)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "avg_confidence": sum(confidences) / len(confidences)
            if confidences
            else 0.0,
            "total_scenarios": len(accuracies),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "brier_score": round(sum(briers) / len(briers), 4) if briers else 0.0,
            "ece": round(self._ece(), 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }


# ===========================================================================
# 经济价值评分（来自 CRPS 研究项目的 cost-loss 框架）
# ===========================================================================


class ValueEvaluator:
    """按钱评分：决策的经济价值（cost-loss 框架）。

    对二元防护决策（预警=花钱预防 C，不预警且事件发生=损失 L）：
        expense = C if 预警 else L * 事件是否发生
    价值分数（Richardson / Murphy 经济价值）：
        V = (E_clim - E_agent) / (E_clim - E_perfect)
    V=1 完美决策；V=0 等同气候学（基准策略）；V<0 比瞎猜还差。

    默认成本比 alpha = C/L（预防成本占损失的比例）按灾种设定，
    可通过构造参数覆盖。这是 EarthBench 从"对错评分"升级到
    "经济价值评分"的核心模块。
    """

    DEFAULT_COST_RATIO = {
        "fire": 0.2,    # 防火响应成本相对损失较低，宁可信其有
        "flood": 0.3,
        "drought": 0.1, # 干旱缓解措施便宜，损失巨大
        "heat": 0.3,
        "ecology": 0.3,
    }

    def __init__(self, cost_ratio: dict[str, float] | None = None):
        self.cost_ratio = dict(self.DEFAULT_COST_RATIO)
        if cost_ratio:
            self.cost_ratio.update(cost_ratio)

    def _expense(self, acted: bool, event: bool, ratio: float) -> float:
        if acted:
            return ratio
        return 1.0 if event else 0.0

    def score(self, results: list[dict]) -> dict:
        """对 BatchEvaluator 的输出列表计算经济价值。

        results 每项需含: predicted(bool), ground_truth(bool), category(str)。
        """
        rows = [r for r in results if "predicted" in r and "error" not in r]
        if not rows:
            return {"value_score": None, "n": 0}

        exp_agent = exp_perf = 0.0
        for r in rows:
            cat = r.get("category", "heat")
            ratio = self.cost_ratio.get(cat, 0.3)
            acted = bool(r["predicted"])
            event = bool(r["ground_truth"])
            exp_agent += self._expense(acted, event, ratio)
            exp_perf += min(ratio, 1.0 if event else 0.0)
        # 气候学基准：样本内更优的常数策略（永远预警 vs 永不预警二选一）
        n = len(rows)
        base_rate = sum(1 for r in rows if r["ground_truth"]) / n
        clim_always = sum(
            self.cost_ratio.get(r.get("category", "heat"), 0.3) for r in rows
        )
        clim_never = sum(1.0 for r in rows if r["ground_truth"])
        exp_clim = min(clim_always, clim_never)

        v = (
            (exp_clim - exp_agent) / (exp_clim - exp_perf)
            if exp_clim > exp_perf else 0.0
        )
        return {
            "value_score": round(v, 4),
            "n": n,
            "expense_agent": round(exp_agent / n, 4),
            "expense_climatology": round(exp_clim / n, 4),
            "expense_perfect": round(exp_perf / n, 4),
            "base_rate": round(base_rate, 3),
            "cost_ratio": self.cost_ratio,
        }
