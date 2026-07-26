"""评测引擎。

参考 SIM 项目的实验框架，提供可量化评估的能力。
"""

from __future__ import annotations

from typing import Protocol

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

    def evaluate(self, prediction: DecisionOutput) -> dict[str, float]:
        """评估一次决策，返回度量指标。"""
        pred_label = 1.0 if prediction.decision else 0.0
        gt_label = 1.0 if self.ground_truth else 0.0

        is_correct = int(pred_label == gt_label)
        confidence_calibration = 1.0 - abs(pred_label - prediction.confidence)

        return {
            "accuracy": float(is_correct),
            "confidence_calibration": confidence_calibration,
            "ground_truth": self.ground_truth,
            "predicted": prediction.decision,
            "confidence": prediction.confidence,
        }


class BatchEvaluator:
    """批量评测引擎。

    对一组场景进行评估，汇总统计指标（准确率、精确率、召回率、F1等）。
    """

    def __init__(self):
        self._results: list[dict] = []

    def run(
        self,
        agent: DecisionAgent,
        contexts: list[ScenarioContext],
        ground_truths: list[bool],
    ) -> list[dict]:
        """运行批量评测。"""
        assert len(contexts) == len(ground_truths), "场景数量与真值数量不匹配"

        results = []
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

    def summary(self) -> dict[str, float]:
        """返回汇总统计。"""
        if not self._results:
            return {
                "accuracy": 0.0, "avg_confidence": 0.0, "total_scenarios": 0,
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "true_positives": 0, "false_positives": 0,
                "true_negatives": 0, "false_negatives": 0,
            }

        accuracies = [r["accuracy"] for r in self._results if "accuracy" in r]
        confidences = [r["confidence"] for r in self._results if "confidence" in r]

        tp = sum(r.get("tp", 0) for r in self._results)
        fp = sum(r.get("fp", 0) for r in self._results)
        tn = sum(r.get("tn", 0) for r in self._results)
        fn = sum(r.get("fn", 0) for r in self._results)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "total_scenarios": len(accuracies),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
