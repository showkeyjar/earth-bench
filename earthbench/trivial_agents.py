"""平凡基线 Agent（基准卫生：给分数一个参照系）。

规则 baseline 在基础套件 100%、对抗套件 0%——这些数字单独看没有语境。
三个 trivial agent 提供「不做任何推理」的下限参照：

- AlwaysAlertAgent：永远预警（宁可错报不可漏报的极端）
- NeverAlertAgent：永远不预警（零成本极端）
- RandomAgent：确定性伪随机（seed 稳定，可复现）

用途：
- 对抗套件上 always 的 FP=8/FN=0 与规则 agent 的双向全败恰好互补，
  说明规则 agent 的失败模式不是类别不平衡造成的；
- 任何新 agent 的得分应同时报告与这三个下限的相对位置；
- 发布日报的「基准体检」段（publish_pipeline）引用。

注意：RandomAgent 的随机性来自 case 派生哈希（region+template+观测数），
同套件内逐例独立、跨运行可复现（无全局状态）。
"""
from __future__ import annotations

import hashlib

from .models import DecisionOutput

__all__ = ["AlwaysAlertAgent", "NeverAlertAgent", "RandomAgent"]


class AlwaysAlertAgent:
    """永远预警的平凡基线。"""

    name = "always-alert"

    def decide(self, context) -> DecisionOutput:
        return DecisionOutput(
            context=context,
            decision=True,
            confidence=0.5,  # 无信息置信度
            evidence_summary={"policy": "always"},
            rationale="平凡基线：无条件预警",
        )


class NeverAlertAgent:
    """永远不预警的平凡基线。"""

    name = "never-alert"

    def decide(self, context) -> DecisionOutput:
        return DecisionOutput(
            context=context,
            decision=False,
            confidence=0.5,
            evidence_summary={"policy": "never"},
            rationale="平凡基线：无条件不预警",
        )


class RandomAgent:
    """确定性伪随机基线（seed 稳定，跨运行可复现）。"""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _coin(self, context) -> bool:
        key = f"{self.seed}|{context.region}|{context.template.value}|{len(context.observations)}"
        return bool(hashlib.sha256(key.encode()).digest()[0] & 1)

    def decide(self, context) -> DecisionOutput:
        decision = self._coin(context)
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=0.5,
            evidence_summary={"policy": "random", "seed": self.seed},
            rationale=f"平凡基线：确定性伪随机（seed={self.seed}）",
        )
