"""决策模板引擎。

每个模板定义了：
1. 输入观测的最低要求
2. 验证逻辑
3. 解释格式
"""

from __future__ import annotations

from .models import DecisionTemplate, Observation, ScenarioContext


class TemplateEngine:
    """决策模板引擎基类。"""

    MIN_OBSERVATIONS = {
        DecisionTemplate.ALERT: 3,
        DecisionTemplate.DISPATCH: 4,
        DecisionTemplate.UPGRADE: 5,
        DecisionTemplate.CLOSE: 3,
        DecisionTemplate.RECOVER: 4,
    }

    @staticmethod
    def validate_context(context: ScenarioContext) -> tuple[bool, str]:
        """验证场景上下文是否满足模板要求。

        Alert 场景要求：至少 3 条观测数据即可判定，不强制要求时序数据。
        时序趋势（L4）是加分项，不是硬性要求。
        """
        tpl = context.template
        min_obs = TemplateEngine.MIN_OBSERVATIONS.get(tpl, 3)

        if len(context.observations) < min_obs:
            return False, (
                f"观测数据不足：需要至少 {min_obs} 条，"
                f"当前 {len(context.observations)} 条"
            )

        return True, "验证通过"

    @staticmethod
    def get_explanation_template(template: DecisionTemplate) -> str:
        """获取自然语言解释模板。"""
        templates = {
            DecisionTemplate.ALERT: (
                "基于 {n} 条多源观测数据，未来 {h} 小时的决策分析如下："
            ),
            DecisionTemplate.DISPATCH: (
                "基于当前态势，资源调度方案分析（{n} 条观测支持）："
            ),
            DecisionTemplate.UPGRADE: (
                "风险等级升级评估（最近 {h} 小时趋势）："
            ),
            DecisionTemplate.CLOSE: (
                "封闭/暂停决策评估："
            ),
            DecisionTemplate.RECOVER: (
                "恢复/解除评估："
            ),
        }
        return templates.get(template, "决策分析：")
