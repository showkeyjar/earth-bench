"""CARS 概率决策 Agent — 集合预报 + 影响优先触发（热浪场景）。

来自 CRPS/CARS 研究项目（D:\\code\\ai\\CRPS）：
- 冻结 SeasonalCarsModel（t2m，2000–2016 训练，15 年验证 15/15 全胜）
- 推理输入为真实 NOAA 业务 GEFS c00 f048 预报（无需分析场，alpha_state=0）
- 决策规则（影响优先）：act iff P(有害高温日) ≥ C/L，其中有害高温日 =
  日均 ≥ 303 K（约日最高 35°C，国标黄色预警线）。即研究项目验证过的
  期望损失规则 act iff P(事件) ≥ 成本比 的可服务化形式。
- 核心原则：只播报对人类有害的事件，不播报统计异常。当月 +2σ 相对异常
  仅作严重度背景（昆明 p2σ=0.33 而 p_harm=0：相对暖日无害）。

概率证据表：earthbench/data/cars_probabilities.json（离线预计算，
每区域×有效日期一条，含危害概率、成员统计与气候基准）。

诚实披露：
- 概率为干球 t2m 口径；场景真值用湿球温度（两者高相关但不等同）
- 模型冻结于 2016 年训练数据；业务 GEFS 与再预报 v12 存在版本漂移
- 域外区域（如乌鲁木齐 87.6°E < 100°E）回退到简单温度阈值规则并降置信度
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .agents import AlertAgent, MultiAlertAgent
from .eval import ValueEvaluator
from .models import DecisionOutput, ScenarioContext

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "data" / "cars_probabilities.json"

# 触发规则（影响优先）：只播报对人类有害的事件，不播报统计异常。
#   有害事件优先用湿球口径：P(午后湿球 ≥ 27°C) ≥ P_TRIG（EarthBench 标准，
#   与场景真值同口径，由成员日最高 + c00 湿度经 Stull 公式派生）。
#   无湿球字段（旧概率表）时回退干球：P(日最高 ≥ 35°C) 或日均 ≥ 303K 近似。
#   当月 +2σ 相对异常仅作为严重度背景信息保留，不参与触发——
#   高原/凉爽城市的"相对暖日"统计上极端但无害（昆明 p2σ=0.33 而 p_harm=0）。
# 冬季对应：有害低温日 = 日最低 ≤ 0°C（结冰/冻害风险）。
# 口径区分（披露，Phase D 收尾）：CARS 的 p_harm_cold 服务的是「冻害」
# （绝对低温线，GB/T 20484 日最低档近似），与基准寒潮真值（24h 降幅 ×
# 日最低 AND（多窗口降幅 OR）是**两个不同的问题**——常态寒冷城市持续 ≤0°C 触发
# 冻害预警但并非寒潮（无骤降）。跨日降幅需前后两日集合配对，本通道
# 暂不服务，见 docs/expansion-plan.md「风险与开放问题」。
# P_TRIG = 热浪成本比 C/L（期望损失规则 act iff P(事件) ≥ C/L）。
# 单一来源：eval.ValueEvaluator.DEFAULT_COST_RATIO["heat"] —— 此前两处硬编码
# 0.3 仅靠人肉保持一致，现改为构造性同步，杜绝漂移。
P_TRIG = ValueEvaluator.DEFAULT_COST_RATIO["heat"]
WB_HARM_C = 27.0
HARM_HEAT_MAX_C = 35.0
HARM_HEAT_INTENSE_C = 37.0
HARM_COLD_MIN_C = 0.0
# 域外回退：国家级高温预警线（°C，日最高温）
FALLBACK_TMAX = 37.0


class CarsProbabilityTable:
    """加载并检索预计算的 CARS 概率证据。"""

    def __init__(self, path: Path | None = None):
        p = path or DATA_FILE
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            self.meta = {k: v for k, v in raw.items() if k != "records"}
            self.records = {
                (r["region"], r["valid_date"]): r for r in raw["records"]
            }
        else:  # 测试/无数据环境：空表，Agent 全部走回退路径
            self.meta = {}
            self.records = {}

    def lookup(self, region: str, valid_date: str) -> dict | None:
        return self.records.get((region, valid_date))


def _latest_obs_date(context: ScenarioContext) -> str | None:
    ts = [o.timestamp for o in context.observations if o.timestamp]
    return max(ts)[:10] if ts else None


def _latest_value(context: ScenarioContext, variable: str) -> float | None:
    vals = [
        (o.timestamp, o.value)
        for o in context.observations
        if o.variable == variable and o.timestamp
    ]
    return max(vals)[1] if vals else None


class CarsHeatAgent(AlertAgent):
    """热浪概率决策 Agent（CARS 集合 + 期望严重度规则）。"""

    def __init__(self, table: CarsProbabilityTable | None = None,
                 p_trig: float = P_TRIG, name: str = "CarsHeatAgent"):
        super().__init__(name)
        self.table = table or CarsProbabilityTable()
        self.p_trig = p_trig

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        valid = _latest_obs_date(context)
        rec = self.table.lookup(context.region, valid) if valid else None

        if rec is not None and rec.get("covered"):
            # 影响优先：湿球口径优先，缺则回退干球（日最高≥35°C 或日均近似）
            if "p_harm_wb" in rec:
                p_harm, basis = rec["p_harm_wb"], f"湿球≥{WB_HARM_C:.0f}°C"
            else:
                p_harm, basis = rec.get("p_harm_heat", 0.0), "日最高≥35°C"
            p_intense = rec.get("p_harm_heat_intense", 0.0)
            p_cold = rec.get("p_harm_cold", 0.0)
            decision = p_harm >= self.p_trig
            confidence = min(0.9, 0.25 + p_harm)
            evidence = {
                "source": "CARS t2m ensemble (frozen 2000-2016)",
                "forecast_valid_date": rec["valid_date"],
                "member_mean_k": rec["t2m_member_mean_k"],
                "member_max_k": rec["t2m_member_max_k"],
                "p_harm": p_harm,
                "harm_basis": basis,
                "p_harm_heat_intense": p_intense,
                "p_harm_cold": p_cold,
                "p_trigger": self.p_trig,
                "rule": "impact-first: act iff P(harmful) >= C/L",
                "severity_context_only": {
                    "p_exceed_2sigma": rec.get("p_exceed_2sigma"),
                    "note": "relative anomaly reported for severity context, "
                            "NOT used as trigger",
                },
                "caveats": "wet bulb from member daily-max + unperturbed c00 "
                           "RH (Stull); GEFS version drift disclosed",
            }
            rationale = (
                f"CARS 30成员集合（GEFS f048，{rec['valid_date']} 有效）："
                f"P(有害事件·{basis})={p_harm:.0%}"
                f"（强档日最高≥37°C 占 {p_intense:.0%}），"
                f"触发阈值 {self.p_trig:.0%}（=成本比 C/L）→ "
                + ("预警" if decision else "不预警")
                + f"（背景：P(超当月+2σ)={rec.get('p_exceed_2sigma', 0):.0%}，"
                  "仅作严重度参考不参与触发）"
            )
            return DecisionOutput(
                context=context, decision=decision,
                confidence=round(confidence, 3),
                evidence_summary=evidence, rationale=rationale,
            )

        # 回退：无概率覆盖（域外/日期缺失）→ 简单阈值 + 低置信度
        tmax = _latest_value(context, "temperature_max")
        if tmax is None:
            return DecisionOutput(
                context=context, decision=False, confidence=0.2,
                evidence_summary={"fallback": "no CARS coverage, no tmax obs"},
                rationale="无 CARS 概率覆���且无温度观测，保守不预警",
            )
        decision = tmax >= FALLBACK_TMAX
        return DecisionOutput(
            context=context, decision=decision, confidence=0.4,
            evidence_summary={
                "fallback": "outside CARS grid (100-140E/20-50N) or date miss",
                "temperature_max": tmax,
                "fallback_threshold_c": FALLBACK_TMAX,
            },
            rationale=(
                f"CARS 无覆盖，回退规则：日最高温 {tmax}°C "
                f"{'≥' if decision else '<'} {FALLBACK_TMAX}°C 国家高温预警线"
            ),
        )


class CarsMultiAgent(MultiAlertAgent):
    """四类场景通用 Agent：热浪走 CARS 概率路径，其余沿用规则引擎。"""

    def __init__(self, p_trig: float = P_TRIG, **kwargs):
        super().__init__(**kwargs)
        self.cars_heat = CarsHeatAgent(p_trig=p_trig)

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        category = (
            context.category.value
            if hasattr(context.category, "value")
            else str(context.category)
        )
        if category in ("heat", "ecology"):
            return self.cars_heat.decide(context)
        return super().decide(context)
