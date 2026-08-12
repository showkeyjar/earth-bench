"""Alert 决策 Agent 实现。

Phase 1 提供四类场景各自专用的规则引擎 baseline Agent。
每类场景都有自己的 Ground Truth 推导逻辑，作为 LLM Agent 的对标基准。

阈值校准: agents.py 的决策阈值可被 calibration 模块动态调整。
通过环境变量 EARTHBENCH_THRESHOLDS_JSON 指定 thresholds.json 路径,
agent 初始化时自动读取对应的校准阈值。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .models import DecisionOutput, ScenarioContext

logger = logging.getLogger(__name__)


# ============================================================================
# 阈值校准: 从 calibration 模块的 thresholds.json 读取动态阈值
# ============================================================================


def _load_calibrated_threshold(category: str, default: float) -> float:
    """从 thresholds.json 读取校准后的阈值。

    通过环境变量 EARTHBENCH_THRESHOLDS_JSON 指定文件路径。
    如果文件不存在或读取失败, 返回 default。
    """
    threshold_file = os.environ.get("EARTHBENCH_THRESHOLDS_JSON", "")
    if not threshold_file:
        return default

    try:
        p = Path(threshold_file)
        if not p.exists():
            return default
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        val = thresholds.get(category)
        if val is not None:
            return float(val)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.debug(f"Failed to load calibrated threshold for {category}: {e}")

    return default


# ===========================================================================
# 专用场景 Agent 接口
# ===========================================================================


class AlertAgent:
    """Alert Agent 基类 — 子类必须实现 decide()."""

    def __init__(self, name: str = "AlertAgent"):
        self.name = name

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        raise NotImplementedError


# ===========================================================================
# Fire Alert Agent
# ===========================================================================


class FireAlertAgent(AlertAgent):
    """森林火险预警 Agent（对标国家林草局 FWI 标准）。

    风险评分模型：
    - FWI（权重 0.40）：主导因子
    - 湿度（权重 0.20）：干燥助燃
    - 风速（权重 0.15）：强风助燃
    - 降雨（权重 -0.15 抑制）：降雨抵消风险
    - 温度（权重 0.15）：高温干燥

    决策阈值: risk_score >= 0.4 → YES (可被 calibration 模块动态调整)

    阈值优先级:
    1. 构造函数显式传入的 threshold 参数
    2. 环境变量 EARTHBENCH_THRESHOLDS_JSON 指定的 JSON 文件中的值
    3. 默认硬编码值 0.4
    """

    def __init__(
        self,
        fwi_threshold: float = 40.0,
        humidity_threshold: float = 20.0,
        wind_threshold: float = 12.0,
        rainfall_suppress: float = 10.0,
        decision_threshold: float | None = None,
    ):
        super().__init__("FireAlertAgent")
        self.fwi_threshold = fwi_threshold
        self.humidity_threshold = humidity_threshold
        self.wind_threshold = wind_threshold
        self.rainfall_suppress = rainfall_suppress
        # 阈值: 显式参数 > calibration 文件 > 默认 0.4
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("fire", 0.4)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        fwi_vals = [o.value for o in obs if o.variable == "FWI"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        wind_vals = [o.value for o in obs if o.variable == "wind_speed"]
        rain_vals = [o.value for o in obs if o.variable == "rainfall"]
        temp_vals = [o.value for o in obs if o.variable == "temperature"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # FWI — 与 GT 推导公式对齐: min(fwi / 60, 1.0)
        if fwi_vals:
            avg_fwi = sum(fwi_vals) / len(fwi_vals)
            fwi_risk = min(avg_fwi / 60.0, 1.0)
            components.append((fwi_risk, 0.40, "FWI"))
            evidence["FWI"] = round(fwi_risk, 3)

        # 湿度 — 与 GT 对齐: max(0, (100 - hum) / 100)
        if hum_vals:
            avg_hum = sum(hum_vals) / len(hum_vals)
            hum_risk = max(0, (100 - avg_hum) / 100)
            components.append((hum_risk, 0.20, "humidity"))
            evidence["humidity"] = round(hum_risk, 3)

        # 风速 — 与 GT 对齐: min(wind / 20, 1.0)
        if wind_vals:
            avg_wind = sum(wind_vals) / len(wind_vals)
            wind_risk = min(avg_wind / 20.0, 1.0)
            components.append((wind_risk, 0.15, "wind_speed"))
            evidence["wind_speed"] = round(wind_risk, 3)

        # 温度 — 与 GT 对齐: max(0, (temp - 20) / 30)
        if temp_vals:
            avg_temp = sum(temp_vals) / len(temp_vals)
            temp_risk = max(0, (avg_temp - 20) / 30.0)
            components.append((temp_risk, 0.15, "temperature"))
            evidence["temperature"] = round(temp_risk, 3)

        # 降雨（抑制因素）
        avg_rain = 0.0
        if rain_vals:
            avg_rain = sum(rain_vals) / len(rain_vals)
            if avg_rain > self.rainfall_suppress:
                suppression = min(avg_rain / 30.0, 1.0)
                evidence["rainfall_suppression"] = -round(suppression * 0.6, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="FireAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        # 降雨抑制（在归一化后应用）
        if rain_vals and avg_rain > self.rainfall_suppress:
            suppression = min(avg_rain / 30.0, 1.0)
            risk_score *= 1.0 - suppression * 0.6

        # 统计用于 rationale
        avg_fwi_val = sum(fwi_vals) / len(fwi_vals) if fwi_vals else 0
        avg_hum_val = sum(hum_vals) / len(hum_vals) if hum_vals else 0
        avg_wind_val = sum(wind_vals) / len(wind_vals) if wind_vals else 0

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"FireAlert: FWI={avg_fwi_val:.1f}, 湿度={avg_hum_val:.1f}%, "
                f"风速={avg_wind_val:.1f}m/s, 降雨={avg_rain:.1f}mm, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Flood Alert Agent
# ===========================================================================


class FloodAlertAgent(AlertAgent):
    """洪涝预警 Agent（对标《国家防汛抗旱应急预案》暴雨标准）。

    风险评分模型：
    - 24h 降雨（权重 0.35）：>=50mm 暴雨标准
    - 6h 短时强降雨（权重 0.25）：>=30mm
    - 土壤湿度饱和（权重 0.15）：>0.85 放大效应
    - 水位超警戒（权重 0.25）：>警戒水位

    决策阈值: risk_score >= 0.45 → YES (可被 calibration 模块动态调整)
    """

    def __init__(
        self,
        water_level_critical: float = 10.0,
        decision_threshold: float | None = None,
    ):
        super().__init__("FloodAlertAgent")
        self.water_level_critical = water_level_critical
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("flood", 0.45)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        rain24_vals = [o.value for o in obs if "rainfall_24h" in o.variable]
        rain6_vals = [o.value for o in obs if "rainfall_6h" in o.variable]
        soil_vals = [o.value for o in obs if "soil_moisture" in o.variable]
        level_obs_list = [o for o in obs if "water_level" in o.variable]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # 24h 降雨
        if rain24_vals:
            avg_rain24 = sum(rain24_vals) / len(rain24_vals)
            rain24_risk = min(avg_rain24 / 100.0, 1.0)
            components.append((rain24_risk, 0.35, "rainfall_24h"))
            evidence["rainfall_24h"] = round(rain24_risk, 3)

        # 6h 短时强降雨
        if rain6_vals:
            avg_rain6 = sum(rain6_vals) / len(rain6_vals)
            rain6_risk = min(avg_rain6 / 50.0, 1.0)
            components.append((rain6_risk, 0.25, "rainfall_6h"))
            evidence["rainfall_6h"] = round(rain6_risk, 3)

        # 土壤湿度
        if soil_vals:
            avg_soil = sum(soil_vals) / len(soil_vals)
            soil_sat_risk = max(0, (avg_soil - 0.6) / 0.4)
            components.append((soil_sat_risk, 0.15, "soil_moisture"))
            evidence["soil_moisture"] = round(soil_sat_risk, 3)

        # 水位
        if level_obs_list:
            avg_level = sum(lo.value for lo in level_obs_list) / len(level_obs_list)
            level_risk = min(avg_level / self.water_level_critical, 1.0)
            components.append((level_risk, 0.25, "water_level"))
            evidence["water_level"] = round(level_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="FloodAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        # 水位趋势检测（与 GT 推导对齐：设风险下限而非加 bonus）
        # 关键：必须按时间戳排序后再判断趋势，否则观测列表顺序可能不一致
        water_t_bonus = 0.0
        if len(level_obs_list) >= 3:
            sorted_levels = sorted(level_obs_list, key=lambda o: o.timestamp)
            levels = [lo.value for lo in sorted_levels]
            increasing = all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))
            latest = levels[-1]
            if increasing and latest > 10.0:
                water_t_bonus = 0.75  # 与 GT 一致：设风险下限 0.75
            elif increasing and latest >= 9.0:
                water_t_bonus = 0.50  # 与 GT 一致：设风险下限 0.50

        if water_t_bonus > 0:
            risk_score = max(risk_score, water_t_bonus)
            evidence["water_trend"] = water_t_bonus

        # 统计用于 rationale
        avg_rain24_val = sum(rain24_vals) / len(rain24_vals) if rain24_vals else 0
        avg_rain6_val = sum(rain6_vals) / len(rain6_vals) if rain6_vals else 0
        avg_soil_val = sum(soil_vals) / len(soil_vals) if soil_vals else 0
        avg_level_val = (
            sum(lo.value for lo in level_obs_list) / len(level_obs_list)
            if level_obs_list
            else 0
        )

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"FloodAlert: 24h={avg_rain24_val:.1f}mm, "
                f"6h={avg_rain6_val:.1f}mm, 土壤={avg_soil_val:.2f}, "
                f"水位={avg_level_val:.1f}m, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Drought Alert Agent
# ===========================================================================


class DroughtAlertAgent(AlertAgent):
    """干旱预警 Agent（对标气象干旱标准）。

    风险评分模型：
    - SPI（标准化降水指数，权重 0.30）：< -1.0 中度干旱
    - Palmer 指数（权重 0.25）：< -0.5 干旱
    - 长期低湿度（权重 0.15）：< 25%
    - 月降雨严重不足（权重 0.15）：< 50mm
    - NDVI 植被胁迫（权重 0.15）：< 0.3

    决策阈值: risk_score >= 0.4 → YES (可被 calibration 模块动态调整)
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("DroughtAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("drought", 0.4)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        spi_vals = [o.value for o in obs if o.variable == "SPI"]
        palm_vals = [o.value for o in obs if o.variable == "palmer_index"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        rain_vals = [o.value for o in obs if o.variable == "rainfall_monthly"]
        ndvi_vals = [o.value for o in obs if o.variable == "NDVI"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # SPI
        if spi_vals:
            avg_spi = sum(spi_vals) / len(spi_vals)
            spi_risk = max(0, (-avg_spi - 0.5) / 2.0)
            components.append((spi_risk, 0.30, "SPI"))
            evidence["SPI"] = round(spi_risk, 3)

        # Palmer
        if palm_vals:
            avg_palmer = sum(palm_vals) / len(palm_vals)
            palm_risk = max(0, (-avg_palmer - 0.25) / 1.5)
            components.append((palm_risk, 0.25, "Palmer"))
            evidence["Palmer"] = round(palm_risk, 3)

        # 湿度
        if hum_vals:
            avg_hum = sum(hum_vals) / len(hum_vals)
            hum_risk = max(0, (30 - avg_hum) / 30.0)
            components.append((hum_risk, 0.15, "humidity"))
            evidence["humidity"] = round(hum_risk, 3)

        # 月降雨 — 阈值与 GT 推导保持一致（30mm）
        if rain_vals:
            avg_rain = sum(rain_vals) / len(rain_vals)
            rain_risk = max(0, (30 - avg_rain) / 30.0)
            components.append((rain_risk, 0.15, "rainfall_monthly"))
            evidence["rainfall_monthly"] = round(rain_risk, 3)

        # NDVI
        if ndvi_vals:
            avg_ndvi = sum(ndvi_vals) / len(ndvi_vals)
            ndvi_risk = max(0, (0.5 - avg_ndvi) / 0.3)
            components.append((ndvi_risk, 0.15, "NDVI"))
            evidence["NDVI"] = round(ndvi_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="DroughtAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        # 统计用于 rationale
        avg_spi_val = sum(spi_vals) / len(spi_vals) if spi_vals else 0
        avg_palmer_val = sum(palm_vals) / len(palm_vals) if palm_vals else 0
        avg_hum_val = sum(hum_vals) / len(hum_vals) if hum_vals else 0
        avg_rain_val = sum(rain_vals) / len(rain_vals) if rain_vals else 0
        avg_ndvi_val = sum(ndvi_vals) / len(ndvi_vals) if ndvi_vals else 0

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"DroughtAlert: SPI={avg_spi_val:.2f}, Palmer={avg_palmer_val:.2f}, "
                f"湿度={avg_hum_val:.1f}%, 月降雨={avg_rain_val:.1f}mm, "
                f"NDVI={avg_ndvi_val:.2f}, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# HeatWave Alert Agent
# ===========================================================================


class HeatWaveAlertAgent(AlertAgent):
    """热浪预警 Agent（对标《中央气象台高温预警信号》）。

    风险评分模型：
    - 最高温（权重 0.35）：>=35°C
    - 持续天数（权重 0.25）：从时序温度记录中推断 ≥3 天持续高温
    - 湿球温度（权重 0.25）：>27°C 致死阈值
    - 湿度（权重 0.15）：高湿加剧闷热

    决策阈值: risk_score >= 0.4 → YES (可被 calibration 模块动态调整)

    关键改进：从多时间戳温度观测中推断"连续高温天数"，
    替代外部 heat_duration_days 参数，使 Agent 能在真实场景中
    仅凭时序温度数据做出热浪预警判断。
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("HeatWaveAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("heat", 0.4)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        temp_vals = [o.value for o in obs if o.variable == "temperature_max"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        wbt_vals = [o.value for o in obs if o.variable == "wet_bulb_temp"]
        duration_vals = [o.value for o in obs if o.variable == "heat_duration_days"]
        temp_ts = [
            (o.timestamp, o.value) for o in obs if o.variable == "temperature_max"
        ]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # --- 最高温 — 与 GT 对齐: max(0, (temp - 28) / 12.0), 权重 0.35 ---
        temp_candidates = (
            temp_vals
            if temp_vals
            else [o.value for o in obs if o.variable == "temperature"]
        )
        if temp_candidates:
            avg_temp = sum(temp_candidates) / len(temp_candidates)
            temp_risk = max(0, (avg_temp - 28) / 12.0)
            weight_key = "temperature_max" if temp_vals else "temperature"
            components.append((temp_risk, 0.35, weight_key))
            evidence[weight_key] = round(temp_risk, 3)
        else:
            avg_temp = 0

        # --- 持续天数 — 与 GT 对齐: min(duration / 5.0, 1.0), 权重 0.25 ---
        # 优先使用观测中的 heat_duration_days，否则从时序温度推断
        if duration_vals:
            heat_duration_days = int(sum(duration_vals) / len(duration_vals))
        elif temp_ts:
            temp_ts.sort(key=lambda x: x[0])
            consecutive_high = 0
            prev_date = None
            for ts, tv in temp_ts:
                if tv >= 35.0:
                    # 提取日期部分用于连续性检查
                    cur_date = ts[:10] if len(ts) >= 10 else ts
                    if prev_date is None or cur_date == prev_date:
                        consecutive_high += 1
                    elif cur_date > prev_date:
                        # 日期递增，继续计数
                        consecutive_high += 1
                    else:
                        # 日期回退，重置
                        consecutive_high = 1
                    prev_date = cur_date
            heat_duration_days = max(consecutive_high, 1)
        elif context.horizon_hours >= 72:
            heat_duration_days = min(context.horizon_hours // 24, 5)
        else:
            heat_duration_days = 1

        duration_factor = min(heat_duration_days / 5.0, 1.0)
        components.append((duration_factor, 0.25, "heat_duration_days"))
        evidence["heat_duration_days"] = round(heat_duration_days)

        # --- 湿球温度 — 与 GT 对齐: max(0, (wb - 23) / 8.0), 权重 0.25 ---
        if wbt_vals:
            avg_wbt = sum(wbt_vals) / len(wbt_vals)
            wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)
            components.append((wbt_risk, 0.25, "wet_bulb_temp"))
            evidence["wet_bulb_temp"] = round(wbt_risk, 3)
        else:
            avg_wbt = None

        # --- 湿度 — 与 GT 对齐: max(0, (hum - 50) / 50.0), 权重 0.15 ---
        if hum_vals:
            avg_hum = sum(hum_vals) / len(hum_vals)
            hum_risk = max(0, (avg_hum - 50) / 50.0)
            components.append((hum_risk, 0.15, "humidity"))
            evidence["humidity"] = round(hum_risk, 3)
        else:
            avg_hum = 0

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="HeatWaveAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"HeatWaveAlert: 最高温={avg_temp:.1f}°C, "
                f"湿球={'%.1f' % avg_wbt if avg_wbt is not None else 'N/A'}°C, "
                f"湿度={avg_hum:.1f}%, 持续≈{heat_duration_days}天, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# MultiAlertAgent — 场景路由
# ===========================================================================


class MultiAlertAgent(AlertAgent):
    """多场景 Alert Agent — 根据场景类别自动路由到专用引擎。

    这是 FireBench → AlertBench 升级后的核心 Agent：
    - category == fire → FireAlertAgent
    - category == flood → FloodAlertAgent
    - category == drought → DroughtAlertAgent
    - category == heat/ecology → HeatWaveAlertAgent
    - 其他 → 降级为 RuleBasedAgent (仅 fire)
    """

    def __init__(
        self,
        fwi_threshold: float = 40.0,
        humidity_threshold: float = 20.0,
        wind_threshold: float = 12.0,
        rainfall_suppress: float = 10.0,
        fire_threshold: float | None = None,
        flood_threshold: float | None = None,
        drought_threshold: float | None = None,
        heat_threshold: float | None = None,
    ):
        super().__init__("MultiAlertAgent")
        self.fire_agent = FireAlertAgent(
            fwi_threshold=fwi_threshold,
            humidity_threshold=humidity_threshold,
            wind_threshold=wind_threshold,
            rainfall_suppress=rainfall_suppress,
            decision_threshold=fire_threshold,
        )
        self.flood_agent = FloodAlertAgent(decision_threshold=flood_threshold)
        self.drought_agent = DroughtAlertAgent(decision_threshold=drought_threshold)
        self.heat_agent = HeatWaveAlertAgent(decision_threshold=heat_threshold)

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """根据场景类别路由到对应的专用 Agent。"""
        category_map = {
            "fire": self.fire_agent,
            "flood": self.flood_agent,
            "drought": self.drought_agent,
            "ecology": self.heat_agent,
            "heat": self.heat_agent,
        }

        category = (
            context.category.value
            if hasattr(context.category, "value")
            else str(context.category)
        )

        agent = category_map.get(category, None)
        if agent is None:
            # 降级：fallback 到 fire agent（不推荐，仅兼容旧逻辑）
            agent = self.fire_agent

        return agent.decide(context)


# ===========================================================================
# Legacy — RuleBasedAgent 保留以兼容旧代码
# ===========================================================================


class RuleBasedAgent(FireAlertAgent):
    """兼容别名：旧 RuleBasedAgent 现在是 FireAlertAgent 的子类。

    保留旧接口名称，确保现有代码不 break。
    但推荐使用 FireAlertAgent（新名称更准确）。
    """

    def __init__(self, **kwargs):
        # 兼容旧参数名
        super().__init__(
            fwi_threshold=kwargs.get("fwi_threshold", 40.0),
            humidity_threshold=kwargs.get("humidity_threshold", 20.0),
            wind_threshold=kwargs.get("wind_threshold", 12.0),
            rainfall_suppress=kwargs.get("rainfall_suppress", 10.0),
        )


class LLMDecisionAgent(AlertAgent):
    """基于 LLM 的决策 Agent（Phase 2+）。

    将场景上下文组装为 prompt，调用 LLM（Gemini/Ollama/OpenAI），
    解析返回的结构化决策。

    优先级: CARM/Mustard集成 > OpenAI兼容API > Ollama本地模型 > 启发式回退
    """

    def __init__(
        self,
        model_name: str = "qwen3:14b",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__("LLMDecisionAgent")
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        import os as _os

        if self.api_key or _os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if self.base_url or _os.environ.get("OLLAMA_BASE_URL"):
            return "ollama"
        return "none"

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        from .templates import TemplateEngine

        valid, msg = TemplateEngine.validate_context(context)
        if not valid:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={"error": msg},
                rationale=f"验证失败: {msg}",
            )

        # 构建 prompt
        tmpl_labels = {
            "alert": "是否预警",
            "dispatch": "是否调度",
            "upgrade": "是否升级",
            "close": "是否关闭",
            "recover": "是否恢复",
        }
        tmpl_label = tmpl_labels.get(context.template.value, "未知")

        obs_lines = []
        for o in context.observations:
            obs_lines.append(
                f"- [{o.source}] {o.variable}={o.value}{o.unit} (conf={o.confidence:.2f}) @ {o.timestamp}"
            )
        obs_text = "\n".join(obs_lines)

        prompt = (
            f"[EarthBench Alert 决策任务]\n"
            f"区域: {context.region}\n"
            f"时间窗口: {context.horizon_hours}小时\n"
            f"决策模板: {tmpl_label}\n"
            f"\n证据列表:\n{obs_text}\n"
            f"\n请做出二元决策：是或否？\n"
            f"只回答 YES 或 NO（大写），然后简要说明理由和置信度。\n"
            f"格式：决策:YES/NO\n置信度:0.xx\n理由:..."
        )

        llm_answer = None
        llm_confidence = 0.5
        llm_rationale = ""

        # 尝试 OpenAI 兼容 API
        if self._provider == "openai":
            try:
                llm_answer, llm_confidence, llm_rationale = self._call_openai(prompt)
            except Exception as e:
                logger.warning(f"OpenAI call failed: {e}", exc_info=True)

        # 尝试 Ollama
        if llm_answer is None:
            try:
                llm_answer, llm_confidence, llm_rationale = self._call_ollama(prompt)
            except Exception as e:
                logger.warning(f"Ollama call failed: {e}", exc_info=True)

        if llm_answer is not None:
            decision = llm_answer in ("yes", "是")
            return DecisionOutput(
                context=context,
                decision=decision,
                confidence=llm_confidence,
                evidence_summary={
                    "llm_model": self.model_name,
                    "provider": self._provider,
                },
                rationale=f"LLM ({self.model_name}): {llm_answer}. {llm_rationale}",
            )

        # 回退到启发式
        return self._heuristic_fallback(context)

    def _call_openai(self, prompt: str) -> tuple[str | None, float, str]:
        import os
        from openai import OpenAI

        base_url = self.base_url or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
        return self._parse_llm_response(text)

    def _call_ollama(self, prompt: str) -> tuple[str | None, float, str]:
        import os
        import urllib.request
        import json

        base_url = self.base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("message", {}).get("content", "")
        return self._parse_llm_response(text)

    @staticmethod
    def _parse_llm_response(text: str) -> tuple[str | None, float, str]:
        import re

        content = text.strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        elif "<antThinking>" in content:
            content = content.split("<antThinking>", 1)[1]

        lower = content.lower()

        dec_match = re.search(r"决策[:：]\s*(yes|no|是|否)", lower)
        if dec_match:
            ans = dec_match.group(1)
            if ans in ("yes", "是"):
                return "yes", 0.85, content[:300]
            else:
                return "no", 0.85, content[:300]

        first_line = (
            lower.split("\n")[0].strip() if "\n" in content else lower[:100].strip()
        )
        if re.match(r"^(yes|no)\b", first_line):
            return ("yes" if first_line.startswith("y") else "no"), 0.8, content[:300]

        if re.search(r"应.*预警|建议.*预警|必须.*预警|需要.*预警|激活.*响应", lower):
            return "yes", 0.7, content[:300]
        if re.search(r"不应.*预警|不建议.*预警|不需要.*预警|不.*发出.*预警", lower):
            return "no", 0.7, content[:300]

        return None, 0.5, content[:300]

    def _heuristic_fallback(self, context: ScenarioContext) -> DecisionOutput:
        """当 LLM 不可用时，使用启发式规则引擎作为兜底。"""
        try:
            from earthbench.integrations import CARMBridge

            bridge = CARMBridge()
            return bridge._decide_heuristic(context)
        except Exception as e:
            logger.warning(f"Heuristic fallback failed: {e}", exc_info=True)
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={"fallback_error": str(e)},
                rationale=f"Heuristic fallback failed: {e}",
            )
