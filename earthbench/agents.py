"""Alert 决策 Agent 实现。

Phase 1 提供四类场景各自专用的规则引擎 baseline Agent。
每类场景 Agent 使用独立的加权评分模型（归一化加权求和 + 阈值判定）做出决策；
Ground Truth 由 scenarios.py 的独立物理标准（国标分级查表）推导，两者相互独立。

阈值校准: agents.py 的决策阈值可被 calibration 模块动态调整。
通过环境变量 EARTHBENCH_THRESHOLDS_JSON 指定 thresholds.json 路径,
agent 初始化时自动读取对应的校准阈值。
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from .llm import domain_hint, parse_yes_no
from .models import DecisionOutput, DecisionTemplate, ScenarioContext
from .scenarios import infer_exposure_class

logger = logging.getLogger(__name__)


def _max_consecutive_days(date_tokens: list[str]) -> int:
    """从观测时间戳推断最大「连续高温」日历天数。

    同一日多条读数去重；仅严格相邻的日期计入连续（修复此前同一日多条
    读数被重复累加、隔日也当成连续的误判）。
    """
    from datetime import date as _date

    days: set = set()
    for tok in date_tokens:
        ts = tok[:10] if isinstance(tok, str) and len(tok) >= 10 else str(tok)
        try:
            days.add(_date.fromisoformat(ts))
        except ValueError:
            continue
    if not days:
        return max(1, len(date_tokens))
    ordered = sorted(days)
    best = run = 1
    for a, b in zip(ordered, ordered[1:]):
        if (b - a).days == 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


# ============================================================================
# 阈值校准: 从 calibration 模块的 thresholds.json 读取动态阈值
# ============================================================================


@lru_cache(maxsize=16)
def _read_thresholds_cached(threshold_file: str) -> dict:
    """按文件路径缓存 thresholds.json 内容（发布流程内 8 个子 agent 各读一遍 → 1 遍）。

    抛出的异常由调用方降级处理。测试若就地改写同一路径文件，需 _read_thresholds_cached.cache_clear()。
    """
    p = Path(threshold_file)
    if not p.exists():
        raise FileNotFoundError(threshold_file)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("thresholds", {})


def _load_calibrated_threshold(category: str, default: float) -> float:
    """从 thresholds.json 读取校准后的阈值。

    通过环境变量 EARTHBENCH_THRESHOLDS_JSON 指定文件路径。
    如果文件不存在或读取失败, 返回 default。
    """
    threshold_file = os.environ.get("EARTHBENCH_THRESHOLDS_JSON", "")
    if not threshold_file:
        return default

    try:
        thresholds = _read_thresholds_cached(threshold_file)
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

        # FWI 归一化（Agent 自身评分模型）: min(fwi / 60, 1.0)
        if fwi_vals:
            avg_fwi = sum(fwi_vals) / len(fwi_vals)
            fwi_risk = min(avg_fwi / 60.0, 1.0)
            components.append((fwi_risk, 0.40, "FWI"))
            evidence["FWI"] = round(fwi_risk, 3)

        # 湿度归一化（Agent 自身评分模型）: max(0, (100 - hum) / 100)
        if hum_vals:
            avg_hum = sum(hum_vals) / len(hum_vals)
            hum_risk = max(0, (100 - avg_hum) / 100)
            components.append((hum_risk, 0.20, "humidity"))
            evidence["humidity"] = round(hum_risk, 3)

        # 风速归一化（Agent 自身评分模型）: min(wind / 20, 1.0)
        if wind_vals:
            avg_wind = sum(wind_vals) / len(wind_vals)
            wind_risk = min(avg_wind / 20.0, 1.0)
            components.append((wind_risk, 0.15, "wind_speed"))
            evidence["wind_speed"] = round(wind_risk, 3)

        # 温度归一化（Agent 自身评分模型）: max(0, (temp - 20) / 30)
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

        # 水位趋势检测（设风险下限而非加 bonus）
        # 关键：必须按时间戳排序后再判断趋势，否则观测列表顺序可能不一致
        water_t_bonus = 0.0
        if len(level_obs_list) >= 3:
            sorted_levels = sorted(level_obs_list, key=lambda o: o.timestamp)
            levels = [lo.value for lo in sorted_levels]
            increasing = all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))
            latest = levels[-1]
            if increasing and latest > 10.0:
                water_t_bonus = 0.75  # 设风险下限 0.75（Agent 评分模型）
            elif increasing and latest >= 9.0:
                water_t_bonus = 0.50  # 设风险下限 0.50（Agent 评分模型）

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

        # 月降雨 — 阈值 30mm（Agent 评分模型自设）
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

        # --- 最高温归一化: max(0, (temp - 28) / 12.0), 权重 0.35 ---
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

        # --- 持续天数归一化: min(duration / 5.0, 1.0), 权重 0.25 ---
        # 优先使用观测中的 heat_duration_days，否则从时序温度推断
        if duration_vals:
            heat_duration_days = int(sum(duration_vals) / len(duration_vals))
        elif temp_ts:
            heat_duration_days = _max_consecutive_days(
                [ts for ts, tv in temp_ts if tv >= 35.0]
            )
        elif context.horizon_hours >= 72:
            heat_duration_days = min(context.horizon_hours // 24, 5)
        else:
            heat_duration_days = 1

        duration_factor = min(heat_duration_days / 5.0, 1.0)
        components.append((duration_factor, 0.25, "heat_duration_days"))
        evidence["heat_duration_days"] = round(heat_duration_days)

        # --- 湿球温度归一化: max(0, (wb - 23) / 8.0), 权重 0.25 ---
        if wbt_vals:
            avg_wbt = sum(wbt_vals) / len(wbt_vals)
            wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)
            components.append((wbt_risk, 0.25, "wet_bulb_temp"))
            evidence["wet_bulb_temp"] = round(wbt_risk, 3)
        else:
            avg_wbt = None

        # --- 湿度归一化: max(0, (hum - 50) / 50.0), 权重 0.15 ---
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
                f"湿球={f'{avg_wbt:.1f}' if avg_wbt is not None else 'N/A'}°C, "
                f"湿度={avg_hum:.1f}%, 持续≈{heat_duration_days}天, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Landslide Alert Agent
# ===========================================================================


class LandslideAlertAgent(AlertAgent):
    """滑坡/泥石流预警 Agent（对标地质灾害气象风险预警四级体系）。

    风险评分模型（线性加权 baseline，刻意不复制查表真值的通道结构）：
    - 1h 激发雨强（权重 0.35）：>= 25mm
    - 24h 降雨（权重 0.35）：/100mm 大暴雨口径
    - 土壤湿度（权重 0.10）：饱和度 /0.85
    - 前 3 日有效降雨（权重 0.10）：前期累积 /150mm
    - 地质易发性（权重 0.10）：静态分级 /3（3=高/2=中/1=低）

    决策阈值: risk_score >= 0.40 → YES (可被 calibration 模块动态调整)

    判别力说明（对抗套件 adv-landslide-* 锚定的两个失效模式）：
    - 低易发 + 激发雨强达标：易发性 AND 条件被加权平均抹掉 → 误报
    - 高易发 + 当日无雨但前期饱和累积：滞后通道权重过低 → 漏报
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("LandslideAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("landslide", 0.40)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        rain1_vals = [o.value for o in obs if o.variable == "rainfall_1h"]
        rain24_vals = [o.value for o in obs if o.variable == "rainfall_24h"]
        soil_vals = [o.value for o in obs if o.variable == "soil_moisture"]
        eff_vals = [o.value for o in obs if o.variable == "effective_rainfall_3d"]
        suscept_vals = [o.value for o in obs if o.variable == "susceptibility"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # 1h 激发雨强
        if rain1_vals:
            avg_rain1 = sum(rain1_vals) / len(rain1_vals)
            rain1_risk = min(avg_rain1 / 25.0, 1.0)
            components.append((rain1_risk, 0.35, "rainfall_1h"))
            evidence["rainfall_1h"] = round(rain1_risk, 3)

        # 24h 降雨
        if rain24_vals:
            avg_rain24 = sum(rain24_vals) / len(rain24_vals)
            rain24_risk = min(avg_rain24 / 100.0, 1.0)
            components.append((rain24_risk, 0.35, "rainfall_24h"))
            evidence["rainfall_24h"] = round(rain24_risk, 3)

        # 土壤湿度（饱和度）
        if soil_vals:
            avg_soil = sum(soil_vals) / len(soil_vals)
            soil_risk = max(0.0, min(avg_soil / 0.85, 1.0))
            components.append((soil_risk, 0.10, "soil_moisture"))
            evidence["soil_moisture"] = round(soil_risk, 3)

        # 前 3 日有效降雨（前期累积）
        if eff_vals:
            avg_eff = sum(eff_vals) / len(eff_vals)
            eff_risk = min(avg_eff / 150.0, 1.0)
            components.append((eff_risk, 0.10, "effective_rainfall_3d"))
            evidence["effective_rainfall_3d"] = round(eff_risk, 3)

        # 地质易发性（静态分级 3/2/1）
        if suscept_vals:
            avg_suscept = sum(suscept_vals) / len(suscept_vals)
            suscept_risk = min(avg_suscept / 3.0, 1.0)
            components.append((suscept_risk, 0.10, "susceptibility"))
            evidence["susceptibility"] = round(suscept_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="LandslideAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        avg_rain1_val = sum(rain1_vals) / len(rain1_vals) if rain1_vals else 0
        avg_rain24_val = sum(rain24_vals) / len(rain24_vals) if rain24_vals else 0
        avg_soil_val = sum(soil_vals) / len(soil_vals) if soil_vals else 0
        avg_eff_val = sum(eff_vals) / len(eff_vals) if eff_vals else 0

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"LandslideAlert: 1h={avg_rain1_val:.1f}mm, "
                f"24h={avg_rain24_val:.1f}mm, 土壤={avg_soil_val:.2f}, "
                f"前3日有效降雨={avg_eff_val:.0f}mm, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Typhoon Alert Agent
# ===========================================================================


class TyphoonAlertAgent(AlertAgent):
    """台风/大风预警 Agent（对标 GB/T 19201-2006 热带气旋等级蒲福氏分档）。

    风险评分模型（线性加权 baseline，刻意不复制查表真值的通道结构）：
    - 日均风（权重 0.40）：/17.2 m/s（8 级线）
    - 阵风（权重 0.25）：/24.5 m/s（10 级临设线）
    - 24h 降雨（权重 0.35）：/100mm 大暴雨口径（风雨耦合的线性近似）

    决策阈值: risk_score >= 0.45 → YES (可被 calibration 模块动态调整)

    判别力说明（对抗套件 adv-typhoon-* 锚定的两个失效模式）：
    - 风雨各差一线（8 级线差 0.7、暴雨线差 2mm，AND 缺一）：线性均值
      把次阈值混合推过阈值 → 误报
    - 日均低但阵风超标（雷雨大风/飑线）：决定性阵风因子权重被稀释 → 漏报
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("TyphoonAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("typhoon", 0.45)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        wind_vals = [o.value for o in obs if o.variable == "wind_speed"]
        gust_vals = [o.value for o in obs if o.variable == "wind_gust"]
        rain24_vals = [o.value for o in obs if o.variable == "rainfall_24h"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # 日均风（8 级线口径）
        if wind_vals:
            avg_wind = sum(wind_vals) / len(wind_vals)
            wind_risk = min(avg_wind / 17.2, 1.0)
            components.append((wind_risk, 0.40, "wind_speed"))
            evidence["wind_speed"] = round(wind_risk, 3)

        # 阵风（10 级临设线口径）
        if gust_vals:
            avg_gust = sum(gust_vals) / len(gust_vals)
            gust_risk = min(avg_gust / 24.5, 1.0)
            components.append((gust_risk, 0.25, "wind_gust"))
            evidence["wind_gust"] = round(gust_risk, 3)

        # 24h 降雨（风雨耦合的线性近似）
        if rain24_vals:
            avg_rain24 = sum(rain24_vals) / len(rain24_vals)
            rain24_risk = min(avg_rain24 / 100.0, 1.0)
            components.append((rain24_risk, 0.35, "rainfall_24h"))
            evidence["rainfall_24h"] = round(rain24_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="TyphoonAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        avg_wind_val = sum(wind_vals) / len(wind_vals) if wind_vals else 0
        avg_gust_val = sum(gust_vals) / len(gust_vals) if gust_vals else 0
        avg_rain24_val = sum(rain24_vals) / len(rain24_vals) if rain24_vals else 0

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"TyphoonAlert: 日均风={avg_wind_val:.1f}m/s, "
                f"阵风={avg_gust_val:.1f}m/s, 24h降雨={avg_rain24_val:.0f}mm, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# ColdWave Alert Agent
# ===========================================================================


class ColdWaveAlertAgent(AlertAgent):
    """寒潮/冰冻预警 Agent（对标 GB/T 20484-2017《冷空气等级》）。

    风险评分模型（线性加权 baseline，刻意不复制查表真值的 AND 结构）：
    - 日最低温（权重 0.40）：(4 - tmin)/8，越冷越高
    - 24h 降温幅度（权重 0.35）：drop/10
    - 风寒（权重 0.25）：wind/12

    决策阈值: risk_score >= 0.45 → YES (可被 calibration 模块动态调整)

    判别力说明（对抗套件 adv-cold-* 锚定的两个失效模式）：
    - 北方常态低温无降幅：绝对低温分量饱和主导 → 误报
      （这正是 cars_cities.json cold_alert_active 按城门控的物理含义）
    - 双线刚过（降幅 8.5 + 极值 3.8）：AND 双中档被均值稀释 → 漏报
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("ColdWaveAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("cold", 0.45)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        tmin_vals = [o.value for o in obs if o.variable == "temperature_min"]
        drop_vals = [o.value for o in obs if o.variable == "temperature_drop_24h"]
        wind_vals = [o.value for o in obs if o.variable == "wind_speed"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # 日最低温（越冷越高）
        if tmin_vals:
            avg_tmin = sum(tmin_vals) / len(tmin_vals)
            tmin_risk = max(0.0, min((4.0 - avg_tmin) / 8.0, 1.0))
            components.append((tmin_risk, 0.40, "temperature_min"))
            evidence["temperature_min"] = round(tmin_risk, 3)

        # 24h 降温幅度
        if drop_vals:
            avg_drop = sum(drop_vals) / len(drop_vals)
            drop_risk = max(0.0, min(avg_drop / 10.0, 1.0))
            components.append((drop_risk, 0.35, "temperature_drop_24h"))
            evidence["temperature_drop_24h"] = round(drop_risk, 3)

        # 风寒
        if wind_vals:
            avg_wind = sum(wind_vals) / len(wind_vals)
            wind_risk = max(0.0, min(avg_wind / 12.0, 1.0))
            components.append((wind_risk, 0.25, "wind_speed"))
            evidence["wind_speed"] = round(wind_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="ColdWaveAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        avg_tmin_val = sum(tmin_vals) / len(tmin_vals) if tmin_vals else 0
        avg_drop_val = sum(drop_vals) / len(drop_vals) if drop_vals else 0

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"ColdWaveAlert: 日最低={avg_tmin_val:.1f}°C, "
                f"24h降幅={avg_drop_val:.1f}°C, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Snow Alert Agent
# ===========================================================================


class SnowAlertAgent(AlertAgent):
    """暴雪/道路结冰预警 Agent（对标 GB/T 28592-2012 降雪等级附表）。

    风险评分模型（线性加权 baseline，刻意不复制查表真值的掩膜/AND 结构）：
    - 24h 降雪量（权重 0.35）：/10mm 暴雪线
    - 路面温度（权重 0.20）：(0 - road)/10，越冷越高
    - 24h 降水总量（权重 0.30）：/40mm —— 不区分雨雪的降水口径
    - 湿度（权重 0.15）：(hum - 50)/50，湿雪附着

    决策阈值: risk_score >= 0.45 → YES (可被 calibration 模块动态调整)

    判别力说明（对抗套件 adv-snow-* 锚定的两个失效模式）：
    - 雨非雪：冻结掩膜（气温 >= 0.5℃ 降水为雨）缺失，降水总量分量
      主导 → 误报（雨雪口径混淆）
    - 结冰双线刚过（中雪 2.6 × 路温 -2）：AND 双中低档被均值稀释 → 漏报
    """

    def __init__(self, decision_threshold: float | None = None):
        super().__init__("SnowAlertAgent")
        self.decision_threshold = (
            decision_threshold
            if decision_threshold is not None
            else _load_calibrated_threshold("snow", 0.45)
        )

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        snow_vals = [o.value for o in obs if o.variable == "snowfall_24h"]
        road_vals = [o.value for o in obs if o.variable == "road_surface_temp"]
        rain_vals = [o.value for o in obs if o.variable == "rainfall_24h"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]

        evidence: dict[str, float] = {}
        components: list[tuple[float, float, str]] = []

        # 24h 降雪量
        if snow_vals:
            avg_snow = sum(snow_vals) / len(snow_vals)
            snow_risk = max(0.0, min(avg_snow / 10.0, 1.0))
            components.append((snow_risk, 0.35, "snowfall_24h"))
            evidence["snowfall_24h"] = round(snow_risk, 3)

        # 路面温度（越冷越高）
        if road_vals:
            avg_road = sum(road_vals) / len(road_vals)
            road_risk = max(0.0, min((0.0 - avg_road) / 10.0, 1.0))
            components.append((road_risk, 0.20, "road_surface_temp"))
            evidence["road_surface_temp"] = round(road_risk, 3)

        # 24h 降水总量（不区分雨雪 —— 对抗用例锚定的口径混淆点）
        if rain_vals:
            avg_rain = sum(rain_vals) / len(rain_vals)
            rain_risk = max(0.0, min(avg_rain / 40.0, 1.0))
            components.append((rain_risk, 0.30, "rainfall_24h"))
            evidence["rainfall_24h"] = round(rain_risk, 3)

        # 湿度（湿雪附着）
        if hum_vals:
            avg_hum = sum(hum_vals) / len(hum_vals)
            hum_risk = max(0.0, min((avg_hum - 50.0) / 50.0, 1.0))
            components.append((hum_risk, 0.15, "humidity"))
            evidence["humidity"] = round(hum_risk, 3)

        if not components:
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={},
                rationale="SnowAlert: no relevant observations",
            )

        # 动态归一化权重
        total_weight = sum(w for _, w, _ in components)
        risk_score = sum(r * (w / total_weight) for r, w, _ in components)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= self.decision_threshold

        avg_snow_val = sum(snow_vals) / len(snow_vals) if snow_vals else 0
        avg_road_val = sum(road_vals) / len(road_vals) if road_vals else 0

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"SnowAlert: 24h降雪={avg_snow_val:.1f}mm, "
                f"路温={avg_road_val:.1f}°C, 风险分={risk_score:.3f}"
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
    - category == landslide → LandslideAlertAgent（Phase A1 扩展）
    - category == typhoon → TyphoonAlertAgent（Phase A2 扩展）
    - category == cold → ColdWaveAlertAgent（Phase B1 扩展）
    - category == snow → SnowAlertAgent（Phase B2 扩展）
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
        landslide_threshold: float | None = None,
        typhoon_threshold: float | None = None,
        cold_threshold: float | None = None,
        snow_threshold: float | None = None,
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
        self.landslide_agent = LandslideAlertAgent(decision_threshold=landslide_threshold)
        self.typhoon_agent = TyphoonAlertAgent(decision_threshold=typhoon_threshold)
        self.cold_agent = ColdWaveAlertAgent(decision_threshold=cold_threshold)
        self.snow_agent = SnowAlertAgent(decision_threshold=snow_threshold)

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """根据决策模板与场景类别路由到对应的决策逻辑。"""
        if context.template != DecisionTemplate.ALERT:
            return self.decide_template(context)
        return self._decide_alert(context)

    def _decide_alert(self, context: ScenarioContext) -> DecisionOutput:
        """ALERT 模板：根据场景类别路由到对应的专用 Agent。"""
        category_map = {
            "fire": self.fire_agent,
            "flood": self.flood_agent,
            "drought": self.drought_agent,
            "ecology": self.heat_agent,
            "heat": self.heat_agent,
            "landslide": self.landslide_agent,
            "typhoon": self.typhoon_agent,
            "cold": self.cold_agent,
            "snow": self.snow_agent,
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

    # ------------------------------------------------------------------
    # 决策模板闭环（Phase C）：dispatch / upgrade / close / recover
    # 基线统一「线性 ALERT 输出的置信度/判定」做启发式，刻意不复制真值
    # 的 AND 查表与持续时间窗结构 —— 边界用例因此双向出错，构成判别力。
    # ------------------------------------------------------------------

    def decide_template(self, context: ScenarioContext) -> DecisionOutput:
        """按模板分派到对应 baseline 决策（Phase C）。"""
        handler = {
            DecisionTemplate.DISPATCH: self._decide_dispatch,
            DecisionTemplate.UPGRADE: self._decide_upgrade,
            DecisionTemplate.CLOSE: self._decide_close,
            DecisionTemplate.RECOVER: self._decide_recover,
        }
        fn = handler.get(context.template)
        if fn is None:
            return self._decide_alert(context)
        return fn(context)

    def _decide_dispatch(self, context: ScenarioContext) -> DecisionOutput:
        """DISPATCH baseline：ALERT 判定且置信度 >= 0.55（忽略 AND 附加条件）。"""
        base = self._decide_alert(context)
        decision = base.decision and base.confidence >= 0.55
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=base.confidence,
            evidence_summary=base.evidence_summary,
            rationale=(
                f"Dispatch: ALERT={base.decision}, 置信度={base.confidence:.3f} "
                f"× 阈值 0.55（忽略附加条件 AND 与暴露复合）"
            ),
        )

    def _decide_upgrade(self, context: ScenarioContext) -> DecisionOutput:
        """UPGRADE baseline：ALERT 判定且置信度 >= 0.60（忽略等级比较/趋势）。"""
        base = self._decide_alert(context)
        decision = base.decision and base.confidence >= 0.60
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=base.confidence,
            evidence_summary=base.evidence_summary,
            rationale=(
                f"Upgrade: ALERT={base.decision}, 置信度={base.confidence:.3f} "
                f"× 阈值 0.60（忽略等级比较与趋势外推）"
            ),
        )

    def _decide_close(self, context: ScenarioContext) -> DecisionOutput:
        """CLOSE baseline：ALERT 判定且置信度 >= 0.70（忽略关键设施细节）。"""
        base = self._decide_alert(context)
        decision = base.decision and base.confidence >= 0.70
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=base.confidence,
            evidence_summary=base.evidence_summary,
            rationale=(
                f"Close: ALERT={base.decision}, 置信度={base.confidence:.3f} "
                f"× 阈值 0.70（忽略危险等级 × 关键设施的矩阵）"
            ),
        )

    def _decide_recover(self, context: ScenarioContext) -> DecisionOutput:
        """RECOVER baseline：「预警解除即恢复」，无持续时间窗校验。

        这是应急管理最常见人命代价来源——「雨停就解除」。真值要求条件解除
        后持续 N 小时无回弹；baseline 只看当前是否已无预警。
        """
        base = self._decide_alert(context)
        decision = not base.decision
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=1.0 - base.confidence,
            evidence_summary=base.evidence_summary,
            rationale=(
                f"Recover: ALERT={base.decision} → 恢复={decision} "
                f"（无持续窗校验——雨停即解除）"
            ),
        )

    # ------------------------------------------------------------------
    # 动作等级决策（Phase B 暴露层）：monitor / alert / dispatch
    # ------------------------------------------------------------------

    def decide_action(self, context: ScenarioContext) -> str:
        """动作等级 baseline：线性置信度作严重度代理 × 暴露分级。

        刻意与真值矩阵不同口径（披露）：
        - 真值用物理 GT score >= 0.90 判高严重度；
        - baseline 用自身线性置信度 >= 0.85 作代理（阈值不同 → 边界
          用例双向出错，构成暴露套件的动作级判别力来源）。
        """
        out = self.decide(context)
        exp_class, _ = infer_exposure_class(context.exposure)

        if not out.decision:
            return "monitor"
        if out.confidence >= 0.85:
            return "dispatch" if exp_class in ("E2", "E3") else "alert"
        return "dispatch" if exp_class == "E3" else "alert"


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

        cat_hint = domain_hint(context.category.value)

        prompt = (
            f"[EarthBench Alert 决策任务]\n"
            f"区域: {context.region}\n"
            f"时间窗口: {context.horizon_hours}小时\n"
            f"决策模板: {tmpl_label}\n"
            f"\n证据列表:\n{obs_text}\n"
            f"{cat_hint}"
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
        import json
        import os
        import urllib.request

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
        """解析 LLM 响应为 (answer, confidence, rationale)（委托 llm.parse_yes_no）。"""
        content = text.strip()
        ans = parse_yes_no(content)
        if ans is None:
            return None, 0.5, content[:300]
        return ans, 0.85, content[:300]

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
