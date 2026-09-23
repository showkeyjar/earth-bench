"""四大高频风险 Alert 场景 + 场景管理 + Ground Truth 推导规则。

每个场景都有独立的 Ground Truth 推导规则（基于国标/行业标准阈值），
确保每输入一组观测数据就能自动产生可验证的 YES/NO。

场景列表：
1. Wildfire (森林火险) — FWI + 湿度 + 风速 + 降雨 → 是否预警
2. Flood (洪涝) — 降水量 + 水位 + 土壤湿度 → 是否预警
3. Drought (干旱) — SPI/Palmer + 湿度 + 降雨 + NDVI → 是否预警
4. HeatWave (热浪) — 温度 + 湿度(Wet Bulb) + 持续时间 → 是否预警
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ScenarioContext, Observation, DecisionTemplate, ScenarioCategory


# ===========================================================================
# ScenarioStore — 场景数据存储与管理（保留原有功能）
# ===========================================================================


class ScenarioStore:
    """场景数据存储与管理。"""

    def __init__(self, data_dir: Path | None = None):
        # 包内数据目录（earthbench/data，含 CARS 工件）；
        # 此前误指 repo 根 /data（不存在）
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self._cache: dict[str, ScenarioContext] = {}

    def register_scenario(self, scenario_id: str, context: ScenarioContext) -> None:
        """注册一个场景到缓存。"""
        self._cache[scenario_id] = context

    def load_fire_scenario(
        self,
        scenario_id: str,
        region: str,
        horizon_hours: int = 72,
        observations: list[dict] | None = None,
    ) -> ScenarioContext:
        """创建一个森林火险场景。"""
        if observations is None:
            observations = []
        obs_list = [Observation(**obs) for obs in observations]
        context = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            observations=obs_list,
            region=region,
            horizon_hours=horizon_hours,
        )
        self.register_scenario(scenario_id, context)
        return context

    def load_scenario_from_dict(
        self,
        scenario_id: str,
        region: str,
        horizon_hours: int = 72,
        observations: list[Observation] | None = None,
        decision_template: DecisionTemplate = DecisionTemplate.ALERT,
        category: ScenarioCategory = ScenarioCategory.FIRE,
    ) -> ScenarioContext:
        """从预构建的 Observation 列表创建场景上下文。"""
        if observations is None:
            observations = []
        context = ScenarioContext(
            category=category,
            template=decision_template,
            observations=observations,
            region=region,
            horizon_hours=horizon_hours,
        )
        self.register_scenario(scenario_id, context)
        return context

    def load_scenario_from_json(self, filepath: Path) -> ScenarioContext:
        """从 JSON 文件加载场景。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        observations = [Observation(**obs) for obs in data.get("observations", [])]
        context = ScenarioContext(
            category=ScenarioCategory(data.get("category", "fire")),
            template=DecisionTemplate(data.get("template", "alert")),
            observations=observations,
            region=data.get("region", "unknown"),
            horizon_hours=data.get("horizon_hours", 72),
        )
        return context

    def get_all_ids(self) -> list[str]:
        return list(self._cache.keys())

    def get(self, scenario_id: str) -> ScenarioContext | None:
        return self._cache.get(scenario_id)


# ===========================================================================
# Ground Truth 推导规则 — 独立物理标准（查表式分级判定）
#
# 说明：Ground Truth 一律依据公开国家标准/行业标准的"分级表"直接判定，
# 不是加权评分公式。这样 benchmark 的真值来源与 agents.py 的
# 评分模型（归一化加权求和）完全脱钩，避免"自证循环"——
# 即用与决策 Agent 相同的公式反推真值，再宣称 Agent 得分 100%。
#
# 各灾种引用的外部标准：
# - Fire   : GB/T 36743-2018《森林火险气象等级》 / 加拿大 FWI 分级体系
# - Flood  : GB/T 28592-2012《降水量等级》(24h 暴雨/大暴雨) + 防汛警戒水位
# - Drought: GB/T 20481-2017《气象干旱等级》(SPI 分级 / Palmer 指数)
# - Heat   : 《中央气象台高温预警信号》(黄/橙/红) + WBGT 湿球热应激阈值
# ===========================================================================

# === Wildfire — 加拿大 FWI 火险天气指数分级表 ===
# FWI < 5 很低 | 5-11 低 | 12-21 中 | 22-33 高 | >33 极高
FWI_LEVEL_VERY_LOW = 5.0
FWI_LEVEL_LOW = 11.0
FWI_LEVEL_MODERATE = 21.0
FWI_LEVEL_HIGH = 22.0
FWI_LEVEL_EXTREME = 33.0
# 极高火险（FWI>33）本身即触发预警；高火险（22~33）需叠加干燥/大风
FIRE_EXTREME_SCORE = 0.95
FIRE_HIGH_TRIGGER_SCORE = 0.75
# 24h 降雨 >= 30mm 视为彻底灭火，强制不预警；>= 20mm 且 FWI < 40 降档
RAINFALL_EXTINGUISH = 30.0
RAINFALL_SUPPRESS_LEVEL = 20.0
FWI_SUPPRESS_CEILING = 40.0

# === Flood — GB/T 28592-2012 降水量等级（24h 暴雨 >= 50mm / 大暴雨 >= 100mm）===
RAINFALL_FLOOD_24H_EXTREME = 100.0  # 大暴雨
RAINFALL_FLOOD_24H_HEAVY = 50.0  # 暴雨
RAINFALL_FLOOD_24H_MODERATE = 30.0  # 中雨偏高（叠加土壤饱和即险）
RAINFALL_FLOOD_6H = 50.0  # 6h 短时强降雨（>= 50mm）
SOIL_MOISTURE_SATURATED = 0.85
WATER_LEVEL_WARNING = 10.0  # 警戒水位（米）
WATER_LEVEL_TREND_RISING = 9.5  # 持续上升逼近警戒即预警

# === Drought — GB/T 20481-2017 SPI 分级（<= -1.0 中旱 / <= -1.5 重旱 / <= -2.0 特旱）===
SPI_DROUGHT_MODERATE = -1.0
SPI_DROUGHT_SEVERE = -1.5
SPI_DROUGHT_EXTREME = -2.0
PALMER_DROUGHT_LIGHT = -1.0  # Palmer <= -1.0 轻旱起步（续以副因子印证）
PALMER_DROUGHT_MODERATE = -2.0  # Palmer <= -2.0 中旱直接预警
HUMIDITY_DROUGHT = 25.0
RAINFALL_DEFICIT = 10.0
NDVI_DEGRADE = 0.3

# === HeatWave — 《中央气象台高温预警信号》+ WBGT 湿球热应激 ===
TEMP_HEAT_RED = 40.0  # 红色预警：单日 >= 40℃
TEMP_HEAT_ORANGE = 37.0  # 橙色预警：单日 >= 37℃
TEMP_HEAT_YELLOW = 35.0  # 黄色预警：连续 3 天 >= 35℃
WET_BULB_TEMP = 27.0  # WBGT 湿球 >= 27℃ 即热应激危险
HEAT_DURATION_DAYS = 3
# 黄色预警需湿热印证（干热不算）：连续 3 天 >=35℃ 且 湿球 >= 24℃
WET_BULB_YELLOW_CONFIRM = 24.0
TEMP_MUGGY = 33.0  # 闷热起点：>=33℃ 且 湿度 >= 70%
HUMIDITY_MUGGY = 70.0


# ===========================================================================
# 独立物理标准推导 — 通用取值辅助
# ===========================================================================


def _obs_vals(obs: list[dict], variable: str) -> list[tuple[str, float]]:
    """返回 (timestamp, value) 列表，按时间戳升序。"""
    items = [
        (str(o.get("timestamp", "")), float(o["value"]))
        for o in obs
        if o["variable"] == variable
    ]
    items.sort(key=lambda x: x[0])
    return items


def _latest(obs: list[dict], variable: str) -> float | None:
    vals = _obs_vals(obs, variable)
    return vals[-1][1] if vals else None


def _max_val(obs: list[dict], variable: str) -> float | None:
    vals = _obs_vals(obs, variable)
    return max(v[1] for v in vals) if vals else None


def _mean_val(obs: list[dict], variable: str) -> float | None:
    vals = _obs_vals(obs, variable)
    return sum(v[1] for v in vals) / len(vals) if vals else None


def _fwi_level(fwi: float) -> str:
    """加拿大 FWI 火险天气指数五级（GB/T 36743-2018 同构分级）。"""
    if fwi > FWI_LEVEL_EXTREME:
        return "极高"
    if fwi > FWI_LEVEL_HIGH:
        return "高"
    if fwi > FWI_LEVEL_MODERATE:
        return "中"
    if fwi > FWI_LEVEL_LOW:
        return "低"
    return "很低"


def infer_fire_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """森林火险 Ground Truth 推导（独立物理标准查表判定）。

    依据 GB/T 36743-2018《森林火险气象等级》的 FWI 五级分级表直接判定，
    不采用加权评分模型，与决策 Agent 的评分逻辑完全脱钩：

    1. 最新 FWI > 33（极高火险）→ 预警（score=0.95）
    2. 最新 FWI 22~33（高火险）且 相对湿度 < 30% 或 风速 >= 12m/s → 预警（0.75）
    3. 抑制条件：最新观测 24h 降雨 >= 30mm → 强制不预警（彻底灭火）；
       降雨 >= 20mm 且 FWI < 40 → 降档不预警
    """
    explanation: dict[str, Any] = {}

    fwi = _latest(obs, "FWI")
    if fwi is None:
        return False, 0.0, {"reason": "no FWI observations; 独立标准无依据"}

    fwi_lvl = _fwi_level(fwi)
    explanation["fwi_level"] = fwi_lvl
    explanation["fwi_latest"] = round(fwi, 1)

    hum = _latest(obs, "humidity")
    wind = _latest(obs, "wind_speed")
    rain24 = _max_val(obs, "rainfall")
    if hum is not None:
        explanation["humidity_latest"] = round(hum, 1)
    if wind is not None:
        explanation["wind_latest"] = round(wind, 1)
    if rain24 is not None:
        explanation["rainfall_max_24h"] = round(rain24, 1)

    # 1) 彻底灭火：24h 降雨 >= 30mm 强制降级为不预警
    if rain24 is not None and rain24 >= RAINFALL_EXTINGUISH:
        explanation["standard"] = "GB/T 36743-2018 降雨抑制"
        explanation["detail"] = f"24h 降雨 {rain24:.0f}mm >= 30mm，彻底灭火，不预警"
        return False, 0.15, explanation

    # 2) 极大暴雨压制（>=20mm 且 FWI 未到高危）也不预警
    if (
        rain24 is not None
        and rain24 >= RAINFALL_SUPPRESS_LEVEL
        and fwi < FWI_SUPPRESS_CEILING
    ):
        explanation["standard"] = "GB/T 36743-2018 降雨抑制"
        explanation["detail"] = (
            f"24h 降雨 {rain24:.0f}mm >= 20mm 且 FWI {fwi:.0f} < 40，风险被压制"
        )
        return False, 0.30, explanation

    # 3) 极高火险（FWI > 33）：无条件预警
    if fwi > FWI_LEVEL_EXTREME:
        explanation["standard"] = "GB/T 36743-2018 FWI 分级表"
        explanation["detail"] = f"FWI {fwi:.1f} > 33，极高火险，触发预警"
        return True, FIRE_EXTREME_SCORE, explanation

    # 4) 高火险（22~33）需叠加干燥或大风
    if fwi > FWI_LEVEL_HIGH:
        if (hum is not None and hum < 30.0) or (wind is not None and wind >= 12.0):
            explanation["standard"] = "GB/T 36743-2018 FWI 分级表"
            explanation["detail"] = (
                f"FWI {fwi:.1f} 属高火险，且 "
                f"{'湿度<30%' if (hum is not None and hum < 30.0) else '风速>=12m/s'}，触发预警"
            )
            return True, FIRE_HIGH_TRIGGER_SCORE, explanation
        explanation["standard"] = "GB/T 36743-2018 FWI 分级表"
        explanation["detail"] = (
            f"FWI {fwi:.1f} 属高火险，但未叠加干燥（湿度<30%）或大风（>=12m/s），不预警"
        )
        return False, 0.50, explanation

    # 5) 中及以下火险：不预警
    explanation["standard"] = "GB/T 36743-2018 FWI 分级表"
    explanation["detail"] = f"FWI {fwi:.1f} 属{fwi_lvl}火险等级，不触发预警"
    return False, 0.40 if fwi > FWI_LEVEL_MODERATE else 0.25, explanation


def infer_flood_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """洪涝 Ground Truth 推导（独立物理标准查表判定）。

    依据 GB/T 28592-2012《降水量等级》逐条查表，任何一个阈值命中即预警：

    1. 24h 降雨 >= 100mm（大暴雨）→ 预警（0.95）
    2. 24h 降雨 >= 50mm（暴雨）→ 预警（0.85）
    3. 6h 降雨 >= 50mm（短时强降雨）→ 预警（0.85）
    4. 24h 降雨 >= 30mm 且 土壤湿度 >= 0.85（饱和）→ 预警（0.80）
    5. 最新水位 >= 10m（警戒水位）→ 预警（0.90）
    6. 最新水位 >= 9.5m 且持续上升 → 预警（0.70）
    """
    explanation: dict[str, Any] = {}

    rain24 = _max_val(obs, "rainfall_24h")
    rain6 = _max_val(obs, "rainfall_6h")
    soil = _max_val(obs, "soil_moisture")
    levels = _obs_vals(obs, "water_level")

    if rain24 is not None:
        explanation["rainfall_24h_max"] = round(rain24, 1)
    if rain6 is not None:
        explanation["rainfall_6h_max"] = round(rain6, 1)
    if soil is not None:
        explanation["soil_moisture_max"] = round(soil, 3)
    if levels:
        explanation["water_level_latest"] = levels[-1][1]
        explanation["water_level_series"] = [v for _, v in levels]

    # 1) 大暴雨
    if rain24 is not None and rain24 >= RAINFALL_FLOOD_24H_EXTREME:
        explanation["standard"] = "GB/T 28592-2012 降水量等级"
        explanation["detail"] = f"24h 降雨 {rain24:.0f}mm >= 100mm（大暴雨），触发预警"
        return True, 0.95, explanation

    # 2) 暴雨
    if rain24 is not None and rain24 >= RAINFALL_FLOOD_24H_HEAVY:
        explanation["standard"] = "GB/T 28592-2012 降水量等级"
        explanation["detail"] = f"24h 降雨 {rain24:.0f}mm >= 50mm（暴雨），触发预警"
        return True, 0.85, explanation

    # 3) 6h 短时强降雨
    if rain6 is not None and rain6 >= RAINFALL_FLOOD_6H:
        explanation["standard"] = "GB/T 28592-2012 短时强降雨"
        explanation["detail"] = f"6h 降雨 {rain6:.0f}mm >= 50mm，触发预警"
        return True, 0.85, explanation

    # 4) 中雨 + 土壤饱和
    if (
        rain24 is not None
        and rain24 >= RAINFALL_FLOOD_24H_MODERATE
        and soil is not None
        and soil >= SOIL_MOISTURE_SATURATED
    ):
        explanation["standard"] = "GB/T 28592-2012 + 土壤饱和判据"
        explanation["detail"] = (
            f"24h 降雨 {rain24:.0f}mm >= 30mm 且 土壤湿度 {soil:.2f} >= 0.85，触发预警"
        )
        return True, 0.80, explanation

    # 5) 超警戒水位
    if levels and levels[-1][1] >= WATER_LEVEL_WARNING:
        explanation["standard"] = "防汛警戒水位判据"
        explanation["detail"] = (
            f"最新水位 {levels[-1][1]:.1f}m >= 10m（警戒），触发预警"
        )
        return True, 0.90, explanation

    # 6) 持续上升逼近警戒
    if len(levels) >= 3:
        values = [v for _, v in levels]
        increasing = all(values[i] < values[i + 1] for i in range(len(values) - 1))
        if increasing and values[-1] >= WATER_LEVEL_TREND_RISING:
            explanation["standard"] = "防汛警戒水位判据（趋势）"
            explanation["detail"] = (
                f"水位持续上升至 {values[-1]:.1f}m >= 9.5m，逼近警戒，触发预警"
            )
            return True, 0.70, explanation

    explanation["standard"] = "GB/T 28592-2012 + 防汛警戒判据"
    explanation["detail"] = "未命中任何暴雨/警戒阈值，不预警"
    return False, 0.20, explanation


def infer_drought_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """干旱 Ground Truth 推导（独立物理标准查表判定）。

    依据 GB/T 20481-2017《气象干旱等级》SPI 分级逐条查表：

    1. 最新 SPI <= -2.0（特旱）→ 预警（0.95）
    2. 最新 SPI <= -1.5（重旱）→ 预警（0.85）
    3. 最新 SPI <= -1.0（中旱）→ 预警（0.75）
    4. Palmer <= -2.0（中旱）→ 预警（0.75）
    5. Palmer <= -1.0（轻旱起步）且任一印证因子
       （湿度<25% / 月降雨<10mm / NDVI<0.3）→ 预警（0.65）
    """
    explanation: dict[str, Any] = {}

    spi = _latest(obs, "SPI")
    palmer = _latest(obs, "palmer_index")
    hum = _latest(obs, "humidity")
    rain_month = _latest(obs, "rainfall_monthly")
    ndvi = _latest(obs, "NDVI")

    if spi is not None:
        explanation["spi_latest"] = spi
    if palmer is not None:
        explanation["palmer_latest"] = palmer
    if hum is not None:
        explanation["humidity_latest"] = round(hum, 1)
    if rain_month is not None:
        explanation["rainfall_monthly_latest"] = round(rain_month, 1)
    if ndvi is not None:
        explanation["ndvi_latest"] = round(ndvi, 3)

    # 1-3) SPI 分级
    if spi is not None:
        if spi <= SPI_DROUGHT_EXTREME:
            explanation["standard"] = "GB/T 20481-2017 SPI 分级"
            explanation["detail"] = f"SPI {spi:.2f} <= -2.0（特旱），触发预警"
            return True, 0.95, explanation
        if spi <= SPI_DROUGHT_SEVERE:
            explanation["standard"] = "GB/T 20481-2017 SPI 分级"
            explanation["detail"] = f"SPI {spi:.2f} <= -1.5（重旱），触发预警"
            return True, 0.85, explanation
        if spi <= SPI_DROUGHT_MODERATE:
            explanation["standard"] = "GB/T 20481-2017 SPI 分级"
            explanation["detail"] = f"SPI {spi:.2f} <= -1.0（中旱），触发预警"
            return True, 0.75, explanation

    # 4) Palmer 中旱
    if palmer is not None and palmer <= PALMER_DROUGHT_MODERATE:
        explanation["standard"] = "GB/T 20481-2017 Palmer 分级"
        explanation["detail"] = f"Palmer {palmer:.2f} <= -2.0（中旱），触发预警"
        return True, 0.75, explanation

    # 5) Palmer 轻旱 + 印证因子
    if palmer is not None and palmer <= PALMER_DROUGHT_LIGHT:
        confirms: list[str] = []
        if hum is not None and hum < HUMIDITY_DROUGHT:
            confirms.append(f"湿度{hum:.0f}%<25%")
        if rain_month is not None and rain_month < RAINFALL_DEFICIT:
            confirms.append(f"月降雨{rain_month:.0f}mm<10mm")
        if ndvi is not None and ndvi < NDVI_DEGRADE:
            confirms.append(f"NDVI {ndvi:.2f}<0.3")
        if confirms:
            explanation["standard"] = "GB/T 20481-2017 多因子印证"
            explanation["detail"] = (
                f"Palmer {palmer:.2f} 轻旱，且印证因子：{'；'.join(confirms)}，触发预警"
            )
            return True, 0.65, explanation

    explanation["standard"] = "GB/T 20481-2017 SPI/Palmer 分级"
    explanation["detail"] = "未达到中旱及以上等级，不预警"
    return False, 0.25, explanation


def infer_heatwave_ground_truth(
    obs: list[dict], heat_duration_days: int = HEAT_DURATION_DAYS
) -> tuple[bool, float, dict[str, Any]]:
    """热浪 Ground Truth 推导（独立物理标准查表判定）。

    依据《中央气象台高温预警信号》色阶 + WBGT 湿球热应激阈值：

    1. 单日最高温 >= 40℃（红色预警）→ 预警（0.95）
    2. 单日最高温 >= 37℃（橙色预警）→ 预警（0.85）
    3. 单日最高温 >= 35℃ 且 持续 >= 3 天 且 湿球 >= 24℃（黄色+湿热印证）→ 预警（0.70）
    4. 湿球温度 >= 27℃（WBGT 热应激危险）→ 预警（0.80）
    5. 最高温 >= 33℃ 且 湿度 >= 70%（闷热）→ 预警（0.60）
    """
    # 优先使用观测中的持续高温天数；缺失时用调用方传入的默认值
    duration_vals = [o["value"] for o in obs if o["variable"] == "heat_duration_days"]
    if duration_vals:
        actual_duration = int(duration_vals[0])
    else:
        actual_duration = heat_duration_days

    explanation: dict[str, Any] = {}

    temp_max = _max_val(obs, "temperature_max")
    if temp_max is None:
        temp_max = _max_val(obs, "temperature")
    wbt = _max_val(obs, "wet_bulb_temp")
    hum = _latest(obs, "humidity")

    if temp_max is not None:
        explanation["temperature_max"] = round(temp_max, 1)
    if wbt is not None:
        explanation["wet_bulb_temp_max"] = round(wbt, 1)
    if hum is not None:
        explanation["humidity_latest"] = round(hum, 1)
    explanation["heat_duration_days"] = actual_duration

    # 1) 红色：>= 40℃
    if temp_max is not None and temp_max >= TEMP_HEAT_RED:
        explanation["standard"] = "中央气象台高温红色预警"
        explanation["detail"] = f"最高温 {temp_max:.1f}℃ >= 40℃，触发预警"
        return True, 0.95, explanation

    # 2) 橙色：>= 37℃
    if temp_max is not None and temp_max >= TEMP_HEAT_ORANGE:
        explanation["standard"] = "中央气象台高温橙色预警"
        explanation["detail"] = f"最高温 {temp_max:.1f}℃ >= 37℃，触发预警"
        return True, 0.85, explanation

    # 3) 黄色：>= 35℃ 连续 3 天 + 湿热印证（干热不算）
    if (
        temp_max is not None
        and temp_max >= TEMP_HEAT_YELLOW
        and actual_duration >= HEAT_DURATION_DAYS
        and wbt is not None
        and wbt >= WET_BULB_YELLOW_CONFIRM
    ):
        explanation["standard"] = "中央气象台高温黄色预警"
        explanation["detail"] = (
            f"最高温 {temp_max:.1f}℃ >= 35℃ 且持续 {actual_duration} 天 "
            f"且湿球 {wbt:.1f}℃ >= 24℃，触发预警"
        )
        return True, 0.70, explanation

    # 4) 湿球热应激
    if wbt is not None and wbt >= WET_BULB_TEMP:
        explanation["standard"] = "WBGT 湿球热应激阈值"
        explanation["detail"] = f"湿球温度 {wbt:.1f}℃ >= 27℃，触发预警"
        return True, 0.80, explanation

    # 5) 闷热
    if (
        temp_max is not None
        and temp_max >= TEMP_MUGGY
        and hum is not None
        and hum >= HUMIDITY_MUGGY
    ):
        explanation["standard"] = "闷热判据"
        explanation["detail"] = (
            f"最高温 {temp_max:.1f}℃ >= 33℃ 且 湿度 {hum:.0f}% >= 70%，触发预警"
        )
        return True, 0.60, explanation

    explanation["standard"] = "中央气象台高温预警信号 + WBGT"
    explanation["detail"] = "未达到任何预警等级，不预警"
    return False, 0.25, explanation


# ===========================================================================
# AlertBenchmarkSuite — 四大类别测试集生成器
# ===========================================================================


def _make_obs(
    source: str,
    variable: str,
    value: float,
    unit: str,
    timestamp: str,
    confidence: float = 0.95,
) -> dict:
    """辅助函数：创建单条 Observation dict。"""
    return {
        "source": source,
        "variable": variable,
        "value": value,
        "unit": unit,
        "timestamp": timestamp,
        "confidence": confidence,
    }


def get_alert_benchmark_suite() -> list[dict[str, Any]]:
    """
    生成 Alert 基准测试套件（四大高频风险场景，共 20 个用例）。

    每个用例包含：
    - case_id, difficulty, category, region, ground_truth
    - observations (list[dict])
    - _gt_fn: Ground Truth 推导函数（会在构建时调用）
    """
    suite: list[dict[str, Any]] = []

    # ---- Wildfire (5 L1-L4) ----
    suite.append(
        {
            "case_id": "fire-l1-extreme-risk",
            "difficulty": "L1",
            "category": "fire",
            "region": "Xiangshan-Beijing",
            "ground_truth": True,
            "observations": [
                _make_obs("ECMWF", "FWI", 52.0, "", "2026-07-10T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 55.0, "", "2026-07-11T12:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 10.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 18.0, "m/s", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "MODIS", "temperature", 38.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_fire_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "fire-l1-low-risk",
            "difficulty": "L1",
            "category": "fire",
            "region": "WestLake-Hangzhou",
            "ground_truth": False,
            "observations": [
                _make_obs("ECMWF", "FWI", 15.0, "", "2026-07-10T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 18.0, "", "2026-07-11T12:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 65.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall", 8.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_fire_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "fire-l2-fwi-high-rain-suppress",
            "difficulty": "L2",
            "category": "fire",
            "region": "ChangbaiMountain",
            "ground_truth": True,
            "observations": [
                _make_obs("ECMWF", "FWI", 48.0, "", "2026-07-10T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 50.0, "", "2026-07-11T06:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 15.0, "%", "2026-07-11T06:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall", 25.0, "mm", "2026-07-11T00:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_fire_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "fire-l3-coastal-humid",
            "difficulty": "L3",
            "category": "fire",
            "region": "Shenzhen-Coast",
            "ground_truth": True,
            "observations": [
                _make_obs("ECMWF", "FWI", 32.0, "", "2026-07-10T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 35.0, "", "2026-07-11T12:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 55.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "MODIS", "temperature", 34.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_fire_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "fire-l4-trend-rising",
            "difficulty": "L4",
            "category": "fire",
            "region": "GreaterKhingan",
            "ground_truth": True,
            "observations": [
                _make_obs("ECMWF", "FWI", 30.0, "", "2026-07-08T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 35.0, "", "2026-07-09T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 39.0, "", "2026-07-10T12:00:00+08:00"),
                _make_obs("ECMWF", "FWI", 43.0, "", "2026-07-11T12:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 25.0, "%", "2026-07-08T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 20.0, "%", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_fire_ground_truth,
        }
    )

    # ---- Flood (5 L1-L4) ----
    suite.append(
        {
            "case_id": "flood-l1-extreme-rainfall",
            "difficulty": "L1",
            "category": "flood",
            "region": "Wuhan-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "rainfall_24h", 120.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_6h", 65.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.90, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 12.5, "m", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_flood_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "flood-l1-no-rain",
            "difficulty": "L1",
            "category": "flood",
            "region": "Lhasa-Tibet",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "rainfall_24h", 0.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.20, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 2.0, "m", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_flood_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "flood-l2-soil-saturated",
            "difficulty": "L2",
            "category": "flood",
            "region": "Guangzhou-PearlR",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "rainfall_24h", 40.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.88, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 8.5, "m", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_flood_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "flood-l3-good-drainage",
            "difficulty": "L3",
            "category": "flood",
            "region": "Guilin-Guangxi",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "rainfall_24h", 80.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.55, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 3.0, "m", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_flood_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "flood-l4-water-trend-rise",
            "difficulty": "L4",
            "category": "flood",
            "region": "Nanjing-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Hydro", "water_level", 9.5, "m", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 10.2, "m", "2026-07-11T06:00:00+08:00"
                ),
                _make_obs(
                    "Hydro", "water_level", 10.8, "m", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_flood_ground_truth,
        }
    )

    # ---- Drought (5 L1-L4) ----
    suite.append(
        {
            "case_id": "drought-l1-severe-spi",
            "difficulty": "L1",
            "category": "drought",
            "region": "Kunming-Yunnan",
            "ground_truth": True,
            "observations": [
                _make_obs("CMA", "SPI", -2.3, "", "2026-07-01T00:00:00+08:00"),
                _make_obs("CMA", "palmer_index", -0.8, "", "2026-07-01T00:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 15.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_monthly", 5.0, "mm", "2026-07-01T00:00:00+08:00"
                ),
                _make_obs("MODIS", "NDVI", 0.15, "", "2026-07-05T00:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "drought-l1-normal",
            "difficulty": "L1",
            "category": "drought",
            "region": "Hangzhou-Zhejiang",
            "ground_truth": False,
            "observations": [
                _make_obs("CMA", "SPI", -0.2, "", "2026-07-01T00:00:00+08:00"),
                _make_obs("CMA", "palmer_index", -0.1, "", "2026-07-01T00:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 60.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_monthly", 150.0, "mm", "2026-07-01T00:00:00+08:00"
                ),
                _make_obs("MODIS", "NDVI", 0.65, "", "2026-07-05T00:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "drought-l2-borderline-spi",
            "difficulty": "L2",
            "category": "drought",
            "region": "Harbin-Heilongjiang",
            "ground_truth": False,
            "observations": [
                _make_obs("CMA", "SPI", -0.8, "", "2026-07-01T00:00:00+08:00"),
                _make_obs("CMA", "palmer_index", -0.3, "", "2026-07-01T00:00:00+08:00"),
                _make_obs(
                    "CMA", "rainfall_monthly", 80.0, "mm", "2026-07-01T00:00:00+08:00"
                ),
                _make_obs("MODIS", "NDVI", 0.50, "", "2026-07-05T00:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "drought-l3-palmerspi-conflict",
            "difficulty": "L3",
            "category": "drought",
            "region": "Urumqi-Xinjiang",
            "ground_truth": True,
            "observations": [
                _make_obs("CMA", "SPI", -0.3, "", "2026-07-01T00:00:00+08:00"),
                _make_obs("CMA", "palmer_index", -1.2, "", "2026-07-01T00:00:00+08:00"),
                _make_obs(
                    "Station", "humidity", 20.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_monthly", 5.0, "mm", "2026-07-01T00:00:00+08:00"
                ),
                _make_obs("MODIS", "NDVI", 0.10, "", "2026-07-05T00:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "drought-l4-spir-declining",
            "difficulty": "L4",
            "category": "drought",
            "region": "Taiyuan-Shanxi",
            "ground_truth": True,
            "observations": [
                _make_obs("CMA", "SPI", -0.5, "", "2026-05-01T00:00:00+08:00"),
                _make_obs("CMA", "SPI", -0.9, "", "2026-06-01T00:00:00+08:00"),
                _make_obs("CMA", "SPI", -1.6, "", "2026-07-01T00:00:00+08:00"),
                _make_obs("CMA", "palmer_index", -0.6, "", "2026-07-01T00:00:00+08:00"),
                _make_obs(
                    "CMA", "rainfall_monthly", 10.0, "mm", "2026-07-01T00:00:00+08:00"
                ),
                _make_obs("MODIS", "NDVI", 0.20, "", "2026-07-05T00:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        }
    )

    # ---- HeatWave (5 L1-L4) ----
    suite.append(
        {
            "case_id": "heat-l1-extreme-wetbulb",
            "difficulty": "L1",
            "category": "heat",
            "region": "Chongqing-HotPotato",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_max", 40.0, "°C", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 38.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 39.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 75.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 30.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 28.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
            ],
            "heat_duration_days": 5,
            "_gt_fn": infer_heatwave_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "heat-l1-cool-comfortable",
            "difficulty": "L1",
            "category": "heat",
            "region": "Kunming-SpringCity",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "temperature_max", 26.0, "°C", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 27.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 25.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 40.0, "%", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "heat_duration_days": 0,
            "_gt_fn": infer_heatwave_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "heat-l2-hot-but-dry",
            "difficulty": "L2",
            "category": "heat",
            "region": "Urumqi-Xinjiang",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "temperature_max", 36.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 15.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 22.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "heat_duration_days": 3,
            "_gt_fn": infer_heatwave_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "heat-l3-borderline",
            "difficulty": "L3",
            "category": "heat",
            "region": "Nanjing-OvenCity",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_max", 34.5, "°C", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 35.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 34.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 72.0, "%", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 27.5, "°C", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "heat_duration_days": 3,
            "_gt_fn": infer_heatwave_ground_truth,
        }
    )
    suite.append(
        {
            "case_id": "heat-l4-progressive-heat",
            "difficulty": "L4",
            "category": "heat",
            "region": "Wuhan-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_max", 31.0, "°C", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 35.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_max", 39.0, "°C", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 24.0, "°C", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 27.0, "°C", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wet_bulb_temp", 29.5, "°C", "2026-07-11T12:00:00+08:00"
                ),
            ],
            "heat_duration_days": 3,
            "_gt_fn": infer_heatwave_ground_truth,
        }
    )

    return suite


def get_adversarial_suite() -> list[dict[str, Any]]:
    """判别力扩展套件 — 规则引擎线性加权与国标查表真值系统性分歧的硬用例。

    与 get_alert_benchmark_suite() 的基础用例不同，这些用例刻意构造在
    「线性评分模型高兴度、但标准查表不触�����的决策边界上（多因子次阈值
    叠加 / AND 条件缺一 / 否定印证因子差一线），用于检验 Agent 能否超越
    规则 baseline —— baseline 在基础套件上为 100%，在对抗套件上必然漏判。

    设计原则：真值仍由 _gt_fn 独立查表推导（与基础套件同源），
    对抗性只体现在观测取值的构造上，不改动任何标准阈值。
    """
    return [
        {
            # FWI=30 属高火险档（22~33），但湿度 40%（未干燥）且风速 8m/s（未大风），
            # 国标要求高火险必须叠加干燥或大风才预警 → 不预警。
            # 线性模型：FWI 0.5×0.40 + 湿度 0.6×0.20 + 风速 0.4×0.15，归一后 ≈0.507 ≥ 0.4 → 误报。
            "case_id": "adv-fire-linear-overweight",
            "difficulty": "L4",
            "category": "fire",
            "region": "Xiangshan-Beijing",
            "ground_truth": False,
            "observations": [
                _make_obs("ECMWF", "FWI", 30.0, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 40.0, "%", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "wind_speed", 8.0, "m/s", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_fire_ground_truth,
        },
        {
            # 24h=45mm（<50 暴雨线）、6h=28mm（<50）、土壤 0.8（<0.85 饱和线）、水位 8.5m（<10 警戒）
            # —— 四个因子全部差一线，任一 OR 条件都不触发 → 不预警。
            # 线性模型四因子同时供分，归一后 ≈0.585 ≥ 0.45 → 误报（OR 逻辑被均值抹平）。
            "case_id": "adv-flood-subthreshold-mix",
            "difficulty": "L4",
            "category": "flood",
            "region": "Wuhan-Yangtze",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "rainfall_24h", 45.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_6h", 28.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "soil_moisture", 0.80, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "water_level", 8.5, "m", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_flood_ground_truth,
        },
        {
            # SPI=-0.6（> -1.0 无旱）、Palmer=-1.5（轻旱，需印证因子），
            # 湿度 26%（≥25）、月降雨 11mm（≥10）、NDVI 0.31（≥0.3）三个印证因子
            # 全部差一线 → 不预警。
            # 线性模型：Palmer 高分 + 月降雨/NDVI 接近满档，归一后 ≈0.433 ≥ 0.4 → 误报。
            "case_id": "adv-drought-palmer-light-no-confirm",
            "difficulty": "L4",
            "category": "drought",
            "region": "ChangbaiMountain",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "SPI", -0.60, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "palmer_index", -1.50, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 26.0, "%", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_monthly", 11.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "NDVI", 0.31, "", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        },
        {
            # 最高温恰 35℃但湿球 23.9℃（<24 黄色湿热印证线）、湿度 69%（<70 闷热线）
            # —— 黄色预警 AND 条件缺一、闷热 OR 条件缺一 → 不预警。
            # 线性模型：温度 0.583×0.35 + 时长 0.6×0.25 + 湿球 0.113×0.25 + 湿度 0.38×0.15
            # ≈ 0.439 ≥ 0.4 → 误报（AND 条件被加权平均稀释）。
            "case_id": "adv-heat-subthreshold-dry",
            "difficulty": "L4",
            "category": "heat",
            "region": "Chongqing-HotPotato",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "temperature_max", 35.0, "°C", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 69.0, "%", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "wet_bulb_temp", 23.9, "°C", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "heat_duration_days", 3.0, "d", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_heatwave_ground_truth,
        },
        # ---------------- 漏报方向（miss）：标准触发但线性评分被稀释 ----------------
        {
            # FWI=22.1 恰入高火险档（>22）且湿度 25%（<30 干燥）→ 国标规则 4 触发预警。
            # 但冬季低温 8℃ 使温度分量归零，线性平均被稀释：
            # (0.368×0.40 + 0.75×0.20 + 0×0.15)/0.75 ≈ 0.396 < 0.4 → 漏报。
            "case_id": "adv-fire-colddry-diluted",
            "difficulty": "L4",
            "category": "fire",
            "region": "Kunming-Yunnan",
            "ground_truth": True,
            "observations": [
                _make_obs("ECMWF", "FWI", 22.1, "", "2026-01-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 25.0, "%", "2026-01-11T12:00:00+08:00"),
                _make_obs("Station", "temperature", 8.0, "°C", "2026-01-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_fire_ground_truth,
        },
        {
            # 24h=30mm（恰达中雨线）+ 土壤 0.85（恰达饱和线）→ 国标规则 4（AND）触发预警。
            # 线性模型两因子各自都不高（0.30 / 0.625），均值 ≈0.298 < 0.45 → 漏报。
            "case_id": "adv-flood-saturation-and",
            "difficulty": "L4",
            "category": "flood",
            "region": "Guilin-Guangxi",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "rainfall_24h", 30.0, "mm", "2026-06-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_6h", 5.0, "mm", "2026-06-11T12:00:00+08:00"),
                _make_obs("Station", "soil_moisture", 0.85, "", "2026-06-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_flood_ground_truth,
        },
        {
            # SPI=-1.0 恰达中旱线（<=-1.0）→ 国标 SPI 分级触发预警；
            # 其余因子正常（Palmer 0 / 湿度 60%）在线性模型中均为 0 分稀释均值
            # ≈0.107 < 0.4 → 漏报（单一决定性因子被正常因子淹没）。
            "case_id": "adv-drought-spi-alone",
            "difficulty": "L4",
            "category": "drought",
            "region": "Taiyuan-Shanxi",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "SPI", -1.00, "", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "palmer_index", 0.00, "", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 60.0, "%", "2026-05-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_drought_ground_truth,
        },
        {
            # 最高温 33.5℃（>=33）+ 湿度 70.5%（>=70）→ 国标闷热判据（OR 通道）触发预警；
            # 线性模型：温度 0.458×0.35 + 时长 0.2×0.25 + 湿球 0×0.25 + 湿度 0.41×0.15
            # ≈ 0.272 < 0.4 → 漏报（低湿球 + 无持续记录把均值拉低）。
            "case_id": "adv-heat-muggy-or",
            "difficulty": "L4",
            "category": "heat",
            "region": "Guangzhou-PearlR",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "temperature_max", 33.5, "°C", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 70.5, "%", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "wet_bulb_temp", 20.0, "°C", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_heatwave_ground_truth,
        },
    ]


# ===========================================================================
# 难度级别常量
# ===========================================================================


class DifficultyLevel:
    """四级难度标签。"""

    L1_EASY = "L1"
    L2_MEDIUM = "L2"
    L3_HARD = "L3"
    L4_BEYOND = "L4"
