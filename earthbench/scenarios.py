"""八大高频风险 Alert 场景 + 场景管理 + Ground Truth 推导规则。

每个场景都有独立的 Ground Truth 推导规则（基于国标/行业标准阈值），
确保每输入一组观测数据就能自动产生可验证的 YES/NO。

场景列表：
1. Wildfire (森林火险) — FWI + 湿度 + 风速 + 降雨 → 是否预警
2. Flood (洪涝) — 降水量 + 水位 + 土壤湿度 → 是否预警
3. Drought (干旱) — SPI/Palmer + 湿度 + 降雨 + NDVI → 是否预警
4. HeatWave (热浪) — 温度 + 湿度(Wet Bulb) + 持续时间 → 是否预警
5. Landslide (滑坡/泥石流) — 激发雨强 + 前期有效降雨 + 土壤饱和 + 易发性
   → 是否预警（Phase A1 扩展，docs/expansion-plan.md）
6. Typhoon (台风/大风) — 日均风蒲福氏分档 + 阵风安全线 + 风雨耦合
   → 是否预警（Phase A2 扩展，docs/expansion-plan.md）
7. ColdWave (寒潮/冰冻) — 多窗口降幅 OR（24h≥8/48h≥10/72h≥12）× 日最低 ≤4℃
   AND（GB/T 20484-2017 四级体系顶档）
   → 是否预警（Phase B1 扩展，docs/expansion-plan.md）
8. Snow (暴雪/道路结冰) — 降雪等级分档 + 结冰复合（AND）+ 冻结掩膜
   → 是否预警（Phase B2 扩展，docs/expansion-plan.md）
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    DecisionTemplate,
    ExposureProfile,
    Observation,
    ScenarioCategory,
    ScenarioContext,
)

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
        exposure: ExposureProfile | None = None,
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
            exposure=exposure,
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
# - Landslide: 自然资源部—中国气象局《地质灾害气象风险预警》四级体系
#             （分级思路；具体数字为披露经验阈值，见下方常量注释）
# - Typhoon : GB/T 19201-2006《热带气旋等级》蒲福氏分档（8/10/12 级下限）
#             + 阵风 10 级临设安全线 + GB/T 28592-2012 暴雨线（风雨耦合）
# - Cold    : GB/T 20484-2017《冷空气等级》四级体系顶档「多窗口降幅 OR × 日最低」AND
#             分档（数字以规范原文为准，此处为对齐分档的披露口径）
# - Snow    : GB/T 28592-2012 降雪等级附表（24h 水当量分档）+「中雪档 ×
#             路温」道路结冰 AND 复合 + 冻结掩膜代理（t2m < 0.5℃ 计雪）
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
# ── 旱灾影响门控：缓发灾种，气象干旱成立 ≠ 影响成立 ──
# 干旱与骤发灾种（台风/暴雨）不同：现代城市受体被供水工程缓冲，
# 短时/局地气象干旱不构成可行动影响，报出来没有决策意义。指数通道
# 成立后还需任一影响通道满足才允许预警（两通道 AND）：
DROUGHT_DURATION_DAYS = 60.0  # 长旱：干旱持续 >= 60 天（两个评估周期）
DROUGHT_EXTENT_RATIO = 0.40  # 大面积：受旱面积占比 >= 40%
URBAN_WATER_DEFICIT = 0.10  # 城市供水缺水率 >= 10%（SL 424-2008 城市旱情中度线；原文本未在线核验，披露口径）
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

# === Landslide — 地质灾害气象风险预警（披露经验阈值） ===
# 分级思路参照自然资源部—中国气象局联合地质灾害气象风险预警四级体系
# （蓝/黄/橙/红）；下列具体数字为披露经验阈值（disclosed empirical），
# 沿用 cars_impact_models.json 中 wind 触发成本比的披露纪律。
# 落地实现时应逐条锚定区域规范/文献原文，并在此处更新来源注释。
RAIN_1H_TRIGGER = 25.0  # 激发雨强：高易发区 1h >= 25mm
RAIN_3H_TRIGGER = 50.0  # 激发雨强：高易发区 3h >= 50mm
EFFECTIVE_RAIN_3D = 100.0  # 前 3 日有效降雨（衰减累积）>= 100mm
SOIL_LANDSLIDE_SATURATED = 0.80  # 累积饱和通道土壤线
RAIN_24H_LANDSLIDE = 50.0  # 复合通道：24h 国标暴雨线（GB/T 28592-2012）
SOIL_LANDSLIDE_COMPOUND = 0.75  # 复合通道土壤线
SUSCEPTIBILITY_HIGH = 2.5  # 易发性分级（3=高/2=中/1=低），>=2.5 即高易发
SUSCEPTIBILITY_MID = 1.5  # >=1.5 即中易发及以上
LANDSLIDE_QUIET_RAIN = 1.0  # 退坡期：当前雨量 < 1mm 视为无雨
SOIL_QUIET = 0.50  # 退坡期：土壤 < 0.50 视为已排水固结

# === Typhoon — GB/T 19201-2006《热带气旋等级》蒲福氏分档 + 阵风安全线 ===
# 蒲福氏风级下限（国标热带气旋等级同构分档）：
# 8 级 17.2（热带风暴）/ 10 级 24.5（强热带风暴）/ 12 级 32.7（台风）
WIND_LEVEL_8 = 17.2  # 8 级下限：交通停运/户外作业停止线（人类活动口径）
WIND_LEVEL_10 = 24.5  # 10 级下限（日均口径参考档）
WIND_LEVEL_12 = 32.7  # 12 级下限：台风级
GUST_SAFETY_10 = 24.5  # 阵风 10 级线：临建设施/塔吊安全线（人类安全口径）
WIND_COMPOUND_7 = 13.9  # 风雨耦合：7 级下限
RAIN_COMPOUND = 50.0  # 风雨耦合：24h 暴雨线（GB/T 28592-2012）

# === Cold Wave — GB/T 20484-2017《冷空气等级》四级体系（披露口径） ===
# 2017 版为四级冷空气体系（弱/较强/强冷空气/寒潮），寒潮为顶档——
# 取代 2006 版的「寒潮/强寒潮/特强寒潮」三档（本实现早期曾误用旧结构，
# 结构性事实已对照维基条目「寒潮·中华人民共和国标准」小节核验）。
# 寒潮判定 = 多窗口降幅 OR × 日最低 AND：
#   24h 日最低降幅 >= 8℃ 或 48h >= 10℃ 或 72h >= 12℃，且日最低 <= 4℃。
# AND 纪律不变：绝对温低 ≠ 对人的异常事件（北方冬季常态低温无降幅
# 不触发）；降幅够但绝对温不低（基础温度高）亦不触发。
# 多窗口数值为披露口径，规范原文（CMA 官网 PDF）：
#   https://www.cma.gov.cn/zfxxgk/gknr/flfgbz/bz/202209/P020220921579662258389.pdf
COLD_DROP_24H = 8.0  # 寒潮：24h 日最低降温 >= 8℃（窗口一，或）
COLD_DROP_48H = 10.0  # 寒潮：48h 日最低降温 >= 10℃（窗口二，或）
COLD_DROP_72H = 12.0  # 寒潮：72h 日最低降温 >= 12℃（窗口三，或）
COLD_TMIN = 4.0  # 且 日最低气温 <= 4℃

# === Snow — GB/T 28592-2012 降雪等级附表（披露口径）+ 道路结冰复合 ===
# 降雪量为水当量（mm）；分档数字以规范原文为准，此处为对齐分档的披露口径。
SNOW_BLIZZARD = 10.0  # 暴雪：24h 降雪 >= 10mm
SNOW_HEAVY_BLIZZARD = 20.0  # 大暴雪：>= 20mm
SNOW_EXTREME_BLIZZARD = 30.0  # 特大暴雪：>= 30mm
SNOW_MODERATE = 2.5  # 中雪档下限（道路结冰复合的降雪条件）
ROAD_ICING_TEMP = 0.0  # 道路结冰：路面温度 <= 0℃
SNOW_FREEZE_MASK_C = 0.5  # 冻结掩膜（披露代理）：气温 < 0.5℃ 的降水计为雪


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


def _parse_ts(ts: str) -> datetime | None:
    """宽松解析 ISO 8601 时间戳。"""
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None


def _latest_ts(obs: list[dict]) -> datetime | None:
    """返回全部观测中的最大时间戳（作为「现在」）。"""
    tss = [_parse_ts(str(o.get("timestamp", ""))) for o in obs]
    tss = [t for t in tss if t is not None]
    return max(tss) if tss else None


def _persist_for(
    obs: list[dict],
    variable: str,
    cond: Any,
    hours: float,
    now: datetime | None = None,
) -> tuple[bool, dict[str, Any]]:
    """持续时间原语：变量在最近 `hours` 小时窗口内是否「持续」满足条件。

    Phase C 新增（docs/expansion-plan.md §6）：这是现有真值函数没有的能力。
    规则（披露口径）：窗口 = [now - hours, now]（now 默认取全部观测最大时间
    戳，闭区间）；窗口内该变量的**每个**采样点都必须满足 cond，且采样点数
    >= 2（避免单点宣称「持续」）。使得「雨停就解除」这类过早恢复错误可测。

    Returns: (是否持续满足, detail)
    """
    series = _obs_vals(obs, variable)
    if not series:
        return False, {"variable": variable, "reason": "无该变量观测"}

    if now is None:
        now = _latest_ts(obs)
    if now is None:
        return False, {"variable": variable, "reason": "无法解析时间戳"}

    start = now - timedelta(hours=hours)
    window: list[tuple[str, float]] = [
        (ts, v)
        for ts, v in series
        if (dt := _parse_ts(ts)) is not None and start <= dt <= now
    ]
    if len(window) < 2:
        return False, {
            "variable": variable,
            "window_hours": hours,
            "points": len(window),
            "reason": f"窗口内采样点 {len(window)} < 2，不足以判定持续",
        }

    all_ok = all(cond(v) for _, v in window)
    return all_ok, {
        "variable": variable,
        "window_hours": hours,
        "points": len(window),
        "min": min(v for _, v in window),
        "max": max(v for _, v in window),
        "all_satisfied": all_ok,
    }


def _detect_template_category(obs: list[dict]) -> str:
    """从观测变量推断模板套件用例所属灾种（供模板真值函数查表用）。

    顺序按特异性排序，避免误判：FWI→fire；water_level→flood；
    wind→typhoon；road_surface_temp/snowfall→snow；降温幅度→cold；
    temperature_max→heat。无匹配→"other"。
    """
    vs = {o["variable"] for o in obs}
    if "FWI" in vs:
        return "fire"
    if "water_level" in vs:
        return "flood"
    if "wind_speed" in vs or "wind_gust" in vs:
        return "typhoon"
    if "road_surface_temp" in vs or "snowfall_24h" in vs:
        return "snow"
    if "temperature_drop_24h" in vs or "temperature_min" in vs:
        return "cold"
    if "temperature_max" in vs:
        return "heat"
    return "other"


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
    """干旱 Ground Truth 推导（独立物理标准查表 × 影响门控）。

    干旱是缓发灾种：现代城市受体被供水工程缓冲，「气象干旱成立」
    不等于「影响成立」——短时、局地的指数红灯报出来没有决策意义。
    判定为两通道 AND 结构：

    A. 指数通道（GB/T 20481-2017《气象干旱等级》）：

       1. 最新 SPI <= -2.0（特旱）→ 档位 0.95
       2. 最新 SPI <= -1.5（重旱）→ 档位 0.85
       3. 最新 SPI <= -1.0（中旱）→ 档位 0.75
       4. Palmer <= -2.0（中旱）→ 档位 0.75
       5. Palmer <= -1.0（轻旱起步）且任一印证因子
          （湿度<25% / 月降雨<10mm / NDVI<0.3）→ 档位 0.65

    B. 影响门控（任一通道满足即通过；缺影响观测视为未验证）：

       - 干旱持续 >= 60 天（长旱）
       - 受旱面积占比 >= 40%（大面积）
       - 城市供水缺水率 >= 10%（SL 424-2008 城市旱情中度线，披露口径）

    A 且 B → 预警（A 的档位分）；A 成立但 B 不成立/缺证据 → 不预警
    （0.30）；A 不成立 → 不预警（0.25）。
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

    # ── A. 指数通道：定档但不立即返回 ──
    tier: tuple[bool, float, str, str] | None = None
    if spi is not None:
        if spi <= SPI_DROUGHT_EXTREME:
            tier = (True, 0.95, "GB/T 20481-2017 SPI 分级",
                    f"SPI {spi:.2f} <= -2.0（特旱）")
        elif spi <= SPI_DROUGHT_SEVERE:
            tier = (True, 0.85, "GB/T 20481-2017 SPI 分级",
                    f"SPI {spi:.2f} <= -1.5（重旱）")
        elif spi <= SPI_DROUGHT_MODERATE:
            tier = (True, 0.75, "GB/T 20481-2017 SPI 分级",
                    f"SPI {spi:.2f} <= -1.0（中旱）")
    if tier is None and palmer is not None and palmer <= PALMER_DROUGHT_MODERATE:
        tier = (True, 0.75, "GB/T 20481-2017 Palmer 分级",
                f"Palmer {palmer:.2f} <= -2.0（中旱）")
    if tier is None and palmer is not None and palmer <= PALMER_DROUGHT_LIGHT:
        confirms: list[str] = []
        if hum is not None and hum < HUMIDITY_DROUGHT:
            confirms.append(f"湿度{hum:.0f}%<25%")
        if rain_month is not None and rain_month < RAINFALL_DEFICIT:
            confirms.append(f"月降雨{rain_month:.0f}mm<10mm")
        if ndvi is not None and ndvi < NDVI_DEGRADE:
            confirms.append(f"NDVI {ndvi:.2f}<0.3")
        if confirms:
            tier = (True, 0.65, "GB/T 20481-2017 多因子印证",
                    f"Palmer {palmer:.2f} 轻旱，印证：{'；'.join(confirms)}")

    # ── B. 影响门控 ──
    duration = _latest(obs, "drought_duration_days")
    extent = _latest(obs, "affected_area_ratio")
    deficit = _latest(obs, "urban_water_deficit_rate")
    if duration is not None:
        explanation["drought_duration_days"] = duration
    if extent is not None:
        explanation["affected_area_ratio"] = extent
    if deficit is not None:
        explanation["urban_water_deficit_rate"] = deficit
    impact_ok: list[str] = []
    if duration is not None and duration >= DROUGHT_DURATION_DAYS:
        impact_ok.append(f"持续{duration:.0f}天≥{DROUGHT_DURATION_DAYS:.0f}")
    if extent is not None and extent >= DROUGHT_EXTENT_RATIO:
        impact_ok.append(f"受旱面积{extent * 100:.0f}%≥{DROUGHT_EXTENT_RATIO * 100:.0f}%")
    if deficit is not None and deficit >= URBAN_WATER_DEFICIT:
        impact_ok.append(f"城市缺水率{deficit * 100:.0f}%≥{URBAN_WATER_DEFICIT * 100:.0f}%")
    explanation["impact_gate"] = bool(impact_ok)
    explanation["impact_reasons"] = impact_ok

    # ── 合成：两通道 AND ──
    if tier is None:
        explanation["standard"] = "GB/T 20481-2017 SPI/Palmer 分级"
        explanation["detail"] = "未达到中旱及以上等级，不预警（指数通道未成立）"
        return False, 0.25, explanation

    _, tier_score, tier_std, tier_detail = tier
    if not impact_ok:
        explanation["standard"] = (
            "GB/T 20481-2017 SPI/Palmer 分级 × 影响门控（缓发灾种）"
        )
        miss = []
        if duration is not None:
            miss.append(f"持续{duration:.0f}天<{DROUGHT_DURATION_DAYS:.0f}")
        if extent is not None:
            miss.append(f"面积{extent * 100:.0f}%<{DROUGHT_EXTENT_RATIO * 100:.0f}%")
        if deficit is not None:
            miss.append(f"缺水率{deficit * 100:.0f}%<{URBAN_WATER_DEFICIT * 100:.0f}%")
        gap = "；".join(miss) if miss else "无任何影响观测（视为未验证）"
        explanation["detail"] = (
            f"指数通道成立（{tier_detail}）但影响门控未通过（{gap}），"
            "短时/局地干旱对现代城市不构成可行动影响，不预警"
        )
        return False, 0.30, explanation

    explanation["standard"] = "GB/T 20481-2017 SPI/Palmer 分级 × 影响门控（缓发灾种）"
    explanation["detail"] = (
        f"{tier_detail}；影响门控通过（{'；'.join(impact_ok)}），触发预警"
    )
    return True, tier_score, explanation


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


def infer_landslide_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """滑坡/泥石流 Ground Truth 推导（查表判定：三通道 + 退坡抑制）。

    分级思路参照自然资源部—中国气象局《地质灾害气象风险预警》四级体系
    （蓝/黄/橙/红）；具体阈值为披露经验阈值（见常量区注释），与 agents.py
    的线性加权评分完全脱钩：

    1. 激发雨强通道（高易发区）：1h >= 25mm 或 3h >= 50mm → 预警（0.95）
    2. 累积饱和通道（任意易发区）：前 3 日有效降雨 >= 100mm 且
       土壤 >= 0.80 → 预警（0.85）——体现「雨停 ≠ 风险停」的滞后性：
       当日无激发雨亦可触发
    3. 复合通道（中易发及以上）：24h 降雨 >= 50mm（GB/T 28592 暴雨线）
       且 土壤 >= 0.75 → 预警（0.80）
    4. 退坡抑制：当前雨量 < 1mm 且 土壤 < 0.50 → 强制不预警（0.15，
       斜坡已排水固结）
    """
    explanation: dict[str, Any] = {}

    rain1 = _max_val(obs, "rainfall_1h")
    rain3 = _max_val(obs, "rainfall_3h")
    rain24 = _max_val(obs, "rainfall_24h")
    eff3d = _latest(obs, "effective_rainfall_3d")
    soil = _max_val(obs, "soil_moisture")
    suscept = _latest(obs, "susceptibility")

    if rain1 is not None:
        explanation["rainfall_1h_max"] = round(rain1, 1)
    if rain3 is not None:
        explanation["rainfall_3h_max"] = round(rain3, 1)
    if rain24 is not None:
        explanation["rainfall_24h_max"] = round(rain24, 1)
    if eff3d is not None:
        explanation["effective_rainfall_3d_latest"] = round(eff3d, 1)
    if soil is not None:
        explanation["soil_moisture_max"] = round(soil, 3)
    if suscept is not None:
        explanation["susceptibility_class"] = int(suscept)

    # 0) 无降雨观测：独立标准无依据
    if rain1 is None and rain3 is None and rain24 is None and eff3d is None:
        return False, 0.0, {"reason": "no rainfall observations; 独立标准无依据"}

    # 1) 退坡抑制：雨停且土壤已排水
    quiet_rain = (rain1 is None or rain1 < LANDSLIDE_QUIET_RAIN) and (
        rain24 is None or rain24 < LANDSLIDE_QUIET_RAIN
    )
    if quiet_rain and soil is not None and soil < SOIL_QUIET:
        explanation["standard"] = "地质灾害气象风险预警（退坡期判据）"
        explanation["detail"] = (
            f"当前雨量 < {LANDSLIDE_QUIET_RAIN:.0f}mm 且土壤 {soil:.2f} < 0.50，"
            f"斜坡已排水固结，不预警"
        )
        return False, 0.15, explanation

    # 2) 激发雨强通道：高易发 + 短时强降雨
    if suscept is not None and suscept >= SUSCEPTIBILITY_HIGH:
        hit_1h = rain1 is not None and rain1 >= RAIN_1H_TRIGGER
        hit_3h = rain3 is not None and rain3 >= RAIN_3H_TRIGGER
        if hit_1h or hit_3h:
            trigger = rain1 if hit_1h else rain3
            window = "1h" if hit_1h else "3h"
            explanation["standard"] = "地质灾害气象风险预警（激发雨强通道）"
            explanation["detail"] = (
                f"高易发区 {window} 降雨 {trigger:.0f}mm 达激发线"
                f"（1h>={RAIN_1H_TRIGGER:.0f} 或 3h>={RAIN_3H_TRIGGER:.0f}），触发预警"
            )
            return True, 0.95, explanation

    # 3) 累积饱和通道：前期有效降雨 + 土壤饱和（当日无激发雨亦可触发）
    if (
        eff3d is not None
        and eff3d >= EFFECTIVE_RAIN_3D
        and soil is not None
        and soil >= SOIL_LANDSLIDE_SATURATED
    ):
        explanation["standard"] = "地质灾害气象风险预警（累积饱和通道）"
        explanation["detail"] = (
            f"前 3 日有效降雨 {eff3d:.0f}mm >= {EFFECTIVE_RAIN_3D:.0f}mm 且 "
            f"土壤 {soil:.2f} >= {SOIL_LANDSLIDE_SATURATED:.2f}，"
            f"饱和斜坡处于临界态（雨停 ≠ 风险停），触发预警"
        )
        return True, 0.85, explanation

    # 4) 复合通道：暴雨 + 土壤偏高 + 中易发及以上
    if (
        rain24 is not None
        and rain24 >= RAIN_24H_LANDSLIDE
        and soil is not None
        and soil >= SOIL_LANDSLIDE_COMPOUND
        and suscept is not None
        and suscept >= SUSCEPTIBILITY_MID
    ):
        explanation["standard"] = "地质灾害气象风险预警（复合通道）"
        explanation["detail"] = (
            f"24h 降雨 {rain24:.0f}mm >= {RAIN_24H_LANDSLIDE:.0f}mm（国标暴雨线）且 "
            f"土壤 {soil:.2f} >= {SOIL_LANDSLIDE_COMPOUND:.2f} 且易发性分级 "
            f"{int(suscept)} >= 2（中易发），触发预警"
        )
        return True, 0.80, explanation

    # 5) 未命中任何通道
    explanation["standard"] = "地质灾害气象风险预警（三通道查表）"
    explanation["detail"] = (
        "未命中任何激发雨强/累积饱和/复合阈值，不预警"
    )
    return False, 0.20, explanation


def infer_typhoon_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """台风/大风 Ground Truth 推导（查表判定：四级风档 + 阵风 + 风雨耦合）。

    依据 GB/T 19201-2006《热带气旋等级》蒲福氏分档（8/10/12 级下限）
    + 阵风 10 级临设安全线 + GB/T 28592-2012 暴雨线（风雨耦合），
    与 agents.py 的线性加权评分完全脱钩：

    1. 日均风 >= 32.7 m/s（12 级，台风级）→ 预警（0.95）
    2. 阵风 >= 24.5 m/s（10 级，临建设施/塔吊安全线）→ 预警（0.85）
       ——日均风不高亦可触发（雷雨大风/飑线场景），人类安全口径独立判定
    3. 日均风 >= 17.2 m/s（8 级，交通停运/户外作业停止线）→ 预警（0.80）
    4. 风雨耦合：日均风 >= 13.9 m/s（7 级）且 24h 降雨 >= 50mm
       （暴雨线）→ 预警（0.70）——AND 结构，两条件缺一不可
    """
    explanation: dict[str, Any] = {}

    wind = _max_val(obs, "wind_speed")
    gust = _max_val(obs, "wind_gust")
    rain24 = _max_val(obs, "rainfall_24h")

    if wind is not None:
        explanation["wind_speed_max"] = round(wind, 1)
    if gust is not None:
        explanation["wind_gust_max"] = round(gust, 1)
    if rain24 is not None:
        explanation["rainfall_24h_max"] = round(rain24, 1)

    # 0) 无风速观测：独立标准无依据
    if wind is None and gust is None:
        return False, 0.0, {"reason": "no wind observations; 独立标准无依据"}

    # 1) 台风级（12 级日均）
    if wind is not None and wind >= WIND_LEVEL_12:
        explanation["standard"] = "GB/T 19201-2006 热带气旋等级（台风级）"
        explanation["detail"] = (
            f"日均风 {wind:.1f} m/s >= {WIND_LEVEL_12} m/s（12 级，台风级），触发预警"
        )
        return True, 0.95, explanation

    # 2) 阵风通道（临建设施/塔吊安全线，日均不高亦可触发）
    if gust is not None and gust >= GUST_SAFETY_10:
        explanation["standard"] = "阵风 10 级临设安全线"
        explanation["detail"] = (
            f"阵风 {gust:.1f} m/s >= {GUST_SAFETY_10} m/s（10 级，临建设施/"
            f"塔吊安全线），触发预警"
        )
        return True, 0.85, explanation

    # 3) 8 级日均（交通停运/户外作业停止线）
    if wind is not None and wind >= WIND_LEVEL_8:
        explanation["standard"] = "GB/T 19201-2006 热带气旋等级（热带风暴级）"
        explanation["detail"] = (
            f"日均风 {wind:.1f} m/s >= {WIND_LEVEL_8} m/s（8 级，交通停运/"
            f"户外作业停止线），触发预警"
        )
        return True, 0.80, explanation

    # 4) 风雨耦合（AND 结构：7 级 + 暴雨，缺一不可）
    if (
        wind is not None
        and wind >= WIND_COMPOUND_7
        and rain24 is not None
        and rain24 >= RAIN_COMPOUND
    ):
        explanation["standard"] = "GB/T 19201-2006 + GB/T 28592-2012 风雨耦合"
        explanation["detail"] = (
            f"日均风 {wind:.1f} m/s >= {WIND_COMPOUND_7} m/s（7 级）且 "
            f"24h 降雨 {rain24:.0f}mm >= {RAIN_COMPOUND:.0f}mm（暴雨线），"
            f"风雨耦合触发预警"
        )
        return True, 0.70, explanation

    # 5) 未命中
    explanation["standard"] = "GB/T 19201-2006 蒲福氏分档查表"
    if wind is not None and wind >= 10.8:
        explanation["detail"] = (
            f"日均风 {wind:.1f} m/s 属 6 级以上但未达 8 级停运线，"
            f"且无阵风/耦合通道命中，不预警"
        )
        return False, 0.30, explanation
    explanation["detail"] = "未达任何风档/阵风/耦合阈值，不预警"
    return False, 0.20, explanation


def infer_cold_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """寒潮 Ground Truth 推导（GB/T 20484-2017 四级体系顶档查表）。

    2017 版《冷空气等级》为四级体系（弱/较强/强冷空气/寒潮），寒潮为
    顶档，判��� = **多窗口降幅 OR × 日最低 AND**：

    1. 24h 日最低降幅 >= 8℃，或 48h >= 10℃，或 72h >= 12℃（任一窗口）
       且 日最低 <= 4℃ → 寒潮，预警（0.95）
    2. AND 结构：任一窗口降幅达线但日最低 > 4℃（基础温度高）→ 不预警；
       日最低 <= 4℃ 但所有窗口均无降幅（北方冬季常态低温）→ 不预警。

    多窗口捕捉缓慢渗透型寒潮（48h 渐进下滑在 24h 窗口可能不达线）；
    AND 纪律保证绝对低温不等于对人的异常事件。

    降幅来源：24h 优先读 temperature_drop_24h 观测，缺失时由
    temperature_min 序列推导（D-1 − D，升温取 0）；48h/72h 窗口由
    序列最近 3/4 个不同日期推导（无对应观测覆写通道，披露）。
    """
    explanation: dict[str, Any] = {}

    tmin = _latest(obs, "temperature_min")
    if tmin is None:
        vals = [v for _, v in _obs_vals(obs, "temperature")]
        tmin = min(vals) if vals else None

    # 各窗口降幅：观测覆写（仅 24h）优先，其余由 tmin 日序列推导
    drops: dict[str, float | None] = {"24h": None, "48h": None, "72h": None}
    drop_source = "none"
    series = _obs_vals(obs, "temperature_min")
    by_day: dict[str, float] = {}
    for ts, v in series:
        by_day[ts[:10]] = v  # 同日多条以时序最后一条为准
    days = sorted(by_day.keys())
    if len(days) >= 2:
        drops["24h"] = by_day[days[-2]] - by_day[days[-1]]
        drop_source = "derived"
    if len(days) >= 3:
        drops["48h"] = by_day[days[-3]] - by_day[days[-1]]
    if len(days) >= 4:
        drops["72h"] = by_day[days[-4]] - by_day[days[-1]]
    drop_obs = _max_val(obs, "temperature_drop_24h")
    if drop_obs is not None:
        drops["24h"] = drop_obs
        drop_source = "observed"

    if tmin is None or drops["24h"] is None:
        return False, 0.0, {"reason": "no tmin/drop observations; 独立标准无依据"}
    drops = {k: max(v, 0.0) if v is not None else None for k, v in drops.items()}

    explanation["tmin_latest"] = round(tmin, 1)
    for k, v in drops.items():
        if v is not None:
            explanation[f"drop_{k}"] = round(v, 1)
    explanation["drop_source"] = drop_source

    # 窗口判定（OR）
    windows = [("24h", COLD_DROP_24H), ("48h", COLD_DROP_48H),
               ("72h", COLD_DROP_72H)]
    fired = [(w, th) for w, th in windows
             if drops[w] is not None and drops[w] >= th]
    any_drop_met = bool(fired)

    # 1) 寒潮（顶档）：任一窗口达线 且 日最低 <= 4℃
    if fired and tmin <= COLD_TMIN:
        fired_txt = "、".join(
            f"{w} 降幅 {drops[w]:.1f}℃ >= {th:.0f}℃" for w, th in fired)
        explanation["standard"] = "GB/T 20484-2017 冷空气等级（寒潮，顶档）"
        explanation["detail"] = (
            f"{fired_txt}（多窗口 OR），且日最低 {tmin:.1f}℃ <= "
            f"{COLD_TMIN:.0f}℃，构成寒潮（四级冷空气体系顶档），触发预警"
        )
        return True, 0.95, explanation

    # 2) AND 缺一：分级归因（不预警）
    explanation["standard"] = "GB/T 20484-2017 冷空气等级（多窗口查表）"
    if any_drop_met and tmin > COLD_TMIN:
        explanation["detail"] = (
            f"降幅达线但日最低 {tmin:.1f}℃ > {COLD_TMIN:.0f}℃，"
            f"绝对温度不低（基础温度偏高），不构成寒潮，不预警"
        )
        return False, 0.30, explanation
    if tmin <= COLD_TMIN and not any_drop_met:
        avail = "、".join(
            f"{w} 仅 {drops[w]:.1f}℃" for w, _ in windows if drops[w] is not None)
        explanation["detail"] = (
            f"日最低 {tmin:.1f}℃ <= {COLD_TMIN:.0f}℃ 属常态低温，但各窗口降幅"
            f"均未达线（{avail}；无寒潮降温，绝对温低 ≠ 对人的异常事件），不预警"
        )
        return False, 0.25, explanation
    explanation["detail"] = "各窗口降幅与极值均未达寒潮判据，不预警"
    return False, 0.20, explanation


def infer_snow_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """暴雪/道路结冰 Ground Truth 推导（查表 + 冻结掩膜代理）。

    降雪等级依据 GB/T 28592-2012 降雪等级附表（24h 降雪量水当量，
    分档数字以规范原文为准，此处为对齐分档的披露口径）；道路结冰为
    「中雪档 × 路温」AND 复合，与 agents.py 的线性加权评分完全脱钩：

    1. 24h 降雪 >= 30mm（特大暴雪）→ 预警（0.95）
    2. 24h 降雪 >= 20mm（大暴雪）→ 预警（0.90）
    3. 24h 降雪 >= 10mm（暴雪）→ 预警（0.85）
    4. 道路结冰复合：24h 降雪 >= 2.5mm（中雪档）且 路面温度 <= 0℃
       → 预警（0.70，AND 结构缺一不可）
    5. 冻结掩膜（披露代理口径）：snowfall 缺失时由 rainfall_24h 与气温
       推导——气温 >= 0.5℃ 的降水为雨，降雪量记 0（无论雨量多大
       都不构成雪灾）。
    """
    explanation: dict[str, Any] = {}

    snow = _max_val(obs, "snowfall_24h")
    snow_source = "observed"
    if snow is None:
        rain = _max_val(obs, "rainfall_24h")
        t2m = _latest(obs, "temperature")
        if rain is not None and t2m is not None:
            snow = rain if t2m < SNOW_FREEZE_MASK_C else 0.0
            snow_source = "proxy"

    road = _latest(obs, "road_surface_temp")
    if road is None:
        road = _latest(obs, "temperature")
    hum = _latest(obs, "humidity")

    if snow is None:
        return False, 0.0, {"reason": "no snowfall/rainfall observations; 独立标准无依据"}

    explanation["snow_24h"] = round(snow, 1)
    explanation["snow_source"] = snow_source
    if road is not None:
        explanation["road_surface_temp"] = round(road, 1)
    if hum is not None:
        explanation["humidity_latest"] = round(hum, 1)

    # 1) 特大暴雪
    if snow >= SNOW_EXTREME_BLIZZARD:
        explanation["standard"] = "GB/T 28592-2012 降雪等级（特大暴雪）"
        explanation["detail"] = (
            f"24h 降雪 {snow:.1f}mm >= {SNOW_EXTREME_BLIZZARD:.0f}mm"
            f"（特大暴雪），触发预警"
        )
        return True, 0.95, explanation

    # 2) 大暴雪
    if snow >= SNOW_HEAVY_BLIZZARD:
        explanation["standard"] = "GB/T 28592-2012 降雪等级（大暴雪）"
        explanation["detail"] = (
            f"24h 降雪 {snow:.1f}mm >= {SNOW_HEAVY_BLIZZARD:.0f}mm（大暴雪），触发预警"
        )
        return True, 0.90, explanation

    # 3) 暴雪
    if snow >= SNOW_BLIZZARD:
        explanation["standard"] = "GB/T 28592-2012 降雪等级（暴雪）"
        explanation["detail"] = (
            f"24h 降雪 {snow:.1f}mm >= {SNOW_BLIZZARD:.0f}mm（暴雪），触发预警"
        )
        return True, 0.85, explanation

    # 4) 道路结冰复合（AND：中雪档 × 路温 <= 0℃）
    if snow >= SNOW_MODERATE and road is not None and road <= ROAD_ICING_TEMP:
        explanation["standard"] = "道路结冰复合判据（中雪档 × 路温）"
        explanation["detail"] = (
            f"24h 降雪 {snow:.1f}mm >= {SNOW_MODERATE}mm（中雪档）且 "
            f"路面温度 {road:.1f}℃ <= {ROAD_ICING_TEMP:.0f}℃，道路结冰风险，触发预警"
        )
        return True, 0.70, explanation

    # 5) 未命中（分级归因）
    explanation["standard"] = "GB/T 28592-2012 降雪等级 + 结冰复合查表"
    if snow_source == "proxy" and snow == 0.0:
        explanation["detail"] = (
            "冻结掩膜：气温 >= 0.5℃，降水为雨而非雪，降雪量记 0，不构成雪灾，不预警"
        )
        return False, 0.15, explanation
    if snow >= SNOW_MODERATE and road is not None and road > ROAD_ICING_TEMP:
        explanation["detail"] = (
            f"24h 降雪 {snow:.1f}mm 达中雪档但路面温度 {road:.1f}℃ > 0℃，"
            f"无结冰复合，且降雪量未达暴雪档，不预警"
        )
        return False, 0.30, explanation
    explanation["detail"] = "降雪量与结冰条件均未达任何分档，不预警"
    return False, 0.20, explanation


# ===========================================================================
# 暴露与脆弱性层（Phase B 扩展，docs/expansion-plan.md 层 2）
#
# 设计不变量：暴露只调制「动作等级」，不改写物理真值。
#   物理等级（各灾种查表，独立） × 暴露分级（下方查表，独立）
#   → 动作矩阵（monitor / alert / dispatch）
# 暴露分级表与动作矩阵均为披露假设（国标无直接对应分级表时按披露纪律
# 发布；GB 51222 城市内涝防治重现期分级可作锚点参考，需核对）。
# ===========================================================================

# 动作矩阵的严重度分界：物理 GT score >= 0.90 视为高严重度（披露口径）
ACTION_SEVERE_SCORE = 0.90

# 动作矩阵（披露假设）：
#   物理 NO          → monitor（任何暴露；暴露不创造风险）
#   物理 YES-moderate: E3 → dispatch；E1/E2/E0 → alert
#   物理 YES-severe  : E3/E2 → dispatch；E1/E0 → alert


def infer_exposure_class(
    exposure: ExposureProfile | None,
) -> tuple[str, dict[str, Any]]:
    """暴露分级查表（E0-E3，披露假设）。

    E3 高暴露：有关键基础设施（医院/学校/机场等），或人口密度 high
    E2 中暴露：人口密度 mid / 城市用地 / 脆弱人群占比 >= 0.30 / 户外活动高
    E1 低暴露：明确的人口稀少或林地/农区
    E0 无画像：未提供 exposure（保守按低暴露处理，行为与 E1 同）

    与 Agent 的线性评分脱钩：本表是被基准真值与 Agent 共同引用的
    「标准常量」（类比双方都引用的国标阈值），判别力来自严重度估计，
    不来自暴露分级本身。
    """
    if exposure is None:
        return "E0", {"basis": "无暴露画像，按低暴露处理（E0）"}

    if exposure.critical_infrastructure:
        return "E3", {
            "basis": f"关键基础设施 {'/'.join(exposure.critical_infrastructure)}",
        }
    if exposure.population_density_class == "high":
        return "E3", {"basis": "人口密度 high"}

    reasons: list[str] = []
    if exposure.population_density_class == "mid":
        reasons.append("人口密度 mid")
    if exposure.land_use == "urban":
        reasons.append("城市用地")
    if exposure.vulnerable_group_ratio is not None and exposure.vulnerable_group_ratio >= 0.30:
        reasons.append(f"脆弱人群占比 {exposure.vulnerable_group_ratio:.0%}")
    if exposure.outdoor_activity_level == "high":
        reasons.append("户外活动高（假期/农忙）")
    if reasons:
        return "E2", {"basis": "；".join(reasons)}

    return "E1", {
        "basis": (
            f"人口 {exposure.population_density_class} / "
            f"用地 {exposure.land_use}"
        ),
    }


def infer_action_ground_truth(
    gt_fn: Any,
    obs: list[dict],
    exposure: ExposureProfile | None = None,
    **kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    """动作等级 Ground Truth 推导（物理等级 × 暴露分级 → 动作矩阵）。

    输入任意灾种的物理真值函数 gt_fn（返回 (decision, score, explanation)）
    与暴露画像，按披露动作矩阵输出动作等级：
    - "monitor"：物理不触发（无论暴露——暴露不创造风险）
    - "alert"  ：物理触发 + 暴露不足以升级动作
    - "dispatch"：物理触发 + 暴露/严重度组合要求资源调度

    真值独立性：物理等级沿用各灾种独立查表；暴露分级独立查表；
    本函数只做二维查表组合，不含任何评分公式。
    """
    decision, score, phys_exp = gt_fn(obs, **kwargs)
    exp_class, exp_detail = infer_exposure_class(exposure)

    explanation: dict[str, Any] = {
        "physical_decision": decision,
        "physical_score": round(score, 2),
        "physical_standard": phys_exp.get("standard", ""),
        "exposure_class": exp_class,
        "exposure_basis": exp_detail.get("basis", ""),
    }

    if not decision:
        action = "monitor"
        explanation["matrix"] = "物理未触发 × 任意暴露 → monitor（暴露不创造风险）"
    elif score >= ACTION_SEVERE_SCORE:
        if exp_class in ("E2", "E3"):
            action = "dispatch"
            explanation["matrix"] = (
                f"物理高严重度(score {score:.2f} >= {ACTION_SEVERE_SCORE}) "
                f"× {exp_class} → dispatch"
            )
        else:
            action = "alert"
            explanation["matrix"] = (
                f"物理高严重度(score {score:.2f}) × {exp_class} → alert"
            )
    else:
        if exp_class == "E3":
            action = "dispatch"
            explanation["matrix"] = (
                f"物理中严重度(score {score:.2f} < {ACTION_SEVERE_SCORE}) "
                f"× {exp_class} → dispatch"
            )
        else:
            action = "alert"
            explanation["matrix"] = (
                f"物理中严重度(score {score:.2f}) × {exp_class} → alert"
            )

    return action, explanation


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
    生成 Alert 基准测试套件（八大高频风险场景，共 40 个用例）。

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
                _make_obs(
                    "CMA-DroughtBulletin", "drought_duration_days", 90.0, "d",
                    "2026-07-01T00:00:00+08:00"
                ),
                _make_obs(
                    "CMA-DroughtBulletin", "affected_area_ratio", 0.50, "",
                    "2026-07-01T00:00:00+08:00"
                ),
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
                _make_obs(
                    "CMA-DroughtBulletin", "drought_duration_days", 75.0, "d",
                    "2026-07-01T00:00:00+08:00"
                ),
                _make_obs(
                    "WaterAuthority", "urban_water_deficit_rate", 0.12, "",
                    "2026-07-01T00:00:00+08:00"
                ),
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
                _make_obs(
                    "CMA-DroughtBulletin", "drought_duration_days", 92.0, "d",
                    "2026-07-01T00:00:00+08:00"
                ),
                _make_obs(
                    "CMA-DroughtBulletin", "affected_area_ratio", 0.45, "",
                    "2026-07-01T00:00:00+08:00"
                ),
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

    # ---- Landslide (5 L1-L4, Phase A1 扩展) ----
    suite.append(
        {
            # 高易发区短时激发雨强直接达线（1h=35 >= 25）→ 激发雨强通道
            "case_id": "landslide-l1-intensity-high-suscept",
            "difficulty": "L1",
            "category": "landslide",
            "region": "Wenchuan-Sichuan",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Station", "rainfall_1h", 35.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_3h", 60.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_24h", 70.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.82, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    90.0,
                    "mm",
                    "2026-07-11T12:00:00+08:00",
                ),
                _make_obs(
                    "GeoSurvey",
                    "susceptibility",
                    3.0,
                    "3=high/2=mid/1=low",
                    "2026-07-11T12:00:00+08:00",
                ),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        }
    )
    suite.append(
        {
            # 雨停已久：当前雨量 < 1mm 且土壤 0.35 已排水 → 退坡期强制不预警
            "case_id": "landslide-l1-quiet-stable",
            "difficulty": "L1",
            "category": "landslide",
            "region": "Chengdu-Panda",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "Station", "rainfall_1h", 0.2, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_3h", 0.8, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_24h", 0.8, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.35, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    15.0,
                    "mm",
                    "2026-07-11T12:00:00+08:00",
                ),
                _make_obs(
                    "GeoSurvey",
                    "susceptibility",
                    2.0,
                    "3=high/2=mid/1=low",
                    "2026-07-11T12:00:00+08:00",
                ),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        }
    )
    suite.append(
        {
            # 1h=20 未达激发线（高易发才适用且易发性仅中），但 24h=55 过
            # 国标暴雨线 + 土壤 0.76 + 中易发 → 复合通道
            "case_id": "landslide-l2-compound-mid-suscept",
            "difficulty": "L2",
            "category": "landslide",
            "region": "Guiyang-Guizhou",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Station", "rainfall_1h", 20.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_3h", 38.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_24h", 55.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.76, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    75.0,
                    "mm",
                    "2026-07-11T12:00:00+08:00",
                ),
                _make_obs(
                    "GeoSurvey",
                    "susceptibility",
                    2.0,
                    "3=high/2=mid/1=low",
                    "2026-07-11T12:00:00+08:00",
                ),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        }
    )
    suite.append(
        {
            # 冲突信号：前期有效降雨 110mm 累积偏高（险），但土壤已排水至
            # 0.48 且当日无激发雨（缓）→ 三通道全不触发，退坡观察期
            "case_id": "landslide-l3-drained-after-storm",
            "difficulty": "L3",
            "category": "landslide",
            "region": "Yichang-ThreeGorges",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "Station", "rainfall_1h", 5.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_3h", 12.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_24h", 12.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.48, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    110.0,
                    "mm",
                    "2026-07-11T12:00:00+08:00",
                ),
                _make_obs(
                    "GeoSurvey",
                    "susceptibility",
                    3.0,
                    "3=high/2=mid/1=low",
                    "2026-07-11T12:00:00+08:00",
                ),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        }
    )
    suite.append(
        {
            # 渐进润湿：土壤 0.55→0.72→0.81 与有效降雨 60→85→102 三日爬升，
            # 当日雨强不高（1h=14）但累积饱和通道在最新时刻达线 → 预警
            "case_id": "landslide-l4-progressive-wetting",
            "difficulty": "L4",
            "category": "landslide",
            "region": "Luding-Sichuan",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Sensor", "soil_moisture", 0.55, "", "2026-07-09T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.72, "", "2026-07-10T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "soil_moisture", 0.81, "", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    60.0,
                    "mm",
                    "2026-07-09T12:00:00+08:00",
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    85.0,
                    "mm",
                    "2026-07-10T12:00:00+08:00",
                ),
                _make_obs(
                    "Station",
                    "effective_rainfall_3d",
                    102.0,
                    "mm",
                    "2026-07-11T12:00:00+08:00",
                ),
                _make_obs(
                    "Station", "rainfall_1h", 14.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_3h", 25.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "rainfall_24h", 20.0, "mm", "2026-07-11T12:00:00+08:00"
                ),
                _make_obs(
                    "GeoSurvey",
                    "susceptibility",
                    3.0,
                    "3=high/2=mid/1=low",
                    "2026-07-11T12:00:00+08:00",
                ),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        }
    )

    # ---- Typhoon (5 L1-L4, Phase A2 扩展) ----
    suite.append(
        {
            # 台风级：日均风 35 m/s（12 级）直击登陆，风雨俱强
            "case_id": "typhoon-l1-typhoon-grade",
            "difficulty": "L1",
            "category": "typhoon",
            "region": "Wenzhou-Zhejiang",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Station", "wind_speed", 35.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 45.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_24h", 120.0, "mm", "2026-07-26T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        }
    )
    suite.append(
        {
            # 内陆盆地无风无雨（深居内陆，台风影响近乎为零）
            "case_id": "typhoon-l1-inland-calm",
            "difficulty": "L1",
            "category": "typhoon",
            "region": "Chengdu-Panda",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "Station", "wind_speed", 5.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 9.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_24h", 3.0, "mm", "2026-07-26T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        }
    )
    suite.append(
        {
            # 风雨耦合：日均 15 m/s（7 级）未达 8 级停运线、阵风 19 未达
            # 10 级线，但 7 级 + 暴雨 60mm 的 AND 复合达线 → 耦合通道
            "case_id": "typhoon-l2-wind-rain-coupled",
            "difficulty": "L2",
            "category": "typhoon",
            "region": "Haikou-Hainan",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Station", "wind_speed", 15.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 19.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_24h", 60.0, "mm", "2026-07-26T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        }
    )
    suite.append(
        {
            # 冲突信号：阵风 17 m/s 显著（8 级阵风）但未达 10 级临设线，
            # 日均仅 5 级、无降雨 → 各通道全不触发
            "case_id": "typhoon-l3-gust-below-line",
            "difficulty": "L3",
            "category": "typhoon",
            "region": "Xiamen-Fujian",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "Station", "wind_speed", 8.5, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 17.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_24h", 6.0, "mm", "2026-07-26T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        }
    )
    suite.append(
        {
            # 台风外围逼近：阵风序列 19→22.5→25 三时次爬升越过 10 级线
            # （峰值口径判定），日均爬升至 6 级 → 阵风通道
            "case_id": "typhoon-l4-gust-approaching",
            "difficulty": "L4",
            "category": "typhoon",
            "region": "Guangzhou-PearlR",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "Station", "wind_speed", 10.0, "m/s", "2026-07-24T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 11.0, "m/s", "2026-07-25T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 12.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 19.0, "m/s", "2026-07-24T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 22.5, "m/s", "2026-07-25T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_gust", 25.0, "m/s", "2026-07-26T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "rainfall_24h", 30.0, "mm", "2026-07-26T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        }
    )

    # ---- ColdWave (5 L1-L4, Phase B1 扩展) ----
    suite.append(
        {
            # 剧烈冷空气过程：24h 降温 14℃ 且日最低 -6℃ —— 多窗口 OR 直击
            # 24h 线（8℃）远超，顶档寒潮
            "case_id": "cold-l1-extreme-coldwave",
            "difficulty": "L1",
            "category": "cold",
            "region": "Nanjing-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_min", -6.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_drop_24h", 14.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 10.0, "m/s", "2026-01-15T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_cold_ground_truth,
        }
    )
    suite.append(
        {
            # 江南冬日常态：低温 1.5℃ 但无降温幅度（降幅 2℃）→ 不预警
            "case_id": "cold-l1-normal-winter",
            "difficulty": "L1",
            "category": "cold",
            "region": "Kunming-SpringCity",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "temperature_min", 1.5, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_drop_24h", 2.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 4.0, "m/s", "2026-01-15T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_cold_ground_truth,
        }
    )
    suite.append(
        {
            # 寒潮档：降幅 9℃ ≥ 8℃ 线且日最低 2.5℃ ≤ 4℃ → 寒潮（顶档）
            "case_id": "cold-l2-coldwave-grade",
            "difficulty": "L2",
            "category": "cold",
            "region": "Wuhan-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_min", 2.5, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_drop_24h", 9.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 7.0, "m/s", "2026-01-15T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_cold_ground_truth,
        }
    )
    suite.append(
        {
            # 冲突信号（AND 缺一）：秋末冷空气降幅 8.5℃ 达线，但基础温度高、
            # 日最低仍有 8℃ > 4℃ → 绝对温度不低，不构成寒潮
            "case_id": "cold-l3-drop-but-mild",
            "difficulty": "L3",
            "category": "cold",
            "region": "Guangzhou-PearlR",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "temperature_min", 8.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_drop_24h", 8.5, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 4.0, "m/s", "2026-01-15T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_cold_ground_truth,
        }
    )
    suite.append(
        {
            # 渐进寒潮：日最低两日下滑 9.5 → -1.5℃（序列推导降幅 11℃，
            # 亦有过程降幅观测印证），风寒 10 m/s → 寒潮顶档
            "case_id": "cold-l4-progressive-coldwave",
            "difficulty": "L4",
            "category": "cold",
            "region": "WestLake-Hangzhou",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "temperature_min", 9.5, "°C", "2026-01-14T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_min", -1.5, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "CMA", "temperature_drop_24h", 11.0, "°C", "2026-01-15T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "wind_speed", 10.0, "m/s", "2026-01-15T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_cold_ground_truth,
        }
    )

    # ---- Snow (5 L1-L4, Phase B2 扩展) ----
    suite.append(
        {
            # 特大暴雪：24h 降雪（水当量）35mm 直击最高档
            "case_id": "snow-l1-extreme-blizzard",
            "difficulty": "L1",
            "category": "snow",
            "region": "Harbin-Heilongjiang",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "snowfall_24h", 35.0, "mm", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "road_surface_temp", -5.0, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 80.0, "%", "2026-01-20T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_snow_ground_truth,
        }
    )
    suite.append(
        {
            # 盆地冬日无雪：微量降雪 0.5mm（小雪档以下）、地表 2℃ 未冻
            "case_id": "snow-l1-no-snow-day",
            "difficulty": "L1",
            "category": "snow",
            "region": "Chengdu-Panda",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "snowfall_24h", 0.5, "mm", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "road_surface_temp", 2.0, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 40.0, "%", "2026-01-20T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_snow_ground_truth,
        }
    )
    suite.append(
        {
            # 暴雪档：24h 降雪 12mm（>= 10 暴雪线，未到 20 大暴雪线）
            "case_id": "snow-l2-blizzard-grade",
            "difficulty": "L2",
            "category": "snow",
            "region": "Urumqi-Xinjiang",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "snowfall_24h", 12.0, "mm", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "road_surface_temp", -2.0, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 75.0, "%", "2026-01-20T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_snow_ground_truth,
        }
    )
    suite.append(
        {
            # 冲突信号（AND 缺一）：中雪 4mm 达中雪档，但白天地表温度
            # 1.5℃ > 0℃ —— 无结冰复合，且未达暴雪档 → 不预警
            "case_id": "snow-l3-moderate-no-icing",
            "difficulty": "L3",
            "category": "snow",
            "region": "Wuhan-Yangtze",
            "ground_truth": False,
            "observations": [
                _make_obs(
                    "CMA", "snowfall_24h", 4.0, "mm", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "road_surface_temp", 1.5, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 55.0, "%", "2026-01-20T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_snow_ground_truth,
        }
    )
    suite.append(
        {
            # 雨夹雪转雪复合结冰：降雪 6mm（大雪档但未达暴雪线）、
            # 气温 0.2℃（冻结掩膜内）、路温 -3℃ → 道路结冰复合通道
            "case_id": "snow-l4-icing-compound",
            "difficulty": "L4",
            "category": "snow",
            "region": "WestLake-Hangzhou",
            "ground_truth": True,
            "observations": [
                _make_obs(
                    "CMA", "snowfall_24h", 6.0, "mm", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "temperature", 0.2, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Sensor", "road_surface_temp", -3.0, "°C", "2026-01-20T12:00:00+08:00"
                ),
                _make_obs(
                    "Station", "humidity", 85.0, "%", "2026-01-20T12:00:00+08:00"
                ),
            ],
            "_gt_fn": infer_snow_ground_truth,
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
            # 城市短旱：SPI=-2.6（特旱档）、湿度 22%、月雨 5mm、NDVI 0.22
            # ——指数通道全红；但持续 22 天（<60）、面积 12%（<40%）、
            # 城市缺水率 4%（<10%）三个影响通道全部未达标 → 不预警
            # （现代城市被供水工程缓冲，短时局地干旱报出来没有行动意义）。
            # 线性模型只读指数类观测：1.05×0.30 + 0.267×0.15 + 0.833×0.15
            # + 0.933×0.15 ≈ 0.620 ≥ 0.4 → 误报（对影响观测天然失明）。
            "case_id": "adv-drought-urban-short-no-impact",
            "difficulty": "L4",
            "category": "drought",
            "region": "Guangzhou-PearlR",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "SPI", -2.60, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 22.0, "%", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_monthly", 5.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "NDVI", 0.22, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("CMA-DroughtBulletin", "drought_duration_days", 22.0, "d", "2026-07-11T12:00:00+08:00"),
                _make_obs("CMA-DroughtBulletin", "affected_area_ratio", 0.12, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("WaterAuthority", "urban_water_deficit_rate", 0.04, "", "2026-07-11T12:00:00+08:00"),
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
            # 中旱+影响成立：SPI=-1.2（恰过中旱线）但持续 95 天（≥60）且
            # 受旱面积 55%（≥40%）→ 指数通道 0.75 × 影响门控通过 → 预警。
            # 线性模型只读指数类观测：SPI 0.35×0.30 + NDVI 0.167×0.15
            # ≈ 0.130 < 0.4 → 漏报（持续/面积两类影响观测对它不存在）。
            "case_id": "adv-drought-impact-gated",
            "difficulty": "L4",
            "category": "drought",
            "region": "Urumqi-Xinjiang",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "SPI", -1.20, "", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "palmer_index", 0.00, "", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "humidity", 58.0, "%", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_monthly", 70.0, "mm", "2026-05-11T12:00:00+08:00"),
                _make_obs("Station", "NDVI", 0.45, "", "2026-05-11T12:00:00+08:00"),
                _make_obs("CMA-DroughtBulletin", "drought_duration_days", 95.0, "d", "2026-05-11T12:00:00+08:00"),
                _make_obs("CMA-DroughtBulletin", "affected_area_ratio", 0.55, "", "2026-05-11T12:00:00+08:00"),
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
        # ---------------- Landslide（Phase A1 扩展）----------------
        {
            # 激发雨强 26mm/1h 已过 25mm 线，但易发性分级 low（1/3）——
            # 激发雨强通道要求高易发（AND 条件缺一）→ 不预警。
            # 24h=45mm < 50 复合线、前期有效降雨 40mm < 100 → 其余通道亦不触发。
            # 线性模型：雨强 1.0×0.35 + 24h 0.45×0.35 + 土壤 0.59×0.10
            # + 有效降雨 0.27×0.10 + 易发 0.33×0.10 ≈ 0.63 >= 0.4 → 误报。
            "case_id": "adv-landslide-intensity-low-suscept",
            "difficulty": "L4",
            "category": "landslide",
            "region": "Wenchuan-Sichuan",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "rainfall_1h", 26.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_3h", 42.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_24h", 45.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Sensor", "soil_moisture", 0.50, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "effective_rainfall_3d", 40.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("GeoSurvey", "susceptibility", 1.0, "3=high/2=mid/1=low", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        },
        {
            # 当日几乎无雨（1h=3mm、24h=8mm），但前 3 日有效降雨 120mm 且
            # 土壤 0.86 饱和 → 累积饱和通道触发（雨停 ≠ 风险停，滞后性）。
            # 线性模型只看当前雨强（3/25=0.12、8/100=0.08），饱和与易发分量
            # 仅各占 0.10 权重，归一后 ≈ 0.35 < 0.4 → 漏报。
            "case_id": "adv-landslide-antecedent-diluted",
            "difficulty": "L4",
            "category": "landslide",
            "region": "Guiyang-Guizhou",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "rainfall_1h", 3.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_3h", 8.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "rainfall_24h", 8.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("Sensor", "soil_moisture", 0.86, "", "2026-07-11T12:00:00+08:00"),
                _make_obs("Station", "effective_rainfall_3d", 120.0, "mm", "2026-07-11T12:00:00+08:00"),
                _make_obs("GeoSurvey", "susceptibility", 3.0, "3=high/2=mid/1=low", "2026-07-11T12:00:00+08:00"),
            ],
            "_gt_fn": infer_landslide_ground_truth,
        },
        # ---------------- Typhoon（Phase A2 扩展）----------------
        {
            # 风雨各差一线的次阈值混合：日均 16.5 < 17.2（8 级线差 0.7）、
            # 阵风 22 < 24.5（10 级线）、耦合通道 7 级已达但降雨 48 < 50
            # （暴雨线差 2mm，AND 缺一）→ 四条线全不触发，不预警。
            # 线性模型：风 0.96×0.40 + 阵风 0.90×0.25 + 雨 0.48×0.35
            # ≈ 0.78 >= 0.45 → 误报（OR 化的均值把 AND 缺一抹平）。
            "case_id": "adv-typhoon-subthreshold-mix",
            "difficulty": "L4",
            "category": "typhoon",
            "region": "Shenzhen-Coast",
            "ground_truth": False,
            "observations": [
                _make_obs("Station", "wind_speed", 16.5, "m/s", "2026-07-26T12:00:00+08:00"),
                _make_obs("Station", "wind_gust", 22.0, "m/s", "2026-07-26T12:00:00+08:00"),
                _make_obs("CMA", "rainfall_24h", 48.0, "mm", "2026-07-26T12:00:00+08:00"),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        },
        {
            # 雷雨大风/飑线：日均风仅 6 m/s（4 级）但阵风 26 m/s（10 级）
            # ——临设/塔吊安全线独立判定（人类安全口径），日均不高亦触发。
            # 线性模型：风 0.35×0.40 + 阵风 1.0×0.25 + 雨 0.02×0.35
            # ≈ 0.40 < 0.45 → 漏报（决定性阵风因子权重被稀释）。
            "case_id": "adv-typhoon-gust-diluted",
            "difficulty": "L4",
            "category": "typhoon",
            "region": "Wuhan-Yangtze",
            "ground_truth": True,
            "observations": [
                _make_obs("Station", "wind_speed", 6.0, "m/s", "2026-07-26T12:00:00+08:00"),
                _make_obs("Station", "wind_gust", 26.0, "m/s", "2026-07-26T12:00:00+08:00"),
                _make_obs("CMA", "rainfall_24h", 2.0, "mm", "2026-07-26T12:00:00+08:00"),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
        },
        # ---------------- ColdWave（Phase B1 扩展）----------------
        {
            # 北方冬季常态低温：日最低 -10℃ 极低，但 24h 降幅仅 3℃ ——
            # 常态低温不是对人的异常事件（AND 缺「降幅」）→ 不预警。
            # 线性模型：温度分量 (4-(-10))/8=1.75→1.0×0.40 + 降幅 0.3×0.35
            # + 风寒 0.5×0.25 ≈ 0.63 >= 0.45 → 误报（绝对低温主导误判）。
            "case_id": "adv-cold-chronic-low-no-drop",
            "difficulty": "L4",
            "category": "cold",
            "region": "Harbin-Heilongjiang",
            "ground_truth": False,
            "observations": [
                _make_obs("CMA", "temperature_min", -10.0, "°C", "2026-01-15T12:00:00+08:00"),
                _make_obs("CMA", "temperature_drop_24h", 3.0, "°C", "2026-01-15T12:00:00+08:00"),
                _make_obs("Station", "wind_speed", 6.0, "m/s", "2026-01-15T12:00:00+08:00"),
            ],
            "_gt_fn": infer_cold_ground_truth,
        },
        {
            # 寒潮双线刚过（AND 边界）：降幅 8.5℃ ≥ 8 且日最低 3.8℃ ≤ 4
            # —— 双条件同时成立，触发寒潮预警。
            # 线性模型：温度分量 (4-3.8)/8=0.025×0.40 + 降幅 0.85×0.35
            # + 风寒 0.25×0.25 ≈ 0.37 < 0.45 → 漏报（AND 双中档被均值稀释，
            # 任一单分量都不足以推过阈值）。
            "case_id": "adv-cold-both-just-met",
            "difficulty": "L4",
            "category": "cold",
            "region": "Chongqing-HotPotato",
            "ground_truth": True,
            "observations": [
                _make_obs("CMA", "temperature_min", 3.8, "°C", "2026-01-15T12:00:00+08:00"),
                _make_obs("CMA", "temperature_drop_24h", 8.5, "°C", "2026-01-15T12:00:00+08:00"),
                _make_obs("Station", "wind_speed", 3.0, "m/s", "2026-01-15T12:00:00+08:00"),
            ],
            "_gt_fn": infer_cold_ground_truth,
        },
        # ---------------- Snow（Phase B2 扩展）----------------
        {
            # 雨非雪（冻结掩膜）：24h 降水 40mm 强势，但气温 3.0℃ >= 0.5℃
            # —— 降水为雨而非雪，降雪量记 0 → 不构成雪灾，不预警。
            # 线性模型只看降水总量（40/40=1.0×0.30）与湿度，掩膜前的
            # 降水分量主导：≈ 0.51 >= 0.45 → 误报（雨雪口径混淆）。
            "case_id": "adv-snow-rain-not-snow",
            "difficulty": "L4",
            "category": "snow",
            "region": "Shenzhen-Coast",
            "ground_truth": False,
            "observations": [
                _make_obs("CMA", "rainfall_24h", 40.0, "mm", "2026-01-20T12:00:00+08:00"),
                _make_obs("Station", "temperature", 3.0, "°C", "2026-01-20T12:00:00+08:00"),
                _make_obs("Sensor", "road_surface_temp", 3.0, "°C", "2026-01-20T12:00:00+08:00"),
                _make_obs("Station", "humidity", 60.0, "%", "2026-01-20T12:00:00+08:00"),
            ],
            "_gt_fn": infer_snow_ground_truth,
        },
        {
            # 结冰双线刚过（AND 边界）：降雪 2.6mm 刚过中雪线 2.5，
            # 路温 -2℃ <= 0℃ —— 双条件同时成立，触发结冰复合预警。
            # 线性模型：降雪 0.26×0.35 + 路温 0.2×0.20 + 湿度 0.1×0.15
            # ≈ 0.21 < 0.45 → 漏报（AND 双中低档被均值稀释）。
            "case_id": "adv-snow-icing-just-met",
            "difficulty": "L4",
            "category": "snow",
            "region": "Chongqing-HotPotato",
            "ground_truth": True,
            "observations": [
                _make_obs("CMA", "snowfall_24h", 2.6, "mm", "2026-01-20T12:00:00+08:00"),
                _make_obs("Sensor", "road_surface_temp", -2.0, "°C", "2026-01-20T12:00:00+08:00"),
                _make_obs("Station", "humidity", 55.0, "%", "2026-01-20T12:00:00+08:00"),
            ],
            "_gt_fn": infer_snow_ground_truth,
        },
    ]


def get_exposure_suite() -> list[dict[str, Any]]:
    """暴露变体套件 — Phase B（层 2）暴露脆弱性建模（docs/expansion-plan.md）。

    设计：同一物理观测 × 不同暴露画像 → 不同动作等级真值（动作矩阵）。
    - 配对用例（成对同观测、异暴露）：检验「暴露调制动作、不改写物理真值」
    - 不变量用例：物理不触发 + 高暴露 → monitor（暴露不创造风险）
    - 两个动作级陷阱（L4）：线性置信度作严重度代理时双向出错——
      · exp-typhoon-severe-metro-E2：物理 severe（GT 0.95）但线性置信度
        0.67 < 0.85 → baseline 欠响应（alert，真值 dispatch）
      · exp-typhoon-moderate-urban-E2：物理 moderate（GT 0.70）但线性
        置信度 0.86 >= 0.85 → baseline 过响应（dispatch，真值 alert）

    每个用例含 observations / exposure（dict，构建时转 ExposureProfile）/
    action_truth / _gt_fn（物理真值函数）/ ground_truth（物理 YES/NO 参考值）。
    """
    ts = "2026-07-26T12:00:00+08:00"

    def _exp(
        pop: str = "unknown",
        land_use: str = "unknown",
        infra: list[str] | None = None,
        vulnerable: float | None = None,
        outdoor: str = "unknown",
    ) -> dict:
        return {
            "population_density_class": pop,
            "land_use": land_use,
            "critical_infrastructure": infra or [],
            "vulnerable_group_ratio": vulnerable,
            "outdoor_activity_level": outdoor,
        }

    # 物理观测组（复用基础套件已验证的形态）
    flood_moderate = [
        _make_obs("CMA", "rainfall_24h", 80.0, "mm", ts),  # 暴雨 0.85（中严重度）
        _make_obs("Sensor", "soil_moisture", 0.55, "", ts),
        _make_obs("Hydro", "water_level", 3.0, "m", ts),
    ]
    typhoon_severe = [
        _make_obs("Station", "wind_speed", 33.0, "m/s", ts),  # 12 级 0.95（高严重度）
        _make_obs("Station", "wind_gust", 41.0, "m/s", ts),
        _make_obs("CMA", "rainfall_24h", 5.0, "mm", ts),
    ]
    typhoon_moderate = [
        # 风雨耦合 0.70（中严重度）：17.1 < 17.2 八级线、24.4 < 24.5 阵风线，
        # 但 7 级 + 暴雨 60mm 的 AND 复合达线 —— 线性置信度 0.86 过代理线
        _make_obs("Station", "wind_speed", 17.1, "m/s", ts),
        _make_obs("Station", "wind_gust", 24.4, "m/s", ts),
        _make_obs("CMA", "rainfall_24h", 60.0, "mm", ts),
    ]
    heat_moderate = [
        # 高温黄色 0.70（中严重度）：35℃ × 3 天 × 湿球 >= 24
        _make_obs("CMA", "temperature_max", 34.5, "°C", "2026-07-24T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 35.0, "°C", "2026-07-25T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 34.0, "°C", "2026-07-26T12:00:00+08:00"),
        _make_obs("Station", "humidity", 72.0, "%", ts),
        _make_obs("Station", "wet_bulb_temp", 27.5, "°C", ts),
        _make_obs("Station", "heat_duration_days", 3.0, "d", ts),
    ]
    landslide_moderate = [
        # 复合通道 0.80（中严重度）
        _make_obs("Station", "rainfall_1h", 20.0, "mm", ts),
        _make_obs("Station", "rainfall_3h", 38.0, "mm", ts),
        _make_obs("Station", "rainfall_24h", 55.0, "mm", ts),
        _make_obs("Sensor", "soil_moisture", 0.76, "", ts),
        _make_obs("Station", "effective_rainfall_3d", 75.0, "mm", ts),
        _make_obs("GeoSurvey", "susceptibility", 2.0, "3=high/2=mid/1=low", ts),
    ]
    flood_none = [
        _make_obs("CMA", "rainfall_24h", 0.0, "mm", ts),
        _make_obs("Sensor", "soil_moisture", 0.20, "", ts),
        _make_obs("Hydro", "water_level", 2.0, "m", ts),
    ]

    def _case(
        case_id: str,
        difficulty: str,
        category: str,
        region: str,
        observations: list[dict],
        exposure: dict | None,
        action_truth: str,
        physical_truth: bool,
        gt_fn: Any,
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "difficulty": difficulty,
            "category": category,
            "region": region,
            "observations": observations,
            "exposure": exposure,
            "action_truth": action_truth,
            "ground_truth": physical_truth,
            "_gt_fn": gt_fn,
        }

    return [
        # ---- 配对 1：暴雨（中严重度）城区 vs 林区 ----
        _case(
            "exp-flood-urban-E3-dispatch", "L2", "flood", "Nanjing-Yangtze",
            flood_moderate,
            _exp(pop="high", land_use="urban", infra=["hospital"]),
            "dispatch", True, infer_flood_ground_truth,
        ),
        _case(
            "exp-flood-forest-E1-alert", "L2", "flood", "GreaterKhingan",
            flood_moderate,
            _exp(pop="none", land_use="forest"),
            "alert", True, infer_flood_ground_truth,
        ),
        # ---- 配对 2：台风级（高严重度） metro vs 近海无暴露 ----
        # metro 案例同时是动作级陷阱 1（欠响应）：物理 severe（GT 0.95）
        # 但线性置信度 0.67 < 0.85 → baseline 给 alert，真值 dispatch
        _case(
            "exp-typhoon-severe-metro-E2-dispatch", "L4", "typhoon", "Guangzhou-PearlR",
            typhoon_severe,
            _exp(pop="mid", land_use="urban"),
            "dispatch", True, infer_typhoon_ground_truth,
        ),
        _case(
            "exp-typhoon-severe-offshore-E1-alert", "L3", "typhoon", "Wenzhou-Zhejiang",
            typhoon_severe,
            _exp(pop="none", land_use="rural"),
            "alert", True, infer_typhoon_ground_truth,
        ),
        # ---- 配对 3：高温黄色（中严重度）医院密集 vs 郊区 ----
        _case(
            "exp-heat-hospital-E3-dispatch", "L3", "heat", "Chongqing-HotPotato",
            heat_moderate,
            _exp(pop="high", land_use="urban", infra=["hospital"], vulnerable=0.40),
            "dispatch", True, infer_heatwave_ground_truth,
        ),
        _case(
            "exp-heat-suburb-E1-alert", "L3", "heat", "Kunming-SpringCity",
            heat_moderate,
            _exp(pop="low", land_use="rural"),
            "alert", True, infer_heatwave_ground_truth,
        ),
        # ---- 配对 4：滑坡复合通道（中严重度）村镇 vs 学校 ----
        _case(
            "exp-landslide-village-E2-alert", "L3", "landslide", "Guiyang-Guizhou",
            landslide_moderate,
            _exp(pop="mid", land_use="rural"),
            "alert", True, infer_landslide_ground_truth,
        ),
        _case(
            "exp-landslide-school-E3-dispatch", "L3", "landslide", "Guiyang-Guizhou",
            landslide_moderate,
            _exp(pop="mid", land_use="rural", infra=["school"]),
            "dispatch", True, infer_landslide_ground_truth,
        ),
        # ---- 不变量：物理不触发 × 高暴露 → monitor ----
        _case(
            "exp-flood-nohazard-metro-E3-monitor", "L3", "flood", "Wuhan-Yangtze",
            flood_none,
            _exp(pop="high", land_use="urban", infra=["subway"]),
            "monitor", False, infer_flood_ground_truth,
        ),
        # ---- 动作级陷阱 2：过响应（物理 moderate，线性置信度 0.86 >= 0.85）----
        _case(
            "exp-typhoon-moderate-urban-E2-overreach", "L4", "typhoon", "Xiamen-Fujian",
            typhoon_moderate,
            _exp(pop="mid", land_use="urban"),
            "alert", True, infer_typhoon_ground_truth,
        ),
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


# ===========================================================================
# 决策模板闭环层（Phase C 扩展，docs/expansion-plan.md §6）
#
# ALERT 已有各灾种 infer_*_ground_truth；本节补齐 DISPATCH / UPGRADE /
# CLOSE / RECOVER 四个模板的查表真值。全部「阈值 + 披露口径」，并复用
# _persist_for 持续时间原语做 RECOVER 的持续窗判定（过早恢复可测）。
# ===========================================================================

# DISPATCH（资源预置）— 阈值 + 附加条件 AND 复合（披露口径）
DISPATCH_FIRE_FWI = 40.0      # FWI 高危险（GB/T 36743 高火险档近似）
DISPATCH_FLOOD_WATER = 4.0    # 0.8 × 5.0m 警戒水位（披露）
DISPATCH_FLOOD_RAIN24 = 30.0  # 过去 24h 降雨持续
DISPATCH_TYPHOON_WIND = 24.5  # 10 级风线（GB/T 19201-2006，与 CLOSE 口径同步）

# UPGRADE（升档）— 等级比较 / 趋势外推（披露口径）
UPGRADE_HEAT_CURRENT = 35.0   # 黄色高温线（已触发档）
UPGRADE_HEAT_NEXT = 36.5      # 逼近橙色档的趋势外推线（披露）
UPGRADE_FLOOD_WATER = 4.5     # 警戒 5.0 - 0.5 的升档线（披露）
UPGRADE_TYPHOON_WIND = 17.2   # 8 级（GB/T 19201-2006）
UPGRADE_TYPHOON_GUST = 24.5   # 阵风 10 级临设线

# CLOSE（封闭/暂停）— 危险等级 × 暴露（披露口径）
CLOSE_TYPHOON_WIND = 24.5     # 10 级
CLOSE_SNOW_SNOW = 10.0        # 暴雪线（GB/T 28592-2012）
CLOSE_SNOW_ROAD = 0.0         # 路面冰点
CLOSE_FLOOD_WATER = 5.0       # 警戒水位（同步 RECOVER 的 5.0 口径）

# RECOVER（解除/恢复）— 条件解除后须持续 N 小时（披露口径）
RECOVER_FIRE_RAIN24 = 30.0    # 24h 透雨
RECOVER_FIRE_FWI = 40.0
RECOVER_FIRE_SUSTAIN_H = 6.0  # 火险降档后 6h 无反弹
RECOVER_FLOOD_WATER = 4.5     # 警戒 5.0 - 0.5
RECOVER_FLOOD_SUSTAIN_H = 6.0
RECOVER_SNOW_ROAD = 0.5       # 路温升至冰点以上 0.5℃
RECOVER_SNOW_SUSTAIN_H = 6.0


def infer_dispatch_ground_truth(
    obs: list[dict], exposure: ExposureProfile | None = None
) -> tuple[bool, float, dict[str, Any]]:
    """DISPATCH（是否预置资源）真值：高危险 + 附加条件 AND 复合（披露）。

    - fire 消防预置：FWI >= 40 且暴露 E2+（景区/假期/关键设施）→ 预置
    - flood 抢险待命：水位 >= 4.0（0.8×警戒）且 24h 降雨 >= 30 → 待命
    - typhoon 防风预置：风速 >= 24.5（10 级）且 E3（关键设施）→ 预置
    与 Agent 的线性评分脱钩：真值做 AND 查表，baseline 做置信度阈值。
    """
    cat = _detect_template_category(obs)
    exp_class, exp_detail = infer_exposure_class(exposure)
    explanation: dict[str, Any] = {
        "category": cat,
        "exposure_class": exp_class,
        "exposure_basis": exp_detail.get("basis", ""),
    }

    if cat == "fire":
        fwi = _latest(obs, "FWI")
        if fwi is None:
            return False, 0.20, {"category": "fire", "reason": "无 FWI 观测"}
        if fwi >= DISPATCH_FIRE_FWI and exp_class in ("E2", "E3"):
            explanation["standard"] = (
                "FWI >= 40 且暴露 E2+（景区/假期/关键设施）→ 消防预置"
            )
            explanation["detail"] = f"FWI {fwi:.1f}, 暴露 {exp_class}"
            return True, 0.90, explanation
        explanation["standard"] = "FWI 高但暴露不足以触发预置"
        explanation["detail"] = f"FWI {fwi:.1f}, 暴露 {exp_class}"
        return False, 0.40, explanation

    if cat == "flood":
        water = _latest(obs, "water_level")
        rain24 = _latest(obs, "rainfall_24h")
        if water is None or rain24 is None:
            return False, 0.20, {"category": "flood", "reason": "缺水位/降雨观测"}
        if water >= DISPATCH_FLOOD_WATER and rain24 >= DISPATCH_FLOOD_RAIN24:
            explanation["standard"] = "水位 >= 4.0 且 24h 降雨 >= 30mm → 抢险待命"
            explanation["detail"] = f"水位 {water:.1f}m, 24h 降雨 {rain24:.1f}mm"
            return True, 0.90, explanation
        explanation["standard"] = "水位与降雨未同时达线（AND 缺一）"
        explanation["detail"] = f"水位 {water:.1f}m, 24h 降雨 {rain24:.1f}mm"
        return False, 0.45, explanation

    if cat == "typhoon":
        wind = _latest(obs, "wind_speed")
        if wind is None:
            return False, 0.20, {"category": "typhoon", "reason": "无风速观测"}
        if wind >= DISPATCH_TYPHOON_WIND and exp_class == "E3":
            explanation["standard"] = "风速 >= 24.5（10 级）且 E3（关键设施）→ 防风预置"
            explanation["detail"] = f"风速 {wind:.1f}m/s, 暴露 {exp_class}"
            return True, 0.90, explanation
        explanation["standard"] = "风速或暴露不足以触发预置"
        explanation["detail"] = f"风速 {wind:.1f}m/s, 暴露 {exp_class}"
        return False, 0.40, explanation

    return False, 0.20, {"category": cat, "reason": "未知灾种"}


def infer_upgrade_ground_truth(
    obs: list[dict], exposure: ExposureProfile | None = None
) -> tuple[bool, float, dict[str, Any]]:
    """UPGRADE（是否升档）真值：等级比较 / 趋势外推（披露）。

    - heat 升档：最高温 >= 35（黄档已触发）且最新值 >= 36.5 且时序上升
    - flood 升档：最新水位 >= 4.5 且水位时序上升（逼近警戒）
    - typhoon 升档：风速 >= 17.2（8 级已预警）且阵风 >= 24.5（10 级线）
    """
    cat = _detect_template_category(obs)
    explanation: dict[str, Any] = {"category": cat}

    if cat == "heat":
        temps = [v for _, v in _obs_vals(obs, "temperature_max")]
        if len(temps) < 2:
            return False, 0.20, {"category": "heat", "reason": "温度时序不足"}
        rmax = max(temps)
        latest = temps[-1]
        rising = latest > temps[-2]
        if rmax >= UPGRADE_HEAT_CURRENT and latest >= UPGRADE_HEAT_NEXT and rising:
            explanation["standard"] = "黄档已触发且最新 >= 36.5 且时序上升 → 升档"
            explanation["detail"] = f"最高 {rmax:.1f}℃, 最新 {latest:.1f}℃, 上升 {rising}"
            return True, 0.90, explanation
        explanation["standard"] = "未满足升档三条件（黄档/逼近橙档/上升）"
        explanation["detail"] = f"最高 {rmax:.1f}℃, 最新 {latest:.1f}℃, 上升 {rising}"
        return False, 0.50, explanation

    if cat == "flood":
        levels = _obs_vals(obs, "water_level")
        if len(levels) < 2:
            return False, 0.20, {"category": "flood", "reason": "水位时序不足"}
        latest = levels[-1][1]
        rising = latest > levels[-2][1]
        if latest >= UPGRADE_FLOOD_WATER and rising:
            explanation["standard"] = "最新水位 >= 4.5 且时序上升 → 升档"
            explanation["detail"] = f"最新水位 {latest:.1f}m, 上升 {rising}"
            return True, 0.90, explanation
        explanation["standard"] = "水位未达升档线或未上升"
        explanation["detail"] = f"最新水位 {latest:.1f}m, 上升 {rising}"
        return False, 0.45, explanation

    if cat == "typhoon":
        wind = _latest(obs, "wind_speed")
        gust = _latest(obs, "wind_gust")
        if wind is None or gust is None:
            return False, 0.20, {"category": "typhoon", "reason": "缺风/阵风观测"}
        if wind >= UPGRADE_TYPHOON_WIND and gust >= UPGRADE_TYPHOON_GUST:
            explanation["standard"] = "8 级已预警且阵风 >= 24.5（10 级线）→ 升档"
            explanation["detail"] = f"风速 {wind:.1f}m/s, 阵风 {gust:.1f}m/s"
            return True, 0.90, explanation
        explanation["standard"] = "风与阵风未同时达升档线（AND 缺一）"
        explanation["detail"] = f"风速 {wind:.1f}m/s, 阵风 {gust:.1f}m/s"
        return False, 0.45, explanation

    return False, 0.20, {"category": cat, "reason": "未知灾种"}


def infer_close_ground_truth(
    obs: list[dict], exposure: ExposureProfile | None = None
) -> tuple[bool, float, dict[str, Any]]:
    """CLOSE（是否封闭道路/景区/设施）真值：危险等级 × 暴露（披露）。

    - typhoon + 机场：风速 >= 24.5 且 critical_infrastructure 含 airport → 封闭
    - snow 封路：降雪 >= 10（暴雪）且路温 <= 0 → 封路（无需暴露）
    - flood 封景区：水位 >= 5.0（警戒）且户外活动高（景区游客）→ 封闭
    """
    cat = _detect_template_category(obs)
    infra = list(exposure.critical_infrastructure) if exposure else []
    outdoor = exposure.outdoor_activity_level if exposure else "unknown"
    explanation: dict[str, Any] = {"category": cat}

    if cat == "typhoon":
        wind = _latest(obs, "wind_speed")
        if wind is None:
            return False, 0.20, {"category": "typhoon", "reason": "无风速观测"}
        if wind >= CLOSE_TYPHOON_WIND and "airport" in infra:
            explanation["standard"] = "风速 >= 24.5 且机场在场 → 封闭"
            explanation["detail"] = f"风速 {wind:.1f}m/s, 关键设施含机场"
            return True, 0.90, explanation
        explanation["standard"] = "风速或设施不满足封闭条件"
        explanation["detail"] = f"风速 {wind:.1f}m/s, 关键设施 {infra or '无'}"
        return False, 0.40, explanation

    if cat == "snow":
        snow = _max_val(obs, "snowfall_24h")
        road = _latest(obs, "road_surface_temp")
        if snow is None or road is None:
            return False, 0.20, {"category": "snow", "reason": "缺降雪/路温观测"}
        if snow >= CLOSE_SNOW_SNOW and road <= CLOSE_SNOW_ROAD:
            explanation["standard"] = "暴雪 >= 10mm 且路温 <= 0℃ → 封路"
            explanation["detail"] = f"降雪 {snow:.1f}mm, 路温 {road:.1f}℃"
            return True, 0.90, explanation
        explanation["standard"] = "降雪或路温未达封路条件"
        explanation["detail"] = f"降雪 {snow:.1f}mm, 路温 {road:.1f}℃"
        return False, 0.40, explanation

    if cat == "flood":
        water = _latest(obs, "water_level")
        if water is None:
            return False, 0.20, {"category": "flood", "reason": "无水位观测"}
        if water >= CLOSE_FLOOD_WATER and outdoor == "high":
            explanation["standard"] = "水位 >= 5.0 且户外活动高（景区）→ 封闭景区"
            explanation["detail"] = f"水位 {water:.1f}m, 户外活动 {outdoor}"
            return True, 0.90, explanation
        explanation["standard"] = "水位或户外活动不满足封闭条件"
        explanation["detail"] = f"水位 {water:.1f}m, 户外活动 {outdoor}"
        return False, 0.40, explanation

    return False, 0.20, {"category": cat, "reason": "未知灾种"}


def infer_recover_ground_truth(
    obs: list[dict], exposure: ExposureProfile | None = None
) -> tuple[bool, float, dict[str, Any]]:
    """RECOVER（是否解除/恢复）真值：条件解除后须持续 N 小时（披露）。

    - fire 解除：24h 降雨 >= 30（透雨）且 FWI < 40 持续 6h 无反弹
    - flood 解除：水位 <= 4.5 持续 6h
    - snow 解除：路温 > 0.5℃ 持续 6h（结冰消解）
    核心：任一时刻回弹即不算「持续解除」——过早恢复（「雨停就解除」）判 False。
    """
    cat = _detect_template_category(obs)
    explanation: dict[str, Any] = {"category": cat}

    if cat == "fire":
        rain24 = _latest(obs, "rainfall_24h")
        ok_rain = rain24 is not None and rain24 >= RECOVER_FIRE_RAIN24
        persist_ok, p_detail = _persist_for(
            obs, "FWI", lambda v: v < RECOVER_FIRE_FWI, RECOVER_FIRE_SUSTAIN_H
        )
        explanation["standard"] = "24h 降雨 >= 30 且 FWI < 40 持续 6h → 解除"
        explanation["detail"] = (
            f"透雨 {rain24 if rain24 is not None else 'N/A'}mm, "
            f"火险持续 {p_detail.get('points', 0)} 点全 < 40: {persist_ok}"
        )
        if ok_rain and persist_ok:
            return True, 0.90, explanation
        return False, 0.40, explanation

    if cat == "flood":
        persist_ok, p_detail = _persist_for(
            obs, "water_level", lambda v: v <= RECOVER_FLOOD_WATER, RECOVER_FLOOD_SUSTAIN_H
        )
        explanation["standard"] = f"水位 <= {RECOVER_FLOOD_WATER}m 持续 {RECOVER_FLOOD_SUSTAIN_H}h → 解除"
        explanation["detail"] = (
            f"窗口 {p_detail.get('points', 0)} 点全 <= {RECOVER_FLOOD_WATER}m: {persist_ok}"
        )
        if persist_ok:
            return True, 0.90, explanation
        return False, 0.40, explanation

    if cat == "snow":
        persist_ok, p_detail = _persist_for(
            obs, "road_surface_temp", lambda v: v > RECOVER_SNOW_ROAD, RECOVER_SNOW_SUSTAIN_H
        )
        explanation["standard"] = f"路温 > {RECOVER_SNOW_ROAD}℃ 持续 {RECOVER_SNOW_SUSTAIN_H}h → 解除结冰"
        explanation["detail"] = (
            f"窗口 {p_detail.get('points', 0)} 点全 > {RECOVER_SNOW_ROAD}℃: {persist_ok}"
        )
        if persist_ok:
            return True, 0.90, explanation
        return False, 0.40, explanation

    return False, 0.20, {"category": cat, "reason": "未知灾种"}


# ===========================================================================
# 模板闭环套件（Phase C）
# ===========================================================================


def _template_case(
    case_id: str,
    difficulty: str,
    category: str,
    region: str,
    observations: list[dict],
    ground_truth: bool,
    gt_fn: Any,
    template: str,
    exposure: dict | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "category": category,
        "region": region,
        "observations": observations,
        "ground_truth": ground_truth,
        "_gt_fn": gt_fn,
        "template": template,
        "exposure": exposure,
    }


def get_dispatch_suite() -> list[dict[str, Any]]:
    """DISPATCH（资源预置）套件 —— 4 用例，含误报/漏报陷阱各一。"""
    ts = "2026-06-15T12:00:00+08:00"
    # fire 消防预置：FWI 45 高危险
    fire_obs = [
        _make_obs("ECMWF", "FWI", 45.0, "", ts),
        _make_obs("Station", "humidity", 25.0, "%", ts),
        _make_obs("Station", "wind_speed", 8.0, "m/s", ts),
        _make_obs("Station", "temperature", 30.0, "°C", ts),
    ]
    # flood 抢险待命
    flood_standby = [
        _make_obs("Hydro", "water_level", 4.2, "m", ts),
        _make_obs("CMA", "rainfall_24h", 35.0, "mm", ts),
        _make_obs("CMA", "rainfall_6h", 20.0, "mm", ts),
        _make_obs("Sensor", "soil_moisture", 0.65, "", ts),
    ]
    flood_water_only = [
        _make_obs("Hydro", "water_level", 4.2, "m", ts),
        _make_obs("CMA", "rainfall_24h", 10.0, "mm", ts),
        _make_obs("CMA", "rainfall_6h", 5.0, "mm", ts),
        _make_obs("Sensor", "soil_moisture", 0.65, "", ts),
    ]
    return [
        _template_case(
            "disp-fire-tourist-holiday", "L3", "fire", "Xiangshan-Beijing",
            fire_obs, True, infer_dispatch_ground_truth, "dispatch",
            {"population_density_class": "mid", "land_use": "rural",
             "critical_infrastructure": ["scenic_area"],
             "outdoor_activity_level": "high"},
        ),
        # 误报陷阱：FWI 同样 45 但无人林区 → 真值不预置，baseline 因线性置信度 0.62 仍预置
        _template_case(
            "disp-fire-rural-low", "L3", "fire", "GreaterKhingan",
            fire_obs, False, infer_dispatch_ground_truth, "dispatch",
            {"population_density_class": "none", "land_use": "forest"},
        ),
        # 漏报陷阱：水位+降雨双达线 → 真值待命，baseline 线性分 0.35 不足
        _template_case(
            "disp-flood-standby", "L3", "flood", "Wuhan-Yangtze",
            flood_standby, True, infer_dispatch_ground_truth, "dispatch",
        ),
        _template_case(
            "disp-flood-water-only", "L3", "flood", "Wuhan-Yangtze",
            flood_water_only, False, infer_dispatch_ground_truth, "dispatch",
        ),
    ]


def get_upgrade_suite() -> list[dict[str, Any]]:
    """UPGRADE（升档）套件 —— 4 用例，含误报/漏报陷阱各一。"""
    typhoon_up = [
        _make_obs("Station", "wind_speed", 18.0, "m/s", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "wind_gust", 26.0, "m/s", "2026-07-11T12:00:00+08:00"),
        _make_obs("CMA", "rainfall_24h", 10.0, "mm", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "humidity", 80.0, "%", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "pressure", 985.0, "hPa", "2026-07-11T12:00:00+08:00"),
    ]
    typhoon_stable8 = [
        _make_obs("Station", "wind_speed", 18.0, "m/s", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "wind_gust", 20.0, "m/s", "2026-07-11T12:00:00+08:00"),
        _make_obs("CMA", "rainfall_24h", 10.0, "mm", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "humidity", 80.0, "%", "2026-07-11T12:00:00+08:00"),
        _make_obs("Station", "pressure", 990.0, "hPa", "2026-07-11T12:00:00+08:00"),
    ]
    heat_approach = [
        _make_obs("CMA", "temperature_max", 35.0, "°C", "2026-07-24T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 36.0, "°C", "2026-07-25T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 37.0, "°C", "2026-07-26T12:00:00+08:00"),
        _make_obs("Station", "humidity", 60.0, "%", "2026-07-26T12:00:00+08:00"),
        _make_obs("Station", "wet_bulb_temp", 26.0, "°C", "2026-07-26T12:00:00+08:00"),
    ]
    heat_stable = [
        _make_obs("CMA", "temperature_max", 35.0, "°C", "2026-07-24T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 34.5, "°C", "2026-07-25T12:00:00+08:00"),
        _make_obs("CMA", "temperature_max", 34.0, "°C", "2026-07-26T12:00:00+08:00"),
        _make_obs("Station", "humidity", 60.0, "%", "2026-07-26T12:00:00+08:00"),
        _make_obs("Station", "wet_bulb_temp", 26.0, "°C", "2026-07-26T12:00:00+08:00"),
    ]
    return [
        _template_case(
            "upg-typhoon-gust-upgrade", "L3", "typhoon", "Wenzhou-Zhejiang",
            typhoon_up, True, infer_upgrade_ground_truth, "upgrade",
        ),
        # 误报陷阱：8 级 + 阵风 20（未达 10 级线）→ 真值不升档，baseline 线性 0.64 过 0.60 仍升
        _template_case(
            "upg-typhoon-stable-8", "L3", "typhoon", "Wenzhou-Zhejiang",
            typhoon_stable8, False, infer_upgrade_ground_truth, "upgrade",
        ),
        # 漏报陷阱：高温三连升逼近橙档 → 真值升档，baseline 线性 0.51 不足 0.60
        _template_case(
            "upg-heat-approach", "L3", "heat", "Chongqing-HotPotato",
            heat_approach, True, infer_upgrade_ground_truth, "upgrade",
        ),
        _template_case(
            "upg-heat-stable", "L3", "heat", "Chongqing-HotPotato",
            heat_stable, False, infer_upgrade_ground_truth, "upgrade",
        ),
    ]


def get_close_suite() -> list[dict[str, Any]]:
    """CLOSE（封闭）套件 —— 5 用例，含误报/漏报陷阱各一。"""
    ts = "2026-08-10T12:00:00+08:00"
    typhoon_close = [
        _make_obs("Station", "wind_speed", 26.0, "m/s", ts),
        _make_obs("Station", "wind_gust", 28.0, "m/s", ts),
        _make_obs("CMA", "rainfall_24h", 30.0, "mm", ts),
    ]
    snow_road = [
        _make_obs("CMA", "snowfall_24h", 12.0, "mm", ts),
        _make_obs("Sensor", "road_surface_temp", -3.0, "°C", ts),
        _make_obs("Station", "humidity", 80.0, "%", ts),
    ]
    snow_wet = [
        _make_obs("CMA", "snowfall_24h", 12.0, "mm", ts),
        _make_obs("Sensor", "road_surface_temp", 2.0, "°C", ts),
        _make_obs("Station", "humidity", 80.0, "%", ts),
    ]
    snow_mild = [
        _make_obs("CMA", "snowfall_24h", 10.5, "mm", ts),
        _make_obs("Sensor", "road_surface_temp", -0.5, "°C", ts),
        _make_obs("Station", "humidity", 55.0, "%", ts),
    ]
    return [
        _template_case(
            "close-typhoon-airport", "L3", "typhoon", "Xiamen-Fujian",
            typhoon_close, True, infer_close_ground_truth, "close",
            {"population_density_class": "high", "land_use": "urban",
             "critical_infrastructure": ["airport"]},
        ),
        # 误报陷阱：同样 10 级风但开阔海面无机场 → 真值不封闭，baseline 线性 0.72 过 0.70 仍封
        _template_case(
            "close-typhoon-open-sea", "L3", "typhoon", "Wenzhou-Zhejiang",
            typhoon_close, False, infer_close_ground_truth, "close",
            {"population_density_class": "none", "land_use": "rural"},
        ),
        _template_case(
            "close-snow-road", "L3", "snow", "Harbin-Heilongjiang",
            snow_road, True, infer_close_ground_truth, "close",
        ),
        _template_case(
            "close-snow-wet-road", "L3", "snow", "Chengdu-Panda",
            snow_wet, False, infer_close_ground_truth, "close",
        ),
        # 漏报陷阱：暴雪 + 路温 -0.5 刚过冰点 → 真值封路，baseline 线性 0.54 不足 0.70 不封
        _template_case(
            "close-snow-mild-just-met", "L4", "snow", "Chongqing-HotPotato",
            snow_mild, True, infer_close_ground_truth, "close",
        ),
    ]


def get_recover_suite() -> list[dict[str, Any]]:
    """RECOVER（解除/恢复）套件 —— 4 用例，含过早恢复陷阱 + 过度保守陷阱。"""
    now = "2026-06-20T12:00:00+08:00"
    flood_sustained = [
        _make_obs("Hydro", "water_level", 4.4, "m", "2026-06-20T06:00:00+08:00"),
        _make_obs("Hydro", "water_level", 4.3, "m", "2026-06-20T09:00:00+08:00"),
        _make_obs("Hydro", "water_level", 4.2, "m", now),
        _make_obs("CMA", "rainfall_24h", 5.0, "mm", now),
    ]
    flood_premature = [
        _make_obs("Hydro", "water_level", 4.7, "m", "2026-06-20T06:00:00+08:00"),
        _make_obs("Hydro", "water_level", 4.4, "m", "2026-06-20T09:00:00+08:00"),
        _make_obs("Hydro", "water_level", 4.3, "m", now),
        _make_obs("CMA", "rainfall_24h", 5.0, "mm", now),
    ]
    flood_still_high = [
        _make_obs("Hydro", "water_level", 5.0, "m", "2026-06-20T06:00:00+08:00"),
        _make_obs("Hydro", "water_level", 5.1, "m", "2026-06-20T09:00:00+08:00"),
        _make_obs("Hydro", "water_level", 5.2, "m", now),
        _make_obs("CMA", "rainfall_24h", 30.0, "mm", now),
        _make_obs("CMA", "rainfall_6h", 30.0, "mm", now),
        _make_obs("Sensor", "soil_moisture", 0.9, "", now),
    ]
    snow_thawed = [
        _make_obs("Sensor", "road_surface_temp", 0.6, "°C", "2026-06-20T06:00:00+08:00"),
        _make_obs("Sensor", "road_surface_temp", 0.8, "°C", "2026-06-20T09:00:00+08:00"),
        _make_obs("Sensor", "road_surface_temp", 1.0, "°C", now),
        _make_obs("CMA", "snowfall_24h", 8.0, "mm", now),
        _make_obs("Station", "humidity", 70.0, "%", now),
    ]
    return [
        _template_case(
            "rec-flood-sustained", "L3", "flood", "Wuhan-Yangtze",
            flood_sustained, True, infer_recover_ground_truth, "recover",
        ),
        # 过早恢复陷阱（应急管理最常见人命代价）：水位 6h 前仍 4.7 > 4.5，
        # 真值不解除，baseline「雨停即恢复」误解除
        _template_case(
            "rec-flood-premature", "L4", "flood", "Wuhan-Yangtze",
            flood_premature, False, infer_recover_ground_truth, "recover",
        ),
        # 过度保守陷阱：路温已持续 6h 冰点以上 → 真值解除，baseline 因
        # 残留降雪量分量仍预警、不解除
        _template_case(
            "rec-snow-thawed", "L4", "snow", "Harbin-Heilongjiang",
            snow_thawed, True, infer_recover_ground_truth, "recover",
        ),
        _template_case(
            "rec-flood-still-high", "L3", "flood", "Nanjing-Yangtze",
            flood_still_high, False, infer_recover_ground_truth, "recover",
        ),
    ]


def get_historical_validation_suite() -> list[dict[str, Any]]:
    """历史灾例回填验证套件 —— 8 用例（7 真实灾害 + 1 合成对照）。

    验证目标：把**公开报道的真实观测值**灌进独立标准真值函数，检验推导
    判定与**实际发生了什么**（官方响应级别/实际灾情/站档记录）是否一致。
    这是对「真值函数只引用外部标准、不自我认证」纪律的外部锚定检验：
    若 `_gt_fn` 在真实灾例上判定为 False（漏报）或对照日误报，
    `AlertBenchEvaluator.gt_divergences` 会立刻暴露。

    数据来源与口径（每条观测值都可溯源，provenance 附 URL）：
    - 郑州 7·20（2021-07-20）：最大 24h 降雨 645.6mm（20日04时–21日04时，
      破 1951 年建站纪录）、最大小时降雨 201.9mm（16–17时，中国大陆小时
      雨强纪录）。实际：郑州启动防汛应急Ⅰ级响应，因灾死亡失踪 398 人。
    - 北京 23·7（2023-07-29~08-01，海河流域性特大洪水）：最大小时雨强
      111.8mm/h（丰台千灵山，31日10–11时）——以其作为任意 6h 窗口的
      严格下界灌入 rainfall_6h 通道（公开报道无干净的峰值 24h 站点值，
      过程极值昌平王家园水库 744.8mm 为 ~83h 窗口，不宜冒充 24h）。
      实际：中央气象台暴雨红色预警 + 北京防汛红色（Ⅰ级）预警响应。
    - 台风杜苏芮（2023-07-28 09:55 登陆福建晋江）：登陆时强台风级
      中心附近最大风速 50 m/s（持续风口径；63 年来首登福建最强台风）。
      口径披露：本真值档位为日均风口径，50 ≫ 32.7（12 级线），口径差
      不改变结论。实际：福建 266.69 万人受灾，直接经济损失 147.55 亿元。
    - 通辽寒潮（2021-11-05→06，锋面过境日）：站档日最低温 4.6 → -6.2°C，
      24h 降幅 10.8°C（远超寒潮 24h 窗口 8℃ 线，GB/T 20484-2017 顶档）。
      站档来源 q-weather 54135 存档（与郑州逐小时序列同类一手站档数据）。
    - 通辽特大暴雪（2021-11-08，主雪日）：站档逐小时 1 小时降水求和
      = 44.6mm 水当量/24h（气温全程 -6.5~-3.3°C，固态无疑义），
      ≥ 30mm 特大暴雪档；积雪深度站档 27cm（08日08时）→ 59cm
      （09日08时）为现实锚定。
    - 川渝极端高温峰值日（2022-08-19）：站档 57516 最高 43.1°C
      （≥ 40°C 红色档）；现实锚定为北碚 45.0°C 破纪录 + 中央气象台
      高温红色预警 + 川渝限电（43.1 已达档，站点差异不敏感）。
    - 重庆主城山火火险日（2022-08-21）：FWI=70.5 由仓库披露的简化
      估算器从 14 时站档快照计算（40.9°C / RH 29% / 5.4 m/s / 零降水），
      对快照时次不敏感（15 时口径仍 >33 极高档）；现实锚定为卫星
      影像确认的多点山火（涪陵/江津/巴南/璧山）。
    - 对照日（合成，非历史事件）：验证 False 通路不误报，披露为合成。

    本轮回填的口径发现（见 HISTORICAL_VALIDATION_FINDINGS）：寒潮
    口径已转化为修复（2006 三档误用 → 2017 四级体系顶档多窗口 OR）；
    滑坡（激发雨强）与干旱（SPI/Palmer 指数）公开溯源断链，按
    「宁缺毋滥」不构造用例——干旱的「偏少 90%」相对口径与官方 MCI
    特旱定性均无法严格标准化为 SPI（需站点气候基线 σ）。

    注意：历史套件不进 83 用例主基准（难度分层与对抗判别力设计针对
    合成场景）；它是**验证报告**（scripts/validate_historical.py →
    docs/historical-validation.md），规则 baseline 得分仅供参考。
    """
    return [
        {
            "case_id": "hist-flood-zhengzhou-720",
            "difficulty": "H1",
            "category": "flood",
            "region": "Zhengzhou-Henan",
            "ground_truth": True,  # 实际：防汛Ⅰ级响应 + 398 死亡失踪
            "observations": [
                _make_obs("CMA-57083", "rainfall_24h", 645.6, "mm",
                          "2021-07-21T04:00:00+08:00"),
                _make_obs("CMA-57083", "rainfall_1h", 201.9, "mm",
                          "2021-07-20T17:00:00+08:00"),
                # 逐小时序列（57083 站存档，维基条目图表）：13-14时、15-16时
                _make_obs("CMA-57083", "rainfall_1h", 30.6, "mm",
                          "2021-07-20T14:00:00+08:00"),
                _make_obs("CMA-57083", "rainfall_1h", 60.6, "mm",
                          "2021-07-20T16:00:00+08:00"),
            ],
            "_gt_fn": infer_flood_ground_truth,
            "provenance": {
                "event": "郑州 7·20 特大暴雨（2021-07-20）",
                "documented_reality": (
                    "郑州启动防汛应急Ⅰ级响应；因灾死亡失踪 398 人，"
                    "直接经济损失 1200.6 亿元"
                ),
                "obs_notes": (
                    "24h=645.6mm（20日04时–21日04时，最大24h，破1951年建站"
                    "纪录）；逐小时序列 201.9/60.6/30.6mm（16-17/15-16/13-14时，"
                    "57083 站存档）；真值函数读 24h 通道，小时值仅作文档记录"
                ),
                "sources": [
                    "https://zh.wikipedia.org/wiki/2021年7月河南水灾",
                    "http://www.xinhuanet.com/2021-07/20/c_1127676145.htm",
                    "https://new.qq.com/omn/20210721/20210721A0ESZA00.html",
                    "https://q-weather.info/weather/57083/history/?date=2021-07-20",
                ],
            },
        },
        {
            "case_id": "hist-flood-beijing-237",
            "difficulty": "H1",
            "category": "flood",
            "region": "Fangshan-Beijing",
            "ground_truth": True,  # 实际：暴雨红色预警 + 防汛红色Ⅰ级响应
            "observations": [
                # 严格下界：任意包含该小时的 6h 窗口 ≥ 111.8mm
                _make_obs("CMA-Beijing", "rainfall_6h", 111.8, "mm",
                          "2023-07-31T11:00:00+08:00"),
                _make_obs("CMA-Beijing", "rainfall_1h", 111.8, "mm",
                          "2023-07-31T11:00:00+08:00"),
            ],
            "_gt_fn": infer_flood_ground_truth,
            "provenance": {
                "event": "海河 23·7 流域性特大洪水（2023-07-29~08-01）",
                "documented_reality": (
                    "中央气象台暴雨红色预警（2010年预警机制以来第2个）；"
                    "北京市升级暴雨红色预警并启动防汛红色（Ⅰ级）预警响应；"
                    "62 人死亡、34 人失踪"
                ),
                "obs_notes": (
                    "最大小时雨强 111.8mm/h（丰台千灵山，31日10–11时），作为"
                    "任意 6h 窗口的严格下界灌入 rainfall_6h 通道（多通道设计"
                    "的意义：无干净 24h 站点值时短时强降雨通道仍可判定）。"
                    "过程极值：昌平王家园水库 744.8mm（~83h）、房山新村"
                    "500.4mm（36h）、邢台临城 1003.0mm"
                ),
                "sources": [
                    "https://zh.wikipedia.org/wiki/2023年京津冀暴雨灾害",
                ],
            },
        },
        {
            "case_id": "hist-typhoon-doksuri-2023",
            "difficulty": "H1",
            "category": "typhoon",
            "region": "Jinjiang-Fujian",
            "ground_truth": True,  # 实际：福建 266.69 万人受灾
            "observations": [
                _make_obs("CMA", "wind_speed", 50.0, "m/s",
                          "2023-07-28T09:55:00+08:00"),
            ],
            "_gt_fn": infer_typhoon_ground_truth,
            "provenance": {
                "event": "台风杜苏芮登陆福建晋江（2023-07-28 09:55）",
                "documented_reality": (
                    "强台风级（50 m/s）登陆，63 年来首登福建最强台风；"
                    "福建 266.69 万人受灾，直接经济损失 147.55 亿元"
                ),
                "obs_notes": (
                    "登陆时中心附近最大风速 50 m/s（持续风口径）。口径披露："
                    "真值档位名义为日均风口径，50 ≫ 32.7（12 级台风线），"
                    "口径差不改变判定结论"
                ),
                "sources": [
                    "https://zh.wikipedia.org/wiki/2023年京津冀暴雨灾害",
                    "https://zh.wikipedia.org/wiki/颱風杜蘇芮_(2023年)",
                ],
            },
        },
        {
            "case_id": "hist-cold-tongliao-2021",
            "difficulty": "H1",
            "category": "cold",
            "region": "Tongliao-InnerMongolia",
            "ground_truth": True,  # 实际：寒潮+暴风雪复合过程
            "observations": [
                # 站档日最低温（q-weather 54135 逐小时存档的最小值）
                _make_obs("CMA-54135", "temperature_min", 4.6, "°C",
                          "2021-11-05T22:00:00+08:00"),
                _make_obs("CMA-54135", "temperature_min", -6.2, "°C",
                          "2021-11-06T21:00:00+08:00"),
                # 锋前暖区午后极值（05日 15:00），文档化锋面对比
                _make_obs("CMA-54135", "temperature", 16.9, "°C",
                          "2021-11-05T15:00:00+08:00"),
            ],
            "_gt_fn": infer_cold_ground_truth,
            "provenance": {
                "event": "通辽寒潮锋面过境（2021-11-05→06，随后特大暴雪）",
                "documented_reality": (
                    "站档 24h 日最低温降 10.8°C（4.6→-6.2°C）；锋前 05 日"
                    "午后 16.9°C，06 日起转雪并持续 40+ 小时，积雪深度"
                    "27cm（08日08时）→ 59cm（09日08时），寒潮+暴风雪复合"
                ),
                "obs_notes": (
                    "日最低温由 q-weather 54135 逐小时站档求当日最小值"
                    "（05日 tmin=4.6@22时、06日 tmin=-6.2@21时）；降幅"
                    "10.8°C 由真值函数从连续两日 tmin 序列推导"
                    "（drop_source=derived）"
                ),
                "sources": [
                    "https://q-weather.info/weather/54135/history/?date=2021-11-05",
                    "https://q-weather.info/weather/54135/history/?date=2021-11-06",
                    "https://q-weather.info/weather/54135/history/?date=2021-11-09",
                ],
            },
        },
        {
            "case_id": "hist-snow-tongliao-2021",
            "difficulty": "H1",
            "category": "snow",
            "region": "Tongliao-InnerMongolia",
            "ground_truth": True,  # 实际：特大暴雪，积雪 59cm
            "observations": [
                # 主雪日 24h 水当量 = 站档逐小时 1 小时降水求和（01-23 时）
                _make_obs("CMA-54135", "snowfall_24h", 44.6, "mm",
                          "2021-11-08T23:00:00+08:00"),
                _make_obs("CMA-54135", "temperature", -3.3, "°C",
                          "2021-11-08T23:00:00+08:00"),
                _make_obs("CMA-54135", "humidity", 92.0, "%",
                          "2021-11-08T23:00:00+08:00"),
            ],
            "_gt_fn": infer_snow_ground_truth,
            "provenance": {
                "event": "通辽特大暴雪主雪日（2021-11-08）",
                "documented_reality": (
                    "积雪深度站档 27cm（08日08时）→ 59cm（09日08时）；"
                    "降雪期间能见度低至 0.4-0.7km、1 小时极大风速 "
                    "16-17 m/s（暴风雪），特大暴雪成灾"
                ),
                "obs_notes": (
                    "24h 降雪水当量 44.6mm = 站档 08日 01-23 时逐小时"
                    "「1小时降水」求和（降水量为水当量口径）；当日气温"
                    "全程 -6.5~-3.3°C（远低于冻结线 0.5°C，固态无疑义，"
                    "积雪深度列直接佐证）"
                ),
                "sources": [
                    "https://q-weather.info/weather/54135/history/?date=2021-11-08",
                    "https://q-weather.info/weather/54135/history/?date=2021-11-09",
                ],
            },
        },
        {
            "case_id": "hist-heat-chongqing-2022",
            "difficulty": "H1",
            "category": "heat",
            "region": "Chongqing-HotPotato",
            "ground_truth": True,  # 实际：1961 年以来最强区域性高温事件
            "observations": [
                # 站档 57516（重庆沙坪坝）8月19日逐小时最高值（17 时）
                _make_obs("CMA-57516", "temperature_max", 43.1, "°C",
                          "2022-08-19T17:00:00+08:00"),
                _make_obs("CMA-57516", "temperature", 43.1, "°C",
                          "2022-08-19T17:00:00+08:00"),
                _make_obs("CMA-57516", "humidity", 23.0, "%",
                          "2022-08-19T17:00:00+08:00"),
                # 条目记载：沙坪坝气温连续 ≥30°C（8月6日-29日）
                _make_obs("CMA-57516", "heat_duration_days", 24.0, "days",
                          "2022-08-19T17:00:00+08:00"),
            ],
            "_gt_fn": infer_heatwave_ground_truth,
            "provenance": {
                "event": "川渝极端高温峰值期（2022-08-19，长江流域复合高温干旱）",
                "documented_reality": (
                    "重庆北碚 8月17日 44.6°C、18日 45.0°C 破重庆气象纪录；"
                    "中央气象台 8月12日发布当年首个高温红色预警；区域性"
                    "高温事件持续 79 天、覆盖超 500 万平方公里、366 个国家"
                    "站破历史极值，综合强度 1961 年以来最强；川渝大规模"
                    "限电（四川工业全停 6+5 天）、多地热射病死亡"
                ),
                "obs_notes": (
                    "站档 57516 8月19日逐小时：最高 43.1°C@17 时、最低"
                    "湿度 23%、全天零降水；北碚站纪录 45.0°C 不在 q-weather"
                    " 存档内，以沙坪坝站档为准（43.1 ≥ 40 已达红色档，"
                    "站点差异不影响档位判定）；heat_duration_days=24 取"
                    "条目记载的沙坪坝连续 ≥30°C 日数"
                ),
                "sources": [
                    "https://zh.wikipedia.org/wiki/2022年中国高温",
                    "https://q-weather.info/weather/57516/history/?date=2022-08-19",
                ],
            },
        },
        {
            "case_id": "hist-fire-chongqing-2022",
            "difficulty": "H1",
            "category": "fire",
            "region": "Chongqing-HotPotato",
            "ground_truth": True,  # 实际：伏秋连旱下主城多点山火
            "observations": [
                # FWI 由仓库披露的简化估算器从 14 时站档快照计算（见 obs_notes）
                _make_obs("FWI-calc/CMA-57516", "FWI", 70.5, "index",
                          "2022-08-21T14:00:00+08:00"),
                _make_obs("CMA-57516", "humidity", 29.0, "%",
                          "2022-08-21T14:00:00+08:00"),
                _make_obs("CMA-57516", "wind_speed", 5.4, "m/s",
                          "2022-08-21T14:00:00+08:00"),
                _make_obs("CMA-57516", "temperature", 40.9, "°C",
                          "2022-08-21T14:00:00+08:00"),
            ],
            "_gt_fn": infer_fire_ground_truth,
            "provenance": {
                "event": "重庆主城多点山火（2022年8月中下旬，伏秋连旱极端火险）",
                "documented_reality": (
                    "干旱天气在重庆主城区多点引发山火（涪陵、江津、巴南、"
                    "璧山，吉林一号卫星影像确认起火点），部分火场地势陡峭"
                    "需志愿者摩托骑士运送物资；同期鄱阳湖/洞庭湖面积严重"
                    "缩小、长江流域直接干旱损失超 500 亿元"
                ),
                "obs_notes": (
                    "FWI=70.5 由本仓库披露的简化 FWI 估算器"
                    "（calculate_fwi_from_weather，生产管线同源）从站档"
                    " 8月21日 14 时快照计算（40.9°C / RH 29% / 风速 "
                    "5.4 m/s / 零降水，无区域水体修正）；对快照时次不"
                    "敏感（15 时口径 FWI≈37.6 仍 > 33 极高档）；8月19/21"
                    "两日站档全天降水 0.0mm 印证连日干苞。8月21日为巴南"
                    "山火主要时段（站档记录的代表性极端火险日）"
                ),
                "sources": [
                    "https://zh.wikipedia.org/wiki/2022年中国高温",
                    "https://zh.wikipedia.org/wiki/2022年长江流域干旱",
                    "https://q-weather.info/weather/57516/history/?date=2022-08-21",
                ],
            },
        },
        {
            "case_id": "hist-control-synthetic-benign",
            "difficulty": "H2",
            "category": "flood",
            "region": "Wuhan-Yangtze",
            "ground_truth": False,  # 对照：无雨无汛，不应预警
            "observations": [
                _make_obs("CMA", "rainfall_24h", 2.0, "mm",
                          "2021-07-15T12:00:00+08:00"),
                _make_obs("Sensor", "soil_moisture", 0.35, "",
                          "2021-07-15T12:00:00+08:00"),
                _make_obs("Hydro", "water_level", 2.5, "m",
                          "2021-07-15T12:00:00+08:00"),
            ],
            "_gt_fn": infer_flood_ground_truth,
            "provenance": {
                "event": "合成对照日（非历史事件，披露）",
                "documented_reality": "无降雨、土壤干燥、水位正常 → 不应预警",
                "obs_notes": (
                    "验证 False 通路：真实灾例均为 True（灾难发生了），"
                    "误报方向由合成对照覆盖——三通道全部远离阈值"
                ),
                "sources": ["synthetic（合成对照，非历史数据）"],
            },
        },
    ]


# 历史回填过程中发现的口径问题（单一来源：验证报告渲染 + 文档引用）。
# 这些是「发现」不是「缺陷」：不改真值代码，列为开放问题供后续决策。
HISTORICAL_VALIDATION_FINDINGS: list[dict[str, str]] = [
    {
        "id": "cold-wave-24h-vs-local-standard",
        "title": "寒潮国标口径 vs 地方预警口径的系统性差异（已按 2017 版修齐多窗口）",
        "detail": (
            "发现与修复：2016年1月世纪寒潮抽查（站档一手数据）显示广州"
            " 59287 站 22/23/24日日最低温 7.1/3.8/1.3°C（相邻日降幅仅 "
            "3.3/2.5°C，48h 降幅 5.8°C）、上海 58367 站 22→23日 2.1→"
            "-4.2°C（24h 降幅 6.3°C，48h 约 9.3°C）——即使按 GB/T "
            "20484-2017 完整多窗口口径（24h≥8 或 48h≥10 或 72h≥12，且"
            "日最低≤4°C）均不达国标线，但实际广州发布红色寒冷预警（地方"
            "标准：绝对低温触发，无降幅要求）、上海发布蓝色寒潮预警（48h "
            "口径）。本轮已把真值从 2006 版三档结构（寒潮/强寒潮/特强"
            "寒潮，误用）修齐为 2017 版四级体系顶档+多窗口 OR。修齐后的"
            "残余差异是真实的国标 vs 地方标准口径分歧：国标要求降幅×极值"
            "双条件，南方地方寒冷预警按绝对低温触发——两者服务对象不同"
            "（寒潮=过程强度分级，地方预警=民生影响），属披露而非缺陷。"
            "结构事实已对照维基「寒潮·中华人民共和国标准」小节核验，多"
            "窗口数值为披露口径（原文 PDF 见来源）。"
        ),
        "sources": [
            "https://zh.wikipedia.org/wiki/寒潮",
            "https://www.cma.gov.cn/zfxxgk/gknr/flfgbz/bz/202209/P020220921579662258389.pdf",
            "https://q-weather.info/weather/59287/history/?date=2016-01-23",
            "https://q-weather.info/weather/59287/history/?date=2016-01-24",
            "https://q-weather.info/weather/58367/history/?date=2016-01-23",
        ],
    },
    {
        "id": "landslide-no-public-rain-intensity",
        "title": "滑坡灾种历史回填受阻：激发雨强定量值公开不可溯源",
        "detail": (
            "2010年舟曲泥石流（1557人死亡，白龙江河谷为中国滑坡泥石流最"
            "发育地区之一，1879 年地震崩积物+2008 汶川震裂山体，成灾机制"
            "与真值函数设计完全吻合）——但中英文维基条目均未给出激发雨强"
            "定量值（仅「暴雨持续约40分钟」定性描述），舟曲国家站存档亦"
            "不可得（q-weather 无该站 2010 年数据；且沟内东山雨量站与县"
            "站差异巨大）。滑坡真值的激发雨强通道（1h≥25mm）无法用公开"
            "数据回填。后续需从文献取数（如 Ren et al. 2014, JGR: "
            "Atmospheres, doi:10.1002/2013JD020881）。本轮按「宁缺毋滥」"
            "纪律不构造该用例。"
        ),
        "sources": [
            "https://zh.wikipedia.org/wiki/2010年舟曲泥石流灾害",
            "https://en.wikipedia.org/wiki/2010_Gansu_mudslide",
        ],
    },
    {
        "id": "drought-index-not-publicly-archived",
        "title": "干旱灾种历史回填受阻：SPI/Palmer 指数公开断链",
        "detail": (
            "2022年长江流域夏秋冬连旱（5000 余万人次受灾、直接损失超 "
            "500 亿元）是理想的回填对象，但干旱真值需要 SPI/Palmer 指数"
            "类观测：站档存档不含指数值，公开报道只有「偏少 90%」相对"
            "口径与官方「特重气象干旱」定性（MCI 口径，非 SPI）——前者"
            "无法严格标准化（SPI=Φ⁻¹(F(x)) 需站点 30 年气候基线的 σ，"
            "无源不造）；后者口径不同不可直接映射。若未来出现可溯源的"
            "站点纪录表述（如「8 月降水为 1951 年以来最少」），可用秩"
            "上界严格推导：纪录事件 ⇒ 经验 F ≤ 1/n ⇒ SPI ≤ Φ⁻¹(1/n)"
            "（n=61 年 → SPI ≤ -2.42，达特旱档）。按「宁缺毋滥」纪律"
            "本轮不构造干旱用例；与滑坡激发雨强同属「指数/强度类输入"
            "公开断链」这一类发现。另注：干旱真值现为指数通道 × 影响"
            "门控双通道（持续≥60天/面积≥40%/城市缺水≥10%，见 "
            "infer_drought_ground_truth），历史回填还需同时溯源影响侧"
            "记录（如流域受旱面积、城市供水通告），断链面更宽。"
        ),
        "sources": [
            "https://zh.wikipedia.org/wiki/2022年长江流域干旱",
            "https://zh.wikipedia.org/wiki/2022年中国高温",
        ],
    },
]
