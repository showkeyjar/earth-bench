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
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
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
# Ground Truth 推导规则 — 国标阈值
# ===========================================================================

# === Wildfire ===
FWI_HIGH_THRESHOLD = 40.0
WIND_HIGH_THRESHOLD = 12.0
HUMIDITY_LOW_THRESHOLD = 20.0
RAINFALL_SUPPRESS = 10.0

# === Flood ===
RAINFALL_FLOOD_24H = 50.0
RAINFALL_FLOOD_6H = 30.0
SOIL_MOISTURE_SATURATED = 0.85

# === Drought ===
SPI_DROUGHT = -1.0
PALMER_DROUGHT = -0.5
HUMIDITY_DROUGHT = 25.0
RAINFALL_DEFICIT = 10.0
NDVI_DEGRADE = 0.3

# === HeatWave ===
TEMP_HEAT_WAVE = 35.0
WET_BULB_TEMP = 27.0
HEAT_DURATION_DAYS = 3


def infer_fire_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """森林火险 Ground Truth 推导（基于国家林草局标准）。

    当某观测变量缺失时，该变量不参与评分，其权重分配给其他变量。
    """
    explanation: dict[str, Any] = {}
    components: list[tuple[float, float, str]] = []  # (risk_score, weight, label)

    fwi_vals = [o["value"] for o in obs if o["variable"] == "FWI"]
    if fwi_vals:
        avg_fwi = sum(fwi_vals) / len(fwi_vals)
        fwi_norm = min(avg_fwi / 60.0, 1.0)
        components.append((fwi_norm, 0.40, f"fwi_mean={round(avg_fwi, 1)}"))

    hum_vals = [o["value"] for o in obs if o["variable"] == "humidity"]
    if hum_vals:
        avg_hum = sum(hum_vals) / len(hum_vals)
        hum_inv = max(0, (100 - avg_hum) / 100)
        components.append((hum_inv, 0.20, f"humidity_mean={round(avg_hum, 1)}"))

    wind_vals = [o["value"] for o in obs if o["variable"] == "wind_speed"]
    if wind_vals:
        avg_wind = sum(wind_vals) / len(wind_vals)
        wind_risk = min(avg_wind / 20.0, 1.0)
        components.append((wind_risk, 0.15, f"wind_mean={round(avg_wind, 1)}"))

    rain_vals = [o["value"] for o in obs if o["variable"] == "rainfall"]
    if rain_vals:
        avg_rain = sum(rain_vals) / len(rain_vals)
        if avg_rain > RAINFALL_SUPPRESS:
            suppression = min(avg_rain / 30.0, 1.0)
            explanation["rainfall_suppression"] = round(suppression, 3)
    else:
        avg_rain = 0.0

    temp_vals = [o["value"] for o in obs if o["variable"] == "temperature"]
    if temp_vals:
        avg_temp = sum(temp_vals) / len(temp_vals)
        temp_risk = max(0, (avg_temp - 20) / 30.0)
        components.append((temp_risk, 0.15, f"temperature_mean={round(avg_temp, 1)}"))

    if not components:
        return False, 0.0, {"reason": "no relevant observations"}

    total_weight = sum(w for _, w, _ in components)
    risk_score = sum(r * (w / total_weight) for r, w, _ in components)

    # 降雨抑制
    if rain_vals:
        avg_rain = sum(rain_vals) / len(rain_vals)
        if avg_rain > RAINFALL_SUPPRESS:
            suppression = min(avg_rain / 30.0, 1.0)
            risk_score *= 1.0 - suppression * 0.6

    for _, _, label in components:
        key, val = label.split("=", 1)
        explanation[key] = val

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


def infer_flood_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """洪涝 Ground Truth 推导（基于《国家防汛抗旱应急预案》）。

    当某观测变量缺失时，该变量不参与评分，其权重分配给其他变量。
    """
    explanation: dict[str, Any] = {}
    components: list[tuple[float, float, str]] = []

    rain24_vals = [o["value"] for o in obs if "rainfall_24h" in o["variable"]]
    if rain24_vals:
        avg_rain24 = sum(rain24_vals) / len(rain24_vals)
        rain24_risk = min(avg_rain24 / 100.0, 1.0)
        components.append((rain24_risk, 0.35, f"rainfall_24h={round(avg_rain24, 1)}"))

    rain6_vals = [o["value"] for o in obs if "rainfall_6h" in o["variable"]]
    if rain6_vals:
        avg_rain6 = sum(rain6_vals) / len(rain6_vals)
        rain6_risk = min(avg_rain6 / 50.0, 1.0)
        components.append((rain6_risk, 0.25, f"rainfall_6h={round(avg_rain6, 1)}"))

    soil_vals = [o["value"] for o in obs if "soil_moisture" in o["variable"]]
    if soil_vals:
        avg_soil = sum(soil_vals) / len(soil_vals)
        soil_sat_risk = max(0, (avg_soil - 0.6) / 0.4)
        components.append((soil_sat_risk, 0.15, f"soil_moisture={round(avg_soil, 3)}"))

    level_vals = [o["value"] for o in obs if "water_level" in o["variable"]]
    if level_vals:
        avg_level = sum(level_vals) / len(level_vals)
        level_risk = min(avg_level / 10.0, 1.0)
        components.append((level_risk, 0.25, f"water_level_mean={round(avg_level, 2)}"))

    if not components:
        return False, 0.0, {"reason": "no relevant observations"}

    total_weight = sum(w for _, w, _ in components)
    risk_score = sum(r * (w / total_weight) for r, w, _ in components)

    # 水位趋势检测（L4 关键能力）
    # 关键：必须按时间戳排序后再判断趋势，否则观测列表顺序可能不一致
    if len(level_vals) >= 3:
        # 按对应的时间戳排序后再取值
        level_obs = [o for o in obs if "water_level" in o["variable"]]
        level_obs_sorted = sorted(level_obs, key=lambda o: o.get("timestamp", ""))
        levels = [o["value"] for o in level_obs_sorted]
        increasing = all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))
        latest = levels[-1]
        if increasing and latest > 10.0:
            risk_score = max(risk_score, 0.75)
            explanation["water_trend"] = "rising_above_critical"
        elif increasing and latest >= 9.0:
            risk_score = max(risk_score, 0.50)
            explanation["water_trend"] = "rising_near_critical"

    for _, _, label in components:
        key, val = label.split("=", 1)
        explanation[key] = val

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.45
    return bool(decision), round(risk_score, 3), explanation


def infer_drought_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """干旱 Ground Truth 推导（基于气象干旱标准）。

    当某观测变量缺失时，该变量不参与评分，其权重分配给其他变量。
    """
    explanation: dict[str, Any] = {}
    components: list[tuple[float, float, str]] = []  # (risk_score, weight, label)

    spi_vals = [o["value"] for o in obs if o["variable"] == "SPI"]
    if spi_vals:
        avg_spi = sum(spi_vals) / len(spi_vals)
        spi_risk = max(0, (-avg_spi - 0.5) / 2.0)
        components.append((spi_risk, 0.30, f"spi_mean={round(avg_spi, 2)}"))

    palmer_vals = [o["value"] for o in obs if o["variable"] == "palmer_index"]
    if palmer_vals:
        avg_palmer = sum(palmer_vals) / len(palmer_vals)
        palmer_risk = max(0, (-avg_palmer - 0.25) / 1.5)
        components.append((palmer_risk, 0.25, f"palmer_mean={round(avg_palmer, 2)}"))

    hum_vals = [o["value"] for o in obs if o["variable"] == "humidity"]
    if hum_vals:
        avg_hum = sum(hum_vals) / len(hum_vals)
        hum_risk = max(0, (30 - avg_hum) / 30.0)
        components.append((hum_risk, 0.15, f"humidity_mean={round(avg_hum, 1)}"))

    rain_vals = [o["value"] for o in obs if o["variable"] == "rainfall_monthly"]
    if rain_vals:
        avg_rain = sum(rain_vals) / len(rain_vals)
        # 阈值从 50mm 降到 30mm，因为 QWeather 7 天预报外推的月估算通常偏低
        rain_deficit = max(0, (30 - avg_rain) / 30.0)
        components.append(
            (rain_deficit, 0.15, f"rainfall_monthly={round(avg_rain, 1)}")
        )

    ndvi_vals = [o["value"] for o in obs if o["variable"] == "NDVI"]
    if ndvi_vals:
        avg_ndvi = sum(ndvi_vals) / len(ndvi_vals)
        ndvi_risk = max(0, (0.5 - avg_ndvi) / 0.3)
        components.append((ndvi_risk, 0.15, f"ndvi_mean={round(avg_ndvi, 3)}"))

    if not components:
        return False, 0.0, {"reason": "no relevant observations"}

    # 动态归一化权重：将缺失变量的权重按比例分配给存在的变量
    total_weight = sum(w for _, w, _ in components)
    risk_score = sum(r * (w / total_weight) for r, w, _ in components)

    for _, _, label in components:
        key, val = label.split("=", 1)
        explanation[key] = val

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


def infer_heatwave_ground_truth(
    obs: list[dict], heat_duration_days: int = 3
) -> tuple[bool, float, dict[str, Any]]:
    """热浪 Ground Truth 推导（基于《中央气象台高温预警信号》）。

    当某观测变量缺失时，该变量不参与评分，其权重分配给其他变量。
    如果观测中包含 heat_duration_days，优先使用该值而非默认参数。
    """
    # 尝试从观测中读取实际持续高温天数
    duration_vals = [o["value"] for o in obs if o["variable"] == "heat_duration_days"]
    if duration_vals:
        actual_duration = int(duration_vals[0])
    else:
        actual_duration = heat_duration_days

    explanation: dict[str, Any] = {}
    components: list[tuple[float, float, str]] = []

    temp_vals = [o["value"] for o in obs if o["variable"] == "temperature_max"]
    if temp_vals:
        avg_temp = sum(temp_vals) / len(temp_vals)
        temp_risk = max(0, (avg_temp - 28) / 12.0)
        components.append(
            (temp_risk, 0.35, f"temperature_max_mean={round(avg_temp, 1)}")
        )

    duration_factor = min(actual_duration / 5.0, 1.0)
    components.append((duration_factor, 0.25, f"heat_duration_days={actual_duration}"))

    wbt_vals = [o["value"] for o in obs if o["variable"] == "wet_bulb_temp"]
    if wbt_vals:
        avg_wbt = sum(wbt_vals) / len(wbt_vals)
        wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)
        components.append((wbt_risk, 0.25, f"wet_bulb_temp_mean={round(avg_wbt, 1)}"))

    hum_vals = [o["value"] for o in obs if o["variable"] == "humidity"]
    if hum_vals:
        avg_hum = sum(hum_vals) / len(hum_vals)
        hum_risk = max(0, (avg_hum - 50) / 50.0)
        components.append((hum_risk, 0.15, f"humidity_mean={round(avg_hum, 1)}"))

    if not components:
        return False, 0.0, {"reason": "no relevant observations"}

    total_weight = sum(w for _, w, _ in components)
    risk_score = sum(r * (w / total_weight) for r, w, _ in components)

    for _, _, label in components:
        key, val = label.split("=", 1)
        explanation[key] = val

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


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


# ===========================================================================
# 难度级别常量
# ===========================================================================


class DifficultyLevel:
    """四级难度标签。"""

    L1_EASY = "L1"
    L2_MEDIUM = "L2"
    L3_HARD = "L3"
    L4_BEYOND = "L4"
