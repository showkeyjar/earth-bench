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
from dataclasses import dataclass, field
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
    """森林火险 Ground Truth 推导（基于国家林草局标准）。"""
    explanation: dict[str, Any] = {}
    risk_score = 0.0
    avg_fwi = sum(o["value"] for o in obs if o["variable"] == "FWI") / max(1, sum(1 for o in obs if o["variable"] == "FWI"))
    avg_hum = sum(o["value"] for o in obs if o["variable"] == "humidity") / max(1, sum(1 for o in obs if o["variable"] == "humidity"))
    avg_wind = sum(o["value"] for o in obs if o["variable"] == "wind_speed") / max(1, sum(1 for o in obs if o["variable"] == "wind_speed"))
    avg_rain = sum(o["value"] for o in obs if o["variable"] == "rainfall") / max(1, sum(1 for o in obs if o["variable"] == "rainfall"))
    avg_temp = sum(o["value"] for o in obs if o["variable"] == "temperature") / max(1, sum(1 for o in obs if o["variable"] == "temperature"))

    fwi_norm = min(avg_fwi / 60.0, 1.0)
    risk_score += fwi_norm * 0.4
    explanation["fwi_mean"] = round(avg_fwi, 1)

    hum_inv = max(0, (100 - avg_hum) / 100)
    risk_score += hum_inv * 0.2
    explanation["humidity_mean"] = round(avg_hum, 1)

    wind_risk = min(avg_wind / 20.0, 1.0)
    risk_score += wind_risk * 0.15
    explanation["wind_mean"] = round(avg_wind, 1)

    if avg_rain > RAINFALL_SUPPRESS:
        suppression = min(avg_rain / 30.0, 1.0)
        risk_score *= (1.0 - suppression * 0.6)
        explanation["rainfall_suppression"] = round(suppression, 3)

    temp_risk = max(0, (avg_temp - 20) / 30.0)
    risk_score += temp_risk * 0.15
    explanation["temperature_mean"] = round(avg_temp, 1)

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


def infer_flood_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """洪涝 Ground Truth 推导（基于《国家防汛抗旱应急预案》）。"""
    explanation: dict[str, Any] = {}
    risk_score = 0.0
    avg_rain24 = sum(o["value"] for o in obs if "rainfall_24h" in o["variable"]) / max(1, sum(1 for o in obs if "rainfall_24h" in o["variable"]))
    avg_rain6 = sum(o["value"] for o in obs if "rainfall_6h" in o["variable"]) / max(1, sum(1 for o in obs if "rainfall_6h" in o["variable"]))
    avg_soil = sum(o["value"] for o in obs if "soil_moisture" in o["variable"]) / max(1, sum(1 for o in obs if "soil_moisture" in o["variable"]))
    avg_level = sum(o["value"] for o in obs if "water_level" in o["variable"]) / max(1, sum(1 for o in obs if "water_level" in o["variable"]))

    rain24_risk = min(avg_rain24 / 100.0, 1.0)
    risk_score += rain24_risk * 0.35
    explanation["rainfall_24h"] = round(avg_rain24, 1)

    rain6_risk = min(avg_rain6 / 50.0, 1.0)
    risk_score += rain6_risk * 0.25
    explanation["rainfall_6h"] = round(avg_rain6, 1)

    soil_sat_risk = max(0, (avg_soil - 0.6) / 0.4)
    risk_score += soil_sat_risk * 0.15
    explanation["soil_moisture"] = round(avg_soil, 3)

    level_risk = min(avg_level / 10.0, 1.0)
    risk_score += level_risk * 0.25
    explanation["water_level_mean"] = round(avg_level, 2)

    # --- 新增：水位趋势检测（L4 关键能力） ---
    level_obs = [o for o in obs if "water_level" in o["variable"]]
    if len(level_obs) >= 3:
        levels = [o["value"] for o in level_obs]
        # 单调递增趋势
        increasing = all(levels[i] < levels[i+1] for i in range(len(levels)-1))
        latest = levels[-1]
        # 最新值超过警戒 + 持续上涨 → 高风险
        if increasing and latest > 10.0:
            risk_score = max(risk_score, 0.75)  # 强制提升为高置信度
            explanation["water_trend"] = "rising_above_critical"
        elif increasing and latest >= 9.0:
            # 接近警戒线且在上涨 → 中等风险
            risk_score = max(risk_score, 0.50)
            explanation["water_trend"] = "rising_near_critical"

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.45
    return bool(decision), round(risk_score, 3), explanation


def infer_drought_ground_truth(obs: list[dict]) -> tuple[bool, float, dict[str, Any]]:
    """干旱 Ground Truth 推导（基于气象干旱标准）。"""
    explanation: dict[str, Any] = {}
    risk_score = 0.0
    avg_spi = sum(o["value"] for o in obs if o["variable"] == "SPI") / max(1, sum(1 for o in obs if o["variable"] == "SPI"))
    avg_palmer = sum(o["value"] for o in obs if o["variable"] == "palmer_index") / max(1, sum(1 for o in obs if o["variable"] == "palmer_index"))
    avg_hum = sum(o["value"] for o in obs if o["variable"] == "humidity") / max(1, sum(1 for o in obs if o["variable"] == "humidity"))
    avg_rain = sum(o["value"] for o in obs if o["variable"] == "rainfall_monthly") / max(1, sum(1 for o in obs if o["variable"] == "rainfall_monthly"))
    avg_ndvi = sum(o["value"] for o in obs if o["variable"] == "NDVI") / max(1, sum(1 for o in obs if o["variable"] == "NDVI"))

    spi_risk = max(0, (-avg_spi - 0.5) / 2.0)
    risk_score += spi_risk * 0.30
    explanation["spi_mean"] = round(avg_spi, 2)

    palmer_risk = max(0, (-avg_palmer - 0.25) / 1.5)
    risk_score += palmer_risk * 0.25
    explanation["palmer_mean"] = round(avg_palmer, 2)

    hum_risk = max(0, (30 - avg_hum) / 30.0)
    risk_score += hum_risk * 0.15
    explanation["humidity_mean"] = round(avg_hum, 1)

    rain_deficit = max(0, (50 - avg_rain) / 50.0)
    risk_score += rain_deficit * 0.15
    explanation["rainfall_monthly"] = round(avg_rain, 1)

    ndvi_risk = max(0, (0.5 - avg_ndvi) / 0.3)
    risk_score += ndvi_risk * 0.15
    explanation["ndvi_mean"] = round(avg_ndvi, 3)

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


def infer_heatwave_ground_truth(obs: list[dict], heat_duration_days: int = 3) -> tuple[bool, float, dict[str, Any]]:
    """热浪 Ground Truth 推导（基于《中央气象台高温预警信号》）。"""
    explanation: dict[str, Any] = {}
    risk_score = 0.0
    avg_temp = sum(o["value"] for o in obs if o["variable"] == "temperature_max") / max(1, sum(1 for o in obs if o["variable"] == "temperature_max"))
    avg_wbt = sum(o["value"] for o in obs if o["variable"] == "wet_bulb_temp") / max(1, sum(1 for o in obs if o["variable"] == "wet_bulb_temp"))
    avg_hum = sum(o["value"] for o in obs if o["variable"] == "humidity") / max(1, sum(1 for o in obs if o["variable"] == "humidity"))

    temp_risk = max(0, (avg_temp - 28) / 12.0)
    risk_score += temp_risk * 0.35
    explanation["temperature_max_mean"] = round(avg_temp, 1)

    duration_factor = min(heat_duration_days / 5.0, 1.0)
    risk_score += duration_factor * 0.25
    explanation["heat_duration_days"] = heat_duration_days

    if avg_wbt:
        wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)
        risk_score += wbt_risk * 0.25
        explanation["wet_bulb_temp_mean"] = round(avg_wbt, 1)

    hum_risk = max(0, (avg_hum - 50) / 50.0)
    risk_score += hum_risk * 0.15
    explanation["humidity_mean"] = round(avg_hum, 1)

    risk_score = max(0.0, min(risk_score, 1.0))
    decision = risk_score >= 0.4
    return bool(decision), round(risk_score, 3), explanation


# ===========================================================================
# AlertBenchmarkSuite — 四大类别测试集生成器
# ===========================================================================

def _make_obs(source: str, variable: str, value: float, unit: str, timestamp: str, confidence: float = 0.95) -> dict:
    """辅助函数：创建单条 Observation dict。"""
    return {
        "source": source, "variable": variable, "value": value,
        "unit": unit, "timestamp": timestamp, "confidence": confidence,
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
    suite.append({
        "case_id": "fire-l1-extreme-risk", "difficulty": "L1", "category": "fire",
        "region": "Xiangshan-Beijing", "ground_truth": True,
        "observations": [
            _make_obs("ECMWF", "FWI", 52.0, "", "2026-07-10T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 55.0, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 10.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wind_speed", 18.0, "m/s", "2026-07-11T12:00:00+08:00"),
            _make_obs("MODIS", "temperature", 38.0, "°C", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_fire_ground_truth,
    })
    suite.append({
        "case_id": "fire-l1-low-risk", "difficulty": "L1", "category": "fire",
        "region": "WestLake-Hangzhou", "ground_truth": False,
        "observations": [
            _make_obs("ECMWF", "FWI", 15.0, "", "2026-07-10T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 18.0, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 65.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "rainfall", 8.0, "mm", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_fire_ground_truth,
    })
    suite.append({
        "case_id": "fire-l2-fwi-high-rain-suppress", "difficulty": "L2", "category": "fire",
        "region": "ChangbaiMountain", "ground_truth": False,
        "observations": [
            _make_obs("ECMWF", "FWI", 48.0, "", "2026-07-10T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 50.0, "", "2026-07-11T06:00:00+08:00"),
            _make_obs("Station", "humidity", 15.0, "%", "2026-07-11T06:00:00+08:00"),
            _make_obs("Station", "rainfall", 25.0, "mm", "2026-07-11T00:00:00+08:00"),
        ],
        "_gt_fn": infer_fire_ground_truth,
    })
    suite.append({
        "case_id": "fire-l3-coastal-humid", "difficulty": "L3", "category": "fire",
        "region": "Shenzhen-Coast", "ground_truth": False,
        "observations": [
            _make_obs("ECMWF", "FWI", 32.0, "", "2026-07-10T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 35.0, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 55.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("MODIS", "temperature", 34.0, "°C", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_fire_ground_truth,
    })
    suite.append({
        "case_id": "fire-l4-trend-rising", "difficulty": "L4", "category": "fire",
        "region": "GreaterKhingan", "ground_truth": True,
        "observations": [
            _make_obs("ECMWF", "FWI", 30.0, "", "2026-07-08T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 35.0, "", "2026-07-09T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 39.0, "", "2026-07-10T12:00:00+08:00"),
            _make_obs("ECMWF", "FWI", 43.0, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 25.0, "%", "2026-07-08T12:00:00+08:00"),
            _make_obs("Station", "humidity", 20.0, "%", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_fire_ground_truth,
    })

    # ---- Flood (5 L1-L4) ----
    suite.append({
        "case_id": "flood-l1-extreme-rainfall", "difficulty": "L1", "category": "flood",
        "region": "Wuhan-Yangtze", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "rainfall_24h", 120.0, "mm", "2026-07-11T12:00:00+08:00"),
            _make_obs("CMA", "rainfall_6h", 65.0, "mm", "2026-07-11T12:00:00+08:00"),
            _make_obs("Sensor", "soil_moisture", 0.90, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Hydro", "water_level", 12.5, "m", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_flood_ground_truth,
    })
    suite.append({
        "case_id": "flood-l1-no-rain", "difficulty": "L1", "category": "flood",
        "region": "Lhasa-Tibet", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "rainfall_24h", 0.0, "mm", "2026-07-11T12:00:00+08:00"),
            _make_obs("Sensor", "soil_moisture", 0.20, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Hydro", "water_level", 2.0, "m", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_flood_ground_truth,
    })
    suite.append({
        "case_id": "flood-l2-soil-saturated", "difficulty": "L2", "category": "flood",
        "region": "Guangzhou-PearlR", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "rainfall_24h", 40.0, "mm", "2026-07-11T12:00:00+08:00"),
            _make_obs("Sensor", "soil_moisture", 0.88, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Hydro", "water_level", 8.5, "m", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_flood_ground_truth,
    })
    suite.append({
        "case_id": "flood-l3-good-drainage", "difficulty": "L3", "category": "flood",
        "region": "Guilin-Guangxi", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "rainfall_24h", 80.0, "mm", "2026-07-11T12:00:00+08:00"),
            _make_obs("Sensor", "soil_moisture", 0.55, "", "2026-07-11T12:00:00+08:00"),
            _make_obs("Hydro", "water_level", 3.0, "m", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_flood_ground_truth,
    })
    suite.append({
        "case_id": "flood-l4-water-trend-rise", "difficulty": "L4", "category": "flood",
        "region": "Nanjing-Yangtze", "ground_truth": True,
        "observations": [
            _make_obs("Hydro", "water_level", 9.5, "m", "2026-07-10T12:00:00+08:00"),
            _make_obs("Hydro", "water_level", 10.2, "m", "2026-07-11T06:00:00+08:00"),
            _make_obs("Hydro", "water_level", 10.8, "m", "2026-07-11T12:00:00+08:00"),
        ],
        "_gt_fn": infer_flood_ground_truth,
    })

    # ---- Drought (5 L1-L4) ----
    suite.append({
        "case_id": "drought-l1-severe-spi", "difficulty": "L1", "category": "drought",
        "region": "Kunming-Yunnan", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "SPI", -2.3, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "palmer_index", -0.8, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("Station", "humidity", 15.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("CMA", "rainfall_monthly", 5.0, "mm", "2026-07-01T00:00:00+08:00"),
            _make_obs("MODIS", "NDVI", 0.15, "", "2026-07-05T00:00:00+08:00"),
        ],
        "_gt_fn": infer_drought_ground_truth,
    })
    suite.append({
        "case_id": "drought-l1-normal", "difficulty": "L1", "category": "drought",
        "region": "Hangzhou-Zhejiang", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "SPI", -0.2, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "palmer_index", -0.1, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("Station", "humidity", 60.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("CMA", "rainfall_monthly", 150.0, "mm", "2026-07-01T00:00:00+08:00"),
            _make_obs("MODIS", "NDVI", 0.65, "", "2026-07-05T00:00:00+08:00"),
        ],
        "_gt_fn": infer_drought_ground_truth,
    })
    suite.append({
        "case_id": "drought-l2-borderline-spi", "difficulty": "L2", "category": "drought",
        "region": "Harbin-Heilongjiang", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "SPI", -0.8, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "palmer_index", -0.3, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "rainfall_monthly", 80.0, "mm", "2026-07-01T00:00:00+08:00"),
            _make_obs("MODIS", "NDVI", 0.50, "", "2026-07-05T00:00:00+08:00"),
        ],
        "_gt_fn": infer_drought_ground_truth,
    })
    suite.append({
        "case_id": "drought-l3-palmerspi-conflict", "difficulty": "L3", "category": "drought",
        "region": "Urumqi-Xinjiang", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "SPI", -0.3, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "palmer_index", -1.2, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("Station", "humidity", 20.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("CMA", "rainfall_monthly", 5.0, "mm", "2026-07-01T00:00:00+08:00"),
            _make_obs("MODIS", "NDVI", 0.10, "", "2026-07-05T00:00:00+08:00"),
        ],
        "_gt_fn": infer_drought_ground_truth,
    })
    suite.append({
        "case_id": "drought-l4-spir-declining", "difficulty": "L4", "category": "drought",
        "region": "Taiyuan-Shanxi", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "SPI", -0.5, "", "2026-05-01T00:00:00+08:00"),
            _make_obs("CMA", "SPI", -0.9, "", "2026-06-01T00:00:00+08:00"),
            _make_obs("CMA", "SPI", -1.6, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "palmer_index", -0.6, "", "2026-07-01T00:00:00+08:00"),
            _make_obs("CMA", "rainfall_monthly", 10.0, "mm", "2026-07-01T00:00:00+08:00"),
            _make_obs("MODIS", "NDVI", 0.20, "", "2026-07-05T00:00:00+08:00"),
        ],
        "_gt_fn": infer_drought_ground_truth,
    })

    # ---- HeatWave (5 L1-L4) ----
    suite.append({
        "case_id": "heat-l1-extreme-wetbulb", "difficulty": "L1", "category": "heat",
        "region": "Chongqing-HotPotato", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "temperature_max", 40.0, "°C", "2026-07-09T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 38.0, "°C", "2026-07-10T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 39.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 75.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 30.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 28.0, "°C", "2026-07-10T12:00:00+08:00"),
        ],
        "heat_duration_days": 5,
        "_gt_fn": infer_heatwave_ground_truth,
    })
    suite.append({
        "case_id": "heat-l1-cool-comfortable", "difficulty": "L1", "category": "heat",
        "region": "Kunming-SpringCity", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "temperature_max", 26.0, "°C", "2026-07-09T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 27.0, "°C", "2026-07-10T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 25.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 40.0, "%", "2026-07-11T12:00:00+08:00"),
        ],
        "heat_duration_days": 0,
        "_gt_fn": infer_heatwave_ground_truth,
    })
    suite.append({
        "case_id": "heat-l2-hot-but-dry", "difficulty": "L2", "category": "heat",
        "region": "Urumqi-Xinjiang", "ground_truth": False,
        "observations": [
            _make_obs("CMA", "temperature_max", 36.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 15.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 22.0, "°C", "2026-07-11T12:00:00+08:00"),
        ],
        "heat_duration_days": 3,
        "_gt_fn": infer_heatwave_ground_truth,
    })
    suite.append({
        "case_id": "heat-l3-borderline", "difficulty": "L3", "category": "heat",
        "region": "Nanjing-OvenCity", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "temperature_max", 34.5, "°C", "2026-07-09T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 35.0, "°C", "2026-07-10T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 34.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "humidity", 72.0, "%", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 27.5, "°C", "2026-07-11T12:00:00+08:00"),
        ],
        "heat_duration_days": 3,
        "_gt_fn": infer_heatwave_ground_truth,
    })
    suite.append({
        "case_id": "heat-l4-progressive-heat", "difficulty": "L4", "category": "heat",
        "region": "Wuhan-Yangtze", "ground_truth": True,
        "observations": [
            _make_obs("CMA", "temperature_max", 31.0, "°C", "2026-07-09T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 35.0, "°C", "2026-07-10T12:00:00+08:00"),
            _make_obs("CMA", "temperature_max", 39.0, "°C", "2026-07-11T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 24.0, "°C", "2026-07-09T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 27.0, "°C", "2026-07-10T12:00:00+08:00"),
            _make_obs("Station", "wet_bulb_temp", 29.5, "°C", "2026-07-11T12:00:00+08:00"),
        ],
        "heat_duration_days": 3,
        "_gt_fn": infer_heatwave_ground_truth,
    })

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
