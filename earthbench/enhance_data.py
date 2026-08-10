"""增强发布数据 — 为 JSON 添加坐标、详细证据链、推理步骤，并生成历史追踪记录。

此模块被 publish_pipeline.py 调用，也可独立运行增强现有 latest.json。
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 北京时间时区
CST = timezone(timedelta(hours=8))

# ===========================================================================
# 区域坐标映射
# ===========================================================================

REGION_COORDS: dict[str, dict[str, Any]] = {
    # (城市级精细范围)
    "Xiangshan-Beijing": {
        "lat": 39.99,
        "lng": 116.16,
        "cn": "北京·香山",
        "radius_km": 20,
    },
    "WestLake-Hangzhou": {
        "lat": 30.25,
        "lng": 120.15,
        "cn": "杭州·西湖",
        "radius_km": 15,
    },
    "Shenzhen-Coast": {"lat": 22.54, "lng": 114.06, "cn": "深圳·海岸", "radius_km": 20},
    "Wuhan-Yangtze": {"lat": 30.59, "lng": 114.31, "cn": "武汉·长江", "radius_km": 25},
    "Guangzhou-PearlR": {
        "lat": 23.13,
        "lng": 113.27,
        "cn": "广州·珠江",
        "radius_km": 25,
    },
    "Guilin-Guangxi": {"lat": 25.27, "lng": 110.29, "cn": "桂林·广西", "radius_km": 20},
    "Nanjing-Yangtze": {
        "lat": 32.06,
        "lng": 118.80,
        "cn": "南京·长江",
        "radius_km": 20,
    },
    "Kunming-Yunnan": {"lat": 25.04, "lng": 102.71, "cn": "昆明·云南", "radius_km": 25},
    "Hangzhou-Zhejiang": {
        "lat": 30.27,
        "lng": 120.16,
        "cn": "杭州·浙江",
        "radius_km": 15,
    },
    "Chongqing-HotPotato": {"lat": 29.43, "lng": 106.91, "cn": "重庆", "radius_km": 20},
    "Taiyuan-Shanxi": {"lat": 37.87, "lng": 112.55, "cn": "太原·山西", "radius_km": 18},
    "Lhasa-Tibet": {"lat": 29.65, "lng": 91.11, "cn": "拉萨·西藏", "radius_km": 15},
    "Harbin-Heilongjiang": {
        "lat": 45.80,
        "lng": 126.53,
        "cn": "哈尔滨·黑龙江",
        "radius_km": 25,
    },
    # (林区/保护区大范围)
    "ChangbaiMountain": {"lat": 42.02, "lng": 128.08, "cn": "长白山", "radius_km": 50},
    "GreaterKhingan": {"lat": 51.17, "lng": 124.39, "cn": "大兴安岭", "radius_km": 60},
    "Urumqi-Xinjiang": {
        "lat": 43.83,
        "lng": 87.62,
        "cn": "乌鲁木齐·新疆",
        "radius_km": 30,
    },
    "Kunming-SpringCity": {
        "lat": 25.04,
        "lng": 102.71,
        "cn": "昆明·春城",
        "radius_km": 20,
    },
    "Nanjing-OvenCity": {"lat": 32.06, "lng": 118.80, "cn": "南京", "radius_km": 18},
}

# ===========================================================================
# 真实地理边界多边形 — 每个区域的多点边界坐标
# 基于实际地形/行政边界近似，替代之前的随机生成多边形
# ===========================================================================

REGION_POLYGONS: dict[str, list[dict[str, float]]] = {
    # 北京·香山 — 香山公园及周边林区
    "Xiangshan-Beijing": [
        {"lat": 40.02, "lng": 116.12},
        {"lat": 40.01, "lng": 116.18},
        {"lat": 39.98, "lng": 116.20},
        {"lat": 39.96, "lng": 116.16},
        {"lat": 39.97, "lng": 116.10},
        {"lat": 40.00, "lng": 116.09},
    ],
    # 杭州·西湖 — 西湖景区及周边城区
    "WestLake-Hangzhou": [
        {"lat": 30.28, "lng": 120.12},
        {"lat": 30.27, "lng": 120.18},
        {"lat": 30.24, "lng": 120.19},
        {"lat": 30.22, "lng": 120.15},
        {"lat": 30.23, "lng": 120.10},
        {"lat": 30.26, "lng": 120.09},
    ],
    # 长白山 — 长白山自然保护区
    "ChangbaiMountain": [
        {"lat": 42.10, "lng": 127.90},
        {"lat": 42.08, "lng": 128.30},
        {"lat": 41.95, "lng": 128.35},
        {"lat": 41.90, "lng": 128.00},
        {"lat": 41.95, "lng": 127.75},
        {"lat": 42.05, "lng": 127.80},
    ],
    # 深圳·海岸 — 深圳沿海地带
    "Shenzhen-Coast": [
        {"lat": 22.58, "lng": 113.95},
        {"lat": 22.56, "lng": 114.15},
        {"lat": 22.50, "lng": 114.20},
        {"lat": 22.48, "lng": 114.10},
        {"lat": 22.50, "lng": 113.92},
        {"lat": 22.55, "lng": 113.90},
    ],
    # 大兴安岭 — 大兴安岭林区
    "GreaterKhingan": [
        {"lat": 51.40, "lng": 124.10},
        {"lat": 51.35, "lng": 124.70},
        {"lat": 51.00, "lng": 124.80},
        {"lat": 50.90, "lng": 124.30},
        {"lat": 50.95, "lng": 123.90},
        {"lat": 51.25, "lng": 123.95},
    ],
    # 武汉·长江 — 武汉长江段及周边
    "Wuhan-Yangtze": [
        {"lat": 30.68, "lng": 114.20},
        {"lat": 30.65, "lng": 114.40},
        {"lat": 30.55, "lng": 114.45},
        {"lat": 30.50, "lng": 114.30},
        {"lat": 30.52, "lng": 114.15},
        {"lat": 30.62, "lng": 114.12},
    ],
    # 广州·珠江 — 广州珠江段
    "Guangzhou-PearlR": [
        {"lat": 23.18, "lng": 113.20},
        {"lat": 23.15, "lng": 113.35},
        {"lat": 23.08, "lng": 113.35},
        {"lat": 23.05, "lng": 113.25},
        {"lat": 23.10, "lng": 113.15},
        {"lat": 23.15, "lng": 113.12},
    ],
    # 桂林·广西 — 桂林岩溶地貌区
    "Guilin-Guangxi": [
        {"lat": 25.32, "lng": 110.20},
        {"lat": 25.30, "lng": 110.38},
        {"lat": 25.22, "lng": 110.40},
        {"lat": 25.18, "lng": 110.28},
        {"lat": 25.22, "lng": 110.18},
        {"lat": 25.30, "lng": 110.15},
    ],
    # 南京·长江 — 南京长江段
    "Nanjing-Yangtze": [
        {"lat": 32.10, "lng": 118.70},
        {"lat": 32.08, "lng": 118.90},
        {"lat": 32.02, "lng": 118.92},
        {"lat": 32.00, "lng": 118.75},
        {"lat": 32.03, "lng": 118.65},
        {"lat": 32.08, "lng": 118.65},
    ],
    # 昆明·云南 — 昆明城区及周边
    "Kunming-Yunnan": [
        {"lat": 25.08, "lng": 102.65},
        {"lat": 25.06, "lng": 102.78},
        {"lat": 25.00, "lng": 102.80},
        {"lat": 24.98, "lng": 102.70},
        {"lat": 25.00, "lng": 102.60},
        {"lat": 25.06, "lng": 102.60},
    ],
    # 杭州·浙江 — 杭州湾区域
    "Hangzhou-Zhejiang": [
        {"lat": 30.30, "lng": 120.10},
        {"lat": 30.28, "lng": 120.22},
        {"lat": 30.22, "lng": 120.25},
        {"lat": 30.20, "lng": 120.15},
        {"lat": 30.24, "lng": 120.08},
        {"lat": 30.28, "lng": 120.08},
    ],
    # 重庆 — 重庆主城区及周边
    "Chongqing-HotPotato": [
        {"lat": 29.50, "lng": 106.80},
        {"lat": 29.48, "lng": 107.00},
        {"lat": 29.38, "lng": 107.02},
        {"lat": 29.35, "lng": 106.85},
        {"lat": 29.38, "lng": 106.72},
        {"lat": 29.45, "lng": 106.72},
    ],
    # 太原·山西 — 太原盆地
    "Taiyuan-Shanxi": [
        {"lat": 37.92, "lng": 112.45},
        {"lat": 37.90, "lng": 112.65},
        {"lat": 37.82, "lng": 112.67},
        {"lat": 37.78, "lng": 112.50},
        {"lat": 37.82, "lng": 112.38},
        {"lat": 37.90, "lng": 112.38},
    ],
    # 拉萨·西藏 — 拉萨河谷
    "Lhasa-Tibet": [
        {"lat": 29.70, "lng": 91.05},
        {"lat": 29.68, "lng": 91.18},
        {"lat": 29.62, "lng": 91.20},
        {"lat": 29.58, "lng": 91.10},
        {"lat": 29.60, "lng": 91.00},
        {"lat": 29.68, "lng": 91.00},
    ],
    # 哈尔滨·黑龙江 — 松嫩平原
    "Harbin-Heilongjiang": [
        {"lat": 45.85, "lng": 126.45},
        {"lat": 45.82, "lng": 126.65},
        {"lat": 45.72, "lng": 126.67},
        {"lat": 45.68, "lng": 126.50},
        {"lat": 45.72, "lng": 126.35},
        {"lat": 45.82, "lng": 126.35},
    ],
    # 乌鲁木齐·新疆 — 乌鲁木齐及周边
    "Urumqi-Xinjiang": [
        {"lat": 43.88, "lng": 87.50},
        {"lat": 43.85, "lng": 87.75},
        {"lat": 43.78, "lng": 87.78},
        {"lat": 43.75, "lng": 87.60},
        {"lat": 43.78, "lng": 87.42},
        {"lat": 43.85, "lng": 87.42},
    ],
    # 昆明·春城 — 昆明主城
    "Kunming-SpringCity": [
        {"lat": 25.08, "lng": 102.68},
        {"lat": 25.06, "lng": 102.76},
        {"lat": 25.02, "lng": 102.78},
        {"lat": 25.00, "lng": 102.70},
        {"lat": 25.02, "lng": 102.62},
        {"lat": 25.06, "lng": 102.62},
    ],
    # 南京 — 南京主城
    "Nanjing-OvenCity": [
        {"lat": 32.08, "lng": 118.72},
        {"lat": 32.06, "lng": 118.88},
        {"lat": 32.00, "lng": 118.90},
        {"lat": 31.98, "lng": 118.78},
        {"lat": 32.02, "lng": 118.68},
        {"lat": 32.06, "lng": 118.68},
    ],
}

# ===========================================================================
# 变量元数据：阈值、显示名、描述
# ===========================================================================

VARIABLE_META: dict[str, dict[str, Any]] = {
    # === Fire ===
    "FWI": {
        "display": "火险指数(FWI)",
        "threshold": 40.0,
        "direction": "high",
        "desc": "综合温度、湿度、风速、降雨的火灾天气指标，值越高火险越大",
        "danger_text": "远超危险阈值",
        "warn_text": "接近危险阈值",
        "safe_text": "处于安全范围",
        "weight": 0.40,
    },
    "humidity": {
        "display": "空气湿度",
        "threshold": 20.0,
        "direction": "low",
        "desc": "湿度低于20%时，可燃物极易点燃，火势难以控制",
        "danger_text": "极度干燥",
        "warn_text": "偏低",
        "safe_text": "湿度正常",
        "weight": 0.20,
    },
    "wind_speed": {
        "display": "风速",
        "threshold": 12.0,
        "direction": "high",
        "desc": "风速超过12 m/s时，火势蔓延速度急剧增加",
        "danger_text": "大风加速蔓延",
        "warn_text": "风速偏高",
        "safe_text": "风速正常",
        "weight": 0.15,
    },
    "temperature": {
        "display": "温度",
        "threshold": 35.0,
        "direction": "high",
        "desc": "高温加速可燃物水分蒸发，降低燃点",
        "danger_text": "高温加剧火险",
        "warn_text": "温度偏高",
        "safe_text": "温度正常",
        "weight": 0.15,
    },
    "rainfall": {
        "display": "降雨量",
        "threshold": 10.0,
        "direction": "suppress",
        "desc": "降雨量超过10mm时，对火险有显著抑制作用",
        "danger_text": "无明显降雨",
        "warn_text": "微量降雨",
        "safe_text": "有效降雨抑制火险",
        "weight": 0.10,
    },
    # === Flood ===
    "rainfall_24h": {
        "display": "24小时降水量",
        "threshold": 50.0,
        "direction": "high",
        "desc": "24小时降水量超过50mm达到暴雨级别，洪涝风险显著增加",
        "danger_text": "暴雨级别",
        "warn_text": "大雨",
        "safe_text": "降水量正常",
        "weight": 0.35,
    },
    "rainfall_6h": {
        "display": "6小时降水量",
        "threshold": 30.0,
        "direction": "high",
        "desc": "短时强降水超过30mm/6h，极易引发城市内涝和山洪",
        "danger_text": "短时强降水",
        "warn_text": "短时降水偏大",
        "safe_text": "降水正常",
        "weight": 0.25,
    },
    "soil_moisture": {
        "display": "土壤含水率",
        "threshold": 0.85,
        "direction": "high",
        "desc": "土壤含水率超过85%表示土壤接近饱和，渗透能力极低",
        "danger_text": "土壤饱和",
        "warn_text": "土壤偏湿",
        "safe_text": "土壤含水正常",
        "weight": 0.15,
    },
    "water_level": {
        "display": "水位",
        "threshold": 10.0,
        "direction": "high",
        "desc": "水位超过10m接近或超过警戒水位，溃堤风险增加",
        "danger_text": "超过警戒水位",
        "warn_text": "接近警戒水位",
        "safe_text": "水位正常",
        "weight": 0.25,
    },
    # === Drought ===
    "SPI": {
        "display": "标准化降水指数(SPI)",
        "threshold": -1.0,
        "direction": "low",
        "desc": "SPI低于-1.0表示中度以上干旱，低于-2.0为极端干旱",
        "danger_text": "中度以上干旱",
        "warn_text": "轻度干旱",
        "safe_text": "降水正常",
        "weight": 0.30,
    },
    "palmer_index": {
        "display": "Palmer干旱指数",
        "threshold": -0.5,
        "direction": "low",
        "desc": "Palmer指数低于-0.5表示干旱开始，低于-2.0为严重干旱",
        "danger_text": "干旱严重",
        "warn_text": "干旱迹象",
        "safe_text": "无干旱",
        "weight": 0.25,
    },
    "rainfall_monthly": {
        "display": "月降雨量",
        "threshold": 30.0,
        "direction": "low",
        "desc": "月降雨量低于30mm表示严重降水亏缺，干旱风险显著增加",
        "danger_text": "严重降水亏缺",
        "warn_text": "降水偏少",
        "safe_text": "降水正常",
        "weight": 0.15,
    },
    "NDVI": {
        "display": "植被指数(NDVI)",
        "threshold": 0.3,
        "direction": "low",
        "desc": "NDVI低于0.3表示植被严重退化，干旱影响显著",
        "danger_text": "植被严重退化",
        "warn_text": "植被偏弱",
        "safe_text": "植被正常",
        "weight": 0.15,
    },
    # === Heat ===
    "temperature_max": {
        "display": "最高温度",
        "threshold": 35.0,
        "direction": "high",
        "desc": "最高温度超过35°C达到高温预警标准",
        "danger_text": "高温预警",
        "warn_text": "温度偏高",
        "safe_text": "温度正常",
        "weight": 0.35,
    },
    "wet_bulb_temp": {
        "display": "湿球温度",
        "threshold": 27.0,
        "direction": "high",
        "desc": "湿球温度超过27°C时人体散热困难，超过31°C极度危险",
        "danger_text": "闷热危险",
        "warn_text": "体感不适",
        "safe_text": "体感正常",
        "weight": 0.25,
    },
    "heat_duration_days": {
        "display": "持续高温天数",
        "threshold": 3.0,
        "direction": "high",
        "desc": "连续高温超过3天，累积热应力显著增加健康风险",
        "danger_text": "持续高温",
        "warn_text": "短期高温",
        "safe_text": "无持续高温",
        "weight": 0.25,
    },
}

# ===========================================================================
# 数据增强函数
# ===========================================================================


def _get_status(value: float, meta: dict[str, Any]) -> str:
    """判断观测值的状态：danger / warn / safe。"""
    threshold = meta["threshold"]
    direction = meta["direction"]

    if direction == "high":
        if value >= threshold:
            return "danger"
        elif value >= threshold * 0.8:
            return "warn"
        else:
            return "safe"
    elif direction == "low":
        if value <= threshold:
            return "danger"
        elif value <= threshold * 1.5:
            return "warn"
        else:
            return "safe"
    elif direction == "suppress":
        if value < threshold * 0.5:
            return "danger"
        elif value < threshold:
            return "warn"
        else:
            return "safe"
    return "safe"


def _aggregate_observations(observations: list[dict]) -> dict[str, list[dict]]:
    """按变量名分组观测数据。"""
    grouped: dict[str, list[dict]] = {}
    for obs in observations:
        var = obs["variable"]
        if var not in grouped:
            grouped[var] = []
        grouped[var].append(obs)
    return grouped


def _build_detailed_evidence(
    observations: list[dict], category: str
) -> list[dict[str, Any]]:
    """构建详细证据列表。"""
    grouped = _aggregate_observations(observations)
    evidence_list = []

    for var, obs_list in grouped.items():
        meta = VARIABLE_META.get(var)
        if not meta:
            continue

        # 取最新值
        latest = obs_list[-1]
        value = latest["value"]

        # 如果有多个观测点，计算趋势
        trend = None
        if len(obs_list) >= 2:
            values = [o["value"] for o in obs_list]
            if values[-1] > values[0]:
                trend = "rising"
            elif values[-1] < values[0]:
                trend = "falling"

        status = _get_status(value, meta)

        # 构建描述
        direction = meta["direction"]
        threshold = meta["threshold"]
        desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}"

        if direction == "high":
            if status == "danger":
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，超过{threshold}{latest.get('unit', '')}的危险阈值"
            elif status == "warn":
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，接近{threshold}{latest.get('unit', '')}的警戒值"
            else:
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，处于安全范围"
        elif direction == "low":
            if status == "danger":
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，低于{threshold}{latest.get('unit', '')}的危险临界值"
            elif status == "warn":
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，接近{threshold}{latest.get('unit', '')}的警戒值"
            else:
                desc = f"最新{meta['display']}为{value}{latest.get('unit', '')}，处于正常范围"
        elif direction == "suppress":
            if status == "danger":
                desc = f"{meta['display']}仅{value}{latest.get('unit', '')}，无明显抑制效果"
            elif status == "warn":
                desc = (
                    f"{meta['display']}为{value}{latest.get('unit', '')}，抑制效果有限"
                )
            else:
                desc = (
                    f"{meta['display']}达{value}{latest.get('unit', '')}，有效抑制风险"
                )

        if trend == "rising" and direction == "high":
            desc += f"，且呈上升趋势（{obs_list[0]['value']}→{value}）"
        elif trend == "falling" and direction == "low":
            desc += f"，且持续下降（{obs_list[0]['value']}→{value}）"
        elif trend == "rising" and direction == "low":
            desc += f"，但呈上升趋势（{obs_list[0]['value']}→{value}）"

        evidence_list.append(
            {
                "variable": var,
                "display_name": meta["display"],
                "value": value,
                "unit": latest.get("unit", ""),
                "source": latest.get("source", ""),
                "threshold": threshold,
                "direction": direction,
                "status": status,
                "status_text": meta[f"{status}_text"],
                "description": desc,
                "explanation": meta["desc"],
                "weight": meta["weight"],
                "timestamp": latest.get("timestamp", ""),
                "trend": trend,
                "history": [
                    {"value": o["value"], "timestamp": o.get("timestamp", "")}
                    for o in obs_list
                ],
            }
        )

    # 按权重降序排列
    evidence_list.sort(key=lambda x: x["weight"], reverse=True)
    return evidence_list


def _build_reasoning_chain(
    evidence_list: list[dict],
    decision: bool,
    confidence: float,
    category: str,
    difficulty: str,
    risk_score: float,
) -> list[dict[str, str]]:
    """构建逐步推理链条。"""
    chain = []
    step = 1

    category_names = {
        "fire": "森林火险",
        "flood": "洪涝灾害",
        "drought": "干旱",
        "heat": "高温热浪",
    }

    # Step 1: 总述
    chain.append(
        {
            "step": step,
            "type": "intro",
            "text": f"正在分析{category_names.get(category, category)}风险场景（难度等级：{difficulty}）",
        }
    )
    step += 1

    # Steps 2-N: 逐条证据
    danger_count = 0
    for ev in evidence_list:
        if ev["status"] == "danger":
            chain.append(
                {
                    "step": step,
                    "type": "evidence",
                    "text": ev["description"],
                }
            )
            step += 1
            danger_count += 1
        elif ev["status"] == "warn":
            chain.append(
                {
                    "step": step,
                    "type": "evidence",
                    "text": ev["description"],
                }
            )
            step += 1

    # 安全的证据也提一下
    safe_evs = [e for e in evidence_list if e["status"] == "safe"]
    if safe_evs:
        safe_text = "；".join(
            [f"{e['display_name']}={e['value']}{e['unit']}" for e in safe_evs[:3]]
        )
        chain.append(
            {
                "step": step,
                "type": "evidence",
                "text": f"安全因素：{safe_text}，降低了风险评分",
            }
        )
        step += 1

    # 分析步骤
    danger_evs = [e for e in evidence_list if e["status"] == "danger"]
    if danger_evs:
        weights_text = "、".join(
            [f"{e['display_name']}(权重{int(e['weight'] * 100)}%)" for e in danger_evs]
        )
        chain.append(
            {
                "step": step,
                "type": "analysis",
                "text": f"多变量加权综合评分：主要风险因素为{weights_text}，综合风险评分={risk_score:.3f}",
            }
        )
        step += 1

    # 决策步骤
    if decision:
        chain.append(
            {
                "step": step,
                "type": "decision",
                "text": f"综合风险评分{risk_score:.3f}超过预警阈值，AI判定为有风险（置信度{confidence:.0%}），建议启动应急预案",
            }
        )
    else:
        chain.append(
            {
                "step": step,
                "type": "decision",
                "text": f"综合风险评分{risk_score:.3f}低于预警阈值，AI判定为暂时安全（置信度{confidence:.0%}），无需特殊措施",
            }
        )

    return chain


# ===========================================================================
# 验证理由生成
# ===========================================================================

# ground_truth_explanation 键到可读名称的映射
_GT_KEY_MAP: dict[str, tuple[str, float | None, str | None]] = {
    "fwi_mean": ("火险指数(FWI)", 40.0, "high"),
    "humidity_mean": ("空气湿度", 20.0, "low"),
    "wind_mean": ("风速", 12.0, "high"),
    "temperature_mean": ("温度", 35.0, "high"),
    "rainfall_24h": ("24小时降水量", 50.0, "high"),
    "rainfall_6h": ("6小时降水量", 30.0, "high"),
    "soil_moisture": ("土壤含水率", 0.85, "high"),
    "water_level_mean": ("水位", 10.0, "high"),
    "water_trend": ("水位趋势", None, None),
    "spi_mean": ("SPI指数", -1.0, "low"),
    "palmer_mean": ("Palmer指数", -0.5, "low"),
    "rainfall_monthly": ("月降雨量", 30.0, "low"),
    "ndvi_mean": ("植被指数(NDVI)", 0.3, "low"),
    "rainfall_suppression": ("降雨抑制率", None, None),
    "temperature_max_mean": ("最高温度", 35.0, "high"),
    "wet_bulb_temp_mean": ("湿球温度", 27.0, "high"),
    "heat_duration_days": ("持续高温天数", 3.0, "high"),
}


def _build_verification_reason(verification: dict, category: str) -> str:
    """构建验证理由的人类可读文本，解释为什么预警命中或未命中。"""
    if not verification:
        return ""

    predicted = verification.get("predicted", False)
    hit = verification.get("hit", False)
    gt_score = verification.get("ground_truth_score", 0)
    gt_explanation = verification.get("ground_truth_explanation", {})

    cat_names = {
        "fire": "森林火险",
        "flood": "洪涝灾害",
        "drought": "干旱",
        "heat": "高温热浪",
    }
    cat_name = cat_names.get(category, category)

    # 将 ground_truth_explanation 转为可读文本
    gt_parts = []
    for key, val in gt_explanation.items():
        if key not in _GT_KEY_MAP:
            continue
        label, threshold, direction = _GT_KEY_MAP[key]
        if not isinstance(val, (int, float)):
            gt_parts.append(f"{label}={val}")
            continue
        if threshold is None or direction is None:
            gt_parts.append(f"{label}={val}")
            continue
        if direction == "high":
            status = "超过阈值" if val >= threshold else "低于阈值"
            gt_parts.append(f"{label}={val}({status}{threshold})")
        elif direction == "low":
            status = "低于阈值" if val <= threshold else "高于阈值"
            gt_parts.append(f"{label}={val}({status}{threshold})")

    gt_text = "；".join(gt_parts) if gt_parts else "验证数据不足"

    if hit:
        if predicted:
            return (
                f"AI 预测有风险，验证数据确认风险成立。{cat_name}场景的实际观测指标："
                f"{gt_text}。综合风险评分 {gt_score:.3f}，超过预警阈值，预警正确。"
            )
        else:
            return (
                f"AI 预测安全，验证数据确认无风险。{cat_name}场景的实际观测指标："
                f"{gt_text}。综合风险评分 {gt_score:.3f}，低于预警阈值，排除正确。"
            )
    else:
        if predicted:
            return (
                f"AI 预测有风险，但实际观测数据未达危险阈值（误报）。{cat_name}场景的实际指标："
                f"{gt_text}。综合风险评分 {gt_score:.3f}，低于预警阈值，预警过于敏感。"
            )
        else:
            return (
                f"AI 预测安全，但实际观测数据已超过危险阈值（漏报）。{cat_name}场景的实际指标："
                f"{gt_text}。综合风险评分 {gt_score:.3f}，超过预警阈值，存在漏报。"
            )


def _build_hit_reason(predicted: bool, actual: bool, hit: bool, category: str) -> str:
    """为历史记录生成简短的命中理由。"""
    cat_short = {"fire": "火险", "flood": "洪涝", "drought": "干旱", "heat": "高温"}
    cat = cat_short.get(category, category)
    if hit:
        if predicted:
            return f"实际{cat}指标超过危险阈值，AI预警正确"
        else:
            return f"实际{cat}指标处于安全范围，AI排除正确"
    else:
        if predicted:
            return f"实际{cat}指标未达危险阈值，AI预警过于敏感（误报）"
        else:
            return f"实际{cat}指标已超过阈值，AI未预警（漏报）"


def enhance_decision(
    decision: dict[str, Any],
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """增强单个决策记录，添加坐标、详细证据、推理链。"""
    region = decision.get("region", "")
    coords = REGION_COORDS.get(
        region, {"lat": 35.0, "lng": 105.0, "cn": region, "radius_km": 50}
    )

    # 添加坐标和中文名
    decision["coordinates"] = {"lat": coords["lat"], "lng": coords["lng"]}
    decision["region_cn"] = coords["cn"]
    decision["affected_radius_km"] = coords["radius_km"]

    # 使用真实地理边界多边形（如果已定义），否则回退到基于种子的近似多边形
    if region in REGION_POLYGONS:
        decision["polygon"] = REGION_POLYGONS[region]
    else:
        import math as _math

        rng = random.Random(region)
        num_sides = 6 + rng.randint(0, 2)  # 7-9边
        radius_m = coords["radius_km"] * 1000
        lat_range = (radius_m / 111320) * 0.8
        lng_range = (
            radius_m / (111320 * _math.cos(coords["lat"] * _math.pi / 180))
        ) * 0.8
        polygon_points = []
        for i in range(num_sides):
            angle = (i / num_sides) * _math.pi * 2 + (rng.random() - 0.5) * 0.5
            r = 0.6 + rng.random() * 0.4
            lat = coords["lat"] + _math.sin(angle) * lat_range * r
            lng = coords["lng"] + _math.cos(angle) * lng_range * r
            polygon_points.append({"lat": round(lat, 6), "lng": round(lng, 6)})
        decision["polygon"] = polygon_points

    # 如果有原始场景数据，构建详细证据
    if scenario and "observations" in scenario:
        observations = scenario["observations"]

        # 保存原始观测数据
        decision["raw_observations"] = [
            {
                "source": o.get("source", ""),
                "variable": o.get("variable", ""),
                "value": o.get("value", 0),
                "unit": o.get("unit", ""),
                "timestamp": o.get("timestamp", ""),
                "confidence": o.get("confidence", 1.0),
            }
            for o in observations
        ]

        # 构建详细证据
        detailed_evidence = _build_detailed_evidence(
            observations, decision.get("category", "")
        )
        decision["detailed_evidence"] = detailed_evidence

        # 从 rationale 中提取风险评分
        risk_score = 0.0
        rationale = decision.get("rationale", "")
        if rationale:
            try:
                import re

                # Agent rationale 使用中文"风险分=0.xxx"格式
                match = re.search(r"风险分[=:]\s*([\d.]+)", rationale)
                if match:
                    risk_score = float(match.group(1))
            except Exception as e:
                logger.warning("Failed to extract risk_score from rationale: %s", e)
        if risk_score == 0:
            # 从 evidence_summary 推算
            ev_sum = decision.get("evidence_summary", {})
            if ev_sum:
                vals = [v for v in ev_sum.values() if isinstance(v, (int, float))]
                risk_score = sum(vals) / len(vals) if vals else 0

        # 构建推理链
        decision["reasoning_chain"] = _build_reasoning_chain(
            detailed_evidence,
            decision.get("llm_decision", False),
            decision.get("confidence", 0),
            decision.get("category", ""),
            decision.get("difficulty", ""),
            risk_score,
        )

        # 计算 ground truth 验证
        gt_fn = scenario.get("_gt_fn")
        if gt_fn:
            try:
                gt_decision, gt_score, gt_explanation = gt_fn(observations)
                decision["verification"] = {
                    "predicted": decision.get("llm_decision", False),
                    "actual": gt_decision,
                    "hit": decision.get("llm_decision", False) == gt_decision,
                    "ground_truth_score": gt_score,
                    "ground_truth_explanation": gt_explanation,
                }
                decision["verification_reason"] = _build_verification_reason(
                    decision["verification"], decision.get("category", "")
                )
            except Exception as e:
                logger.warning("Failed to build verification for decision: %s", e)

    return decision


# ===========================================================================
# 历史数据生成
# ===========================================================================


def generate_history(
    decisions: list[dict], days: int = 7, output_dir=None, window_days: int = 14
) -> dict[str, Any]:
    """生成历史预警追踪数据。

    今日数据基于 decisions 中的 ground_truth 和 llm_decision 字段。
    过去 N-1 天的数据从已保存的 decisions_YYYYMMDD.json 中加载真实记录，
    无历史文件的天次不生成记录（不伪造数据）。
    """
    history = []
    today = datetime.now(CST)

    # 当前日期的验证结果
    current_alerts = []
    # 修正：confidence 字段是"风险评分"(0=安全,1=极危,阈值0.4)，并非决策把握度，
    # 不能用 confidence>=0.7 来判定是否为警报（低风险日会漏掉全部记录）。
    # 正确口径：以 AI 决策(llm_decision)为准，统计所有被判为风险的场景作为"警报"。
    for d in decisions:
        predicted = bool(d.get("llm_decision", False))
        actual = bool(d.get("ground_truth", False))
        hit = predicted == actual
        cat = d.get("category", "")
        hit_reason = d.get("verification_reason") or _build_hit_reason(
            predicted, actual, hit, cat
        )
        current_alerts.append(
            {
                "region": d.get("region_cn", d.get("region", "")),
                "category": cat,
                "predicted": predicted,
                "actual": actual,
                "hit": hit,
                "hit_reason": hit_reason,
            }
        )

    current_hits = sum(1 for a in current_alerts if a["hit"])
    current_total = len(current_alerts)

    history.append(
        {
            "date": today.strftime("%Y-%m-%d"),
            "date_short": today.strftime("%m-%d"),
            "total": current_total,
            "alerts": sum(1 for a in current_alerts if a["predicted"]),
            "hits": current_hits,
            "misses": current_total - current_hits,
            "accuracy": round(current_hits / current_total, 3) if current_total else 0,
            "is_today": True,
            "details": current_alerts,
        }
    )

    # 从已保存的 decisions_YYYYMMDD.json 中加载真实历史数据
    past_days = _load_history_from_files(output_dir, today, days - 1)
    history.extend(past_days)

    # 按日期降序排列
    history.sort(key=lambda x: x["date"], reverse=True)

    # 真实近期预警（扩展时间范围）：当“今天无预警”时，回看过去 window_days 天里
    # 真实触发过的风险预警。数据来自已保留的 decisions_YYYYMMDD.json
    # （由 daily-report.yml 从 gh-pages 拉回历史文件）。
    recent_alerts = _collect_recent_alerts(output_dir, today, window_days)

    return {
        "days": history,
        "recent_alerts": recent_alerts,
        "recent_window_days": window_days,
    }


def _load_history_from_files(
    output_dir, today: datetime, max_days: int
) -> list[dict[str, Any]]:
    """从已保存的 decisions_YYYYMMDD.json 加载真实历史天次记录。

    只加载 max_days 天内的历史文件；无文件的天次不生成记录。
    """
    past: list[dict[str, Any]] = []
    if not output_dir:
        # 也尝试从仓库根目录加载
        root = Path(__file__).resolve().parent.parent
        candidates = [root]
    else:
        out = Path(output_dir)
        root = Path(__file__).resolve().parent.parent
        candidates = [out, root] if out != root else [out]

    for i in range(1, max_days + 1):
        date = today - timedelta(days=i)
        tag = date.strftime("%Y%m%d")
        found = False
        for base in candidates:
            fp = base / f"decisions_{tag}.json"
            if not fp.exists():
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                decs = data.get("decisions", []) if isinstance(data, dict) else data
                if not decs:
                    continue
                total = len(decs)
                alerts = sum(1 for d in decs if bool(d.get("llm_decision", False)))
                hits = sum(
                    1
                    for d in decs
                    if bool(d.get("llm_decision", False))
                    == bool(d.get("ground_truth", False))
                )
                details = []
                for d in decs:
                    predicted = bool(d.get("llm_decision", False))
                    actual = bool(d.get("ground_truth", False))
                    hit = predicted == actual
                    cat = d.get("category", "")
                    details.append(
                        {
                            "region": d.get("region_cn", d.get("region", "")),
                            "category": cat,
                            "predicted": predicted,
                            "actual": actual,
                            "hit": hit,
                            "hit_reason": d.get("verification_reason")
                            or _build_hit_reason(predicted, actual, hit, cat),
                        }
                    )
                past.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "date_short": date.strftime("%m-%d"),
                        "total": total,
                        "alerts": alerts,
                        "hits": hits,
                        "misses": total - hits,
                        "accuracy": round(hits / total, 3) if total else 0,
                        "is_today": False,
                        "details": details,
                    }
                )
                found = True
                break
            except Exception as e:
                logger.warning("Failed to load history from %s: %s", fp, e)
                continue
        if not found:
            continue
    return past


def _collect_recent_alerts(
    output_dir, today: datetime, window_days: int
) -> list[dict[str, Any]]:
    """扫描已保留的 decisions_YYYYMMDD.json，收集真实近期预警（llm_decision=True）。

    仅用于“今天无预警”时回看历史；今天文件与超出窗口的文件均跳过。

    扫描两个位置（合并去重）：
      1. output_dir（published_reports，CI 每次 fresh checkout + gh-pages force_orphan
         导致通常只保留 1-2 天历史）；
      2. 仓库根目录的 decisions_*.json（已提交到 main，CI checkout 后稳定可用）。
    若无 output_dir 或根目录均无文件，则回退到仅 output_dir。
    """
    alerts: list[dict[str, Any]] = []
    cutoff = today - timedelta(days=window_days)

    # 收集候选目录：output_dir + 仓库根目录（enhance_data.py 的上上级）
    candidates: list[Path] = []
    if output_dir:
        out = Path(output_dir)
        if out.exists():
            candidates.append(out)
    root = Path(__file__).resolve().parent.parent
    if root.exists() and root not in candidates:
        candidates.append(root)

    seen_dates: set[tuple[str, str, str]] = set()
    for base in candidates:
        for fp in base.glob("decisions_*.json"):
            tag = fp.stem.replace("decisions_", "")
            try:
                dt = datetime.strptime(tag, "%Y%m%d")
            except ValueError:
                continue
            if dt.date() >= today.date():
                continue  # 跳过今天（今天已在主表展示）
            if dt < cutoff:
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fp, e)
                continue
            decs = data.get("decisions", []) if isinstance(data, dict) else data
            for d in decs:
                if not bool(d.get("llm_decision", False)):
                    continue
                predicted = True
                actual = bool(d.get("ground_truth", False))
                hit = predicted == actual
                cat = d.get("category", "")
                date_str = dt.strftime("%Y-%m-%d")
                # 同一天同一 region 去重，避免根目录与 output_dir 重复
                key = (date_str, str(d.get("region", "")), str(cat))
                if key in seen_dates:
                    continue
                seen_dates.add(key)
                alerts.append(
                    {
                        "date": date_str,
                        "region": d.get("region_cn", d.get("region", "")),
                        "category": cat,
                        "predicted": predicted,
                        "actual": actual,
                        "hit": hit,
                        "hit_reason": d.get("verification_reason")
                        or _build_hit_reason(predicted, actual, hit, cat),
                    }
                )
    alerts.sort(key=lambda x: x["date"], reverse=True)
    return alerts[:20]


# ===========================================================================
# 主入口
# ===========================================================================


def enhance_latest_json(json_path: str | Path, suite: list[dict] | None = None) -> None:
    """增强 latest.json，添加坐标、详细证据、推理链，并生成 history.json。"""
    json_path = Path(json_path)
    output_dir = json_path.parent

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 如果没有传入场景数据，尝试加载
    if suite is None:
        try:
            from earthbench.scenarios import get_alert_benchmark_suite

            suite = get_alert_benchmark_suite()
        except Exception:
            suite = []

    # 构建 case_id -> scenario 的映射
    scenario_map = {s["case_id"]: s for s in suite} if suite else {}

    # 增强每个决策
    for decision in data.get("decisions", []):
        case_id = decision.get("case_id", "")
        scenario = scenario_map.get(case_id)
        enhance_decision(decision, scenario)

    # 写回增强后的 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 生成历史数据
    history = generate_history(
        data.get("decisions", []), output_dir=output_dir, window_days=14
    )
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Enhanced {json_path.name} with {len(data.get('decisions', []))} decisions")
    print(f"Generated {history_path.name} with {len(history['days'])} days of history")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    latest_path = project_root / "published_reports" / "latest.json"
    if latest_path.exists():
        enhance_latest_json(latest_path)
    else:
        print(f"File not found: {latest_path}")
