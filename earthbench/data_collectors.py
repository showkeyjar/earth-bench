"""真实气象数据采集器 — 对接和风天气 QWeather API。

取代 scenarios.py 中的硬编码模拟数据，为每个监测区域采集实时气象观测值。
如果 API 调用失败，回退到 AlertBench 内置场景作为 fallback。

配置方式：
    export QWEATHER_API_KEY="your_api_key"
    export QWEATHER_HOST="na2tupd7ah.re.qweatherapi.com"  # 开发者 ID 专属域名
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger("earthbench.data_collectors")


# ===========================================================================
# 和风天气配置（通过环境变量或硬编码）
# ===========================================================================

QWEATHER_API_KEY = os.environ.get("QWEATHER_API_KEY", "")
QWEATHER_HOST = os.environ.get(
    "QWEATHER_HOST",
    "na2tupd7ah.re.qweatherapi.com",  # 开发者 ID: Q980449E05
)

# NASA FIRMS MAP_KEY（卫星热异常检测，需免费注册）
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")

# 和风天气城市编码映射表
REGION_LOCATION_MAP: dict[str, dict[str, Any]] = {
    # === Fire ===
    "Xiangshan-Beijing": {"location_id": "101010100", "name": "北京"},
    "WestLake-Hangzhou": {"location_id": "101210101", "name": "杭州"},
    "Shenzhen-Coast": {"location_id": "101280601", "name": "深圳"},
    "Wuhan-Yangtze": {"lon": 114.31, "lat": 30.52, "name": "武汉"},
    "Guangzhou-PearlR": {"location_id": "101280101", "name": "广州"},
    "Guilin-Guangxi": {"location_id": "101250701", "name": "桂林"},
    "Nanjing-Yangtze": {"location_id": "101190101", "name": "南京"},
    "Nanjing-OvenCity": {"location_id": "101190101", "name": "南京"},
    "Kunming-Yunnan": {"location_id": "101290101", "name": "昆明"},
    "Kunming-SpringCity": {"location_id": "101290101", "name": "昆明"},
    "Hangzhou-Zhejiang": {"location_id": "101210101", "name": "杭州"},
    "Chongqing-HotPotato": {"location_id": "101040100", "name": "重庆"},
    "Taiyuan-Shanxi": {"location_id": "101170201", "name": "太原"},
    "Lhasa-Tibet": {"location_id": "101260201", "name": "拉萨"},
    "Harbin-Heilongjiang": {"location_id": "101050101", "name": "哈尔滨"},
    "GreaterKhingan": {"lon": 124.39, "lat": 51.17, "name": "大兴安岭"},
    "Urumqi-Xinjiang": {"location_id": "101130101", "name": "乌鲁木齐"},
    # === Drought / Heat ===
    "ChangbaiMountain": {"lon": 128.08, "lat": 42.02, "name": "长白山"},
}


# ===========================================================================
# HTTP 请求工具（支持 gzip 解压）
# ===========================================================================


def qweather_request(path: str, params: dict[str, str]) -> dict[str, Any] | None:
    """向和风天气 API 发起请求，返回 JSON 或 None（失败时）。

    响应可能经过 gzip 压缩，需要自动解压。
    """
    url = f"https://{QWEATHER_HOST}{path}?key={QWEATHER_API_KEY}"
    if params:
        url += "&" + "&".join(f"{k}={v}" for k, v in params.items())

    try:
        req = Request(url, headers={"User-Agent": "EarthBench/1.0"})
        with urlopen(req, timeout=15) as resp:
            raw_data = resp.read()

            # 处理 gzip 压缩响应
            if raw_data[:2] == b"\x1f\x8b":  # gzip magic number
                raw_data = gzip.decompress(raw_data)

            data = json.loads(raw_data.decode("utf-8"))
            if data.get("code") == "200":
                return data
            else:
                logger.warning(
                    f"QWeather API error: code={data.get('code')}, msg={data.get('msg')}"
                )
                return None
    except URLError as e:
        logger.warning(f"QWeather request failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"QWeather unexpected error: {e}")
        return None


# ===========================================================================
# 实时天气采集
# ===========================================================================


def fetch_realtime_weather(
    location_id: str | None, lon: float | None, lat: float | None
) -> dict[str, Any]:
    """获取实时天气数据。

    Args:
        location_id: 和风天气城市编码
        lon: 经度（备用定位方式）
        lat: 纬度（备用定位方式）

    Returns:
        包含温度、湿度、风速、降水等字段的字典，失败返回空字典
    """
    location = location_id or f"{lon},{lat}"
    data = qweather_request("/v7/weather/now", {"location": location})

    if not data:
        return {}

    now = data.get("now", {})
    return {
        "temp": float(now.get("temp", 0)),
        "humidity": int(now.get("humidity", 50)),
        "wind_speed_level": int(now.get("windScale", "0")),  # 风力等级 0-12
        "wind_speed_ms": float(now.get("windSpeed", "0")) / 3.6,  # km/h → m/s
        "wind_dir": now.get("windDir", ""),
        "precip_1h": float(now.get("precip", 0)),  # 过去1小时降水量 mm
        "pressure": int(now.get("pressure", 1000)),
        "vis": float(now.get("vis", 10)),
        "text": now.get("text", ""),
        "cloud": int(now.get("cloud", 0)),
        "dew": float(now.get("dew", 0)),
        "obsTime": now.get("obsTime", ""),
    }


def fetch_hourly_forecast(location: str, hours: int = 24) -> list[dict]:
    """获取逐小时天气预报。

    Args:
        location: location_id 或 "lon,lat" 经纬度字符串
        hours: 返回小时数（自动选择 24h/72h/168h 端点）
    """
    if hours <= 24:
        endpoint = "/v7/weather/24h"
    elif hours <= 72:
        endpoint = "/v7/weather/72h"
    else:
        endpoint = "/v7/weather/168h"

    data = qweather_request(endpoint, {"location": location, "lang": "zh"})
    if not data:
        return []

    hourly = data.get("hourly", [])[:hours]
    return [
        {
            "fxTime": h.get("fxTime", ""),
            "temp": float(h.get("temp", "0")),
            "humidity": int(h.get("humidity", "0")),
            "windSpeed": float(h.get("windSpeed", "0")),
            "windDir": h.get("windDir", ""),
            "precip": float(h.get("precip", "0")),
            "weather": h.get("text", ""),
        }
        for h in hourly
    ]


def fetch_daily_forecast(location: str, days: int = 3) -> list[dict]:
    """获取每日天气预报（高低温、降水、风力）。

    Args:
        location: location_id 或 "lon,lat" 经纬度字符串
        days: 返回天数（自动选择 3d/7d/15d 端点）
    """
    if days <= 3:
        endpoint = "/v7/weather/3d"
    elif days <= 7:
        endpoint = "/v7/weather/7d"
    else:
        endpoint = "/v7/weather/15d"

    data = qweather_request(endpoint, {"location": location, "lang": "zh"})
    if not data:
        return []

    daily = data.get("daily", [])[:days]
    return [
        {
            "date": d.get("date", ""),
            "tempMax": float(d.get("tempMax", "0")),
            "tempMin": float(d.get("tempMin", "0")),
            "humidity": int(d.get("humidity", "0")),
            "windSpeed": float(d.get("windSpeed", "0")),
            "precip": float(d.get("precip", "0")),
            "weather": d.get("text", ""),
        }
        for d in daily
    ]


# ===========================================================================
# FWI 森林火险指数计算（简化版）
# ===========================================================================

# 区域类型映射（用于水体修正）— 这些区域有大型水体/城区，需要降低火险指数
WATER_BODY_REGIONS = {
    "WestLake-Hangzhou": 0.3,  # 西湖大型水体，火险大幅降低
    "Wuhan-Yangtze": 0.5,  # 长江区域，有一定水体调节
    "Guangzhou-PearlR": 0.5,  # 珠江区域
    "Nanjing-Yangtze": 0.5,  # 长江区域
    "Shenzhen-Coast": 0.4,  # 沿海区域
    "Xiangshan-Beijing": 1.0,  # 北京香山，无水体修正（正常林地）
}


def calculate_fwi_from_weather(realtime: dict, region_key: str | None = None) -> float:
    """根据实时天气数据估算 FFMC/FWI。

    参考: https://en.wikipedia.org/wiki/Forest_Fire_Weather_Index

    这是一个简化模型，真实 FWI 需要复杂的 6 个因子计算。
    此处根据温度、湿度、风速、降水进行加权估算，并考虑区域水体修正。
    """
    temp = realtime.get("temp", 20)
    hum = realtime.get("humidity", 50)
    wind_ms = realtime.get("wind_speed_ms", 3)  # m/s
    precip = realtime.get("precip_1h", 0)

    # FFMC (Fine Fuel Moisture Code) — 细燃料干燥码
    # 湿度越低、温度越高、风速越大 → FFMC 越高
    base_ffmc = 90 * (1 - hum / 100) + temp * 0.3 + wind_ms * 2.0
    base_ffmc = max(0, min(101, base_ffmc))

    # DMC (Duff Moisture Code) — 枯枝潮湿码 (简化)
    base_dmc = max(0, 30 - precip * 0.5 + temp * 0.2 - hum * 0.1)

    # ISI (Initial Spread Index) — 初始蔓延指数
    base_isi = base_ffmc * wind_ms * 0.3

    # FWI (Fire Weather Index)
    fwi = math.sqrt(base_isi * base_dmc) if base_dmc > 0 else base_isi
    fwi = max(0.0, min(100.0, fwi))

    # ==================== 区域水体修正 ====================
    # 对于有大型水体的区域（如杭州西湖），根据水体调节效应降低 FWI
    # 这是因为城区内有大面积水体时，实际火险远低于裸露林地
    if region_key:
        correction_factor = WATER_BODY_REGIONS.get(region_key, 1.0)
        original_fwi = fwi
        # 应用非线性修正：对高FWI值显著降低，对低值影响较小
        if fwi > 40:
            fwi = fwi * correction_factor
            # 水体修正后最低保留 60% 原值，避免过度修正
            fwi = max(fwi, original_fwi * 0.6)
        elif fwi > 20:
            # 中度影响
            fwi = fwi * (0.5 + 0.5 * correction_factor)
            fwi = max(fwi, original_fwi * 0.6)
        # correction_factor == 1.0 时不做任何修正（如北京香山）

    # ==================== 结束区域修正 ====================

    return round(fwi, 1)


# ===========================================================================
# 转换为场景观测列表
# ===========================================================================


def weather_to_observations(
    realtime: dict,
    hourly: list[dict],
    daily: list[dict],
    category: str,
    region_key: str,
    region_name: str,
) -> list[dict]:
    """将和风天气 API 返回的数据转换为 publish_pipeline 需要的 observations 格式。

    Args:
        region_key: 区域键值（如 WestLake-Hangzhou），用于 FWI 水体修正
    """
    obs_list = []
    now_time = realtime.get(
        "obsTime", datetime.now(timezone(timedelta(hours=8))).isoformat()
    )

    # --- Fire: FWI + humidity + wind + temperature ---
    if "fire" in category:
        # 温度
        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "temperature",
                "value": realtime.get("temp", 0),
                "unit": "°C",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

        # 湿度
        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "humidity",
                "value": realtime.get("humidity", 50),
                "unit": "%",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

        # 风速 (m/s)
        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "wind_speed",
                "value": round(realtime.get("wind_speed_ms", 0), 1),
                "unit": "m/s",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

        # FWI 估算 — 传入 region_key 以便进行水体修正
        fwi = calculate_fwi_from_weather(realtime, region_key)
        obs_list.append(
            {
                "source": "QWeather/Calculated",
                "variable": "FWI",
                "value": fwi,
                "unit": "",
                "timestamp": now_time,
                "confidence": 0.90,
            }
        )

        # 添加历史 FWI 作为趋势参考（用前一条观测）
        # 注意：QWeather 实时 API 不提供历史 FWI，此处仅用当天数据
        # 如果需要趋势分析，应通过 enhance_data 从历史 decisions_*.json 中获取

    # --- Flood: 降雨量 + 水位 ---
    elif "flood" in category:
        today_precip = daily[0].get("precip", 0) if daily else 0
        recent_precip = realtime.get("precip_1h", 0)

        # 用 hourly 预报更准确地估算 6h 和 24h 降雨量
        if hourly:
            hourly_precips = [h.get("precip", 0) for h in hourly[:24]]
            total_24h = sum(hourly_precips)
            total_6h = (
                sum(hourly_precips[:6])
                if len(hourly_precips) >= 6
                else sum(hourly_precips)
            )
        else:
            # 降级：用实时降水 + daily 降水估算
            total_24h = max(today_precip, recent_precip * 24)
            total_6h = recent_precip * 6

        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "rainfall_24h",
                "value": round(total_24h, 1),
                "unit": "mm",
                "timestamp": now_time,
                "confidence": 0.92,
            }
        )

        obs_list.append(
            {
                "source": "QWeather/Hourly",
                "variable": "rainfall_6h",
                "value": round(total_6h, 1),
                "unit": "mm",
                "timestamp": now_time,
                "confidence": 0.90,
            }
        )

        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "temperature",
                "value": realtime.get("temp", 0),
                "unit": "°C",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

    # --- Drought: SPI/Palmer + 湿度 + 降雨 ---
    elif "drought" in category:
        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "humidity",
                "value": realtime.get("humidity", 50),
                "unit": "%",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

        # 用 daily 预报多日累计降水量来估算月降雨量
        # daily API 返回未来最多 7 天的预报，累加后按 30 天外推
        if daily:
            daily_total = sum(d.get("precip", 0) for d in daily[:7])
            # 将 7 天累计外推到 30 天估算
            monthly_est = round(daily_total / 7.0 * 30.0, 1)
        else:
            monthly_est = 0.0

        obs_list.append(
            {
                "source": "QWeather/Daily7d",
                "variable": "rainfall_monthly",
                "value": monthly_est,
                "unit": "mm",
                "timestamp": now_time,
                "confidence": 0.80,
            }
        )

        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "temperature",
                "value": realtime.get("temp", 0),
                "unit": "°C",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

    # --- Heat: 温度 + 湿度 + 湿球温度 ---
    elif "heat" in category:
        temp = realtime.get("temp", 0)
        hum = realtime.get("humidity", 50)

        # 从 daily 预报中取最高温（今天+未来几天），用最高值作为 temperature_max
        if daily:
            temp_max_values = [d.get("tempMax", temp) for d in daily[:3]]
            # 同时统计连续高温天数（持续 >= 35°C 的天数）
            heat_days = sum(1 for t in temp_max_values if float(t) >= 35.0)
        else:
            temp_max_values = [temp]
            heat_days = 1

        avg_temp_max = sum(float(t) for t in temp_max_values) / len(temp_max_values)

        obs_list.append(
            {
                "source": "QWeather/Daily3d",
                "variable": "temperature_max",
                "value": round(avg_temp_max, 1),
                "unit": "°C",
                "timestamp": now_time,
                "confidence": 0.92,
            }
        )

        # 湿球温度估算 — 使用 Stull (2011) 近似公式
        # Tw = T * atan(0.151977 * (RH + 8.313659)^0.5) + atan(T + RH)
        #       - atan(RH - 1.676331) + 0.00391838 * RH^1.5 * atan(0.023101 * RH) - 4.686035
        # 参考: Stull, R. (2011), "Wet-Bulb Temperature from Relative Humidity and Air Temperature"
        # 该公式在 -20°C~50°C、5%~99% RH 范围内误差 < 1°C
        import math as _m

        rh = hum
        tw = (
            temp * _m.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + _m.atan(temp + rh)
            - _m.atan(rh - 1.676331)
            + 0.00391838 * rh**1.5 * _m.atan(0.023101 * rh)
            - 4.686035
        )
        wet_bulb = round(tw, 1)
        obs_list.append(
            {
                "source": "QWeather/Calculated",
                "variable": "wet_bulb_temp",
                "value": round(wet_bulb, 1),
                "unit": "°C",
                "timestamp": now_time,
                "confidence": 0.85,
            }
        )

        obs_list.append(
            {
                "source": "QWeather/Realtime",
                "variable": "humidity",
                "value": hum,
                "unit": "%",
                "timestamp": now_time,
                "confidence": 0.95,
            }
        )

        # 连续高温天数作为热浪持续时间依据
        if daily:
            obs_list.append(
                {
                    "source": "QWeather/Daily3d",
                    "variable": "heat_duration_days",
                    "value": heat_days,
                    "unit": "days",
                    "timestamp": now_time,
                    "confidence": 0.85,
                }
            )

    return obs_list


# ===========================================================================
# 主入口：为单个场景采集气象数据
# ===========================================================================


def collect_region_weather(region_key: str, category: str) -> list[dict]:
    """为一个监测区域采集气象观测数据。

    如果和风天气 API 成功，返回真实观测；
    如果失败，返回空列表（由调用方 fallback 到 AlertBench 场景）。
    """
    loc_info = REGION_LOCATION_MAP.get(region_key)
    if not loc_info:
        logger.warning(f"Unknown region key: {region_key}, skipping")
        return []

    location_id = loc_info.get("location_id")
    lon = loc_info.get("lon")
    lat = loc_info.get("lat")

    if not location_id and (not lon or not lat):
        logger.warning(f"No valid location for {region_key}: {loc_info}")
        return []

    # 采集实时天气
    realtime = fetch_realtime_weather(location_id, lon, lat)
    if not realtime:
        logger.warning(
            f"Failed to fetch realtime weather for {loc_info.get('name', region_key)}"
        )
        return []

    # 采集逐小时和每日预报（location_id 和经纬度均可）
    hourly = []
    daily = []
    location_str = location_id or (f"{lon},{lat}" if lon and lat else "")
    if location_str:
        hourly = fetch_hourly_forecast(location_str, hours=24)
        daily = fetch_daily_forecast(location_str, days=7)

    # 转换为场景观测列表 — 传递 region_key 以便 FWI 计算时进行水体修正
    obs_list = weather_to_observations(
        realtime=realtime,
        hourly=hourly,
        daily=daily,
        category=category,
        region_key=region_key,
        region_name=loc_info.get("name", region_key),
    )

    logger.info(
        f"[{loc_info['name']}] Collected {len(obs_list)} real observations: "
        f"temp={realtime['temp']}°C, hum={realtime['humidity']}%, "
        f"wind={realtime.get('wind_speed_ms', 0):.1f}m/s, precip={realtime.get('precip_1h', 0)}mm"
    )

    return obs_list


# ===========================================================================
# Fallback: 使用 AlertBench 模拟数据
# ===========================================================================


def fallback_to_scenarios(
    suite: list[dict], category_filter: Optional[list[str]] = None
) -> list[dict]:
    """当 QWeather API 不可用时，回退到 AlertBench 内置场景。"""
    results = []
    for item in suite:
        cat = item.get("category", "")
        if category_filter and cat not in category_filter:
            continue

        scenario_copy = item.copy()
        scenario_copy["_data_source"] = "fallback_alertbench"
        results.append(scenario_copy)

    logger.warning(
        f"Fallback: using {len(results)} AlertBench scenarios (QWeather API unavailable)"
    )
    return results


# ===========================================================================
# NASA FIRMS 卫星热异常检测（追加）
# ===========================================================================


def fetch_firms_hotspots(lat: float, lon: float, radius_km: int = 50) -> list[dict]:
    """使用 NASA FIRMS API 查询指定区域内近7天的卫星热异常（火点）数据。"""
    if not FIRMS_MAP_KEY:
        return []

    lat_range = radius_km / 111.0
    lng_range = radius_km / (111.0 * math.cos(math.radians(lat)))

    west = round(lon - lng_range, 2)
    south = round(lat - lat_range, 2)
    east = round(lon + lng_range, 2)
    north = round(lat + lat_range, 2)
    area_str = f"{west},{south},{east},{north}"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{area_str}/7"

    try:
        req = Request(url, headers={"User-Agent": "EarthBench/1.0"})
        with urlopen(req, timeout=30) as resp:
            raw_data = resp.read()
            if raw_data[:2] == b"\x1f\x8b":
                raw_data = gzip.decompress(raw_data)

            text = raw_data.decode("utf-8-sig")
            lines = text.strip().split("\n")

            if len(lines) < 2:
                return []

            header = lines[0].strip().split(",")
            hotspots = []
            for line in lines[1:]:
                fields = line.strip().split(",")
                if len(fields) < len(header):
                    continue
                row = dict(zip(header, fields))
                try:
                    hotspots.append(
                        {
                            "latitude": float(row.get("latitude", 0)),
                            "longitude": float(row.get("longitude", 0)),
                            "confidence_level": row.get("confidence", "unknown"),
                            "brightness_ti4": float(row.get("bright_ti4", 0)),
                            "frp": float(row.get("frp", 0)),
                            "satellite": row.get("satellite", ""),
                            "instrument": row.get("instrument", ""),
                            "acq_date": row.get("acq_date", ""),
                            "daynight": row.get("daynight", "N"),
                        }
                    )
                except (ValueError, KeyError):
                    continue

            return hotspots
    except Exception as e:
        logger.warning(f"FIRMS query failed: {e}")
        return []


def has_active_fire(latitude: float, longitude: float, radius_km: int = 20) -> bool:
    """检查指定区域内是否有活跃火点。"""
    hotspots = fetch_firms_hotspots(latitude, longitude, radius_km)
    significant = [
        h
        for h in hotspots
        if float(h.get("frp", 0)) > 0.5 or h.get("confidence_level") == "high"
    ]

    if significant:
        frp_total = sum(h["frp"] for h in significant)
        logger.info(
            f"[FIRMS] Found {len(significant)} active fire(s) within {radius_km}km, "
            f"total FRP={frp_total:.1f} MW/m2"
        )
        return True

    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Testing QWeather data collection")
    print("=" * 60)

    print("\n--- Fetching Beijing realtime ---")
    bj_weather = fetch_realtime_weather("101010100", None, None)
    print(json.dumps(bj_weather, indent=2, ensure_ascii=False))

    print("\n--- Fetching Beijing 24h forecast ---")
    bj_hourly = fetch_hourly_forecast("101010100", hours=6)
    print(json.dumps(bj_hourly, indent=2, ensure_ascii=False))

    print("\n--- Converting to fire scenario observations ---")
    fire_obs = weather_to_observations(
        realtime=bj_weather,
        hourly=bj_hourly,
        daily=[],
        category="fire",
        region_key="Xiangshan-Beijing",
        region_name="北京",
    )
    print(json.dumps(fire_obs, indent=2, ensure_ascii=False))

    print("\n--- Testing FIRMS hotspot detection ---")
    hotspots = fetch_firms_hotspots(39.99, 116.16, radius_km=50)
    print(f"Found {len(hotspots)} hotspots")
