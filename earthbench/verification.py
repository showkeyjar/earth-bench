"""EarthBench 闭环验证模块 -- 用网上真实数据验证历史 AI 预测。

核心流程:
  1. 读取 N 天前的 decisions_YYYYMMDD.json (AI 当时做了什么预测)
  2. 从独立数据源获取该日的真实数据:
     - QWeather /v7/historical/weather (历史天气)
     - NASA FIRMS 卫星火点 (真实火灾)
  3. 用真实数据判断预测是否命中
  4. 累积验证结果 -> accuracy_trend.json (precision/recall/f1)
  5. 前端展示精度趋势

与 scenarios.py 中 infer_*_ground_truth 的关键区别:
  - infer_*_ground_truth: 基于同源 QWeather 实时数据 + 规则推断 (训练用)
  - verification: 基于独立的 QWeather 历史天气 API + NASA FIRMS (验证用)
  虽然历史天气和实时天气都来自 QWeather, 但历史天气 API (/v7/historical/weather)
  是独立的数据接口, 数据在 T+2 后才完全稳定, 能避免实时接口的延迟和修正。
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from earthbench.data_collectors import (
    QWEATHER_API_KEY,
    fetch_firms_hotspots,
    qweather_request,
)
from earthbench.enhance_data import REGION_COORDS

logger = logging.getLogger("earthbench.verification")

CST = timezone(timedelta(hours=8))


# ============================================================================
# 工具函数
# ============================================================================


def _wet_bulb(temp_c: float, rh: float) -> float:
    """Stull 2011 湿球温度近似公式。

    Tw = T * atan(0.151977 * sqrt(RH + 8.313659))
         + atan(T + RH) - atan(RH - 1.676331)
         + 0.00391838 * RH^1.5 * atan(0.023101 * RH) - 4.686035
    """
    t = float(temp_c)
    rh = float(rh)
    tw = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return round(tw, 2)


def _date_str(d: datetime) -> str:
    """格式化为 YYYYMMDD 字符串。"""
    return d.strftime("%Y%m%d")


def _iso_date(d: datetime) -> str:
    """格式化为 ISO 日期字符串 YYYY-MM-DD。"""
    return d.strftime("%Y-%m-%d")


# ============================================================================
# 独立数据源获取
# ============================================================================


def fetch_historical_weather(location_id: str, date_str: str) -> dict[str, Any]:
    """从 QWeather 历史天气 API 获取指定日期的真实天气数据。

    使用 /v7/historical/weather 接口 (与实时天气 API 不同的端点)。
    需要 QWEATHER_API_KEY。
    """
    if not QWEATHER_API_KEY:
        return {"error": "QWEATHER_API_KEY not set"}

    # QWeather 历史天气 API: /v7/historical/weather
    # 参数: location=城市ID或坐标, date=YYYYMMDD
    params = {"location": location_id, "date": date_str}
    result = qweather_request("/v7/historical/weather", params)

    if result is None:
        return {"error": "API request failed"}

    # 统一解析数据结构 (QWeather 返回 weatherDaily/weatherHourly)
    weather_daily = result.get("weatherDaily", {})
    weather_hourly = result.get("weatherHourly", [])

    # 转换为统一格式 {daily: [...], hourly: [...]}
    return {
        "daily": [weather_daily] if weather_daily else [],
        "hourly": weather_hourly,
    }


def fetch_firms_fire_data(region_id: str) -> list[dict]:
    """获取 NASA FIRMS 卫星火点数据 (复用 data_collectors 中的函数)。

    返回该区域附近的近7天火点列表。
    """
    coords = REGION_COORDS.get(region_id)
    if not coords:
        return []

    lat = float(coords["lat"])
    lng = float(coords["lng"])
    radius = int(coords.get("radius_km", 30))

    return fetch_firms_hotspots(lat, lng, radius_km=radius)


# ============================================================================
# 灾种验证器 -- 用真实数据判断预测是否命中
# ============================================================================


def verify_fire(prediction: dict[str, Any], region_id: str) -> dict[str, Any]:
    """验证火灾预测: 用 NASA FIRMS 卫星火点数据验证。

    如果 FIRMS 在该区域检测到火点 -> 实际发生火灾
    否则 -> 无火灾
    """
    firms_data = fetch_firms_fire_data(region_id)

    fire_detected = len(firms_data) > 0

    evidence = (
        f"NASA FIRMS 卫星检测到 {len(firms_data)} 个火点"
        if fire_detected
        else "NASA FIRMS 卫星未检测到火点"
    )

    return {
        "actual": fire_detected,
        "verification_source": "NASA FIRMS",
        "verification_evidence": evidence,
        "verification_status": "verified",
        "fire_count": len(firms_data),
    }


def verify_flood(
    prediction: dict[str, Any], hist_weather: dict[str, Any]
) -> dict[str, Any]:
    """验证洪水预测: 用 QWeather 历史日降水量验证。

    日降水量 >= 50mm (暴雨级别) -> 可能发生洪水
    """
    if hist_weather.get("error"):
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": hist_weather.get("error", "数据不可用"),
            "verification_status": "insufficient_data",
        }

    daily_list = hist_weather.get("daily", [])
    hourly_list = hist_weather.get("hourly", [])

    if not daily_list:
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": "历史天气数据为空",
            "verification_status": "insufficient_data",
        }

    daily = daily_list[0]
    precip = float(daily.get("precip", 0))

    # 日降水量 >= 50mm 判定为洪水风险
    flood_actual = precip >= 50.0

    # hourly 降水明细 (如果有)
    hourly_precip = (
        sum(float(h.get("precip", 0)) for h in hourly_list) if hourly_list else 0
    )
    max_hourly = (
        max(float(h.get("precip", 0)) for h in hourly_list) if hourly_list else 0
    )

    evidence = (
        f"日降水量={precip:.1f}mm, 小时累计={hourly_precip:.1f}mm, "
        f"最大小时降水={max_hourly:.1f}mm"
    )

    return {
        "actual": flood_actual,
        "verification_source": "QWeather Historical",
        "verification_evidence": evidence,
        "verification_status": "verified",
        "precip_mm": precip,
    }


def verify_drought(
    prediction: dict[str, Any], hist_weather: dict[str, Any]
) -> dict[str, Any]:
    """验证干旱预测: 用 QWeather 历史降水量验证。

    近30天累计降水量 < 20mm -> 干旱条件
    (单日数据不足, 但可以用日降水量=0 作为辅助判断)
    """
    if hist_weather.get("error"):
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": hist_weather.get("error", "数据不可用"),
            "verification_status": "insufficient_data",
        }

    daily_list = hist_weather.get("daily", [])

    if not daily_list:
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": "历史天气数据为空",
            "verification_status": "insufficient_data",
        }

    daily = daily_list[0]
    precip = float(daily.get("precip", 0))
    humidity = float(daily.get("humidity", 50))

    # 日降水=0 且湿度低 -> 干旱条件辅助判断
    drought_actual = precip < 0.1 and humidity < 50

    evidence = f"日降水量={precip:.1f}mm, 湿度={humidity:.0f}%"

    return {
        "actual": drought_actual,
        "verification_source": "QWeather Historical",
        "verification_evidence": evidence,
        "verification_status": "verified",
        "precip_mm": precip,
        "humidity": humidity,
    }


def verify_heat(
    prediction: dict[str, Any], hist_weather: dict[str, Any]
) -> dict[str, Any]:
    """验证热浪预测: 用 QWeather 历史最高温和湿球温度验证。

    日最高温 >= 35degC 或湿球温度 >= 28degC -> 热浪条件
    """
    if hist_weather.get("error"):
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": hist_weather.get("error", "数据不可用"),
            "verification_status": "insufficient_data",
        }

    daily_list = hist_weather.get("daily", [])

    if not daily_list:
        return {
            "actual": None,
            "verification_source": "QWeather Historical",
            "verification_evidence": "历史天气数据为空",
            "verification_status": "insufficient_data",
        }

    daily = daily_list[0]
    temp_max = float(daily.get("tempMax", 0))
    temp_min = float(daily.get("tempMin", 0))
    humidity_avg = float(daily.get("humidity", 50))

    # 湿球温度估算 (Stull 2011 近似)
    tw = _wet_bulb(temp_max, humidity_avg)

    is_heat = temp_max >= 35.0 or tw >= 28.0

    evidence = (
        f"日最高温={temp_max:.1f}degC, 最低温={temp_min:.1f}degC, "
        f"湿度={humidity_avg:.0f}%, 湿球温度={tw:.1f}degC"
    )

    return {
        "actual": is_heat,
        "verification_source": "QWeather Historical",
        "verification_evidence": evidence,
        "verification_status": "verified",
        "temp_max": temp_max,
        "wet_bulb": tw,
    }


# ============================================================================
# 单条预测验证
# ============================================================================


def verify_prediction(
    prediction: dict[str, Any], hist_weather: dict[str, Any] | None = None
) -> dict[str, Any]:
    """验证单条预测, 返回验证结果。

    Args:
        prediction: 决策记录, 包含 category, region_id, llm_decision 等
        hist_weather: 该区域该日的历史天气数据 (可选, 火灾验证不需要)

    Returns:
        验证结果 dict, 包含:
        - hit: 预测是否命中 (True/False/None=无法验证)
        - actual: 实际是否发生灾害
        - predicted: 预测是否报警
        - source: 验证数据来源
        - evidence: 验证证据描述
        - status: verified / insufficient_data
    """
    category = prediction.get("category", "")
    region_id = prediction.get("region_id", "") or prediction.get("region", "")
    predicted = bool(prediction.get("llm_decision", False))

    result: dict[str, Any] = {
        "category": category,
        "region_id": region_id,
        "region_name": prediction.get("region_name", "")
        or prediction.get("region_cn", "")
        or region_id,
        "predicted": predicted,
        "hit": None,
        "actual": None,
        "verification_source": "",
        "verification_evidence": "",
        "verification_status": "skipped",
    }

    if category == "fire":
        # 火灾: 用 NASA FIRMS 卫星火点验证
        verif = verify_fire(prediction, region_id)
    elif category == "flood":
        verif = verify_flood(prediction, hist_weather or {})
    elif category == "drought":
        verif = verify_drought(prediction, hist_weather or {})
    elif category == "heat":
        verif = verify_heat(prediction, hist_weather or {})
    else:
        result["verification_evidence"] = f"未知灾种: {category}"
        return result

    result["actual"] = verif.get("actual")
    result["verification_source"] = verif.get("verification_source", "")
    result["verification_evidence"] = verif.get("verification_evidence", "")
    result["verification_status"] = verif.get("verification_status", "unknown")
    result["verification_details"] = {
        k: v
        for k, v in verif.items()
        if k
        not in (
            "actual",
            "verification_source",
            "verification_evidence",
            "verification_status",
        )
    }

    # 只有验证成功时才计算 hit
    if result["verification_status"] == "verified" and result["actual"] is not None:
        result["hit"] = predicted == result["actual"]
        # 如果预测报警且实际发生 -> 命中 (tp)
        # 如果预测不报警且实际不发生 -> 命中 (tn)
        # 如果预测报警但实际不发生 -> 误报 (fp)
        # 如果预测不报警但实际发生 -> 漏报 (fn)

    return result


# ============================================================================
# 批量验证 -- 验证某日所有预测
# ============================================================================


def verify_date_predictions(
    target_date: datetime,
    output_dir: Path,
) -> dict[str, Any]:
    """验证指定日期的所有预测, 输出 verification_YYYYMMDD.json。

    Args:
        target_date: 要验证的日期 (北京时间)
        output_dir: 输出目录 (published_reports/)

    Returns:
        验证结果汇总
    """
    date_str = _date_str(target_date)
    iso_date = _iso_date(target_date)

    # 读取该日的决策文件
    decisions_file = output_dir / f"decisions_{date_str}.json"
    if not decisions_file.exists():
        logger.info(f"No decisions file for {date_str}, skipping verification")
        return {
            "date": iso_date,
            "status": "no_data",
            "total": 0,
            "verified": 0,
            "accuracy": None,
        }

    with open(decisions_file, encoding="utf-8") as f:
        decisions_data = json.load(f)

    decisions = decisions_data.get("decisions", [])

    # 获取每个区域的 location_id
    from earthbench.data_collectors import REGION_LOCATION_MAP

    verifications: list[dict[str, Any]] = []
    verified_count = 0
    tp = fp = fn = tn = 0

    for pred in decisions:
        region_id = pred.get("region_id", "") or pred.get("region", "")
        category = pred.get("category", "")

        # 获取该区域的 location_id (优先从 REGION_LOCATION_MAP, 退化用坐标查城市)
        loc_info = REGION_LOCATION_MAP.get(region_id, {})
        location_id = loc_info.get("location_id", "")

        # 如果没有 location_id, 尝试用坐标查 QWeather 城市搜索
        if not location_id:
            coords = pred.get("coordinates") or REGION_COORDS.get(region_id, {})
            lat = coords.get("lat") if coords else None
            lng = coords.get("lng") if coords else None
            if lat and lng:
                location_id = f"{lng:.2f},{lat:.2f}"

        hist_weather = None
        if category in ("flood", "drought", "heat") and location_id:
            hist_weather = fetch_historical_weather(location_id, date_str)

        result = verify_prediction(pred, hist_weather)
        verifications.append(result)

        if result["verification_status"] == "verified":
            verified_count += 1
            if result["hit"] is True:
                if result["predicted"]:
                    tp += 1
                else:
                    tn += 1
            elif result["hit"] is False:
                if result["predicted"]:
                    fp += 1
                else:
                    fn += 1

    total = len(decisions)
    correct = tp + tn
    accuracy = round(correct / verified_count, 4) if verified_count > 0 else None

    output = {
        "date": iso_date,
        "generated_at": datetime.now(CST).isoformat(),
        "total_predictions": total,
        "verified": verified_count,
        "unverified": total - verified_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "verifications": verifications,
    }

    # 写入 verification_YYYYMMDD.json
    output_file = output_dir / f"verification_{date_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Verified {date_str}: {verified_count}/{total} verified, "
        f"tp={tp}, fp={fp}, fn={fn}, tn={tn}, accuracy={accuracy}"
    )

    return output


# ============================================================================
# 精度趋势 -- 累积所有验证结果
# ============================================================================


def build_accuracy_trend(output_dir: Path, keep_days: int = 14) -> dict[str, Any]:
    """读取所有 verification_*.json, 构建精度趋势数据。

    输出 accuracy_trend.json, 包含:
    - 总体 precision/recall/f1
    - 按灾种分类的精度
    - 每日趋势
    """
    now = datetime.now(CST)
    cutoff = now - timedelta(days=keep_days)

    # 收集所有验证文件
    verif_files = sorted(output_dir.glob("verification_*.json"))

    daily_trend: list[dict[str, Any]] = []

    total_tp = total_fp = total_fn = total_tn = 0
    by_category: dict[str, dict[str, int]] = {}

    for vf in verif_files:
        with open(vf, encoding="utf-8") as f:
            data = json.load(f)

        date_str = data.get("date", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=CST)
        except (ValueError, TypeError):
            continue

        if file_date < cutoff:
            continue

        tp = data.get("tp", 0)
        fp = data.get("fp", 0)
        fn = data.get("fn", 0)
        tn = data.get("tn", 0)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        verified = data.get("verified", 0)
        accuracy = data.get("accuracy")

        daily_trend.append(
            {
                "date": date_str,
                "verified": verified,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "accuracy": accuracy,
            }
        )

        # 按灾种统计
        for v in data.get("verifications", []):
            category = v.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}

            by_category[category]["total"] += 1
            if v.get("verification_status") == "verified":
                if v.get("hit") is True:
                    if v["predicted"]:
                        by_category[category]["tp"] += 1
                    else:
                        by_category[category]["tn"] += 1
                elif v.get("hit") is False:
                    if v["predicted"]:
                        by_category[category]["fp"] += 1
                    else:
                        by_category[category]["fn"] += 1

    # 总体精度
    total_verified = total_tp + total_fp + total_fn + total_tn
    accuracy = (
        round((total_tp + total_tn) / total_verified, 4) if total_verified > 0 else None
    )

    precision = (
        round(total_tp / (total_tp + total_fp), 4)
        if (total_tp + total_fp) > 0
        else None
    )
    recall = (
        round(total_tp / (total_tp + total_fn), 4)
        if (total_tp + total_fn) > 0
        else None
    )
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall and (precision + recall) > 0
        else None
    )

    # 按灾种计算 precision/recall
    by_category_stats: dict[str, dict[str, Any]] = {}
    for cat, counts in by_category.items():
        c_tp = counts["tp"]
        c_fp = counts["fp"]
        c_fn = counts["fn"]
        c_tn = counts["tn"]
        c_verified = c_tp + c_fp + c_fn + c_tn
        c_acc = round((c_tp + c_tn) / c_verified, 4) if c_verified > 0 else None
        c_prec = round(c_tp / (c_tp + c_fp), 4) if (c_tp + c_fp) > 0 else None
        c_rec = round(c_tp / (c_tp + c_fn), 4) if (c_tp + c_fn) > 0 else None
        by_category_stats[cat] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "tn": c_tn,
            "total": counts["total"],
            "verified": c_verified,
            "accuracy": c_acc,
            "precision": c_prec,
            "recall": c_rec,
        }

    trend = {
        "generated_at": datetime.now(CST).isoformat(),
        "keep_days": keep_days,
        "total_verified": total_verified,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "by_category": by_category_stats,
        "daily_trend": daily_trend,
    }

    # 写入 accuracy_trend.json
    trend_file = output_dir / "accuracy_trend.json"
    with open(trend_file, "w", encoding="utf-8") as f:
        json.dump(trend, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Accuracy trend: {total_verified} verified, "
        f"precision={precision}, recall={recall}, f1={f1}"
    )

    return trend


# ============================================================================
# Pipeline 入口 -- 延迟验证 T-N 的预测
# ============================================================================


def run_delayed_verification(
    output_dir: Path,
    delay_days: int = 2,
) -> dict[str, Any]:
    """验证 delay_days 天前的预测 (默认 T-2)。

    T-2 的原因: 历史天气数据通常需要 1-2 天才能完全稳定,
    FIRMS 卫星数据也有延迟, T-2 确保数据已上线。

    如果该日期已有验证文件, 则跳过。
    """
    now = datetime.now(CST)
    target_date = now - timedelta(days=delay_days)

    date_str = _date_str(target_date)
    verif_file = output_dir / f"verification_{date_str}.json"

    if verif_file.exists():
        logger.info(f"Verification for {date_str} already exists, skipping")
        with open(verif_file, encoding="utf-8") as f:
            return json.load(f)

    logger.info(
        f"Starting delayed verification for {_iso_date(target_date)} (T-{delay_days})"
    )

    result = verify_date_predictions(target_date, output_dir)

    return result
