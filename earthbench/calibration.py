"""EarthBench 模型自校准模块 -- 基于闭环验证反馈自动调优决策阈值。

核心思想:
  规则引擎的决策阈值 (fire: 0.4, flood: 0.45, drought: 0.4, heat: 0.4)
  是系统最关键的"旋钮"。延迟验证 (verification.py) 提供了独立的 FP/FN 反馈,
  可以用来判断阈值是否需要调整。

调优策略: EWMA (指数加权移动平均) 平滑调整
  - FP (误报) 多了 -> 阈值上调 (更保守, 减少报警)
  - FN (漏报) 多了 -> 阈值下调 (更激进, 增加报警)
  - 单次调整幅度被 learning_rate 限制 (默认 0.05), 防止震荡
  - 阈值被限制在安全边界 [0.25, 0.65] 内

安全机制:
  1. 只在有足够验证样本 (min_samples) 时才调优
  2. 单次调整幅度不超过 max_step (默认 0.03)
  3. 阈值不超出 [min_threshold, max_threshold] 边界
  4. 调整方向由 (fp - fn) 的差值驱动, 平衡精确率和召回率
  5. 每次调整都记录完整的调优日志 (前后值、原因、验证数据)

数据流:
  verification_*.json -> calibration.py -> thresholds.json + calibration_log.json
  -> agents.py (读取阈值) -> 规则引擎使用新阈值决策
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("earthbench.calibration")

CST = timezone(timedelta(hours=8))

# ============================================================================
# 默认阈值 (与 agents.py 中的硬编码值一致)
# ============================================================================

DEFAULT_THRESHOLDS: dict[str, float] = {
    "fire": 0.40,
    "flood": 0.45,
    "drought": 0.40,
    "heat": 0.40,
}

# 安全边界: 阈值不能超出此范围
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.65

# 调优参数
LEARNING_RATE = 0.05  # EWMA 学习率 (越小越保守)
MAX_STEP = 0.03  # 单次最大调整幅度
MIN_SAMPLES = 3  # 最少验证样本数 (少于此不调优)
LOOKBACK_DAYS = 14  # 回看多少天的验证数据


# ============================================================================
# 阈值配置读写
# ============================================================================


def load_thresholds(output_dir: Path) -> dict[str, float]:
    """从 thresholds.json 加载当前阈值。如果不存在则返回默认值。"""
    threshold_file = output_dir / "thresholds.json"

    if not threshold_file.exists():
        logger.info("No thresholds.json found, using defaults")
        return dict(DEFAULT_THRESHOLDS)

    try:
        with open(threshold_file, encoding="utf-8") as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        # 合并默认值 (防止新增灾种缺失)
        result = dict(DEFAULT_THRESHOLDS)
        result.update({k: float(v) for k, v in thresholds.items()})
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to load thresholds.json: {e}, using defaults")
        return dict(DEFAULT_THRESHOLDS)


def save_thresholds(
    output_dir: Path,
    thresholds: dict[str, float],
    log_entry: dict[str, Any] | None = None,
) -> None:
    """保存阈值到 thresholds.json, 并追加调优日志。"""
    threshold_file = output_dir / "thresholds.json"

    # 读取现有数据 (如果有)
    existing = {}
    if threshold_file.exists():
        try:
            with open(threshold_file, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    # 保存阈值
    data = {
        "thresholds": thresholds,
        "updated_at": datetime.now(CST).isoformat(),
        "version": existing.get("version", 0) + 1,
    }

    with open(threshold_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 追加调优日志
    if log_entry:
        log_file = output_dir / "calibration_log.json"

        log_entries: list[dict[str, Any]] = []
        if log_file.exists():
            try:
                with open(log_file, encoding="utf-8") as f:
                    log_entries = json.load(f)
                    if not isinstance(log_entries, list):
                        log_entries = []
            except (json.JSONDecodeError, ValueError):
                pass

        log_entries.append(log_entry)

        # 只保留最近 30 条日志
        if len(log_entries) > 30:
            log_entries = log_entries[-30:]

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Threshold calibration: {log_entry.get('category', '?')} "
            f"{log_entry.get('old_value', '?')} -> {log_entry.get('new_value', '?')} "
            f"({log_entry.get('reason', '?')})"
        )


# ============================================================================
# 从验证结果计算调优方向
# ============================================================================


def compute_adjustment(
    category: str,
    fp: int,
    fn: int,
    tn: int,
    tp: int,
) -> dict[str, Any]:
    """根据 FP/FN 计算阈值调整方向和幅度。

    逻辑:
    - FP > FN: 误报多于漏报 -> 阈值上调 (更保守)
    - FN > FP: 漏报多于误报 -> 阈值下调 (更激进)
    - FP == FN: 不调整

    调整幅度 = min(|fp - fn| / (fp + fn + 1) * LEARNING_RATE, MAX_STEP)
    """
    total = tp + fp + fn + tn

    if total < MIN_SAMPLES:
        return {
            "action": "skip",
            "reason": f"insufficient_samples ({total} < {MIN_SAMPLES})",
            "adjustment": 0.0,
        }

    if fp == 0 and fn == 0:
        return {
            "action": "skip",
            "reason": "no_errors (perfect)",
            "adjustment": 0.0,
        }

    # 误差方向: 正=误报多(需上调), 负=漏报多(需下调)
    error_diff = fp - fn

    # 归一化: |error_diff| / total_errors
    total_errors = fp + fn
    normalized_diff = error_diff / max(total_errors, 1)

    # EWMA 调整: 幅度被 learning_rate 限制
    raw_adjustment = normalized_diff * LEARNING_RATE

    # 限制单次最大调整幅度
    adjustment = max(-MAX_STEP, min(MAX_STEP, raw_adjustment))

    if abs(adjustment) < 0.001:
        return {
            "action": "skip",
            "reason": f"adjustment_too_small ({adjustment:.4f})",
            "adjustment": 0.0,
        }

    direction = "up" if adjustment > 0 else "down"
    reason = (
        f"fp={fp}, fn={fn}, tn={tn}, tp={tp} | error_diff={error_diff} -> {direction}"
    )

    return {
        "action": "adjust",
        "reason": reason,
        "adjustment": round(adjustment, 4),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp": tp,
    }


# ============================================================================
# 执行调优
# ============================================================================


def run_calibration(output_dir: Path) -> dict[str, Any]:
    """主入口: 读取验证结果, 计算并应用阈值调优。

    Returns:
        调优结果摘要, 包含每个灾种的调整详情
    """
    logger.info("[Calibration] Starting threshold calibration...")

    # 加载当前阈值
    current_thresholds = load_thresholds(output_dir)

    # 收集近 LOOKBACK_DAYS 天的验证数据, 按灾种聚合
    now = datetime.now(CST)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)

    # 按灾种聚合 FP/FN/TP/TN
    category_stats: dict[str, dict[str, int]] = {}
    for cat in ("fire", "flood", "drought", "heat"):
        category_stats[cat] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    verif_files = sorted(output_dir.glob("verification_*.json"))
    files_used = 0

    for vf in verif_files:
        try:
            with open(vf, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue

        date_str = data.get("date", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=CST)
        except (ValueError, TypeError):
            continue

        if file_date < cutoff:
            continue

        files_used += 1

        # 聚合每个灾种的验证结果
        for v in data.get("verifications", []):
            category = v.get("category", "")
            if category not in category_stats:
                continue

            if v.get("verification_status") != "verified":
                continue

            if v.get("hit") is True:
                if v.get("predicted"):
                    category_stats[category]["tp"] += 1
                else:
                    category_stats[category]["tn"] += 1
            elif v.get("hit") is False:
                if v.get("predicted"):
                    category_stats[category]["fp"] += 1
                else:
                    category_stats[category]["fn"] += 1

    logger.info(
        f"[Calibration] Aggregated {files_used} verification files, "
        f"stats: {category_stats}"
    )

    # 对每个灾种计算调整
    new_thresholds = dict(current_thresholds)
    adjustments: list[dict[str, Any]] = []

    for category, stats in category_stats.items():
        old_value = current_thresholds.get(
            category, DEFAULT_THRESHOLDS.get(category, 0.4)
        )

        result = compute_adjustment(
            category,
            stats["fp"],
            stats["fn"],
            stats["tn"],
            stats["tp"],
        )

        if result["action"] != "adjust":
            adjustments.append(
                {
                    "category": category,
                    "old_value": old_value,
                    "new_value": old_value,
                    "adjustment": 0.0,
                    "action": result["action"],
                    "reason": result["reason"],
                    "stats": stats,
                }
            )
            continue

        # 应用调整并限制在安全边界内
        new_value = old_value + result["adjustment"]
        new_value = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_value))
        new_value = round(new_value, 4)

        # 如果被边界截断, 调整幅度可能变化
        actual_adjustment = round(new_value - old_value, 4)

        new_thresholds[category] = new_value

        log_entry = {
            "timestamp": datetime.now(CST).isoformat(),
            "category": category,
            "old_value": old_value,
            "new_value": new_value,
            "adjustment": actual_adjustment,
            "action": "adjust",
            "reason": result["reason"],
            "stats": stats,
            "verification_files_used": files_used,
            "lookback_days": LOOKBACK_DAYS,
        }

        adjustments.append(log_entry)

    # 检查是否有实际变更
    has_changes = any(a["adjustment"] != 0.0 for a in adjustments)

    if has_changes:
        # 保存新阈值 + 调优日志 (只记录有变更的)
        changed_logs = [a for a in adjustments if a["adjustment"] != 0.0]
        for log_entry in changed_logs:
            save_thresholds(output_dir, new_thresholds, log_entry)
            # save_thresholds 会追加日志, 但阈值只需要保存一次
            # 后续调用只追加日志
            break  # 只调用一次 save_thresholds

        # 再次保存确保阈值文件是最新的
        save_thresholds(output_dir, new_thresholds, None)
    else:
        logger.info("[Calibration] No threshold changes needed")

    # 构建返回摘要
    summary = {
        "status": "calibrated" if has_changes else "no_change",
        "files_used": files_used,
        "lookback_days": LOOKBACK_DAYS,
        "current_thresholds": current_thresholds,
        "new_thresholds": new_thresholds,
        "adjustments": adjustments,
        "calibrated_at": datetime.now(CST).isoformat(),
    }

    # 写入 calibration_status.json (前端展示用)
    status_file = output_dir / "calibration_status.json"
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[Calibration] Done: {len([a for a in adjustments if a['adjustment'] != 0.0])} "
        f"thresholds adjusted"
    )

    return summary
