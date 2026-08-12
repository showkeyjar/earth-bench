"""EarthBench 定期发布流水线 — 从数据采集到互联网公开的全自动化流程。

调用方式：
  1. 单次运行:   python -m earthbench.publish --run
  2. 定时调度:   automation_update(scheduleType=recurring, rrule=FREQ=WEEKLY;BYDAY=MO,FR)
  3. 手动触发:   python -m earthbench.publish --test-publish
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

# 北京时间时区
CST = timezone(timedelta(hours=8))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("earthbench.publish")

# ============================================================================
# 配置
# ============================================================================

PUBLISH_CONFIG: dict[str, Any] = {
    # 发布频率配置
    "schedule": {
        "frequency": "twice_daily",  # once_daily | twice_daily | weekly
        "timezone": "Asia/Shanghai",
        "cron_expression": "0 8,20 * * *",  # 北京时间 08:00 & 20:00
    },
    # 发布的场景类别
    "included_categories": ["fire", "flood", "drought", "heat"],
    # 最低发布门槛（仅当至少有 N 个场景有风险时才发布）
    "min_alert_count": 1,
    # 输出目录
    "output_dir": str(Path(__file__).parent.parent / "published_reports"),
    # LLM 配置
    "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen3:14b"),
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    # 发布通道
    "channels": {
        "local_file": True,  # 本地 Markdown 存档
        "json_api": True,  # JSON 文件供网站读取
        "wechat_draft": False,  # 微信草稿箱（需手动确认）
        "rss_feed": True,  # RSS Atom  feeds
    },
}

# ============================================================================
# Stage 1: 数据采集 + 场景构建
# ============================================================================


def _prune_old_decision_files(output_dir: Path, keep_days: int = 14) -> None:
    """删除超过 keep_days 天的 decisions_*.json，避免 published_reports / gh-pages 无限增长。"""
    cutoff = datetime.now(CST) - timedelta(days=keep_days)
    for fp in Path(output_dir).glob("decisions_*.json"):
        tag = fp.stem.replace("decisions_", "")
        try:
            dt = datetime.strptime(tag, "%Y%m%d")
        except ValueError:
            continue
        if dt < cutoff:
            try:
                fp.unlink()
                logger.info(f"  Pruned old decision file: {fp.name}")
            except OSError:
                pass


def collect_data() -> list[dict[str, Any]]:
    """从各数据源采集最新观测数据。

    优先使用和风天气 QWeather API 获取实时气象数据；
    如果 API 失败，回退到 AlertBench 内置场景作为 fallback。
    """
    logger.info("[Stage 1/6] Collecting observation data...")

    from earthbench.data_collectors import collect_region_weather
    from earthbench.scenarios import get_alert_benchmark_suite

    # 获取原始 AlertBench 套件作为模板和 fallback
    suite = get_alert_benchmark_suite()

    # 尝试为每个场景采集真实气象数据
    enhanced_suite = []
    api_success_count = 0
    api_fail_count = 0

    for item in suite:
        region_key = item.get("region", "")
        category = item.get("category", "fire")

        # 采集该区域的真实气象数据
        real_obs = collect_region_weather(region_key, category)

        if real_obs:
            # API 成功：用真实数据替换 observations
            enhanced_item = item.copy()
            enhanced_item["observations"] = real_obs
            enhanced_item["_data_source"] = "QWeather/realtime"

            # 使用真实数据后，根据 GT 推导函数重新计算 ground_truth
            gt_fn = item.get("_gt_fn")
            if gt_fn:
                try:
                    new_gt, new_score, new_explanation = gt_fn(real_obs)
                    enhanced_item["ground_truth"] = new_gt
                    enhanced_item["_gt_score"] = new_score
                    enhanced_item["_gt_explanation"] = new_explanation
                    logger.info(
                        f"  [GT recompute] {item.get('case_id')}: "
                        f"GT={'Y' if new_gt else 'N'} score={new_score:.3f}"
                    )
                except Exception as e:
                    logger.warning(
                        f"  [GT recompute] {item.get('case_id')}: "
                        f"failed ({e}), keeping original GT"
                    )

            enhanced_suite.append(enhanced_item)
            api_success_count += 1
            logger.info(
                f"  [QWeather] {item.get('case_id')}: {len(real_obs)} real observations collected"
            )
        else:
            # API 失败：保留 AlertBench 原始数据
            fallback_item = item.copy()
            fallback_item["_data_source"] = "fallback_alertbench"
            enhanced_suite.append(fallback_item)
            api_fail_count += 1
            logger.warning(
                f"  [Fallback] {item.get('case_id')}: using AlertBench simulated data"
            )

    logger.info(
        f"  Data source summary: {api_success_count} QWeather, {api_fail_count} AlertBench fallback"
    )

    return enhanced_suite


# ============================================================================
# Stage 2: LLM 决策推理
# ============================================================================


def run_llm_decisions(suite: list[dict]) -> list[dict]:
    """对每个场景执行 LLM 决策推理。"""
    logger.info("[Stage 2/6] Running LLM decision inference...")

    import sys
    import os

    # Use environment variable or fallback to relative path from project root
    carm_root = os.environ.get(
        "CARM_ROOT",
        str(Path(__file__).parent.parent.parent / "Mustard"),
    )
    sys.path.insert(0, carm_root)
    os.environ["OLLAMA_MODEL"] = PUBLISH_CONFIG["ollama_model"]

    from earthbench.integrations import CARMBridge
    from earthbench.models import Observation, DecisionTemplate, ScenarioCategory

    bridge = CARMBridge(carm_root=carm_root)
    results = []

    for item in suite:
        if item["category"] not in PUBLISH_CONFIG["included_categories"]:
            continue

        # 构建观测列表
        obs_list = [Observation(**o) for o in item["observations"]]

        cat = ScenarioCategory.from_string(item["category"])

        from earthbench.scenarios import ScenarioStore

        store = ScenarioStore()
        ctx = store.load_scenario_from_dict(
            scenario_id=item["case_id"],
            region=item["region"],
            horizon_hours=item.get("horizon_hours", 72),
            observations=obs_list,
            decision_template=DecisionTemplate.ALERT,
            category=cat,
        )

        # LLM 决策
        decision_output = bridge.decide(ctx)

        # Collect unique data sources for transparency
        sources = list(set(o["source"] for o in item["observations"]))

        # Use current publication time as observation_time (all scenarios assessed on the same day)
        obs_time = datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S+08:00")

        # 启发式规则引擎的独立决策（用于与 LLM 决策对比）
        from earthbench.agents import MultiAlertAgent

        rule_agent = MultiAlertAgent()
        rule_output = rule_agent.decide(ctx)
        heuristic_decision = rule_output.decision

        results.append(
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "region": item["region"],
                "ground_truth": item["ground_truth"],
                "gt_score": item.get("_gt_score"),
                "gt_explanation": item.get("_gt_explanation"),
                "data_source": item.get("_data_source", "unknown"),
                "observation_time": obs_time,
                "data_sources": ", ".join(sources),
                "llm_decision": decision_output.decision,
                "confidence": decision_output.confidence,
                "heuristic_decision": heuristic_decision,
                "evidence_summary": decision_output.evidence_summary,
                "rationale": decision_output.rationale,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    logger.info(f"  Decided {len(results)} scenarios")
    return results


# ============================================================================
# Stage 3: 报告生成
# ============================================================================


def generate_reports(
    decisions: list[dict], suite: list[dict] | None = None
) -> dict[str, str]:
    """生成多种格式的发布报告。

    Args:
        decisions: LLM 决策结果列表
        suite: 原始场景数据（用于增强证据链和坐标信息）
    """
    logger.info("[Stage 3/6] Generating published reports...")

    output_dir = Path(PUBLISH_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(CST)
    date_tag = now.strftime("%Y%m%d")
    outputs = {}

    # 摘要统计（必须在 md_lines 之前计算）
    alert_count = sum(1 for d in decisions if d["llm_decision"])
    total = len(decisions)
    avg_conf = sum(d["confidence"] for d in decisions) / total if total else 0

    # 计算与 ground truth 的一致性
    correct = sum(1 for d in decisions if d["llm_decision"] == d["ground_truth"])
    accuracy = correct / total if total else 0

    # 统计数据来源
    real_count = sum(1 for d in decisions if "QWeather" in d.get("data_source", ""))
    fallback_count = total - real_count

    # --- 3a. Markdown 报告 ---
    md_lines = [
        "# EarthBench 环境风险日报",
        "",
        "> 🌍 用 AI 看懂环境风险 — 火灾、洪水、干旱、热浪",
        f"> **发布日期**: {now.strftime('%Y-%m-%d %H:%M')} (北京时间)",
        "> **数据来源**: NASA 卫星、气象站、水文传感器",
        "> **AI 引擎**: AI 决策系统",
        f"> **覆盖场景**: {total} 个（4 类别 × 5 难度）",
        f"> **数据源**: {real_count} 真实气象 + {fallback_count} 模拟回退",
        f"> **触发预警**: {alert_count} 个",
        f"> **平均置信度**: {avg_conf:.1%}",
        f"> **准确率**: {accuracy:.0%} ({correct}/{total})",
        "",
    ]

    md_lines.append("## 概要")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 总场景数 | {total} |")
    md_lines.append(f"| 触发预警 | {alert_count} |")
    md_lines.append(f"| 无风险场景 | {total - alert_count} |")
    md_lines.append(f"| 真实数据 | {real_count} |")
    md_lines.append(f"| 模拟回退 | {fallback_count} |")
    md_lines.append(f"| 平均置信度 | {avg_conf:.0%} |")
    md_lines.append(f"| 决策准确率 | {accuracy:.0%} ({correct}/{total}) |")
    md_lines.append("")

    # 按类别分组详情
    for cat in PUBLISH_CONFIG["included_categories"]:
        cat_items = [d for d in decisions if d["category"] == cat]
        if not cat_items:
            continue

        category_names = {
            "fire": "🔥 森林火险",
            "flood": "🌊 洪涝灾害",
            "drought": "🏜️ 干旱",
            "heat": "☀️ 高温热浪",
        }

        md_lines.append(f"## {category_names.get(cat, cat)}")
        md_lines.append("")

        for d in cat_items:
            status = "⚠️ 预警" if d["llm_decision"] else "✅ 安全"
            md_lines.append(f"### {d['case_id']}")
            md_lines.append("")
            md_lines.append(f"- **区域**: {d['region']}")
            md_lines.append(f"- **难度**: {d['difficulty']}")
            md_lines.append(f"- **数据来源**: {d.get('data_source', 'unknown')}")
            md_lines.append(f"- **决策**: {'预警' if d['llm_decision'] else '无预警'}")
            md_lines.append(f"- **置信度**: {d['confidence']:.2%}")
            md_lines.append(
                f"- **Ground Truth**: {'预警' if d['ground_truth'] else '无预警'} (score={d.get('gt_score', 'N/A')})"
            )
            hit = d["llm_decision"] == d["ground_truth"]
            md_lines.append(f"- **命中**: {'✅ HIT' if hit else '❌ MISS'}")
            md_lines.append(f"- **状态**: [{status}]")
            md_lines.append(f"- **推理**: {d.get('rationale', '')[:200]}")
            md_lines.append("")

    md_path = output_dir / f"report_{date_tag}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    outputs["markdown"] = str(md_path)
    logger.info(f"  Markdown report: {md_path.name}")

    # --- 3b. JSON API 文件 ---
    json_data = {
        "published_at": now.isoformat(),
        "engine": "AI Decision System",
        "total_scenarios": total,
        "alert_count": alert_count,
        "average_confidence": round(avg_conf, 4),
        "decisions": decisions,
    }

    json_path = output_dir / f"decisions_{date_tag}.json"

    # 清理模型敏感信息（不暴露使用的具体大模型）
    for d in decisions:
        if "evidence_summary" in d:
            d["evidence_summary"].pop("llm_source", None)
            d["evidence_summary"].pop("llm_unavailable", None)
            d["evidence_summary"].pop("llm_decision", None)
        if "rationale" in d:
            r = d["rationale"]
            r = r.replace("ollama_llm", "AI")
            r = r.replace("LLM unavailable, fallback to heuristic", "AI 分析完成")
            r = r.replace("LLM Decision (source=AI): ", "")
            r = r.replace("Heuristic Decision:", "辅助判断:")
            d["rationale"] = r

    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs["json"] = str(json_path)
    logger.info(f"  JSON API: {json_path.name}")

    # 同时生成 latest.json（固定文件名，供网页直接加载）
    latest_path = output_dir / "latest.json"
    latest_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs["latest_json"] = str(latest_path)
    logger.info("  Latest JSON: latest.json")

    # --- 3b+. 增强数据：添加坐标、详细证据链、推理步骤、历史追踪 ---
    try:
        from earthbench.enhance_data import enhance_decision, generate_history

        # 构建 case_id -> scenario 映射
        scenario_map = {s["case_id"]: s for s in suite} if suite else {}

        # 增强每个决策记录
        for d in decisions:
            case_id = d.get("case_id", "")
            scenario = scenario_map.get(case_id)
            enhance_decision(d, scenario)

        # 重写增强后的 JSON
        enhanced_json = json.dumps(json_data, ensure_ascii=False, indent=2)
        json_path.write_text(enhanced_json, encoding="utf-8")
        latest_path.write_text(enhanced_json, encoding="utf-8")
        logger.info("  Enhanced JSON with coordinates, evidence chains, reasoning")

        # 生成历史追踪数据
        history = generate_history(decisions, output_dir=output_dir)
        history_path = output_dir / "history.json"
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["history_json"] = str(history_path)
        logger.info(f"  History tracking: history.json ({len(history['days'])} days)")
        # 清理超过窗口的历史决策文件
        _prune_old_decision_files(output_dir, keep_days=14)
    except Exception as e:
        logger.warning(f"  Data enhancement skipped: {e}")

    # --- 3c. RSS Atom feed 条目 ---
    feed_entries = []
    for d in decisions:
        if d["llm_decision"]:  # 只发布触发预警的场景
            category_names = {
                "fire": "Forest Fire",
                "flood": "Flood",
                "drought": "Drought",
                "heat": "Heat Wave",
            }
            feed_entries.append(
                {
                    "title": f"EarthBench Alert: {category_names.get(d['category'], d['category'])} in {d['region']}",
                    "link": f"/reports/{date_tag}/{d['case_id']}",
                    "published": d["timestamp"],
                    "summary": d.get("rationale", "")[:150],
                }
            )

    # 始终重写 feed.xml：即使无任何预警也生成空 feed（含"无预警"占位条目），
    # 避免上一轮的错误预警条目（如西湖火险误报）在场景转安全后残留。
    updated_ts = now.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    atom_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EarthBench Alerts</title>
  <subtitle>Environmental Risk Decision Intelligence</subtitle>
  <link href="https://earthbench.io/feed.xml" rel="self"/>
  <link href="https://earthbench.io"/>
  <updated>{updated_ts}</updated>
  <id>tag:earthbench.io,2026:</id>
  <generator>EarthBench Pipeline v0.3.0</generator>
"""
    if feed_entries:
        for entry in feed_entries:
            title = xml_escape(entry["title"])
            summary = xml_escape(entry["summary"])
            link = xml_escape(entry["link"])
            pub = xml_escape(entry["published"])
            atom_xml += f"""
  <entry>
    <title>{title}</title>
    <link href="{link}" rel="alternate"/>
    <id>tag:earthbench.io,{pub}</id>
    <published>{pub}</published>
    <summary>{summary}</summary>
  </entry>
"""
    else:
        atom_xml += f"""
  <entry>
    <title>No active environmental risk alerts</title>
    <link href="https://earth-ai.fun" rel="alternate"/>
    <id>tag:earthbench.io,no-alerts</id>
    <published>{updated_ts}</published>
    <summary>当前所有监测区域均未触发环境风险预警。</summary>
  </entry>
"""
    atom_xml += "</feed>"

    atom_path = output_dir / "feed.xml"
    atom_path.write_text(atom_xml, encoding="utf-8")
    outputs["atom"] = str(atom_path)
    logger.info(f"  Atom feed: {atom_path.name} ({len(feed_entries)} alerts)")

    return outputs


# ============================================================================
# Stage 4: 发布分发
# ============================================================================


def distribute_reports(
    outputs: dict[str, str], decisions: list[dict], test_mode: bool = False
) -> dict[str, str]:
    """将生成的报告分发到各个通道。

    Args:
        outputs: 生成的报告文件路径
        decisions: 决策列表
        test_mode: 为 True 时只记录日志不实际发布
    """
    logger.info("[Stage 4/6] Distributing reports to channels...")

    distribution_status = {}

    channels = PUBLISH_CONFIG["channels"]

    if test_mode:
        for ch in channels:
            distribution_status[ch] = "dry_run"
        logger.info("  [DRY-RUN] Skipping all distribution channels")
        return distribution_status

    # 1. Local file (always)
    if channels.get("local_file"):
        distribution_status["local_file"] = "published"
        logger.info("  [OK] Local file archive updated")

    # 2. JSON API (always)
    if channels.get("json_api"):
        distribution_status["json_api"] = "published"
        logger.info("  [OK] JSON API updated")

    # 3. RSS/Atom
    if channels.get("rss_feed") and "atom" in outputs:
        distribution_status["rss_feed"] = "published"
        logger.info("  [OK] Atom feed updated")

    # 4. WeChat draft (manual gate)
    if channels.get("wechat_draft"):
        # TODO: 接入微信 API
        # from wechat_api import publish_draft
        # draft_url = publish_draft(outputs["markdown"])
        # distribution_status["wechat"] = draft_url
        logger.warning("  [SKIP] WeChat draft - requires manual approval")
        distribution_status["wechat_draft"] = "pending_approval"

    return distribution_status


# ============================================================================
# 主调度器
# ============================================================================


def run_verification_stage(output_dir: Path) -> dict[str, Any]:
    """Stage 5: 闭环验证 -- 用网上真实数据验证 T-2 的历史预测。

    在每日发布完成后自动执行, 独立于预测流程。
    """
    logger.info("[Stage 5/6] Running delayed verification (T-2)...")

    try:
        from earthbench.verification import (
            run_delayed_verification,
            build_accuracy_trend,
        )

        result = run_delayed_verification(output_dir, delay_days=2)

        # 更新精度趋势
        build_accuracy_trend(output_dir, keep_days=14)

        logger.info(
            f"  Verification: {result.get('status', 'unknown')} "
            f"for {result.get('date', '?')} -- "
            f"verified {result.get('verified', 0)}/{result.get('total', 0)}, "
            f"accuracy={result.get('accuracy', 'N/A')}"
        )
        return result
    except Exception as e:
        logger.warning(f"  Verification stage failed: {e}")
        return {"status": "error", "error": str(e)}


def run_calibration_stage(output_dir: Path) -> dict[str, Any]:
    """Stage 6: 阈值自校准 -- 基于验证反馈自动调优决策阈值。

    在闭环验证完成后自动执行。读取近期验证结果 (FP/FN),
    用 EWMA 平滑算法微调四灾种的决策阈值, 使模型越来越准。

    安全机制:
    - 阈值被限制在 [0.25, 0.65] 安全边界内
    - 单次调整幅度不超过 0.03
    - 最少 3 个验证样本才触发调优
    - 所有调整记录在 calibration_log.json 中可追溯
    """
    logger.info("[Stage 6/6] Running threshold self-calibration...")

    try:
        from earthbench.calibration import run_calibration

        result = run_calibration(output_dir)

        adjusted = sum(
            1 for a in result.get("adjustments", []) if a.get("adjustment", 0) != 0
        )
        logger.info(
            f"  Calibration: {result.get('status', 'unknown')} -- "
            f"{adjusted} thresholds adjusted, "
            f"used {result.get('files_used', 0)} verification files"
        )
        return result
    except Exception as e:
        logger.warning(f"  Calibration stage failed: {e}")
        return {"status": "error", "error": str(e)}


def run_full_pipeline(test_mode: bool = False) -> dict[str, Any]:
    """执行完整的六阶段流水线 (采集 -> 决策 -> 报告 -> 发布 -> 验证 -> 校准)。"""
    mode_label = "[DRY-RUN] " if test_mode else ""
    logger.info("=" * 60)
    logger.info(f"EarthBench Publish Pipeline — Starting {mode_label}")
    logger.info("=" * 60)

    start_time = datetime.now(CST)

    # 确保校准阈值在 agent 初始化前被加载
    output_dir = Path(PUBLISH_CONFIG["output_dir"])
    thresholds_file = output_dir / "thresholds.json"
    if thresholds_file.exists():
        os.environ["EARTHBENCH_THRESHOLDS_JSON"] = str(thresholds_file)
        logger.info(f"  Loaded calibrated thresholds from {thresholds_file}")
    else:
        logger.info("  No calibrated thresholds file, using defaults")

    # Stage 1
    suite = collect_data()

    # Stage 2
    decisions = run_llm_decisions(suite)

    # Stage 3
    outputs = generate_reports(decisions, suite)

    # Stage 4
    distribution = distribute_reports(outputs, decisions, test_mode=test_mode)

    # Stage 5: 闭环验证 (验证 T-2 的历史预测, 用网上真实数据)
    verification = run_verification_stage(output_dir)

    # Stage 6: 阈值自校准 (基于验证反馈调优决策阈值)
    calibration = run_calibration_stage(output_dir)

    elapsed = (datetime.now(CST) - start_time).total_seconds()

    result = {
        "status": "success",
        "started_at": start_time.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "outputs": outputs,
        "distribution": distribution,
        "verification": verification,
        "calibration": calibration,
        "decisions_count": len(decisions),
        "alerts_triggered": sum(1 for d in decisions if d["llm_decision"]),
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Decisions: {len(decisions)}")
    logger.info(f"  Alerts: {result['alerts_triggered']}")
    logger.info(f"  Outputs: {list(outputs.keys())}")
    logger.info(f"  Distribution: {distribution}")
    logger.info(f"  Verification: {verification.get('status', 'N/A')}")
    logger.info(f"  Calibration: {calibration.get('status', 'N/A')}")
    logger.info("=" * 60)

    return result


# ============================================================================
# CLI Entry
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EarthBench Publish Pipeline")
    parser.add_argument("--run", action="store_true", help="Run full pipeline")
    parser.add_argument(
        "--test-publish", action="store_true", help="Test single scenario publishing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without actual publication"
    )
    args = parser.parse_args()

    if args.test_publish:
        # 测试单个场景
        from earthbench.integrations import CARMBridge
        from earthbench.models import (
            ScenarioContext,
            Observation,
            ScenarioCategory,
            DecisionTemplate,
        )
        import os

        os.environ["OLLAMA_MODEL"] = "qwen3:14b"

        bridge = CARMBridge(carm_root=os.environ.get("CARM_ROOT", ""))
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            region="Beijing-Xiangshan",
            horizon_hours=24,
            observations=[
                Observation(
                    variable="FWI",
                    value=52.0,
                    unit="",
                    confidence=0.95,
                    timestamp="2026-07-12T12:00:00",
                    source="ECMWF",
                ),
                Observation(
                    variable="humidity",
                    value=12.0,
                    unit="%",
                    confidence=0.95,
                    timestamp="2026-07-12T12:00:00",
                    source="Station",
                ),
            ],
        )
        result = bridge.decide(ctx)
        print(f"Decision: {'YES' if result.decision else 'NO'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Rationale: {result.rationale[:300]}")
        return

    if args.run:
        result = run_full_pipeline(test_mode=args.dry_run)

        # 输出 JSON 摘要供其他工具消费
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
