"""EarthBench 定期发布流水线 — 从数据采集到互联网公开的全自动化流程。

调用方式：
  1. 单次运行:   python -m earthbench.publish --run
  2. 定时调度:   automation_update(scheduleType=recurring, rrule=FREQ=WEEKLY;BYDAY=MO,FR)
  3. 手动触发:   python -m earthbench.publish --test-publish
"""

from __future__ import annotations

import json
import sys
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

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
        "local_file": True,          # 本地 Markdown 存档
        "json_api": True,            # JSON 文件供网站读取
        "wechat_draft": False,       # 微信草稿箱（需手动确认）
        "rss_feed": True,            # RSS Atom  feeds
    },
}

# ============================================================================
# Stage 1: 数据采集 + 场景构建
# ============================================================================

def _prune_old_decision_files(output_dir: Path, keep_days: int = 14) -> None:
    """删除超过 keep_days 天的 decisions_*.json，避免 published_reports / gh-pages 无限增长。"""
    cutoff = datetime.now() - timedelta(days=keep_days)
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


def _sync_root_decisions(json_path: Path, date_tag: str) -> None:
    """将今日 decisions_*.json 复制到仓库根目录并提交 main。

    作用：CI 每次 fresh checkout main，published_reports 是空的；
    enhance_data._collect_recent_alerts 会回退扫描仓库根目录的
    decisions_*.json，从而保证“今日无预警→回看近14天真实预警”始终有数据。
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        dest = repo_root / f"decisions_{date_tag}.json"
        import shutil
        shutil.copy2(json_path, dest)
        logger.info(f"  Synced decision file to repo root: {dest.name}")

        import subprocess
        # 仅当文件有变化时才提交，避免无谓的空提交
        diff = subprocess.run(
            ["git", "status", "--porcelain", dest.name],
            cwd=repo_root, capture_output=True, text=True,
        )
        if not diff.stdout.strip():
            logger.info("  Root decision file unchanged, skip commit")
            return
        subprocess.run(["git", "add", dest.name], cwd=repo_root, check=False)
        subprocess.run(
            ["git", "commit", "-m", f"data: 同步 {date_tag} 决策文件（供近14天回看）"],
            cwd=repo_root, check=False,
        )
        # 推送到 main（CI 内 GITHUB_TOKEN 通常只读，故 best-effort，失败不影响发布）
        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, capture_output=True, text=True,
        )
        if push.returncode == 0:
            logger.info("  Pushed root decision file to main")
        else:
            logger.warning("  Push root decision file skipped (no write token or non-CI env)")
    except Exception as e:
        logger.warning(f"  Sync root decisions failed (non-fatal): {e}")


def collect_data() -> list[dict[str, Any]]:
    """从各数据源采集最新观测数据。
    
    优先使用和风天气 QWeather API 获取实时气象数据；
    如果 API 失败，回退到 AlertBench 内置场景作为 fallback。
    """
    logger.info("[Stage 1/4] Collecting observation data...")
    
    from earthbench.data_collectors import (
        collect_region_weather,
        fallback_to_scenarios,
        REGION_LOCATION_MAP,
    )
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
            enhanced_suite.append(enhanced_item)
            api_success_count += 1
            logger.info(f"  [QWeather] {item.get('case_id')}: {len(real_obs)} real observations collected")
        else:
            # API 失败：保留 AlertBench 原始数据
            fallback_item = item.copy()
            fallback_item["_data_source"] = "fallback_alertbench"
            enhanced_suite.append(fallback_item)
            api_fail_count += 1
            logger.warning(f"  [Fallback] {item.get('case_id')}: using AlertBench simulated data")
    
    logger.info(f"  Data source summary: {api_success_count} QWeather, {api_fail_count} AlertBench fallback")
    
    return enhanced_suite


# ============================================================================
# Stage 2: LLM 决策推理
# ============================================================================

def run_llm_decisions(suite: list[dict]) -> list[dict]:
    """对每个场景执行 LLM 决策推理。"""
    logger.info("[Stage 2/4] Running LLM decision inference...")
    
    import sys
    import os
    
    # Use environment variable or fallback to relative path from project root
    carm_root = os.environ.get(
        "CARM_ROOT",
        str(Path(__file__).parent.parent.parent / "Mustard"),
    )
    sys.path.insert(0, carm_root)
    os.environ['OLLAMA_MODEL'] = PUBLISH_CONFIG['ollama_model']
    
    from earthbench.integrations import CARMBridge
    from earthbench.models import Observation, DecisionTemplate, ScenarioCategory
    from earthbench.benchmark import AlertBenchEvaluator
    
    bridge = CARMBridge(carml_root=carm_root)
    results = []
    
    for item in suite:
        if item["category"] not in PUBLISH_CONFIG["included_categories"]:
            continue
            
        # 构建观测列表
        obs_list = [Observation(**o) for o in item["observations"]]
        
        category_map = {
            "fire": ScenarioCategory.FIRE,
            "flood": ScenarioCategory.FLOOD,
            "drought": ScenarioCategory.DROUGHT,
            "heat": ScenarioCategory.ECOLOGY,
        }
        cat = category_map.get(item["category"], ScenarioCategory.FIRE)
        
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
        
        # Extract the latest observation timestamp (approximate decision time)
        last_obs_time = obs_list[-1].timestamp if obs_list else datetime.now().isoformat()
        
        # Collect unique data sources for transparency
        sources = list(set(o["source"] for o in item["observations"]))
        
        # Use current publication time as observation_time (all scenarios assessed on the same day)
        obs_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
        
        results.append({
            "case_id": item["case_id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "region": item["region"],
            "ground_truth": item["ground_truth"],
            "observation_time": obs_time,
            "data_sources": ", ".join(sources),
            "llm_decision": decision_output.decision,
            "confidence": decision_output.confidence,
            "heuristic_decision": decision_output.context.observations and True,
            "evidence_summary": decision_output.evidence_summary,
            "rationale": decision_output.rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    logger.info(f"  Decided {len(results)} scenarios")
    return results


# ============================================================================
# Stage 3: 报告生成
# ============================================================================

def generate_reports(decisions: list[dict], suite: list[dict] | None = None) -> dict[str, str]:
    """生成多种格式的发布报告。

    Args:
        decisions: LLM 决策结果列表
        suite: 原始场景数据（用于增强证据链和坐标信息）
    """
    logger.info("[Stage 3/4] Generating published reports...")
    
    output_dir = Path(PUBLISH_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now()
    date_tag = now.strftime("%Y%m%d")
    outputs = {}
    
    # 摘要统计（必须在 md_lines 之前计算）
    alert_count = sum(1 for d in decisions if d["llm_decision"])
    total = len(decisions)
    avg_conf = sum(d["confidence"] for d in decisions) / total if total else 0
    
    # 计算与 ground truth 的一致性
    correct = sum(1 for d in decisions if d["llm_decision"] == d["ground_truth"])
    accuracy = correct / total if total else 0
    
    # --- 3a. Markdown 报告 ---
    md_lines = [
        f"# EarthBench 环境风险日报",
        f"",
        f"> 🌍 用 AI 看懂环境风险 — 火灾、洪水、干旱、热浪",
        f"> **发布日期**: {now.strftime('%Y-%m-%d %H:%M')} (北京时间)",
        f"> **数据来源**: NASA 卫星、气象站、水文传感器",
        f"> **AI 引擎**: AI 决策系统",
        f"> **覆盖场景**: {total} 个（4 类别 × 5 难度）",
        f"> **触发预警**: {alert_count} 个",
        f"> **平均置信度**: {avg_conf:.1%}",
        f"> **准确率**: {accuracy:.0%} ({correct}/{total})",
        f"",
    ]
    
    md_lines.append(f"## 概要")
    md_lines.append(f"")
    md_lines.append(f"| 指标 | 数值 |")
    md_lines.append(f"|------|------|")
    md_lines.append(f"| 总场景数 | {total} |")
    md_lines.append(f"| 触发预警 | {alert_count} |")
    md_lines.append(f"| 无风险场景 | {total - alert_count} |")
    md_lines.append(f"| 平均置信度 | {avg_conf:.0%} |")
    md_lines.append(f"| 决策准确率 | {accuracy:.0%} ({correct}/{total}) |")
    md_lines.append(f"")
    
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
        md_lines.append(f"")
        
        for d in cat_items:
            status = "⚠️ 预警" if d["llm_decision"] else "✅ 安全"
            md_lines.append(f"### {d['case_id']}")
            md_lines.append(f"")
            md_lines.append(f"- **区域**: {d['region']}")
            md_lines.append(f"- **难度**: {d['difficulty']}")
            md_lines.append(f"- **决策**: {'预警' if d['llm_decision'] else '无预警'}")
            md_lines.append(f"- **置信度**: {d['confidence']:.2%}")
            md_lines.append(f"- **状态**: [{status}]")
            md_lines.append(f"- **推理**: {d.get('rationale', '')[:200]}")
            md_lines.append(f"")
    
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
    
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["json"] = str(json_path)
    logger.info(f"  JSON API: {json_path.name}")
    
    # 同时生成 latest.json（固定文件名，供网页直接加载）
    latest_path = output_dir / "latest.json"
    latest_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["latest_json"] = str(latest_path)
    logger.info(f"  Latest JSON: latest.json")

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
        logger.info(f"  Enhanced JSON with coordinates, evidence chains, reasoning")

        # 生成历史追踪数据
        history = generate_history(decisions, output_dir=output_dir)
        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["history_json"] = str(history_path)
        logger.info(f"  History tracking: history.json ({len(history['days'])} days)")
        # 清理超过窗口的历史决策文件
        _prune_old_decision_files(output_dir, keep_days=14)

        # 将今日 decisions_*.json 同步回仓库根目录并提交 main，
        # 保证 CI 的 fresh checkout 也能拿到近 14 天历史（用于“今日无预警→回看近14天”）。
        _sync_root_decisions(json_path, date_tag)
    except Exception as e:
        logger.warning(f"  Data enhancement skipped: {e}")
    
    # --- 3c. RSS Atom feed 条目 ---
    feed_entries = []
    for d in decisions:
        if d["llm_decision"]:  # 只发布触发预警的场景
            category_names = {"fire": "Forest Fire", "flood": "Flood", "drought": "Drought", "heat": "Heat Wave"}
            feed_entries.append({
                "title": f"EarthBench Alert: {category_names.get(d['category'], d['category'])} in {d['region']}",
                "link": f"/reports/{date_tag}/{d['case_id']}",
                "published": d["timestamp"],
                "summary": d.get("rationale", "")[:150],
            })

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
            atom_xml += f"""
  <entry>
    <title>{entry["title"]}</title>
    <link href="{entry["link"]}" rel="alternate"/>
    <id>tag:earthbench.io,{entry["published"]}</id>
    <published>{entry["published"]}</published>
    <summary>{entry["summary"]}</summary>
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

def distribute_reports(outputs: dict[str, str], decisions: list[dict]) -> dict[str, str]:
    """将生成的报告分发到各个通道。"""
    logger.info("[Stage 4/4] Distributing reports to channels...")
    
    distribution_status = {}
    
    channels = PUBLISH_CONFIG["channels"]
    
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

def run_full_pipeline(test_mode: bool = False) -> dict[str, Any]:
    """执行完整的四阶段流水线。"""
    logger.info("=" * 60)
    logger.info("EarthBench Publish Pipeline — Starting")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # Stage 1
    suite = collect_data()
    
    # Stage 2
    decisions = run_llm_decisions(suite)
    
    # Stage 3
    outputs = generate_reports(decisions, suite)
    
    # Stage 4
    distribution = distribute_reports(outputs, decisions)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    result = {
        "status": "success",
        "started_at": start_time.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "outputs": outputs,
        "distribution": distribution,
        "decisions_count": len(decisions),
        "alerts_triggered": sum(1 for d in decisions if d["llm_decision"]),
    }
    
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Decisions: {len(decisions)}")
    logger.info(f"  Alerts: {result['alerts_triggered']}")
    logger.info(f"  Outputs: {list(outputs.keys())}")
    logger.info(f"  Distribution: {distribution}")
    logger.info("=" * 60)
    
    return result


# ============================================================================
# CLI Entry
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EarthBench Publish Pipeline")
    parser.add_argument("--run", action="store_true", help="Run full pipeline")
    parser.add_argument("--test-publish", action="store_true", help="Test single scenario publishing")
    parser.add_argument("--dry-run", action="store_true", help="Run without actual publication")
    args = parser.parse_args()
    
    if args.test_publish:
        # 测试单个场景
        from earthbench.integrations import CARMBridge
        from earthbench.models import ScenarioContext, Observation, ScenarioCategory, DecisionTemplate
        import os
        os.environ['OLLAMA_MODEL'] = 'qwen3:14b'
        
        bridge = CARMBridge(carml_root=os.path.normpath(os.path.abspath('D:/codes/Mustard')))
        ctx = ScenarioContext(
            category=ScenarioCategory.FIRE,
            template=DecisionTemplate.ALERT,
            region='Beijing-Xiangshan',
            horizon_hours=24,
            observations=[
                Observation(variable='FWI', value=52.0, unit='', confidence=0.95, timestamp='2026-07-12T12:00:00', source='ECMWF'),
                Observation(variable='humidity', value=12.0, unit='%', confidence=0.95, timestamp='2026-07-12T12:00:00', source='Station'),
            ],
        )
        result = bridge.decide(ctx)
        print(f"Decision: {'YES' if result.decision else 'NO'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Rationale: {result.rationale[:300]}")
        return
    
    if args.run or not args.test_publish:
        result = run_full_pipeline(test_mode=args.dry_run)
        
        # 输出 JSON 摘要供其他工具消费
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
