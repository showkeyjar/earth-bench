#!/usr/bin/env python3
"""
公众号推文摘要生成器 - 从每日 JSON 日报提取关键信息
生成适合微信推送的简短摘要文本
"""

import json
import sys
from datetime import datetime, timezone, timedelta

# China Standard Time
CST = timezone(timedelta(hours=8))


def load_latest_report():
    """Load the latest decision report."""
    today = datetime.now(CST).strftime("%Y%m%d")
    for fname in [f"decisions_{today}.json", "latest.json"]:
        try:
            with open(f"published_reports/{fname}") as f:
                data = json.load(f)
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None


def extract_summary(data):
    """Extract key information for WeChat article."""
    if not data or "decisions" not in data:
        return None

    decisions = data.get("decisions", [])

    # Count alerts vs safe
    alerts = [d for d in decisions if d.get("llm_decision", False)]
    safe_count = len(decisions) - len(alerts)

    # Group alerts by category
    alert_categories = {}
    for d in alerts:
        cat = d.get("category", "unknown")
        region = d.get("region_cn", d.get("region", "?"))
        if cat not in alert_categories:
            alert_categories[cat] = []
        alert_categories[cat].append(
            {"region": region, "confidence": d.get("confidence", 0)}
        )

    # Build summary text
    publish_date = datetime.now(CST).strftime("%Y年%m月%d日")
    total = len(decisions)
    alert_count = len(alerts)
    avg_conf = (
        sum(d.get("confidence", 0) for d in decisions) / max(len(decisions), 1) * 100
    )

    summary_lines = [
        f"📊 EarthBench · {publish_date} 环境风险日报",
        "",
        f"今日监测 {total} 个区域，触发 {alert_count} 次预警，"
        f"{safe_count} 个区域安全，"
        f"平均置信度 {avg_conf:.0f}%。",
        "",
    ]

    if alert_categories:
        summary_lines.append("⚠️ 重点关注:")
        emoji_map = {"fire": "🔥", "flood": "🌊", "drought": "🏜️", "heat": "☀️"}
        for cat, items in alert_categories.items():
            cat_name = {
                "fire": "森林火险",
                "flood": "洪涝灾害",
                "drought": "干旱",
                "heat": "高温热浪",
            }
            emoji = emoji_map.get(cat, "")
            name = cat_name.get(cat, cat)
            for item in sorted(items, key=lambda x: x["confidence"], reverse=True)[:3]:
                conf = int(item["confidence"] * 100)
                summary_lines.append(f"  • {item['region']}: {name} {emoji}({conf}%)")
        summary_lines.append("")

    summary_lines.append("完整报告: https://earth-ai.fun")
    summary_lines.append("#EarthBench #AI决策智能 #环境风险 #防灾减灾")

    return "\n".join(summary_lines)


if __name__ == "__main__":
    data = load_latest_report()
    summary = extract_summary(data)
    if summary:
        print(summary)
        sys.exit(0)
    else:
        print("ERROR: No report data available", file=sys.stderr)
        sys.exit(1)
