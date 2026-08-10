"""EarthBench CLI 入口。"""

from __future__ import annotations

import argparse
import json

from earthbench.scenarios import ScenarioStore
from earthbench.agents import RuleBasedAgent, MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.integrations import CARMBridge


def main():
    parser = argparse.ArgumentParser(
        description="EarthBench CLI — Earth Decision Intelligence Benchmark"
    )
    parser.add_argument("--demo", action="store_true", help="Run single-scenario demo")
    parser.add_argument(
        "--benchmark",
        "--bench",
        action="store_true",
        help="Run full AlertBench suite (fire+flood+drought+heat)",
    )
    parser.add_argument(
        "--agent",
        choices=["rule", "carm"],
        default="rule",
        help="Agent type to evaluate (default: rule)",
    )
    parser.add_argument(
        "--carm-root",
        type=str,
        default=None,
        help="Path to Mustard/CARM repository (for --agent carm)",
    )
    parser.add_argument(
        "--category",
        choices=["all", "fire", "flood", "drought", "heat"],
        default="all",
        help="Filter by category (default: all)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run the publish pipeline (data collection → LLM decision → report generation → distribution)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run standalone scenario evaluation from JSON input",
    )
    parser.add_argument(
        "--eval-input",
        type=str,
        default=None,
        help="Path to JSON file with observations for --eval mode",
    )
    args = parser.parse_args()

    if args.publish:
        from earthbench.publish_pipeline import run_full_pipeline

        result = run_full_pipeline()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.demo:
        run_demo()
    elif args.benchmark:
        run_alert_benchmark(args.agent, args.carm_root, args.category)
    elif args.eval:
        run_eval_mode(args.eval_input)


def run_demo():
    """运行演示：一个森林火险场景的规则 Agent 评测。"""
    print("=" * 60)
    print("EarthBench Demo — Forest Fire Decision Benchmark")
    print("=" * 60)

    store = ScenarioStore()
    context = store.load_fire_scenario(
        scenario_id="demo-fire-001",
        region="Beijing-Xiangshan",
        horizon_hours=72,
        observations=[
            {
                "source": "ECMWF",
                "variable": "FWI",
                "value": 42.0,
                "unit": "",
                "timestamp": "2026-07-11T12:00:00+08:00",
                "confidence": 0.95,
            },
            {
                "source": "ECMWF",
                "variable": "FWI",
                "value": 38.0,
                "unit": "",
                "timestamp": "2026-07-10T12:00:00+08:00",
                "confidence": 0.95,
            },
            {
                "source": "Station",
                "variable": "humidity",
                "value": 12.0,
                "unit": "%",
                "timestamp": "2026-07-11T12:00:00+08:00",
                "confidence": 0.98,
            },
            {
                "source": "Station",
                "variable": "wind_speed",
                "value": 16.0,
                "unit": "m/s",
                "timestamp": "2026-07-11T12:00:00+08:00",
                "confidence": 0.92,
            },
            {
                "source": "MODIS",
                "variable": "temperature",
                "value": 35.0,
                "unit": "°C",
                "timestamp": "2026-07-11T12:00:00+08:00",
                "confidence": 0.90,
            },
        ],
    )

    agent = RuleBasedAgent(
        fwi_threshold=38.0, humidity_threshold=15.0, wind_threshold=15.0
    )
    output = agent.decide(context)

    print("\n决策详情：")
    print(f"  区域：{context.region}")
    print(f"  决策：{'需要预警 ✓' if output.decision else '无需预警'}")
    print(f"  置信度：{output.confidence:.2f}")
    print(f"  证据：{json.dumps(output.evidence_summary, indent=2)}")
    print(f"  推理：{output.rationale}")
    print("\n" + "=" * 60)


def run_alert_benchmark(
    agent_type: str, carm_root: str | None, category_filter: str = "all"
):
    """运行 AlertBench 基准评测。"""
    print("=" * 60)
    print("AlertBench — Full Benchmark (Fire + Flood + Drought + Heat)")
    print("=" * 60)

    # 选择 Agent
    if agent_type == "rule":
        agent = MultiAlertAgent(
            fwi_threshold=40.0,
            humidity_threshold=20.0,
            wind_threshold=12.0,
            rainfall_suppress=10.0,
        )
        agent_name = "MultiAlertAgent (4 categories)"
    elif agent_type == "carm":
        if not carm_root:
            print("[ERROR] --agent carm requires --carm-root to be set.")
            return
        bridge = CARMBridge(carm_root=carm_root)
        agent = bridge
        agent_name = f"CARM ({'LLM' if bridge._loaded else 'heuristic-fallback'})"
    else:
        print(f"Unknown agent type: {agent_type}")
        return

    print(f"\n评测 Agent: {agent_name}")
    print(f"场景过滤: {category_filter}")

    # 运行评测
    bench_eval = AlertBenchEvaluator()

    if category_filter != "all":
        # 重新初始化 evaluator 并过滤 raw_suite
        bench_eval.raw_suite = [
            item for item in bench_eval.raw_suite if item["category"] == category_filter
        ]
        bench_eval.test_cases = []
        bench_eval.results = []
        bench_eval._build_test_cases()

    bench_eval.evaluate_agent(agent)

    # 输出汇总
    report = bench_eval.summary_report()
    print(f"\n{'=' * 55}")
    print(f"AlertBench 评测报告 (v{report['version']})")
    print(f"{'=' * 55}")
    print(f"  总场景数：{report['total_cases']}")
    print(f"  整体准确率：{report['overall_accuracy']:.2%}")

    print("\n  按场景类别：")
    for cat, stats in report["by_category"].items():
        if stats["total"] > 0:
            print(
                f"    {cat:10s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
            )

    print("\n  按难度分级：")
    for dl, stats in report["by_difficulty"].items():
        if stats["total"] > 0:
            label = {"L1": "Easy", "L2": "Medium", "L3": "Hard", "L4": "Beyond"}.get(
                dl, dl
            )
            print(
                f"    {dl} ({label:7s}): {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
            )

    print("\n  各场景详情：")
    for case in report["cases"]:
        print(
            f"    {case['id']:35s} {case['difficulty']:4s} {case['category']:7s} GT={case['ground_truth']}"
        )

    print(f"\n{'=' * 60}")


def run_eval_mode(eval_input: str | None):
    """独立评测模式：从文件或标准输入加载场景，运行评测。"""
    import sys
    from earthbench.models import Observation, ScenarioCategory

    # 读取输入
    if eval_input:
        with open(eval_input, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("Enter JSON observations (end with empty line or Ctrl-D):")
        text = sys.stdin.read()
        data = json.loads(text)

    observations_raw = data.get("observations", [])
    obs_list = [Observation(**o) for o in observations_raw]

    cat_str = data.get("category", "fire")
    category = ScenarioCategory.from_string(cat_str)

    region = data.get("region", "unknown")
    horizon = data.get("horizon_hours", 72)

    store = ScenarioStore()
    ctx = store.load_scenario_from_dict(
        scenario_id=data.get("scenario_id", "eval-stdin"),
        region=region,
        horizon_hours=horizon,
        observations=obs_list,
        category=category,
    )

    agent_type = data.get("agent", "multi")
    if agent_type == "fire":
        from earthbench.agents import FireAlertAgent

        agent = FireAlertAgent()
    elif agent_type == "flood":
        from earthbench.agents import FloodAlertAgent

        agent = FloodAlertAgent()
    elif agent_type == "drought":
        from earthbench.agents import DroughtAlertAgent

        agent = DroughtAlertAgent()
    elif agent_type == "heat":
        from earthbench.agents import HeatWaveAlertAgent

        agent = HeatWaveAlertAgent()
    elif agent_type == "llm":
        from earthbench.agents import LLMDecisionAgent

        agent = LLMDecisionAgent()
    else:
        from earthbench.agents import MultiAlertAgent

        agent = MultiAlertAgent()

    result = agent.decide(ctx)

    output = {
        "scenario_id": data.get("scenario_id", ctx.region),
        "region": ctx.region,
        "category": category.value,
        "observations_count": len(obs_list),
        "decision": {"yes": result.decision, "no": not result.decision},
        "confidence": round(result.confidence, 4),
        "evidence_summary": result.evidence_summary,
        "rationale": result.rationale,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
