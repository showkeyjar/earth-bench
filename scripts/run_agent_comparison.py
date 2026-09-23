# -*- coding: utf-8 -*-
"""Agent 对比报告：rule vs cars（准确率 + 经济价值 V）。

用法:
    python scripts/run_agent_comparison.py [--category heat]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from earthbench.agents import MultiAlertAgent
from earthbench.benchmark import AlertBenchEvaluator
from earthbench.cars_agent import CarsMultiAgent
from earthbench.eval import ValueEvaluator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="all",
                    choices=["all", "fire", "flood", "drought", "heat"])
    args = ap.parse_args()

    bench = AlertBenchEvaluator()
    if args.category != "all":
        bench.raw_suite = [i for i in bench.raw_suite
                           if i["category"] == args.category]
        bench.test_cases = []
        bench.results = []
        bench._build_test_cases()

    agents = {
        "rule": MultiAlertAgent(),
        "cars": CarsMultiAgent(),
    }
    ve = ValueEvaluator()
    print(f"{'agent':6s} {'accuracy':>9s} {'V score':>8s} "
          f"{'exp/agent':>9s} {'exp/clim':>9s} {'exp/perf':>9s}")
    summary = {}
    for name, agent in agents.items():
        results = bench.evaluate_agent(agent)
        acc = sum(r.get("accuracy", 0) for r in results
                  if "accuracy" in r) / max(1, sum(1 for r in results
                                                   if "accuracy" in r))
        val = ve.score(results)
        summary[name] = {"accuracy": round(acc, 3), "value": val}
        print(f"{name:6s} {acc:9.3f} {val['value_score']:8.3f} "
              f"{val['expense_agent']:9.3f} {val['expense_climatology']:9.3f} "
              f"{val['expense_perfect']:9.3f}")

    # 阈值敏感性：P_TRIG（期望损失 P ≥ C/L）权衡
    print("\nP_TRIG sensitivity (heat cases, CARS agent):")
    sens = {}
    for p_trig in (0.1, 0.3, 0.5):
        results = bench.evaluate_agent(CarsMultiAgent(p_trig=p_trig))
        heat = [r for r in results if r.get("category") == "heat"]
        acc_h = sum(r["accuracy"] for r in heat) / max(1, len(heat))
        fn = sum(r.get("fn", 0) for r in heat)
        fp = sum(r.get("fp", 0) for r in heat)
        sens[p_trig] = {"heat_accuracy": round(acc_h, 3),
                        "misses": fn, "false_alarms": fp}
        print(f"  P_TRIG={p_trig:.1f}: heat acc={acc_h:.2f} "
              f"漏报={fn} 空报={fp}")
    summary["cars_p_trig_sensitivity"] = sens

    # 细节：热浪场景逐案
    print("\nheat case detail (cars):")
    results = bench.evaluate_agent(agents["cars"])
    for r in results:
        if r.get("category") == "heat":
            print(f"  {r.get('case_id','?'):28s} gt={str(r['ground_truth']):5s} "
                  f"pred={str(r['predicted']):5s} "
                  f"{'✓' if r['accuracy'] == 1 else '✗'}")

    out = Path(__file__).parent.parent / "agent_comparison.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()