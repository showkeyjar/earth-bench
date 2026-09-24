"""LLM Agent 基准评测 CLI —— 基准的真正使用方。

用法：
    python scripts/evaluate_llm_agent.py --backend mock --suite adversarial
    python scripts/evaluate_llm_agent.py --backend carm --suite basic,adversarial --limit 8
    python scripts/evaluate_llm_agent.py --backend carm --blind --suite adversarial

对比面板：LLM vs 规则 baseline vs 平凡基线（always/never/random），
与日报「基准体检」同一张表口径。报告写入 docs/llm-eval-<backend>.md。

注意：
- carm 后端为本地服务（http://127.0.0.1:8642，Qwen3.6-35B），
  10-30s/例——全量 83 例约 15-40 分钟，建议先 --limit 冒烟；
- 退出码恒 0（评测快照，不作门禁；真值分歧断言属 pytest 职责）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from earthbench.agents import MultiAlertAgent  # noqa: E402
from earthbench.benchmark import AlertBenchEvaluator  # noqa: E402
from earthbench.llm_agent import (  # noqa: E402
    CarmBackend,
    LLMAlertAgent,
    MockLLMBackend,
)
from earthbench.scenarios import (  # noqa: E402
    get_adversarial_suite,
    get_alert_benchmark_suite,
    get_close_suite,
    get_dispatch_suite,
    get_exposure_suite,
    get_recover_suite,
    get_upgrade_suite,
)
from earthbench.trivial_agents import (  # noqa: E402
    AlwaysAlertAgent,
    NeverAlertAgent,
    RandomAgent,
)

SUITES = {
    "basic": ("基础 40（baseline 满分层，无判别力）", get_alert_benchmark_suite),
    "adversarial": ("对抗 16（判别力主战场）", get_adversarial_suite),
    "dispatch": ("dispatch 4", get_dispatch_suite),
    "upgrade": ("upgrade 4", get_upgrade_suite),
    "close": ("close 5", get_close_suite),
    "recover": ("recover 4", get_recover_suite),
    "exposure": ("暴露 10（物理层）", get_exposure_suite),
}

REPORT = REPO / "docs"


def _run(agent, suite) -> tuple[float, int, int]:
    ev = AlertBenchEvaluator(suite=list(suite))
    res = [r for r in ev.evaluate_agent(agent) if "error" not in r]
    if not res:
        return 0.0, 0, 0
    acc = sum(r["accuracy"] for r in res) / len(res)
    fp = sum(1 for r in res if r["fp"])
    fn = sum(1 for r in res if r["fn"])
    return acc, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM Agent benchmark evaluation")
    ap.add_argument("--backend", choices=["mock", "carm"], default="mock")
    ap.add_argument("--suite", default="adversarial",
                    help=f"逗号分隔：{','.join(SUITES)} 或 all")
    ap.add_argument("--limit", type=int, default=0,
                    help="每套件截取前 N 例（0=全量；carm 冒烟建议 8）")
    ap.add_argument("--blind", action="store_true",
                    help="盲模式：提示词不含阈值摘要（考内化知识）")
    ap.add_argument("--timeout", type=int, default=300,
                    help="CARM 单例超时秒（默认 300）")
    ap.add_argument("--think", action="store_true",
                    help="开启 Qwen3.6 思考模式（默认关：长链会耗尽 token）")
    args = ap.parse_args()

    names = list(SUITES) if args.suite == "all" else args.suite.split(",")
    for n in names:
        if n not in SUITES:
            print(f"未知套件: {n}；可选 {','.join(SUITES)}")
            return 2

    backend = MockLLMBackend() if args.backend == "mock" else CarmBackend(
        timeout=args.timeout, enable_thinking=args.think)
    llm = LLMAlertAgent(backend=backend, include_thresholds=not args.blind)
    mode = "blind（不给阈值）" if args.blind else "informed（给阈值摘要）"

    print(f"backend={llm.name} mode={mode} suites={names} limit={args.limit}")
    lines = [
        f"# LLM Agent 基准评测（{llm.name}）",
        "",
        f"- 后端：{llm.name}（{'本地 Mustard CARM / Qwen3.6-35B，无云调用' if args.backend == 'carm' else '确定性离线 mock，管线冒烟'}）",
        f"- 模式：{mode}",
        f"- 套件：{', '.join(names)}；limit={args.limit or '全量'}",
        "",
        "| 套件 | LLM | 规则 baseline | always | never | random | LLM FP/FN | 解析失败 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for key in names:
        label, loader = SUITES[key]
        suite = list(loader())
        if args.limit:
            suite = suite[:args.limit]
            label = f"{label.split('（')[0]}（前 {len(suite)} 例）"
        llm.parse_failures = 0
        acc, fp, fn = _run(llm, suite)
        r_acc, _, _ = _run(MultiAlertAgent(), suite)
        a_acc, _, _ = _run(AlwaysAlertAgent(), suite)
        n_acc, _, _ = _run(NeverAlertAgent(), suite)
        d_acc, _, _ = _run(RandomAgent(), suite)
        row = (f"| {label} | {acc:.0%} | {r_acc:.0%} | {a_acc:.0%} | "
               f"{n_acc:.0%} | {d_acc:.0%} | {fp}/{fn} | "
               f"{llm.parse_failures} |")
        lines.append(row)
        print(row)

    lines += [
        "",
        "## 首例提示词（审计用）",
        "",
        "```",
        (llm.last_prompt or "（未运行）")[:1500],
        "```",
        "",
        "## 首例原始输出（审计用）",
        "",
        "```",
        (llm.last_raw or "（未运行）")[:800],
        "```",
    ]
    out = REPORT / f"llm-eval-{args.backend}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
