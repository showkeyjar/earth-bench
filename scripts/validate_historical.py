"""历史灾例回填验证 —— 真值函数对真实事件的外部锚定检验。

用法：
    python scripts/validate_historical.py

流程：
  1. 载入 get_historical_validation_suite()（3 真实灾害 + 1 合成对照，
     观测值与来源见每例 provenance）；
  2. 用 AlertBenchEvaluator 运行时推导 Ground Truth（独立标准查表），
     与「实际发生了什么」（文档化现实）对照 → gt_divergences 应为空；
  3. 规则 baseline（MultiAlertAgent）得分仅作参考（灾难级事件理应明显）；
  4. 生成 docs/historical-validation.md 验证报告（含来源 URL）。

退出码：全部一致 0；任何一例判定与实际不一致 → 1（CI 可用）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from earthbench.agents import MultiAlertAgent  # noqa: E402
from earthbench.benchmark import AlertBenchEvaluator  # noqa: E402
from earthbench.scenarios import (  # noqa: E402
    HISTORICAL_VALIDATION_FINDINGS,
    get_historical_validation_suite,
)

REPORT = Path(__file__).resolve().parents[1] / "docs" / "historical-validation.md"


def main() -> int:
    suite = get_historical_validation_suite()
    evaluator = AlertBenchEvaluator(suite=list(suite))
    agent = MultiAlertAgent()
    results = evaluator.evaluate_agent(agent)

    by_id = {r.get("case_id"): r for r in results if "error" not in r}
    # BatchEvaluator 异常结果不含 case_id，按与 suite 相同的顺序对齐
    errs = {i: r["error"] for i, r in enumerate(results) if "error" in r}
    n_ok = 0
    print("=" * 78)
    print("历史灾例回填验证（真实观测 → 独立标准真值 → 实际现实对照）")
    print("=" * 78)
    for i, item in enumerate(suite):
        cid = item["case_id"]
        prov = item["provenance"]
        decision, score, expl = item["_gt_fn"](item["observations"])
        doc = item["ground_truth"]
        match = decision == doc
        n_ok += int(match)
        if cid in by_id:
            base = by_id[cid].get("predicted")
            base_note = "一致" if base == doc else "偏离（仅供参考）"
        else:
            base = None
            base_note = f"n/a（agent 守卫：{errs.get(i, '未知错误')}）"
        print(f"\n[{cid}] {'MATCH' if match else '*** MISMATCH ***'}")
        print(f"  事件: {prov['event']}")
        print(f"  实际现实: {prov['documented_reality']}")
        print(f"  真值推导: alert={decision} score={score:.2f} "
              f"标准: {expl.get('standard', '-')}")
        print(f"    判据: {expl.get('detail', expl.get('reason', '-'))}")
        print(f"  baseline: predicted={base} （{base_note}）")

    print("\n" + "=" * 78)
    print(f"gt_divergences（推导 vs 文档现实不一致）: "
          f"{len(evaluator.gt_divergences)} 例")
    for d in evaluator.gt_divergences:
        print(f"  !! {d}")
    n_match = sum(
        1 for item in suite
        if item["_gt_fn"](item["observations"])[0] == item["ground_truth"])
    n_eval = len(by_id)
    base_acc = (
        sum(r["accuracy"] for r in by_id.values()) / n_eval if n_eval else None
    )
    print(f"判定一致率: {n_match}/{len(suite)}")
    if base_acc is not None:
        print(f"规则 baseline 参考（{n_eval}/{len(suite)} 例可评，其余被"
              f" ≥3 观测守卫跳过）: {base_acc:.0%}")
    else:
        print("规则 baseline 参考: 无可评用例")

    _write_report(suite, evaluator, by_id, errs, n_match, base_acc)

    if evaluator.gt_divergences or n_match != len(suite):
        print("验证失败：真值判定与实际现实存在分歧")
        return 1
    print("验证通过：独立标准真值与三场真实灾害 + 合成对照全部一致")
    print(f"报告已写入: {REPORT}")
    return 0


def _write_report(suite, evaluator, by_id, errs, n_match: float,
                  base_acc) -> None:
    lines = [
        "# 历史灾例回填验证报告",
        "",
        "> 生成自 `scripts/validate_historical.py`；套件定义见",
        "> `earthbench/scenarios.py::get_historical_validation_suite()`。",
        "> 验证逻辑：把公开报道的**真实观测值**灌入独立标准真值函数，",
        "> 检验推导判定与**实际发生了什么**（官方响应/实际灾情）一致。",
        "",
        f"- 判定一致：**{n_match:.0f}/{len(suite)}**；"
        f"gt_divergences：**{len(evaluator.gt_divergences)}**",
        (f"- 规则 baseline（参考，非验收；{len(by_id)}/{len(suite)} 例可评，"
         f"其余被 ≥3 观测守卫跳过）：{base_acc:.0%}"
         if base_acc is not None else
         "- 规则 baseline（参考，非验收）：无可评用例"),
        "",
        "| 用例 | 事件 | 关键文档观测 | 真值判定 | 实际现实 | 一致 | baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, item in enumerate(suite):
        cid = item["case_id"]
        prov = item["provenance"]
        decision, score, _ = item["_gt_fn"](item["observations"])
        obs_kv = "; ".join(
            f"{o['variable']}={o['value']}{o['unit']}"
            for o in item["observations"]
        )
        if cid in by_id:
            base = "✅" if by_id[cid].get("predicted") == item["ground_truth"] \
                else "⚠️ 偏离"
        else:
            base = f"n/a（守卫：{errs.get(i, '')}）"
        lines.append(
            f"| `{cid}` | {prov['event']} | {obs_kv} | "
            f"{'预警' if decision else '不预警'}（{score:.2f}） | "
            f"{'应预警' if item['ground_truth'] else '不应预警'} | "
            f"{'✅' if decision == item['ground_truth'] else '❌'} | {base} |"
        )
    lines += ["", "## 来源与口径披露", ""]
    for item in suite:
        prov = item["provenance"]
        lines += [
            f"### {item['case_id']} — {prov['event']}",
            "",
            f"- **实际现实**：{prov['documented_reality']}",
            f"- **观测口径**：{prov['obs_notes']}",
            "- **来源**：",
        ]
        lines += [f"  - {s}" for s in prov["sources"]]
        lines.append("")
    lines += [
        "## 方法说明",
        "",
        "- 真值函数（`infer_flood_ground_truth` / `infer_typhoon_ground_truth`",
        "  / `infer_cold_ground_truth` / `infer_snow_ground_truth` /",
        "  `infer_heatwave_ground_truth` / `infer_fire_ground_truth`）",
        "  只引用国标查表（GB/T 28592-2012 降水量等级、GB/T 19201-2006",
        "  热带气旋等级、GB/T 20484-2017 冷空气等级、GB/T 28592-2012",
        "  降雪等级附表、中央气象台高温预警信号色阶、GB/T 36743-2018",
        "  森林火险气象等级），与 agents.py 线性权重完全脱钩——本验证",
        "  是对「不自我认证」纪律的外部锚定。",
        "- 历史真灾例均为 True 方向（灾难确实发生）；False 方向由合成对照日",
        "  覆盖（三通道远离阈值）。误报判别力的系统性检验仍由对抗套件承担。",
        "- 历史套件不进入 83 用例主基准计数（验证报告性质，非难度分层用例）。",
        "",
        "## 回填过程中的口径发现（含已清偿与仍开放的口径问题）",
        "",
    ]
    for f in HISTORICAL_VALIDATION_FINDINGS:
        lines += [
            f"### {f['title']}",
            "",
            f["detail"],
            "",
        ]
        lines += [f"- 来源：{s}" for s in f["sources"]]
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
