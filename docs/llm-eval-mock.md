# LLM Agent 基准评测（llm-mock）

- 后端：llm-mock（确定性离线 mock，管线冒烟）
- 模式：informed（给阈值摘要）
- 套件：basic, adversarial, dispatch, upgrade, close, recover, exposure；limit=全量

| 套件 | LLM | 规则 baseline | always | never | random | LLM FP/FN | 解析失败 |
|---|---|---|---|---|---|---|---|
| 基础 40（baseline 满分层，无判别力） | 68% | 100% | 65% | 35% | 60% | 7/6 | 0 |
| 对抗 16（判别力主战场） | 38% | 0% | 50% | 50% | 56% | 6/4 | 0 |
| dispatch 4 | 50% | 50% | 50% | 50% | 50% | 1/1 | 0 |
| upgrade 4 | 50% | 50% | 50% | 50% | 50% | 2/0 | 0 |
| close 5 | 60% | 60% | 60% | 40% | 80% | 1/1 | 0 |
| recover 4 | 75% | 50% | 50% | 50% | 50% | 0/1 | 0 |
| 暴露 10（物理层） | 100% | 100% | 90% | 10% | 80% | 0/0 | 0 |

## 首例提示词（审计用）

```
判断当前观测是否触发【typhoon】预警。

【区域】Xiamen-Fujian
【决策模板】alert
【决策时间窗】72 小时

【暴露画像】{"population_density_class": "mid", "land_use": "urban", "critical_infrastructure": [], "vulnerable_group_ratio": null, "outdoor_activity_level": "unknown"}
【观测】
- wind_speed = 17.1 m/s @ 2026-07-26T12:00:00+08:00（来源 Station）
- wind_gust = 24.4 m/s @ 2026-07-26T12:00:00+08:00（来源 Station）
- rainfall_24h = 60.0 mm @ 2026-07-26T12:00:00+08:00（来源 CMA）

【判定纪律】
1. 国标阈值查表思维：逐条对照阈值，不是加权平均——平均会稀释 AND 复合条件
2. AND 结构：复合条件（如「降雨 AND 土壤湿度」「降幅 AND 极值」）缺一即不触发
3. 常态 ≠ 异常：北方冬季常态低温、雨季正常降雨、干燥气候常态不构成预警
4. 单位与口径：先核对单位再比较；观测值是瞬时/累计/序列要区分
5. 仅当对人类活动或安全有实际威胁时才 YES

【typhoon 阈值参考（国标/披露口径）】
持续风 ≥32.7 m/s（台风级）；阵风 ≥24.5 m/s（设施安全线）；持续风 ≥17.2 且 24h降雨 ≥50.0mm（风雨耦合）

【输出格式】只输出一行 JSON，不要任何其他文字：
{"decision": "YES 或 NO", "confidence": 0.0到1.0, "evidence": {"观测变量名": 权重0到1}, "rationale": "一句话判定依据"}
```

## 首例原始输出（审计用）

```
{"decision": "YES", "confidence": 0.9, "evidence": {}, "rationale": "mock-heuristic（≥40 触发，非智能）"}
```