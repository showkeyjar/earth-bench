# EarthBench 设计文档

## 1. 问题定义

当前 Earth System AI 评测（如 WeatherBench、GraphCast）聚焦于**预测精度**——即数值预测的准确性。

但真实世界中，决策者关心的不是数字本身，而是**这个数字意味着什么行动**。

EarthBench 解决的是从"观测数据"到"决策"的跨越，核心挑战：

1. **多源融合**：遥感、气象站、传感器、社交媒体
2. **时序理解**：不仅要知道"现在是什么"，还要知道"趋势如何变化"
3. **决策可验证**：二元决策（YES/NO）可以有 Ground Truth
4. **能力复用**：决策能力应模块化、可移植

## 2. 架构设计

```
┌─────────────────────────────────────────────────┐
│                    EarthBench                     │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  Scenarios   │  │ Decision    │  │ Eval      │ │
│  │  Management  │→ │ Agents      │→ │ Engine    │ │
│  └─────────────┘  └─────────────┘  └───────────┘ │
│         │                │                        │
│         ▼                ▼                        │
│  ┌──────────────────────────────────────────┐    │
│  │         Decision Templates (5 types)      │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │         Capability Packs (modular)        │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## 3. Phase 规划

### Phase 1 — FireBench（当前）
- 森林火险决策场景
- Rule-Based Agent baseline
- 5 决策模板验证
- 基础评测指标

### Phase 2 — LLM Agent Integration
- 整合 Mustard/CARM 推理框架
- LLM Decision Agent
- Prompt 工程优化

### Phase 3 — Capability System
- 能力包管理系统（借鉴 SIM）
- 经验回放与在线进化
- 治理机制

### Phase 4 — Multi-modal Agent
- 遥感图像理解
- 多模态融合（图像+数值+文本）
- 端到端决策 Agent

## 4. 数据流

```
原始数据 ─→ 观测结构化 ─→ 场景组装 ─→ Agent 推理 ─→ 决策输出 ─→ 评测
              │              │              │             │
              ▼              ▼              ▼             ▼
          MODIS/ECMWF   Context       Rule/LLM     Ground Truth
          Station        Template      Agent        Comparison
```

## 5. 与现有项目的关系

| 项目 | 关系 |
|------|------|
| WeatherBench | 差异化：预测精度 vs 决策质量 |
| GraphCast | 差异化：数值预报 vs 行动建议 |
| Mustard/CARM | 复用：Agent 推理框架 → Phase 2 整合 |
| SIM | 复用：能力载体与治理机制 → Phase 3 整合 |
