# Capability Packs — 能力包

此目录存储可复用的决策能力包（知识包/技能包/工具包），借鉴 SIM 项目的设计。

## 能力包格式

每个能力包是一个 JSON 文件，包含：
- `pack_id`: 唯一标识符
- `domain`: 领域（如 "fire"）
- `type`: 类型（knowledge / skill / tool / eval）
- `triggers`: 触发条件列表
- `inputs`: 输入变量
- `outputs`: 输出变量
- `version`: 版本号
- `status`: 状态（draft / active / deprecated）

## 示例

```json
{
    "pack_id": "fire-fwi-knowledge-001",
    "domain": "fire",
    "type": "knowledge",
    "description": "FWI（野火气象指数）知识库：各子指数的含义与阈值",
    "triggers": ["FWI"],
    "inputs": ["temperature", "humidity", "wind_speed", "rainfall"],
    "outputs": ["FWI_value", "risk_level"],
    "version": "1.0.0",
    "status": "draft"
}
```
