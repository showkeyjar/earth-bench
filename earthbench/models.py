"""EarthBench 核心数据结构定义。

借鉴 SIM 项目的 PackManifest 设计，采用 Pydantic 保证类型安全。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DecisionTemplate(Enum):
    """五大决策模板。"""
    ALERT = "alert"           # ① 是否预警？
    DISPATCH = "dispatch"     # ② 是否调度？
    UPGRADE = "upgrade"       # ③ 是否升级？
    CLOSE = "close"           # ④ 是否关闭？
    RECOVER = "recover"       # ⑤ 是否恢复？


class ScenarioCategory(str, Enum):
    """场景类别。"""
    FIRE = "fire"             # 森林防火
    FLOOD = "flood"           # 洪涝灾害
    DROUGHT = "drought"       # 干旱
    ECOLOGY = "ecology"       # 生态风险


class Observation(BaseModel):
    """单一观测值（来自遥感、气象站、传感器等）。"""
    source: str               # 数据源（如 "MODIS", "ECMWF"）
    variable: str             # 变量名（如 "FWI", "NDVI", "rainfall"）
    value: float
    unit: str                 # 单位
    timestamp: str            # ISO 8601
    confidence: float = 1.0   # 0-1


class ScenarioContext(BaseModel):
    """场景上下文 — 多源时序观测集合。"""
    category: ScenarioCategory
    template: DecisionTemplate
    observations: list[Observation]
    region: str
    horizon_hours: int = 72   # 决策时间窗口


class DecisionOutput(BaseModel):
    """决策输出。"""
    context: ScenarioContext
    decision: bool            # YES=True, NO=False
    confidence: float         # 0-1
    evidence_summary: dict[str, Any] = Field(default_factory=dict)  # 各证据对决策的贡献权重
    rationale: str = ""       # 自然语言推理过程


class CapabilityPack(BaseModel):
    """能力包（借鉴 SIM 项目的能力载体系统）。

    用于存储可复用的决策能力（知识包/技能包/工具包）。
    """
    pack_id: str
    domain: str
    type: str                 # knowledge / skill / tool / eval
    description: str
    triggers: list[str]       # 触发条件
    inputs: list[str]
    outputs: list[str]
    version: str = "1.0.0"
    status: str = "draft"     # draft / active / deprecated
