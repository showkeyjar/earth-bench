"""EarthBench 核心数据结构定义。

借鉴 SIM 项目的 PackManifest 设计，采用 Pydantic 保证类型安全。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionTemplate(Enum):
    """五大决策模板。"""

    ALERT = "alert"  # ① 是否预警？
    DISPATCH = "dispatch"  # ② 是否调度？
    UPGRADE = "upgrade"  # ③ 是否升级？
    CLOSE = "close"  # ④ 是否关闭？
    RECOVER = "recover"  # ⑤ 是否恢复？


class ScenarioCategory(str, Enum):
    """场景类别。"""

    FIRE = "fire"  # 森林防火
    FLOOD = "flood"  # 洪涝灾害
    DROUGHT = "drought"  # 干旱
    HEAT = "heat"  # 高温热浪
    ECOLOGY = "ecology"  # 生态风险
    LANDSLIDE = "landslide"  # 滑坡/泥石流（Phase A1 扩展）
    TYPHOON = "typhoon"  # 台风/大风（Phase A2 扩展）
    COLD = "cold"  # 寒潮/冰冻（Phase B1 扩展）
    SNOW = "snow"  # 暴雪/道路结冰（Phase B2 扩展）

    @classmethod
    def from_string(cls, value: str) -> ScenarioCategory:
        """将字符串安全地转换为 ScenarioCategory，未知值回退到 FIRE（并告警）。

        此前未知类别静默回退 FIRE，拼写错误（如 "haet"）会被当成火险场景
        处理且不留任何痕迹；保留回退行为以兼容，但显式记录告警。
        """
        cat = _CATEGORY_MAP.get(value)
        if cat is None:
            logging.getLogger(__name__).warning(
                "Unknown scenario category %r — falling back to FIRE "
                "(expected one of: fire/flood/drought/heat/ecology/landslide/typhoon/cold/snow)",
                value,
            )
            return cls.FIRE
        return cat


_CATEGORY_MAP: dict[str, ScenarioCategory] = {
    "fire": ScenarioCategory.FIRE,
    "flood": ScenarioCategory.FLOOD,
    "drought": ScenarioCategory.DROUGHT,
    "heat": ScenarioCategory.HEAT,
    "ecology": ScenarioCategory.ECOLOGY,
    "landslide": ScenarioCategory.LANDSLIDE,
    "typhoon": ScenarioCategory.TYPHOON,
    "cold": ScenarioCategory.COLD,
    "snow": ScenarioCategory.SNOW,
}


class Observation(BaseModel):
    """单一观测值（来自遥感、气象站、传感器等）。"""

    source: str  # 数据源（如 "MODIS", "ECMWF"）
    variable: str  # 变量名（如 "FWI", "NDVI", "rainfall"）
    value: float
    unit: str  # 单位
    timestamp: str  # ISO 8601
    confidence: float = 1.0  # 0-1


class ExposureProfile(BaseModel):
    """暴露与脆弱性画像（Phase B 扩展，docs/expansion-plan.md 层 2）。

    设计不变量：暴露只调制「该采取哪个层级的动作」（alert/dispatch），
    不改写物理真值——FWI 52 在无人区依然是极高火险，但动作等级不同。

    全部字段可选/缺省，向后兼容：不提供 exposure 的场景行为完全不变。
    分级查表（E0-E3）见 scenarios.py::infer_exposure_class（披露假设）。
    """

    population_density_class: str = "unknown"  # high / mid / low / none
    land_use: str = "unknown"  # urban / rural / forest / farmland
    critical_infrastructure: list[str] = Field(default_factory=list)  # 如 ["hospital"]
    vulnerable_group_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    outdoor_activity_level: str = "unknown"  # high(假期/农忙) / normal / low


class ScenarioContext(BaseModel):
    """场景上下文 — 多源时序观测集合。"""

    category: ScenarioCategory
    template: DecisionTemplate
    observations: list[Observation]
    region: str
    horizon_hours: int = 72  # 决策时间窗口
    exposure: ExposureProfile | None = None  # 缺省 → 无暴露画像，行为不变


class DecisionOutput(BaseModel):
    """决策输出。"""

    context: ScenarioContext
    decision: bool  # YES=True, NO=False
    confidence: float  # 0-1
    evidence_summary: dict[str, Any] = Field(
        default_factory=dict
    )  # 各证据对决策的贡献权重
    rationale: str = ""  # 自然语言推理过程


class CapabilityPack(BaseModel):
    """能力包（借鉴 SIM 项目的能力载体系统）。

    用于存储可复用的决策能力（知识包/技能包/工具包）。
    """

    pack_id: str
    domain: str
    type: str  # knowledge / skill / tool / eval
    description: str
    triggers: list[str]  # 触发条件
    inputs: list[str]
    outputs: list[str]
    version: str = "1.0.0"
    status: str = "draft"  # draft / active / deprecated
