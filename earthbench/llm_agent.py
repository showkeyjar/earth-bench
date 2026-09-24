"""LLM Agent 评测 harness —— 把基准的真正使用方接入。

README 架构中的「LLM Agent via CARM」环节此前只有规则 baseline 在跑；
本模块补齐三件事：

1. ``ScenarioPromptBuilder``：``ScenarioContext`` → 结构化提示词。
   观测表逐条嵌入（变量/数值/单位/时次），判定协议显式要求查表思维
   （非加权平均）、AND 复合缺一不触发、常态 ≠ 异常。
   阈值双模式：informed（默认，从 ``scenarios.py`` 常量单一来源生成
   国标/披露阈值摘要——与规则 baseline 同一信息类，对抗套件变成纯
   推理测试）/ blind（不给阈值，考 LLM 内化的标准知识）。
2. ``parse_llm_decision``：鲁棒解析（裸 JSON / 围栏 JSON / 散文
   YES-NO），解析失败显式计数不静默。
3. 后端：``CarmBackend``（本地 Qwen3.6-35B，直连 llama.cpp OpenAI
   兼容服务 127.0.0.1:8082——Mustard CARM bigmodel_proxy 的同一底层
   引擎；不走 CARM 路由端点，路由器会把判定提示词误判为检索类查询；
   网络失败优雅降级）
   与 ``MockLLMBackend``（确定性离线后端，管线冒烟用，非智能）。

``LLMAlertAgent.decide`` 满足评测接口，可直接进
``AlertBenchEvaluator.evaluate_agent`` 与规则 baseline / 平凡基线同表
对比（scripts/evaluate_llm_agent.py）。

口径披露：
- informed 模式给 LLM 的是与真值函数同源的阈值表，但**不给判定结果**
  ——测试的是推理组合（对抗套件的 AND 缺一/边界值陷阱正是为此设计）；
- blind 模式测试内化知识，预期分数更低，属不同赛道；
- CARM 后端 temperature=0 求稳定，本地模型仍有非确定性，报告以
  单次运行快照为准；
- CARM 的 /v1/chat/completions 是工具路由端点（检索/计算/代码），
  对本任务会把提示词误路由到 search（实测 8/8 解析失败）——因此
  直连其底层 llama.cpp 引擎（同模型同配置），这是接口适配披露。
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.request import Request, urlopen

from .models import DecisionOutput, ScenarioContext
from .scenarios import (
    COLD_DROP_24H,
    COLD_DROP_48H,
    COLD_DROP_72H,
    COLD_TMIN,
    FWI_LEVEL_EXTREME,
    FWI_LEVEL_HIGH,
    GUST_SAFETY_10,
    HUMIDITY_MUGGY,
    RAIN_COMPOUND,
    SNOW_BLIZZARD,
    SNOW_EXTREME_BLIZZARD,
    SNOW_MODERATE,
    SPI_DROUGHT_EXTREME,
    SPI_DROUGHT_SEVERE,
    TEMP_HEAT_ORANGE,
    TEMP_HEAT_RED,
    TEMP_MUGGY,
    WIND_LEVEL_8,
    WIND_LEVEL_12,
)

# 阈值摘要（单一来源：scenarios.py 常量；滑坡为披露经验阈值，与
# infer_landslide_ground_truth 文档一致）
CATEGORY_THRESHOLD_DIGEST: dict[str, str] = {
    "flood": (
        "24h降雨 ≥100mm 特大暴雨；≥50mm 暴雨；6h ≥50mm 短时强降雨；"
        "24h≥30mm 且土壤湿度 ≥0.85 饱和复合；水位 ≥10m"
    ),
    "typhoon": (
        f"持续风 ≥{WIND_LEVEL_12} m/s（台风级）；阵风 ≥{GUST_SAFETY_10} m/s"
        f"（设施安全线）；持续风 ≥{WIND_LEVEL_8} 且 24h降雨 ≥{RAIN_COMPOUND}mm"
        "（风雨耦合）"
    ),
    "cold": (
        f"24h日最低降幅 ≥{COLD_DROP_24H}°C 或 48h ≥{COLD_DROP_48H}°C 或 "
        f"72h ≥{COLD_DROP_72H}°C（任一窗口），且日最低 ≤{COLD_TMIN}°C"
        "（AND 双条件：常态低温无降幅不触发）"
    ),
    "snow": (
        f"24h降雪水当量 ≥{SNOW_EXTREME_BLIZZARD}mm 特大暴雪；"
        f"≥{SNOW_BLIZZARD}mm 暴雪；道路结冰复合：降雪 ≥{SNOW_MODERATE}mm"
        " 且路温 ≤0°C；气温 ≥0.5°C 的降水为雨不是雪"
    ),
    "heat": (
        f"日最高 ≥{TEMP_HEAT_RED}°C 红色；≥{TEMP_HEAT_ORANGE}°C 橙色；"
        f"≥{TEMP_MUGGY}°C 且湿度 ≥{HUMIDITY_MUGGY}% 闷热；"
        "湿球 ≥27°C 热应激"
    ),
    "fire": (
        f"FWI >{FWI_LEVEL_EXTREME} 极高火险；FWI {FWI_LEVEL_HIGH}-{FWI_LEVEL_EXTREME} "
        "高火险且湿度 <30% 或风速 ≥12 m/s；24h降雨 ≥30mm 彻底灭火（强制不预警）"
    ),
    "drought": (
        f"SPI ≤{SPI_DROUGHT_EXTREME} 特旱；≤{SPI_DROUGHT_SEVERE} 重旱；"
        "≤-1.0 中旱（GB/T 20481 查表）——且必须过影响门控（AND）："
        "持续 ≥60天 或 受旱面积 ≥40% 或 城市缺水率 ≥10%；"
        "短时/局地城市干旱（指数红但影响不成立）不预警"
    ),
    "landslide": (
        "高易发区：1h雨 ≥25mm 或 3h ≥50mm 激发雨强；3日有效降雨 ≥100mm "
        "且土壤湿度 ≥0.80；24h ≥50mm 且土壤 ≥0.75 且易发性 ≥2 级"
        "（经验阈值，披露口径）"
    ),
}

_DECISION_PROTOCOL = """【判定纪律】
1. 国标阈值查表思维：逐条对照阈值，不是加权平均——平均会稀释 AND 复合条件
2. AND 结构：复合条件（如「降雨 AND 土壤湿度」「降幅 AND 极值」）缺一即不触发
3. 常态 ≠ 异常：北方冬季常态低温、雨季正常降雨、干燥气候常态不构成预警
4. 单位与口径：先核对单位再比较；观测值是瞬时/累计/序列要区分
5. 仅当对人类活动或安全有实际威胁时才 YES"""

_OUTPUT_SPEC = """【输出格式】只输出一行 JSON，不要任何其他文字：
{"decision": "YES 或 NO", "confidence": 0.0到1.0, "evidence": {"观测变量名": 权重0到1}, "rationale": "一句话判定依据"}"""


class ScenarioPromptBuilder:
    """ScenarioContext → (system, user) 结构化提示词。"""

    def __init__(self, include_thresholds: bool = True):
        self.include_thresholds = include_thresholds

    def build(self, ctx: ScenarioContext) -> tuple[str, str]:
        category = ctx.category.value if hasattr(ctx.category, "value") else str(ctx.category)
        obs_lines = [
            f"- {o.variable} = {o.value} {o.unit} @ {o.timestamp}"
            f"（来源 {o.source}）"
            for o in ctx.observations
        ]
        exposure_note = ""
        if getattr(ctx, "exposure", None) is not None:
            exposure_dump = json.dumps(
                ctx.exposure.model_dump(), ensure_ascii=False, default=str
            )
            exposure_note = f"\n【暴露画像】{exposure_dump}"
        threshold_block = ""
        if self.include_thresholds:
            digest = CATEGORY_THRESHOLD_DIGEST.get(category, "（无摘要）")
            threshold_block = f"\n【{category} 阈值参考（国标/披露口径）】\n{digest}\n"

        system = (
            "你是环境风险预警决策 Agent。基于观测与标准阈值判定是否触发"
            "对人类安全的预警。严格按输出格式返回一行 JSON。"
        )
        user = (
            f"判断当前观测是否触发【{category}】预警。\n\n"
            f"【区域】{ctx.region}\n"
            f"【决策模板】{ctx.template.value}\n"
            f"【决策时间窗】{ctx.horizon_hours} 小时\n"
            f"{exposure_note}\n"
            f"【观测】\n" + "\n".join(obs_lines) + "\n\n"
            f"{_DECISION_PROTOCOL}\n"
            f"{threshold_block}\n"
            f"{_OUTPUT_SPEC}"
        )
        return system, user


def parse_llm_decision(
    text: str,
) -> tuple[bool, float, dict[str, Any], str, bool]:
    """解析 LLM 输出 → (decision, confidence, evidence, rationale, ok)。

    三级解析：整段 JSON → 去围栏/截取 JSON → 散文 YES/NO 兜底
    （兜底成功 ok=False，计入解析失败统计）。
    """
    text = (text or "").strip()

    def _from_obj(obj: dict[str, Any]) -> tuple[bool, float, dict, str]:
        raw_dec = str(obj.get("decision", obj.get("alert", ""))).strip().upper()
        decision = raw_dec.startswith(("YES", "是", "TRUE", "1"))
        conf = obj.get("confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5
        evidence = obj.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        rationale = str(obj.get("rationale", ""))[:500]
        return decision, conf, evidence, rationale

    # 1) 整段 JSON
    try:
        return (*_from_obj(json.loads(text)), True)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) 围栏 / 截取 JSON
    stripped = re.sub(r"```(?:json)?|```", "", text).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return (*_from_obj(json.loads(stripped[start:end + 1])), True)
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) 散文兜底
    m = re.search(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if m:
        decision = m.group(1).upper() == "YES"
        cm = re.search(r"(?:置信度|confidence)[:\s]*([0-9]*\.?[0-9]+)", text,
                       re.IGNORECASE)
        conf = 0.5
        if cm:
            try:
                conf = max(0.0, min(1.0, float(cm.group(1))))
            except ValueError:
                pass
        return decision, conf, {}, text[:200], False

    return False, 0.0, {}, text[:200], False


class CarmBackend:
    """本地 Qwen3.6-35B 后端（llama.cpp OpenAI 兼容服务，无云调用）。

    直连 ``http://127.0.0.1:8082/v1``——即 Mustard CARM bigmodel_proxy
    使用的同一底层引擎与同一模型。不走 CARM 的 /v1/chat/completions：
    该端点经工具路由器，会把判定提示词误判为检索类查询（实测路由至
    search 且 ok=false，8/8 解析失败）；基准评测需要原始补全协议，
    直连引擎层是干净路径且不改动 CARM 服务状态。
    """

    name = "carm"

    def __init__(self, base_url: str = "http://127.0.0.1:8082/v1",
                 model: str = "qwen3.6-35b", timeout: int = 300,
                 max_tokens: int = 1536, temperature: float = 0.0,
                 enable_thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def _build_payload(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Qwen3 系思考模型默认长链思考：思维链写 reasoning_content、
        # 答案写 content，1536 tokens 常在思考阶段耗尽导致 content 空
        # （实测 15/16 截断）。基准判定任务用非思考模式：62 tokens 出
        # 干净 JSON；思考模式仍可显式开启（--think 或 ctor 参数）。
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(self._build_payload(system, user)).encode()
        req = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # Qwen3.6 为思考模型：思维链在 reasoning_content、答案在 content；
        # 若 token 上限在思考阶段耗尽导致 content 为空，回落思维链文本
        # （解析器的 JSON 截取逻辑可从中恢复答案）
        if not content.strip():
            content = msg.get("reasoning_content") or ""
        return content


class MockLLMBackend:
    """确定性离线后端（管线冒烟，非智能）。

    从提示词观测行提取数值，用粗阈值（任意值 ≥40 触发）产生决策，
    以 JSON 返回——让评测管线、解析器、报告全链路离线可测。
    """

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        vals = [
            float(v)
            for _var, v in re.findall(r"- (\w+) = (-?\d+(?:\.\d+)?)", user)
        ]
        yes = any(v >= 40.0 for v in vals)
        return json.dumps(
            {
                "decision": "YES" if yes else "NO",
                "confidence": 0.9 if yes else 0.6,
                "evidence": {},
                "rationale": "mock-heuristic（≥40 触发，非智能）",
            },
            ensure_ascii=False,
        )


class LLMAlertAgent:
    """LLM 决策 Agent（满足 decide(ctx) -> DecisionOutput 评测接口）。"""

    def __init__(self, backend: Any | None = None,
                 include_thresholds: bool = True):
        self.backend = backend or CarmBackend()
        self.builder = ScenarioPromptBuilder(
            include_thresholds=include_thresholds
        )
        self.last_prompt: str | None = None
        self.last_raw: str | None = None
        self.parse_failures = 0
        self.backend_errors = 0

    @property
    def name(self) -> str:
        return f"llm-{getattr(self.backend, 'name', 'custom')}"

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        system, user = self.builder.build(context)
        self.last_prompt = user
        try:
            raw = self.backend.complete(system, user)
        except Exception as exc:  # noqa: BLE001 — 网络失败优雅降级
            self.backend_errors += 1
            return DecisionOutput(
                context=context,
                decision=False,
                confidence=0.0,
                evidence_summary={"error": f"backend-unavailable: {exc}"},
                rationale="后端不可用（本地 CARM 未启动/超时），降级为不预警",
            )
        self.last_raw = raw
        decision, conf, evidence, rationale, ok = parse_llm_decision(raw)
        if not ok:
            self.parse_failures += 1
        summary: dict[str, Any] = dict(evidence) if evidence else {}
        summary["parse_ok"] = ok
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=conf,
            evidence_summary=summary,
            rationale=rationale or "（LLM 未给出依据）",
        )


__all__ = [
    "CATEGORY_THRESHOLD_DIGEST",
    "CarmBackend",
    "LLMAlertAgent",
    "MockLLMBackend",
    "ScenarioPromptBuilder",
    "parse_llm_decision",
]
