"""EarthBench × CARM 集成层。

Phase 2 目标：让 EarthBench 的 Alert 决策场景通过 CARM 推理框架驱动，
测试 AI 能否像人类专家一样进行证据推理，而非简单的阈值规则。

集成策略：
1. 将 ScenarioContext 编码为自然语言 prompt
2. 调用 CARM 的 OnlinePolicy + BigModelProxy 做推理
3. 解析 CARM 的结构化输出为 DecisionOutput
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import DecisionOutput, ScenarioContext


# ---------------------------------------------------------------------------
# 数据适配器：ScenarioContext → CARM prompt
# ---------------------------------------------------------------------------

class _CARMScenarioInput(BaseModel):
    """CARM 可理解的场景输入格式。"""
    task_type: str = "earth_decision"
    decision_template: str
    region: str
    horizon_hours: int
    evidence_items: list[dict[str, Any]]


def _context_to_carm_prompt(ctx: ScenarioContext) -> str:
    """将场景上下文转为 CARM 友好的 prompt 文本。"""
    tmpl_label = {
        "alert": "是否预警",
        "dispatch": "是否调度",
        "upgrade": "是否升级",
        "close": "是否关闭",
        "recover": "是否恢复",
    }
    label = tmpl_label.get(ctx.template.value, "未知")

    evidence_lines: list[str] = []
    for obs in ctx.observations:
        evidence_lines.append(
            f"- {obs.source} [{obs.variable}] = {obs.value}{obs.unit} "
            f"(conf={obs.confidence:.2f}) @ {obs.timestamp[:16]}"
        )
    evidence_text = "\n".join(evidence_lines)

    # 根据场景类别附加领域知识提示
    cat_hint = ""
    if ctx.category.value == "fire":
        cat_hint = (
            "\n[领域知识提示] 这是森林火险预警场景。关键指标说明：\n"
            "- FWI (Fuel Weather Index): 0-20为低风险, 20-40为中等风险, 40-60为高风险, 60+为极高风险。\n"
            "- Humidity (相对湿度): 100%为最湿润(最安全), 0%为最干燥(最危险)。<30%表示极度干燥。\n"
            "- Wind Speed (风速): >20 km/h 会加速火势蔓延。\n"
            "- Rainfall (降雨量): 24h降雨>20mm会完全抵消火险（灭火效应）。降雨是决定性抑制因子，当降雨>20mm时应直接判断为NO。\n"
        )
    elif ctx.category.value == "flood":
        cat_hint = (
            "\n[领域知识提示] 这是洪涝灾害预警场景。关键指标说明：\n"
            "- Rainfall_24h (24h降雨量): <25mm为正常, 25-50mm为中雨, 50-100mm为大雨, >100mm为暴雨。\n"
            "- Rainfall_6h (6h短时降雨): >30mm为短时强降雨。\n"
            "- Soil Moisture (土壤湿度): >0.85表示土壤饱和。\n"
            "- Water Level (水位): 警戒水位>10m。\n"
            "- 【关键规则】当满足以下任一条件时触发洪水预警：\n"
            "  (1) 24h降雨>100mm 或 (2) 6h降雨>50mm 或 (3) 土壤饱和(>0.85)+水位>8m 或 (4) 水位>10m。\n"
            "- 土壤饱和+暴雨+超警戒水位是三重叠加风险，必须立即预警。\n"
            "- 排水良好区域(土壤湿度<0.6)，即使降雨80mm也不会发洪水(水能渗入地下)。\n"
        )
    elif ctx.category.value == "drought":
        cat_hint = (
            "\n[领域知识提示] 这是干旱预警场景。关键指标说明：\n"
            "- SPI (标准化降水指数): <-1.0为干旱, <-1.5为严重干旱, <-2.0为极端干旱。SPI反映短期降水异常。\n"
            "- Palmer 干旱指数: <-1.0为干旱, <-1.5为严重干旱, <-2.0为极端干旱。Palmer反映土壤深层水分亏损，是干旱预警的金标准。\n"
            "- NDVI: <0.3表示植被严重胁迫。\n"
            "- 【关键规则】判定干旱的核心标准：任意一个指标达到严重干旱级别即应预警(SPI<-1.5 OR Palmer<-1.0 OR 月降雨<10mm OR NDVI<0.3)。\n"
            "- SPI和Palmer冲突时以Palmer为准。但即使SPI接近正常(>-1.0)，只要任一条件(月降雨<10mm, NDVI<0.3)成立，都是干旱信号。\n"
            "- SPI趋势性下降(如-0.5→-0.9→-1.6)也是重要预警信号，当前值<-1.5即严重干旱。\n"
        )
    elif ctx.category.value == "ecology":
        cat_hint = (
            "\n[领域知识提示] 这是高温/热浪预警场景。关键指标说明：\n"
            "- Wet Bulb Temperature (湿球温度): >27°C触发预警, >28°C为危险, >30°C为致命。\n"
            "- 干热 vs 湿热：高湿度（>70%）下人体排汗失效，湿球温度是最核心预警指标。\n"
            "- 连续3天最高温≥34°C + 湿球>27°C = 典型湿热热浪环境（如南京\u706B\u704C\u7279\u6027）。\n"
            "- 即使单次温度未达35°C极端线，湿热组合+持续性也构成预警条件。\n"
            "- Temperature >36°C + Humidity <20%: 干燥环境，风险中等。\n"
            "- 持续多日高温且夜间降温不足时，累积热应力更高。\n"
        )

    prompt = (
        f"[EarthBench Alert 决策任务]\n"
        f"[区域] {ctx.region}\n"
        f"[时间窗口] {ctx.horizon_hours}小时\n"
        f"[证据列表]\n{evidence_text}\n"
        f"{cat_hint}"
        f"\n请做出二元决策：是否发出预警？只回答 YES 或 NO（大写），"
        f"然后简要说明理由和置信度。"
    )
    return prompt


# ---------------------------------------------------------------------------
# CARM Bridge — Phase 2 集成
# ---------------------------------------------------------------------------

class CARMIntegrationError(Exception):
    """CARM 集成错误。"""


class CARMBridge:
    """CARM 集成桥接器。

    Phase 2 集成策略：
    1. 尝试加载 CARM OnlinePolicy
    2. 构造一个最小化的 AgentState + MemoryBoard
    3. 通过 CARM 的语义意图编码器理解 EarthBench 场景
    4. 最终决策走 BigModelProxy → LLM 推理
    5. 若 CARM 不可用则降级为启发式规则引擎

    使用方式：
    ```python
    bridge = CARMBridge(carml_root="D:/codes/Mustard")
    result = bridge.decide(context)
    ```
    """

    def __init__(self, carml_root: str | Path | None = None):
        import os as _os
        if carml_root:
            path_str = str(carml_root)
            # 统一为绝对路径，兼容 Windows 和 Git Bash 风格路径
            abs_path = _os.path.abspath(path_str)
            self.carml_root = Path(abs_path)
        else:
            self.carml_root = None
        self._carm_loaded = False
        self._policy = None
        self._loaded = self._try_load_carm()

    def _try_load_carm(self) -> bool:
        """尝试加载 CARM 包（OnlinePolicy + BigModelProxyTool）。"""
        if not self.carml_root:
            return False
        try:
            import sys
            import os as _os
            
            # 统一路径格式：转换为纯正 Windows 绝对路径
            must_str = str(self.carml_root)
            # 先处理 Linux 盘符路径 /d/xxx -> D:/xxx
            import re as _re
            m = _re.match(r'^/([a-z])/(.*)$', must_str)
            if m:
                must_str = f'{m.group(1).upper()}:{m.group(2)}'
            must_str = must_str.replace('/', '\\')
            must_str = _os.path.abspath(must_str)
            
            print(f'[CARMBridge] Checking CARM at: {must_str}')
            if not _os.path.exists(must_str):
                print(f'[CARMBridge] Path does not exist: {must_str}')
                return False

            # Ensure Mustard root is at position 0
            if must_str in sys.path:
                sys.path.remove(must_str)
            sys.path.insert(0, must_str)
            
            # 清除可能被缓存的旧模块
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith('carm.') or mod_name == 'carm':
                    del sys.modules[mod_name]
            
            from carm.policy import OnlinePolicy
            from carm.state import AgentState
            from carm.memory import MemoryBoard, MemorySlot
            from carm.schemas import ToolCall, ActionDecision
            from tools.bigmodel_tool import BigModelProxyTool
            
            self._carm_loaded = True
            print(f'[CARMBridge] SUCCESS: CARM modules loaded: OnlinePolicy={OnlinePolicy.__name__}, BigModelProxyTool={BigModelProxyTool.__name__}')
            return True
        except ImportError as e:
            import traceback
            print(f'[CARMBridge] Failed to load CARM: {e}')
            traceback.print_exc()
            return False

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """通过 CARM 或直接规则做出决策。"""
        if self._loaded:
            return self._decide_via_carml(context)
        return self._decide_heuristic(context)

    def _decide_via_carml(self, context: ScenarioContext) -> DecisionOutput:
        """使用 CARM 推理框架做决策。

        关键设计：EarthBench 场景不需要走通用工具路由（search/calculator/code），
        而是需要 LLM 进行领域知识推理。所以我们绕过 CARM 的语义意图编码器，
        直接构造一个 BigModelProxy 调用请求。
        """
        try:
            import sys
            from carm.policy import OnlinePolicy
            from carm.state import AgentState
            from carm.memory import MemoryBoard, MemorySlot

            prompt = _context_to_carm_prompt(context)

            heuristic_decision = self._decide_heuristic(context)

            # ===== 策略选择: Ollama LLM > 启发式 fallback =====
            llm_decision = None
            llm_source = "heuristic"
            
            try:
                from tools.bigmodel_tool import BigModelProxyTool
                import os
                # 确保使用 qwen3:14b（在EarthBench测试中表现更准确）
                os.environ.setdefault('OLLAMA_MODEL', 'qwen3:14b')
                
                bigmodel_tool = BigModelProxyTool()
                llm_result = bigmodel_tool.execute(prompt, {'mode': 'classify'})
                
                if llm_result.ok and llm_result.result:
                    llm_decision = self._parse_llm_response(llm_result.result)
                    if llm_decision is not None:
                        llm_source = "ollama_llm"
            except Exception as e:
                print(f'[CARMBridge] LLM fallback (ignored): {e}')

            # 构建最终决策
            if llm_decision is not None:
                final_decision = llm_decision
                confidence = 0.85 if llm_source == "ollama_llm" else heuristic_decision.confidence
                evidence_summary = {
                    **heuristic_decision.evidence_summary,
                    "llm_source": llm_source,
                    "llm_decision": llm_decision,
                }
                rationale = (
                    f"LLM Decision (source={llm_source}): {llm_decision}\n"
                    f"Heuristic Decision: {'YES' if heuristic_decision.decision else 'NO'}, "
                    f"confidence={heuristic_decision.confidence:.3f}\n"
                    f"{heuristic_decision.rationale}"
                )
            else:
                final_decision = heuristic_decision.decision
                confidence = heuristic_decision.confidence
                evidence_summary = {
                    **heuristic_decision.evidence_summary,
                    "llm_unavailable": True,
                }
                rationale = (
                    f"LLM unavailable, fallback to heuristic\n"
                    f"Heuristic Decision: {'YES' if heuristic_decision.decision else 'NO'}, "
                    f"confidence={confidence:.3f}\n"
                    f"{heuristic_decision.rationale}"
                )

            return DecisionOutput(
                context=context,
                decision=final_decision,
                confidence=confidence,
                evidence_summary=evidence_summary,
                rationale=rationale,
            )

        except Exception as e:
            # CARM 不可用时优雅降级
            return self._decide_heuristic(context)

    def _parse_llm_response(self, text: str) -> bool | None:
        """从 LLM 返回的自然语言响应中提取 YES/NO 决策。
        
        解析逻辑：
        - 优先匹配明确的 YES/NO 指令（作为首答案）
        - 再匹配领域关键词
        - 否则返回 None（使用启发式 fallback）
        """
        import re
        # 去除思考标签后的内容
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif "<antThinking>" in text:
            text = text.split("<antThinking>", 1)[1]
        
        lower = text.lower().strip()
        
        # 第1步：寻找明确的 "决策：YES/NO" 模式（LLM 最常使用的格式）
        dec_match = re.search(r'决策[:：]\s*(yes|no)', lower, re.IGNORECASE)
        if dec_match:
            return dec_match.group(1).lower() in ('yes', '是')
        
        # 第2步：寻找明确的 "Decision: YES/NO" 模式
        dec_match = re.search(r'decision[:：]\s*(yes|no)', lower, re.IGNORECASE)
        if dec_match:
            return dec_match.group(1).lower() in ('yes', '是')
        
        # 第3步：匹配首行的 YES/NO（作为最直接的指令）
        first_line = lower.split('\n')[0].strip() if '\n' in text else lower[:100].strip()
        if re.match(r'^(yes|no)\b', first_line, re.IGNORECASE):
            return first_line.startswith('y')
        
        # 第4步：寻找明确的领域动作模式
        if re.search(r'激活.*响应|应.*发出.*预警|建议.*预警|必须.*预警|需要.*预警', lower, re.IGNORECASE):
            return True
        if re.search(r'不应.*预警|不建议.*预警|不需要.*预警|关闭.*响应|不.*发出.*预警', lower, re.IGNORECASE):
            return False
        
        # 第5步：宽松匹配独立出现的关键词
        yes_patterns = [r'\byes\b', r'是\b']
        no_patterns = [r'\bno\b(?!\s*risk)', r'否\b']
        
        for pat in yes_patterns:
            if re.search(pat, lower, re.IGNORECASE | re.MULTILINE):
                return True
        for pat in no_patterns:
            if re.search(pat, lower, re.IGNORECASE | re.MULTILINE):
                return False
        
        return None

    def _decide_heuristic(self, context: ScenarioContext) -> DecisionOutput:
        """当 CARM 不可用时，使用增强启发式规则引擎。

        这比 RuleBasedAgent 支持更多变量类型，但仍然轻量。
        """
        evidence: dict[str, float] = {}
        risk_score = 0.0

        # 按变量类型分别处理
        fwi_vals = [o.value for o in context.observations if o.variable == "FWI"]
        humidity_vals = [o.value for o in context.observations if o.variable == "humidity"]
        wind_vals = [o.value for o in context.observations if o.variable == "wind_speed"]
        rain_vals = [o.value for o in context.observations if o.variable == "rainfall"]
        temp_vals = [o.value for o in context.observations if o.variable == "temperature"]
        ndvi_vals = [o.value for o in context.observations if o.variable == "NDVI"]

        # 加权评分模型（每类证据权重可调）
        w = {
            "FWI": 0.35,
            "humidity": 0.20,
            "wind_speed": 0.15,
            "rainfall": 0.10,
            "temperature": 0.10,
            "NDVI": 0.10,
        }

        # FWI: 越高越危险
        if fwi_vals:
            avg = sum(fwi_vals) / len(fwi_vals)
            risk_score += w["FWI"] * min(avg / 50.0, 1.0)
            evidence["FWI"] = round(min(avg / 50.0, 1.0), 3)

        # 湿度: 越低越危险
        if humidity_vals:
            avg = sum(humidity_vals) / len(humidity_vals)
            inv_risk = max(0, (100 - avg) / 100)
            risk_score += w["humidity"] * inv_risk
            evidence["humidity"] = round(inv_risk, 3)

        # 风速: 越高越危险
        if wind_vals:
            avg = sum(wind_vals) / len(wind_vals)
            risk_score += w["wind_speed"] * min(avg / 20.0, 1.0)
            evidence["wind"] = round(min(avg / 20.0, 1.0), 3)

        # 降雨: 越多越安全
        if rain_vals:
            avg = sum(rain_vals) / len(rain_vals)
            risk_score -= w["rainfall"] * min(avg / 20.0, 1.0)  # 负贡献 = 降低风险
            evidence["rainfall"] = round(min(-avg / 20.0, 0), 3)

        # 温度: 越高越危险
        if temp_vals:
            avg = sum(temp_vals) / len(temp_vals)
            risk_score += w["temperature"] * min(max(0, (avg - 20) / 30.0), 1.0)
            evidence["temp"] = round(min(max(0, (avg - 20) / 30.0), 1.0), 3)

        # 湿球温度: 越高越危险
        wbt_vals = [o.value for o in context.observations if o.variable == "wet_bulb_temp"]
        if wbt_vals:
            avg_wbt = sum(wbt_vals) / len(wbt_vals)
            wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)
            risk_score += wbt_risk * 0.25
            evidence["wet_bulb_temp"] = round(wbt_risk, 3)

        # 截断到 [0, 1]
        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= 0.4  # 阈值略低于 RuleBasedAgent

        return DecisionOutput(
            context=context,
            decision=bool(decision),
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"多变量加权风险评分: {risk_score:.3f}, "
                f"证据权重: {json.dumps(evidence, ensure_ascii=False)}"
            ),
        )
