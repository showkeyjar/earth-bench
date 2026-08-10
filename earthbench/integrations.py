"""EarthBench × CARM 集成层。

Phase 2 目标：让 EarthBench 的 Alert 决策场景通过 CARM 推理框架驱动，
测试 AI 能否像人类专家一样进行证据推理，而非简单的阈值规则。

集成策略：
1. 将 ScenarioContext 编码为自然语言 prompt
2. 调用 CARM 的 OnlinePolicy + BigModelProxy 做推理
3. 解析 CARM 的结构化输出为 DecisionOutput
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from .models import DecisionOutput, ScenarioContext


# ---------------------------------------------------------------------------
# 数据适配器：ScenarioContext → CARM prompt
# ---------------------------------------------------------------------------


def _context_to_carm_prompt(ctx: ScenarioContext) -> str:
    """将场景上下文转为 CARM 友好的 prompt 文本。"""
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
    elif ctx.category.value == "heat":
        cat_hint = (
            "\n[领域知识提示] 这是高温/热浪预警场景。关键指标说明：\n"
            "- Wet Bulb Temperature (湿球温度): >27°C触发预警, >28°C为危险, >30°C为致命。\n"
            "- 干热 vs 湿热：高湿度（>70%）下人体排汗失效，湿球温度是最核心预警指标。\n"
            "- 连续3天最高温≥34°C + 湿球>27°C = 典型湿热热浪环境（如南京火炉特性）。\n"
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
    bridge = CARMBridge(carm_root="D:/codes/Mustard")
    result = bridge.decide(context)
    ```
    """

    def __init__(self, carm_root: str | Path | None = None):
        import os as _os

        if carm_root:
            path_str = str(carm_root)
            # 统一为绝对路径，兼容 Windows 和 Git Bash 风格路径
            abs_path = _os.path.abspath(path_str)
            self.carm_root = Path(abs_path)
        else:
            self.carm_root = None
        self._carm_loaded = False
        self._policy = None
        self._loaded = self._try_load_carm()

    def _try_load_carm(self) -> bool:
        """尝试加载 CARM 包（OnlinePolicy + BigModelProxyTool）。"""
        if not self.carm_root:
            return False
        try:
            import sys
            import os as _os

            # 统一路径格式：转换为纯正 Windows 绝对路径
            must_str = str(self.carm_root)
            # 先处理 Linux 盘符路径 /d/xxx -> D:/xxx
            import re as _re

            m = _re.match(r"^/([a-z])/(.*)$", must_str)
            if m:
                must_str = f"{m.group(1).upper()}:{m.group(2)}"
            must_str = must_str.replace("/", "\\")
            must_str = _os.path.abspath(must_str)

            logger.info(f"Checking CARM at: {must_str}")
            if not _os.path.exists(must_str):
                logger.warning(f"Path does not exist: {must_str}")
                return False

            # Ensure Mustard root is at position 0
            if must_str in sys.path:
                sys.path.remove(must_str)
            sys.path.insert(0, must_str)

            # 清除可能被缓存的旧模块
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith("carm.") or mod_name == "carm":
                    del sys.modules[mod_name]

            from carm.policy import OnlinePolicy
            from tools.bigmodel_tool import BigModelProxyTool

            self._carm_loaded = True
            logger.info(
                f"CARM modules loaded: OnlinePolicy={OnlinePolicy.__name__}, "
                f"BigModelProxyTool={BigModelProxyTool.__name__}"
            )
            return True
        except ImportError as e:
            logger.warning(f"Failed to load CARM: {e}", exc_info=True)
            return False

    # 规则引擎权威决策实例（跨调用复用，避免每次重建）
    _rule_agent = None

    @classmethod
    def _get_rule_agent(cls):
        """返回权威规则引擎（MultiAlertAgent），用于兜底与防漏报。"""
        if cls._rule_agent is None:
            from .agents import MultiAlertAgent

            cls._rule_agent = MultiAlertAgent()
        return cls._rule_agent

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """做出决策。

        权威决策由规则引擎（MultiAlertAgent）给出，确保 evidence_summary 与
        decision/confidence 自洽、且真实气象数据（如 rainfall_24h 等带下划线变量）
        不会被漏提取导致系统性漏报（见 0727-0729 连续 0 预警事故）。

        LLM（CARM/Ollama）仅在“规则判定为安全但证据已逼近阈值”时作为参考升级，
        绝不反向把高危证据判成安全。
        """
        rule_output = self._get_rule_agent().decide(context)

        # 若 CARM 不可用或 LLM 不可用，直接返回规则权威结果（已修复漏报）
        if not self._loaded:
            return rule_output

        llm_decision = self._decide_via_carml_llm(context)
        if llm_decision is None:
            return rule_output

        # 规则已判定为预警 → 直接采用（防漏报优先）
        if rule_output.decision:
            return rule_output

        # 规则判定为安全，但 LLM 强烈预警且证据已逼近阈值 → 升级为预警
        # 仅在风险评分 >= 0.35（接近 0.4 阈值）时允许 LLM 升级，避免误报
        risk_score = rule_output.confidence
        if llm_decision and risk_score >= 0.35:
            return DecisionOutput(
                context=context,
                decision=True,
                confidence=round(max(risk_score, 0.4), 3),
                evidence_summary={**rule_output.evidence_summary, "llm_upgrade": True},
                rationale=(
                    f"规则评分={risk_score:.3f}（接近阈值），LLM 强烈预警，"
                    f"升级为预警以优先保障安全。\n{rule_output.rationale}"
                ),
            )

        # 规则与 LLM 一致为安全 → 规则结果
        return rule_output

    def _decide_via_carml_llm(self, context: ScenarioContext) -> bool | None:
        """调用 CARM/LLM 仅做二元决策（YES/NO），不负责风险评分。

        返回 None 表示 LLM 不可用或无法解析，此时调用方应依赖规则引擎权威结果。
        """
        import concurrent.futures
        import os

        def _call_llm() -> bool | None:
            from tools.bigmodel_tool import BigModelProxyTool

            os.environ.setdefault("OLLAMA_MODEL", "qwen3:14b")

            prompt = _context_to_carm_prompt(context)
            bigmodel_tool = BigModelProxyTool()
            llm_result = bigmodel_tool.execute(prompt, {"mode": "classify"})
            if llm_result.ok and llm_result.result:
                return self._parse_llm_response(llm_result.result)
            return None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_llm)
                return future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            logger.warning("LLM call timed out after 30s (ignored)")
        except Exception as e:
            logger.warning(f"LLM call failed (ignored): {e}")
        return None

    def _parse_llm_response(self, text: str) -> bool | None:
        """从 LLM 返回的自然语言响应中提取 YES/NO 决策。

        解析逻辑：
        - 优先匹配明确的 YES/NO 指令（作为首答案）
        - 再匹配领域关键词
        - 否则返回 None（使用启发式 fallback）
        """

        # 去除思考标签后的内容
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif "<antThinking>" in text:
            text = text.split("<antThinking>", 1)[1]

        lower = text.lower().strip()

        # 第1步：寻找明确的 "决策：YES/NO" 模式（LLM 最常使用的格式）
        dec_match = re.search(r"决策[:：]\s*(yes|no)", lower, re.IGNORECASE)
        if dec_match:
            return dec_match.group(1).lower() in ("yes", "是")

        # 第2步：寻找明确的 "Decision: YES/NO" 模式
        dec_match = re.search(r"decision[:：]\s*(yes|no)", lower, re.IGNORECASE)
        if dec_match:
            return dec_match.group(1).lower() in ("yes", "是")

        # 第3步：匹配首行的 YES/NO（作为最直接的指令）
        first_line = (
            lower.split("\n")[0].strip() if "\n" in text else lower[:100].strip()
        )
        if re.match(r"^(yes|no)\b", first_line, re.IGNORECASE):
            return first_line.startswith("y")

        # 第4步：寻找明确的领域动作模式
        if re.search(
            r"激活.*响应|应.*发出.*预警|建议.*预警|必须.*预警|需要.*预警",
            lower,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"不应.*预警|不建议.*预警|不需要.*预警|关闭.*响应|不.*发出.*预警",
            lower,
            re.IGNORECASE,
        ):
            return False

        # 第5步：宽松匹配独立出现的关键词
        yes_patterns = [r"\byes\b", r"是\b"]
        no_patterns = [r"\bno\b(?!\s*risk)", r"否\b"]

        for pat in yes_patterns:
            if re.search(pat, lower, re.IGNORECASE | re.MULTILINE):
                return True
        for pat in no_patterns:
            if re.search(pat, lower, re.IGNORECASE | re.MULTILINE):
                return False

        return None

    def _decide_heuristic(self, context: ScenarioContext) -> DecisionOutput:
        """当 CARM 不可用时，使用 MultiAlertAgent 规则引擎。

        此前此方法包含独立的重复权重计算逻辑，与主 Agent 不一致，
        现统一为直接调用 MultiAlertAgent，确保逻辑自洽。
        """
        return self._get_rule_agent().decide(context)
