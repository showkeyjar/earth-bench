"""Alert 决策 Agent 实现。

Phase 1 提供四类场景各自专用的规则引擎 baseline Agent。
每类场景都有自己的 Ground Truth 推导逻辑，作为 LLM Agent 的对标基准。
"""

from __future__ import annotations

from typing import Any

from .models import DecisionOutput, ScenarioContext
from .templates import TemplateEngine


# ===========================================================================
# 专用场景 Agent 接口
# ===========================================================================

class AlertAgent:
    """Alert Agent 基类 — 子类必须实现 decide()."""

    def __init__(self, name: str = "AlertAgent"):
        self.name = name

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        raise NotImplementedError


# ===========================================================================
# Fire Alert Agent
# ===========================================================================

class FireAlertAgent(AlertAgent):
    """森林火险预警 Agent（对标国家林草局 FWI 标准）。

    风险评分模型：
    - FWI（权重 0.40）：主导因子
    - 湿度（权重 0.20）：干燥助燃
    - 风速（权重 0.15）：强风助燃
    - 降雨（权重 -0.15 抑制）：降雨抵消风险
    - 温度（权重 0.15）：高温干燥

    决策阈值: risk_score >= 0.4 → YES
    """

    def __init__(
        self,
        fwi_threshold: float = 40.0,
        humidity_threshold: float = 20.0,
        wind_threshold: float = 12.0,
        rainfall_suppress: float = 10.0,
    ):
        super().__init__("FireAlertAgent")
        self.fwi_threshold = fwi_threshold
        self.humidity_threshold = humidity_threshold
        self.wind_threshold = wind_threshold
        self.rainfall_suppress = rainfall_suppress

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        fwi_vals = [o.value for o in obs if o.variable == "FWI"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        wind_vals = [o.value for o in obs if o.variable == "wind_speed"]
        rain_vals = [o.value for o in obs if o.variable == "rainfall"]
        temp_vals = [o.value for o in obs if o.variable == "temperature"]

        evidence: dict[str, float] = {}
        risk_score = 0.0

        # FWI (主导)
        avg_fwi = sum(fwi_vals) / len(fwi_vals) if fwi_vals else 0
        fwi_risk = min(avg_fwi / 60.0, 1.0)
        risk_score += fwi_risk * 0.40
        evidence["FWI"] = round(fwi_risk, 3)

        # 湿度
        avg_hum = sum(hum_vals) / len(hum_vals) if hum_vals else 100
        hum_risk = max(0, (100 - avg_hum) / 100)
        risk_score += hum_risk * 0.20
        evidence["humidity"] = round(hum_risk, 3)

        # 风速
        avg_wind = sum(wind_vals) / len(wind_vals) if wind_vals else 0
        wind_risk = min(avg_wind / 20.0, 1.0)
        risk_score += wind_risk * 0.15
        evidence["wind"] = round(wind_risk, 3)

        # 降雨（抑制因素）
        avg_rain = sum(rain_vals) / len(rain_vals) if rain_vals else 0
        if avg_rain > self.rainfall_suppress:
            suppression = min(avg_rain / 30.0, 1.0)
            risk_score *= (1.0 - suppression * 0.6)
            evidence["rainfall_suppression"] = -round(suppression * 0.6, 3)
        else:
            evidence["rainfall_suppression"] = 0.0

        # 温度
        avg_temp = sum(temp_vals) / len(temp_vals) if temp_vals else 20
        temp_risk = max(0, (avg_temp - 20) / 30.0)
        risk_score += temp_risk * 0.15
        evidence["temperature"] = round(temp_risk, 3)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= 0.4

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"FireAlert: FWI={avg_fwi:.1f}, 湿度={avg_hum:.1f}%, "
                f"风速={avg_wind:.1f}m/s, 降雨={avg_rain:.1f}mm, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Flood Alert Agent
# ===========================================================================

class FloodAlertAgent(AlertAgent):
    """洪涝预警 Agent（对标《国家防汛抗旱应急预案》暴雨标准）。

    风险评分模型：
    - 24h 降雨（权重 0.35）：>=50mm 暴雨标准
    - 6h 短时强降雨（权重 0.25）：>=30mm
    - 土壤湿度饱和（权重 0.15）：>0.85 放大效应
    - 水位超警戒（权重 0.25）：>警戒水位

    决策阈值: risk_score >= 0.45 → YES
    """

    def __init__(
        self,
        water_level_critical: float = 10.0,
    ):
        super().__init__("FloodAlertAgent")
        self.water_level_critical = water_level_critical

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        rain24_vals = [o.value for o in obs if "rainfall_24h" in o.variable]
        rain6_vals = [o.value for o in obs if "rainfall_6h" in o.variable]
        soil_vals = [o.value for o in obs if "soil_moisture" in o.variable]
        level_vals = [o.value for o in obs if "water_level" in o.variable]

        evidence: dict[str, float] = {}
        risk_score = 0.0

        # 24h 降雨
        avg_rain24 = sum(rain24_vals) / len(rain24_vals) if rain24_vals else 0
        rain24_risk = min(avg_rain24 / 100.0, 1.0)
        risk_score += rain24_risk * 0.35
        evidence["rainfall_24h"] = round(rain24_risk, 3)

        # 6h 短时强降雨
        avg_rain6 = sum(rain6_vals) / len(rain6_vals) if rain6_vals else 0
        rain6_risk = min(avg_rain6 / 50.0, 1.0)
        risk_score += rain6_risk * 0.25
        evidence["rainfall_6h"] = round(rain6_risk, 3)

        # 土壤湿度
        avg_soil = sum(soil_vals) / len(soil_vals) if soil_vals else 0.5
        soil_sat_risk = max(0, (avg_soil - 0.6) / 0.4)
        risk_score += soil_sat_risk * 0.15
        evidence["soil_moisture"] = round(soil_sat_risk, 3)

        # 水位
        level_obs_list = [o for o in obs if "water_level" in o.variable]
        avg_level = sum(lo.value for lo in level_obs_list) / len(level_obs_list) if level_obs_list else 0
        
        # --- 水位趋势检测（L4 关键能力） ---
        water_t_bonus = 0.0
        if len(level_obs_list) >= 3:
            levels = [lo.value for lo in level_obs_list]
            increasing = all(levels[i] < levels[i+1] for i in range(len(levels)-1))
            latest = levels[-1]
            if increasing and latest > 10.0:
                water_t_bonus = 0.50  # 强制提升为高置信度
            elif increasing and latest >= 9.0:
                water_t_bonus = 0.25  # 接近警戒线且在上涨
        
        level_risk = min(avg_level / self.water_level_critical, 1.0)
        risk_score += level_risk * 0.25
        evidence["water_level"] = round(level_risk, 3)
        
        if water_t_bonus > 0:
            risk_score += water_t_bonus
            evidence["water_trend"] = water_t_bonus

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= 0.45

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"FloodAlert: 24h={avg_rain24:.1f}mm, "
                f"6h={avg_rain6:.1f}mm, 土壤={avg_soil:.2f}, "
                f"水位={avg_level:.1f}m, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# Drought Alert Agent
# ===========================================================================

class DroughtAlertAgent(AlertAgent):
    """干旱预警 Agent（对标气象干旱标准）。

    风险评分模型：
    - SPI（标准化降水指数，权重 0.30）：< -1.0 中度干旱
    - Palmer 指数（权重 0.25）：< -0.5 干旱
    - 长期低湿度（权重 0.15）：< 25%
    - 月降雨严重不足（权重 0.15）：< 50mm
    - NDVI 植被胁迫（权重 0.15）：< 0.3

    决策阈值: risk_score >= 0.4 → YES
    """

    def __init__(self):
        super().__init__("DroughtAlertAgent")

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        spi_vals = [o.value for o in obs if o.variable == "SPI"]
        palm_vals = [o.value for o in obs if o.variable == "palmer_index"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        rain_vals = [o.value for o in obs if o.variable == "rainfall_monthly"]
        ndvi_vals = [o.value for o in obs if o.variable == "NDVI"]

        evidence: dict[str, float] = {}
        risk_score = 0.0

        # SPI
        avg_spi = sum(spi_vals) / len(spi_vals) if spi_vals else 0
        spi_risk = max(0, (-avg_spi - 0.5) / 2.0)
        risk_score += spi_risk * 0.30
        evidence["SPI"] = round(spi_risk, 3)

        # Palmer
        avg_palmer = sum(palm_vals) / len(palm_vals) if palm_vals else 0
        palm_risk = max(0, (-avg_palmer - 0.25) / 1.5)
        risk_score += palm_risk * 0.25
        evidence["Palmer"] = round(palm_risk, 3)

        # 湿度
        avg_hum = sum(hum_vals) / len(hum_vals) if hum_vals else 50
        hum_risk = max(0, (30 - avg_hum) / 30.0)
        risk_score += hum_risk * 0.15
        evidence["humidity"] = round(hum_risk, 3)

        # 月降雨
        avg_rain = sum(rain_vals) / len(rain_vals) if rain_vals else 100
        rain_risk = max(0, (50 - avg_rain) / 50.0)
        risk_score += rain_risk * 0.15
        evidence["rainfall_monthly"] = round(rain_risk, 3)

        # NDVI
        avg_ndvi = sum(ndvi_vals) / len(ndvi_vals) if ndvi_vals else 0.6
        ndvi_risk = max(0, (0.5 - avg_ndvi) / 0.3)
        risk_score += ndvi_risk * 0.15
        evidence["NDVI"] = round(ndvi_risk, 3)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= 0.4

        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"DroughtAlert: SPI={avg_spi:.2f}, Palmer={avg_palmer:.2f}, "
                f"湿度={avg_hum:.1f}%, 月降雨={avg_rain:.1f}mm, "
                f"NDVI={avg_ndvi:.2f}, 风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# HeatWave Alert Agent
# ===========================================================================

class HeatWaveAlertAgent(AlertAgent):
    """热浪预警 Agent（对标《中央气象台高温预警信号》）。

    风险评分模型：
    - 最高温（权重 0.35）：>=35°C
    - 持续天数（权重 0.25）：从时序温度记录中推断 ≥3 天持续高温
    - 湿球温度（权重 0.25）：>27°C 致死阈值
    - 湿度（权重 0.15）：高湿加剧闷热

    决策阈值: risk_score >= 0.4 → YES

    关键改进：从多时间戳温度观测中推断"连续高温天数"，
    替代外部 heat_duration_days 参数，使 Agent 能在真实场景中
    仅凭时序温度数据做出热浪预警判断。
    """

    def __init__(self):
        super().__init__("HeatWaveAlertAgent")

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        obs = context.observations

        temp_vals = [o.value for o in obs if o.variable == "temperature_max"]
        hum_vals = [o.value for o in obs if o.variable == "humidity"]
        wbt_vals = [o.value for o in obs if o.variable == "wet_bulb_temp"]
        temp_ts = [(o.timestamp, o.value) for o in obs if o.variable == "temperature_max"]
        wbt_ts = [(o.timestamp, o.value) for o in obs if o.variable == "wet_bulb_temp"]

        evidence: dict[str, float] = {}
        risk_score = 0.0

        # --- 最高温 ---
        avg_temp = sum(temp_vals) / len(temp_vals) if temp_vals else 25
        temp_risk = max(0, (avg_temp - 28) / 12.0)  # 28°C 以下无风险
        risk_score += temp_risk * 0.35
        evidence["temperature_max"] = round(temp_risk, 3)

        # --- 持续性检测：从多时间戳温度记录推断连续高温天数 ---
        # 排序时间戳
        temp_ts.sort(key=lambda x: x[0])
        heat_duration_days = 0
        if temp_ts:
            # 统计温度 >= 35°C 的时间戳数量（作为至少多少天经历高温）
            # L4 progressive 场景：如果趋势性升温到 ≥35°C，也算持续风险
            consecutive_high = 0
            for _, tv in temp_ts:
                if tv >= 35.0:
                    consecutive_high += 1
            heat_duration_days = max(consecutive_high, len(temp_ts) - 1)  # 至少是间隔天数
        elif context.horizon_hours >= 72:
            # 如果没有多时间戳但有长 horizon，假设 >= 3 天
            heat_duration_days = min(context.horizon_hours // 24, 5)

        # duration_factor: 3天=0.6, 5天=1.0
        duration_factor = min(heat_duration_days / 5.0, 1.0)
        risk_score += duration_factor * 0.25
        evidence["heat_duration_days"] = round(heat_duration_days)

        # --- 湿球温度 ---
        if wbt_vals:
            avg_wbt = sum(wbt_vals) / len(wbt_vals)
            wbt_risk = max(0, (avg_wbt - 23.0) / 8.0)  # 27°C 危险阈值
            risk_score += wbt_risk * 0.25
            evidence["wet_bulb_temp"] = round(wbt_risk, 3)

        # --- 湿度 ---
        avg_hum = sum(hum_vals) / len(hum_vals) if hum_vals else 50
        hum_risk = max(0, (avg_hum - 50) / 50.0)
        risk_score += hum_risk * 0.15
        evidence["humidity"] = round(hum_risk, 3)

        risk_score = max(0.0, min(risk_score, 1.0))
        decision = risk_score >= 0.4

        avg_wbt_val = sum(wbt_vals) / len(wbt_vals) if wbt_vals else None
        return DecisionOutput(
            context=context,
            decision=decision,
            confidence=round(risk_score, 3),
            evidence_summary=evidence,
            rationale=(
                f"HeatWaveAlert: 最高温={avg_temp:.1f}°C, "
                f"湿球={'%.1f' % avg_wbt_val if avg_wbt_val is not None else 'N/A'}°C, "
                f"湿度={avg_hum:.1f}%, 持续≈{heat_duration_days}天, "
                f"风险分={risk_score:.3f}"
            ),
        )


# ===========================================================================
# MultiAlertAgent — 场景路由
# ===========================================================================

class MultiAlertAgent(AlertAgent):
    """多场景 Alert Agent — 根据场景类别自动路由到专用引擎。

    这是 FireBench → AlertBench 升级后的核心 Agent：
    - category == fire → FireAlertAgent
    - category == flood → FloodAlertAgent
    - category == drought → DroughtAlertAgent
    - category == heat/ecology → HeatWaveAlertAgent
    - 其他 → 降级为 RuleBasedAgent (仅 fire)
    """

    def __init__(
        self,
        fwi_threshold: float = 40.0,
        humidity_threshold: float = 20.0,
        wind_threshold: float = 12.0,
        rainfall_suppress: float = 10.0,
    ):
        super().__init__("MultiAlertAgent")
        self.fire_agent = FireAlertAgent(
            fwi_threshold=fwi_threshold,
            humidity_threshold=humidity_threshold,
            wind_threshold=wind_threshold,
            rainfall_suppress=rainfall_suppress,
        )
        self.flood_agent = FloodAlertAgent()
        self.drought_agent = DroughtAlertAgent()
        self.heat_agent = HeatWaveAlertAgent()

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        """根据场景类别路由到对应的专用 Agent。"""
        category_map = {
            "fire": self.fire_agent,
            "flood": self.flood_agent,
            "drought": self.drought_agent,
            "ecology": self.heat_agent,
            "heat": self.heat_agent,
        }

        category = context.category.value if hasattr(context.category, "value") else str(context.category)

        agent = category_map.get(category, None)
        if agent is None:
            # 降级：fallback 到 fire agent（不推荐，仅兼容旧逻辑）
            agent = self.fire_agent

        return agent.decide(context)


# ===========================================================================
# Legacy — RuleBasedAgent 保留以兼容旧代码
# ===========================================================================

class RuleBasedAgent(FireAlertAgent):
    """兼容别名：旧 RuleBasedAgent 现在是 FireAlertAgent 的子类。

    保留旧接口名称，确保现有代码不 break。
    但推荐使用 FireAlertAgent（新名称更准确）。
    """

    def __init__(self, **kwargs):
        # 兼容旧参数名
        super().__init__(
            fwi_threshold=kwargs.get("fwi_threshold", 40.0),
            humidity_threshold=kwargs.get("humidity_threshold", 20.0),
            wind_threshold=kwargs.get("wind_threshold", 12.0),
            rainfall_suppress=kwargs.get("rainfall_suppress", 10.0),
        )


class LLMDecisionAgent(AlertAgent):
    """基于 LLM 的决策 Agent（Phase 2+）。

    将场景上下文组装为 prompt，调用 LLM（Gemini/Ollama/OpenAI），
    解析返回的结构化决策。

    优先级: CARM/Mustard集成 > OpenAI兼容API > Ollama本地模型 > 启发式回退
    """

    def __init__(self, model_name: str = "qwen3:14b", base_url: str | None = None, api_key: str | None = None):
        super().__init__("LLMDecisionAgent")
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        import os as _os
        if self.api_key or _os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if self.base_url or _os.environ.get("OLLAMA_BASE_URL"):
            return "ollama"
        return "none"

    def decide(self, context: ScenarioContext) -> DecisionOutput:
        from .templates import TemplateEngine

        valid, msg = TemplateEngine.validate_context(context)
        if not valid:
            return DecisionOutput(
                context=context, decision=False, confidence=0.0,
                evidence_summary={"error": msg},
                rationale=f"验证失败: {msg}",
            )

        # 构建 prompt
        tmpl_labels = {
            "alert": "是否预警", "dispatch": "是否调度", "upgrade": "是否升级",
            "close": "是否关闭", "recover": "是否恢复",
        }
        tmpl_label = tmpl_labels.get(context.template.value, "未知")

        obs_lines = []
        for o in context.observations:
            obs_lines.append(f"- [{o.source}] {o.variable}={o.value}{o.unit} (conf={o.confidence:.2f}) @ {o.timestamp}")
        obs_text = "\n".join(obs_lines)

        prompt = (
            f"[EarthBench Alert 决策任务]\n"
            f"区域: {context.region}\n"
            f"时间窗口: {context.horizon_hours}小时\n"
            f"决策模板: {tmpl_label}\n"
            f"\n证据列表:\n{obs_text}\n"
            f"\n请做出二元决策：是或否？\n"
            f"只回答 YES 或 NO（大写），然后简要说明理由和置信度。\n"
            f"格式：决策:YES/NO\n置信度:0.xx\n理由:..."
        )

        llm_answer = None
        llm_confidence = 0.5
        llm_rationale = ""

        # 尝试 OpenAI 兼容 API
        if self._provider == "openai":
            try:
                llm_answer, llm_confidence, llm_rationale = self._call_openai(prompt)
            except Exception as e:
                print(f"[LLMDecisionAgent] OpenAI call failed: {e}")

        # 尝试 Ollama
        if llm_answer is None:
            try:
                llm_answer, llm_confidence, llm_rationale = self._call_ollama(prompt)
            except Exception as e:
                print(f"[LLMDecisionAgent] Ollama call failed: {e}")

        if llm_answer is not None:
            decision = llm_answer in ("yes", "是")
            return DecisionOutput(
                context=context, decision=decision, confidence=llm_confidence,
                evidence_summary={"llm_model": self.model_name, "provider": self._provider},
                rationale=f"LLM ({self.model_name}): {llm_answer}. {llm_rationale}",
            )

        # 回退到启发式
        return self._heuristic_fallback(context)

    def _call_openai(self, prompt: str) -> tuple[str | None, float, str]:
        import os
        from openai import OpenAI

        base_url = self.base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
        return self._parse_llm_response(text)

    def _call_ollama(self, prompt: str) -> tuple[str | None, float, str]:
        import os
        import urllib.request
        import json
        base_url = self.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        payload = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("message", {}).get("content", "")
        return self._parse_llm_response(text)

    @staticmethod
    def _parse_llm_response(text: str) -> tuple[str | None, float, str]:
        import re

        content = text.strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        elif "<antThinking>" in content:
            content = content.split("<antThinking>", 1)[1]

        lower = content.lower()

        dec_match = re.search(r'决策[:：]\s*(yes|no|是|否)', lower)
        if dec_match:
            ans = dec_match.group(1)
            if ans in ("yes", "是"):
                return "yes", 0.85, content[:300]
            else:
                return "no", 0.85, content[:300]

        first_line = lower.split('\n')[0].strip() if '\n' in content else lower[:100].strip()
        if re.match(r'^(yes|no)\b', first_line):
            return ("yes" if first_line.startswith('y') else "no"), 0.8, content[:300]

        if re.search(r'应.*预警|建议.*预警|必须.*预警|需要.*预警|激活.*响应', lower):
            return "yes", 0.7, content[:300]
        if re.search(r'不应.*预警|不建议.*预警|不需要.*预警|不.*发出.*预警', lower):
            return "no", 0.7, content[:300]

        return None, 0.5, content[:300]

    def _heuristic_fallback(self, context: ScenarioContext) -> DecisionOutput:
        from earthbench.integrations import CARMBridge
        bridge = CARMBridge()
        return bridge._decide_heuristic(context)
