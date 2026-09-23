"""LLM 决策解析与 prompt 领域提示（单一来源）。

此前 `agents.LLMDecisionAgent` 与 `integrations.CARMBridge` 各有一套几乎相同的
YES/NO 解析正则，漂移风险高；统一到本模块，供两边复用。
"""

from __future__ import annotations

import re


def domain_hint(category: str) -> str:
    """按灾种返回领域知识提示（供 prompt 组装）。"""
    if category == "fire":
        return (
            "\n[领域知识提示] 这是森林火险预警场景。关键指标说明：\n"
            "- FWI (Fuel Weather Index): 0-20为低风险, 20-40为中等风险, 40-60为高风险, 60+为极高风险。\n"
            "- Humidity (相对湿度): 100%为最湿润(最安全), 0%为最干燥(最危险)。<30%表示极度干燥。\n"
            "- Wind Speed (风速): >20 km/h 会加速火势蔓延。\n"
            "- Rainfall (降雨量): 24h降雨>20mm会完全抵消火险（灭火效应）。降雨是决定性抑制因子，当降雨>20mm时应直接判断为NO。\n"
        )
    if category == "flood":
        return (
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
    if category == "drought":
        return (
            "\n[领域知识提示] 这是干旱预警场景。关键指标说明：\n"
            "- SPI (标准化降水指数): <-1.0为干旱, <-1.5为严重干旱, <-2.0为极端干旱。SPI反映短期降水异常。\n"
            "- Palmer 干旱指数: <-1.0为干旱, <-1.5为严重干旱, <-2.0为极端干旱。Palmer反映土壤深层水分亏损，是干旱预警的金标准。\n"
            "- NDVI: <0.3表示植被严重胁迫。\n"
            "- 【关键规则】判定干旱的核心标准：任意一个指标达到严重干旱级别即应预警(SPI<-1.5 OR Palmer<-1.0 OR 月降雨<10mm OR NDVI<0.3)。\n"
            "- SPI和Palmer冲突时以Palmer为准。但即使SPI接近正常(>-1.0)，只要任一条件(月降雨<10mm, NDVI<0.3)成立，都是干旱信号。\n"
            "- SPI趋势性下降(如-0.5→-0.9→-1.6)也是重要预警信号，当前值<-1.5即严重干旱。\n"
        )
    if category == "heat":
        return (
            "\n[领域知识提示] 这是高温/热浪预警场景。关键指标说明：\n"
            "- Wet Bulb Temperature (湿球温度): >27°C触发预警, >28°C为危险, >30°C为致命。\n"
            "- 干热 vs 湿热：高湿度（>70%）下人体排汗失效，湿球温度是最核心预警指标。\n"
            "- 连续3天最高温≥34°C + 湿球>27°C = 典型湿热热浪环境（如南京火炉特性）。\n"
            "- 即使单次温度未达35°C极端线，湿热组合+持续性也构成预警条件。\n"
            "- Temperature >36°C + Humidity <20%: 干燥环境，风险中等。\n"
            "- 持续多日高温且夜间降温不足时，累积热应力更高。\n"
        )
    return ""


def parse_yes_no(text: str) -> str | None:
    """从 LLM 自然语言响应中提取二元决策，返回 'yes' / 'no' / None（不可判）。"""
    if not text:
        return None

    content = text.strip()
    if " response" in content:
        content = content.split(" response", 1)[1]
    elif "<antThinking>" in content:
        content = content.split("<antThinking>", 1)[1]
    lower = content.lower()

    # 1) 明确的「决策：YES/NO」或「Decision: YES/NO」或中文「是/否」
    m = re.search(r"(?:决策|decision)[:：]\s*(yes|no|是|否)", lower)
    if m:
        return "yes" if m.group(1) in ("yes", "是") else "no"

    # 2) 首行直接 YES/NO
    first_line = (
        lower.split("\n")[0].strip() if "\n" in lower else lower[:100].strip()
    )
    if re.match(r"^(yes|no)\b", first_line):
        return "yes" if first_line.startswith("y") else "no"

    # 3) 领域动作短语（先判否定，避免「不建议预警/不需要预警」被「建议…预警/需要…预警」误判为 yes）
    if re.search(r"不应.*预警|不建议.*预警|不需要.*预警|关闭.*响应|不.*发出.*预警", lower):
        return "no"
    if re.search(r"激活.*响应|应.*发出.*预警|建议.*预警|必须.*预警|需要.*预警", lower):
        return "yes"

    # 4) 宽松的英文关键词
    if re.search(r"\byes\b", lower):
        return "yes"
    if re.search(r"\bno\b(?!\s*risk)", lower):
        return "no"

    return None