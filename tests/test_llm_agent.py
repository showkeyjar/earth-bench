"""LLM Agent 评测 harness 测试（全离线：mock 后端，不触网络）。"""
from __future__ import annotations

import json

from earthbench.benchmark import AlertBenchEvaluator
from earthbench.llm_agent import (
    CATEGORY_THRESHOLD_DIGEST,
    CarmBackend,
    LLMAlertAgent,
    MockLLMBackend,
    ScenarioPromptBuilder,
    parse_llm_decision,
)
from earthbench.scenarios import get_adversarial_suite, get_alert_benchmark_suite


def _ctx():
    ev = AlertBenchEvaluator(suite=list(get_alert_benchmark_suite()[:2]))
    return ev.test_cases[0].to_context()


# ---------------------------------------------------------------------------
# 提示词构建
# ---------------------------------------------------------------------------


def test_prompt_contains_observations_and_protocol():
    ctx = _ctx()
    system, user = ScenarioPromptBuilder(include_thresholds=True).build(ctx)
    for o in ctx.observations:
        assert o.variable in user and str(o.value) in user
    assert ctx.region in user
    assert "判定纪律" in user and "AND" in user
    assert "JSON" in system or "JSON" in user


def test_prompt_threshold_modes():
    ctx = _ctx()
    _, informed = ScenarioPromptBuilder(include_thresholds=True).build(ctx)
    _, blind = ScenarioPromptBuilder(include_thresholds=False).build(ctx)
    digest = CATEGORY_THRESHOLD_DIGEST[ctx.category.value]
    assert "阈值参考" in informed
    # informed 摘要实际进入提示词（而非仅存在于字典）
    key_token = digest.split("；")[0][:6]
    assert key_token in informed
    assert "阈值参考" not in blind
    assert digest not in blind


def test_threshold_digest_covers_all_categories():
    assert set(CATEGORY_THRESHOLD_DIGEST) == {
        "flood", "typhoon", "cold", "snow", "heat", "fire", "drought",
        "landslide",
    }


# ---------------------------------------------------------------------------
# 解析器鲁棒性
# ---------------------------------------------------------------------------


def test_parse_clean_json():
    d, c, e, r, ok = parse_llm_decision(
        '{"decision": "YES", "confidence": 0.8, "evidence": {"rainfall_24h": 0.9}, "rationale": "超阈值"}'
    )
    assert ok and d is True and c == 0.8 and e["rainfall_24h"] == 0.9
    assert "超阈值" in r


def test_parse_fenced_json_with_prose():
    text = '分析如下：\n```json\n{"decision": "NO", "confidence": 0.7, "rationale": "AND 缺一"}\n```\n综上不预警'
    d, c, _e, _r, ok = parse_llm_decision(text)
    assert ok and d is False and c == 0.7


def test_parse_confidence_clamped_and_default():
    d, c, _e, _r, ok = parse_llm_decision(
        '{"decision": "YES", "confidence": 7.5}'
    )
    assert ok and c == 1.0
    d, c, _e, _r, ok = parse_llm_decision('{"decision": "NO"}')
    assert ok and c == 0.5


def test_parse_prose_fallback_counts_as_failure():
    d, _c, _e, _r, ok = parse_llm_decision("我的判断是 YES，置信度 0.9")
    assert d is True and ok is False
    d, _c, _e, _r, ok = parse_llm_decision("完全无法解析的输出")
    assert d is False and ok is False


# ---------------------------------------------------------------------------
# Mock 后端 + Agent 管线（离线）
# ---------------------------------------------------------------------------


def test_mock_backend_deterministic():
    b = MockLLMBackend()
    a = b.complete("sys", "user\n- rainfall_24h = 120.0 mm @ t")
    assert a == b.complete("sys", "user\n- rainfall_24h = 120.0 mm @ t")
    obj = json.loads(a)
    assert obj["decision"] == "YES"  # 120 ≥ 40


def test_llm_agent_with_mock_offline_pipeline():
    agent = LLMAlertAgent(backend=MockLLMBackend())
    out = agent.decide(_ctx())
    assert out.decision in (True, False)
    assert 0.0 <= out.confidence <= 1.0
    assert out.evidence_summary.get("parse_ok") is True
    assert agent.parse_failures == 0 and agent.backend_errors == 0


def test_llm_agent_backend_failure_degrades():
    class _Dead:
        name = "dead"

        def complete(self, system, user):
            raise ConnectionError("no service")

    agent = LLMAlertAgent(backend=_Dead())
    out = agent.decide(_ctx())
    assert out.decision is False and out.confidence == 0.0
    assert agent.backend_errors == 1
    assert "后端不可用" in out.rationale


def test_carm_backend_payload_construction():
    payload = CarmBackend()._build_payload("sys", "user")
    assert payload["model"] == "qwen3.6-35b"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["temperature"] == 0.0
    # 思考模型默认关思考（长链会把 1536 tokens 耗尽在思维链上）
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 1536
    thinking = CarmBackend(enable_thinking=True)._build_payload("s", "u")
    assert "chat_template_kwargs" not in thinking
    # 直连引擎层（非 CARM 路由端点）
    assert CarmBackend().base_url.endswith("8082/v1")


def test_mock_agent_evaluable_on_adversarial():
    """mock 后端全链路可评（离线），且不抛错、有结果。"""
    agent = LLMAlertAgent(backend=MockLLMBackend())
    ev = AlertBenchEvaluator(suite=list(get_adversarial_suite()))
    res = [r for r in ev.evaluate_agent(agent) if "error" not in r]
    assert len(res) == 16
    acc = sum(r["accuracy"] for r in res) / len(res)
    assert 0.0 <= acc <= 1.0
