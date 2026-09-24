"""历史灾例回填验证测试 —— 真值函数对真实事件的外部锚定。

郑州 7·20 / 北京 23·7 / 台风杜苏芮 的公开报道观测值灌入独立标准
真值函数，推导判定必须与实际现实（官方响应/实际灾情）一致。
"""
from __future__ import annotations

from earthbench.benchmark import AlertBenchEvaluator
from earthbench.scenarios import get_historical_validation_suite


def _case(cid: str) -> dict:
    return next(c for c in get_historical_validation_suite()
                if c["case_id"] == cid)


def test_historical_suite_shape_and_provenance():
    suite = get_historical_validation_suite()
    assert len(suite) == 8
    real = [c for c in suite if not c["case_id"].startswith("hist-control")]
    synthetic = [c for c in suite if c["case_id"].startswith("hist-control")]
    assert len(real) == 7 and len(synthetic) == 1
    # 真实灾例必须逐条溯源（URL），合成对照必须显式披露
    for c in real:
        assert c["provenance"]["sources"], c["case_id"]
        assert any(s.startswith("http") for s in c["provenance"]["sources"])
        assert "documented_reality" in c["provenance"]
        assert "obs_notes" in c["provenance"]
    assert "合成" in synthetic[0]["provenance"]["event"]


def test_gt_verdicts_match_documented_reality():
    """核心验证：独立标准推导 == 实际发生了什么（gt_divergences 为空）。"""
    ev = AlertBenchEvaluator(suite=list(get_historical_validation_suite()))
    assert ev.gt_divergences == []
    assert len(ev.test_cases) == 8
    # 文档真值本身就是「实际现实」的编码
    assert all(tc.ground_truth == item["ground_truth"]
               for tc, item in zip(ev.test_cases,
                                   get_historical_validation_suite()))


def test_zhengzhou_720_flood_gt_tier():
    """645.6mm 24h → 大暴雨档（≥100mm，GB/T 28592 查表顶档 0.95）。"""
    decision, score, expl = _case("hist-flood-zhengzhou-720")["_gt_fn"](
        _case("hist-flood-zhengzhou-720")["observations"])
    assert decision is True and score == 0.95
    assert "大暴雨" in expl["detail"] or "100" in expl["detail"]


def test_beijing_237_flood_gt_via_6h_channel():
    """无干净 24h 站点值 → 6h 短时强降雨通道（111.8 ≥ 50）命中 0.85。

    多通道查表设计的实战意义：公开报道口径不齐时，任一独立通道
    即可完成判定。
    """
    decision, score, expl = _case("hist-flood-beijing-237")["_gt_fn"](
        _case("hist-flood-beijing-237")["observations"])
    assert decision is True and score == 0.85
    assert "6h" in expl["detail"]


def test_doksuri_2023_typhoon_gt_tier():
    """登陆风速 50 m/s ≥ 32.7（12 级台风线，GB/T 19201）→ 0.95。"""
    decision, score, expl = _case("hist-typhoon-doksuri-2023")["_gt_fn"](
        _case("hist-typhoon-doksuri-2023")["observations"])
    assert decision is True and score == 0.95
    assert "台风级" in expl["detail"]


def test_tongliao_2021_cold_gt_tier():
    """站档 tmin 序列 4.6 → -6.2°C：推导降幅 10.8 ≥ 8（24h 窗口）且 ≤4
    → 寒潮顶档 0.95（GB/T 20484-2017 四级体系）。

    验证降幅推导路径（drop_source=derived，从连续两日 tmin 序列差值）。
    """
    decision, score, expl = _case("hist-cold-tongliao-2021")["_gt_fn"](
        _case("hist-cold-tongliao-2021")["observations"])
    assert decision is True and score == 0.95
    assert expl["drop_source"] == "derived"
    assert expl["drop_24h"] == 10.8 and expl["tmin_latest"] == -6.2
    assert "寒潮" in expl["standard"]


def test_tongliao_2021_snow_gt_tier():
    """主雪日 24h 水当量 44.6mm ≥ 30 → 特大暴雪 0.95（GB/T 28592 附表顶档）。"""
    decision, score, expl = _case("hist-snow-tongliao-2021")["_gt_fn"](
        _case("hist-snow-tongliao-2021")["observations"])
    assert decision is True and score == 0.95
    assert expl["snow_24h"] == 44.6 and expl["snow_source"] == "observed"
    assert "特大暴雪" in expl["detail"]


def test_chongqing_2022_heat_gt_tier():
    """站档 tmax 43.1°C ≥ 40 → 红色高温档 0.95（中央气象台预警信号色阶）。"""
    decision, score, expl = _case("hist-heat-chongqing-2022")["_gt_fn"](
        _case("hist-heat-chongqing-2022")["observations"])
    assert decision is True and score == 0.95
    assert expl["temperature_max"] == 43.1
    assert expl["heat_duration_days"] == 24
    assert "红色" in expl["standard"]


def test_chongqing_2022_fire_gt_tier():
    """站档 14 时快照计算 FWI=70.5 > 33 → 极高火险 0.95（GB/T 36743）。"""
    decision, score, expl = _case("hist-fire-chongqing-2022")["_gt_fn"](
        _case("hist-fire-chongqing-2022")["observations"])
    assert decision is True and score == 0.95
    assert expl["fwi_latest"] == 70.5
    assert "极高" in expl["detail"]


def test_historical_findings_disclosed():
    """回填口径发现可追溯：寒潮口径差异 + 滑坡/干旱数据受阻，均附来源。"""
    from earthbench.scenarios import HISTORICAL_VALIDATION_FINDINGS
    ids = {f["id"] for f in HISTORICAL_VALIDATION_FINDINGS}
    assert ids == {
        "cold-wave-24h-vs-local-standard",
        "landslide-no-public-rain-intensity",
        "drought-index-not-publicly-archived",
    }
    for f in HISTORICAL_VALIDATION_FINDINGS:
        assert f["title"] and f["detail"]
        assert any(s.startswith("http") for s in f["sources"])


def test_synthetic_control_no_false_alarm():
    """对照日：三通���远离阈值 → 不预警（False 通路）。"""
    decision, score, _ = _case("hist-control-synthetic-benign")["_gt_fn"](
        _case("hist-control-synthetic-benign")["observations"])
    assert decision is False
    assert score < 0.5


def test_historical_regions_have_coordinates():
    """历史灾例区域进位置表（供 LLM 上下文/数据采集定位）。"""
    from earthbench.data_collectors import REGION_LOCATION_MAP
    for r in ("Zhengzhou-Henan", "Fangshan-Beijing", "Jinjiang-Fujian",
              "Tongliao-InnerMongolia", "Chongqing-HotPotato"):
        info = REGION_LOCATION_MAP[r]
        assert "lon" in info and "lat" in info
