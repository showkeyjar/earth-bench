"""CARS 冲击变量（暴雨/大风）独立推理 + 日报服务 — 落地流水线（判据 v2）。

与 cars_serve.py（t2m，冻结 deep-NGR 连续 CDF）平行的第二通道：
- 模型：SeasonalCarsModel 在完整 2000–2016 档案上**现场拟合**（fit < 2s；
  档案 npz 即工件，无 pickle 兼容性问题）。超参 = 滚动回测诚实选择众数
  （tp/wind 一致：k30/pca16/alpha0/beta1/w121，各 10/15 年），见
  earthbench/data/cars_impact_models.json（单一来源，含出处与阈值）。
- alpha_state=0 → 状态通道对误差选取无贡献，x := forecast（与 t2m DeepCars
  同法；业务上 ERA5 状态滞后 5 天不可得，此配置免疫该问题）。
- 尾部档位由漏报深度判据 v2 决定（CRPS 四变量验证）：
  * tp（deficit +1.57σ，深破裂）→ 高尾补全 q0.2/i1.5：漏报 22.9%（raw 49.1%），
    经济价值 6×（真保险）；
  * wind（deficit +0.52σ，浅越界）→ raw：补全亏价值 −10~20%（化妆品），
    校准值钱（SSR 0.90 近校准）。

数据：NOAA 业务 GEFS c00 pgrb2a 0p50（与 t2m 同桶同格式）：
- tp：f054/f060/f066/f072 的 surface APCP 6h 累计求和 → 有效日 24h 累计
  （mm；窗口 (48,72]，与再预报口径一致；已实测 ops 6h 倍数时效=6h 累计窗）。
- wind：f048–f072（6h×5）的 UGRD/VGRD 10m（heightAboveGround=10）瞬时
  → 逐快照风速 → 均值（m/s；日均近似，6h 采样，披露）。

影响优先（只播报对人类有害的事件）：
- p_harm_rain        P(24h 降水 ≥ 50 mm)（国标暴雨线）
- p_harm_rain_intense P(≥ 100 mm)（大暴雨线）
- p_harm_wind        P(日均风速 ≥ 10.8 m/s)（六级大风线，日均口径弱代理，披露）
- p_harm_wind_intense P(≥ 13.9 m/s)（七级线）
- p_harm_wind_typhoon P(≥ 17.2 m/s)（八级线：交通停运/户外作业停止，
                       GB/T 19201-2006，Phase D 与基准台风真值同口径同步）
- p_harm_wind_extreme P(≥ 24.5 m/s)（十级线：临设/塔吊安全参考档，日均口径）
- p_harm_snow        P(降雪 ≥ 10 mm 水当量)（GB/T 28592-2012 暴雪线；
                       APCP × 冻结掩膜 t2m < 0.5°C，Phase D 新增）
- p_harm_snow_intense P(≥ 20 mm)（大暴雪线）
- 触发：P ≥ 30%（rain 用 flood 成本比；wind/snow 无既有类别取 0.3，披露假设；
  snow 与 eval.ValueEvaluator["snow"] 构造性同步）。

诚实披露（caveats 一并写入 JSON）：
- 概率为 30 成员计数（地板 1/30 ≈ 3.3%）；t2m 通道是连续 CDF，两通道口径不同。
- 业务 GEFS 与再预报 v12 存在版本漂移（与 t2m 通道同款风险）。
- 无阵风（gust）数据：日均风速是大风的弱代理（10.8 m/s 日均已近该格点 p99）；
  17.2/24.5 高档位同为日均口径（日均 ≥17.2 已是灾害性持续大风；基准阵风
  安全线 GUST 24.5 无法由本通道服务，披露）。
- 降雪冻结掩膜取未扰动 c00 日均气温（确定性判定，非逐成员扰动）——
  概率仅来自 tp 成员散布，冻结/降雨相位不确定性未入集合（披露）。
- 风/降水检验闭环 = cars_verify_impact.py（模型锚定观测：有效日 00z f006-f024，
  检验的是概率层校准与 48h 相对技巧，非对地球真值的绝对校准——绝对校准由
  CRPS 侧 ERA5 out-of-sample 审计承担）。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .eval import ValueEvaluator

logger = logging.getLogger(__name__)

PACKAGE_DATA = Path(__file__).parent / "data"
IMPACT_CONFIG = PACKAGE_DATA / "cars_impact_models.json"
CITIES_FILE = PACKAGE_DATA / "cars_cities.json"
OPS = "https://noaa-gefs-pds.s3.amazonaws.com"

TP_STEPS = ("054", "060", "066", "072")           # 6h acc 窗求和 → (48,72] 24h
WIND_STEPS = ("048", "054", "060", "066", "072")  # 6h×5 瞬时快照均值

RAIN_P_TRIG = ValueEvaluator.DEFAULT_COST_RATIO["flood"]  # 0.3
WIND_P_TRIG = 0.3  # 无既有成本类别，取 flood 同值（披露假设）
# Phase D：snow 触发线与基准成本比构造性同步（eval.ValueEvaluator 单一来源）
SNOW_P_TRIG = ValueEvaluator.DEFAULT_COST_RATIO["snow"]  # 0.3
# 冻结掩膜阈值：与基准暴雪真值同口径（scenarios.py SNOW_FREEZE_MASK_C）
SNOW_FREEZE_C = 0.5

GRID_LAT = np.arange(50.0, 19.99, -1.0)
GRID_LON = np.arange(100.0, 140.01, 1.0)


def data_dir() -> Path:
    d = os.environ.get("CARS_DATA_DIR")
    return Path(d) if d else PACKAGE_DATA


def out_json_path() -> Path:
    return data_dir() / "cars_probabilities_impact_daily.json"


def history_dir() -> Path:
    return data_dir() / "cars_history"


def load_cities() -> list[dict]:
    cfg = json.loads((data_dir() / "cars_cities.json"
                      if (data_dir() / "cars_cities.json").exists()
                      else CITIES_FILE).read_text(encoding="utf-8"))
    return cfg["cities"] if isinstance(cfg, dict) else cfg


def load_impact_config() -> dict:
    return json.loads(IMPACT_CONFIG.read_text(encoding="utf-8"))


def _fetch_grib(init: str, step: str, hour: str = "00") -> Path:
    """下载并缓存一个业务 f 文件（pgrb2a 0p50，单时效多变量）。"""
    from .gefs_io import fetch_grib

    return fetch_grib(init, step, hour)


def _read_field(path: Path, keys: dict, var_hint: tuple[str, ...]) -> np.ndarray:
    """读一个 grib 消息 → 域内 1° 网格 (31,41) float32。"""
    from .gefs_io import read_field

    return read_field(path, keys, var_hint)


def fetch_ops_apcp(init: str, hour: str = "00") -> np.ndarray:
    """业务 c00 4×6h APCP 累计求和 → 有效日 24h 累计（mm，kg m⁻² 同值）。"""
    total = np.zeros((31, 41), dtype=np.float64)
    for s in TP_STEPS:
        p = _fetch_grib(init, s, hour)
        total += _read_field(p, {"typeOfLevel": "surface", "shortName": "tp"},
                             ("tp", "apcp", "prate"))
    return np.maximum(total, 0.0).astype(np.float32)


def fetch_ops_wind(init: str, hour: str = "00") -> np.ndarray:
    """业务 c00 5 快照 10m 风速均值 → 日均风速近似（m/s）。"""
    from .gefs_io import read_fields

    speeds = []
    for s in WIND_STEPS:
        p = _fetch_grib(init, s, hour)
        w = read_fields(p, {"typeOfLevel": "heightAboveGround", "level": 10},
                        {"u": ("u10", "u"), "v": ("v10", "v")})
        speeds.append(np.sqrt(w["u"].astype(np.float64) ** 2 + w["v"] ** 2))
    return np.mean(speeds, axis=0).astype(np.float32)


def fetch_ops_t2m_mean(init: str, hour: str = "00") -> np.ndarray:
    """业务 c00 5 快照 2m 气温均值 → 冻结掩膜用日均气温近似（°C）。

    与 fetch_ops_wind 读同一批已缓存 grib（pgrb2a 含 heightAboveGround=2
    的 t2m），无额外下载。Phase D：snowfall 通道的确定性冻结判据。
    """
    vals = []
    for s in WIND_STEPS:
        p = _fetch_grib(init, s, hour)
        t = _read_field(p, {"typeOfLevel": "heightAboveGround", "level": 2},
                        ("t2m", "t"))
        vals.append(t.astype(np.float64) - 273.15)
    return np.mean(vals, axis=0).astype(np.float32)


def snowfall_members(
    tp_members: np.ndarray, t2m_mean_c: np.ndarray | float,
    freeze_c: float = SNOW_FREEZE_C,
) -> np.ndarray:
    """冻结掩膜：t2m < freeze_c 处降水计为降雪（mm 水当量），否则 0。

    Phase D 新增，与基准暴雪真值同口径（scenarios.py 的 SNOW_FREEZE_MASK_C
    掩膜逻辑：气温 ≥ 0.5°C 的降水是雨不是雪）。支持任意可广播形状：
    t2m_mean_c可为 (31,41) 网格、标量，或与 tp_members 逐成员对齐的数组
    （若未来有逐成员气温，掩膜自动升级为逐成员判定）。
    """
    mask = (np.asarray(t2m_mean_c) < freeze_c).astype(
        np.asarray(tp_members).dtype)
    return (np.asarray(tp_members) * mask).astype(np.float32)


class ImpactCars:
    """冻结 SeasonalCars（档案现场拟合）冲击变量集合生成器。

    ``var`` ∈ {"tp", "wind"}；配置（超参/补全/截断/阈值）全部来自
    cars_impact_models.json。alpha_state=0 ⇒ x := forecast（无 ERA5 依赖）。
    """

    def __init__(self, var: str, config_path: Path | None = None,
                 archive_dir: Path | None = None):
        try:
            from crps_cars.models import SeasonalCarsModel, _inflate_high_tail
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "crps_cars is required for impact serving: "
                "`pip install -e ../CRPS`"
            ) from exc
        self._inflate_high_tail = staticmethod(_inflate_high_tail)
        cfg_all = json.loads((config_path or IMPACT_CONFIG).read_text(
            encoding="utf-8"))
        cfg = cfg_all[var]
        self.var = var
        self.cfg = cfg
        self.thresholds = cfg["thresholds"]
        self.clip_min = cfg.get("clip_min")
        art = (archive_dir or data_dir()) / cfg["archive"]
        if not art.exists():
            art = PACKAGE_DATA / cfg["archive"]
        z = np.load(art)
        x, f, t, d = z["x"], z["forecast"], z["truth"], z["dates"]
        m_cfg = cfg["config"]
        self.model = SeasonalCarsModel(
            members=m_cfg["members"], k=m_cfg["k"],
            pca_components=m_cfg["pca_components"],
            alpha_state=m_cfg["alpha_state"],
            beta_forecast=m_cfg["beta_forecast"],
            day_window=m_cfg["day_window"], seed=m_cfg["seed"],
        ).fit(x, f, (t - f).astype(np.float32), d)
        self.n_train = int(x.shape[0])

    def predict_for(self, forecast: np.ndarray, valid: date) -> np.ndarray:
        """forecast (T,1,31,41) → 成员 (T,30,1,31,41)，按有效日 doy 走季节窗。

        尾部档位按判据 v2（tp 高尾补全 / wind raw），最后按 clip_min 截断。
        """
        d = np.asarray([np.datetime64(valid.isoformat(), "D")] *
                       forecast.shape[0])
        members = self.model.predict(forecast, forecast, d)
        comp = self.cfg.get("tail_completion")
        if comp and comp.get("mode") == "high" and comp.get("infl", 0) > 0:
            members = self._inflate_high_tail(
                members, float(comp["infl"]), float(comp.get("q", 0.2)))
        if self.clip_min is not None:
            members = np.maximum(members, self.clip_min)
        return members.astype(np.float32)


def _frac_ge(m: np.ndarray, thr: float) -> float:
    return float(np.mean(m >= thr))


def _city_records(tp_members: np.ndarray, wind_members: np.ndarray,
                  valid: date, tp: ImpactCars, wind: ImpactCars,
                  cities: list[dict],
                  t2m_mean_c: np.ndarray | None = None,
                  snow_cfg: dict | None = None) -> list[dict]:
    records = []
    for c in cities:
        ix = int(np.argmin(np.abs(GRID_LON - c["lon"])))
        iy = int(np.argmin(np.abs(GRID_LAT - c["lat"])))
        mt = tp_members[0, :, 0, iy, ix].astype(np.float64)   # mm/day
        mw = wind_members[0, :, 0, iy, ix].astype(np.float64)  # m/s
        rt, wt = tp.thresholds, wind.thresholds
        rec = {
            "region": c["region"], "lat": c["lat"], "lon": c["lon"],
            "name_zh": c.get("name_zh", c["region"]),
            "valid_date": valid.isoformat(), "covered": True,
            "cell_lat": float(GRID_LAT[iy]), "cell_lon": float(GRID_LON[ix]),
            "rain_member_mean_mm": round(float(mt.mean()), 1),
            "rain_member_max_mm": round(float(mt.max()), 1),
            "p_harm_rain": round(_frac_ge(mt, rt["hard"]), 3),
            "p_harm_rain_intense": round(_frac_ge(mt, rt["intense"]), 3),
            "wind_member_mean_ms": round(float(mw.mean()), 1),
            "wind_member_max_ms": round(float(mw.max()), 1),
            "p_harm_wind": round(_frac_ge(mw, wt["hard"]), 3),
            "p_harm_wind_intense": round(_frac_ge(mw, wt["intense"]), 3),
        }
        # Phase D：wind 高档位（17.2/24.5，与基准台风真值同口径；
        # 旧配置缺键时优雅跳过）
        wty, wex = wt.get("typhoon"), wt.get("extreme")
        if wty is not None:
            rec["p_harm_wind_typhoon"] = round(_frac_ge(mw, wty), 3)
        if wex is not None:
            rec["p_harm_wind_extreme"] = round(_frac_ge(mw, wex), 3)
        # Phase D：snowfall 通道（APCP × 冻结掩膜；需 t2m 场与 snow 配置）
        if snow_cfg is not None and t2m_mean_c is not None:
            st = snow_cfg["thresholds"]
            freeze_c = float(snow_cfg.get("freeze_mask_c", SNOW_FREEZE_C))
            t2m_c = float(t2m_mean_c[iy, ix])
            snow = snowfall_members(mt, t2m_c, freeze_c)
            rec["t2m_mean_c"] = round(t2m_c, 1)
            rec["snow_freeze_mask_on"] = bool(t2m_c < freeze_c)
            rec["p_harm_snow"] = round(_frac_ge(snow, st["hard"]), 3)
            rec["p_harm_snow_intense"] = round(_frac_ge(snow, st["intense"]), 3)
        records.append(rec)
    return records


def daily_update_impact(valid: date | None = None, hour: str = "00") -> dict:
    """主入口：拉业务 GEFS（默认有效日 = 今天+2）、双变量推理、写概率表+历史。

    Phase D：同批 grib 加读 2m 气温（冻结掩膜），输出 snowfall 通道与
    wind 高档位（17.2/24.5）。
    """
    from .gefs_io import prune_gefs_cache

    try:
        prune_gefs_cache(keep_days=14)
    except Exception as e:  # noqa: BLE001 — 清理失败不阻断发布
        logger.warning(f"gefs cache prune skipped: {e}")
    valid = valid or (date.today() + timedelta(days=2))
    init = valid - timedelta(days=2)
    tp_fc = fetch_ops_apcp(init.isoformat(), hour)[None, None]   # (1,1,31,41)
    wd_fc = fetch_ops_wind(init.isoformat(), hour)[None, None]
    t2m_fc = fetch_ops_t2m_mean(init.isoformat(), hour)          # (31,41) °C

    tp = ImpactCars("tp")
    wind = ImpactCars("wind")
    tp_members = tp.predict_for(tp_fc, valid)                    # (1,30,1,31,41)
    wind_members = wind.predict_for(wd_fc, valid)

    cfg_all = load_impact_config()
    doc = {
        "model": "SeasonalCarsModel tp/wind serving "
                 "(2000-2016 archives, honest-selected modal hyperparams)",
        "tail_policy": "tp=high completion q0.2/i1.5 (miss-depth +1.57σ, "
                       "deep rupture); wind=raw (+0.52σ, shallow — "
                       "calibration wins). Criterion v2.",
        "forecast_source": f"NOAA ops GEFS c00 pgrb2a 0p50 "
                           f"APCP f{'+'.join(TP_STEPS)} sum; "
                           f"UGRD/VGRD10 f{'+'.join(WIND_STEPS)} mean; "
                           f"T2M f{'+'.join(WIND_STEPS)} mean (freeze mask)",
        "generated_for": valid.isoformat(),
        "p_trigger": {"rain": RAIN_P_TRIG, "wind": WIND_P_TRIG,
                      "snow": SNOW_P_TRIG},
        "provenance": {v: cfg_all[v]["provenance"] for v in ("tp", "wind")},
        "caveats": [
            "概率为 30 成员计数（地板 1/30≈3.3%），与 t2m 通道连续 CDF 口径不同",
            "业务 GEFS 与再预报 v12 存在版本漂移（与 t2m 通道同款风险）",
            "无阵风数据：日均风速是大风弱代理；17.2/24.5 高档位同为日均口径"
            "（基准阵风安全线 GUST 24.5 无法由本通道服务）",
            "降雪冻结掩膜取未扰动 c00 日均气温（确定性，非逐成员）——概率仅"
            "来自 tp 成员散布，雨雪相位不确定性未入集合",
            "检验闭环为模型锚定观测（cars_verify_impact，有效日 00z 短时效场），"
            "绝对校准由 CRPS 侧 ERA5 out-of-sample 审计承担",
            "tp 高尾补全档有 +1.7% CRPS 代价（防灾口径：漏报 49%→23%，价值 6×）",
        ],
        "records": _city_records(tp_members, wind_members, valid,
                                 tp, wind, load_cities(),
                                 t2m_mean_c=t2m_fc,
                                 snow_cfg=cfg_all.get("snow")),
    }
    out = out_json_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    hd = history_dir()
    hd.mkdir(parents=True, exist_ok=True)
    (hd / f"forecast_impact_{valid.isoformat()}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("impact daily update wrote %d records -> %s",
                len(doc["records"]), out)
    return doc


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=None,
                    help="有效日 YYYY-MM-DD（回填模式；业务桶保留约 10 天）")
    args = ap.parse_args()
    v = date.fromisoformat(args.valid) if args.valid else None
    print(json.dumps(daily_update_impact(v), ensure_ascii=False, indent=2))
