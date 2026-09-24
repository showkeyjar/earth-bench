"""CARS 冻结模型独立推理 + 日报服务 — 落地流水线核心。

零外部 ML 训练依赖：纯 numpy 直载 DeepNgrModel（深度 NGR：预报场 → 条件误差
高斯/EVT 混合分布），无需手写复刻（模型文件即单一来源）。工件为 2000–2015
训练的一次性冻结（earthbench/data/cars_t2m_deep_ngr.npz），气候/网格见
cars_climo.npz。

影响优先原则（只播报对人类有害的事件）：
- 模型量：日均 t2m 成员（与训练口径一致）
- 危害量（由 c00 的 tmax/tmin 日循环偏移 + r2 湿度派生，偏移不扰动，披露）：
  * p_harm_heat       P(日最高 ≥ 35°C)（国标高温黄色预警线）
  * p_harm_heat_intense P(日最高 ≥ 37°C)（橙色预警线）
  * p_harm_cold       P(日最低 ≤ 0°C)（结冰/冻害；仅冬季温暖城市启用预警。
                    口径披露：这是「冻害」绝对低温线，非基准寒潮真值的
                    「多窗口降幅 OR × 日最低」双条件——跨日降幅需 2-4 日集合配对，暂不服务）
  * p_harm_wb         P(午后湿球 ≥ 27°C)（EarthBench 湿球热应力标准，Stull 公式）
- 触发：P ≥ 30%（期望损失规则 P ≥ C/L）
- 当月 +2σ 相对异常仅为严重度背景，不参与触发。

城市配置：earthbench/data/cars_cities.json（可运维扩展，无需改码）。
数据目录：环境变量 CARS_DATA_DIR（CI 用 published_reports 做 gh-pages
持久化），默认 earthbench/data。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .eval import ValueEvaluator
from .weather import wet_bulb_stull as _wb_scalar

logger = logging.getLogger(__name__)

PACKAGE_DATA = Path(__file__).parent / "data"
DEEP_ARTIFACT = PACKAGE_DATA / "cars_t2m_deep_ngr.npz"
CLIMO_ARTIFACT = PACKAGE_DATA / "cars_climo.npz"
CITIES_FILE = PACKAGE_DATA / "cars_cities.json"
OPS = "https://noaa-gefs-pds.s3.amazonaws.com"

# 触发与阈值常量（P_TRIG 单一来源：eval.ValueEvaluator，与 cars_agent 构造性一致）
P_TRIG = ValueEvaluator.DEFAULT_COST_RATIO["heat"]
HEAT_MAX_C = 35.0          # 国标高温黄色预警线（日最高）
HEAT_INTENSE_C = 37.0      # 橙色预警线
COLD_MIN_C = 0.0           # 结冰/冻害线（日最低）
WB_NOON_C = 27.0           # EarthBench 湿球热应力标准
HEAT_ORANGE_P = 0.60       # 橙色发布线：P(强高温) 高位

GRID_LAT = np.arange(50.0, 19.99, -1.0)
GRID_LON = np.arange(100.0, 140.01, 1.0)


def data_dir() -> Path:
    """数据目录：CARS_DATA_DIR 覆盖（CI 持久化），默认包内 data。"""
    d = os.environ.get("CARS_DATA_DIR")
    return Path(d) if d else PACKAGE_DATA


def out_json_path() -> Path:
    return data_dir() / "cars_probabilities_daily.json"


def history_dir() -> Path:
    return data_dir() / "cars_history"


def load_cities() -> list[dict]:
    cfg = json.loads((data_dir() / "cars_cities.json"
                      if (data_dir() / "cars_cities.json").exists()
                      else CITIES_FILE).read_text(encoding="utf-8"))
    return cfg["cities"] if isinstance(cfg, dict) else cfg


def wet_bulb_stull(t_c, rh_pct):
    """Stull (2011) 湿球温度近似（向量化包装，公式单源在 weather.wet_bulb_stull）。

    支持标量或 ndarray：t_c °C，rh_pct %，误差典型 <1°C。
    """
    return np.vectorize(_wb_scalar)(t_c, rh_pct)


def _wb_inv(rh_pct: float, target: float = WB_NOON_C) -> float:
    """满足 wet_bulb_stull(t, rh)=target 的 t（湿球单调递增于 t，二分求逆）。"""
    lo, hi = -60.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _wb_scalar(mid, rh_pct) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DeepCars:
    """冻结 deep-NGR t2m 集合生成器（DeepNgrModel.load 直载，无复刻）。"""

    def __init__(self, artifact: Path | None = None, climo: Path | None = None):
        try:
            from crps_cars.deep_ngr import DeepNgrModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "crps_cars is required for deep-NGR serving: `pip install -e ../CRPS`"
            ) from exc
        self.model = DeepNgrModel.load(artifact or DEEP_ARTIFACT)
        c = np.load(climo or CLIMO_ARTIFACT)
        self.climo_mean = c["climo_mean"]          # (12,1,31,41)
        self.climo_std = c["climo_std"]
        self.lats, self.lons = c["grid_lat"], c["grid_lon"]

    def predict(self, forecast: np.ndarray) -> np.ndarray:
        """forecast: (T,1,31,41) K；返回 (T,30,1,31,41) 成员（日均口径）。

        alpha_state=0 时状态对误差无贡献，故 x := forecast（预测只依赖预报场）。
        """
        return self.model.predict(forecast, forecast)

    def cdf(self, forecast: np.ndarray, y) -> np.ndarray:
        """P(日均 t2m ≤ y) 逐格点连续累积概率（EVT 混合密度，非成员计数）。"""
        return self.model.predict_cdf(forecast, forecast, y)


def fetch_ops_fields(init: str, hour: str = "00") -> dict[str, np.ndarray]:
    """拉业务 GEFS c00 f048 t2m/r2/tmax/tmin → 1° 网格 (31,41)。

    tmax/tmin 为 c00 自带的日峰谷（用于成员日循环偏移，不扰动，披露）。
    """
    import xarray as xr

    from .gefs_io import fetch_grib

    local = fetch_grib(init, "048", hour)
    ds = xr.open_dataset(
        local, engine="cfgrib", backend_kwargs={"indexpath": ""},
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    sel = dict(latitude=slice(50.0, 20.0), longitude=slice(100.0, 140.0))
    grid = dict(latitude=GRID_LAT, longitude=GRID_LON)
    out: dict[str, np.ndarray] = {}
    for var in ("t2m", "r2", "tmax", "tmin"):
        if var in ds:
            out[var] = ds[var].sel(**sel).interp(**grid
                                                 ).values.astype(np.float32)
    ds.close()
    if "t2m" not in out:
        raise RuntimeError(f"t2m missing from {local}")
    return out


def _city_records(members: np.ndarray, fields: dict[str, np.ndarray],
                  valid: date, cars: DeepCars,
                  cities: list[dict]) -> list[dict]:
    mi = valid.month - 1
    tmax = fields.get("tmax")
    tmin = fields.get("tmin")
    r2 = fields.get("r2")
    fcst = fields["t2m"][None, None]                        # (1,1,31,41) K
    records = []
    for c in cities:
        ix = int(np.argmin(np.abs(cars.lons - c["lon"])))
        iy = int(np.argmin(np.abs(cars.lats - c["lat"])))
        m = members[0, :, 0, iy, ix].astype(np.float64)      # 日均成员 K（仅摘要统计）
        cm = cars.climo_mean[mi, 0, iy, ix]
        cs = cars.climo_std[mi, 0, iy, ix]
        thr = cm + 2.0 * cs
        # 连续 CDF：去掉 30-member 概率地板
        rec = {
            "region": c["region"], "lat": c["lat"], "lon": c["lon"],
            "name_zh": c.get("name_zh", c["region"]),
            "cold_alert_active": c.get("cold_alert_active", True),
            "valid_date": valid.isoformat(), "covered": True,
            "cell_lat": float(cars.lats[iy]), "cell_lon": float(cars.lons[ix]),
            "thr_2sigma_k": round(float(thr), 2),
            "p_exceed_2sigma": round(1.0 - float(cars.cdf(fcst, thr)[0, 0, iy, ix]), 3),
            "t2m_member_min_k": round(float(m.min()), 2),
            "t2m_member_mean_k": round(float(m.mean()), 2),
            "t2m_member_max_k": round(float(m.max()), 2),
        }
        # ---- 危害量：日循环偏移派生（连续概率） ----
        if tmax is not None:
            off_max = float(tmax[iy, ix] - fields["t2m"][iy, ix])  # K
            rec.update({
                "p_harm_heat": round(1.0 - float(cars.cdf(
                    fcst, HEAT_MAX_C + 273.15 - off_max)[0, 0, iy, ix]), 3),
                "p_harm_heat_intense": round(1.0 - float(cars.cdf(
                    fcst, HEAT_INTENSE_C + 273.15 - off_max)[0, 0, iy, ix]), 3),
                "tmax_offset_k": round(off_max, 2),
                "member_max_c_median": round(float(np.median(m + off_max - 273.15)), 1),
            })
            if r2 is not None:
                rh = float(r2[iy, ix])
                tw_thr = _wb_inv(rh)                          # 湿球 27°C 对应的日最高(°C)
                rec.update({
                    "rh_c00_pct": round(rh, 1),
                    "p_harm_wb": round(1.0 - float(cars.cdf(
                        fcst, tw_thr + 273.15 - off_max)[0, 0, iy, ix]), 3),
                })
        else:
            # 无 tmax 场时退回日均口径（303K ≈ 日最高 35°C 的旧近似）
            rec.update({
                "p_harm_heat": round(1.0 - float(cars.cdf(fcst, 303.0)[0, 0, iy, ix]), 3),
                "p_harm_heat_intense": round(1.0 - float(cars.cdf(fcst, 305.0)[0, 0, iy, ix]), 3),
            })
        if tmin is not None:
            off_min = float(tmin[iy, ix] - fields["t2m"][iy, ix])
            rec.update({
                "p_harm_cold": round(float(cars.cdf(
                    fcst, COLD_MIN_C + 273.15 - off_min)[0, 0, iy, ix]), 3),
            })
        else:
            rec.update({"p_harm_cold": round(float(cars.cdf(fcst, 273.15)[0, 0, iy, ix]), 3)})
        records.append(rec)
    return records


def daily_update(valid: date | None = None) -> dict:
    """主入口：拉 f048（默认有效日 = 今天+2）、推理、写概率表 + 历史归档。"""
    from .gefs_io import prune_gefs_cache

    try:
        prune_gefs_cache(keep_days=14)
    except Exception as e:  # noqa: BLE001 — 清理失败不阻断发布
        logger.warning(f"gefs cache prune skipped: {e}")
    valid = valid or (date.today() + timedelta(days=2))
    init = valid - timedelta(days=2)
    fields = fetch_ops_fields(init.isoformat())
    fcst = fields["t2m"][None, None]                        # (1,1,31,41)
    cars = DeepCars()
    members = cars.predict(fcst)                             # (1,30,1,31,41)

    records = _city_records(members, fields, valid, cars, load_cities())
    doc = {
        "model": "DeepNgrModel t2m serving (2000-2015 artifact, evt tail)",
        "forecast_source": "NOAA ops GEFS c00 pgrb2a 0p50 f048 "
                           "(t2m/r2/tmax/tmin at level 2)",
        "generated_for": valid.isoformat(),
        "p_trigger": P_TRIG,
        "caveats": [
            "成员为日均 t2m 扰动；日峰谷由 c00 偏移派生（未扰动，披露）",
            "湿度取 c00 单场（未扰动）；湿球为 Stull 近似（典型误差 <1°C）",
            "深度 NGR 冻结于 2000-2015 训练（2016 早停 + GPD 尾部）；业务 GEFS "
            "与再预报 v12 存在版本漂移",
        ],
        "records": records,
    }
    out = out_json_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    # 历史归档（检验闭环用：valid 日 2 天后对答案）
    hd = history_dir()
    hd.mkdir(parents=True, exist_ok=True)
    (hd / f"forecast_{valid.isoformat()}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("cars daily update wrote %d records -> %s", len(records), out)
    return doc


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=None,
                    help="有效日 YYYY-MM-DD（回填模式；GEFS 业务桶约保留 10 天）")
    args = ap.parse_args()
    v = date.fromisoformat(args.valid) if args.valid else None
    print(json.dumps(daily_update(v), ensure_ascii=False, indent=2))
