"""CARS 冲击变量检验闭环 — 暴雨/大风预报 → 有效日对答案 → 技能分数累积。

与 cars_verify.py（t2m）平行的第二检验通道，时序相同：
  D 日 daily_update_impact() 发布 D+2 概率表并归档 forecast_impact_{D+2}.json
  D+3 起 verify_impact() 对答案

观测源（无 QWeather 逐日降水/风速存档， disclosed）：
  GEFS c00 **有效日 00z 自身短时效场**（模型锚定，非站点真值）：
  - 降水 = f006/f012/f018/f024 的 surface APCP 6h 累计求和 = 有效日 (0,24] 24h 累计
    （与预报侧 (48,72] 窗口同构，只是换成有效日当日起报）
  - 风速 = f000/f006/f012/f018/f024 的 UGRD/VGRD10 瞬时 → 逐快照风速 → 均值
    （5 快照 ×6h，与预报侧结构一致）
  检验口径：**与业务模式自身的一致性 + CARS 概率层的校准**（48h 概率对 0-24h
  模式场的相对技巧），不是对真实地球的绝对校准——绝对校准由 CRPS 侧 ERA5
  out-of-sample 审计（2017–2019）承担。

累积指标（earthbench/data/cars_impact_verification_history.json）：
  - Brier：p_harm_rain vs obs≥50mm；p_harm_wind vs obs≥10.8 m/s
  - 触发线 P≥30% 的命中/漏报/空报与 CSI（两变量分别计）
  - 可靠性分箱（10 档）

用法：
    python -m earthbench.cars_verify_impact            # 最近一个可检验有效日
    python -m earthbench.cars_verify_impact --valid 2026-09-23
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .cars_serve_impact import (
    GRID_LAT,
    GRID_LON,
    PACKAGE_DATA,
    RAIN_P_TRIG,
    WIND_P_TRIG,
    _read_field,
    data_dir,
    history_dir,
    load_cities,
    load_impact_config,
)

logger = logging.getLogger(__name__)

HIST_FILE = "cars_impact_verification_history.json"

TP_OBS_STEPS = ("006", "012", "018", "024")     # (0,24] 6h 窗
WIND_OBS_STEPS = ("000", "006", "012", "018", "024")  # 5 快照 ×6h


def _hist_path() -> Path:
    return data_dir() / HIST_FILE


def _load_hist() -> dict:
    p = _hist_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"records": []}


def _save_hist(h: dict) -> None:
    p = _hist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_grib_cached(valid: date, step: str) -> Path:
    ymd = valid.strftime("%Y%m%d")
    name = f"gec00.t00z.pgrb2a.0p50.f{step}"
    cache = PACKAGE_DATA / "gefs_cache"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{ymd}00_{name}"
    if not local.exists():
        urllib.request.urlretrieve(
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{ymd}/00/atmos/"
            f"pgrb2ap5/{name}", local)
    return local


def obs_gefs_valid_day(valid: date) -> tuple[np.ndarray, np.ndarray]:
    """有效日 00z 模型锚定观测：(24h 降水 mm, 日均风速 m/s)，(31,41)。"""
    tp = np.zeros((31, 41), dtype=np.float64)
    for s in TP_OBS_STEPS:
        p = _fetch_grib_cached(valid, s)
        tp += _read_field(p, {"typeOfLevel": "surface", "shortName": "tp"},
                          ("tp", "apcp"))
    speeds = []
    for s in WIND_OBS_STEPS:
        p = _fetch_grib_cached(valid, s)
        u = _read_field(p, {"typeOfLevel": "heightAboveGround", "level": 10},
                        ("u10", "u")).astype(np.float64)
        v = _read_field(p, {"typeOfLevel": "heightAboveGround", "level": 10},
                        ("v10", "v"))
        speeds.append(np.sqrt(u ** 2 + v ** 2))
    wind = np.mean(speeds, axis=0)
    return (np.maximum(tp, 0.0).astype(np.float32),
            wind.astype(np.float32))


def verify_impact(valid: date) -> dict:
    """对一个有效日的暴雨/大风预报对答案并累积历史。"""
    fpath = history_dir() / f"forecast_impact_{valid.isoformat()}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"no archived impact forecast for {valid}")
    doc = json.loads(fpath.read_text(encoding="utf-8"))
    thr = load_impact_config()
    rain_thr = float(thr["tp"]["thresholds"]["hard"])   # 50 mm
    wind_thr = float(thr["wind"]["thresholds"]["hard"])  # 10.8 m/s

    obs_tp, obs_wd = obs_gefs_valid_day(valid)

    hist = _load_hist()
    new = []
    for r in doc["records"]:
        iy = int(np.argmin(np.abs(GRID_LAT - r["lat"])))
        ix = int(np.argmin(np.abs(GRID_LON - r["lon"])))
        o_rain = float(obs_tp[iy, ix])
        o_wind = float(obs_wd[iy, ix])
        ev_rain = o_rain >= rain_thr
        ev_wind = o_wind >= wind_thr
        p_rain = r.get("p_harm_rain", 0.0)
        p_wind = r.get("p_harm_wind", 0.0)
        new.append({
            "region": r["region"], "valid_date": valid.isoformat(),
            "lead_hours": 48, "obs_source": "gefs_validday_00z",
            "obs_rain_mm": round(o_rain, 1), "obs_wind_ms": round(o_wind, 1),
            "p_harm_rain": p_rain, "observed_rain": bool(ev_rain),
            "brier_rain": round((p_rain - float(ev_rain)) ** 2, 4),
            "triggered_rain": p_rain >= RAIN_P_TRIG,
            "p_harm_wind": p_wind, "observed_wind": bool(ev_wind),
            "brier_wind": round((p_wind - float(ev_wind)) ** 2, 4),
            "triggered_wind": p_wind >= WIND_P_TRIG,
        })

    keep = [x for x in hist["records"]
            if x["valid_date"] != valid.isoformat()]
    hist["records"] = keep + new
    _save_hist(hist)
    logger.info("impact-verified %d records for %s -> %s",
                len(new), valid, _hist_path())
    return {"valid_date": valid.isoformat(), "n": len(new), "records": new}


def _var_summary(recs: list[dict], var: str) -> dict:
    """单变量（rain/wind）滚动摘要：Brier/命中漏报空报/CSI/可靠性分箱。"""
    p_key = f"p_harm_{var}"
    ev_key = f"observed_{var}"
    trig_key = f"triggered_{var}"
    if not recs:
        return {"n": 0}
    brier = [r[f"brier_{var}"] for r in recs]
    hit = sum(1 for r in recs if r[trig_key] and r[ev_key])
    miss = sum(1 for r in recs if not r[trig_key] and r[ev_key])
    fa = sum(1 for r in recs if r[trig_key] and not r[ev_key])
    csi = hit / (hit + miss + fa) if (hit + miss + fa) else None
    bins = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        grp = [r for r in recs if lo <= min(r[p_key], 0.9999) < hi]
        if grp:
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}", "n": len(grp),
                "mean_p": round(float(np.mean([r[p_key] for r in grp])), 3),
                "event_rate": round(float(np.mean(
                    [r[ev_key] for r in grp])), 3),
            })
    return {
        "n": len(recs),
        "mean_brier": round(float(np.mean(brier)), 4),
        "hit": hit, "miss": miss, "false_alarm": fa,
        "csi": round(csi, 3) if csi is not None else None,
        "event_rate": round(float(np.mean([r[ev_key] for r in recs])), 4),
        "reliability_bins": bins,
    }


def summary() -> dict:
    """双变量滚动摘要（供面板引用）。"""
    recs = _load_hist()["records"]
    if not recs:
        return {"n": 0}
    out = {
        "n": len(recs),
        "since": min(r["valid_date"] for r in recs),
        "last": max(r["valid_date"] for r in recs),
        "obs_source": "gefs_validday_00z (model-anchored)",
    }
    for var in ("rain", "wind"):
        out[var] = _var_summary(recs, var)
    return out


def latest_verifiable_impact() -> date | None:
    """最近一个已过有效日且有归档冲击预报的日期（今天-1 起回看 10 天）。"""
    hd = history_dir()
    for back in range(1, 11):
        d = date.today() - timedelta(days=back)
        if (hd / f"forecast_impact_{d.isoformat()}.json").exists():
            return d
    return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()
    v = date.fromisoformat(args.valid) if args.valid else latest_verifiable_impact()
    if v is None:
        print("no archived impact forecast to verify yet")
    else:
        out = verify_impact(v)
        print(json.dumps({"verified": out["n"], "valid": out["valid_date"],
                          "summary": summary()}, ensure_ascii=False, indent=2))
