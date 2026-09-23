"""CARS 检验闭环 — 每日预报 → 有效日对答案 → 技能分数累积。

流水线时序：
  D 日 08:00  daily_update() 发布 D+2 有效日的概率表并归档
  D+3 起      verify() 对 D+2 的预报对答案（观测已可得）

观测源优先级（每条记录标注实际来源，诚实披露）：
  1. QWeather 实时气温（20:00 北京时运行时读数 ≈ 日最高代理，误差 1-3°C）
  2. GEFS c00 f000 当日场（分析近似；检验的是"与业务模式自身的一致性+48h 技巧"，
     不是站点真值）

累积指标（earthbench/data/cars_verification_history.json）：
  - Brier 分数（p_harm_heat vs 实际是否 ≥35°C）
  - 触发线 P≥30% 的命中/漏报/空报计数与 CSI（临界成功指数）
  - 样本量按城市累积

用法：
    python -m earthbench.cars_verify            # 自动找最近一个可检验的有效日
    python -m earthbench.cars_verify --valid 2026-09-21
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .cars_serve import (
    HEAT_MAX_C,
    PACKAGE_DATA,
    P_TRIG,
    data_dir,
    history_dir,
    load_cities,
)

logger = logging.getLogger(__name__)

HIST_FILE = "cars_verification_history.json"


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
    p.write_text(json.dumps(h, ensure_ascii=False, indent=2),
                 encoding="utf-8")


def _obs_qweather(cities: list[dict]) -> dict[str, float]:
    """QWeather 实时气温（°C）。返回 region -> 温度；无 key/失败返回空。"""
    import os

    key = os.environ.get("QWEATHER_API_KEY", "")
    if not key:
        return {}
    out: dict[str, float] = {}
    for c in cities:
        try:
            url = (f"https://devapi.qweather.com/v7/weather/now?"
                   f"key={key}&location={c['lon']:.2f},{c['lat']:.2f}")
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.loads(r.read().decode())
            if d.get("code") == "200":
                out[c["region"]] = float(d["now"]["temp"])
        except Exception as e:  # noqa: BLE001 — 单城失败不阻断
            logger.debug(f"QWeather failed for {c['region']}: {e}")
    return out


def _obs_gefs_f000(valid: date, cities: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """GEFS c00 当日 00z f000：返回 (t2m 读数°C, tmax 场°C)。"""
    import xarray as xr

    ymd = valid.isoformat().replace("-", "")
    name = "gec00.t00z.pgrb2a.0p50.f000"
    cache = PACKAGE_DATA / "gefs_cache"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{ymd}00_{name}"
    if not local.exists():
        urllib.request.urlretrieve(
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{ymd}/00/atmos/"
            f"pgrb2ap5/{name}", local)
    ds = xr.open_dataset(
        local, engine="cfgrib", backend_kwargs={"indexpath": ""},
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    sel = dict(latitude=slice(50.0, 20.0), longitude=slice(100.0, 140.0))
    t2m: dict[str, float] = {}
    tmax: dict[str, float] = {}
    for c in cities:
        try:
            pt = ds.sel(latitude=c["lat"], longitude=c["lon"],
                        method="nearest")
            t2m[c["region"]] = float(pt["t2m"].values) - 273.15
            if "tmax" in ds:
                tmax[c["region"]] = float(pt["tmax"].values) - 273.15
        except KeyError:
            continue
    ds.close()
    return t2m, tmax


def verify(valid: date) -> dict:
    """对一个有效日的全部城市预报对答案并累积历史。"""
    fpath = history_dir() / f"forecast_{valid.isoformat()}.json"
    if not fpath.exists():
        raise FileNotFoundError(f"no archived forecast for {valid}")
    doc = json.loads(fpath.read_text(encoding="utf-8"))
    cities = load_cities()

    obs_qw = _obs_qweather(cities)
    obs_g, obs_gmax = _obs_gefs_f000(valid, cities)

    hist = _load_hist()
    new = []
    for r in doc["records"]:
        region = r["region"]
        # 观测源决策：QWeather 优先，缺则 GEFS f000
        if region in obs_qw and obs_qw[region] is not None:
            obs_temp, obs_src = obs_qw[region], "qweather_now"
        elif region in obs_gmax:
            obs_temp, obs_src = obs_gmax[region], "gefs_f000_tmax"
        elif region in obs_g:
            obs_temp, obs_src = obs_g[region], "gefs_f000_t2m"
        else:
            continue
        event = obs_temp >= HEAT_MAX_C
        p = r.get("p_harm_heat", 0.0)
        rec = {
            "region": region, "valid_date": valid.isoformat(),
            "lead_hours": 48,
            "p_harm_heat": p,
            "observed_temp_c": round(obs_temp, 1),
            "observed_event": bool(event),
            "obs_source": obs_src,
            "brier": round((p - float(event)) ** 2, 4),
            "triggered": p >= P_TRIG,
        }
        new.append(rec)

    # 去重（同日重跑覆盖）
    keep = [x for x in hist["records"]
            if x["valid_date"] != valid.isoformat()]
    hist["records"] = keep + new
    _save_hist(hist)
    logger.info("verified %d records for %s -> %s", len(new), valid,
                _hist_path())
    return {"valid_date": valid.isoformat(), "n": len(new), "records": new}


def summary() -> dict:
    """滚动技能摘要（供面板与 latest.json 引用）。

    含可靠性分箱（reliability diagram 原料：每档预报概率的实测事件率）。
    """
    hist = _load_hist()
    recs = hist["records"]
    if not recs:
        return {"n": 0}
    brier = [r["brier"] for r in recs]
    hit = sum(1 for r in recs if r["triggered"] and r["observed_event"])
    miss = sum(1 for r in recs if not r["triggered"] and r["observed_event"])
    fa = sum(1 for r in recs if r["triggered"] and not r["observed_event"])
    csi = hit / (hit + miss + fa) if (hit + miss + fa) else None
    # 可靠性分箱（10 档；p=1.0 归入最高档）
    bins = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        grp = [r for r in recs
               if lo <= min(r["p_harm_heat"], 0.9999) < hi]
        if grp:
            bins.append({
                "bin": f"{lo:.1f}-{hi:.1f}", "n": len(grp),
                "mean_p": round(float(np.mean(
                    [r["p_harm_heat"] for r in grp])), 3),
                "event_rate": round(float(np.mean(
                    [r["observed_event"] for r in grp])), 3),
            })
    return {
        "n": len(recs),
        "mean_brier": round(float(np.mean(brier)), 4),
        "hit": hit, "miss": miss, "false_alarm": fa,
        "csi": round(csi, 3) if csi is not None else None,
        "p_trigger": P_TRIG,
        "since": min(r["valid_date"] for r in recs),
        "last": max(r["valid_date"] for r in recs),
        "obs_sources": sorted({r["obs_source"] for r in recs}),
        "reliability_bins": bins,
    }


def latest_verifiable() -> date | None:
    """最近一个已过有效日且有归档预报的日期（今天-1 起，最多回看 10 天）。"""
    hd = history_dir()
    for back in range(1, 11):
        d = date.today() - timedelta(days=back)
        if (hd / f"forecast_{d.isoformat()}.json").exists():
            return d
    return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()
    v = date.fromisoformat(args.valid) if args.valid else latest_verifiable()
    if v is None:
        print("no archived forecast to verify yet")
    else:
        out = verify(v)
        print(json.dumps({"verified": out["n"], "valid": out["valid_date"],
                          "summary": summary()}, ensure_ascii=False, indent=2))
