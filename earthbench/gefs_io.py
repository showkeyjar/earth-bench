"""GEFS grib 下载 / 缓存 / 读取的单一实现（四胞胎收口）。

cars_serve / cars_serve_impact / cars_verify / cars_verify_impact 此前各持一份
几乎相同的「下载到包内 gefs_cache + cfgrib 打开 + 域内 1° 插值」实现，修一个
bug 要同步四处。本模块收口为一个实现，并补上缓存目录的过期清理——GEFS 业务
档在线仅保留 ~10 天，append-only 缓存（本地实测 17 个文件 / 241MB 且只增不减）
在 init 日期滚出保留窗口后就是死重。
"""

from __future__ import annotations

import logging
import re
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PACKAGE_DATA = Path(__file__).parent / "data"
GEFS_CACHE = PACKAGE_DATA / "gefs_cache"  # 固定包内（不进 CARS_DATA_DIR，防 CI 把 grib 发布上网）
OPS = "https://noaa-gefs-pds.s3.amazonaws.com"

# 业务 GEFS 域内 1° 目标网格（与 CARS 冻结模型网格一致）
GRID_LAT = np.arange(50.0, 19.99, -1.0)
GRID_LON = np.arange(100.0, 140.01, 1.0)

_GEF_FILE_RE = re.compile(r"^(\d{8})\d{2}_gec00\.t\d{2}z\.pgrb2a\.0p50\.f\d+$")


def fetch_grib(init: str, step: str, hour: str = "00") -> Path:
    """下载（若缺）并返回业务 c00 pgrb2a 0p50 单时效文件的本地缓存路径。"""
    ymd = init.replace("-", "")
    name = f"gec00.t{hour}z.pgrb2a.0p50.f{step}"
    GEFS_CACHE.mkdir(parents=True, exist_ok=True)
    local = GEFS_CACHE / f"{ymd}{hour}_{name}"
    if not local.exists():
        urllib.request.urlretrieve(
            f"{OPS}/gefs.{ymd}/{hour}/atmos/pgrb2ap5/{name}", local)
    return local


def prune_gefs_cache(keep_days: int = 14) -> int:
    """删除 init 日期早于 keep_days 天前的缓存 grib，返回删除数量。

    文件名首 8 位即 init YYYYMMDD，文件按 init 日期不可变，按名清理安全；
    命名不匹配的文件不动。保留窗口取 14 天（GEFS 在线 ~10 天 + 回填惯例）。
    """
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y%m%d")
    if not GEFS_CACHE.exists():
        return 0
    removed = 0
    for fp in GEFS_CACHE.iterdir():
        m = _GEF_FILE_RE.match(fp.name)
        if not m or m.group(1) >= cutoff:
            continue
        try:
            fp.unlink()
            removed += 1
        except OSError as e:
            logger.warning(f"[gefs_io] prune failed for {fp.name}: {e}")
    if removed:
        logger.info(
            f"[gefs_io] pruned {removed} cached grib file(s) older than {keep_days} days")
    return removed


def read_field(path: Path, keys: dict, var_hint: tuple[str, ...]) -> np.ndarray:
    """读一个 grib 消息 → 域内 1° 网格 (31,41) float32。"""
    return read_fields(path, keys, {"field": var_hint})["field"]


def read_fields(
    path: Path, keys: dict, hints: dict[str, tuple[str, ...]]
) -> dict[str, np.ndarray]:
    """一次 cfgrib 打开读取同一 filter keys 下的多个变量。

    此前 u10/v10 各开一次（同一文件同 keys），5 个时效 ×3 变量 = 15 次打开；
    合并后同 keys 的变量共享一次 open。返回 {标签: (31,41) float32}。
    """
    import xarray as xr

    ds = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs={"indexpath": ""},
        filter_by_keys=keys)
    try:
        names = [n for n in ds.data_vars if n not in ("lat", "lon")]
        out: dict[str, np.ndarray] = {}
        for label, hint in hints.items():
            pick = next((n for n in names if n.startswith(hint)), names[0])
            f = ds[pick].sel(latitude=slice(50.0, 20.0),
                             longitude=slice(100.0, 140.0))
            out[label] = f.interp(latitude=GRID_LAT, longitude=GRID_LON
                                  ).values.astype(np.float32)
        return out
    finally:
        ds.close()
