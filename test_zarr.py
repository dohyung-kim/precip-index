"""
Zarr pre-conversion benchmark for SPI computation.

Two steps:
  1. --convert  Extract one 720×720 tile from all per-year NetCDF files
                and write to a Zarr store (chunked along full time axis).
  2. --bench    Compare I/O load time (files vs Zarr) and run a full
                SPI-12 on the Zarr tile with spi_global().

Usage:
    python test_zarr.py              # convert + bench
    python test_zarr.py --convert    # only convert
    python test_zarr.py --bench      # only benchmark (needs existing Zarr)
    python test_zarr.py --convert --overwrite  # re-convert
"""

import argparse
import gc
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import xarray as xr

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR   = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'output', 'zarr_test')
PRECIP_GLOB = os.path.join(INPUT_DIR, 'TerraClimate_ppt_*.nc')
ZARR_STORE  = os.path.join(OUTPUT_DIR, 'tile_ppt.zarr')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full global grid (4320 × 8640)
TILE_LAT = slice(0, 4320)
TILE_LON = slice(0, 8640)

# Zarr chunking strategy:
#   time=-1  → all 816 months in one temporal chunk per spatial block
#   lat/lon  → spatial block size (180 fits ~100 MB per chunk at float32)
ZARR_CHUNK_TIME = -1
ZARR_CHUNK_LAT  = 180
ZARR_CHUNK_LON  = 180

# SPI settings (used in the full SPI run on the Zarr tile)
CALIB_START = 1991
CALIB_END   = 2020
SCALE       = 12
CHUNK_SIZE  = 180   # spatial chunk size passed to spi_global


# ============================================================
# STEP 1 — CONVERT
# ============================================================

def convert_tile_to_zarr(overwrite: bool = False) -> None:
    if os.path.exists(ZARR_STORE) and not overwrite:
        print(f"Zarr store already exists: {ZARR_STORE}")
        print("Pass --overwrite to redo.")
        return

    files = sorted(glob.glob(PRECIP_GLOB))
    if not files:
        print(f"No files found: {PRECIP_GLOB}")
        return

    print(f"Found {len(files)} per-year files.")
    print(f"Opening with open_mfdataset …", end=' ', flush=True)
    t0 = time.perf_counter()
    ds = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 12, 'lat': ZARR_CHUNK_LAT, 'lon': ZARR_CHUNK_LON})
    print(f"{time.perf_counter() - t0:.1f}s")

    tile = ds[['ppt']].isel(lat=TILE_LAT, lon=TILE_LON)
    n_time = tile.sizes['time']
    n_lat  = tile.sizes['lat']
    n_lon  = tile.sizes['lon']
    chunk_t = n_time if ZARR_CHUNK_TIME == -1 else ZARR_CHUNK_TIME

    print(f"Tile shape : time={n_time}, lat={n_lat}, lon={n_lon}")
    print(f"Zarr chunks: time={chunk_t}, lat={ZARR_CHUNK_LAT}, lon={ZARR_CHUNK_LON}")

    tile_chunked = tile.chunk({
        'time': chunk_t,
        'lat':  ZARR_CHUNK_LAT,
        'lon':  ZARR_CHUNK_LON,
    }).astype(np.float32)

    # Remove encoding from source files so Zarr can set its own
    for var in tile_chunked.data_vars:
        tile_chunked[var].encoding.clear()

    print(f"Writing Zarr → {ZARR_STORE} …", flush=True)
    t0 = time.perf_counter()
    tile_chunked.to_zarr(ZARR_STORE, mode='w')
    elapsed = time.perf_counter() - t0

    size_gb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(ZARR_STORE)
        for f in fs
    ) / 1e9

    print(f"Done  |  elapsed: {elapsed:.1f}s  |  Zarr size on disk: {size_gb:.2f} GB")


# ============================================================
# STEP 2 — BENCHMARK
# ============================================================

def _time_load(da: xr.DataArray, label: str, max_load_gb: float = 4.0) -> float:
    """
    Time loading a DataArray into RAM.

    For large tiles that would exceed max_load_gb, load only the first
    time slice to get a representative I/O rate without crashing.
    """
    total_gb = np.prod(da.shape) * da.dtype.itemsize / 1e9
    if total_gb > max_load_gb:
        # Sample: first time step only → same I/O path, safe on 16 GB
        sample = da.isel(time=0)
        print(f"  [{label}] tile too large ({total_gb:.1f} GB > {max_load_gb:.0f} GB limit) "
              f"— timing first time-step load …", end=' ', flush=True)
        t0 = time.perf_counter()
        arr = sample.load()
        elapsed = time.perf_counter() - t0
        slice_gb = arr.values.nbytes / 1e9
        # Extrapolate to full tile
        n_time = da.sizes['time']
        est_total = elapsed * n_time
        print(f"{elapsed:.1f}s / step  (×{n_time} steps ≈ {est_total:.0f}s est.  "
              f"slice={slice_gb*1e3:.0f} MB)")
        del arr
        return est_total
    else:
        print(f"  [{label}] loading into RAM …", end=' ', flush=True)
        t0 = time.perf_counter()
        arr = da.load()
        elapsed = time.perf_counter() - t0
        gb = arr.values.nbytes / 1e9
        print(f"{elapsed:.1f}s  ({gb:.2f} GB)")
        del arr
        return elapsed


def benchmark() -> None:
    files = sorted(glob.glob(PRECIP_GLOB))
    if not files:
        print(f"No input files found: {PRECIP_GLOB}")
        return

    if not os.path.exists(ZARR_STORE):
        print(f"Zarr store not found: {ZARR_STORE}")
        print("Run with --convert first.")
        return

    print("\n=== I/O BENCHMARK: per-year NetCDF vs Zarr ===\n")
    print(f"Tile: lat[{TILE_LAT.start}:{TILE_LAT.stop}]  "
          f"lon[{TILE_LON.start}:{TILE_LON.stop}]")

    # --- A: open_mfdataset + isel ---
    print("\n[A] Per-year NetCDF files")
    t0 = time.perf_counter()
    ds_files = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 12, 'lat': ZARR_CHUNK_LAT, 'lon': ZARR_CHUNK_LON})
    da_files = ds_files['ppt'].isel(lat=TILE_LAT, lon=TILE_LON)
    open_a = time.perf_counter() - t0
    print(f"  open_mfdataset + isel: {open_a:.2f}s")
    load_a = _time_load(da_files, 'files')
    total_a = open_a + load_a
    print(f"  → total open+load: {total_a:.1f}s")
    del da_files, ds_files
    gc.collect()

    # --- B: open_zarr ---
    print("\n[B] Zarr store")
    t0 = time.perf_counter()
    ds_zarr = xr.open_zarr(ZARR_STORE)
    da_zarr = ds_zarr['ppt']
    open_b = time.perf_counter() - t0
    print(f"  open_zarr: {open_b:.4f}s")
    load_b = _time_load(da_zarr, 'zarr')
    total_b = open_b + load_b
    print(f"  → total open+load: {total_b:.1f}s")
    del da_zarr, ds_zarr
    gc.collect()

    # --- Summary ---
    print("\n=== RESULTS ===")
    print(f"  Per-year files  open+load: {total_a:.1f}s")
    print(f"  Zarr            open+load: {total_b:.1f}s")
    if total_b > 0:
        print(f"  I/O speedup:   {total_a / total_b:.2f}x")

    # --- SPI-12 on Zarr tile ---
    print(f"\n=== SPI-{SCALE} on Zarr tile ===\n")
    from indices import spi_global
    from gpu import gpu_info
    print(f"Device: {gpu_info()}")

    spi_out = os.path.join(OUTPUT_DIR, f'tile_spi_{SCALE}.nc')
    ds_for_spi = xr.open_zarr(ZARR_STORE)
    t0 = time.perf_counter()
    spi_global(
        precip_path=ds_for_spi,   # pass Dataset directly — avoids open_nc path
        output_path=spi_out,
        scale=SCALE,
        calibration_start_year=CALIB_START,
        calibration_end_year=CALIB_END,
        chunk_size=CHUNK_SIZE,
        distribution='gamma',
        save_params=True,
        use_gpu=True,
    )
    spi_elapsed = time.perf_counter() - t0
    print(f"\nSPI-{SCALE} done in {spi_elapsed:.1f}s  →  {spi_out}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert',   action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--bench',     action='store_true')
    args = parser.parse_args()

    if not args.convert and not args.bench:
        args.convert = True
        args.bench   = True

    if args.convert:
        convert_tile_to_zarr(overwrite=args.overwrite)

    if args.bench:
        benchmark()
