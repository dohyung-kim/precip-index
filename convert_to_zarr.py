"""
Convert TerraClimate per-year NetCDF files to Zarr stores.

Run once before run_spi.py to pre-convert input data.  Conversion is skipped
for any variable whose Zarr store already exists (pass --overwrite to redo).

Usage:
    python convert_to_zarr.py            # convert ppt + pet
    python convert_to_zarr.py --overwrite
"""

import argparse
import glob
import os
import time

import numpy as np
import xarray as xr

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'input')
ZARR_DIR   = os.path.join(os.path.dirname(__file__), 'output', 'zarr_test')

VARIABLES = {
    'ppt': {
        'glob':  os.path.join(INPUT_DIR, 'TerraClimate_ppt_*.nc'),
        'store': os.path.join(ZARR_DIR, 'tile_ppt.zarr'),
    },
    'pet': {
        'glob':  os.path.join(INPUT_DIR, 'TerraClimate_pet_*.nc'),
        'store': os.path.join(ZARR_DIR, 'tile_pet.zarr'),
    },
}

# Chunking: full time axis per spatial block, 180×180 spatial tiles
ZARR_CHUNK_LAT = 180
ZARR_CHUNK_LON = 180

# ============================================================
# CONVERSION
# ============================================================

def convert_variable(var: str, glob_pattern: str, zarr_store: str, overwrite: bool) -> None:
    if os.path.exists(zarr_store) and not overwrite:
        print(f"[{var}] Zarr store already exists — skipping. (--overwrite to redo)")
        print(f"       {zarr_store}")
        return

    files = sorted(glob.glob(glob_pattern))
    if not files:
        print(f"[{var}] No files found: {glob_pattern}")
        return

    print(f"[{var}] Found {len(files)} files.")
    print(f"[{var}] Opening with open_mfdataset …", end=' ', flush=True)
    t0 = time.perf_counter()
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        chunks={'time': 12, 'lat': ZARR_CHUNK_LAT, 'lon': ZARR_CHUNK_LON},
    )
    print(f"{time.perf_counter() - t0:.1f}s")

    tile = ds[[var]]
    n_time = tile.sizes['time']
    n_lat  = tile.sizes['lat']
    n_lon  = tile.sizes['lon']
    chunk_t = n_time  # full time axis per spatial block

    print(f"[{var}] Shape  : time={n_time}, lat={n_lat}, lon={n_lon}")
    print(f"[{var}] Chunks : time={chunk_t}, lat={ZARR_CHUNK_LAT}, lon={ZARR_CHUNK_LON}")

    tile_chunked = tile.chunk({
        'time': chunk_t,
        'lat':  ZARR_CHUNK_LAT,
        'lon':  ZARR_CHUNK_LON,
    }).astype(np.float32)

    for v in tile_chunked.data_vars:
        tile_chunked[v].encoding.clear()

    os.makedirs(ZARR_DIR, exist_ok=True)
    print(f"[{var}] Writing → {zarr_store} …", flush=True)
    t0 = time.perf_counter()
    tile_chunked.to_zarr(zarr_store, mode='w')
    elapsed = time.perf_counter() - t0

    size_gb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(zarr_store)
        for f in fs
    ) / 1e9
    print(f"[{var}] Done  |  {elapsed:.1f}s  |  {size_gb:.2f} GB on disk\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-convert even if Zarr store already exists')
    args = parser.parse_args()

    for var, cfg in VARIABLES.items():
        convert_variable(var, cfg['glob'], cfg['store'], overwrite=args.overwrite)

    print("All done.")
