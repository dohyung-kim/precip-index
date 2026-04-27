"""
Run SPI and SPEI computation on TerraClimate global data.

Reads from Zarr stores (fast) if available, otherwise falls back to per-year
NetCDF files.  Run convert_to_zarr.py first to build the Zarr stores.

Usage:
    python convert_to_zarr.py   # once, to build Zarr stores
    python run_spi.py

Adjust chunk_size and use_gpu below to match your hardware.
System: 16 GB RAM + 16 GB VRAM  →  chunk_size=720, use_gpu=True
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import xarray as xr
from gpu import gpu_info
from indices import spi_global, spei_global

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'netcdf')
PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'output', 'params')
ZARR_DIR   = os.path.join(os.path.dirname(__file__), 'output', 'zarr_test')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Zarr stores (built by convert_to_zarr.py)
PPT_ZARR = os.path.join(ZARR_DIR, 'tile_ppt.zarr')
PET_ZARR = os.path.join(ZARR_DIR, 'tile_pet.zarr')

# Fallback: glob patterns for per-year NetCDF files
PRECIP_GLOB = os.path.join(INPUT_DIR, 'TerraClimate_ppt_*.nc')
PET_GLOB    = os.path.join(INPUT_DIR, 'TerraClimate_pet_*.nc')

# WMO standard calibration period
CALIB_START = 1991
CALIB_END   = 2020

# Hardware settings
# chunk_size=720 → 72 chunks, ~6 GB VRAM peak, ~5 GB RAM peak
CHUNK_SIZE = 720
USE_GPU    = True   # set False to force CPU


def open_precip():
    if os.path.exists(PPT_ZARR):
        print(f"Precip: Zarr  ({PPT_ZARR})")
        return xr.open_zarr(PPT_ZARR)
    print(f"Precip: NetCDF glob  ({PRECIP_GLOB})")
    return PRECIP_GLOB


def open_pet():
    if os.path.exists(PET_ZARR):
        print(f"PET:    Zarr  ({PET_ZARR})")
        return xr.open_zarr(PET_ZARR)
    import glob as _glob
    pet_files = sorted(_glob.glob(PET_GLOB))
    if not pet_files:
        return None
    print(f"PET:    NetCDF glob  ({len(pet_files)} files)")
    return PET_GLOB


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print(f"Device: {gpu_info()}")
    print(f"Calibration: {CALIB_START}-{CALIB_END}")
    print(f"Chunk size: {CHUNK_SIZE}  GPU: {USE_GPU}")
    print()

    precip_src = open_precip()
    pet_src    = open_pet()
    print()

    # ----------------------------------------------------------
    # SPI  (precipitation only)
    # ----------------------------------------------------------
    for scale in [12]:
        print(f"=== SPI-{scale} ===")
        spi_global(
            precip_path=precip_src,
            output_path=os.path.join(OUTPUT_DIR, f'tc_spi_gamma_{scale}_month.nc'),
            scale=scale,
            calibration_start_year=CALIB_START,
            calibration_end_year=CALIB_END,
            chunk_size=CHUNK_SIZE,
            distribution='gamma',
            save_params=True,
            use_gpu=USE_GPU,
        )
        print(f"  → output/netcdf/tc_spi_gamma_{scale}_month.nc")

    # ----------------------------------------------------------
    # SPEI  (requires PET)
    # ----------------------------------------------------------
    if pet_src is None:
        print("\nSPEI skipped: no PET Zarr store or NetCDF files found.")
    else:
        print()
        for scale in [12]:
            print(f"=== SPEI-{scale} ===")
            spei_global(
                precip_path=precip_src,
                pet_path=pet_src,
                output_path=os.path.join(OUTPUT_DIR, f'tc_spei_gamma_{scale}_month.nc'),
                scale=scale,
                calibration_start_year=CALIB_START,
                calibration_end_year=CALIB_END,
                chunk_size=CHUNK_SIZE,
                distribution='gamma',
                save_params=True,
                use_gpu=USE_GPU,
            )
            print(f"  → output/netcdf/tc_spei_gamma_{scale}_month.nc")

    print("\nDone.")
