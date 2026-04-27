"""
SPEI Computation + Probability Layers → GeoTIFF
================================================

Computes SPEI for each configured scale from per-year TerraClimate NetCDF
files (ppt + pet), then generates 4-band Cloud-Optimised GeoTIFFs matching
the SPI probability output format.

OUTPUT per scale:
  output/netcdf/tc_spei_gamma_{scale}_month.nc       — SPEI values
  output/netcdf/tc_spei_gamma_{scale}_month_params.nc — gamma fit params
  output/geotiff/spei{scale}_prob_quality.tif         — 4-band COG

GeoTIFF bands:
  Band 1: P(SPEI ≤ -1.0) — moderate drought or worse
  Band 2: P(SPEI ≤ -1.5) — severe drought or worse
  Band 3: P(SPEI ≤ -2.0) — extreme drought
  Band 4: quality_flag   — 1=high / 2=medium / 3=low / nodata=ocean

Usage:
    python run_spei.py
"""

import gc
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gpu import gpu_info
from indices import spei_global

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_NC  = os.path.join(os.path.dirname(__file__), 'output', 'netcdf')
OUTPUT_TIF = Path(os.path.dirname(__file__)) / 'output' / 'geotiff'
CACHE_DIR  = OUTPUT_TIF / '.spei_cache'

PRECIP_GLOB = os.path.join(INPUT_DIR, 'TerraClimate_ppt_*.nc')
PET_GLOB    = os.path.join(INPUT_DIR, 'TerraClimate_pet_*.nc')

YEAR_START  = 1958
YEAR_END    = 2025

CALIB_START = 1991
CALIB_END   = 2020

# Scales to compute
SPEI_SCALES = [12]

# chunk_size=720 → 72 chunks, safe for 16 GB VRAM + 16 GB RAM
CHUNK_SIZE = 720
USE_GPU    = True

# Probability thresholds (drought categories)
THRESHOLDS = [-1.0, -1.5, -2.0]
NODATA     = -9999.0

# Quality flag thresholds (same as SPI, based on raw PPT record)
QUALITY_HIGH_SAMPLES   = 30
QUALITY_MEDIUM_SAMPLES = 10
QUALITY_HIGH_PRECIP    = 200   # mm/year
QUALITY_MEDIUM_PRECIP  = 100   # mm/year

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_prob_bands(nc_path, var_name, cache_dir):
    """
    Load SPEI from nc_path and compute per-pixel drought probability bands.
    Divides by valid (non-NaN) time steps per pixel — ocean/nodata → NaN.
    Results are cached to avoid recomputation if the GeoTIFF step fails.
    """
    cache_probs  = cache_dir / f'{var_name}_prob_bands.npy'
    cache_coords = cache_dir / f'{var_name}_coords.npz'

    if cache_probs.exists() and cache_coords.exists():
        print('  Loading cached probability bands...')
        prob_bands = list(np.load(cache_probs))
        coords = np.load(cache_coords)
        lat, lon = coords['lat'], coords['lon']
        nlat, nlon = len(lat), len(lon)
        print(f'  Grid: {nlat}×{nlon}  bands: {len(prob_bands)}')
        return prob_bands, lat, lon, nlat, nlon

    print(f'  Opening {nc_path} ...')
    ds = xr.open_dataset(
        nc_path,
        chunks={'time': -1, 'lat': 60, 'lon': 1440},
    )
    da = ds[var_name]
    nlat, nlon, n_time = da.sizes['lat'], da.sizes['lon'], da.sizes['time']
    lat = ds['lat'].values
    lon = ds['lon'].values
    print(f'  Shape: {da.shape}')

    print('  Counting valid time steps per pixel...')
    valid_count = da.notnull().sum('time').compute().values.astype(np.float32)
    has_data    = valid_count > 0

    prob_bands = []
    for thresh in THRESHOLDS:
        label = f"prob_spei_{str(thresh).replace('-','neg').replace('.','p')}"
        print(f'  Computing {label} ...', end=' ', flush=True)
        below = (da <= thresh).sum('time').compute().values.astype(np.float32)
        prob  = np.where(has_data, below / np.where(has_data, valid_count, 1), np.nan)
        prob_bands.append(prob.astype(np.float32))
        valid_vals = prob[has_data]
        print(f'range {valid_vals.min():.4f}–{valid_vals.max():.4f}  '
              f'({(~has_data).sum():,} nodata pixels)')

    ds.close()
    gc.collect()

    np.save(cache_probs, np.stack(prob_bands))
    np.savez(cache_coords, lat=lat, lon=lon)
    print('  Cached.')

    return prob_bands, lat, lon, nlat, nlon


def compute_quality(nlat, nlon, nodata_mask, cache_dir):
    """
    Compute quality flag from raw PPT files.
    Ocean/permanent-nodata pixels (from nodata_mask) get NaN, not 'low'.
    Result is cached so it can be reused across multiple SPEI scales.
    """
    cache_quality = cache_dir / 'quality.npy'

    if cache_quality.exists():
        print('  Loading cached quality flag...')
        quality = np.load(cache_quality)   # float32, NaN = nodata
        valid_q = ~np.isnan(quality)
        print(f'  High={(quality[valid_q]==1).sum():,}  '
              f'Medium={(quality[valid_q]==2).sum():,}  '
              f'Low={(quality[valid_q]==3).sum():,}  '
              f'NoData={(~valid_q).sum():,} pixels')
        return quality

    years = range(YEAR_START, YEAR_END + 1)
    n_years = len(years)
    nonzero_count = np.zeros((nlat, nlon), dtype=np.int32)
    total_precip  = np.zeros((nlat, nlon), dtype=np.float64)

    print(f'  Reading {n_years} PPT files for quality flag...')
    for year in years:
        ppt_file = os.path.join(INPUT_DIR, f'TerraClimate_ppt_{year}.nc')
        sys.stdout.write(f'\r  {year} ({year - YEAR_START + 1}/{n_years})...')
        sys.stdout.flush()
        ds_ppt = xr.open_dataset(ppt_file)
        ppt = ds_ppt['ppt'].values
        nonzero_count += (ppt > 0).sum(axis=0).astype(np.int32)
        total_precip  += np.nansum(ppt, axis=0)
        ds_ppt.close()
        del ppt
        gc.collect()
    print()

    mean_annual = total_precip / n_years

    # Default: low (3) for land pixels, NaN for ocean/nodata
    quality = np.where(nodata_mask, np.nan, 3.0).astype(np.float32)

    medium_mask = (
        (nonzero_count >= QUALITY_MEDIUM_SAMPLES) &
        (mean_annual    >= QUALITY_MEDIUM_PRECIP) &
        (~nodata_mask)
    )
    quality[medium_mask] = 2

    high_mask = (
        (nonzero_count >= QUALITY_HIGH_SAMPLES) &
        (mean_annual    >= QUALITY_HIGH_PRECIP) &
        (~nodata_mask)
    )
    quality[high_mask] = 1

    n_high   = int(high_mask.sum())
    n_medium = int(medium_mask.sum()) - n_high
    n_low    = int((~nodata_mask & ~medium_mask).sum())
    print(f'  High={n_high:,}  Medium={n_medium:,}  Low={n_low:,}  '
          f'NoData={nodata_mask.sum():,} pixels')

    del nonzero_count, total_precip, mean_annual, medium_mask, high_mask
    gc.collect()

    np.save(cache_quality, quality)
    print('  Cached.')
    return quality


def write_geotiff(output_path, prob_bands, quality, lat, lon, nlat, nlon,
                  scale, label):
    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    transform = from_bounds(
        lon.min() - res_lon / 2, lat.min() - res_lat / 2,
        lon.max() + res_lon / 2, lat.max() + res_lat / 2,
        nlon, nlat,
    )

    band_meta = [
        (f'prob_spei_neg1p0', f'P({label} ≤ -1.0) moderate drought or worse'),
        (f'prob_spei_neg1p5', f'P({label} ≤ -1.5) severe drought or worse'),
        (f'prob_spei_neg2p0', f'P({label} ≤ -2.0) extreme drought'),
        ('quality_flag',      '1=high 2=medium 3=low data quality'),
    ]

    all_bands = [np.where(np.isnan(b), NODATA, b).astype(np.float32)
                 for b in prob_bands + [quality]]

    print(f'  Writing {output_path.name} ...', flush=True)
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=nlat, width=nlon, count=4, dtype='float32',
        crs=CRS.from_epsg(4326), transform=transform,
        compress='deflate', predictor=2, tiled=True,
        blockxsize=256, blockysize=256, BIGTIFF='YES',
        nodata=NODATA,
    ) as dst:
        for i, (data, (bname, bdesc)) in enumerate(zip(all_bands, band_meta), start=1):
            dst.write(data, i)
            dst.update_tags(i, name=bname, description=bdesc)
        dst.update_tags(
            source=f'precip-index / TerraClimate {label} {YEAR_START}-{YEAR_END}',
            calibration=f'{CALIB_START}-{CALIB_END}',
            thresholds=str(THRESHOLDS),
            nodata=str(NODATA),
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f'  Done  {size_mb:.0f} MB  →  {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Verify inputs exist
    ppt_files = sorted(glob.glob(PRECIP_GLOB))
    pet_files = sorted(glob.glob(PET_GLOB))
    if not ppt_files:
        sys.exit(f'ERROR: No PPT files found matching {PRECIP_GLOB}')
    if not pet_files:
        sys.exit(f'ERROR: No PET files found matching {PET_GLOB}')

    os.makedirs(OUTPUT_NC,  exist_ok=True)
    OUTPUT_TIF.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Device     : {gpu_info()}')
    print(f'PPT files  : {len(ppt_files)} ({ppt_files[0]} … {ppt_files[-1]})')
    print(f'PET files  : {len(pet_files)} ({pet_files[0]} … {pet_files[-1]})')
    print(f'Calibration: {CALIB_START}–{CALIB_END}')
    print(f'Scales     : {SPEI_SCALES}')
    print(f'Chunk size : {CHUNK_SIZE}  GPU: {USE_GPU}')
    print()

    quality = None   # computed once, reused across scales

    for scale in SPEI_SCALES:
        label    = f'SPEI-{scale}'
        var_name = f'spei_gamma_{scale}_month'
        nc_out   = os.path.join(OUTPUT_NC, f'tc_spei_gamma_{scale}_month.nc')
        tif_out  = OUTPUT_TIF / f'spei{scale}_prob_quality.tif'

        # ── Step 1: Compute SPEI ──────────────────────────────────────────────
        print(f'{"="*60}')
        print(f'STEP 1: Compute {label}')
        print(f'{"="*60}')
        spei_global(
            precip_path=PRECIP_GLOB,
            pet_path=PET_GLOB,
            output_path=nc_out,
            scale=scale,
            calibration_start_year=CALIB_START,
            calibration_end_year=CALIB_END,
            chunk_size=CHUNK_SIZE,
            distribution='gamma',
            save_params=True,
            use_gpu=USE_GPU,
        )
        print(f'  → {nc_out}')
        print()

        # ── Step 2: Probability bands ─────────────────────────────────────────
        print(f'{"="*60}')
        print(f'STEP 2: Probability bands for {label}')
        print(f'{"="*60}')
        prob_bands, lat, lon, nlat, nlon = compute_prob_bands(
            nc_out, var_name, CACHE_DIR
        )
        print()

        # ── Step 3: Quality flag (computed once from PPT, reused across scales)
        print(f'{"="*60}')
        print(f'STEP 3: Quality flag')
        print(f'{"="*60}')
        if quality is None:
            nodata_mask = np.isnan(prob_bands[0])
            quality = compute_quality(nlat, nlon, nodata_mask, CACHE_DIR)
        else:
            print('  Reusing quality flag from previous scale.')
        print()

        # ── Step 4: Write GeoTIFF ─────────────────────────────────────────────
        print(f'{"="*60}')
        print(f'STEP 4: Write GeoTIFF for {label}')
        print(f'{"="*60}')
        write_geotiff(tif_out, prob_bands, quality, lat, lon, nlat, nlon,
                      scale, label)
        print()

    print('All done.')
    print()
    print('Outputs:')
    for scale in SPEI_SCALES:
        nc_out  = os.path.join(OUTPUT_NC,  f'tc_spei_gamma_{scale}_month.nc')
        tif_out = OUTPUT_TIF / f'spei{scale}_prob_quality.tif'
        for p in [nc_out, str(tif_out)]:
            if os.path.exists(p):
                print(f'  {p}  ({os.path.getsize(p)/1e6:.0f} MB)')
