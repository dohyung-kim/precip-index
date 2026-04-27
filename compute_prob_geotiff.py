"""
SPI Probability Layers + Quality Flags → GeoTIFF
=================================================

Replicates UN-SPIDER methodology from spi_unspider_perpixel_final.py
using local TerraClimate SPI-12 output.

OUTPUT (4-band GeoTIFF):
  Band 1: prob_spi_neg1p0  — P(SPI-12 ≤ -1.0) across all months
  Band 2: prob_spi_neg1p5  — P(SPI-12 ≤ -1.5)
  Band 3: prob_spi_neg2p0  — P(SPI-12 ≤ -2.0)
  Band 4: quality_flag     — 1=high, 2=medium, 3=low

Quality flag criteria (matching GEE script):
  High   : non-zero precip months ≥ 30  AND mean annual precip ≥ 200 mm
  Medium : non-zero precip months ≥ 10  AND mean annual precip ≥ 100 mm
  Low    : everything else
"""

import gc
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPI_FILE   = "output/zarr_test/tile_spi_12.nc"
PPT_DIR    = "input"
OUTPUT_DIR = Path("output/geotiff")
OUTPUT_FILE = OUTPUT_DIR / "spi12_prob_quality.tif"

THRESHOLDS = [-1.0, -1.5, -2.0]
YEARS      = range(1958, 2026)   # 68 years → 816 months
NODATA     = -9999.0             # fill value written to GeoTIFF bands

# Quality thresholds (from GEE script constants)
QUALITY_HIGH_SAMPLES   = 30
QUALITY_MEDIUM_SAMPLES = 10
QUALITY_HIGH_PRECIP    = 200   # mm / year
QUALITY_MEDIUM_PRECIP  = 100   # mm / year

# Spatial chunk size for dask (controls peak RAM during prob reduction)
# Each chunk: time_full × CHUNK_LAT × CHUNK_LON × 4 bytes
# 816 × 60 × 1440 × 4 ≈ 283 MB — safe on 8 GB available RAM
CHUNK_LAT = 60
CHUNK_LON = 1440

# Intermediate checkpoint dir (avoids redoing Steps 1+2 if Step 3 fails)
CACHE_DIR = Path("output/geotiff/.cache")

# ---------------------------------------------------------------------------
# Step 1: Probability maps from SPI-12
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PROBS   = CACHE_DIR / "prob_bands.npy"
CACHE_QUALITY = CACHE_DIR / "quality.npy"
CACHE_COORDS  = CACHE_DIR / "coords.npz"

print("=" * 60)
print("STEP 1: Probability maps from SPI-12")
print("=" * 60)

if CACHE_PROBS.exists() and CACHE_COORDS.exists():
    print("  Loading cached probability bands...")
    prob_bands = list(np.load(CACHE_PROBS))
    coords = np.load(CACHE_COORDS)
    lat, lon = coords["lat"], coords["lon"]
    nlat, nlon = len(lat), len(lon)
    print(f"  Loaded {len(prob_bands)} bands, grid {nlat}×{nlon}")
else:
    print(f"  Opening {SPI_FILE} (lazy)...")
    ds_spi = xr.open_dataset(
        SPI_FILE,
        chunks={"time": -1, "lat": CHUNK_LAT, "lon": CHUNK_LON},
    )
    spi = ds_spi["spi_gamma_12_month"]

    nlat   = spi.sizes["lat"]
    nlon   = spi.sizes["lon"]
    n_time = spi.sizes["time"]
    lat    = ds_spi["lat"].values
    lon    = ds_spi["lon"].values

    print(f"  Shape: {spi.shape}  ({n_time} months, {nlat}×{nlon} grid)")

    # Count valid (non-NaN) time steps per pixel.
    # Ocean and permanent no-data pixels have NaN for every time step → valid_count=0.
    print("  Computing valid-time-step count per pixel...")
    valid_count = spi.notnull().sum("time").compute().values.astype(np.float32)
    has_data = valid_count > 0   # True where at least one valid SPI value exists

    prob_bands = []
    for thresh in THRESHOLDS:
        name = f"prob_spi_{str(thresh).replace('-','neg').replace('.','p')}"
        print(f"  Computing {name}...")
        below = (spi <= thresh).sum("time").compute().values.astype(np.float32)
        # Divide by per-pixel valid count; pixels with no valid data → NaN (NoData)
        prob = np.where(has_data, below / np.where(has_data, valid_count, 1), np.nan)
        prob_bands.append(prob.astype(np.float32))
        valid_vals = prob[has_data]
        print(f"    range: {valid_vals.min():.4f} – {valid_vals.max():.4f}  "
              f"({(~has_data).sum():,} nodata pixels)")

    ds_spi.close()
    gc.collect()

    print("  Caching prob bands to disk...")
    np.save(CACHE_PROBS, np.stack(prob_bands))
    np.savez(CACHE_COORDS, lat=lat, lon=lon)

# ---------------------------------------------------------------------------
# Step 2: Quality flags from raw PPT
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2: Quality flags from PPT (year-by-year)")
print("=" * 60)

if CACHE_QUALITY.exists():
    print("  Loading cached quality flag...")
    quality = np.load(CACHE_QUALITY)   # float32, NaN = nodata
    valid_q = ~np.isnan(quality)
    n_high   = (quality[valid_q] == 1).sum()
    n_medium = (quality[valid_q] == 2).sum()
    n_low    = (quality[valid_q] == 3).sum()
    print(f"  High={n_high:,}  Medium={n_medium:,}  Low={n_low:,}  NoData={(~valid_q).sum():,} pixels")
else:
    nonzero_count  = np.zeros((nlat, nlon), dtype=np.int32)
    total_precip   = np.zeros((nlat, nlon), dtype=np.float64)
    n_years        = len(list(YEARS))

    for year in YEARS:
        ppt_file = f"{PPT_DIR}/TerraClimate_ppt_{year}.nc"
        sys.stdout.write(f"  {year} ({year - min(YEARS) + 1}/{n_years})...\r")
        sys.stdout.flush()

        ds_ppt = xr.open_dataset(ppt_file)
        ppt = ds_ppt["ppt"].values  # (12, 4320, 8640) float64

        nonzero_count += (ppt > 0).sum(axis=0).astype(np.int32)
        total_precip  += np.nansum(ppt, axis=0)

        ds_ppt.close()
        del ppt
        gc.collect()

    print()

    mean_annual_precip = total_precip / n_years
    print(f"  Non-zero count range : {nonzero_count.min()} – {nonzero_count.max()}")
    print(f"  Mean annual precip   : {np.nanmin(mean_annual_precip):.1f} – {np.nanmax(mean_annual_precip):.1f} mm/yr")

    print("  Building quality flag...")
    # Use the SPI NoData mask so ocean/permanent-nodata pixels get NaN, not "low".
    # prob_bands must already be computed before reaching here.
    nodata_mask = np.isnan(prob_bands[0])

    # Default: low quality (3) for all land pixels; NaN for nodata pixels
    quality = np.where(nodata_mask, np.nan, 3.0).astype(np.float32)

    medium_mask = (
        (nonzero_count >= QUALITY_MEDIUM_SAMPLES) &
        (mean_annual_precip >= QUALITY_MEDIUM_PRECIP) &
        (~nodata_mask)
    )
    quality[medium_mask] = 2

    high_mask = (
        (nonzero_count >= QUALITY_HIGH_SAMPLES) &
        (mean_annual_precip >= QUALITY_HIGH_PRECIP) &
        (~nodata_mask)
    )
    quality[high_mask] = 1

    n_high   = int(high_mask.sum())
    n_medium = int(medium_mask.sum()) - n_high
    n_low    = int((~nodata_mask & ~medium_mask).sum())
    print(f"  High={n_high:,}  Medium={n_medium:,}  Low={n_low:,}  NoData={nodata_mask.sum():,} pixels")

    del nonzero_count, total_precip, mean_annual_precip, medium_mask, high_mask, nodata_mask
    gc.collect()

    print("  Caching quality flag to disk...")
    np.save(CACHE_QUALITY, quality)  # saved as float32 with NaN for nodata

# ---------------------------------------------------------------------------
# Step 3: Write Cloud-Optimised GeoTIFF
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3: Writing GeoTIFF")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build affine transform from coordinate arrays
# lat is descending (north-first), lon is ascending
res_lat = abs(lat[1] - lat[0])
res_lon = abs(lon[1] - lon[0])

transform = from_bounds(
    lon.min() - res_lon / 2,  # west
    lat.min() - res_lat / 2,  # south
    lon.max() + res_lon / 2,  # east
    lat.max() + res_lat / 2,  # north
    nlon,                     # width
    nlat,                     # height
)

BAND_META = [
    ("prob_spi_neg1p0", "P(SPI-12 ≤ -1.0) — moderate drought or worse"),
    ("prob_spi_neg1p5", "P(SPI-12 ≤ -1.5) — severe drought or worse"),
    ("prob_spi_neg2p0", "P(SPI-12 ≤ -2.0) — extreme drought"),
    ("quality_flag",   "1=high 2=medium 3=low data quality"),
]

# Replace NaN with NODATA sentinel before writing so rasterio clients
# (GDAL, QGIS, etc.) correctly mask ocean and permanent-nodata pixels.
all_bands = [np.where(np.isnan(b), NODATA, b).astype(np.float32)
             for b in prob_bands + [quality]]

print(f"  Output : {OUTPUT_FILE}")
print(f"  Size   : {nlat} × {nlon}, 4 bands  nodata={NODATA}")

with rasterio.open(
    OUTPUT_FILE,
    "w",
    driver="GTiff",
    height=nlat,
    width=nlon,
    count=4,
    dtype="float32",
    crs=CRS.from_epsg(4326),
    transform=transform,
    compress="deflate",
    predictor=2,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="YES",
    nodata=NODATA,
) as dst:
    for band_idx, (data, (bname, bdesc)) in enumerate(zip(all_bands, BAND_META), start=1):
        print(f"  Writing band {band_idx}: {bname}...")
        dst.write(data, band_idx)
        dst.update_tags(band_idx, name=bname, description=bdesc)

    dst.update_tags(
        source="precip-index / TerraClimate SPI-12 1958-2025",
        spi_timescale="12",
        calibration="1991-2020",
        thresholds=str(THRESHOLDS),
    )

print()
size_gb = OUTPUT_FILE.stat().st_size / 1e9
print(f"  Done!  {size_gb:.2f} GB  →  {OUTPUT_FILE}")
print()
print("Band summary:")
for i, (bname, bdesc) in enumerate(BAND_META, start=1):
    print(f"  Band {i}: {bname}  — {bdesc}")
