"""
Patch missing lat rows in tile_spi_12.nc.

The VS Code crash during chunked SPI computation dropped two lat chunks
covering rows 1800–2160 (lat 14.98° to 0.02°, tropical Africa/Asia/Americas).
This script recomputes SPI-12 for just those rows and patches them back in.
"""

import os
import sys
import tempfile
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import xarray as xr
import netCDF4 as nc4

from indices import spi_global

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
ZARR_STORE  = "output/zarr_test/tile_ppt.zarr"
SPI_FILE    = "output/zarr_test/tile_spi_12.nc"
TEMP_SPI    = "output/zarr_test/tile_spi_12_patch.nc"

MISSING_LAT_START = 1800   # row index (inclusive)
MISSING_LAT_END   = 2160   # row index (exclusive)

CALIB_START = 1991
CALIB_END   = 2020
SCALE       = 12
CHUNK_SIZE  = 180

print("=" * 60)
print("PATCH: missing SPI rows 1800–2160 (lat 14.98° to 0.02°)")
print("=" * 60)

# -----------------------------------------------------------------------
# Step 1: Extract missing lat slice from zarr PPT store
# -----------------------------------------------------------------------
print("\n[1] Reading PPT slice from zarr...")
ds_zarr = xr.open_zarr(ZARR_STORE)
ppt_slice = ds_zarr["ppt"].isel(lat=slice(MISSING_LAT_START, MISSING_LAT_END))
print(f"    Shape: {ppt_slice.shape}")
print(f"    Lat  : {float(ppt_slice.lat[0]):.2f} to {float(ppt_slice.lat[-1]):.2f}")

# Wrap in a Dataset so spi_global can consume it
ds_slice = xr.Dataset({"ppt": ppt_slice})

# -----------------------------------------------------------------------
# Step 2: Compute SPI-12 on the slice
# -----------------------------------------------------------------------
print(f"\n[2] Computing SPI-{SCALE} on slice → {TEMP_SPI}")
if os.path.exists(TEMP_SPI):
    os.remove(TEMP_SPI)

spi_global(
    precip_path=ds_slice,
    output_path=TEMP_SPI,
    scale=SCALE,
    calibration_start_year=CALIB_START,
    calibration_end_year=CALIB_END,
    chunk_size=CHUNK_SIZE,
    distribution="gamma",
    save_params=False,
    use_gpu=True,
)

ds_zarr.close()
del ds_slice, ppt_slice
gc.collect()

# -----------------------------------------------------------------------
# Step 3: Read patch result and verify
# -----------------------------------------------------------------------
print(f"\n[3] Verifying patch output...")
ds_patch = xr.open_dataset(TEMP_SPI)
# Find the SPI variable name
spi_var = [v for v in ds_patch.data_vars if "spi" in v.lower()][0]
patch_data = ds_patch[spi_var].values  # (time, lat_patch, lon)
print(f"    Variable: {spi_var}")
print(f"    Shape   : {patch_data.shape}")
print(f"    Non-NaN : {(~np.isnan(patch_data)).sum():,}")
print(f"    Range   : {np.nanmin(patch_data):.3f} to {np.nanmax(patch_data):.3f}")
ds_patch.close()

# -----------------------------------------------------------------------
# Step 4: Write patch rows back into tile_spi_12.nc
# -----------------------------------------------------------------------
print(f"\n[4] Patching {SPI_FILE} rows {MISSING_LAT_START}:{MISSING_LAT_END}...")

with nc4.Dataset(SPI_FILE, "r+") as dst:
    # Find the SPI variable in the target file
    spi_vars = [v for v in dst.variables if "spi" in v.lower() and v not in ("time", "lat", "lon")]
    if not spi_vars:
        raise RuntimeError(f"No SPI variable found in {SPI_FILE}. Variables: {list(dst.variables)}")
    target_var = spi_vars[0]
    print(f"    Target variable: {target_var}")
    print(f"    Target shape   : {dst.variables[target_var].shape}")

    # Write: variable is (time, lat, lon)
    dst.variables[target_var][:, MISSING_LAT_START:MISSING_LAT_END, :] = patch_data

print("    Done writing.")

# -----------------------------------------------------------------------
# Step 5: Verify patch was applied
# -----------------------------------------------------------------------
print(f"\n[5] Verifying patch in {SPI_FILE}...")
ds_check = xr.open_dataset(SPI_FILE)
check_var = [v for v in ds_check.data_vars if "spi" in v.lower()][0]
sample = ds_check[check_var].isel(
    time=400,
    lat=slice(MISSING_LAT_START, MISSING_LAT_END)
).load().values

valid = (~np.isnan(sample)).sum()
print(f"    Valid pixels in patched rows (time=400): {valid:,}")
ds_check.close()

if valid > 0:
    print("\n✓ Patch successful!")
    print(f"  Cleaning up temp file: {TEMP_SPI}")
    os.remove(TEMP_SPI)
else:
    print("\n✗ Patch failed — no valid data in patched rows.")
    print(f"  Temp file kept at: {TEMP_SPI}")

print("\nDone. Re-run compute_prob_geotiff.py to regenerate the GeoTIFF.")
print("(Delete output/geotiff/.cache/ first to force recomputation)")
