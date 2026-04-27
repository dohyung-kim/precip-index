# GPU-Accelerated SPI & SPEI from TerraClimate

Global drought and wetness indices — Standardized Precipitation Index (SPI) and Standardized Precipitation-Evapotranspiration Index (SPEI) — computed from TerraClimate data using GPU-accelerated processing.

---

## What are SPI and SPEI?

**SPI (Standardized Precipitation Index)** measures precipitation anomalies relative to a long-term climatology. It uses only precipitation as input, fitted to a gamma distribution, and is standardized to a normal distribution so that values are comparable across locations and time scales.

**SPEI (Standardized Precipitation-Evapotranspiration Index)** extends SPI by incorporating the demand side of the water balance. It uses precipitation minus potential evapotranspiration (P − PET) as input, fitted to a Pearson Type III distribution. SPEI captures the effect of rising temperatures on drought severity — a dry period that was tolerable historically may become more severe drought when temperatures are higher.

Both indices use the same classification:

| Value | Category |
|-------|----------|
| ≥ 2.0 | Extremely wet |
| 1.5 to 1.99 | Very wet |
| 1.0 to 1.49 | Moderately wet |
| −0.99 to 0.99 | Near normal |
| −1.0 to −1.49 | Moderately dry |
| −1.5 to −1.99 | Severely dry |
| ≤ −2.0 | Extremely dry |

This repository computes both indices at **12-month time scale** (SPI-12, SPEI-12), which reflects long-term moisture conditions and is most relevant for agricultural drought, groundwater, and streamflow impacts.

---

## Input Data

**Source:** [TerraClimate](https://www.climatologylab.org/terraclimate.html) — a high-resolution global dataset of monthly climate variables.

| Parameter | Value |
|-----------|-------|
| Spatial resolution | 0.0417° (~4.6 km at equator) |
| Grid size | 4,320 × 8,640 pixels |
| Time period | January 1958 – December 2025 (67 years, 804 monthly time steps) |
| Calibration period | 1991–2020 (WMO standard) |
| Variables used | `ppt` (precipitation, mm) for SPI; `ppt` + `pet` (potential evapotranspiration, mm) for SPEI |
| Distribution | Gamma (SPI), Pearson Type III (SPEI) |
| Source URL | https://climate.northwestknowledge.net/TERRACLIMATE-DATA |

---

## Results on Google Earth Engine

The computed SPI-12 and SPEI-12 are published as assets on Google Earth Engine under the UNICEF CCRI project.

### SPI-12

**Asset:** `projects/unicef-ccri/assets/droughts/spi12_TerraClimate_1958-2025`

```javascript
var spi12 = ee.ImageCollection('projects/unicef-ccri/assets/droughts/spi12_TerraClimate_1958-2025');
```

### SPEI-12

**Asset:** `projects/unicef-ccri/assets/droughts/spei12_TerraClimate_1958-2025`

```javascript
var spei12 = ee.ImageCollection('projects/unicef-ccri/assets/droughts/spei12_TerraClimate_1958-2025');
```

### Band information

Each image in the collection represents one month. The index value is stored as a float32 band:

| Band | Name | Description | Range |
|------|------|-------------|-------|
| 1 | `spi_gamma_12_month` / `spei_gamma_12_month` | Standardized index value | −3.09 to 3.09 |

Negative values indicate drought (dry conditions), positive values indicate wet conditions. Pixels over ocean and permanent ice are masked (NoData).

### Example GEE code

```javascript
// Load SPI-12 ImageCollection
var spi12 = ee.ImageCollection('projects/unicef-ccri/assets/droughts/spi12_TerraClimate_1958-2025');

// Filter to a specific month (e.g. December 2023)
var spi_dec2023 = spi12.filterDate('2023-12-01', '2024-01-01').first();

// Visualize
Map.addLayer(spi_dec2023, {
  bands: ['spi_gamma_12_month'],
  min: -2.5,
  max: 2.5,
  palette: ['8B1A1A', 'DE2929', 'F3641D', 'FDC404', 'FFFFFF', '89CDFF', '2C7BB6', '1A3399']
}, 'SPI-12 Dec 2023');

// Extract drought area (SPI <= -1.0)
var drought = spi_dec2023.lte(-1.0);
```

---

## Pipeline

```
Step 1 — Download data
    python download_terraclimate.py --vars ppt pet

Step 2 — Process (Colab notebook)
    notebooks/01_compute_SPI_SPEI.ipynb

Step 3 — Validate (Colab notebook)
    notebooks/02_validation.ipynb
```

### Step 1: Download

`download_terraclimate.py` downloads TerraClimate per-year NetCDF files from the official server with validation and resume support:

```bash
python download_terraclimate.py               # ppt + pet (default)
python download_terraclimate.py --vars ppt    # ppt only (SPI only)
python download_terraclimate.py --years 2020 2025  # specific year range
```

Alternatively, use the bundled wget script for bulk download:

```bash
bash input/terraclimate_wget.sh
```

### Step 2: Compute SPI & SPEI

Open `notebooks/01_compute_SPI_SPEI.ipynb` in Google Colab. The notebook handles the full pipeline:

1. Mount Google Drive and restore cached files
2. Download TerraClimate data (ppt, pet) with Drive caching
3. Convert to Zarr for fast I/O
4. Compute SPI-12 and SPEI-12 using GPU (CuPy on A100)
5. Generate probability layers and quality flags as Cloud-Optimised GeoTIFF
6. Save all outputs to Google Drive

**Runtime:** Select `Runtime → Change runtime type → A100 GPU`

### Step 3: Validate

Open `notebooks/02_validation.ipynb` to verify the algorithm produces correct results by comparing the GPU output against the original author's reference output on the same Bali input data.

---

## GPU Acceleration

The three computationally intensive operations are accelerated with [CuPy](https://cupy.dev/) (CUDA-based drop-in replacement for NumPy):

| Operation | CPU approach | GPU approach |
|-----------|-------------|-------------|
| Rolling sum | Sliding window over time axis | Cumulative-sum trick on GPU |
| Gamma parameter fitting | Loop over calendar months | Parallel method-of-moments for all pixels simultaneously |
| Normal transform | `scipy.stats.gamma.cdf` + `norm.ppf` | `cupyx.scipy.special.gammainc` + `ndtri` |

Processing is done in spatial chunks via `ChunkedProcessor` so each chunk fits in GPU VRAM. Tested with a 16 GB GPU on the full global TerraClimate grid (4,320 × 8,640, 1958–2025).

CuPy is **optional** — if not installed or no CUDA device is found, all functions fall back silently to the CPU path.

---

## Validation

The GPU implementation was validated against the original author's reference outputs on identical input data (Bali subset, TerraClimate ~2022 vintage).

| Index | Correlation | RMSE | Notes |
|-------|-------------|------|-------|
| SPI-12 | **1.000000** | **0.000024** | Negligible floating-point rounding only |
| SPEI-12 | **1.000000** | **0.000024** | Negligible floating-point rounding only |

The differences are entirely due to float32 rounding — the algorithms are numerically identical. See `notebooks/02_validation.ipynb` for the full comparison including scatter plots, spatial bias maps, and per-pixel correlation maps.

---

## Repository Structure

```
precip-index/
├── notebooks/
│   ├── 01_compute_SPI_SPEI.ipynb   # Colab: download + compute + probability layer
│   └── 02_validation.ipynb          # Colab: compare against original reference output
│
├── src/                             # Core library
│   ├── gpu.py                       # CuPy GPU acceleration (rolling sum, gamma fit, transform)
│   ├── indices.py                   # spi_global(), spei_global() — top-level functions
│   ├── compute.py                   # Vectorized SPI/SPEI computation (CPU + GPU paths)
│   ├── chunked.py                   # Spatial tiling and checkpoint logic
│   ├── distributions.py             # Distribution fitting (Pearson III, Log-Logistic, etc.)
│   ├── config.py                    # Central configuration
│   └── utils.py                     # I/O, logging, array utilities
│
├── download_terraclimate.py         # Step 1: Download + validate TerraClimate NetCDF files
├── convert_to_zarr.py               # Pre-convert NetCDF → Zarr for fast I/O (local GPU runs)
├── run_spi.py                       # Local GPU: compute SPI on full global data
├── run_spei.py                      # Local GPU: compute SPEI on full global data
├── compute_prob_geotiff.py          # Generate probability layer GeoTIFF from SPI output
├── compare_bali_old_data.py         # Validation: compare outputs against original reference
├── patch_spi_missing.py             # Utility: patch missing lat rows from interrupted runs
├── test_zarr.py                     # Benchmark: compare NetCDF vs Zarr I/O performance
│
├── tests/
│   ├── terraclimate_bali_ppt_1958_2024.nc   # Reference precipitation data (Bali subset)
│   └── terraclimate_bali_pet_1958_2024.nc   # Reference PET data (Bali subset)
│
├── input/
│   └── terraclimate_wget.sh         # Bulk wget download script
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/dohyung-kim/precip-index.git
cd precip-index
pip install -r requirements.txt

# Optional GPU support (choose the version matching your CUDA install):
pip install cupy-cuda12x   # CUDA 12.x
# pip install cupy-cuda11x # CUDA 11.x
```

---

## Credits

**Original implementation:** [Benny Istanto](https://github.com/bennyistanto), GOST/DEC Data Group, The World Bank.  
Original repository: [bennyistanto/precip-index](https://github.com/bennyistanto/precip-index)  
Documentation: https://bennyistanto.github.io/precip-index/

This repository adapts the original CPU-based SPI/SPEI implementation with:
- **GPU acceleration** via CuPy for the three bottleneck operations (rolling sum, gamma fitting, normal transform)
- **Global-scale pipeline** for TerraClimate data (1958–2025, 4320 × 8640 grid)
- **Probability layer output** as Cloud-Optimised GeoTIFF (exceedance probabilities + quality flags)
- **Google Colab notebooks** for end-to-end processing on an A100 GPU
- **Validation notebooks** confirming numerical equivalence with the original algorithm

Both implementations build on the foundational work of [climate-indices](https://github.com/monocongo/climate_indices) by James Adams.

---

## License

BSD-3-Clause — see [LICENSE](LICENSE) for details.
