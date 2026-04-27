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

The final outputs are **drought probability layers** published as GEE assets under the UNICEF CCRI project. Rather than showing the SPI/SPEI index value for a single month, each asset answers the question: *"based on the full 1958–2025 record, how often does this pixel experience drought of a given severity?"*

Each asset is a single multi-band image at 0.0417° (~4.6 km) resolution covering the full globe.

### SPI-12 Probability Layer

**Asset:** `projects/unicef-ccri/assets/droughts/spi12_TerraClimate_1958-2025`

### SPEI-12 Probability Layer

**Asset:** `projects/unicef-ccri/assets/droughts/spei12_TerraClimate_1958-2025`

### Band information

| Band | Name | Description | Range |
|------|------|-------------|-------|
| 1 | `prob_spi_neg1p0` / `prob_spei_neg1p0` | P(index ≤ −1.0) — probability of moderate drought or worse | 0 to 1 |
| 2 | `prob_spi_neg1p5` / `prob_spei_neg1p5` | P(index ≤ −1.5) — probability of severe drought or worse | 0 to 1 |
| 3 | `prob_spi_neg2p0` / `prob_spei_neg2p0` | P(index ≤ −2.0) — probability of extreme drought | 0 to 1 |
| 4 | `quality_flag` | Data quality: 1 = high, 2 = medium, 3 = low | 1, 2, or 3 |

**Quality flag criteria:**
- **High (1):** ≥ 30 months with non-zero precipitation AND mean annual precipitation ≥ 200 mm
- **Medium (2):** ≥ 10 months with non-zero precipitation AND mean annual precipitation ≥ 100 mm
- **Low (3):** all other land pixels (very arid or sparse data)
- **NoData:** ocean, permanent ice, and pixels with no valid SPI/SPEI values

Probability values are computed from the full 1958–2025 record (804 monthly time steps). A pixel with `prob_spi_neg1p0 = 0.15` means it experienced moderate-or-worse drought in 15% of all months over that 67-year period.

### Example GEE code

```javascript
// Load SPI-12 probability layer
var spi12_prob = ee.Image('projects/unicef-ccri/assets/droughts/spi12_TerraClimate_1958-2025');

// Visualize probability of moderate drought (Band 1)
Map.addLayer(spi12_prob.select('prob_spi_neg1p0'), {
  min: 0, max: 0.4,
  palette: ['white', 'yellow', 'orange', 'red', 'darkred']
}, 'P(SPI-12 ≤ -1.0)');

// Visualize probability of extreme drought (Band 3)
Map.addLayer(spi12_prob.select('prob_spi_neg2p0'), {
  min: 0, max: 0.15,
  palette: ['white', 'orange', 'red']
}, 'P(SPI-12 ≤ -2.0)');

// Mask to high-quality pixels only
var high_quality = spi12_prob.select('quality_flag').eq(1);
var prob_hq = spi12_prob.select('prob_spi_neg1p0').updateMask(high_quality);

// Load SPEI-12 probability layer
var spei12_prob = ee.Image('projects/unicef-ccri/assets/droughts/spei12_TerraClimate_1958-2025');
```


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
