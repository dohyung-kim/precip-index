"""
Compare SPI-12 and SPEI-12 results against the original author's Bali reference.

Reference files : tests/output/netcdf/
Our results     : output/netcdf/tc_spi_gamma_12_month.nc
                  output/netcdf/tc_spei_gamma_12_month.nc  (if computed)

Prints pixel-level and grid-mean statistics, then saves a multi-panel
figure to output/comparison_bali.png.
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
REF_DIR      = ROOT / 'tests' / 'output' / 'netcdf'
OUR_DIR      = ROOT / 'output' / 'netcdf'
OUTPUT_FIGURE = ROOT / 'output' / 'comparison_bali.png'

REFS = {
    'spi':  REF_DIR / 'spi_12_gamma.nc',
    'spei': REF_DIR / 'spei_12_pet_gamma.nc',
}
OURS = {
    'spi':  OUR_DIR / 'tc_spi_gamma_12_month.nc',
    'spei': OUR_DIR / 'tc_spei_gamma_12_month.nc',
}
VAR = {
    'spi':  'spi_gamma_12_month',
    'spei': 'spei_gamma_12_month',
}

# ── Load & align ──────────────────────────────────────────────────────────────

def load_and_align(index):
    ref_path = REFS[index]
    our_path = OURS[index]
    var      = VAR[index]

    if not ref_path.exists():
        print(f'  Reference not found: {ref_path}')
        return None, None
    if not our_path.exists():
        print(f'  Our output not found: {our_path}')
        return None, None

    print(f'\n[{index.upper()}-12] Loading reference ...')
    ds_ref = xr.open_dataset(ref_path)
    da_ref = ds_ref[var]

    # Bali bbox from reference coordinates
    lat_min = float(da_ref.lat.min()) - 0.02
    lat_max = float(da_ref.lat.max()) + 0.02
    lon_min = float(da_ref.lon.min()) - 0.02
    lon_max = float(da_ref.lon.max()) + 0.02

    print(f'  Bali bbox: lat [{lat_min:.3f}, {lat_max:.3f}]  '
          f'lon [{lon_min:.3f}, {lon_max:.3f}]')

    print(f'  Loading our global output and subsetting to Bali ...')
    ds_our = xr.open_dataset(our_path)
    da_our = ds_our[var].sel(
        lat=slice(lat_max, lat_min),   # lat is descending in our file
        lon=slice(lon_min, lon_max),
    )

    # Align to common time axis (inner join — only shared months)
    common_time = np.intersect1d(da_ref.time.values, da_our.time.values)
    da_ref = da_ref.sel(time=common_time)
    da_our = da_our.sel(time=common_time)

    print(f'  Reference grid : {da_ref.sizes}')
    print(f'  Our grid       : {da_our.sizes}')
    print(f'  Common months  : {len(common_time)}  '
          f'({str(common_time[0])[:7]} – {str(common_time[-1])[:7]})')

    # Sanity check — catch all-NaN output (incomplete/interrupted run)
    our_loaded = da_our.load()
    valid_frac = float(np.isfinite(our_loaded.values).mean())
    if valid_frac == 0.0:
        print(f'\n  *** ERROR: our {index.upper()}-12 output is entirely NaN. ***')
        print(f'  The file at {our_path}')
        print(f'  was pre-allocated but computation never wrote values to it.')
        print(f'  Delete it and re-run run_{index}.py before comparing.')
        return None, None
    if valid_frac < 0.1:
        print(f'  WARNING: only {valid_frac*100:.1f}% of our values are non-NaN '
              f'— computation may be incomplete.')

    # Interpolate our coarser-or-finer grid onto the reference grid
    da_our_interp = our_loaded.interp(
        lat=da_ref.lat, lon=da_ref.lon, method='linear'
    )

    return da_ref.load(), da_our_interp


# ── Statistics ────────────────────────────────────────────────────────────────

def stats(ref, our, label):
    """Compute flat pixel statistics between two aligned DataArrays."""
    r = ref.values.ravel()
    o = our.values.ravel()
    mask = np.isfinite(r) & np.isfinite(o)
    r, o = r[mask], o[mask]

    bias   = float(np.mean(o - r))
    mae    = float(np.mean(np.abs(o - r)))
    rmse   = float(np.sqrt(np.mean((o - r) ** 2)))
    corr   = float(np.corrcoef(r, o)[0, 1])
    n      = int(mask.sum())

    print(f'\n  {label} pixel-level statistics (n={n:,}):')
    print(f'    Correlation : {corr:.4f}')
    print(f'    RMSE        : {rmse:.4f}')
    print(f'    MAE         : {mae:.4f}')
    print(f'    Bias (ours−ref): {bias:+.4f}')

    # Spatial correlation map: correlation at each pixel over time
    n_lat, n_lon = ref.sizes['lat'], ref.sizes['lon']
    corr_map = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            rv = ref.values[:, i, j]
            ov = our.values[:, i, j]
            m  = np.isfinite(rv) & np.isfinite(ov)
            if m.sum() > 10:
                corr_map[i, j] = np.corrcoef(rv[m], ov[m])[0, 1]

    return dict(bias=bias, mae=mae, rmse=rmse, corr=corr, n=n,
                corr_map=corr_map, r_flat=r, o_flat=o)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure(results):
    indices = [k for k, v in results.items() if v is not None]
    n_idx   = len(indices)

    fig = plt.figure(figsize=(18, 6 * n_idx))
    fig.suptitle('SPI/SPEI-12 Comparison: Our Results vs Original Author (Bali)',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, index in enumerate(indices):
        da_ref, da_our, st = results[index]
        label = index.upper() + '-12'
        lat = da_ref.lat.values
        lon = da_ref.lon.values

        # Pick a central sample pixel
        ci = len(lat) // 2
        cj = len(lon) // 2
        ts_ref = da_ref.values[:, ci, cj]
        ts_our = da_our.values[:, ci, cj]
        time   = da_ref.time.values

        gs  = GridSpec(2, 3, figure=fig, top=1 - row / n_idx - 0.02,
                       bottom=1 - (row + 1) / n_idx + 0.04,
                       hspace=0.45, wspace=0.35)

        # Panel 1: Time series at sample pixel
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, ts_ref, lw=0.8, label='Reference', color='steelblue')
        ax1.plot(time, ts_our, lw=0.8, label='Ours', color='tomato',
                 linestyle='--', alpha=0.85)
        ax1.axhline(0, color='k', lw=0.4)
        ax1.set_title(f'{label} — time series at '
                      f'({lat[ci]:.2f}°N, {lon[cj]:.2f}°E)', fontsize=9)
        ax1.set_ylabel(label)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Scatter plot (all pixels × all times)
        ax2 = fig.add_subplot(gs[0, 2])
        lim  = max(abs(st['r_flat']).max(), abs(st['o_flat']).max())
        lim  = min(lim, 4.0)
        ax2.hexbin(st['r_flat'], st['o_flat'], gridsize=60, cmap='Blues',
                   mincnt=1, extent=(-lim, lim, -lim, lim))
        ax2.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        ax2.set_xlabel('Reference')
        ax2.set_ylabel('Ours')
        ax2.set_title(f'{label} scatter  r={st["corr"]:.3f}  '
                      f'RMSE={st["rmse"]:.3f}', fontsize=9)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)

        # Panel 3: Spatial correlation map
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(st['corr_map'], vmin=0.9, vmax=1.0, cmap='RdYlGn',
                         origin='upper',
                         extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         aspect='auto')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.set_title(f'{label} per-pixel correlation', fontsize=9)
        ax3.set_xlabel('Lon')
        ax3.set_ylabel('Lat')

        # Panel 4: Spatial bias map (mean over time)
        bias_map = (da_our - da_ref).mean('time').values
        ax4 = fig.add_subplot(gs[1, 1])
        vmax = max(abs(np.nanmin(bias_map)), abs(np.nanmax(bias_map)), 0.05)
        im4 = ax4.imshow(bias_map, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                         origin='upper',
                         extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         aspect='auto')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        ax4.set_title(f'{label} mean bias (ours − ref)', fontsize=9)
        ax4.set_xlabel('Lon')
        ax4.set_ylabel('Lat')

        # Panel 5: Stats text box
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        txt = (
            f'{label} pixel statistics\n'
            f'(n = {st["n"]:,} valid points)\n\n'
            f'Correlation : {st["corr"]:.4f}\n'
            f'RMSE        : {st["rmse"]:.4f}\n'
            f'MAE         : {st["mae"]:.4f}\n'
            f'Bias        : {st["bias"]:+.4f}\n\n'
            f'Corr map min: {np.nanmin(st["corr_map"]):.3f}\n'
            f'Corr map max: {np.nanmax(st["corr_map"]):.3f}\n'
            f'Corr map mean: {np.nanmean(st["corr_map"]):.4f}'
        )
        ax5.text(0.05, 0.95, txt, transform=ax5.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}

    for index in ['spi', 'spei']:
        da_ref, da_our = load_and_align(index)
        if da_ref is None:
            results[index] = None
            continue
        st = stats(da_ref, da_our, index.upper() + '-12')
        results[index] = (da_ref, da_our, st)

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        sys.exit('No valid comparisons could be made.')

    print(f'\nSaving figure → {OUTPUT_FIGURE}')
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(results)
    fig.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches='tight')
    plt.close(fig)
    size_kb = OUTPUT_FIGURE.stat().st_size / 1024
    print(f'Done  ({size_kb:.0f} KB)  →  {OUTPUT_FIGURE}')


if __name__ == '__main__':
    main()
