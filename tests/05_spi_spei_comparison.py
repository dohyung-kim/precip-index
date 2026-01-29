#!/usr/bin/env python
"""
Test 05: SPI vs SPEI Comparison

This test compares SPI and SPEI to analyze the effect of including
evapotranspiration in the drought index:
- Correlation between SPI and SPEI
- Differences in drought detection
- Impact of temperature trends
- Drought area percentage comparison

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    print_stat, setup_output_dirs, load_data, check_input_files,
    compare_arrays, print_comparison, write_report, setup_matplotlib,
    OUTPUT_NETCDF, OUTPUT_PLOTS, OUTPUT_REPORTS,
    CALIBRATION_START, CALIBRATION_END, DATA_START_YEAR,
    SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np
import xarray as xr

# Import functions
from indices import spi, spei, get_drought_area_percentage, save_index_to_netcdf


def calculate_spi_spei_pair(precip, tmean, lat, scale: int) -> tuple:
    """Calculate SPI and SPEI at the same scale for comparison."""
    print_info(f"Calculating SPI-{scale}...")
    spi_result = spi(
        precip=precip,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution='gamma'
    )

    print_info(f"Calculating SPEI-{scale} (Thornthwaite)...")
    spei_result = spei(
        precip=precip,
        temperature=tmean,
        latitude=lat,
        scale=scale,
        periodicity='monthly',
        data_start_year=DATA_START_YEAR,
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution='gamma',
        pet_method='thornthwaite'
    )

    return spi_result, spei_result


def analyze_spi_spei_correlation(spi_da, spei_da, scale: int) -> str:
    """Analyze correlation between SPI and SPEI."""
    print_subheader(f"SPI-{scale} vs SPEI-{scale} Correlation")

    report_lines = [f"\nSPI-{scale} vs SPEI-{scale} Analysis:\n"]

    # Overall correlation
    comp = compare_arrays('SPI', spi_da.values, 'SPEI', spei_da.values)

    print(f"  Overall Correlation: r = {comp['correlation']:.4f}")
    print(f"  RMSE: {comp['rmse']:.4f}")
    print(f"  Bias (SPI - SPEI): {-comp['bias']:.4f}")  # Note: bias is (a-b), we want SPEI-SPI

    report_lines.append(f"Overall Correlation: r={comp['correlation']:.4f}")
    report_lines.append(f"RMSE: {comp['rmse']:.4f}")
    report_lines.append(f"Mean Difference (SPEI - SPI): {-comp['bias']:.4f}")

    # Per-pixel correlation
    print("\n  Spatial correlation analysis:")

    correlations = []
    for i in range(spi_da.shape[1]):
        for j in range(spi_da.shape[2]):
            spi_ts = spi_da[:, i, j].values
            spei_ts = spei_da[:, i, j].values
            valid = ~(np.isnan(spi_ts) | np.isnan(spei_ts))
            if valid.sum() > 30:
                r = np.corrcoef(spi_ts[valid], spei_ts[valid])[0, 1]
                correlations.append(r)

    if correlations:
        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)

        print(f"    Mean pixel correlation: r = {mean_corr:.4f}")
        print(f"    Range: [{min_corr:.4f}, {max_corr:.4f}]")

        report_lines.append(f"\nSpatial Correlation:")
        report_lines.append(f"  Mean pixel correlation: {mean_corr:.4f}")
        report_lines.append(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")

    return "\n".join(report_lines)


def analyze_drought_detection_difference(spi_da, spei_da, scale: int, threshold: float = -1.0) -> str:
    """Compare drought detection between SPI and SPEI."""
    print_subheader(f"Drought Detection Comparison (threshold={threshold})")

    report_lines = [f"\nDrought Detection (threshold={threshold}):\n"]

    # Count drought months
    spi_drought = (spi_da < threshold).sum().values
    spei_drought = (spei_da < threshold).sum().values
    total_valid = (~np.isnan(spi_da.values)).sum()

    spi_pct = spi_drought / total_valid * 100
    spei_pct = spei_drought / total_valid * 100

    print(f"  SPI drought months: {spi_drought} ({spi_pct:.1f}%)")
    print(f"  SPEI drought months: {spei_drought} ({spei_pct:.1f}%)")
    print(f"  Difference: {spei_drought - spi_drought} ({spei_pct - spi_pct:.1f}%)")

    report_lines.append(f"SPI-{scale} drought months: {spi_drought} ({spi_pct:.1f}%)")
    report_lines.append(f"SPEI-{scale} drought months: {spei_drought} ({spei_pct:.1f}%)")
    report_lines.append(f"Difference: {spei_drought - spi_drought} ({spei_pct - spi_pct:.1f}%)")

    # Agreement analysis
    spi_is_drought = spi_da.values < threshold
    spei_is_drought = spei_da.values < threshold
    valid = ~(np.isnan(spi_da.values) | np.isnan(spei_da.values))

    both_drought = (spi_is_drought & spei_is_drought & valid).sum()
    spi_only = (spi_is_drought & ~spei_is_drought & valid).sum()
    spei_only = (~spi_is_drought & spei_is_drought & valid).sum()
    neither = (~spi_is_drought & ~spei_is_drought & valid).sum()

    print(f"\n  Agreement Matrix:")
    print(f"    Both detect drought: {both_drought}")
    print(f"    Only SPI detects: {spi_only}")
    print(f"    Only SPEI detects: {spei_only}")
    print(f"    Neither detects: {neither}")

    agreement_pct = (both_drought + neither) / valid.sum() * 100
    print(f"    Agreement rate: {agreement_pct:.1f}%")

    report_lines.append(f"\nAgreement Matrix:")
    report_lines.append(f"  Both detect: {both_drought}")
    report_lines.append(f"  Only SPI: {spi_only}")
    report_lines.append(f"  Only SPEI: {spei_only}")
    report_lines.append(f"  Agreement rate: {agreement_pct:.1f}%")

    return "\n".join(report_lines)


def analyze_drought_area_timeseries(spi_da, spei_da, scale: int, threshold: float = -1.0) -> str:
    """Compare drought area percentage time series."""
    print_subheader("Drought Area Percentage Comparison")

    report_lines = ["\nDrought Area Percentage:\n"]

    spi_area = get_drought_area_percentage(spi_da, threshold=threshold)
    spei_area = get_drought_area_percentage(spei_da, threshold=threshold)

    # Statistics
    spi_mean = float(spi_area.mean())
    spei_mean = float(spei_area.mean())
    spi_max = float(spi_area.max())
    spei_max = float(spei_area.max())

    print(f"  SPI-{scale}: mean={spi_mean:.1f}%, max={spi_max:.1f}%")
    print(f"  SPEI-{scale}: mean={spei_mean:.1f}%, max={spei_max:.1f}%")
    print(f"  Mean difference: {spei_mean - spi_mean:.1f}%")

    report_lines.append(f"SPI-{scale}: mean={spi_mean:.1f}%, max={spi_max:.1f}%")
    report_lines.append(f"SPEI-{scale}: mean={spei_mean:.1f}%, max={spei_max:.1f}%")
    report_lines.append(f"Mean difference: {spei_mean - spi_mean:.1f}%")

    return "\n".join(report_lines), spi_area, spei_area


def create_spi_spei_comparison_plots(spi_da, spei_da, spi_area, spei_area, scale: int):
    """Create comparison plots for SPI vs SPEI."""
    print_subheader("Creating Comparison Plots")

    plt = setup_matplotlib()

    # Get sample location
    lat_idx = min(SAMPLE_LAT_IDX, spi_da.shape[1] - 1)
    lon_idx = min(SAMPLE_LON_IDX, spi_da.shape[2] - 1)

    # Plot 1: Time series comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # SPI time series
    spi_ts = spi_da[:, lat_idx, lon_idx]
    axes[0].fill_between(spi_da.time, 0, spi_ts.where(spi_ts >= 0), color='blue', alpha=0.5)
    axes[0].fill_between(spi_da.time, 0, spi_ts.where(spi_ts < 0), color='red', alpha=0.5)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axhline(y=-1, color='orange', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel(f'SPI-{scale}')
    axes[0].set_title(f'SPI-{scale} Time Series')
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].grid(True, alpha=0.3)

    # SPEI time series
    spei_ts = spei_da[:, lat_idx, lon_idx]
    axes[1].fill_between(spei_da.time, 0, spei_ts.where(spei_ts >= 0), color='blue', alpha=0.5)
    axes[1].fill_between(spei_da.time, 0, spei_ts.where(spei_ts < 0), color='red', alpha=0.5)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axhline(y=-1, color='orange', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel(f'SPEI-{scale}')
    axes[1].set_title(f'SPEI-{scale} Time Series')
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].grid(True, alpha=0.3)

    # Difference (SPEI - SPI)
    diff_ts = spei_ts - spi_ts
    axes[2].fill_between(spi_da.time, 0, diff_ts.where(diff_ts >= 0), color='green', alpha=0.5, label='SPEI > SPI')
    axes[2].fill_between(spi_da.time, 0, diff_ts.where(diff_ts < 0), color='purple', alpha=0.5, label='SPEI < SPI')
    axes[2].axhline(y=0, color='k', linewidth=0.5)
    axes[2].set_ylabel('SPEI - SPI')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Difference (SPEI - SPI)')
    axes[2].set_ylim(-2, 2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / f'spi_spei_timeseries_{scale}.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")

    # Plot 2: Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    spi_flat = spi_da.values.flatten()
    spei_flat = spei_da.values.flatten()
    valid = ~(np.isnan(spi_flat) | np.isnan(spei_flat))

    ax.scatter(spi_flat[valid], spei_flat[valid], alpha=0.1, s=5, c='steelblue')
    ax.plot([-3, 3], [-3, 3], 'r--', linewidth=1, label='1:1 line')

    # Correlation
    r = np.corrcoef(spi_flat[valid], spei_flat[valid])[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.4f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'SPI-{scale}')
    ax.set_ylabel(f'SPEI-{scale}')
    ax.set_title(f'SPI-{scale} vs SPEI-{scale} Scatter Plot')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / f'spi_spei_scatter_{scale}.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")

    # Plot 3: Drought area comparison
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(spi_area.time, spi_area, label=f'SPI-{scale}', color='steelblue', linewidth=1)
    ax.plot(spei_area.time, spei_area, label=f'SPEI-{scale}', color='darkgreen', linewidth=1)

    ax.axhline(y=float(spi_area.mean()), color='steelblue', linestyle='--', alpha=0.5,
               label=f'SPI mean: {float(spi_area.mean()):.1f}%')
    ax.axhline(y=float(spei_area.mean()), color='darkgreen', linestyle='--', alpha=0.5,
               label=f'SPEI mean: {float(spei_area.mean()):.1f}%')

    ax.set_xlabel('Time')
    ax.set_ylabel('Drought Area (%)')
    ax.set_title(f'Drought Area Percentage: SPI-{scale} vs SPEI-{scale} (threshold < -1.0)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / f'spi_spei_drought_area_{scale}.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def main():
    """Run SPI vs SPEI comparison tests."""
    print_header("TEST 05: SPI VS SPEI COMPARISON")

    setup_output_dirs()

    # Load data
    print_subheader("Loading Input Data")

    status = check_input_files()

    try:
        precip = load_data('precip')
        print_ok(f"Precipitation: {precip.shape}")
    except Exception as e:
        print_fail(f"Could not load precipitation: {e}")
        return False

    if not status['tmean']['exists']:
        print_fail("Temperature data required for SPEI comparison")
        return False

    tmean = load_data('tmean')
    print_ok(f"Temperature: {tmean.shape}")

    lat = precip.lat

    # Calculate SPI and SPEI for each scale
    all_reports = []

    for scale in [3, 12]:
        print_subheader(f"Processing Scale {scale}")

        spi_da, spei_da = calculate_spi_spei_pair(precip, tmean, lat, scale)

        # Analysis
        corr_report = analyze_spi_spei_correlation(spi_da, spei_da, scale)
        all_reports.append(corr_report)

        detection_report = analyze_drought_detection_difference(spi_da, spei_da, scale)
        all_reports.append(detection_report)

        area_report, spi_area, spei_area = analyze_drought_area_timeseries(spi_da, spei_da, scale)
        all_reports.append(area_report)

        # Create plots
        create_spi_spei_comparison_plots(spi_da, spei_da, spi_area, spei_area, scale)

    # Write report
    full_report = "\n".join(all_reports)
    write_report("05_spi_spei_comparison_report.txt", "SPI vs SPEI Comparison Report", full_report)

    # Summary
    print_header("TEST 05 COMPLETE")
    print_ok("SPI vs SPEI comparison complete")
    print_ok(f"Plots saved to: {OUTPUT_PLOTS}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
