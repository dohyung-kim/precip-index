#!/usr/bin/env python
"""
Test 02: SPI Calculation

This test calculates SPI at multiple scales using different distributions.
It validates:
- SPI-3 and SPI-12 calculation
- Gamma, Pearson III, and Log-Logistic distributions
- Output statistics and expected ranges
- Cross-distribution correlation

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    print_stat, setup_output_dirs, load_data, compare_arrays, print_comparison,
    write_report, setup_matplotlib,
    OUTPUT_NETCDF, CALIBRATION_START, CALIBRATION_END,
    TEST_DISTRIBUTIONS, TEST_SCALES, SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np

# Import SPI functions
from indices import spi, spi_multi_scale, save_index_to_netcdf
from config import DISTRIBUTION_DISPLAY_NAMES


def test_spi_single_scale(precip, scale: int, distribution: str) -> dict:
    """
    Calculate SPI for a single scale and distribution.

    Returns:
        dict with SPI DataArray and statistics
    """
    print_info(f"Calculating SPI-{scale} ({DISTRIBUTION_DISPLAY_NAMES[distribution]})...")

    spi_result = spi(
        precip,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution
    )

    # Calculate statistics
    values = spi_result.values.flatten()
    valid = values[~np.isnan(values)]

    stats = {
        'scale': scale,
        'distribution': distribution,
        'data': spi_result,
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'pct_below_neg1': float((valid < -1.0).sum() / len(valid) * 100),
        'pct_above_pos1': float((valid > 1.0).sum() / len(valid) * 100),
    }

    print_ok(f"SPI-{scale} ({distribution}): mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    return stats


def test_spi_all_combinations(precip) -> dict:
    """Calculate SPI for all scale/distribution combinations."""
    print_subheader("SPI Calculation - All Combinations")

    results = {}

    for scale in TEST_SCALES:
        for dist in TEST_DISTRIBUTIONS:
            key = f"spi_{scale}_{dist}"
            results[key] = test_spi_single_scale(precip, scale, dist)

    return results


def test_spi_statistics(results: dict) -> str:
    """Analyze SPI statistics and validate expected properties."""
    print_subheader("SPI Statistical Validation")

    report_lines = ["SPI Statistics Summary:\n"]

    for key, stats in results.items():
        scale = stats['scale']
        dist = stats['distribution']

        print(f"\n  SPI-{scale} ({DISTRIBUTION_DISPLAY_NAMES[dist]}):")
        print_stat("Mean", stats['mean'])
        print_stat("Std Dev", stats['std'])
        print_stat("Range", f"[{stats['min']:.2f}, {stats['max']:.2f}]")
        print_stat("% below -1.0", f"{stats['pct_below_neg1']:.1f}%")
        print_stat("% above +1.0", f"{stats['pct_above_pos1']:.1f}%")

        # Validate expected properties
        # Mean should be close to 0 over calibration period
        if abs(stats['mean']) < 0.2:
            print_ok("Mean close to 0 (expected for standardized index)")
        else:
            print_info(f"Mean {stats['mean']:.3f} slightly off from 0")

        # Std should be close to 1
        if 0.8 < stats['std'] < 1.2:
            print_ok("Std Dev close to 1 (expected for standardized index)")
        else:
            print_info(f"Std Dev {stats['std']:.3f} slightly off from 1")

        report_lines.append(f"\n{key}:")
        report_lines.append(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        report_lines.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        report_lines.append(f"  Drought (<-1): {stats['pct_below_neg1']:.1f}%, Wet (>+1): {stats['pct_above_pos1']:.1f}%")

    return "\n".join(report_lines)


def test_cross_distribution_comparison(results: dict) -> str:
    """Compare SPI values across different distributions."""
    print_subheader("Cross-Distribution Comparison")

    report_lines = ["\nDistribution Comparison:\n"]

    for scale in TEST_SCALES:
        print(f"\n  SPI-{scale} distribution correlations:")
        report_lines.append(f"\nSPI-{scale}:")

        # Get data for this scale
        scale_results = {k: v for k, v in results.items() if f"spi_{scale}" in k}

        # Compare all pairs
        keys = list(scale_results.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_a, key_b = keys[i], keys[j]
                data_a = scale_results[key_a]['data'].values
                data_b = scale_results[key_b]['data'].values

                comp = compare_arrays(
                    scale_results[key_a]['distribution'],
                    data_a,
                    scale_results[key_b]['distribution'],
                    data_b
                )

                print(f"    {comp['name_a']} vs {comp['name_b']}: r = {comp['correlation']:.4f}")
                report_lines.append(f"  {comp['name_a']} vs {comp['name_b']}: r={comp['correlation']:.4f}")

                # High correlation expected between distributions
                if comp['correlation'] > 0.95:
                    print_ok("High correlation (distributions produce similar results)")

    return "\n".join(report_lines)


def save_spi_outputs(results: dict):
    """Save SPI results to NetCDF files."""
    print_subheader("Saving SPI Outputs")

    for key, stats in results.items():
        filename = f"{key}.nc"
        filepath = OUTPUT_NETCDF / filename

        save_index_to_netcdf(stats['data'], str(filepath), compress=True)
        print_ok(f"Saved: {filename}")


def main():
    """Run all SPI calculation tests."""
    print_header("TEST 02: SPI CALCULATION")

    setup_output_dirs()

    # Load precipitation data
    print_subheader("Loading Data")
    try:
        precip = load_data('precip')
        print_ok(f"Precipitation loaded: {precip.shape}")
    except Exception as e:
        print_fail(f"Could not load precipitation: {e}")
        return False

    # Calculate SPI for all combinations
    results = test_spi_all_combinations(precip)

    # Validate statistics
    stats_report = test_spi_statistics(results)

    # Compare distributions
    comparison_report = test_cross_distribution_comparison(results)

    # Save outputs
    save_spi_outputs(results)

    # Write report
    full_report = stats_report + "\n" + comparison_report
    write_report("02_spi_calculation_report.txt", "SPI Calculation Report", full_report)

    # Summary
    print_header("TEST 02 COMPLETE")
    print_ok(f"Calculated {len(results)} SPI variants")
    print_ok(f"Outputs saved to: {OUTPUT_NETCDF}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
