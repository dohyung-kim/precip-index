#!/usr/bin/env python
"""
Test 04: SPEI Calculation

This test calculates SPEI using different PET methods and distributions:
- SPEI with pre-computed PET (TerraClimate)
- SPEI with Thornthwaite PET
- SPEI with Hargreaves PET
- Multiple distributions (Gamma, Pearson III, Log-Logistic)
- Multiple scales (SPEI-3, SPEI-12)

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    print_stat, setup_output_dirs, load_data, check_input_files,
    compare_arrays, write_report,
    OUTPUT_NETCDF, CALIBRATION_START, CALIBRATION_END, DATA_START_YEAR,
    TEST_DISTRIBUTIONS, TEST_SCALES, SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np

# Import SPEI functions
from indices import spei, save_index_to_netcdf
from config import DISTRIBUTION_DISPLAY_NAMES


def calculate_spei_with_pet(precip, pet, scale: int, distribution: str) -> dict:
    """Calculate SPEI using pre-computed PET."""
    print_info(f"SPEI-{scale} with PET ({DISTRIBUTION_DISPLAY_NAMES[distribution]})...")

    spei_result = spei(
        precip=precip,
        pet=pet,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution
    )

    values = spei_result.values.flatten()
    valid = values[~np.isnan(values)]

    stats = {
        'method': 'PET',
        'scale': scale,
        'distribution': distribution,
        'data': spei_result,
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
    }

    print_ok(f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    return stats


def calculate_spei_with_thornthwaite(precip, tmean, lat, scale: int, distribution: str) -> dict:
    """Calculate SPEI using Thornthwaite PET method."""
    print_info(f"SPEI-{scale} with Thornthwaite ({DISTRIBUTION_DISPLAY_NAMES[distribution]})...")

    spei_result = spei(
        precip=precip,
        temperature=tmean,
        latitude=lat,
        scale=scale,
        periodicity='monthly',
        data_start_year=DATA_START_YEAR,
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        pet_method='thornthwaite'
    )

    values = spei_result.values.flatten()
    valid = values[~np.isnan(values)]

    stats = {
        'method': 'Thornthwaite',
        'scale': scale,
        'distribution': distribution,
        'data': spei_result,
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
    }

    print_ok(f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    return stats


def calculate_spei_with_hargreaves(precip, tmean, tmin, tmax, lat, scale: int, distribution: str) -> dict:
    """Calculate SPEI using Hargreaves PET method."""
    print_info(f"SPEI-{scale} with Hargreaves ({DISTRIBUTION_DISPLAY_NAMES[distribution]})...")

    spei_result = spei(
        precip=precip,
        temperature=tmean,
        latitude=lat,
        scale=scale,
        periodicity='monthly',
        data_start_year=DATA_START_YEAR,
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        pet_method='hargreaves',
        temp_min=tmin,
        temp_max=tmax
    )

    values = spei_result.values.flatten()
    valid = values[~np.isnan(values)]

    stats = {
        'method': 'Hargreaves',
        'scale': scale,
        'distribution': distribution,
        'data': spei_result,
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
    }

    print_ok(f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    return stats


def test_spei_all_combinations(data: dict) -> dict:
    """Calculate SPEI for all method/scale/distribution combinations."""
    results = {}

    precip = data['precip']
    tmean = data.get('tmean')
    tmin = data.get('tmin')
    tmax = data.get('tmax')
    pet = data.get('pet')
    lat = precip.lat

    # Test with pre-computed PET
    if pet is not None:
        print_subheader("SPEI with Pre-computed PET")
        for scale in TEST_SCALES:
            for dist in TEST_DISTRIBUTIONS:
                key = f"spei_{scale}_pet_{dist}"
                results[key] = calculate_spei_with_pet(precip, pet, scale, dist)

    # Test with Thornthwaite
    if tmean is not None:
        print_subheader("SPEI with Thornthwaite")
        for scale in TEST_SCALES:
            for dist in TEST_DISTRIBUTIONS:
                key = f"spei_{scale}_thornthwaite_{dist}"
                results[key] = calculate_spei_with_thornthwaite(precip, tmean, lat, scale, dist)

    # Test with Hargreaves
    if tmean is not None and tmin is not None and tmax is not None:
        print_subheader("SPEI with Hargreaves")
        for scale in TEST_SCALES:
            for dist in TEST_DISTRIBUTIONS:
                key = f"spei_{scale}_hargreaves_{dist}"
                results[key] = calculate_spei_with_hargreaves(precip, tmean, tmin, tmax, lat, scale, dist)

    return results


def analyze_spei_results(results: dict) -> str:
    """Analyze SPEI results and generate report."""
    print_subheader("SPEI Results Analysis")

    report_lines = ["SPEI Calculation Summary:\n"]

    # Group by scale
    for scale in TEST_SCALES:
        print(f"\n  SPEI-{scale} Summary:")
        report_lines.append(f"\nSPEI-{scale}:")

        scale_results = {k: v for k, v in results.items() if f"spei_{scale}_" in k}

        for key, stats in scale_results.items():
            method = stats['method']
            dist = DISTRIBUTION_DISPLAY_NAMES[stats['distribution']]
            print(f"    {method} ({dist}): mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            report_lines.append(f"  {method} ({dist}): mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    return "\n".join(report_lines)


def compare_pet_methods_in_spei(results: dict) -> str:
    """Compare SPEI calculated with different PET methods."""
    print_subheader("PET Method Comparison in SPEI")

    report_lines = ["\nPET Method Comparison in SPEI:\n"]

    for scale in TEST_SCALES:
        print(f"\n  SPEI-{scale} PET method correlations:")
        report_lines.append(f"\nSPEI-{scale}:")

        # Get results for this scale with gamma distribution
        pet_key = f"spei_{scale}_pet_gamma"
        thorn_key = f"spei_{scale}_thornthwaite_gamma"
        harg_key = f"spei_{scale}_hargreaves_gamma"

        comparisons = []

        if pet_key in results and thorn_key in results:
            comp = compare_arrays(
                'PET', results[pet_key]['data'].values,
                'Thornthwaite', results[thorn_key]['data'].values
            )
            comparisons.append(('PET vs Thornthwaite', comp))

        if pet_key in results and harg_key in results:
            comp = compare_arrays(
                'PET', results[pet_key]['data'].values,
                'Hargreaves', results[harg_key]['data'].values
            )
            comparisons.append(('PET vs Hargreaves', comp))

        if thorn_key in results and harg_key in results:
            comp = compare_arrays(
                'Thornthwaite', results[thorn_key]['data'].values,
                'Hargreaves', results[harg_key]['data'].values
            )
            comparisons.append(('Thornthwaite vs Hargreaves', comp))

        for name, comp in comparisons:
            print(f"    {name}: r = {comp['correlation']:.4f}")
            report_lines.append(f"  {name}: r={comp['correlation']:.4f}")

    return "\n".join(report_lines)


def save_spei_outputs(results: dict):
    """Save SPEI results to NetCDF files."""
    print_subheader("Saving SPEI Outputs")

    for key, stats in results.items():
        filename = f"{key}.nc"
        filepath = OUTPUT_NETCDF / filename

        save_index_to_netcdf(stats['data'], str(filepath), compress=True)
        print_ok(f"Saved: {filename}")


def main():
    """Run all SPEI calculation tests."""
    print_header("TEST 04: SPEI CALCULATION")

    setup_output_dirs()

    # Load all required data
    print_subheader("Loading Input Data")

    data = {}
    status = check_input_files()

    try:
        data['precip'] = load_data('precip')
        print_ok(f"Precipitation: {data['precip'].shape}")
    except Exception as e:
        print_fail(f"Could not load precipitation: {e}")
        return False

    if status['tmean']['exists']:
        data['tmean'] = load_data('tmean')
        print_ok(f"Temperature (mean): {data['tmean'].shape}")

    if status['tmin']['exists']:
        data['tmin'] = load_data('tmin')
        print_ok(f"Temperature (min): {data['tmin'].shape}")

    if status['tmax']['exists']:
        data['tmax'] = load_data('tmax')
        print_ok(f"Temperature (max): {data['tmax'].shape}")

    if status['pet']['exists']:
        data['pet'] = load_data('pet')
        print_ok(f"PET (pre-computed): {data['pet'].shape}")

    # Calculate SPEI for all combinations
    results = test_spei_all_combinations(data)

    if not results:
        print_fail("No SPEI results calculated (missing temperature or PET data)")
        return False

    # Analyze results
    analysis_report = analyze_spei_results(results)

    # Compare PET methods
    comparison_report = compare_pet_methods_in_spei(results)

    # Save outputs
    save_spei_outputs(results)

    # Write report
    full_report = analysis_report + "\n" + comparison_report
    write_report("04_spei_calculation_report.txt", "SPEI Calculation Report", full_report)

    # Summary
    print_header("TEST 04 COMPLETE")
    print_ok(f"Calculated {len(results)} SPEI variants")
    print_ok(f"Outputs saved to: {OUTPUT_NETCDF}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
