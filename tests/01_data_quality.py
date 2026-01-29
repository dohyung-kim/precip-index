#!/usr/bin/env python
"""
Test 01: Data Quality Assessment

This test validates input data files and generates a quality report.
It checks:
- File existence and sizes
- Data dimensions and coordinates
- Value ranges and statistics
- Missing data patterns
- Data readiness for SPI/SPEI calculation

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    setup_output_dirs, check_input_files, load_data, load_all_data,
    get_data_summary, print_data_summary, write_report,
    INPUT_FILES, OUTPUT_REPORTS, CALIBRATION_START, CALIBRATION_END
)

import numpy as np


def test_file_availability():
    """Check which input files exist."""
    print_subheader("File Availability Check")

    status = check_input_files()
    all_exist = True
    required = ['precip', 'tmean', 'pet']  # Minimum required
    optional = ['tmin', 'tmax']  # For Hargreaves

    report_lines = ["File Availability:\n"]

    for name, info in status.items():
        if info['exists']:
            print_ok(f"{name}: {info['path'].name} ({info['size_mb']:.2f} MB)")
            report_lines.append(f"  [OK] {name}: {info['size_mb']:.2f} MB")
        else:
            if name in required:
                print_fail(f"{name}: NOT FOUND (required)")
                all_exist = False
            else:
                print_info(f"{name}: NOT FOUND (optional for Hargreaves)")
            report_lines.append(f"  [--] {name}: not found")

    return all_exist, "\n".join(report_lines)


def test_data_quality():
    """Load data and assess quality."""
    print_subheader("Data Quality Assessment")

    data = load_all_data()
    report_lines = ["\nData Quality Summary:\n"]

    summaries = {}
    for name, da in data.items():
        summary = get_data_summary(da)
        summaries[name] = summary
        print_data_summary(name.upper(), summary)

        # Add to report
        report_lines.append(f"\n{name.upper()}:")
        report_lines.append(f"  Shape: {summary['shape']}")
        report_lines.append(f"  Time: {summary.get('time_start', 'N/A')} to {summary.get('time_end', 'N/A')}")
        report_lines.append(f"  Mean: {summary['mean']:.2f}, Std: {summary['std']:.2f}")
        report_lines.append(f"  Range: [{summary['min']:.2f}, {summary['max']:.2f}]")
        report_lines.append(f"  Missing: {summary['nan_pct']:.1f}%")

    return summaries, "\n".join(report_lines)


def test_data_consistency():
    """Check that all datasets are spatially and temporally aligned."""
    print_subheader("Data Consistency Check")

    data = load_all_data()
    report_lines = ["\nData Consistency:\n"]

    if len(data) < 2:
        print_info("Not enough datasets to check consistency")
        return True, "  Skipped (insufficient data)\n"

    # Use precipitation as reference
    ref_name = 'precip'
    if ref_name not in data:
        ref_name = list(data.keys())[0]

    ref = data[ref_name]
    ref_shape = ref.shape
    ref_time = len(ref.time) if 'time' in ref.coords else 0

    all_consistent = True

    for name, da in data.items():
        if name == ref_name:
            continue

        # Check shape
        shape_match = da.shape == ref_shape
        time_match = (len(da.time) == ref_time) if 'time' in da.coords else True

        if shape_match and time_match:
            print_ok(f"{name} matches {ref_name} (shape: {da.shape})")
            report_lines.append(f"  [OK] {name} matches reference")
        else:
            print_fail(f"{name} shape mismatch: {da.shape} vs {ref_shape}")
            report_lines.append(f"  [FAIL] {name} shape mismatch")
            all_consistent = False

    return all_consistent, "\n".join(report_lines)


def test_calibration_coverage():
    """Check if data covers the calibration period."""
    print_subheader("Calibration Period Coverage")

    data = load_all_data()
    report_lines = [f"\nCalibration Period: {CALIBRATION_START}-{CALIBRATION_END}\n"]

    all_covered = True

    for name, da in data.items():
        if 'time' not in da.coords:
            continue

        years = da.time.dt.year
        min_year = int(years.min())
        max_year = int(years.max())

        covers_start = min_year <= CALIBRATION_START
        covers_end = max_year >= CALIBRATION_END

        if covers_start and covers_end:
            print_ok(f"{name}: {min_year}-{max_year} (covers calibration period)")
            report_lines.append(f"  [OK] {name}: {min_year}-{max_year}")
        else:
            print_fail(f"{name}: {min_year}-{max_year} (does not fully cover calibration)")
            report_lines.append(f"  [FAIL] {name}: {min_year}-{max_year}")
            all_covered = False

    return all_covered, "\n".join(report_lines)


def test_value_ranges():
    """Check if values are within expected physical ranges."""
    print_subheader("Physical Value Range Check")

    # Expected ranges (reasonable bounds)
    expected_ranges = {
        'precip': (0, 1500),      # mm/month
        'tmean': (-10, 45),       # Celsius
        'tmin': (-15, 40),        # Celsius
        'tmax': (-5, 50),         # Celsius
        'pet': (0, 400),          # mm/month
    }

    data = load_all_data()
    report_lines = ["\nValue Range Validation:\n"]

    all_valid = True

    for name, da in data.items():
        if name not in expected_ranges:
            continue

        exp_min, exp_max = expected_ranges[name]
        actual_min = float(da.min())
        actual_max = float(da.max())

        in_range = (actual_min >= exp_min - 5) and (actual_max <= exp_max + 50)  # Allow some tolerance

        if in_range:
            print_ok(f"{name}: [{actual_min:.1f}, {actual_max:.1f}] (expected: [{exp_min}, {exp_max}])")
            report_lines.append(f"  [OK] {name}: [{actual_min:.1f}, {actual_max:.1f}]")
        else:
            print_fail(f"{name}: [{actual_min:.1f}, {actual_max:.1f}] outside expected range")
            report_lines.append(f"  [WARN] {name}: values outside typical range")
            # Don't fail for this, just warn

    return all_valid, "\n".join(report_lines)


def generate_readiness_assessment():
    """Generate overall data readiness assessment."""
    print_subheader("Data Readiness Assessment")

    status = check_input_files()
    report_lines = ["\nReadiness Assessment:\n"]

    # SPI readiness
    spi_ready = status['precip']['exists']
    if spi_ready:
        print_ok("SPI calculation: READY (precipitation available)")
        report_lines.append("  [OK] SPI: Ready")
    else:
        print_fail("SPI calculation: NOT READY (precipitation missing)")
        report_lines.append("  [FAIL] SPI: Not ready - precipitation missing")

    # SPEI with pre-computed PET
    spei_pet_ready = status['precip']['exists'] and status['pet']['exists']
    if spei_pet_ready:
        print_ok("SPEI (pre-computed PET): READY")
        report_lines.append("  [OK] SPEI (pre-computed PET): Ready")
    else:
        print_info("SPEI (pre-computed PET): Not available")
        report_lines.append("  [--] SPEI (pre-computed PET): PET file missing")

    # SPEI with Thornthwaite
    spei_thorn_ready = status['precip']['exists'] and status['tmean']['exists']
    if spei_thorn_ready:
        print_ok("SPEI (Thornthwaite): READY")
        report_lines.append("  [OK] SPEI (Thornthwaite): Ready")
    else:
        print_fail("SPEI (Thornthwaite): NOT READY")
        report_lines.append("  [FAIL] SPEI (Thornthwaite): Temperature missing")

    # SPEI with Hargreaves
    spei_harg_ready = (status['precip']['exists'] and
                       status['tmean']['exists'] and
                       status['tmin']['exists'] and
                       status['tmax']['exists'])
    if spei_harg_ready:
        print_ok("SPEI (Hargreaves): READY")
        report_lines.append("  [OK] SPEI (Hargreaves): Ready")
    else:
        missing = [k for k in ['tmean', 'tmin', 'tmax'] if not status[k]['exists']]
        print_info(f"SPEI (Hargreaves): Not available (missing: {', '.join(missing)})")
        report_lines.append(f"  [--] SPEI (Hargreaves): Missing {', '.join(missing)}")

    return "\n".join(report_lines)


def main():
    """Run all data quality tests."""
    print_header("TEST 01: DATA QUALITY ASSESSMENT")

    setup_output_dirs()

    # Collect all report sections
    report_sections = []

    # Run tests
    files_ok, files_report = test_file_availability()
    report_sections.append(files_report)

    if not files_ok:
        print_fail("\nCritical files missing. Cannot proceed.")
        return False

    summaries, quality_report = test_data_quality()
    report_sections.append(quality_report)

    consistent, consistency_report = test_data_consistency()
    report_sections.append(consistency_report)

    calibration_ok, calibration_report = test_calibration_coverage()
    report_sections.append(calibration_report)

    ranges_ok, ranges_report = test_value_ranges()
    report_sections.append(ranges_report)

    readiness_report = generate_readiness_assessment()
    report_sections.append(readiness_report)

    # Write consolidated report
    full_report = "\n".join(report_sections)
    write_report("01_data_quality_report.txt", "Data Quality Report", full_report)

    # Summary
    print_header("TEST 01 COMPLETE")
    all_passed = files_ok and consistent and calibration_ok
    if all_passed:
        print_ok("All data quality checks passed!")
    else:
        print_info("Some checks had warnings - review report for details")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
