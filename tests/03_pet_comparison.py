#!/usr/bin/env python
"""
Test 03: PET Method Comparison

This test compares different PET calculation methods:
- TerraClimate pre-computed PET (Penman-Monteith reference)
- Thornthwaite method (temperature-based)
- Hargreaves-Samani method (requires Tmin/Tmax)

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    print_stat, setup_output_dirs, load_data, check_input_files,
    compare_arrays, print_comparison, write_report, setup_matplotlib,
    OUTPUT_NETCDF, OUTPUT_PLOTS, DATA_START_YEAR,
    SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np
import xarray as xr

# Import PET functions
from utils import calculate_pet


def load_pet_input_data() -> dict:
    """Load all data needed for PET comparison."""
    print_subheader("Loading Input Data")

    data = {}
    status = check_input_files()

    # Required for all methods
    if status['tmean']['exists']:
        data['tmean'] = load_data('tmean')
        print_ok(f"Temperature (mean): {data['tmean'].shape}")

    # Required for Hargreaves
    if status['tmin']['exists']:
        data['tmin'] = load_data('tmin')
        print_ok(f"Temperature (min): {data['tmin'].shape}")

    if status['tmax']['exists']:
        data['tmax'] = load_data('tmax')
        print_ok(f"Temperature (max): {data['tmax'].shape}")

    # Reference PET (TerraClimate Penman-Monteith)
    if status['pet']['exists']:
        data['pet_reference'] = load_data('pet')
        print_ok(f"PET reference (TerraClimate): {data['pet_reference'].shape}")

    return data


def calculate_thornthwaite_pet(data: dict) -> xr.DataArray:
    """Calculate PET using Thornthwaite method."""
    print_subheader("Thornthwaite PET Calculation")

    if 'tmean' not in data:
        print_fail("Temperature data not available")
        return None

    tmean = data['tmean']
    lat = tmean.lat

    print_info("Calculating PET using Thornthwaite method...")
    pet_thornthwaite = calculate_pet(
        tmean,
        latitude=lat,
        data_start_year=DATA_START_YEAR,
        method='thornthwaite'
    )

    print_ok(f"Thornthwaite PET calculated: {pet_thornthwaite.shape}")
    print_stat("Mean", float(pet_thornthwaite.mean()), "mm/month")
    print_stat("Range", f"[{float(pet_thornthwaite.min()):.1f}, {float(pet_thornthwaite.max()):.1f}]", "mm/month")

    return pet_thornthwaite


def calculate_hargreaves_pet(data: dict) -> xr.DataArray:
    """Calculate PET using Hargreaves-Samani method."""
    print_subheader("Hargreaves-Samani PET Calculation")

    required = ['tmean', 'tmin', 'tmax']
    missing = [k for k in required if k not in data]

    if missing:
        print_info(f"Hargreaves not available (missing: {', '.join(missing)})")
        return None

    tmean = data['tmean']
    tmin = data['tmin']
    tmax = data['tmax']
    lat = tmean.lat

    print_info("Calculating PET using Hargreaves-Samani method...")
    pet_hargreaves = calculate_pet(
        tmean,
        latitude=lat,
        data_start_year=DATA_START_YEAR,
        method='hargreaves',
        temp_min=tmin,
        temp_max=tmax
    )

    print_ok(f"Hargreaves PET calculated: {pet_hargreaves.shape}")
    print_stat("Mean", float(pet_hargreaves.mean()), "mm/month")
    print_stat("Range", f"[{float(pet_hargreaves.min()):.1f}, {float(pet_hargreaves.max()):.1f}]", "mm/month")

    return pet_hargreaves


def compare_pet_methods(pet_reference, pet_thornthwaite, pet_hargreaves) -> str:
    """Compare PET values across methods."""
    print_subheader("PET Method Comparison")

    report_lines = ["PET Method Comparison:\n"]

    methods = {}
    if pet_reference is not None:
        methods['TerraClimate'] = pet_reference.values
    if pet_thornthwaite is not None:
        methods['Thornthwaite'] = pet_thornthwaite.values
    if pet_hargreaves is not None:
        methods['Hargreaves'] = pet_hargreaves.values

    if len(methods) < 2:
        print_info("Not enough PET methods for comparison")
        return "Insufficient PET methods for comparison\n"

    # Summary statistics
    print("\n  Summary Statistics:")
    report_lines.append("\nSummary Statistics:")

    for name, values in methods.items():
        valid = values[~np.isnan(values)]
        mean = float(np.mean(valid))
        std = float(np.std(valid))
        print(f"    {name}: mean={mean:.1f}, std={std:.1f} mm/month")
        report_lines.append(f"  {name}: mean={mean:.1f}, std={std:.1f} mm/month")

    # Pairwise comparisons
    print("\n  Correlation with TerraClimate (Penman-Monteith):")
    report_lines.append("\nCorrelation with TerraClimate:")

    if 'TerraClimate' in methods:
        ref = methods['TerraClimate']

        for name, values in methods.items():
            if name == 'TerraClimate':
                continue

            comp = compare_arrays('TerraClimate', ref, name, values)
            print(f"    {name}: r = {comp['correlation']:.4f}, RMSE = {comp['rmse']:.2f}, Bias = {comp['bias']:.2f}")
            report_lines.append(f"  {name}: r={comp['correlation']:.4f}, RMSE={comp['rmse']:.2f}, Bias={comp['bias']:.2f}")

            # Interpretation
            if comp['correlation'] > 0.8:
                print_ok(f"{name} has good correlation with reference")
            elif comp['correlation'] > 0.6:
                print_info(f"{name} has moderate correlation with reference")
            else:
                print_info(f"{name} has low correlation with reference")

    # Compare Thornthwaite vs Hargreaves directly
    if 'Thornthwaite' in methods and 'Hargreaves' in methods:
        comp = compare_arrays('Thornthwaite', methods['Thornthwaite'],
                             'Hargreaves', methods['Hargreaves'])
        print(f"\n  Thornthwaite vs Hargreaves: r = {comp['correlation']:.4f}")
        report_lines.append(f"\nThornthwaite vs Hargreaves: r={comp['correlation']:.4f}")

    return "\n".join(report_lines)


def create_pet_comparison_plots(pet_reference, pet_thornthwaite, pet_hargreaves):
    """Create comparison plots for PET methods."""
    print_subheader("Creating PET Comparison Plots")

    plt = setup_matplotlib()

    methods = {}
    if pet_reference is not None:
        methods['TerraClimate (PM)'] = pet_reference
    if pet_thornthwaite is not None:
        methods['Thornthwaite'] = pet_thornthwaite
    if pet_hargreaves is not None:
        methods['Hargreaves'] = pet_hargreaves

    if len(methods) < 2:
        print_info("Not enough PET methods for plotting")
        return

    # Get sample location
    lat_idx = min(SAMPLE_LAT_IDX, methods[list(methods.keys())[0]].shape[1] - 1)
    lon_idx = min(SAMPLE_LON_IDX, methods[list(methods.keys())[0]].shape[2] - 1)

    # Plot 1: Time series comparison (last 10 years)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    n_months = 120  # Last 10 years
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    ax1 = axes[0]
    for i, (name, da) in enumerate(methods.items()):
        ts = da[-n_months:, lat_idx, lon_idx].values
        ax1.plot(range(n_months), ts, label=name, color=colors[i % len(colors)], alpha=0.8)

    ax1.set_xlabel('Months (last 10 years)')
    ax1.set_ylabel('PET (mm/month)')
    ax1.set_title('PET Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter comparison with reference
    ax2 = axes[1]

    if 'TerraClimate (PM)' in methods:
        ref_flat = methods['TerraClimate (PM)'].values.flatten()

        for i, (name, da) in enumerate(methods.items()):
            if name == 'TerraClimate (PM)':
                continue

            calc_flat = da.values.flatten()
            valid = ~(np.isnan(ref_flat) | np.isnan(calc_flat))

            ax2.scatter(ref_flat[valid], calc_flat[valid],
                       alpha=0.1, s=5, label=name, color=colors[(i+1) % len(colors)])

        # 1:1 line
        max_val = float(np.nanmax(ref_flat))
        ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')

        ax2.set_xlabel('TerraClimate PET (mm/month)')
        ax2.set_ylabel('Calculated PET (mm/month)')
        ax2.set_title('PET Method Comparison vs TerraClimate Reference')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'pet_comparison_timeseries_scatter.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")

    # Plot 3: Spatial mean comparison
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for i, (name, da) in enumerate(methods.items()):
        mean_pet = da.mean(dim='time')
        im = axes[i].pcolormesh(da.lon, da.lat, mean_pet, cmap='YlOrRd', vmin=50, vmax=200)
        axes[i].set_title(f'{name}\n(Mean PET)')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        plt.colorbar(im, ax=axes[i], label='mm/month')

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'pet_comparison_spatial.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def save_pet_outputs(pet_thornthwaite, pet_hargreaves):
    """Save calculated PET to NetCDF files."""
    print_subheader("Saving PET Outputs")

    if pet_thornthwaite is not None:
        filepath = OUTPUT_NETCDF / 'pet_thornthwaite.nc'
        pet_thornthwaite.to_netcdf(filepath)
        print_ok(f"Saved: {filepath.name}")

    if pet_hargreaves is not None:
        filepath = OUTPUT_NETCDF / 'pet_hargreaves.nc'
        pet_hargreaves.to_netcdf(filepath)
        print_ok(f"Saved: {filepath.name}")


def main():
    """Run all PET comparison tests."""
    print_header("TEST 03: PET METHOD COMPARISON")

    setup_output_dirs()

    # Load input data
    data = load_pet_input_data()

    if 'tmean' not in data:
        print_fail("Temperature data required for PET calculation")
        return False

    # Calculate PET with different methods
    pet_reference = data.get('pet_reference')
    pet_thornthwaite = calculate_thornthwaite_pet(data)
    pet_hargreaves = calculate_hargreaves_pet(data)

    # Compare methods
    comparison_report = compare_pet_methods(pet_reference, pet_thornthwaite, pet_hargreaves)

    # Create plots
    create_pet_comparison_plots(pet_reference, pet_thornthwaite, pet_hargreaves)

    # Save outputs
    save_pet_outputs(pet_thornthwaite, pet_hargreaves)

    # Write report
    write_report("03_pet_comparison_report.txt", "PET Method Comparison Report", comparison_report)

    # Summary
    print_header("TEST 03 COMPLETE")

    methods_available = []
    if pet_reference is not None:
        methods_available.append("TerraClimate")
    if pet_thornthwaite is not None:
        methods_available.append("Thornthwaite")
    if pet_hargreaves is not None:
        methods_available.append("Hargreaves")

    print_ok(f"Compared {len(methods_available)} PET methods: {', '.join(methods_available)}")
    print_ok(f"Plots saved to: {OUTPUT_PLOTS}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
