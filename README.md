# Precipitation Index - SPI & SPEI for Climate Extremes Monitoring

<img src="./docs/images/logo-white-background.jpg" alt="PRECIP-INDEX Logo" width="300" height="300">
</br>

**precip-index** is a lightweight set of Python scripts for calculating precipitation-based climate indices (SPI and SPEI) and analyzing **dry and wet extremes** using **run theory**, designed for gridded `xarray` workflows.  

📚 Documentation: https://bennyistanto.github.io/precip-index/

## Key Features

- **SPI / SPEI** at 1, 3, 6, 12, 24-month scales (xarray + CF-compliant NetCDF outputs)
- **Bidirectional extremes**: drought (dry) and flood-prone (wet) conditions in one framework
- **Multi-distribution fitting**: Gamma, Pearson Type III, Log-Logistic
- **Run theory events**: duration, magnitude, intensity, peak, interarrival + gridded summaries
- **Operational mode**: save fitted parameters, load and apply to new data without refitting
- **Scalable processing**: chunked tiling, memory estimation, streaming I/O for global datasets
- **Visualization**: event-highlighted time series, 11-category WMO classification, spatial maps

## Why precip-index?

- **Dry + wet symmetry**: same API and methodology for negative (drought) and positive (wet) thresholds
- **Distribution-aware SPI/SPEI**: choose the best-fit distribution per workflow (Gamma / P-III / Log-Logistic)
- **Production-ready monitoring**: calibrate once, save parameters, apply consistently to new observations
- **Event analytics included**: run theory metrics beyond simple threshold exceedance
- **Designed for large grids**: practical for CHIRPS / ERA5-Land / TerraClimate via chunked processing

## Global Output

SPI-12 (Gamma) calculated from **CHIRPS v3** at 0.05° resolution.

![SPI-12 computed from global CHIRPS v3 dataset (0.05°) — December 2025](./docs/images/global-spi12-202512.png)

SPEI-12 (Pearson III) calculated from **TerraClimate** at 0.0417° ~ 4km resolution.

![SPEI-12 computed from global TerraClimate dataset (0.0417° ~ 4km) — December 2024](./docs/images/global-spei12-202412.png)

## Credits

**Benny Istanto**, GOST/DEC Data Group, The World Bank

Built upon the foundation of [climate-indices](https://github.com/monocongo/climate_indices) by James Adams, with substantial additions for multi-distribution support, bidirectional event analysis, operational mode (parameter persistence), and scalable processing.

## License

BSD-3-Clause — see [LICENSE](LICENSE) for details.
