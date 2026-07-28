# Data Dictionary — SolarMagix

**Dataset:** `hourly_pv_weather_station.csv`

| Column                           | Type     | Description                                  | Role       |
| -------------------------------- | -------- | -------------------------------------------- | ---------- |
| station_hash_id                  | string   | Unique station identifier                    | Excluded   |
| measured_ts                      | datetime | Hourly timestamp                             | Time       |
| total_produced_energy            | float    | Energy produced during the hour              | Feature    |
| source                           | int      | Data source identifier                       | Excluded   |
| capacity_factor                  | float    | total_produced_energy / station rated kW     | **Target** |
| temperature_2m                   | float    | Air temperature                              | Feature    |
| shortwave_radiation              | float    | direct radiation + diffuse radiation         | Feature    |
| direct_radiation                 | float    | Direct solar radiation                       | Feature    |
| diffuse_radiation                | float    | Diffuse solar radiation                      | Feature    |
| global_tilted_irradiance         | float    | Tilted Irradiance (GTI)                      | Feature    |
| terrestrial_radiation            | float    | Solar radiation because it behaves like that | Feature    |
| terrestrial_radiation_instant    | float    | Instantaneous terrestrial radiation          | Unreliable |
| global_tilted_irradiance_instant | float    | Instantaneous GTI                            | Unreliable |
| direct_normal_irradiance_instant | float    | Instantaneous DNI                            | Unreliable |
| diffuse_radiation_instant        | float    | Instantaneous diffuse radiation              | Unreliable |
| direct_radiation_instant         | float    | Instantaneous direct radiation               | Unreliable |
| shortwave_radiation_instant      | float    | Instantaneous shortwave radiation            | Unreliable |
| ac_power                         | float    | AC output power                              | Excluded   |
| dc_power                         | float    | DC input power                               | Excluded   |
| efficiency                       | float    | Inverter efficiency                          | Excluded   |
| device_temperature               | float    | Inverter temperature                         | Excluded   |
| perc_state_on                    | float    | % time inverter was ON                       | Excluded   |
| perc_state_off                   | float    | % time inverter was OFF                      | Excluded   |
| perc_state_error                 | float    | % time inverter was in ERROR state           | Excluded   |
| direct_normal_irradiance         | float    | Direct Normal Irradiance (DNI)               | Excluded   |
| tilt                             | float    | Solar panel tilt angle                       | Excluded   |
| azimuth                          | float    | Solar panel orientation                      | Excluded   |

---

## Notes

- **Primary Key:** (`station_hash_id`, `measured_ts`)
- **Granularity:** Hourly per station
- **Target:** `capacity_factor`
- **ML Task:** Regression / Time-series Forecasting

## Features Categories

- **terrestrial_radiation:** Top-of-atmosphere solar radiation on a horizontal surface. Named misleadingly.
- **shortwave_radiation:** Total incoming shortwave solar radiation. Verified to equal direct_radiation + diffuse_radiation exactly.
- **direct_radiation:** Direct beam solar radiation reaching the surface without atmospheric scattering.
- **diffuse_radiation:** Solar radiation scattered by clouds and the atmosphere before reaching the surface.
- **\_instant:** **Excluded**, Instantaneous measurements were excluded from the baseline feature set.
  Their relationship to the hourly-mean variables was not sufficiently established during data validation.

## Target

- capacity_factor (for regression)
- capacity_factor (h hour ahead using historical data)

---

## Derived Feature

- `cos_zenith`: Deterministic astronomical computation derived from top-of-atmosphere radiation and timestamp.
  - Rule: `cos(zenith) = TOA / (1361 * E0(doy))`, clipped to [0, 1].
  - Notes: replaces previous weather-based implementation (DNI/DN)

- `kt` (clearness index): `shortwave_radiation / terrestrial_radiation`.
  - Guard: set to `NaN` when `terrestrial_radiation` (TOA) ≤ 10 W/m².

- `is_daylight`: boolean derived from TOA (`terrestrial_radiation > 10 W/m²`).

- `hour_sin`, `hour_cos`: cyclic hour features derived from `measured_ts`.

- `doy_sin`, `doy_cos`: cyclic day-of-year features derived from `measured_ts`.
