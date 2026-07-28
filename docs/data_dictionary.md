# Data Dictionary — SolarMagix

**Dataset:** `hourly_pv_weather_station.csv`

| Column                           | Type     | Description                                  | Role       |
| -------------------------------- | -------- | -------------------------------------------- | ---------- |
| station_hash_id                  | string   | Unique station identifier                    | ID         |
| measured_ts                      | datetime | Hourly timestamp                             | Time       |
| total_produced_energy            | float    | Energy produced during the hour              | **Target** |
| source                           | int      | Data source identifier                       | Metadata   |
| ac_power                         | float    | AC output power                              | Feature    |
| dc_power                         | float    | DC input power                               | Feature    |
| efficiency                       | float    | Inverter efficiency                          | Feature    |
| device_temperature               | float    | Inverter temperature                         | Feature    |
| perc_state_on                    | float    | % time inverter was ON                       | Feature    |
| perc_state_off                   | float    | % time inverter was OFF                      | Feature    |
| perc_state_error                 | float    | % time inverter was in ERROR state           | Feature    |
| temperature_2m                   | float    | Air temperature                              | Feature    |
| shortwave_radiation              | float    | direct radiation + diffuse radiation         | Feature    |
| direct_radiation                 | float    | Direct solar radiation                       | Feature    |
| diffuse_radiation                | float    | Diffuse solar radiation                      | Feature    |
| direct_normal_irradiance         | float    | Direct Normal Irradiance (DNI)               | Feature    |
| global_tilted_irradiance         | float    | Tilted Irradiance (GTI)                      | Feature    |
| terrestrial_radiation            | float    | Solar radiation because it behaves like that | Feature    |
| terrestrial_radiation_instant    | float    | Instantaneous terrestrial radiation          | Feature    |
| global_tilted_irradiance_instant | float    | Instantaneous GTI                            | Feature    |
| direct_normal_irradiance_instant | float    | Instantaneous DNI                            | Feature    |
| diffuse_radiation_instant        | float    | Instantaneous diffuse radiation              | Feature    |
| direct_radiation_instant         | float    | Instantaneous direct radiation               | Feature    |
| shortwave_radiation_instant      | float    | Instantaneous shortwave radiation            | Feature    |
| tilt                             | float    | Solar panel tilt angle                       | Feature    |
| azimuth                          | float    | Solar panel orientation                      | Feature    |

---

## Notes

- **Primary Key:** (`station_hash_id`, `measured_ts`)
- **Granularity:** Hourly per station
- **Target:** `total_produced_energy`
- **ML Task:** Regression / Time-series Forecasting

## Features Categories

- **terrestrial_radiation:** Behaves as solar radiation (≈0 during nighttime). Although named terrestrial_radiation, exploratory analysis indicates it represents solar radiation rather than continuous terrestrial longwave infrared radiation.
- **shortwave_radiation:** Total incoming shortwave solar radiation. Verified to equal direct_radiation + diffuse_radiation exactly.
- **direct_radiation:** Direct beam solar radiation reaching the surface without atmospheric scattering.
- **diffuse_radiation:** Solar radiation scattered by clouds and the atmosphere before reaching the surface.
- **\_instant:** **Excluded**, Instantaneous measurements were excluded from the baseline feature set. Their relationship to the hourly-mean variables was not sufficiently established during data validation.

## Target

- total_produced_energy (for regression)
- total_energy_produced (h hour ahead using historical data)

---

## Derived Feature

- kt = shortwave_radiation / terrestrial_radiation
- cos(zenith)
