# Data Dictionary — SolarMagix

**Dataset:** `hourly_pv_weather_station.csv`

| Column                           | Type     | Description                                  | Role       |
| -------------------------------- | -------- | -------------------------------------------- | ---------- |
| station_hash_id                  | string   | Unique station identifier                    | Excluded   |
| measured_ts                      | datetime | UTC interval-start timestamp                 | Time       |
| total_produced_energy            | float    | Energy in kWh over [T, T+1 hour)              | Target source |
| source                           | int      | Data source identifier                       | Excluded   |
| capacity_factor                  | float    | hourly kWh / (retained inverter kW * 1 hour)     | **Target** |
| temperature_2m                   | float    | Air temperature                              | Feature    |
| shortwave_radiation              | float    | Global horizontal irradiance (GHI), W/m²         | Derived-feature input |
| direct_radiation                 | float    | Direct solar radiation                       | Feature    |
| diffuse_radiation                | float    | Diffuse solar radiation                      | Feature    |
| global_tilted_irradiance         | float    | Tilted Irradiance (GTI)                      | Feature    |
| terrestrial_radiation            | float    | Top-of-atmosphere solar irradiance on a horizontal surface | Derived-feature input |
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
- `capacity_factor` at T, forecast 24 hours ahead using information available at T−24

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

---

## Model Matrix — feature counts

The input matrix is specified by `model_matrix:` in `configs/features.yaml` and
built by `build_features()` in `src/solarfl/data/features.py`, which reads that
yaml at import. The counts below are derived from the spec, not maintained by
hand — if they disagree with the yaml, the yaml wins.

| Group            | Count | Prefix in `X`      | Available at forecast time?             |
| ---------------- | ----- | ------------------ | --------------------------------------- |
| `history`        | 1     | `history_`         | Yes — capacity_factor at T−24 (D-012)   |
| `geometry`       | 6     | *(none)*           | Yes — deterministic astronomy           |
| `weather_past`   | 5     | `weather_past_`    | Yes — ERA5 observed at T−24             |
| `weather_future` | 5     | `weather_future_`  | **No** — ERA5 reanalysis at T itself    |

### Variant A — operational (past-only): **12 features**

`history` + `geometry` + `weather_past`. This is the past-only offline configuration at a nominal 24-hour timestamp
horizon. A complete hour labeled T−24 arrives after that hour ends; real-time
input availability is not verified. See [solar semantics](solar_semantics.md).

### Variant B — perfect-weather upper bound: **17 features**

Variant A + `weather_future`. Adds ERA5 reanalysis for the predicted hour
itself, which stands in for a weather forecast that would in reality carry its
own error. Scores from this variant are an **upper bound**, not an achievable
operational result, and must be labelled as such wherever they are reported
(D-014).

The archived A/B contrast is exploratory seed-0 validation evidence about
target-hour input availability, not a causal cost estimate or a test endpoint.

`build_features()` returns 17 columns; Variant-A callers explicitly select
`PAST_MODEL_MATRIX` (12 columns). `row_set="past"` requires only past predictors,
target and daylight; `row_set="common"` additionally requires target-hour
reanalysis. See the existing variant-row-mask audit for their observed equality.

Actual input widths remain 12/17. The constant `is_daylight` column is retained,
leaving 11/16 nonconstant columns in pooled eligible data; these are not claims
about independent information dimensions. MLP parameter counts remain
2,945/3,265. The executable audit is `python -m solarfl.eval.manuscript_assets`.

[Solar variable definitions and timestamp conventions](solar_semantics.md)
define GHI, GTI, TOA, θz, E0 and units with provider/source citations.
[Manuscript evidence](manuscript_evidence.md) documents capacity reconstruction and scope.
