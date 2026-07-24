# Data Dictionary — SolarMagix


**Dataset:** `hourly_pv_weather_inverter.csv`

| Column | Type | Description | Role |
|--------|------|-------------|------|
| device_hash_id | string | Unique inverter identifier | ID |
| measured_ts | datetime | Hourly timestamp | Time |
| total_produced_energy | float | Energy produced during the hour | **Target** |
| source | int | Data source identifier | Metadata |
| ac_power | float | AC output power | Feature |
| dc_power | float | DC input power | Feature |
| efficiency | float | Inverter efficiency | Feature |
| device_temperature | float | Inverter temperature | Feature |
| perc_state_on | float | % time inverter was ON | Feature |
| perc_state_off | float | % time inverter was OFF | Feature |
| perc_state_error | float | % time inverter was in ERROR state | Feature |
| temperature_2m | float | Air temperature | Feature |
| shortwave_radiation | float | Shortwave solar radiation | Feature |
| direct_radiation | float | Direct solar radiation | Feature |
| diffuse_radiation | float | Diffuse solar radiation | Feature |
| direct_normal_irradiance | float | Direct Normal Irradiance (DNI) | Feature |
| global_tilted_irradiance | float | Global Tilted Irradiance (GTI) | Feature |
| terrestrial_radiation | float | Terrestrial radiation | Feature |
| terrestrial_radiation_instant | float | Instantaneous terrestrial radiation | Feature |
| global_tilted_irradiance_instant | float | Instantaneous GTI | Feature |
| direct_normal_irradiance_instant | float | Instantaneous DNI | Feature |
| diffuse_radiation_instant | float | Instantaneous diffuse radiation | Feature |
| direct_radiation_instant | float | Instantaneous direct radiation | Feature |
| shortwave_radiation_instant | float | Instantaneous shortwave radiation | Feature |
| tilt | float | Solar panel tilt angle | Feature |
| azimuth | float | Solar panel orientation | Feature |

---

## Notes

- **Primary Key:** (`device_hash_id`, `measured_ts`)
- **Granularity:** Hourly per inverter
- **Target:** `total_produced_energy`
- **ML Task:** Regression / Time-series Forecasting