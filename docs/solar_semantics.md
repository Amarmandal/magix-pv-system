# Solar variables and timestamp convention (review point 7)

Irradiance is radiant **power** per unit area, in W/m². Irradiation is radiant
**energy** per unit area integrated over an interval, in J/m² or Wh/m².
The radiation columns used here are irradiances, even when hourly averaged.

| Symbol / field | Meaning | Units |
| --- | --- | --- |
| GHI / `shortwave_radiation` | Global horizontal surface irradiance: direct horizontal plus diffuse horizontal | W/m² |
| GTI / `global_tilted_irradiance` | Global irradiance on the specified tilted plane | W/m² |
| TOA / `terrestrial_radiation` | Extraterrestrial solar irradiance on a horizontal plane; not Earth-emitted longwave radiation | W/m² |
| θz | Solar zenith angle, measured from the upward vertical | radians in trigonometric expressions |
| `cos_zenith` | Dimensionless normalized TOA feature approximating the cosine of solar zenith over the hourly interval | 1 |
| E0(doy) | Approximate Earth–Sun distance correction, `1 + 0.033*cos(2*pi*doy/365)` | 1 |
| doy | UTC timestamp's day of year, starting at 1 | day index |
| 1361 | Solar constant used by the frozen implementation | W/m² |
| kt | Clearness index, GHI / TOA where TOA > 10 W/m², otherwise NaN | 1 |
| `is_daylight` | 1 where hourly TOA > 10 W/m², otherwise 0 | 1 |

The exact implemented expressions are

```text
cos_zenith = clip(TOA / (1361 * (1 + 0.033*cos(2*pi*doy/365))), 0, 1)
kt = shortwave_radiation / terrestrial_radiation   if TOA > 10, else NaN
```

**GTI is not the numerator of kt.** It is a separate weather predictor. The
hourly-average TOA and approximate E0 mean that `cos_zenith` must not be
described as an instantaneous angle or as exactly equivalent to a standard
solar-position calculation at T. The fixed 365-day approximation also remains
in leap years to preserve the frozen computation.

## Two timestamp conventions

Open-Meteo documents its non-instant radiation fields as **means over the
preceding hour**. Its public implementation explicitly computes
`terrestrial_radiation` using `extraTerrestrialRadiationBackwards` and returns
W/m²; the `_instant` variant uses a separate instantaneous calculation.
[API documentation](https://open-meteo.com/en/docs/historical-weather-api),
[solar-variable documentation](https://open-meteo.com/en/docs/chmi-api),
[official source](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/Controllers/ForecastapiController.swift).

The SolarMagix authors describe UTC **hour-floor** PV aggregation, with the
hour beginning at T. Before joining, they move each preceding-hour weather
aggregate from T+1 to T. Thus the distributed merged rows describe energy and
mean irradiance over **[T, T+1 hour)**. This repository consumes that alignment
as supplied; it does not shift weather a second time. CSV timestamps carry no
explicit timezone suffix; the upstream UTC convention is used without local
time or DST conversion. See the source article's time-zone and gold-layer
sections: [SolarMagix, DOI 10.1016/j.dib.2026.112830](https://doi.org/10.1016/j.dib.2026.112830).

The capacity-factor target is `E_[T,T+1h) / (P_rated * 1 hour)`, where energy
is kWh and capacity is kW. This ratio is dimensionless.

The lag code uses **24 hours between stored interval-start labels**. A complete
aggregate labeled T−24 is available only after that interval ends at T−23,
before any ingestion delay. Therefore a literal dispatch at the instant T−24
would not yet have that complete aggregate. ERA5 access latency is also not
simulated. Describe Variant A as the past-only offline configuration with a
nominal 24-hour timestamp horizon; do not claim a verified real-time day-ahead
deployment. Changing the origin convention or lag now would require a separate
experiment, not silently relabeling or modifying the frozen result.

Sources checked 6 September 2026. The upstream article's longwave description
of `terrestrial_radiation` conflicts with the provider's solar implementation;
use the provider's semantics for this variable.
