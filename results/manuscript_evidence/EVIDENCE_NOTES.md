# Manuscript evidence

The implementation and audit assets below support the manuscript methodology,
reporting scope and reproducibility statements. The manuscript itself has not
been edited by this evidence-generation workflow.
No additional model was fitted or selected on the test split.

## Upload inventory

This folder contains 10 files: this evidence note, six CSV tables, capacity
source provenance, and the two current author-supplied PNG figures. Upload
these files together. Table layout and LaTeX can be created directly in Prism.

- `experiment_scope.csv`: validation versus test scope, settings and archived means.
- `station_capacity.csv`: station-level denominator counts and capacities.
- `capacity_device_audit.csv`: device-level inclusion/exclusion evidence.
- `capacity_provenance.json`: versioned source URLs, hashes and reconstruction result.
- `feature_dimensions.csv`: input widths, constant columns and parameter counts.
- `split_counts.csv`: raw and retained counts and time boundaries by station/split.
- `optimization_protocol.csv`: optimizer, seed, aggregation and stopping settings.
- `overall_pipeline_2x.png` and `federated_architecture_2x.png`: current figures.

CSV values are preserved from the original audit. No new training or test
selection was performed. Redundant formats and historical diagrams are not
part of this upload packet. The solar definitions and sources are included below.

## 4. Scope of evidence

`experiment_scope.csv` records each archived comparison's split, seed scope,
variant and role. Broad local/centralized Ridge/MLP and FedAvg/FedProx sweeps,
including E=5 and perfect-weather B, are exploratory seed-0 validation results.
The past-only validation robustness study additionally uses seeds 0–4 for
centralized MLP, FedAvg E1 and FedProx E1. Only centralized MLP versus selected
FedProx E1, mu=1, Variant A, seeds 0–4 supports the frozen confirmatory claim.
Persistence remains the skill reference, not an additional inferential endpoint.

Suggested revised title for author consideration: **“Federated versus
Centralized Day-Ahead PV Capacity-Factor Forecasting: A Non-Inferiority Study
Across Seven Stations in North Macedonia.”** The repository retains the
existing citation title until the manuscript authors adopt a new one.

Use `overall_pipeline_2x.png` for the revised evaluation-flow figure. Perfect-
weather improvements must be described as seed-0 validation comparisons.
The means in the broad-sweep inventory are computed from archived rounded
per-client scores and are not newly estimated high-precision results.

## 5. Data locality and privacy

The implementation has the following data flows and limitations.
One station per client is an analytical partition representing a possible data
silo; independent ownership or a confidentiality obligation is not established.
The study is a single-process simulation on public data. Client training
updates use separate station arrays, but the host can access all arrays, and
validation predictors and targets are concatenated centrally for early stopping.
Weights and aggregate scaling statistics are unprotected. No threat model,
attack evaluation, differential privacy or secure aggregation is implemented.

Use data-governance or commercial-confidentiality concerns as motivation only.
The measured endpoint is predictive utility under the simulated training
partition, not demonstrated privacy protection or enforced data isolation.

## 6. Capacity denominator

`solarfl.labels.capacity_audit` verifies official V2 file IDs, sizes and SHA-256
hashes from `docs/source_manifest.json`, then independently reconstructs labels
and capacities. It fails on a mismatch and never overwrites frozen configs.
The correct station fact filename is **`hourly_pv_weather_station.csv`**.

`station_capacity.csv` reports all rated/retained/excluded counts and kW sums;
`capacity_device_audit.csv` gives all 66 device records, inclusion reasons,
observation counts and first/last timestamps. `capacity_provenance.json` records
that the reconstructed labels match the frozen configuration.

The historical rule retains a device with nonmissing positive `max_power` if
its ID appears at least once in `hourly_pv_weather_inverter.csv`, anywhere in
the full released period, 2024-06-27 12:00 through 2025-05-31 23:00 UTC.
It does **not** require positive production or use commissioning records.
Fifty of 53 rated devices are retained; 13 unrated metadata devices are omitted.
Three S5 devices, each 55 kW, are absent from the fact table: S5 therefore uses
605 kW rather than 770 kW. Absence does not establish that they were
uncommissioned. The manuscript should remove that unsupported status claim.

Final S1–S7 capacities are 44, 44, 220, 440, 605, 605 and 1300 kW. `max_power`
is the dataset's inverter rating field, not an independently verified plant
nameplate rating. Station metadata validates membership; device metadata
supplies power. Equal capacities are ordered by station hash to reproduce labels.
This is a retrospective, full-period target-definition rule, not a denominator
learned solely from training data or a time-varying commissioning inventory.

## 7. Radiation and timestamps

See the solar definitions, formulas and authoritative sources below. GHI is `shortwave_radiation`; GTI is a separate tilted-plane input.
`cos_zenith` is a normalized hourly-mean geometry proxy, not an angle or exact
instantaneous solar position. Code comments now state these distinctions.
The upstream weather shift and interval-start convention are documented;
no second shift or numerical feature change was introduced.

The timestamp clarification also exposes a deployment qualification: lag 24
means 24 hours between interval labels, while a complete past-hour aggregate
arrives after its interval ends. Do not assert verified availability at the
instant T−24. Resolving an actual dispatch-time experiment would require a
separate protocol; the frozen nominal-horizon comparison is preserved here.

## 8. Width versus nonconstant columns; row counts

`feature_dimensions.csv` audits the matrices by split.
Widths remain **12 and 17**; pooled nonconstant counts are **11 and 16** because
`is_daylight` stays in the matrices. These counts do not establish independent
information dimensions or matrix rank. Parameter totals from the actual MLP
are **2,945 and 3,265**. The architecture is input → 64 ReLU → 32 ReLU → 1 linear. Test-side B counts are a descriptive mask audit,
not evidence that a perfect-weather test model was fitted.

`split_counts.csv` distinguishes raw
from retained counts. Raw train/validation/test totals are 12,586 / 2,696 / 2,701;
retained totals are **8,719 / 2,059 / 2,094**. Their sum, **12,872**, means
model-eligible observations across all splits, not training examples.
For these totals, filter `split_counts.csv` to variant A and row_set past,
then sum across the seven clients within each split. Other variant/row-set
rows describe alternative views of the same observations, not additional data.

## 9. Algorithm and regime differences

The algorithm takes seed s as an input; the confirmatory comparison loops
over paired seeds 0–4. Each fit initializes the global model using s, broadcasts
weights, trains each client for E epochs, aggregates by retained training-row
count, and checks pooled validation MAE. FedProx uses a squared-distance penalty
to the broadcast weights; mu=0 yields the FedAvg objective. The best weights
are restored when stopping, including when the round budget is reached. `optimization_protocol.csv` captures function defaults and
optimizer-state, aggregation and stopping differences. Both training routines
already accepted and used s; no seed or training dynamics needed changing.

The phrase to use is “hold architecture and primary hyperparameters fixed
while comparing training regimes.” Centralized Adam retains moments across
epochs; federated Adam is fresh per client-round. Data exposure at E=1 is only
approximately compute matched. The actual batching uses `array_split` with a
nominal batch size, not an exact fixed size. Test metrics clip predictions;
validation early stopping uses un-clipped predictions.

The existing decision log remains an historical record. Authors should record
a new D-entry acknowledging the reporting clarifications, retrospective
capacity-membership rule, and interval-availability limitation. This audit
does not retroactively alter the frozen decisions or their dates.

---

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
