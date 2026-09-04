# Framing and decision log

Last updated: 2026-09-04

This file records choices that define the experiment. Implementation detail
belongs in code, and extended analysis belongs in the cited notebooks or result
tables.

## Research questions

**RQ1 — Primary:** Can federated learning achieve forecasting performance
comparable to centralized training without pooling raw station observations?
“Comparable” is evaluated through the pre-specified non-inferiority protocol in
D-029–D-031.

**RQ2 — Future direction:** How does differential privacy affect the
privacy–utility trade-off under explicit privacy budgets? The current system
does not implement differential privacy, secure aggregation, or protected
parameter exchange.

## Data

Daskalov, Andrej; Zdravevski, Eftim (2026), “Dataset for unified
photovoltaic-weather multi-station analysis in North Macedonia,” Mendeley Data,
V2, DOI: [`10.17632/4zgsckxpdy.2`](https://doi.org/10.17632/4zgsckxpdy.2).

## D-001 — Use the hourly station fact table

**Status:** Decided

**Decision:** Use `hourly_pv_weather_station.csv` as the experiment fact table.

**Why:** It contains the hourly station-level production and weather records
required by the forecasting task.

**Evidence:** `notebooks/01_eda.ipynb` file inventory.

**Rejected:** Other source files do not provide the required station-hour grain.

## D-002 — Treat each station as one client

**Status:** Decided

**Decision:** Use one PV station per federated client, giving seven clients.

**Why:** Station boundaries represent the natural data silos and preserve local
observations during federated training.

**Evidence:** `notebooks/01_eda.ipynb`, `data/processed/client/`, and
`configs/splits.json`.

**Rejected:** Arbitrary row partitions would not represent real station silos.

## D-003 — Use per-client chronological 70/15/15 splits

**Status:** Decided

**Decision:** Sort each client by timestamp and split by row position into 70%
train, 15% validation, and 15% test. Splits are deterministic and frozen in
`configs/splits.json`; boundary dates may differ by client.

**Why:** Stations have different commissioning dates and roughly 21% inter-day
missingness, so global-calendar or elapsed-time cuts can produce unrepresentative
partitions.

**Evidence:** `src/solarfl/data/splits.py` and `configs/splits.json`.

**Rejected:** Random splits leak adjacent-hour information; a global calendar
cut can leave clients empty; elapsed-time ratios do not preserve row ratios.

## D-004 — Define an h-ahead forecasting task

**Status:** Superseded by D-012

**Decision:** Frame the problem as forecasting a future target at a fixed
horizon rather than contemporaneous estimation. D-012 fixes the horizon at 24
hours.

**Why:** A declared horizon determines which inputs are available and which lags
are leakage-safe.

**Evidence:** D-011 and D-012.

**Rejected:** Contemporaneous prediction does not answer the day-ahead question.

## D-005 — Exclude constant and client-identity features

**Status:** Decided

**Decision:** Exclude `tilt`, `azimuth`, `station_hash_id`, and `source` from
model inputs.

**Why:** `tilt` and `azimuth` are globally constant; `station_hash_id` and
`source` vary only across clients and would let a pooled model memorize station
identity rather than learn a shared weather-to-power relationship.

**Evidence:** Variance audit in `notebooks/04_scope-features.ipynb` and
`configs/features.yaml`.

**Rejected:** Retaining identifiers would undermine the client-invariant
comparison.

## D-006 — Remove redundant shortwave radiation

**Status:** Decided

**Decision:** Retain `direct_radiation` and `diffuse_radiation`; exclude
`shortwave_radiation` from model inputs.

**Why:** `shortwave_radiation = direct_radiation + diffuse_radiation`, so the sum
adds no information and creates perfect multicollinearity. The components retain
the direct-versus-diffuse distinction.

**Evidence:** Exact zero residual across 13,444 daylight observations in
`notebooks/04_scope-features.ipynb`.

**Rejected:** Keeping all three radiation variables is redundant.

## D-007 — Exclude instantaneous radiation variables

**Status:** Decided

**Decision:** Exclude all `*_instant` radiation variables from the baseline
feature specification.

**Why:** Their timestamp semantics could not be established; correlations with
the corresponding hourly means ranged only from about −0.35 to 0.22.

**Evidence:** `notebooks/04_scope-features.ipynb` and `configs/features.yaml`.

**Rejected:** Treating instantaneous values as hourly counterparts without
validated timing could misalign predictors and targets.

## D-008 — Derive solar geometry deterministically

**Status:** Decided

**Decision:** Derive `cos_zenith` from top-of-atmosphere
`terrestrial_radiation`, replacing the weather-dependent
`direct_radiation / direct_normal_irradiance` calculation.

**Why:** The deterministic method is available for future timestamps and avoids
dropping rows when direct-normal irradiance is small.

**Evidence:** Correlation `0.9991`, maximum absolute deviation `0.0442`, and
coverage increase from 11,962 to 13,444 daylight rows (+11%) in
`notebooks/04_scope-features.ipynb`.

**Rejected:** The former weather-derived ratio reduced coverage and made solar
geometry depend on measured weather.

## D-009 — Define capacity factor using producing inverters

**Status:** Decided · 2026-07-25

**Decision:** Define hourly capacity factor as production divided by the summed
rated power of producing inverters for that station.

**Why:** Three non-producing SUN2000-50KTL-M3 inverters at station
`b59685487` contribute 165 kW of nominal capacity but no production; including
them would create false chronic underperformance.

**Evidence:** `notebooks/03_derive-station-capacity.ipynb`.

**Rejected:** Summing all 53 rated inverters mismatches the numerator’s device
set.

## D-010 — Blank physically impossible capacity factors

**Status:** Decided · 2026-07-28

**Decision:** Set `capacity_factor > 1.0` to `NaN` without deleting rows.

**Why:** Values above rated hourly capacity are physically invalid, while row
deletion would invalidate the frozen split counts.

**Evidence:** 87 of 17,983 rows (0.48%): one at S3 and 86 at S7, identified in
`notebooks/05_target-capacity-factor.ipynb`. S7 anomalies occur on a regular
hourly grid and are not timestamp-gap artifacts.

**Rejected:** Deleting rows breaks split integrity; keeping the values corrupts
the target.

## D-011 — Exclude contemporaneous system state

**Status:** Decided · 2026-07-28

**Decision:** Exclude `ac_power`, `dc_power`, `efficiency`,
`device_temperature`, `perc_state_on`, `perc_state_off`, and
`perc_state_error` at the target time.

**Why:** They measure the target, its consequences, or inverter state during the
predicted hour and are unavailable at forecast time. They may be used only as
lags of at least the forecast horizon or in non-forecasting tasks.

**Evidence:** `configs/features.yaml` and `src/solarfl/data/features.py`.

**Rejected:** Contemporaneous use is target leakage for every horizon `H ≥ 1`.

## D-012 — Fix the forecast horizon at 24 hours

**Status:** Decided · 2026-07-30 · resolves Q-001

**Decision:** Predict 24 hours ahead and require every historical feature to use
`lag ≥ 24`.

**Why:** Day-ahead forecasting is operationally relevant and emphasizes a
shared weather-to-power relationship; a one-hour task would be dominated by
persistence. Lag-24 remains available for approximately 96–98% of usable rows.

**Evidence:** `configs/features.yaml`, `src/solarfl/data/features.py`, and
`notebooks/06_timeseries-eda.ipynb`.

**Rejected:** `H = 1` provides little scope for federated knowledge transfer and
does not answer the paper’s day-ahead question.

## D-013 — Replace planned personalization with FedProx

**Status:** Partly superseded by D-026 · 2026-08-04

**Decision:** Retain the completed FedAvg evaluation, but replace the proposed
personalized-FL experiment with FedProx.

**Why:** Clients show heterogeneous levels but broadly shared irradiance-response
patterns. FedProx tests heterogeneous optimization within the existing global
model pipeline.

**Evidence:** FedAvg and FedProx runners plus D-026.

**Rejected:** FedProx must not be described as personalized; genuine
personalization remains future work.

## D-014 — Separate deployable and perfect-weather variants

**Status:** Decided · 2026-07-30

**Decision:** Build `weather_past_*` from observations at T−24 for the deployable
variant and `weather_future_*` from ERA5 at T for an explicitly labelled
perfect-weather upper bound.

**Why:** The two variants distinguish model-transfer performance from future
weather forecast error.

**Evidence:** `configs/features.yaml` and `src/solarfl/data/features.py`.

**Rejected:** ERA5 reanalysis at T is not a real day-ahead forecast and cannot
support deployment claims.

## D-015 — Train and evaluate on daylight rows

**Status:** Decided · 2026-08-02

**Decision:** Train and report metrics only where top-of-atmosphere radiation is
above 10 W/m² and the full shared model matrix is present. Past- and
perfect-weather variants use the same rows.

**Why:** Night production is trivially zero and would artificially improve
metrics; `kt` is undefined at night. Requiring the shared matrix makes variant
comparisons paired.

**Evidence:** 17,983 raw rows yield 13,398 daylight rows and 12,872 model rows;
per-client retention is 95.3–96.7% of daylight rows. Persistence MAE is
identical across variants in `results/baselines_val.csv`.

**Rejected:** Imputing night `kt = 0` plus a night indicator would reward trivial
night predictions.

## D-016 — Compute lags before applying split filters

**Status:** Decided · 2026-08-02

**Decision:** Compute `history_capacity_factor` and `weather_past_*` on each
client’s full chronological timeline, then select train, validation, or test
rows using the frozen half-open boundaries.

**Why:** A validation or test observation may legitimately use information from
T−24 even when that earlier row lies in the preceding split.

**Evidence:** `_split_mask` and operation order in
`src/solarfl/data/features.py`; counts are checked against
`configs/splits.json`.

**Rejected:** Filtering first would unnecessarily blank the first 24 hours of
each split.

## D-017 — Clip predictions to the physical range

**Status:** Decided · 2026-08-02

**Decision:** Clip every fitted-model prediction to `[0, 1]` before computing
metrics. Persistence is already bounded through D-010.

**Why:** Capacity factor cannot fall outside `[0, 1]`, and clipping materially
affects linear models.

**Evidence:** Ridge produced 117/2,059 negative predictions for the past variant
(5.7%) and 253/2,059 for perfect weather (12.3%); implementation is in
`src/solarfl/models/baselines.py`.

**Rejected:** Scoring physically impossible predictions distorts the comparison.

## D-018 — Share one MLP architecture across regimes

**Status:** Decided · 2026-08-02

**Decision:** Local, centralized, FedAvg, and FedProx use `MLP(hidden=(64, 32))`
with ReLU, Adam (`lr=1e-3`), MSE loss, batch size 256, and validation-MAE early
stopping with patience 25 and best-weight restore. Scaling is fitted on training
data only. Baseline runs use seed 0; D-030 defines final-test seeds.

**Why:** A shared architecture and selection criterion isolate the effect of the
training regime. PyTorch exposes the parameters required for aggregation.

**Evidence:** `src/solarfl/models/mlp.py` and reproducible
`results/baselines_val.csv`.

**Rejected:** `sklearn`’s MLP did not expose a suitable `state_dict`. The fixed,
untuned architecture remains a stated capacity confound.

## D-019 — Define MAE skill against persistence

**Status:** Decided · 2026-08-02

**Decision:** Report `skill = 1 − MAE(model) / MAE(persistence)`, where
persistence predicts the realized capacity factor from T−24. Positive skill
beats persistence; persistence has skill zero. Report MAE and RMSE separately.

**Why:** MAE is the primary error measure and persistence has heavier error
tails, so RMSE-based skill would systematically inflate apparent gains.

**Evidence:** `src/solarfl/eval/metrics.py`; validation RMSE/MAE ratios are
1.46–1.63 for persistence and 1.27–1.61 for fitted models.

**Rejected:** RMSE-based skill answers a different, tail-weighted question.

## D-020 — Select models on validation before opening test

**Status:** Decided · 2026-08-02

**Decision:** Use train data for fitting and validation data for early stopping,
algorithm comparison, and FedProx `mu` selection. Freeze D-029–D-031 before the
one-time final test evaluation.

**Why:** Keeping test data outside model selection preserves the final
confirmatory comparison.

**Evidence:** `results/*_val.csv`, the experiment runners, and
`src/solarfl/eval/test_evaluation.py`.

**Boundary:** Validation has only 2,059 correlated daylight rows (roughly 25
days per client); an approximate per-client MAE-difference SE of 0.006–0.008
makes small validation gaps exploratory rather than conclusive.

## D-021 — Implement FedAvg directly

**Status:** Decided · 2026-08-04

**Decision:** Implement the FedAvg simulation directly: broadcast the global
model, train every client, collect parameters, aggregate by sample count,
evaluate pooled validation MAE, and restore the best global model.

**Why:** The study compares algorithms offline on one machine; a compact loop
makes aggregation, optimizer state, and stopping behavior transparent without
distributed-system abstractions.

**Evidence:** `src/solarfl/federated/fedavg.py`.

**Rejected:** Flower adds deployment machinery not needed here. The direct
simulation does not model networks, unavailable clients, or asynchronous
training.

## D-022 — Reconstruct a global scaler from aggregates

**Status:** Decided · 2026-08-04

**Decision:** Each client contributes feature `sum`, `sum_sq`, and `count`; the
server reconstructs global training means and standard deviations and returns
one scaler to all clients.

**Why:** This exactly matches centralized normalization without transmitting raw
feature vectors and prevents preprocessing differences from confounding the
training-regime comparison.

**Evidence:** Federated scaler code in `src/solarfl/federated/`.

**Boundary:** Aggregate statistics still cross client boundaries and are not a
formal privacy mechanism.

## D-023 — Fix the FedAvg training configuration

**Status:** Decided · 2026-08-04

**Decision:** Use full client participation, one local epoch per round (`E=1`),
a fresh Adam optimizer for each client-round, sample-count-weighted aggregation,
and pooled-validation-MAE early stopping with patience measured in rounds.

**Why:** E=1 approximately matches one centralized pass over all training data;
fresh optimizers avoid retaining moments after global parameters are replaced;
sample weighting follows standard FedAvg.

**Evidence:** `src/solarfl/federated/fedavg.py` and D-028.

**Rejected:** Partial participation, uniform weighting, persistent client
optimizer state, and decentralized stopping are outside this study.

## D-024 — Retain FedAvg as the federated baseline

**Status:** Decided · 2026-08-04

**Decision:** Keep FedAvg as the reference federated method, not as the preferred
predictive model.

**Why:** It performs close to but generally worse than centralized MLP, while
local MLP wins on most clients, suggesting that one global model misses some
station heterogeneity.

**Evidence:** `results/fedavg_val.csv` and `results/baselines_val.csv`.

**Boundary:** This motivates heterogeneity-aware methods but does not prove that
personalization closes the gap; FedProx is interpreted under D-026.

## D-025 — Select FedProx mu on validation

**Status:** Decided and implemented · 2026-08-04

**Decision:** Evaluate `mu ∈ {0.001, 0.01, 0.1, 1.0}` and select the value with
the lowest pooled validation MAE.

**Why:** Proximal strength depends on client heterogeneity, so an arbitrary fixed
value would not be defensible.

**Evidence:** `src/solarfl/federated/fedprox.py` and
`results/fedprox_val.csv`.

**Boundary:** FedProx receives four-way validation selection while FedAvg does
not. Because improvements can be comparable to the 0.006–0.008 validation noise
scale from D-020, validation superiority is exploratory. Nested validation was
not used because of dataset size and project scope.

## D-026 — Use FedProx as the second federated method

**Status:** Decided and implemented · 2026-08-04

**Decision:** Evaluate FedProx as the heterogeneity-aware extension to FedAvg,
replacing the personalized-FL experiment proposed in D-013.

**Why:** FedProx changes only the local objective through a proximal penalty,
allowing a controlled comparison with the existing aggregation pipeline.

**Evidence:** `src/solarfl/federated/fedprox.py`.

**Boundary:** FedProx still returns one shared global model and is not
personalized FL. FedPer, FedBN, and client-specific fine-tuning remain future
work.

## D-027 — Separate optimization and privacy claims

**Status:** Decided · 2026-08-22

**Decision:** RQ1 tests predictive non-inferiority without centrally pooling raw
station rows. Differential privacy is a separate future RQ2 requiring an
explicit privacy unit, adjacency relation, clipping rule, noise mechanism,
accountant, and `(epsilon, delta)` guarantee.

**Why:** Federated optimization alone does not protect model updates or shared
statistics and therefore cannot substantiate a formal privacy claim.

**Evidence:** Current FedAvg/FedProx and aggregate-scaling implementations.

**Rejected:** Do not describe the current system as differentially private,
secure aggregation, secure parameter exchange, or protected client updates.

## D-028 — Use E=1 for the primary comparison

**Status:** Decided · 2026-08-22

**Decision:** Compare centralized MLP with full-participation FedAvg/FedProx at
`E=1` for primary RQ1. Treat `E=5` as a secondary
communication–computation experiment, comparing FedAvg E5 vs E1, FedProx E5 vs
E1, and FedProx E5 vs FedAvg E5.

**Why:** One centralized epoch uses about 34 optimizer steps and one E=1 round
uses about 32 client steps; E=5 performs roughly five times the local work per
round and is not compute-matched.

**Evidence:** Training-set sizes and runners in `src/solarfl/federated/`.

**Boundary:** E=5 cannot support equal-compute or communication-efficiency
claims without round histories and a pre-specified performance threshold. It
does not determine the primary non-inferiority result.

## D-029 — Freeze a 0.005 non-inferiority margin

**Status:** Decided · 2026-08-22

**Decision:** For FedProx E1 with validation-selected `mu=1` versus centralized
MLP, define `D = macro-MAE_FedProx − macro-MAE_centralized`. Declare FedProx
non-inferior only when the upper endpoint of its paired two-sided 95% confidence
interval is strictly below `0.005`.

**Why:** A margin of 0.005 is 0.5 percentage points of normalized capacity-factor
MAE and about 4.5% of the centralized validation macro-MAE (~0.112). It was
fixed before test evaluation.

**Evidence:** D-030–D-031 and `results/test_evaluation_summary.csv`.

**Boundary:** Crossing zero does not prevent non-inferiority; crossing 0.005
makes the result inconclusive. The margin applies only to the primary
past-weather E1 comparison and cannot be changed after observing test results.

## D-030 — Average the final comparison across five seeds

**Status:** Decided · 2026-08-23

**Decision:** Train centralized MLP and FedProx E1 (`mu=1`) with paired seeds
`0, 1, 2, 3, 4`. For each seed compute its client-macro-MAE gap, then average
the five gaps. Hold these ten fitted models fixed during bootstrap resampling.

**Why:** Paired fixed seeds reduce dependence on a favorable initialization
without selecting a seed using test performance.

**Evidence:** `src/solarfl/eval/seed_robustness.py`,
`src/solarfl/eval/test_evaluation.py`, and
`results/test_evaluation_by_seed.csv`.

**Rejected:** Best-seed selection is test leakage; seeds are not independent
test sets; bootstrap retraining would estimate a different source of
uncertainty.

## D-031 — Bootstrap paired complete calendar days

**Status:** Decided · 2026-08-23

**Decision:** Match centralized and FedProx errors by client, timestamp, and
seed; resample complete calendar dates with replacement across all clients and
seeds; run 10,000 repetitions with RNG seed 0; and report the 2.5th and 97.5th
percentiles. Compute unweighted client-macro-MAE per seed before the five-seed
average. Timestamps are timezone-naive and are not converted.

**Why:** Pairing removes condition mismatch, while date blocks preserve
within-day weather and solar-cycle dependence.

**Evidence:** `src/solarfl/eval/test_evaluation.py` and
`results/test_evaluation_bootstrap.csv`.

**Rejected:** Hour-level resampling breaks temporal dependence; independent
method resampling breaks pairing; a normal-theory interval assumes an
unsupported distribution. The two-sided 95% rule remains as frozen.

## D-032 — Use portable provenance in the split manifest

**Status:** Decided · 2026-09-04

**Decision:** Set `configs/splits.json` source to the versioned Mendeley DOI.
Pin the exact source file separately by file ID and SHA-256 in
`data/README.md` and `scripts/prepare_data.py`.

**Why:** A DOI identifies the public source independently of any researcher’s
checkout; the file ID and checksum identify its exact bytes.

**Evidence:** `configs/splits.json`, `data/README.md`, and
`scripts/prepare_data.py`.

**Rejected:** Absolute paths are machine-specific; relative paths describe only
local placement. This metadata correction did not regenerate or otherwise
change the frozen splits.

## Resolved questions

- **Q-001 — Forecast horizon:** resolved by D-012 (`H=24`).
- **Q-002 — Meaning of `terrestrial_radiation`:** verified from the data as
  top-of-atmosphere solar radiation on a horizontal surface, not terrestrial
  infrared. Night mean is 0.1 W/m² (97.4% zero); daylight mean is 597.2 W/m²,
  consistent with `1361 × cos(zenith)` near 41.5°N. It is retained for deriving
  `kt` and excluded from direct model inputs.

## Protocol freeze

D-029–D-031 define the pre-test primary comparison, margin, seed policy, and
uncertainty procedure. The final test outputs in `results/test_evaluation_*.csv`
were produced under that fixed protocol. Any future change must receive a new
decision entry and must not retroactively alter the archived claim.
