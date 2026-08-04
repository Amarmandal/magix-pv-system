# P0 — Framing & Decision Log

Last updated: 2026-08-04

## 1. Problem

Can federated learning achieve performance comparable to centralized training while preserving the privacy of each solar station's data?

## 2. Data

Daskalov, Andrej; Zdravevski, Eftim (2026), “Dataset for unified photovoltaic-weather multi-station analysis in North Macedonia”, Mendeley Data, V2, doi: 10.17632/4zgsckxpdy.2

## 3. Decisions

### D-001 :- Fact table = `hourly_pv_weather_stations.csv`

**Status:** Decided

**Evidence:** `01_eda.ipynb` file inventory

### D-002 :- Client Grain = station -> 7 clients

**Status:** Decided

**Evidence:** `01_eda.ipynb` notebook, `processed/client` and `data/processed/splits.json`

### D-003 :- Data split = per-client temporal, 70/15/15, cut by row position

**Status:** Decided

**Evidence:** `splits.py` inside solarlf package, `processed/client` and `data/processed/splits.json`

Each client is split on its own timeline: first 70% of its observations to
train, next 15% val, last 15% test.

**Rejected:** global calendar cut — stations commissioned at different dates,
so some clients would get an empty or unrepresentative train set. Cut by time
_span_ — with ~21% mean inter-day missingness, a time-based 70% mark does not
put 70% of observations in train. Random split — adjacent hours are near
duplicates; leaks by construction.

**Consequence:** cut dates differ per client, so each test set has a different
seasonal composition. Must be stated alongside results, not buried.
Determinism: no seed is involved — sorting and slicing is reproducible by
construction.

### D-004 :- Task Family = h-ahead forecasting

**Status:** Decided . horizon H deferred

**Evidence:**

### D-005 :- Feature Selection for Client-Invariant Training

**Status:** Decided

**Decision:**
Drop the features `tilt`, `azimuth`, `station_hash_id`, and `source` before model training.

**Evidence:**
A variance audit was performed to identify features that remain constant within each client (station). These features were then evaluated across the pooled dataset. `04_scope-feature.ipynb`

- `tilt` and `azimuth` were found to be **globally constant** (`n_unique_pooled = 1`). Since every observation shares the same value, they contain no predictive information and only increase feature dimensionality. Their removal is standard data-cleaning (housekeeping) and does not affect model behaviour.

- `station_hash_id` and `source` were found to vary **only across clients**, not within clients. These features encode client identity rather than the underlying physical relationship between weather variables and power generation. Retaining them would allow the model to memorize station-specific behaviour instead of learning patterns that generalize to unseen stations.

Because the experimental protocol evaluates generalization using Leave-One-Source-Out (LOSO), retaining client identifiers would introduce information leakage and compromise the validity of the evaluation. Therefore, these features are intentionally removed as a methodological design decision rather than a preprocessing convenience.

**Rationale:**
The objective of the study is to evaluate whether Federated Learning can generalize to previously unseen clients. Feature selection should therefore preserve only predictive variables that represent the underlying forecasting problem and exclude features that either:

1. provide no information (`tilt`, `azimuth`), or
2. reveal client identity (`station_hash_id`, `source`).

This ensures that model performance reflects learned relationships between input variables and photovoltaic power generation rather than memorization of individual client characteristics.

### D-006 :- Remove Redundant Shortwave Radiation Feature

**Status:** Decided

**Decision**

Retain `direct_radiation` and `diffuse_radiation`.
Remove `shortwave_radiation` during model training.

**Evidence**

The physical identity

shortwave_radiation = direct_radiation + diffuse_radiation

was verified exactly over all 13,444 daytime observations. For detail refer to `04_scope-feature.ipynb`.  
and cell 4

Residual statistics:

- mean = 0.0
- std = 0.0
- max = 0.0

**Rationale**

`shortwave_radiation` is a deterministic linear combination of the other two features and therefore contributes no additional information.

The direct and diffuse components preserve cloudiness information that the summed measurement does not.

Removing the derived feature reduces redundancy and avoids perfect multicollinearity in linear models.

### D-007 :- Exclude Instantaneous Radiation Features

**Status:** Decided

**Decision**

Exclude all \*\_instant radiation variables from the baseline feature specification.

**Evidence**

The correlation between each hourly-mean radiation variable and its corresponding instantaneous measurement was weak (approximately −0.35 to 0.22), indicating that the instantaneous variables do not behave as direct counterparts of the hourly observations.

**Rationale**

Because their temporal relationship could not be established confidently, the instantaneous variables were excluded from the baseline feature set. They may be revisited after validating their timestamp semantics.

### D-008 :- Replace Weather-derived cos_zenith with Deterministic Astronomical Computation

**Status:** Decided

**Decision**

Replace the previous cos_zenith implementation based on

$$\cos(\text{zenith}) = \frac{\text{direct\_radiation}}{\text{direct\_normal\_irradiance}}$$

with the deterministic implementation derived from top-of-atmosphere (terrestrial) radiation.

**Evidence**

- Correlation with previous implementation: 0.9991
- Maximum absolute deviation: 0.0442
- Coverage increased from 11,962 to 13,444 daytime observations (+1,482 rows, +11%).
- `04_scope-features.ipynb` Check the Resolving problematic cos zenith cell

**Rationale**

The previous implementation depended on measured weather variables (direct_radiation and direct_normal_irradiance) and excluded observations where DNI ≤ 10, reducing coverage and introducing weather-dependent feature availability.

The new implementation derives cos_zenith from deterministic astronomical quantities, making it available for all daytime observations and for any future prediction timestamp without requiring weather measurements.

### D-009 :- Target = capacity factor, denominator = producing inverters only

**Status:** Decided on 2026-07-25

**Evidence:** `03_derive-station-capacity.ipynb`

Capacity factor = hourly production ÷ (Σ max_power over PRODUCING inverters × 1h).
The 3 non-producing inverters (all b59685487, 3× SUN2000-50KTL-M3 @ 55kW = 165kW)
are excluded so the denominator matches the numerator's device set.

**Rejected:** summing all 53 rated inverters — would divide real output by 165kW
of never-commissioned capacity, making b59685487 a false chronic underperformer.

### D-010 — Impossible Capacity Factors Set to NaN

**Status:** Decided (2026-07-28)  
**Evidence:** `05_target-capacity-factor.ipynb`

**Decision**

Set all `capacity_factor > 1.0` values to `NaN`.

**Rationale**

A capacity factor greater than **1.0** is physically impossible—a photovoltaic system cannot produce more than its rated capacity over an hourly interval.

**Evidence**

- **87** invalid rows detected out of **17,983** total rows (**0.48%**).
- Distribution:
  - **S3:** 1 row
  - **S7:** 86 rows

**Implementation**

Values were replaced with `NaN` rather than deleting the rows because `splits.json` preserves row indices. Removing rows would invalidate the existing train/test splits and break `load_split()`.

**Validation**

The anomalies are **not** caused by missing timestamps:

- All 86 affected rows in **S7** occur at a regular **1-hour interval** (`std = 0.0`).
- The actual timestamp gaps occur in other (valid) observations, with gaps of up to **77 hours**.

### D-011 — Exclude Contemporaneous System-State Columns

**Status:** Decided (2026-07-28)

**Decision**

Exclude the following columns from the model feature set:

- `ac_power`
- `dc_power`
- `efficiency`
- `device_temperature`
- `perc_state_on`
- `perc_state_off`
- `perc_state_error`

**Rationale**

These features are **not available at forecast time**.

- `ac_power`, `dc_power`, `efficiency`, and `device_temperature` are measurements of the target or its direct consequences, introducing **target leakage**.
- `perc_state_on`, `perc_state_off`, and `perc_state_error` describe the inverter's operating state during the **predicted hour**, which cannot be known before the prediction is made.

**Scope**

This decision is valid for **all forecasting horizons** (`H ≥ 1`) and is therefore independent of **Q-001**.

**Future Use**

These features remain valuable as **lagged features** once the forecasting horizon is fixed (`lag ≥ H`). They are also suitable for **anomaly detection** tasks, where the current system state is the signal of interest rather than a source of leakage.

## D-012 — Forecast Horizon: H = 24 Hours

**Status:** ✅ Decided (2026-07-30)  
**Resolves:** Q-001

### Decision

Use a **24-hour (day-ahead)** forecast horizon. All historical (lag) features must use **lag ≥ 24 hours**.

### Rationale

- At **H = 1**, **persistence** (current production ≈ next-hour production) is the dominant signal. Each client can solve the task independently, leaving little benefit for Federated Learning (FL).
- At **H = 24**, persistence is much weaker. The prediction depends primarily on the **weather → power generation** relationship, which is governed by the same underlying physics across all solar plants.
- This shared relationship is exactly the type of knowledge that **Federated Averaging** can learn and transfer between clients.
- A 24-hour horizon also aligns with **day-ahead electricity market operations**, making the task practically relevant.

### Trade-offs

- Higher forecast errors compared to short-horizon forecasting.
- Lag-24 features require a complete hourly time index. After reindexing, approximately
  96–98% of samples have a valid lag-24 feature; the remaining missing values arise
  primarily from inter-day gaps and quality filtering (e.g., invalid capacity factors).
- Weather and solar geometry features remain available for all samples.

### Consequence

The project evaluates whether Federated Learning can learn a **shared weather-to-production mapping**, rather than simply matching a strong persistence baseline.

## D-013 — Evaluate FedAvg and Personalized FL

**Status:** ✅ Decided (2026-07-30)

### Decision

Evaluate both **FedAvg** and a **personalized Federated Learning** variant against a centralized baseline.

### Rationale

Exploratory analysis shows consistent client-level differences (approximately 2× spread in capacity factor) while the irradiance response curves remain approximately parallel. This suggests a shared weather-to-power relationship with client-specific offsets rather than different underlying physics. A personalized FL approach is therefore expected to better capture client heterogeneity than a single global FedAvg model.

### Trade-offs

- Requires implementing and evaluating a second FL method.
- Personalization strategy (e.g., which layers remain local) must be defined.

### Consequence

The FL evaluation includes both FedAvg and Personalized FL. We predict that FedAvg will exhibit systematic bias on extreme clients, while personalization will reduce this gap and approach centralized performance.

## D-014 — Weather features: lagged-observed vs perfect-forecast

**Status:** ✅ Decided (2026-07-30)

### Decision

The builder produces two sets of weather features:

- **`weather_past_*`**: observed weather at the forecast origin (**T−24**). These values would be available when making the prediction.
- **`weather_future_*`**: weather at the target time (**T**). These represent a **perfect forecast** and would not be available in a real deployment.

### Rationale

Both feature sets are generated, and each model variant chooses which one to use.

- **Lagged-only weather** provides a realistic baseline.
- **Perfect-forecast weather** provides an optimistic upper bound, allowing us to isolate whether the weather→power relationship transfers across clients under FedAvg without forecast error.

### Trade-offs

- Results using **`weather_future_*`** assume perfect knowledge of future weather and therefore cannot be deployed in practice.
- Real day-ahead weather forecasts (e.g., Open-Meteo) include forecast errors and are not used in this project.

### Consequence

All results using **`weather_future_*`** are explicitly labelled as **perfect-forecast** and interpreted as an upper bound. They are not directly comparable to a real-world forecasting system.

## D-015 — Evaluation is daylight-only

**Status:** ✅ Decided (2026-08-02)

### Decision

Model training and all reported metrics cover **daylight hours only**. Night rows
are excluded from the model matrix as a consequence of `kt` being undefined when
top-of-atmosphere radiation ≤ 10 W/m² (D-008 guard), combined with the
`dropna(subset=MODEL_MATRIX + ["capacity_factor"])` in `build_features`.

Rows are additionally required to have **all** feature columns present, including
`weather_future_*`. The `past` and `perfect` variants are therefore scored on an
identical row set.

### Evidence

`src/solarfl/data/features.py`, measured 2026-08-02:

| client | raw rows | daylight | in model matrix | matrix/daylight |
| ------ | -------- | -------- | --------------- | --------------- |
| S1     | 2470     | 1453     | 1392            | 0.958           |
| S2     | 1627     | 1453     | 1392            | 0.958           |
| S3     | 1789     | 1604     | 1549            | 0.966           |
| S4     | 1769     | 1601     | 1548            | 0.967           |
| S5     | 1770     | 1601     | 1548            | 0.967           |
| S6     | 1759     | 1588     | 1536            | 0.967           |
| S7     | 6799     | 4098     | 3907            | 0.953           |

- Total: 17,983 raw → 13,398 daylight (74.5%) → 12,872 in matrix (71.6% of raw).
- The 95.3–96.7% matrix/daylight retention is the lag-24 availability rate, and
  independently confirms the 96–98% figure asserted in D-012.
- Persistence MAE is identical across the `past` and `perfect` variants for every
  client in `data/results/baselines_val.csv`, confirming the two variants share a
  row set.

### Rejected

Imputing `kt = 0` at night plus a night indicator, which would retain all 17,983
rows.

### Rationale

- Keeping the night hour rows is not sensible because predicting night does not demonstrate forecast ability
- It will just make the metrics look artificially better
- Also, forecasting the Night time production is trivial (Night -> 0)

### Consequence

<!-- YOURS. Prompt: how must every metric in this project be labelled from now
     on, and what comparison would become invalid if a future result silently
     included night hours? -->

---

## D-016 — Lag features computed on the full timeline, split filter applied afterwards

**Status:** ✅ Decided (2026-08-02)

### Decision

`build_features(station_id, split=...)` computes `history_capacity_factor` and  
`weather_past_*` against the client's **entire** timeline, then filters rows down  
to the requested split. A val or test row therefore retains a T−24 value that may  
originate from a row belonging to an earlier split.

Split boundaries are read from `configs/splits.json` and the resulting row counts
are verified against the manifest, raising on mismatch.

### Evidence

- `_split_mask` and the ordering of operations in
  `src/solarfl/data/features.py`.
- Half-open interval convention matches `splits._assign`:
  `[.., train_end) [train_end, val_end) [val_end, ..]`.
- Manifest verification is enforced twice: total row count against `n_total`, and
  per-split selected count against `counts[split]`.

### Rejected

Filtering to the split first, then computing lags within it. This would blank the
first 24 hours of every split.

### Rationale

<!-- YOURS. The key question an interviewer will ask: "isn't a val row reading a
     train row leakage?" Prompts:
  - At the moment you forecast time T, is the value at T−24 known or unknown?
  - Does it matter which split T−24 was administratively assigned to?
  - What IS the thing leakage prohibits — using data from the past, or using
    data from the future relative to the forecast origin?
  - Separately: why does the scaler (mu/sigma in mlp.py) NOT get this same
    latitude? What's different about it?
-->

### Consequence

<!-- YOURS. Prompt: what does the manifest count check buy you, given the
     "never delete rows" invariant? What failure does it catch? -->

---

## D-017 — Predictions clipped to the physical range [0, 1]

**Status:** ✅ Decided (2026-08-02)

### Decision

All model predictions are clipped to `[0, 1]` before any metric is computed
(`_clip` in `src/solarfl/models/baselines.py`). Persistence is not clipped, as it
is already a realised capacity factor and bounded by construction (D-010).

### Evidence

The clip is not cosmetic — it binds frequently on the linear model:

| model | variant | predictions below 0 | rate  |
| ----- | ------- | ------------------- | ----- |
| ridge | past    | 117 / 2059          | 5.7%  |
| ridge | perfect | 253 / 2059          | 12.3% |

### Rejected

Leaving predictions unclipped.

### Rationale

<!-- YOURS. Prompts:
  - Ridge has no way to know CF ≥ 0. Is clipping supplying information the model
    lacked, or correcting an output the domain already forbids?
  - Which model family benefits more from the clip, and does that make the
    ridge-vs-MLP comparison more or less fair? (Note the 12.3% figure.)
  - Would you defend clipping if you were reporting ONLY ridge?
-->

### Consequence

<!-- YOURS. Prompt: does this flatter ridge relative to the MLP, and how should
     that be disclosed alongside the ridge-vs-MLP comparison? -->

---

## D-018 — One shared MLP architecture across every training regime

**Status:** ✅ Decided (2026-08-02)

### Decision

Local, centralized, and (forthcoming) federated runs all use the same model class:
`MLP(hidden=(64, 32))`, ReLU, Adam at lr 1e-3, MSE loss, batch size 256, early
stopping on validation MAE with patience 25 and best-weight restore, seed 0.
Implemented in PyTorch (`src/solarfl/models/mlp.py`).

Feature standardisation (`mu`, `sigma`) is fitted on the **train split only** and
is carried with the fitted model in `FittedMLP`.

### Evidence

- `src/solarfl/models/mlp.py`.
- Run-to-run and machine-to-machine reproducibility confirmed: the results table
  in `data/results/baselines_val.csv` reproduced bit-for-bit on 2026-08-02.

### Rejected

`sklearn.neural_network.MLPRegressor` — no accessible `state_dict`, which FedAvg
requires for weight averaging.

### Open / not yet decided

Hyperparameters are **untuned defaults**, not search results. Capacity is held
fixed at (64, 32) even though the centralized model sees ~7× the training data of
any local model — this is an acknowledged confound in the local-vs-centralized
comparison and is not yet resolved.

### Rationale

<!-- YOURS. Prompts:
  - If centralized used a bigger network than local, what would a
    "centralized wins" result actually prove?
  - Why does the standardiser have to travel inside FittedMLP rather than being
    recomputed at predict time?
  - Why does the centralized model early-stop on POOLED val rather than each
    client's own val? What would per-client stopping quietly grant it?
-->

### Consequence

<!-- YOURS. Prompt: what must stay frozen for the FedAvg numbers to be
     comparable to these baselines? -->

---

## D-019 — Skill score defined on MAE against same-hour-yesterday persistence

**Status:** ✅ Decided (2026-08-02)

### Decision

`skill = 1 − MAE(model) / MAE(persistence)`, where persistence is the
`history_capacity_factor` feature (the realised capacity factor at T−24).
Positive skill beats persistence; persistence scores exactly 0 by construction.
MAE and RMSE are both reported, but skill is computed on MAE only.

### Evidence

`src/solarfl/eval/metrics.py`. RMSE/MAE ratio by model (val, 2026-08-02):

- persistence: 1.46 – 1.63 (highest in almost every client row)
- fitted models: 1.27 – 1.61

Persistence has visibly fatter error tails. Defining skill on RMSE instead would
raise every model's score — e.g. S1 MLP-local rises from 0.153 to 0.228.

### Rationale

<!-- YOURS. Prompts:
  - Raw MAE for S1 (44 kW) and S7 (1300 kW) — why can't you compare those two
    numbers directly, and what does normalising by persistence fix?
  - RMSE-skill would make your models look better. Why is choosing the metric
    that flatters you less the defensible move here?
  - What kind of event produces a single huge persistence error at H=24, and why
    shouldn't one such hour dominate the ratio?
-->

### Consequence

<!-- YOURS. Prompt: the project invariant says no result is meaningful without a
     persistence comparison. What does a NEGATIVE skill score oblige you to
     report rather than quietly drop? (See the ridge/`past` result.) -->

---

## D-020 — Model selection on validation; test split untouched until protocol freeze

**Status:** ✅ Decided (2026-08-02)

### Decision

All results reported to date — `data/results/baselines_val.csv` — are computed on
the **validation** split. The test split has not been read by any model or metric.
Early stopping, and any future hyperparameter tuning, select on val.

### Evidence

`_load_all` in `src/solarfl/models/baselines.py` loads only `train` and `val`.

### Known limitation

Validation sets are small: 229–593 daylight rows per client (2,059 total). Because
adjacent hours are correlated, the effective sample size is closer to the number
of distinct days (~25 per client) than to the row count. An approximate paired
standard error on a per-client MAE difference is therefore ~0.006–0.008, meaning
**no individual local-vs-centralized gap in the current results is statistically
conclusive.** The MLP `past` finding rests on the consistency of its sign
(6/7 clients, median gap −0.0123), not on any single client's margin.

### Rationale

<!-- YOURS. Prompts:
  - Early stopping reads val every epoch. In what sense is val therefore already
    "used up", and what does that imply about reporting a final number on it?
  - Given the noise floor above, what claim are you entitled to make from the
    current table, and what claim would be overreach?
-->

### Consequence

<!-- YOURS. Prompt: what specifically must be frozen at the protocol-freeze line
     (section 6) before the test split is read, and how many times may it be
     read? -->

---

## D-021 — Hand-Rolled FedAvg Instead of Flower

**Status:** ✅ Decided (2026-08-04)

### Decision

FedAvg was implemented directly in `fedavg.py` rather than using a federated learning framework such as Flower.

The implementation explicitly performs the standard FedAvg round:

1. Broadcast the current global model to all clients.
2. Train each client locally.
3. Collect the updated model parameters (`state_dict`).
4. Aggregate the client models using sample-count-weighted averaging.
5. Evaluate the aggregated model on the pooled validation set.
6. Apply early stopping and restore the best-performing model.

### Rationale

A hand-written implementation provides complete control over every stage of the algorithm and makes the implementation directly comparable with the existing centralized training loop.

Using Flower would introduce additional abstractions (client processes, communication APIs, server strategies) that are unnecessary for an offline simulation where all client datasets are already available locally. Since the goal of this study is algorithmic comparison rather than distributed deployment, a minimal implementation improves transparency and reproducibility.

### Consequences

#### Advantages

- Every optimization step is visible and easy to inspect.
- Training is directly comparable with the centralized baseline.
- Easier experimentation with aggregation strategies, stopping criteria, and optimizer behavior.
- Minimal implementation complexity.

#### Limitations

- Does not model real network communication.
- Does not support heterogeneous client availability or asynchronous updates.
- Not directly deployable as a production federated learning system.

---

## D-022 — Global Feature Scaling Using Aggregate Statistics

**Status:** ✅ Decided (2026-08-04)

### Decision

Reconstruct a global feature scaler using per-client aggregate statistics (`sum`, `sum_sq`, and `count`) rather than fitting the scaler on centrally collected data.

Each client shares only:

- feature sums
- feature squared sums
- sample counts

The server reconstructs the **global mean** and **standard deviation** from these aggregates and distributes the resulting normalization parameters to all clients.

### Rationale

This approach produces exactly the same normalization parameters as centralized preprocessing while ensuring that raw feature vectors never leave the client boundary.

Using a common global scaler isolates the effect of federated optimization from differences in preprocessing. Consequently, any performance differences between FedAvg and the centralized MLP arise from the training procedure rather than inconsistent feature normalization.

### Consequences

#### Advantages

- Identical preprocessing to the centralized baseline.
- No raw training samples are transmitted.
- Communication cost is extremely small.
- Fair comparison between centralized and federated training.

#### Limitations

Although substantially more privacy-preserving than sharing raw data, aggregate statistics still cross the client boundary and therefore represent a small relaxation compared with a fully local preprocessing pipeline.

---

## D-023 — FedAvg Configuration

**Status:** ✅ Decided (2026-08-04)

### Decision

The following configuration was adopted:

- Full client participation every communication round.
- One local epoch per communication round (`local_epochs = 1`).
- Adam optimizer recreated independently for every client and every communication round.
- **Sample-count-weighted** model aggregation.
- Early stopping using pooled validation MAE with patience measured in communication rounds.

### Rationale

Using one local epoch per round makes each communication round perform approximately the same amount of gradient computation as one centralized training epoch. This enables direct comparison of training budgets between centralized and federated learning.

Recreating the Adam optimizer each round treats clients as stateless, preventing optimizer moments from becoming inconsistent after the server replaces model parameters with the aggregated global model.

Sample-count weighting follows the original FedAvg algorithm and gives larger datasets proportionally greater influence on the global model.

Monitoring pooled validation MAE mirrors the centralized baseline (D-018), ensuring identical model-selection criteria.

### Consequences

#### Advantages

- Fair comparison with centralized training.
- Configuration closely follows the original FedAvg algorithm.
- Minimal implementation complexity.
- Easy to reproduce.

#### Limitations

Several common federated learning variants remain unexplored:

- Multiple local epochs (`E > 1`)
- Partial client participation
- Uniform client weighting
- Alternative aggregation methods
- Client-specific validation or decentralized stopping criteria

These remain possible directions for future work.

---

## D-024 — Interpretation of FedAvg Results

**Status:** ✅ Decided (2026-08-04)

### Context

FedAvg was evaluated against the centralized MLP, local MLP, and persistence baselines using the same validation protocol.

### Decision

FedAvg is retained as the representative federated baseline but is **not** adopted as the preferred predictive model.

### Rationale

FedAvg consistently performs close to, but slightly worse than, the centralized MLP, while the local MLP achieves the best performance on most clients. This suggests that client-specific characteristics are better captured through personalization than by a single shared global model.

### Consequence

The remainder of the analysis emphasizes the value of personalization while using FedAvg as the federated reference implementation for comparison.

---

## 4. Open — blocking

Question · why it blocks · what would resolve it · which notebook owns it

## 5. Open — deferred

### Q-001: Resolved - See D-012

### Q-002 · RESOLVED — terrestrial_radiation is top-of-atmosphere solar

Night: mean 0.1 W/m², 97.4% exactly zero.  
Day: mean 597.2 W/m², matching
1361 × cos(zenith) at 41.5°N (~600).  
The paper's description ("infrared
emitted by Earth's surface") is incorrect; the column is extraterrestrial
solar radiation on the horizontal.
Unblocks kt (clearness index). Both terrestrial_radiation and
shortwave_radiation retained in `load` as kt inputs, excluded from `features`.
Note for writeup: column semantics verified against data, not the source
publication.

## 6. Protocol freeze

The date after which splits, features, metrics and seeds stop changing.
Everything below that line ran under a fixed protocol.
