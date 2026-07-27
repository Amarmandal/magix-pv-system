# P0 — Framing & Decision Log

Last updated: 2026-07-25

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

## 4. Open — blocking

Question · why it blocks · what would resolve it · which notebook owns it

### D-005 :- Target = capacity factor, denominator = producing inverters only

**Status:** Decided on 2026-07-25

**Evidence:** `03_derive-station-capacity.ipynb`

Capacity factor = hourly production ÷ (Σ max_power over PRODUCING inverters × 1h).
The 3 non-producing inverters (all b59685487, 3× SUN2000-50KTL-M3 @ 55kW = 165kW)
are excluded so the denominator matches the numerator's device set.

**Rejected:** summing all 53 rated inverters — would divide real output by 165kW
of never-commissioned capacity, making b59685487 a false chronic underperformer.

## 5. Open — deferred

### Q-001: Forecast Horizon H

## 6. Protocol freeze

The date after which splits, features, metrics and seeds stop changing.
Everything below that line ran under a fixed protocol.
