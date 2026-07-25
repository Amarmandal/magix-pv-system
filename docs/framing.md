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
