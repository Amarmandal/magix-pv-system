# SolarMagix — federated PV forecasting

Day-ahead (H=24) capacity-factor forecasting across 7 PV stations in North
Macedonia. Primary research question: can federated learning achieve performance
comparable to centralized training without pooling raw station observations?
"Comparable" is to be defined by a non-inferiority margin before test
evaluation. The current implementation has no formal differential-privacy
guarantee; differential privacy is an additional privacy–utility research
direction.

## Working protocol — read this first

I am learning ML/DS and must be able to defend every line of this codebase in an
interview. Optimise for my understanding, not for finishing fast.

1. **Explain before writing.** For any new module or non-trivial function, first
   state in chat: what you'll build, which library you'll use, and why that one
   over the obvious alternative. Wait for my go-ahead.
2. **No new dependencies without asking.** Prefer pandas / numpy / stdlib.
   Justify any new package by what it actually saves.
3. **Comment the why, not the what.** `# lag >= H or we leak the future` is
   useful. `# loop over rows` is not.
4. **After writing, list what I should review** — 3–5 bullets covering the
   places you made a judgment call.
5. **Never silently refactor** code that already exists. Propose it, don't do it.
6. **If you're unsure, stop and ask.** A wrong guess costs me more than a
   question does.

## Invariants — breaking these breaks the project

- **Never delete rows from `data/processed/client/*.csv`.** `configs/splits.json`
  stores per-client row counts and `load_split()` raises on mismatch. Blank bad
  values to NaN instead. Pattern: `src/solarfl/labels/capacity.py`.
- **`configs/` is hand-maintained.** `features.yaml`, `splits.json`, and
  `station_labels.json` were bootstrapped by notebooks that are now historical
  records. Never regenerate them from a notebook.
- **H = 24 (D-012).** Every lag feature must use lag >= 24. A lag of 1 is
  leakage at this horizon.
- **`configs/features.yaml` is the spec, not a suggestion.** Code conforms to the
  yaml. If code needs a column the yaml excludes, stop and tell me — do not edit
  the yaml to make code work.
- **Splits are frozen.** Do not regenerate `configs/splits.json`.
- **Baselines before models.** No model result is meaningful until it's compared
  against same-hour-yesterday persistence.
- **RQ1 non-inferiority uses E=1.** Treat centralized MLP versus full-participation
  FedAvg/FedProx with one local epoch per round as the primary, approximately
  compute-matched comparison. E=5 is a secondary communication–computation
  trade-off experiment; do not use it for the primary non-inferiority claim or
  call it communication-efficient without round-history evidence (D-028).

## Decision log

Every non-obvious choice gets an entry in `docs/framing.md` as `D-0xx`, with
status, date, rationale, and what was ruled out. Highest so far: D-028.

When you make a judgment call, tell me it needs a D-entry. Don't write the entry
yourself — writing it is how I confirm I understood the choice.

## Layout

- `configs/` — hand-maintained specs and frozen artifacts
- `src/solarfl/` — the library: `data/`, `labels/`, `eval/`, `federated/`, `models/`
- `notebooks/` — numbered, exploratory, historical. Never imported from.
- `docs/` — `framing.md` (decision log), `data_dictionary.md`, `domain_knowledge.md`
- `data/` — gitignored, local only

Notebooks decide; modules execute. Anything that runs more than once belongs in
`src/`.

## Domain terms

- **capacity_factor** — the target. Hourly kWh / station rated kW. NaN where > 1.0.
- **client** — one PV station. 7 total, labelled S1–S7 in `station_labels.json`.
- **kt** — clearness index. Ratio of surface shortwave irradiance to
  top-of-atmosphere irradiance. Clear sky ≈ 0.75, heavy cloud ≈ 0.15.
- **cos_zenith** — solar geometry, derived from `terrestrial_radiation` (D-008).

## Gotchas

- `terrestrial_radiation` is top-of-atmosphere **solar** radiation, not infrared.
  The source paper's description is wrong; verified against the data in DK-003.
- Weather is ERA5 **reanalysis**, not forecast. Using it as a known-future
  covariate is a perfect-forecast setup — label results accordingly.
- Hourly-grid coverage is only 45–84% per device, but the holes are structured:
  the same deep-night hours are missing every day, and whole-day loss comes as
  2–3 contiguous outages, not scattered days. So lag-24 is available for 96–98%
  of rows — verified in 06_timeseries-eda. Weather and geometry have no gaps.
- Stations are all within ~200 km, so client data is weather-correlated. They are
  not independent silos.

## Commands

<!-- fill these in and delete this comment -->

- Install: `uv sync`
- Run a module: `uv run python -m solarfl.<module>`
- Import check: `uv run python -c "import solarfl"`
