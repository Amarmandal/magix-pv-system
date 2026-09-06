# Data acquisition and provenance

The data are not duplicated in this repository. The forecasting matrix and capacity audit use versioned files
from the following public dataset:

> Daskalov, Andrej; Zdravevski, Eftim (2026), “Dataset for unified
> photovoltaic-weather multi-station analysis in North Macedonia”, Mendeley
> Data, V2. <https://doi.org/10.17632/4zgsckxpdy.2>

The source dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The MIT license in
the repository root applies to this repository's software, not to the source
dataset.

## Exact input

| Property | Value |
| --- | --- |
| File | `hourly_pv_weather_station.csv` |
| Mendeley V2 file ID | `fac7b49d-96d8-4a4a-8d9b-4aa008683339` |
| Size | 4,130,562 bytes |
| SHA-256 | `fd504ae8ec39fad64f3b08adb546d2158c8f94f567bca03752616eb5555ddbd5` |
| Rows excluding header | 17,983 |

The Mendeley landing-page description says that the station-level view contains
13,193 rows. The downloadable V2 file contains 17,983 rows; its official API
checksum and the checksum above agree. Reproduction uses the file, not the row
count in the prose description.

## Prepare the seven clients

From the repository root, download the exact source file and partition it by
station:

```bash
uv run python scripts/prepare_data.py --download --audit-capacity
```

The command verifies the source checksum and the station IDs and row counts in
`configs/splits.json`. It then writes one byte-reproducible CSV per station to
`data/processed/client/`. If files already exist and match, it verifies them
without overwriting them. A different existing file is never overwritten
unless `--overwrite` is explicitly supplied.

To use a source file downloaded manually:

```bash
uv run python scripts/prepare_data.py \
  --source /path/to/hourly_pv_weather_station.csv
```

## Capacity inputs and deterministic reconstruction

`--audit-capacity` also verifies `devices.csv`, `stations.csv`, and
`hourly_pv_weather_inverter.csv`. Their exact V2 IDs, byte sizes, download URLs
and official SHA-256 hashes are pinned in [source_manifest.json](../docs/source_manifest.json).
Existing mismatched files fail verification and are never silently replaced.

```bash
uv run python -m solarfl.labels.capacity_audit --download
```

The audit sums metadata `max_power` (kW) for rated devices whose IDs appear in
the full released hourly inverter fact table. The term “producing” in the
historical notebook means observed membership, not positive energy or verified
commissioning. It excludes three S5 rated devices totaling 165 kW and reproduces
all seven frozen denominators. Device decisions, observation ranges and station
sums are written under `reproduced-results/manuscript_evidence_audit/` by
default; the config is only compared, never rewritten. Source-derived audit
tables retain CC BY 4.0 attribution to the dataset authors above.

The membership rule is retrospective over the full release, not training-only.
For energy units and the source's UTC interval-start convention, see
[solar_semantics.md](../docs/solar_semantics.md).
