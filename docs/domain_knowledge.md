# Most Important Technical Terms

These are the key domain concepts to understand before performing EDA or building machine learning models.

---

## 1. Irradiance

**Definition:**  
The amount of solar energy reaching a surface at a given moment.

**Unit:**  
`W/m²` (Watts per square meter)

**Why it matters:**  
Irradiance is one of the strongest predictors of solar energy production. More sunlight generally leads to higher power generation.

---

## 2. Efficiency

**Definition:**  
Measures how efficiently an inverter converts **DC power into AC power**.

**Formula:**

```text
Efficiency = Useful Output / Input
```

**Why it matters:**  
Lower efficiency may indicate inverter degradation or faults and directly impacts energy production.

---

## 3. Time Series

**Definition:**  
The observations are ordered by time and are **not independent**.

**Granularity:**  
Hourly

**Why it matters:**

- Previous observations influence future observations.
- Daily and seasonal patterns are important.
- Data should be split chronologically, not randomly.

---

## 4. Tilt

**Definition:**  
The angle of the solar panel relative to the ground.

**Why it matters:**  
Tilt determines how much sunlight the panel receives throughout the day and year.

---

## 5. Azimuth

**Definition:**  
The orientation (direction) the solar panel faces.

**Why it matters:**  
Panel orientation affects the amount of solar radiation captured during the day.

---

## 6. Granularity

**Definition:**  
The level of detail represented by each row.

**Granularity of this dataset:**

> One row represents **one station at one hour**.

Inverter-level data exists in the source dataset (`hourly_pv_weather_inverter.csv`),
but D-001 selected the station-level fact table, so the working grain is
(`station_hash_id`, `measured_ts`).

**Why it matters:**  
Granularity determines how the data should be aggregated, analyzed, and modeled.

## 7. Fanout

**Definition:**
A fan-out happens when one input row turns into multiple output rows during an operation like a database join or merge.

## 8. Rated Power

**Definition:**
The rated power of an inverter is the maximum continuous electrical output (measured in watts or kilowatts) that the device can supply safely and stably over a long period without overheating or shutting down

## 9. Capacity Factor

**Definition:**
The fraction of its theoretical maximum that a station actually produced.

> Formula: Actual Energy Produced / (Rated Power \* Time Period)

## 10. Baseline Models ⭐⭐⭐⭐⭐

### Core Idea

Every ML project starts with a **simple baseline**.

The purpose is to answer:

> "Is my model actually better than a simple solution?"

### Examples

| Problem                 | Baseline                    |
| ----------------------- | --------------------------- |
| Regression              | Predict the mean            |
| Classification          | Predict the majority class  |
| Time Series Forecasting | Persistence                 |

### Example

PV Forecasting:

```text
Current Production = 120 kWh
Persistence Prediction (Next Hour) = 120 kWh
```

Without a baseline, model performance has no context.

---

## 11. Skill Score ⭐⭐⭐⭐⭐

### Core Idea

Measures how much better your model is than the baseline.

Formula:

```text
Skill = 1 - MAE_model / MAE_baseline
```

Interpretation:

| Skill | Meaning              |
| ----- | -------------------- |
| < 0   | Worse than baseline  |
| 0     | Same as baseline     |
| 0.15  | 15% better           |
| 0.30  | 30% better           |
| 1.0   | Perfect prediction   |

### Lesson

Never ask:

> "Is MAE = 0.1 good?"

Instead ask:

> "How much better is my model than the baseline?"

---

## 12. Oracle / Upper Bound ⭐⭐⭐⭐⭐

### Core Idea

Determine the **best possible performance** under ideal conditions.

Example:

| Setup                    | Skill |
| ------------------------ | ----- |
| Past weather             | 0.157 |
| Perfect weather (oracle) | 0.286 |

Meaning:

Your model has already achieved

```text
0.157 / 0.286 ≈ 55%
```

of the maximum possible improvement.

### Lesson

The oracle tells you whether improving the model is worthwhile.

---

## 13. Bottleneck Analysis ⭐⭐⭐⭐⭐

### Core Idea

Instead of asking

> "How do I improve the model?"

Ask

> "What is actually limiting performance?"

Possible bottlenecks:

- Poor features
- Low-quality labels
- Bad weather forecasts
- Insufficient data
- Model capacity
- Optimization

### Lesson

Find the bottleneck before improving the model.

---

## 14. Error Analysis ⭐⭐⭐⭐⭐

### Core Idea

Don't just compute one number.

Investigate:

- Where does the model fail?
- Which stations?
- Which seasons?
- Which weather conditions?
- Sunrise?
- Sunset?
- Cloudy days?

### Lesson

Most model improvements come from understanding failures.

---

### DK-003 :- Interpretation of `terrestrial_radiation`

**Question**

Does the `terrestrial_radiation` feature represent Earth's emitted longwave infrared radiation or solar radiation?

**Hypotheses**

- H1: Solar radiation → should approach zero during nighttime.
- H2: Longwave terrestrial radiation → should remain substantial during both day and night.

**Evidence**

Nighttime statistics:

- mean = 0.1 W/m²
- max = 25.1 W/m²
- 97.4% of observations are exactly zero

Daytime statistics:

- mean = 597.2 W/m²
- max = 1254.1 W/m²

**Conclusion**

The observed behavior strongly matches solar radiation rather than continuous terrestrial longwave infrared emission. The feature name appears misleading or inconsistent with the accompanying documentation.

### DK-006 :- Verification of Physics-Derived Features

Two derived features were validated using the observed data.  
Notebook: `04_scope-features.ipynb`.  
Cell: 5

**Recovered solar geometry**

> **Superseded by D-008.** The formula below is no longer the implementation in
> use. It is kept as the record of the original validation, and as the baseline
> the replacement was checked against.

cos(zenith) = direct_radiation / direct_normal_irradiance

Observed statistics:

- min = 0.058
- max = 0.946

All recovered values lie within the physically valid interval [0,1], confirming the feature behaves as expected.

**Recovered clearness index**

kt = shortwave_radiation / terrestrial_radiation

Observed statistics:

- mean = 0.477
- range = [0.014, 0.791]

The distribution matches the expected behavior of a clearness index, providing additional evidence that `terrestrial_radiation` behaves as top-of-atmosphere solar irradiance rather than terrestrial longwave radiation.
