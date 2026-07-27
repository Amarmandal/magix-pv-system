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

> One row represents **one inverter at one hour**.

**Why it matters:**  
Granularity determines how the data should be aggregated, analyzed, and modeled.

## 7. Fanout

**Definition:**
A fan-out happens when one input row turns into multiple output rows during an operation like a database join or merge.

## 8. Rated Power

**Definition:**
The rated power of an inverter is the maximum continuous electrical output (measured in watts or kilowatts) that the device can supply safely and stably over a long period without overheating or shutting down

## 9. Capacity Factor

**Defnition:**
The maximum energy produced out of its theoritical limit

> Formula: Actual Energy Produced / (Rated Power \* Time Period)

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
