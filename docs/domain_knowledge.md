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
