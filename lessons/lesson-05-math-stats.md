# Lesson 5 · Mathematics & Statistics for Big Data

> **Week 5** | Lecture + Lab  
> Slides: [Lecture 5](https://docs.google.com/presentation/d/1B9yaTQZuA0n6acvNI_Vs2K5aLKTeHl9H3D-MQzSpNoQ/edit?usp=sharing)

---

## Learning Objectives

- Apply hypothesis testing to big-data A/B experiments
- Use PCA on Spark to reduce high-dimensional data
- Perform bootstrap sampling and confidence interval estimation at scale
- Understand the special challenges statistics faces when n is very large

---

## 1 · Statistics at Scale — What Changes?

When n is very large, classical statistics behaves differently:

| Effect | Small data | Big data |
|--------|-----------|----------|
| Sampling error | High | Near zero |
| Statistical significance | Hard to achieve | Almost everything is significant |
| Practical significance | Aligned with statistical | Often diverges — need effect size |
| Computation | Fast | Needs distributed algorithms |

> **Key insight:** With 100 million rows, a difference of 0.001% will be statistically significant (p < 0.001) even if it's practically meaningless. Always report **effect sizes** alongside p-values.

---

## 2 · Descriptive Statistics on Spark

```python
# Built-in summary
df.describe().show()
df.summary("count", "mean", "stddev", "min", "25%", "75%", "max").show()

# Individual statistics
from pyspark.sql.functions import (
    mean, stddev, skewness, kurtosis,
    percentile_approx, corr
)

df.select(
    mean("amount").alias("mean"),
    stddev("amount").alias("std"),
    skewness("amount").alias("skew"),
    kurtosis("amount").alias("kurt"),
    percentile_approx("amount", [0.25, 0.50, 0.75]).alias("quartiles"),
).show()

# Pearson correlation between two columns
df.select(corr("amount", "frequency")).show()
```

---

## 3 · Hypothesis Testing

### Chi-Square Test (categorical independence)

```python
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import VectorAssembler

# Test whether 'gender' is independent of 'purchase_category'
# First encode categories as numeric
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="purchase_category", outputCol="label")
df_idx = indexer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=["gender_encoded"], outputCol="features")
df_vec = assembler.transform(df_idx)

result = ChiSquareTest.test(df_vec, "features", "label")
print(f"p-value:  {result.select('pValues').collect()[0][0]}")
print(f"statistic:{result.select('statistics').collect()[0][0]}")
```

### T-Test (manual via Spark stats)

```python
import scipy.stats as stats
import numpy as np

# Sample small random samples from each group (stratified)
# (pulling all data to driver is fine for final stat test after aggregation)

group_a = df.filter(df.group == "A").sample(fraction=0.01).select("revenue").toPandas()["revenue"]
group_b = df.filter(df.group == "B").sample(fraction=0.01).select("revenue").toPandas()["revenue"]

t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
effect_size = (group_a.mean() - group_b.mean()) / np.sqrt(
    (group_a.std()**2 + group_b.std()**2) / 2
)  # Cohen's d

print(f"t = {t_stat:.4f}, p = {p_value:.6f}")
print(f"Cohen's d = {effect_size:.4f}")
print(f"Practically significant: {abs(effect_size) > 0.2}")
```

---

## 4 · Principal Component Analysis (PCA)

PCA reduces dimensionality by finding directions of maximum variance. Essential for:
- Visualising high-dimensional data in 2D/3D
- Removing correlated features before ML
- Noise reduction

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline

# Step 1: Assemble features
feature_cols = ["age", "income", "spending_score", "purchase_frequency",
                "avg_order_value", "days_since_last_purchase"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")

# Step 2: Standardise (PCA is scale-sensitive)
scaler = StandardScaler(
    inputCol="raw_features", outputCol="scaled_features",
    withMean=True, withStd=True
)

# Step 3: PCA
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")

# Build pipeline
pipeline = Pipeline(stages=[assembler, scaler, pca])
model = pipeline.fit(df)
df_pca = model.transform(df)

# Explained variance
pca_model = model.stages[-1]
explained_var = pca_model.explainedVariance.toArray()
print(f"PC1 explains {explained_var[0]*100:.1f}% of variance")
print(f"PC2 explains {explained_var[1]*100:.1f}% of variance")
print(f"Total: {explained_var.sum()*100:.1f}%")
```

### Visualise PCA in 2D

```python
import matplotlib.pyplot as plt
import pandas as pd

# Collect PCA results (small sample only)
pdf = df_pca.select("pca_features", "label").limit(2000).toPandas()
pdf["pc1"] = pdf["pca_features"].apply(lambda v: float(v[0]))
pdf["pc2"] = pdf["pca_features"].apply(lambda v: float(v[1]))

plt.figure(figsize=(8, 6))
for label, group in pdf.groupby("label"):
    plt.scatter(group["pc1"], group["pc2"], label=label, alpha=0.5, s=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — 2D Projection")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 5 · Bootstrap Confidence Intervals

When you can't assume a distribution, use bootstrapping to estimate confidence intervals.

```python
import numpy as np

def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """
    Bootstrap confidence interval for any statistic.
    data    — 1-D numpy array
    stat_fn — function that computes the statistic (e.g., np.mean)
    """
    boot_stats = np.array([
        stat_fn(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (
        np.percentile(boot_stats, alpha * 100),
        np.percentile(boot_stats, (1 - alpha) * 100),
    )


# Pull a sample from Spark and bootstrap
sample = df.select("revenue").sample(fraction=0.005).toPandas()["revenue"].values

lo, hi = bootstrap_ci(sample, np.mean, n_boot=2000, ci=0.95)
print(f"Mean revenue: {sample.mean():.2f}")
print(f"95% CI: [{lo:.2f}, {hi:.2f}]")
```

---

## 6 · A/B Testing at Scale

```python
from pyspark.sql.functions import count, mean as spark_mean

# Aggregate by experiment group — cheap on large data
agg = df.filter(col("experiment_id") == "exp_2024_checkout") \
        .groupBy("variant") \
        .agg(
            count("*").alias("n"),
            spark_mean("converted").alias("conversion_rate"),
            spark_mean("revenue").alias("avg_revenue"),
        )

agg.show()
# +-------+--------+---------------+------------+
# |variant|       n|conversion_rate|avg_revenue |
# +-------+--------+---------------+------------+
# |control| 145230 |         0.0312|      842.11|
# |treated| 143987 |         0.0341|      917.44|
# +-------+--------+---------------+------------+
```

**Checklist before calling a winner:**

- [ ] Sufficient sample size (power analysis: n per group ≥ 10,000 typical)
- [ ] Experiment ran ≥ 1 full business week (avoid day-of-week bias)
- [ ] No novelty effect (don't end experiment in first 48 h)
- [ ] p-value < 0.05 AND Cohen's d > 0.1 (practical significance)
- [ ] No data quality issues (verify equal group sizes, check for leakage)

---

## Exercises

### Exercise 5.1 — The Multiple Testing Problem

An analyst runs 50 A/B tests simultaneously, each at significance level α = 0.05.

1. What is the probability of at least one false positive by chance?
2. What Bonferroni-corrected α should they use to keep the family-wise error rate at 5%?
3. Run a simulation in Python to verify your answer

```python
import numpy as np

def simulate_false_positives(n_tests: int, alpha: float, n_simulations: int = 10_000) -> float:
    """Returns fraction of simulations with ≥1 false positive."""
    # Under the null, p-values are uniform(0,1)
    p_values = np.random.uniform(0, 1, size=(n_simulations, n_tests))
    any_significant = np.any(p_values < alpha, axis=1)
    return any_significant.mean()

# TODO: call simulate_false_positives with n_tests=50, alpha=0.05
# TODO: find Bonferroni-corrected alpha and verify the rate drops to ~5%
```

### Exercise 5.2 — PCA on Real Data

1. Load the [Credit Card Default dataset](https://www.kaggle.com/mishra5001/credit-card)
2. Select all numeric features
3. Apply PCA and find the number of components needed to explain 90% of variance
4. Plot the cumulative explained variance curve (scree plot)

### Exercise 5.3 — Effect Size

A company reports: *"Our new recommendation engine increased click-through rate from 2.00% to 2.05%, p < 0.0001."*

1. Compute Cohen's h (effect size for proportions)
2. Is this practically significant for a business?
3. How many users did the experiment need to achieve p < 0.0001 with such a tiny difference? Use `statsmodels.stats.proportion.proportions_ztest`.

---

## Further Reading

- *Statistics Done Wrong* — Alex Reinhart ([free online](https://www.statisticsdonewrong.com/))
- [PySpark MLlib Statistics](https://spark.apache.org/docs/latest/mllib-statistics.html)
- [Notebook: PCA on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture5_%20PCA_on_spark.ipynb)
- [Notebook: Testing & Sampling on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture6_%20Testing_%26_Sampling_on%20Spark.ipynb)

---

*Previous: [Lesson 4 ← Algorithms](./lesson-04-algorithms.md) · Next: [Lesson 6 → ML for Big Data](./lesson-06-ml-bigdata.md)*
