# Lesson 8 · Unsupervised Learning on Spark

> **Week 11** | Lecture + Lab  
> Slides: [Lecture 8](https://docs.google.com/presentation/d/1ht0Bv9xpE5Ek_VCoo0u9AiGjumT2yJzbXCbyiLV1r4U/edit?usp=sharing)  
> Notebook: [Unsupervised Modeling on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture8_%20Modeling%20on%20spark%20-%20unsupervised%20learning.ipynb)

---

## Learning Objectives

- Apply k-means and bisecting k-means clustering in Spark
- Evaluate clustering quality with silhouette score and elbow method
- Implement Gaussian Mixture Models for soft clustering
- Use dimensionality reduction (PCA, t-SNE) to visualise clusters

---

## 1 · Unsupervised Learning Overview

Unsupervised learning finds structure in data **without labels**. Main tasks:

| Task | Goal | Algorithms |
|------|------|-----------|
| **Clustering** | Group similar records | K-means, GMM, DBSCAN |
| **Dimensionality reduction** | Compress features | PCA, t-SNE, UMAP |
| **Association rules** | Find co-occurrence patterns | FP-Growth, Apriori |
| **Anomaly detection** | Flag unusual records | Isolation Forest, Autoencoders |

---

## 2 · K-Means Clustering

K-means partitions data into k clusters by minimising within-cluster variance.

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Fit k-means with k=4
kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=4,
    maxIter=20,
    seed=42,
    distanceMeasure="euclidean"
)

km_model = kmeans.fit(train_df)
predictions = km_model.transform(df)

# Cluster centres
centres = km_model.clusterCenters()
for i, c in enumerate(centres):
    print(f"Cluster {i}: {c.round(3)}")
```

### Silhouette Score

```python
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.4f}")
# Range: -1 to 1. Higher is better (>0.5 is good, >0.7 is strong)
```

### Elbow Method — Choosing k

```python
import matplotlib.pyplot as plt

wssse = []   # Within-cluster Sum of Squared Errors
k_range = range(2, 12)

for k in k_range:
    km = KMeans(featuresCol="features", k=k, seed=42)
    model = km.fit(df)
    wssse.append(model.summary.trainingCost)

plt.figure(figsize=(8, 4))
plt.plot(list(k_range), wssse, "o-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WSSSE")
plt.title("Elbow Method for Optimal k")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3 · Bisecting K-Means

Bisecting k-means is a hierarchical variant that starts with one cluster and repeatedly splits the "worst" one. Often produces better clusters than regular k-means.

```python
from pyspark.ml.clustering import BisectingKMeans

bkm = BisectingKMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=5,
    maxIter=20,
    seed=42,
    minDivisibleClusterSize=20.0
)

bkm_model = bkm.fit(df)
predictions = bkm_model.transform(df)

sil = evaluator.evaluate(predictions)
print(f"Bisecting KMeans Silhouette: {sil:.4f}")
```

---

## 4 · Gaussian Mixture Model (GMM)

GMM is a **soft** clustering algorithm — each point gets a probability of belonging to each cluster. Better when clusters overlap or have different shapes.

```python
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(
    featuresCol="features",
    predictionCol="cluster",
    probabilityCol="probability",
    k=4,
    maxIter=50,
    tol=1e-4,
    seed=42,
)

gmm_model = gmm.fit(df)
predictions = gmm_model.transform(df)

# Each row now has a probability vector
predictions.select("features", "cluster", "probability").show(5, truncate=False)

# Gaussian parameters
for i, g in enumerate(gmm_model.gaussians):
    print(f"\nCluster {i}:")
    print(f"  Mean:     {g.mean.toArray().round(3)}")
    print(f"  Diagonal: {g.cov.toArray().diagonal().round(3)}")
```

---

## 5 · Visualising Clusters with PCA

```python
from pyspark.ml.feature import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Reduce to 2D for visualisation
pca = PCA(k=2, inputCol="features", outputCol="pca_2d")
pca_model = pca.fit(predictions)
df_vis = pca_model.transform(predictions)

# Sample and collect for plotting
pdf = df_vis.select("pca_2d", "cluster").limit(5000).toPandas()
pdf["x"] = pdf["pca_2d"].apply(lambda v: float(v[0]))
pdf["y"] = pdf["pca_2d"].apply(lambda v: float(v[1]))

# Plot
fig, ax = plt.subplots(figsize=(9, 7))
colours = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
for cluster_id, group in pdf.groupby("cluster"):
    ax.scatter(group["x"], group["y"],
               c=colours[cluster_id % len(colours)],
               label=f"Cluster {cluster_id}",
               alpha=0.4, s=12)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-Means Clusters (PCA 2D projection)")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 6 · Association Rule Mining (FP-Growth)

Find items that frequently appear together — the foundation of recommendation systems and market basket analysis.

```python
from pyspark.ml.fpm import FPGrowth

# Each row: a basket (list of items)
baskets = spark.createDataFrame([
    ([1, 2, 5],),
    ([2, 4],),
    ([2, 3],),
    ([1, 2, 4],),
    ([1, 3],),
    ([2, 3],),
    ([1, 3],),
    ([1, 2, 3, 5],),
    ([1, 2, 3],),
], ["items"])

fp = FPGrowth(
    itemsCol="items",
    minSupport=0.3,     # item set must appear in ≥30% of baskets
    minConfidence=0.8,  # rule A→B: P(B|A) ≥ 0.8
)

fp_model = fp.fit(baskets)

print("=== Frequent Itemsets ===")
fp_model.freqItemsets.show()

print("=== Association Rules ===")
fp_model.associationRules.show()

print("=== Predictions (complete missing items) ===")
fp_model.transform(baskets).show()
```

---

## Exercises

### Exercise 8.1 — Customer Segmentation

Using the [Big Sales Data](https://www.kaggle.com/datasets/pigment/big-sales-data):

1. Aggregate customers to the level of `(customer_id, total_spend, num_orders, avg_order_value, days_since_last_order)`
2. Scale the features
3. Find the optimal k (2–10) using both elbow method and silhouette score
4. Profile each cluster — what is the "business persona" of each segment?
5. Visualise clusters in 2D using PCA

### Exercise 8.2 — Anomaly Detection via Clustering

Use the k-means model from Exercise 8.1:

1. Compute each customer's **distance to their cluster centre**
2. Flag the top 1% of customers by distance as anomalies
3. Inspect these anomalies — do they look like high-value, dormant, or fraudulent accounts?

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors
import numpy as np

centres_broadcast = spark.sparkContext.broadcast(km_model.clusterCenters())

@udf(DoubleType())
def dist_to_centre(features, cluster):
    centre = np.array(centres_broadcast.value[cluster])
    point  = np.array(features.toArray())
    return float(np.linalg.norm(point - centre))

predictions = predictions.withColumn(
    "dist_to_centre", dist_to_centre("features", "cluster")
)

# Flag top 1% as anomalies
threshold = predictions.approxQuantile("dist_to_centre", [0.99], 0.01)[0]
anomalies = predictions.filter(col("dist_to_centre") > threshold)
```

### Exercise 8.3 — Market Basket Analysis

Using the [Google Play Store dataset](https://www.kaggle.com/gauthamp10/google-playstore-apps):

1. Treat each app's **category tags** as a basket
2. Find frequent category co-occurrences with FP-Growth (`minSupport=0.05`)
3. Find the top 10 association rules by lift
4. Interpret: which category pairs are most strongly associated?

---

## Further Reading

- [Spark MLlib Clustering](https://spark.apache.org/docs/latest/ml-clustering.html)
- [Notebook: Unsupervised Modeling on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture8_%20Modeling%20on%20spark%20-%20unsupervised%20learning.ipynb)
- *Introduction to Statistical Learning* (ISL) — Chapter 10: Unsupervised Learning ([free PDF](https://www.statlearning.com/))

---

*Previous: [Lesson 7 ← Supervised Learning](./lesson-07-supervised.md) · Next: [Lesson 9 → Use Cases](./lesson-09-use-cases.md)*
