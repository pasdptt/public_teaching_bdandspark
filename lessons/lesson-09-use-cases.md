# Lesson 9 · Use Cases — Telco, IoT, Retail & Banking

> **Weeks 12–14** | Case Studies  
> Slides: [Lecture 9 — RFM & Segmentation](https://docs.google.com/presentation/d/1lH-vy3kyBeW5bL_bd0pCK57Foa8nrfuPEeNRvscj5iY/edit?usp=sharing)  
> Notebook: [Customer Segmentation](https://github.com/pasdptt/PasdPublicNB/blob/master/Customer_segmentation.ipynb)

---

## Learning Objectives

- Map big data techniques to concrete industry problems
- Implement an end-to-end RFM customer segmentation pipeline
- Build a churn predictor for a Telco dataset
- Apply anomaly detection to IoT sensor streams
- Design a product recommendation system for retail

---

## Use Case 1 · Telco — Churn Prediction

### Business Context

A mobile operator with 5 million customers loses ~2% monthly to churn. Retaining a customer costs 5× less than acquiring a new one. The goal: predict churners 30 days in advance so marketing can intervene.

### Data Sources

```
subscribers(subscriber_id, plan, tenure_months, monthly_charge, contract_type)
cdr(call_date, subscriber_id, duration_sec, direction, call_type)  ← call detail records
complaints(ticket_id, subscriber_id, created_at, resolved_at, category)
payments(payment_id, subscriber_id, payment_date, amount, is_late)
```

### Feature Engineering

```python
from pyspark.sql.functions import (
    count, sum as spark_sum, avg, datediff,
    max as spark_max, min as spark_min,
    when, col, lit
)
from pyspark.sql.window import Window

# ── CDR features (last 30 days) ──────────────────────────────────────────
cdr_features = (
    cdr.filter(col("call_date") >= "2024-11-01")
       .groupBy("subscriber_id").agg(
           count("*").alias("n_calls_30d"),
           spark_sum("duration_sec").alias("total_duration_30d"),
           avg("duration_sec").alias("avg_duration_30d"),
           spark_sum(when(col("direction") == "outbound", 1).otherwise(0))
               .alias("n_outbound_30d"),
       )
)

# ── Complaint features ───────────────────────────────────────────────────
complaint_features = (
    complaints.groupBy("subscriber_id").agg(
        count("*").alias("n_complaints"),
        avg(datediff("resolved_at", "created_at")).alias("avg_resolution_days"),
    )
)

# ── Payment features ─────────────────────────────────────────────────────
payment_features = (
    payments.groupBy("subscriber_id").agg(
        spark_sum(when(col("is_late") == True, 1).otherwise(0)).alias("late_payments"),
        avg("amount").alias("avg_payment"),
    )
)

# ── Join all features ────────────────────────────────────────────────────
features_df = (
    subscribers
    .join(cdr_features,       on="subscriber_id", how="left")
    .join(complaint_features, on="subscriber_id", how="left")
    .join(payment_features,   on="subscriber_id", how="left")
    .fillna(0)
)
```

### Model & Outcome

```python
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Assume churn label is already available for training period
gbt = GBTClassifier(featuresCol="features", labelCol="churn", maxIter=50, maxDepth=5)
model = pipeline.fit(train_df)
preds = model.transform(test_df)

auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(preds)
print(f"AUC-ROC: {auc:.4f}")

# Score current subscribers — rank by churn probability
at_risk = (
    model.transform(active_subscribers)
         .select("subscriber_id", "probability", "prediction")
         .withColumn("churn_prob", col("probability").getItem(1))
         .orderBy(col("churn_prob").desc())
         .limit(10_000)   # top 10k most at-risk customers for campaign
)
at_risk.show(10)
```

---

## Use Case 2 · IoT — Sensor Anomaly Detection

### Business Context

A factory deploys 500 vibration sensors on machines. Unusual vibration patterns precede equipment failure by 6–48 hours. The goal: detect anomalies in real time to enable predictive maintenance.

### Data Schema

```
sensor_readings(
    sensor_id      STRING,
    machine_id     STRING,
    timestamp      TIMESTAMP,
    vibration_x    DOUBLE,
    vibration_y    DOUBLE,
    vibration_z    DOUBLE,
    temperature    DOUBLE,
    rpm            DOUBLE,
)
```

### Feature Engineering — Rolling Statistics

```python
from pyspark.sql.functions import (
    avg, stddev, min as spark_min, max as spark_max,
    col, window
)

# Rolling 5-minute aggregation per sensor
rolling = (
    sensor_readings
    .groupBy("sensor_id", window("timestamp", "5 minutes", "1 minute"))
    .agg(
        avg("vibration_x").alias("vib_x_mean"),
        stddev("vibration_x").alias("vib_x_std"),
        avg("temperature").alias("temp_mean"),
        spark_max("rpm").alias("rpm_max"),
    )
    .withColumn("window_start", col("window.start"))
    .drop("window")
)
```

### Isolation Forest via PySpark (using sklearn in a Pandas UDF)

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from sklearn.ensemble import IsolationForest

feature_cols = ["vib_x_mean", "vib_x_std", "temp_mean", "rpm_max"]

@pandas_udf(DoubleType())
def anomaly_score(*cols) -> pd.Series:
    """Compute Isolation Forest anomaly score per sensor batch."""
    X = pd.concat(cols, axis=1)
    X.columns = feature_cols
    X = X.fillna(X.median())

    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X)
    # score_samples: higher = more normal; negate so higher = more anomalous
    return pd.Series(-iso.score_samples(X))


rolling = rolling.withColumn(
    "anomaly_score",
    anomaly_score(*[col(c) for c in feature_cols])
)

# Alert on high-anomaly windows
alerts = rolling.filter(col("anomaly_score") > 0.6) \
                .orderBy(col("anomaly_score").desc())
alerts.show(10)
```

---

## Use Case 3 · Retail & Banking — RFM + Customer Segmentation

### Business Context

An e-commerce platform wants to personalise marketing spend. High-value customers deserve retention campaigns; dormant customers need re-activation offers; new customers need onboarding nudges.

### RFM Framework

| Dimension | Question | Measure |
|-----------|---------|---------|
| **Recency** | How recently did they buy? | Days since last purchase |
| **Frequency** | How often do they buy? | Number of orders in period |
| **Monetary** | How much do they spend? | Total or average order value |

```python
from pyspark.sql.functions import (
    datediff, lit, count, spark_sum, current_date,
    percentile_approx, col, ntile
)
from pyspark.sql.window import Window

SNAPSHOT_DATE = "2024-12-31"

rfm = (
    orders
    .filter(col("order_date") <= SNAPSHOT_DATE)
    .groupBy("customer_id")
    .agg(
        datediff(lit(SNAPSHOT_DATE), spark_max("order_date")).alias("recency"),
        count("order_id").alias("frequency"),
        spark_sum("order_value").alias("monetary"),
    )
)

# Score R/F/M into quintiles (1=worst, 5=best)
# Note: for Recency, lower days = better → reverse ranking
w = Window.orderBy(col("recency").asc())    # low recency = recent = 5
rfm = rfm.withColumn("R", 6 - ntile(5).over(w))  # invert: 1 day → score 5

w_f = Window.orderBy(col("frequency").asc())
rfm = rfm.withColumn("F", ntile(5).over(w_f))

w_m = Window.orderBy(col("monetary").asc())
rfm = rfm.withColumn("M", ntile(5).over(w_m))

rfm = rfm.withColumn("RFM_score", col("R") + col("F") + col("M"))

rfm.show(10)
```

### Segment Mapping

```python
from pyspark.sql.functions import when

rfm = rfm.withColumn("segment",
    when((col("R") >= 4) & (col("F") >= 4),         "Champions")
   .when((col("R") >= 3) & (col("F") >= 3),         "Loyal Customers")
   .when((col("R") >= 4) & (col("F") <= 2),         "Recent Customers")
   .when((col("R") >= 3) & (col("M") >= 4),         "Potential Loyalists")
   .when((col("R") <= 2) & (col("F") >= 4),         "At Risk")
   .when((col("R") <= 2) & (col("F") >= 2),         "Hibernating")
   .when((col("R") == 1) & (col("F") == 1),         "Lost")
   .otherwise("Needs Attention")
)

# Segment distribution
rfm.groupBy("segment").count().orderBy(col("count").desc()).show()
```

### Product Recommendation (Collaborative Filtering)

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# ALS requires integer IDs
als = ALS(
    userCol="customer_id_int",
    itemCol="product_id_int",
    ratingCol="purchase_count",   # implicit feedback: times purchased
    implicitPrefs=True,           # implicit (purchase count) vs. explicit (rating)
    rank=20,                      # latent factors
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",     # handle new users/items without crashing
    seed=42,
)

als_model = als.fit(train_df)

# Generate top-5 product recommendations for every customer
recs = als_model.recommendForAllUsers(5)
recs.show(5, truncate=False)
```

---

## Use Case 4 · Banking — Fraud Detection

### Business Context

A bank processes 10 million card transactions per day. ~0.01% are fraudulent. The cost of a missed fraud is ~฿5,000; the cost of a false positive (blocking a legitimate transaction) is customer dissatisfaction and ~฿200 in support cost.

### Key Design Decisions

```python
# 1. Optimise for recall (catch frauds), not accuracy
#    → use precision-recall AUC, not ROC AUC

# 2. Set decision threshold based on cost matrix
cost_matrix = {
    "TP": 5000,   # fraud caught: save ฿5,000
    "FP": -200,   # false alarm: cost ฿200
    "FN": -5000,  # missed fraud: lose ฿5,000
    "TN": 0,      # correct approval: no cost
}

def expected_value(threshold, preds_pdf):
    preds_pdf["pred"] = (preds_pdf["score"] >= threshold).astype(int)
    tp = ((preds_pdf["label"] == 1) & (preds_pdf["pred"] == 1)).sum()
    fp = ((preds_pdf["label"] == 0) & (preds_pdf["pred"] == 1)).sum()
    fn = ((preds_pdf["label"] == 1) & (preds_pdf["pred"] == 0)).sum()
    tn = ((preds_pdf["label"] == 0) & (preds_pdf["pred"] == 0)).sum()
    return (tp * cost_matrix["TP"] + fp * cost_matrix["FP"] +
            fn * cost_matrix["FN"] + tn * cost_matrix["TN"])

# 3. Feature engineering: velocity features
velocity = (
    transactions
    .withWatermark("txn_time", "1 hour")
    .groupBy("card_id", window("txn_time", "1 hour"))
    .agg(
        count("*").alias("txn_count_1h"),
        spark_sum("amount").alias("total_amount_1h"),
        count_distinct("merchant_country").alias("n_countries_1h"),
    )
)
```

---

## Final Project Checklist

Your project report should answer these questions:

- [ ] **Problem statement** — what business question are you solving?
- [ ] **Data description** — source, size, schema, quality issues found
- [ ] **EDA** — distributions, correlations, class balance, missing values
- [ ] **Feature engineering** — what features did you create and why?
- [ ] **Modelling** — which algorithms did you try? How did you evaluate them?
- [ ] **Results** — what were the key findings? Include metrics and visualisations
- [ ] **Business recommendation** — what action should the business take?
- [ ] **Limitations** — what could be improved with more time or data?

---

## Exercises

### Exercise 9.1 — Telco Churn End-to-End

Using the [Telco Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn):

1. Run the full pipeline: EDA → feature engineering → modelling → evaluation
2. Calculate the **expected monthly revenue saved** if the model is used to target the top 500 at-risk customers with a ฿300 retention incentive
3. Write a one-page executive summary of your findings

### Exercise 9.2 — RFM Segmentation

Using the [Big Sales Data](https://www.kaggle.com/datasets/pigment/big-sales-data):

1. Compute RFM scores and assign segments
2. For each segment, compute mean revenue per customer per month
3. Design a marketing action for each segment (in a table)
4. Estimate the total addressable revenue if all "At Risk" customers are recovered

### Exercise 9.3 — Anomaly Detection

Using the [APS Failure at Scania Trucks](https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set):

1. Apply k-means clustering and flag outliers by distance to cluster centre
2. Compare flagged outliers with the ground-truth failure labels
3. What precision and recall does your unsupervised approach achieve?

---

## Further Reading

- [Notebook: Customer Segmentation (RFM)](https://github.com/pasdptt/PasdPublicNB/blob/master/Customer_segmentation.ipynb)
- [ALS Collaborative Filtering — Spark Docs](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [Interpretable Machine Learning — Christoph Molnar (free book)](https://christophm.github.io/interpretable-ml-book/)
- [Responsible AI Practices — Google](https://ai.google/responsibilities/responsible-ai-practices/)

---

*Previous: [Lesson 8 ← Unsupervised Learning](./lesson-08-unsupervised.md) · Back to [README](../README.md)*
