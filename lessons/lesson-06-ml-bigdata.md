# Lesson 6 · ML for Big Data — PCA, Feature Engineering & Pipelines

> **Week 6–7** | Lecture + Lab  
> Slides: [Lecture 6](https://docs.google.com/presentation/d/1_xyojlGvXhHEUmDVAZgCwQfhm-JZ2n7HxQwQ2I0cNME/edit?usp=sharing)  
> Notebooks: [PCA on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture5_%20PCA_on_spark.ipynb) · [Testing & Sampling](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture6_%20Testing_%26_Sampling_on%20Spark.ipynb)

---

## Learning Objectives

- Build end-to-end ML pipelines with Spark MLlib
- Encode categorical features correctly
- Handle class imbalance at scale
- Evaluate models with appropriate metrics
- Tune hyperparameters using cross-validation

---

## 1 · Spark MLlib Overview

Spark MLlib provides two APIs:

| API | Package | Status |
|-----|---------|--------|
| **DataFrame-based (ML)** | `pyspark.ml` | ✅ Use this |
| RDD-based (MLlib) | `pyspark.mllib` | ⚠️ Legacy |

### Core abstractions

- **Transformer** — takes a DataFrame and produces a new DataFrame (e.g., `StandardScaler`, `StringIndexer`)
- **Estimator** — fits on data to produce a Transformer (e.g., `LogisticRegression.fit()`)
- **Pipeline** — chains Transformers and Estimators into a reproducible workflow

---

## 2 · Feature Engineering

### Encoding Categorical Variables

```python
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, MinMaxScaler, Imputer
)

# 1. StringIndexer: string → integer index
indexer = StringIndexer(
    inputCols=["city", "gender", "category"],
    outputCols=["city_idx", "gender_idx", "category_idx"],
    handleInvalid="keep"      # handle unseen categories at predict time
)

# 2. OneHotEncoder: integer index → sparse binary vector
ohe = OneHotEncoder(
    inputCols=["city_idx", "gender_idx", "category_idx"],
    outputCols=["city_ohe", "gender_ohe", "category_ohe"],
    dropLast=True             # avoid multicollinearity
)
```

### Handling Missing Values

```python
# Numeric: impute with mean/median/mode
imputer = Imputer(
    inputCols=["age", "income", "tenure"],
    outputCols=["age_imp", "income_imp", "tenure_imp"],
    strategy="median"
)

# Categorical: fill before indexing
df = df.fillna({"city": "Unknown", "gender": "Unknown"})
```

### Assembling the Feature Vector

```python
feature_cols = [
    "age_imp", "income_imp", "tenure_imp",  # numeric
    "city_ohe", "gender_ohe", "category_ohe",  # encoded categoricals
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)
```

---

## 3 · Building a Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer as LabelIndexer

# Label encoding
label_indexer = LabelIndexer(inputCol="churn", outputCol="label")

# Model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5      # 0=Ridge, 1=Lasso, 0.5=ElasticNet
)

# Full pipeline
pipeline = Pipeline(stages=[
    indexer,       # StringIndexer for categoricals
    ohe,           # OneHotEncoder
    imputer,       # Imputer for numerics
    assembler,     # VectorAssembler
    scaler,        # StandardScaler
    label_indexer, # Label encoding
    lr             # Model
])

# Train / test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Fit the entire pipeline on training data
model = pipeline.fit(train_df)

# Predict on test data
predictions = model.transform(test_df)
predictions.select("label", "prediction", "probability").show(10)
```

---

## 4 · Model Evaluation

```python
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# AUC-ROC (binary classification)
auc_eval = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
)
auc = auc_eval.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Accuracy, F1, Precision, Recall
mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    mc_eval.setMetricName(metric)
    print(f"{metric:20s}: {mc_eval.evaluate(predictions):.4f}")
```

### Confusion Matrix (manual)

```python
from pyspark.sql.functions import col

# Collect confusion matrix values
tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)

print(f"Confusion Matrix:")
print(f"              Predicted 0    Predicted 1")
print(f"Actual 0:     {tn:12,}   {fp:12,}")
print(f"Actual 1:     {fn:12,}   {tp:12,}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
```

---

## 5 · Class Imbalance

Most real datasets are imbalanced (e.g., fraud = 0.1% of transactions). Strategies:

### Oversampling (SMOTE-like via pandas_udf)

```python
# Simple oversampling: upsample minority class
majority = df.filter(col("label") == 0)
minority = df.filter(col("label") == 1)

ratio = majority.count() / minority.count()
oversampled = minority.sample(withReplacement=True, fraction=ratio, seed=42)

balanced_df = majority.unionAll(oversampled)
print(f"Balanced class distribution:")
balanced_df.groupBy("label").count().show()
```

### Class Weights in the Model

```python
from pyspark.sql.functions import when

# Compute class weights
n_total = df.count()
n_pos = df.filter(col("label") == 1).count()
n_neg = n_total - n_pos

df = df.withColumn("weight",
    when(col("label") == 1, n_total / (2 * n_pos))
   .otherwise(n_total / (2 * n_neg))
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight"   # ← use class weights
)
```

---

## 6 · Hyperparameter Tuning

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

param_grid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.001, 0.01, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .build())

cross_val = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=auc_eval,
    numFolds=5,
    parallelism=4      # run folds in parallel
)

cv_model = cross_val.fit(train_df)

# Best parameters
best_lr = cv_model.bestModel.stages[-1]
print(f"Best regParam:         {best_lr.getRegParam()}")
print(f"Best elasticNetParam:  {best_lr.getElasticNetParam()}")
print(f"Best AUC (CV):         {max(cv_model.avgMetrics):.4f}")
```

---

## Exercises

### Exercise 6.1 — Feature Engineering Pipeline

Using the [Smart Meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london/data) dataset:

1. Parse the timestamp and extract `hour`, `day_of_week`, `month`, `is_weekend`
2. Handle missing values in energy readings
3. Compute per-household rolling 7-day average energy consumption
4. Assemble a feature vector and standardise it

### Exercise 6.2 — Imbalanced Classification

Using the [Credit Card Fraud](https://www.kaggle.com/datasets/joebeachcapital/credit-card-fraud) dataset (~0.17% fraud rate):

1. Train a baseline Logistic Regression **without** addressing imbalance
2. Train with class weights
3. Train with oversampling of the minority class
4. Compare AUC, precision, recall for all three approaches
5. Which metric matters most for fraud detection? Why?

### Exercise 6.3 — Pipeline Serialisation

1. Train any pipeline from Exercises 6.1–6.2
2. Save the trained pipeline to Databricks DBFS: `model.write().overwrite().save("dbfs:/models/my_pipeline")`
3. Load it back: `from pyspark.ml import PipelineModel; loaded = PipelineModel.load("...")`
4. Verify predictions match

---

## Further Reading

- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Feature Engineering for Machine Learning — Alice Zheng (O'Reilly)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Imbalanced Classification with PySpark — Databricks](https://docs.databricks.com/machine-learning/train-model/imbalanced-data.html)

---

*Previous: [Lesson 5 ← Math & Statistics](./lesson-05-math-stats.md) · Next: [Lesson 7 → Supervised Learning](./lesson-07-supervised.md)*
