# Lesson 7 · Supervised Learning on Spark

> **Week 10–11** | Lecture + Lab  
> Slides: [Lecture 7](https://docs.google.com/presentation/d/1zoMutn4idkDkuGurtgH4LizOqRZe8R7J5euLTv2hO1U/edit?usp=sharing)  
> Notebook: [Supervised Modeling on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture7_%20Modeling%20on%20Spark_%20supervised%20learning.ipynb)

---

## Learning Objectives

- Implement classification and regression with Spark MLlib
- Compare linear models, tree-based models, and gradient boosting
- Interpret model outputs (feature importance, coefficients)
- Serve predictions at scale

---

## 1 · Algorithm Map

```
Supervised Learning
├── Classification (predict a category)
│   ├── Logistic Regression          → binary / multi-class, interpretable
│   ├── Decision Tree                → interpretable, non-linear
│   ├── Random Forest                → robust, handles missing values
│   ├── Gradient Boosted Trees (GBT) → highest accuracy, slower to train
│   └── Linear SVM                  → fast, high-dimensional text
│
└── Regression (predict a number)
    ├── Linear Regression            → interpretable baseline
    ├── Decision Tree Regressor
    ├── Random Forest Regressor
    └── GBT Regressor                → often best accuracy
```

---

## 2 · Logistic Regression

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,        # L2 regularisation strength
    elasticNetParam=0.0,  # 0=Ridge, 1=Lasso
    threshold=0.5,        # decision boundary
)

lr_model = lr.fit(train_df)
predictions = lr_model.transform(test_df)

# Coefficients
import pandas as pd
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coefficients.toArray()
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.head(10))

# AUC
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"AUC: {evaluator.evaluate(predictions):.4f}")
```

---

## 3 · Random Forest

Random Forest trains many decision trees on random subsets of data and features, then aggregates their predictions (bagging + feature randomness).

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,           # more trees = better but slower
    maxDepth=8,
    minInstancesPerNode=10, # minimum records per leaf
    featureSubsetStrategy="sqrt",  # features per split: sqrt(p) for classification
    seed=42,
)

rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)

# Feature importance
import matplotlib.pyplot as plt
import numpy as np

importances = rf_model.featureImportances.toArray()
feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.xlabel("Importance (Gini impurity reduction)")
plt.title("Random Forest — Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

## 4 · Gradient Boosted Trees (GBT)

GBT trains trees **sequentially** — each tree corrects the errors of the previous one. Usually the most accurate algorithm for tabular data.

```python
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=50,       # number of boosting rounds
    maxDepth=5,       # shallow trees work well for GBT
    stepSize=0.1,     # learning rate — lower is more robust but needs more rounds
    subsamplingRate=0.8,  # row sampling per tree (reduces overfitting)
    seed=42,
)

gbt_model = gbt.fit(train_df)
predictions = gbt_model.transform(test_df)

print(f"AUC: {evaluator.evaluate(predictions):.4f}")
```

> **GBT limitation:** Spark's `GBTClassifier` supports binary classification only. For multi-class, use `RandomForestClassifier` or `OneVsRest`.

---

## 5 · Multi-Class Classification

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# RF supports multi-class natively
rf_multi = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",       # label must be 0, 1, 2, ... (integer)
    numTrees=100,
    maxDepth=6,
)

rf_multi_model = rf_multi.fit(train_df)
preds = rf_multi_model.transform(test_df)

mc_eval = MulticlassClassificationEvaluator(labelCol="label")
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    mc_eval.setMetricName(metric)
    print(f"{metric:20s}: {mc_eval.evaluate(preds):.4f}")
```

---

## 6 · Regression

```python
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator

# Linear Regression
lr_reg = LinearRegression(
    featuresCol="features",
    labelCol="revenue",
    maxIter=100,
    regParam=0.01,
)
lr_reg_model = lr_reg.fit(train_df)

# Evaluation
reg_eval = RegressionEvaluator(labelCol="revenue", predictionCol="prediction")
for metric in ["rmse", "mae", "r2"]:
    reg_eval.setMetricName(metric)
    print(f"{metric}: {reg_eval.evaluate(lr_reg_model.transform(test_df)):.4f}")

# Summary
summary = lr_reg_model.summary
print(f"R²:   {summary.r2:.4f}")
print(f"RMSE: {summary.rootMeanSquaredError:.4f}")
```

---

## 7 · Model Comparison

```python
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=100),
    "Random Forest":       RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50),
    "GBT":                 GBTClassifier(featuresCol="features", labelCol="label", maxIter=30),
}

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
results = []

for name, model in models.items():
    t0 = time.time()
    fitted = model.fit(train_df)
    preds = fitted.transform(test_df)
    auc = evaluator.evaluate(preds)
    elapsed = time.time() - t0
    results.append({"Model": name, "AUC": round(auc, 4), "Train time (s)": round(elapsed, 1)})

results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
print(results_df.to_string(index=False))
```

---

## Exercises

### Exercise 7.1 — Churn Prediction

Using the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn):

1. Perform EDA: churn rate, feature distributions
2. Build a feature engineering pipeline (encode categoricals, impute, scale)
3. Train all three classifiers (LR, RF, GBT) and compare AUC
4. For the best model, plot the ROC curve
5. Find the optimal decision threshold to maximise F1

```python
# ROC curve helper
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Collect probability scores from Spark (use a sample if large)
pdf = predictions.select("label", "probability").limit(50_000).toPandas()
pdf["score"] = pdf["probability"].apply(lambda v: float(v[1]))

fpr, tpr, thresholds = roc_curve(pdf["label"], pdf["score"])
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

### Exercise 7.2 — House Price Regression

Using the [Big Sales Data](https://www.kaggle.com/datasets/pigment/big-sales-data):

1. Build a regression pipeline to predict `price`
2. Compare Linear Regression, Random Forest Regressor, and GBT Regressor using RMSE and R²
3. Plot predicted vs. actual values for the best model
4. Analyse residuals — are they normally distributed?

### Exercise 7.3 — Interpretability

For your best classification model from Exercise 7.1:

1. List the top 10 most important features
2. For Logistic Regression: which features increase churn probability the most?
3. Write a 200-word business summary of your findings suitable for a non-technical manager

---

## Further Reading

- [MLlib Classification and Regression — Spark Docs](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [Notebook: Supervised Modeling on Spark](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture7_%20Modeling%20on%20Spark_%20supervised%20learning.ipynb)
- *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* — Chapters 3–7

---

*Previous: [Lesson 6 ← ML Pipelines](./lesson-06-ml-bigdata.md) · Next: [Lesson 8 → Unsupervised Learning](./lesson-08-unsupervised.md)*
