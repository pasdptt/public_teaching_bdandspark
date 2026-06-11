# Lesson 3 · Spark DataFrame Programming

> **Week 4** | Lab  
> Slides: [Lecture 3](https://docs.google.com/presentation/d/1rpq2Rmok5lwCpKeEJ1FduUJWGXY1izro-SuMY8R8W6I/edit?usp=sharing) · Notebook: [Intro to Spark DataFrame](https://github.com/pasdptt/PasdPublicNB/blob/master/Lecture3_%20Intro2Spark_Dataframe.ipynb)

---

## Learning Objectives

- Define schemas explicitly and understand Spark data types
- Apply `map`, `flatMap`, and UDFs for custom transformations
- Understand partitioning and how to tune it for performance
- Read and write partitioned Parquet datasets
- Cache DataFrames appropriately

---

## 1 · Defining Schemas

Letting Spark infer schema from CSV is convenient but slow and error-prone on large files. Define schemas explicitly:

```python
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType, BooleanType
)

schema = StructType([
    StructField("order_id",    IntegerType(), nullable=False),
    StructField("customer_id", IntegerType(), nullable=True),
    StructField("product",     StringType(),  nullable=True),
    StructField("quantity",    IntegerType(), nullable=True),
    StructField("price",       DoubleType(),  nullable=True),
    StructField("order_date",  DateType(),    nullable=True),
    StructField("is_returned", BooleanType(), nullable=True),
])

df = (spark.read
      .schema(schema)
      .option("header", "true")
      .option("dateFormat", "yyyy-MM-dd")
      .csv("dbfs:/data/orders.csv"))

df.printSchema()
```

### Spark Type Cheat Sheet

| Python type | Spark SQL type | Spark `StructField` class |
|-------------|---------------|--------------------------|
| `int` | INT / BIGINT | `IntegerType()` / `LongType()` |
| `float` | DOUBLE | `DoubleType()` |
| `str` | STRING | `StringType()` |
| `bool` | BOOLEAN | `BooleanType()` |
| `datetime.date` | DATE | `DateType()` |
| `datetime.datetime` | TIMESTAMP | `TimestampType()` |
| `list` | ARRAY | `ArrayType(elementType)` |
| `dict` | MAP | `MapType(keyType, valueType)` |

---

## 2 · User-Defined Functions (UDFs)

When built-in functions are not enough, write a Python function and register it as a UDF.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Pure Python function
def classify_amount(amount):
    if amount is None:
        return "Unknown"
    elif amount < 500:
        return "Low"
    elif amount < 5000:
        return "Medium"
    else:
        return "High"

# Register as UDF
classify_udf = udf(classify_amount, StringType())

# Apply
df = df.withColumn("spend_tier", classify_udf(df["amount"]))
df.show(5)
```

> ⚠️ **UDFs are slow** — Spark can't optimise them and must serialize data to Python. Prefer built-in `pyspark.sql.functions` whenever possible. For complex logic, consider **Pandas UDFs** (vectorised).

### Pandas UDF (faster)

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf(StringType())
def classify_amount_fast(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: (
        "Unknown" if pd.isna(x) else
        "Low"     if x < 500   else
        "Medium"  if x < 5000  else
        "High"
    ))

df = df.withColumn("spend_tier", classify_amount_fast(df["amount"]))
```

---

## 3 · Working with Arrays & Structs

Real-world data — especially JSON — contains nested arrays and structs.

```python
from pyspark.sql.functions import (
    explode, array_contains, size,
    col, struct, array
)

# Sample nested DataFrame
data = [
    (1, "Alice", ["Python", "SQL", "Spark"]),
    (2, "Bob",   ["Java", "Scala"]),
    (3, "Carol", ["Python", "R", "SAS", "Julia"]),
]
df = spark.createDataFrame(data, ["id", "name", "skills"])

# Explode array → one row per element
df_exploded = df.select("id", "name", explode("skills").alias("skill"))
df_exploded.show()

# Filter rows where array contains a value
df.filter(array_contains("skills", "Python")).show()

# Count array length
df.withColumn("num_skills", size("skills")).show()
```

---

## 4 · Partitioning & Performance

Spark splits DataFrames into **partitions** — the unit of parallelism.

```python
# Check current partition count
print(df.rdd.getNumPartitions())

# Repartition (shuffles data — expensive)
df = df.repartition(8)

# Coalesce (merge partitions — no shuffle, only for reducing count)
df = df.coalesce(4)

# Repartition by column (useful before writing partitioned Parquet)
df = df.repartition("region")
```

### Rule of thumb for partition count

```
partitions ≈ total_data_bytes / 128_MB
# or
partitions ≈ number_of_CPU_cores × 2
```

### Write Partitioned Parquet

```python
# Partition on disk by year and region → fast predicate pushdown on reads
df.write \
  .mode("overwrite") \
  .partitionBy("year", "region") \
  .parquet("dbfs:/output/sales_partitioned/")

# Read — Spark reads only the relevant partitions (partition pruning)
df_filtered = spark.read.parquet("dbfs:/output/sales_partitioned/") \
                   .filter((col("year") == 2024) & (col("region") == "North"))
```

---

## 5 · Caching

Cache a DataFrame when you access it multiple times in the same session:

```python
df.cache()           # lazy — cached on first action
df.count()           # triggers materialisation
df.show()            # served from cache

# Check if cached
print(df.is_cached)  # True

# Unpersist when done (free memory)
df.unpersist()
```

**Storage levels:**

```python
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)        # fastest; drops if insufficient RAM
df.persist(StorageLevel.MEMORY_AND_DISK)    # spills to disk if needed (default for cache())
df.persist(StorageLevel.DISK_ONLY)          # always on disk
```

---

## 6 · The Execution Plan

Use `explain()` to see how Spark will execute your query:

```python
df.filter(col("amount") > 1000) \
  .groupBy("category") \
  .agg({"amount": "sum"}) \
  .explain(mode="formatted")
```

Look for:
- **FileScan** — did partition pruning happen?
- **HashAggregate** — aggregation in memory
- **Exchange** — shuffle (expensive; minimise these)
- **BroadcastHashJoin** — small table broadcast (fast join)

---

## Exercises

### Exercise 3.1 — Schema & Load

1. Download any CSV from [Kaggle](https://www.kaggle.com/) (≥ 50,000 rows)
2. Inspect the first 5 rows and define an explicit `StructType` schema
3. Load with your schema and confirm `df.count()` matches the expected row count
4. Report how long loading took with `inferSchema=True` vs. your explicit schema (use `%%time` in a notebook)

### Exercise 3.2 — UDF vs Built-ins

You have a column `phone_number` (string) formatted as `"0812345678"`. Write two implementations of a function that formats it as `"081-234-5678"`:

- Version A: Python UDF
- Version B: using only `pyspark.sql.functions` (`substring`, `concat_ws`, or `regexp_replace`)

Which is faster on 1M rows? Benchmark with `%%time`.

### Exercise 3.3 — Explode & Aggregate

Given a dataset of customer purchase baskets:

```python
baskets = [
    (1, "Alice", ["milk", "bread", "eggs"]),
    (2, "Bob",   ["bread", "butter"]),
    (3, "Carol", ["milk", "yogurt", "cheese", "butter"]),
    (4, "Dave",  ["eggs", "bread", "milk", "yogurt"]),
]
df = spark.createDataFrame(baskets, ["basket_id", "customer", "items"])
```

1. Find the **top 3 most frequently purchased items**
2. Find the **average basket size**
3. Find customers who bought **both "milk" and "bread"**

---

## Further Reading

- [PySpark DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
- [Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_ps.html)
- [Understanding Spark Partitions — Databricks](https://www.databricks.com/blog/2022/09/28/understanding-apache-spark-partitions.html)

---

*Previous: [Lesson 2 ← SQL & DataFrame Querying](./lesson-02-sql.md) · Next: [Lesson 4 → Algorithms for Big Data](./lesson-04-algorithms.md)*
