# Lesson 1 · Hadoop & Spark Ecosystem

> **Week 2** | Lecture  
> Slides: [Lecture 1](https://docs.google.com/presentation/d/1H7AhGWjTld6n7qznNB6vw9xIrcGAtYFsVyVwfwjESxA/edit?usp=sharing) · Notebook: [Python + Spark in Colab](https://github.com/pasdptt/PasdPublicNB/blob/master/Python_Example.ipynb)

---

## Learning Objectives

- Describe the core components of the Hadoop ecosystem
- Explain how HDFS stores data across a cluster
- Compare MapReduce with Spark's in-memory execution model
- Start a SparkSession and verify the cluster is running

---

## 1 · Hadoop Ecosystem

Hadoop is an open-source framework for distributed storage and processing. Three components form its core:

```
┌─────────────────────────────────────────┐
│              YARN (Resource Manager)    │  ← schedules jobs across nodes
├─────────────────────────────────────────┤
│              MapReduce / Spark          │  ← compute layer
├─────────────────────────────────────────┤
│              HDFS                       │  ← storage layer
└─────────────────────────────────────────┘
```

### HDFS — Hadoop Distributed File System

- Files are split into **blocks** (default 128 MB each)
- Each block is replicated **3×** across different nodes
- A **NameNode** stores metadata; **DataNodes** store actual blocks

```
File: sales_2024.csv (512 MB)
  ↓  split into 4 blocks of 128 MB
  ↓  each block replicated ×3

Block 1 → Node A, Node C, Node E
Block 2 → Node B, Node D, Node F
Block 3 → Node A, Node D, Node E
Block 4 → Node B, Node C, Node F
```

### MapReduce — The Original Compute Engine

MapReduce processes data in two phases:

| Phase | What happens | Example |
|-------|-------------|---------|
| **Map** | Apply function to each record independently | tokenize each log line |
| **Reduce** | Aggregate results by key | count tokens per word |

**Limitation:** Every MapReduce step writes results back to disk → slow for iterative workloads (e.g., machine learning).

---

## 2 · Apache Spark

Spark solves MapReduce's disk I/O problem by keeping intermediate results **in memory**.

### Speed Comparison

| Operation | MapReduce | Spark |
|-----------|-----------|-------|
| Iterative ML (logistic regression, 10 iterations) | ~110× slower | baseline |
| Interactive queries | Minutes | Seconds |
| Streaming | Near-real-time (micro-batch) | Sub-second |

### Core Abstractions

```
RDD (Resilient Distributed Dataset)   ← low-level, fault-tolerant collection
  └── DataFrame                        ← tabular, schema-aware (use this!)
        └── Dataset[T]                 ← typed DataFrame (Scala/Java only)
```

> **For this course we always use DataFrames.** They are optimised by Spark's Catalyst query planner and are easier to write.

### Spark Architecture

```
Driver Program
  │
  ├── SparkContext / SparkSession
  │
  └── Cluster Manager (YARN / Kubernetes / Standalone)
        ├── Executor (Node 1) → Task 1, Task 2
        ├── Executor (Node 2) → Task 3, Task 4
        └── Executor (Node 3) → Task 5, Task 6
```

- **Driver** — your Python script; coordinates the job
- **Executor** — JVM process on a worker node; runs tasks
- **Task** — the smallest unit of work (one partition, one function)

### Lazy Evaluation

Spark does **not** run transformations immediately. It builds a DAG (Directed Acyclic Graph) of operations and only executes when an **action** is called.

```python
# Transformations (lazy — nothing runs yet)
df2 = df.filter(df.age > 25)
df3 = df2.select("name", "age")
df4 = df3.withColumn("age_group", (df3.age / 10).cast("int") * 10)

# Action (triggers execution of the entire DAG)
df4.show()          # ← NOW Spark runs everything above
df4.count()         # ← another action = another job
df4.write.parquet("output/")
```

---

## 3 · Starting Spark

### In Databricks

A `SparkSession` named `spark` is already available — no setup needed.

```python
# Verify it works
spark.version          # e.g., '3.5.0'
spark.sparkContext.defaultParallelism   # number of CPU cores available
```

### In Google Colab

```python
!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null
!pip install pyspark --quiet

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .master("local[*]")
         .appName("BigDataCourse")
         .config("spark.driver.memory", "4g")
         .getOrCreate())

print(f"Spark version: {spark.version}")
```

### Hello World in PySpark

```python
from pyspark.sql import Row

# Create a small DataFrame from Python objects
data = [
    Row(name="Alice", age=30, city="Bangkok"),
    Row(name="Bob",   age=25, city="Chiang Mai"),
    Row(name="Carol", age=35, city="Bangkok"),
    Row(name="Dave",  age=28, city="Phuket"),
]

df = spark.createDataFrame(data)
df.printSchema()
df.show()
```

**Expected output:**

```
root
 |-- name: string (nullable = true)
 |-- age: long (nullable = true)
 |-- city: string (nullable = true)

+-----+---+----------+
| name|age|      city|
+-----+---+----------+
|Alice| 30|   Bangkok|
|  Bob| 25|Chiang Mai|
|Carol| 35|   Bangkok|
| Dave| 28|    Phuket|
+-----+---+----------+
```

---

## 4 · Transformations vs. Actions — Quick Reference

### Transformations (lazy)

```python
df.select("col1", "col2")          # choose columns
df.filter(df.col > value)          # row filter
df.withColumn("new", expr)         # add/replace column
df.groupBy("key").agg(...)         # aggregation
df.join(other, on="id")            # join two DataFrames
df.orderBy("col")                  # sort
df.limit(100)                      # first N rows
df.distinct()                      # deduplicate
```

### Actions (eager — triggers execution)

```python
df.show(n=20)                      # print rows
df.collect()                       # pull all rows to driver (careful!)
df.count()                         # row count
df.first()                         # first row
df.toPandas()                      # convert to pandas (small data only)
df.write.parquet("path/")          # write to storage
df.write.csv("path/", header=True)
```

---

## Exercises

### Exercise 1.1 — Architecture Diagram

Draw (or describe in text) the flow of data when a user runs `df.count()` on a Spark cluster with 4 worker nodes. Include: driver, executors, tasks, and the final result.

### Exercise 1.2 — Your First DataFrame

```python
# TODO: Create a DataFrame with at least 6 rows representing students.
# Columns: student_id (int), name (str), score_midterm (float), score_final (float)
# Then:
#   1. Filter to students with midterm score > 60
#   2. Add a column `total` = midterm * 0.20 + final * 0.35
#   3. Sort by `total` descending
#   4. Show the result

# Write your solution below
students = [...]   # fill this in
```

### Exercise 1.3 — Lazy Evaluation

Explain in your own words: why does this code print **nothing** until `.show()` is called?

```python
df2 = df.filter(df.score > 50)     # (a)
df3 = df2.withColumn("grade",      # (b)
          (df2.score / 10).cast("int"))
df3.show()                         # (c)
```

---

## Further Reading

- [Spark Architecture Deep Dive — Databricks](https://www.databricks.com/glossary/apache-spark-architecture)
- [PySpark Getting Started — Official Docs](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)
- *Learning Spark* (2nd ed.) — Chapters 1–3

---

*Previous: [Lesson 0 ← Introduction](./lesson-00-introduction.md) · Next: [Lesson 2 → SQL & DataFrames](./lesson-02-sql.md)*
