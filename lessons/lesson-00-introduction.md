# Lesson 0 · Introduction to Big Data

> **Week 1** | Lecture  
> Slides: [Lecture 0](https://docs.google.com/presentation/d/1LA_PkymDFetWhc33ssI0E4dhgFPqdcXMSxVmJRvsTZE/edit?usp=sharing)

---

## Learning Objectives

By the end of this lesson you will be able to:

- Define "big data" using the 5 Vs framework
- Distinguish between structured, semi-structured, and unstructured data
- Describe the high-level architecture of a distributed data system
- Explain why traditional tools break at scale and what replaces them

---

## 1 · What Is Big Data?

Big data refers to datasets that are too large, fast-moving, or varied to be processed efficiently by traditional relational databases or single-machine tools.

### The 5 Vs

| V | Meaning | Example |
|---|---------|---------|
| **Volume** | Scale of data | Petabytes of sensor logs |
| **Velocity** | Speed of generation | 500,000 tweets/minute |
| **Variety** | Diversity of formats | CSV + JSON + images + video |
| **Veracity** | Trustworthiness of data | Noisy IoT sensors, missing fields |
| **Value** | Business utility extracted | Churn prediction saves ฿50M/year |

---

## 2 · Types of Data

```
Data
├── Structured      → tables, rows, columns (MySQL, Postgres)
├── Semi-structured → JSON, XML, Parquet, Avro
└── Unstructured    → images, audio, free text, logs
```

**Why it matters:** ~80% of enterprise data is unstructured. Big data systems must handle all three types in the same pipeline.

---

## 3 · The Limits of Traditional Systems

| Problem | Single Machine | Distributed System |
|---------|---------------|-------------------|
| 1 TB CSV | Runs out of RAM | Splits across nodes |
| Real-time streams | Batch only | Spark Streaming / Kafka |
| Fault tolerance | One crash = lost job | Replication & retry |
| Scalability | Buy a bigger server | Add more nodes |

> **Rule of thumb:** If your data doesn't fit in memory on one machine, you need a distributed framework.

---

## 4 · Big Data Architecture Overview

A typical big data pipeline has five layers:

```
[Sources]  →  [Ingestion]  →  [Storage]  →  [Processing]  →  [Serving]

  IoT            Kafka          HDFS /           Spark           Dashboard
  Logs           Flume          S3 / GCS         Hive            ML API
  APIs           Sqoop          Delta Lake        Flink           Report
```

### Key components you will use in this course

- **HDFS** — Hadoop Distributed File System (storage)
- **Apache Spark** — in-memory distributed compute engine (processing)
- **Databricks** — managed Spark platform (environment)
- **MLlib** — Spark's built-in machine learning library

---

## 5 · Quick Python Warmup

Before the next lecture, confirm your Python environment works:

```python
# Check versions
import sys, platform
print(f"Python {sys.version}")
print(f"Platform: {platform.system()}")

# Basic list comprehension
data = [1, 4, 9, 16, 25, 36]
roots = [x**0.5 for x in data]
print(roots)  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Simple statistics
import statistics
values = [12, 7, 3, 14, 6, 11, 5, 4]
print(f"Mean:   {statistics.mean(values):.2f}")
print(f"Median: {statistics.median(values):.2f}")
print(f"Stdev:  {statistics.stdev(values):.2f}")
```

---

## 6 · Discussion Questions

1. Give three examples from your daily life where big data is collected without you noticing it.
2. A hospital stores 10 years of patient records as scanned PDFs. Which of the 5 Vs are most challenging here?
3. Why is "veracity" often considered the hardest V to address?

---

## Exercises

### Exercise 0.1 — Data Classification

Classify each data source below as *structured*, *semi-structured*, or *unstructured*. Justify your answer.

| Source | Your Classification | Reason |
|--------|---------------------|--------|
| MySQL sales table | | |
| Twitter JSON feed | | |
| CCTV footage | | |
| Apache access log | | |
| Parquet file on S3 | | |

### Exercise 0.2 — Scale Estimation

A smart city deploys 50,000 air-quality sensors. Each sends one reading every 10 seconds. Each reading is 200 bytes.

- How many readings per day?
- How many GB per day?
- How many TB per year?

```python
sensors    = 50_000
freq_sec   = 10
bytes_per  = 200
seconds    = 86_400  # per day

readings_per_day = sensors * (seconds // freq_sec)
gb_per_day = (readings_per_day * bytes_per) / 1e9
tb_per_year = gb_per_day * 365 / 1000

print(f"Readings/day : {readings_per_day:,}")
print(f"GB/day       : {gb_per_day:.2f}")
print(f"TB/year      : {tb_per_year:.2f}")
```

---

## Further Reading

- [The 5 Vs of Big Data — IBM](https://www.ibm.com/topics/big-data)
- *Mining of Massive Datasets* — Chapter 1 ([free PDF](http://www.mmds.org/))
- [What is a Data Lake? — AWS](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/)

---

*Next: [Lesson 1 → Hadoop & Spark Ecosystem](./lesson-01-hadoop-spark.md)*
