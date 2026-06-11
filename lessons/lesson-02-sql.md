# Lesson 2 · SQL & DataFrame Querying

> **Week 3–4** | Lecture + Lab  
> Slides: [Lecture 2](https://docs.google.com/presentation/d/1jplaSRbpJCYkUsDI7l4ZApvWhJRCwp4qi407lZllDhc/edit?usp=sharing) · [W3Schools SQL Try-It](https://www.w3schools.com/sql/trymysql.asp)

---

## Learning Objectives

- Write SQL queries against Spark temporary views
- Translate SQL to PySpark DataFrame API and vice versa
- Use window functions for ranking and running totals
- Load real-world CSV/JSON data from HDFS or cloud storage

---

## 1 · Why SQL in Big Data?

SQL is the lingua franca of data. Spark supports it natively through **Spark SQL**, which:

- Accepts standard SQL syntax
- Optimises queries with the **Catalyst** planner and **Tungsten** execution engine
- Works on DataFrames registered as **temporary views**

```python
# Register a DataFrame as a SQL-queryable view
df.createOrReplaceTempView("sales")

# Run plain SQL — returns another DataFrame
result = spark.sql("""
    SELECT region, SUM(revenue) AS total_revenue
    FROM   sales
    WHERE  year = 2024
    GROUP  BY region
    ORDER  BY total_revenue DESC
""")
result.show()
```

---

## 2 · Loading Data

### CSV

```python
df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("dbfs:/FileStore/sales_2024.csv"))

df.printSchema()
df.show(5)
```

### JSON

```python
df = spark.read.json("dbfs:/FileStore/events/*.json")
```

### Parquet (preferred for big data)

```python
# Write
df.write.mode("overwrite").parquet("dbfs:/output/sales_parquet/")

# Read
df = spark.read.parquet("dbfs:/output/sales_parquet/")
```

> **Parquet vs CSV:** Parquet is columnar, compressed, and schema-preserving. A 10 GB CSV commonly shrinks to ~1.5 GB as Parquet and reads 5–10× faster.

---

## 3 · Core SQL Operations

### SELECT & WHERE

```sql
-- SQL
SELECT customer_id, amount, category
FROM   sales
WHERE  amount > 1000
  AND  category IN ('Electronics', 'Appliances');
```

```python
# PySpark equivalent
from pyspark.sql.functions import col

df.select("customer_id", "amount", "category") \
  .filter((col("amount") > 1000) &
          (col("category").isin("Electronics", "Appliances"))) \
  .show()
```

### GROUP BY & Aggregation

```sql
-- SQL
SELECT   category,
         COUNT(*)        AS orders,
         SUM(amount)     AS revenue,
         AVG(amount)     AS avg_order
FROM     sales
GROUP BY category
ORDER BY revenue DESC;
```

```python
# PySpark
from pyspark.sql.functions import count, sum as spark_sum, avg, round

df.groupBy("category").agg(
    count("*").alias("orders"),
    spark_sum("amount").alias("revenue"),
    round(avg("amount"), 2).alias("avg_order")
).orderBy(col("revenue").desc()).show()
```

### JOIN

```sql
-- SQL
SELECT s.order_id, s.amount, c.name, c.segment
FROM   sales     s
JOIN   customers c ON s.customer_id = c.customer_id
WHERE  c.segment = 'Premium';
```

```python
# PySpark
sales.join(customers, on="customer_id", how="inner") \
     .filter(col("segment") == "Premium") \
     .select("order_id", "amount", "name", "segment") \
     .show()
```

**Join types:**

| Type | SQL keyword | PySpark `how=` |
|------|------------|----------------|
| Inner | `JOIN` | `"inner"` |
| Left outer | `LEFT JOIN` | `"left"` |
| Right outer | `RIGHT JOIN` | `"right"` |
| Full outer | `FULL OUTER JOIN` | `"outer"` |
| Cross | `CROSS JOIN` | `"cross"` |

---

## 4 · Window Functions

Window functions compute a result **across a set of rows related to the current row** — without collapsing them with `GROUP BY`.

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank, row_number, lag, sum as spark_sum

# Define the window: partition by region, order by revenue desc
window_spec = Window.partitionBy("region").orderBy(col("revenue").desc())

df_ranked = df.withColumn("rank",        rank().over(window_spec)) \
              .withColumn("dense_rank",  dense_rank().over(window_spec)) \
              .withColumn("row_number",  row_number().over(window_spec))

df_ranked.show()
```

### Running Total

```python
window_cumsum = Window.partitionBy("region") \
                      .orderBy("order_date") \
                      .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df = df.withColumn("running_revenue", spark_sum("revenue").over(window_cumsum))
df.show()
```

### Lag / Lead (time-series comparison)

```python
window_time = Window.partitionBy("product_id").orderBy("month")

df = df.withColumn("prev_month_sales", lag("sales", 1).over(window_time)) \
       .withColumn("mom_growth",
           (col("sales") - col("prev_month_sales")) / col("prev_month_sales"))
df.show()
```

---

## 5 · Common Data-Cleaning Operations

```python
from pyspark.sql.functions import when, coalesce, lit, trim, upper, to_date

# Fill nulls
df = df.fillna({"amount": 0, "category": "Unknown"})

# Conditional column
df = df.withColumn("size_label",
    when(col("amount") < 100,  "Small")
   .when(col("amount") < 1000, "Medium")
   .otherwise("Large"))

# Date parsing
df = df.withColumn("order_date", to_date(col("order_date_str"), "yyyy-MM-dd"))

# String cleanup
df = df.withColumn("category", trim(upper(col("category"))))

# Drop duplicates on key columns
df = df.dropDuplicates(["order_id"])
```

---

## Exercises

### Exercise 2.1 — SQL Translation

Translate the following SQL into PySpark DataFrame API code:

```sql
SELECT   product_name,
         region,
         COUNT(*)     AS num_orders,
         SUM(revenue) AS total_revenue
FROM     orders
WHERE    order_date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY product_name, region
HAVING   SUM(revenue) > 50000
ORDER BY total_revenue DESC
LIMIT    20;
```

### Exercise 2.2 — Window Ranking

Using the `sales` dataset:

1. Rank products by total revenue **within each region** (dense rank)
2. Show only the **top-3** products per region
3. Add a column showing how far each product's revenue is from the regional leader (`revenue_gap`)

### Exercise 2.3 — Join & Aggregate

You have two DataFrames:

- `orders(order_id, customer_id, product_id, quantity, price)`
- `products(product_id, product_name, category, cost)`

Write a query that returns the **top 5 most profitable categories**, where profit = `(price - cost) * quantity`.

---

## Further Reading

- [Spark SQL Programming Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [PySpark SQL Functions Reference](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
- [Window Functions in Spark — Databricks Blog](https://www.databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)

---

*Previous: [Lesson 1 ← Hadoop & Spark](./lesson-01-hadoop-spark.md) · Next: [Lesson 3 → Spark DataFrame Programming](./lesson-03-spark-coding.md)*
