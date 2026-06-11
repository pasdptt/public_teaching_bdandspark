# Lesson 4 · Algorithms for Big Data

> **Week 6** | Lecture + Lab  
> Slides: [Lecture 4](https://docs.google.com/presentation/d/1T8ALERFxHpIHQwOYd2ybBnbHjXWJDWH44LXfGUMpYh4/edit?usp=sharing)

---

## Learning Objectives

- Explain why classical algorithms fail at scale and what "streaming" replacements exist
- Implement probabilistic data structures: Bloom filters, HyperLogLog, Count-Min Sketch
- Use locality-sensitive hashing (LSH) for approximate similarity search
- Apply reservoir sampling to draw unbiased samples from unbounded streams

---

## 1 · The "Too Big to Fit" Problem

Many classical algorithms assume the dataset fits in memory:

| Algorithm | Memory requirement | Breaks at |
|-----------|-------------------|-----------|
| Exact distinct count | O(n) | ~10⁸ items |
| Exact frequency table | O(n) | large cardinality |
| Exact nearest neighbour | O(n²) | millions of points |
| Exact set membership | O(n) | billions of items |

Big data algorithms trade **exact answers for probabilistic approximations** with bounded error guarantees and O(1) or O(log n) memory.

---

## 2 · Bloom Filter — Set Membership

A Bloom filter answers "**is this element in the set?**" with:
- **No false negatives** — if it says "no", the element is definitely absent
- **Controlled false positive rate** — if it says "yes", it *might* be wrong (with probability p)

### How it works

```
Set S = {"alice", "bob", "carol"}
k = 3 hash functions, m = 20 bits

Insert "alice":
  h1("alice") = 2  → set bit 2
  h2("alice") = 7  → set bit 7
  h3("alice") = 14 → set bit 14

Query "dave":
  h1("dave") = 2   → bit 2 is set ✓
  h2("dave") = 5   → bit 5 is NOT set ✗ → "dave" is ABSENT (definitive)

Query "eve":
  h1("eve") = 2  ✓, h2("eve") = 7  ✓, h3("eve") = 14  ✓
  → "eve" is PRESENT (but it might be a false positive!)
```

### Python implementation

```python
import hashlib
import math

class BloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        capacity   — expected number of elements
        error_rate — desired false positive probability
        """
        self.m = math.ceil(
            -capacity * math.log(error_rate) / (math.log(2) ** 2)
        )  # number of bits
        self.k = math.ceil(
            (self.m / capacity) * math.log(2)
        )  # number of hash functions
        self.bits = bytearray(self.m)

    def _hashes(self, item: str):
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str):
        for idx in self._hashes(item):
            self.bits[idx] = 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[idx] for idx in self._hashes(item))


# Usage
bf = BloomFilter(capacity=1_000_000, error_rate=0.001)

seen_ids = ["user_001", "user_042", "user_999"]
for uid in seen_ids:
    bf.add(uid)

print("user_001" in bf)   # True  (in set)
print("user_100" in bf)   # False (not in set — with 99.9% confidence)
```

**Use case:** Before querying a database for a key, check the Bloom filter. Saves expensive I/O on misses.

---

## 3 · HyperLogLog — Approximate Count Distinct

**Problem:** `SELECT COUNT(DISTINCT user_id) FROM logs` on 10 billion rows requires storing all distinct IDs.

**HyperLogLog** estimates cardinality using O(log log n) memory with ~1–2% error.

```python
# Spark has built-in HLL
from pyspark.sql.functions import approx_count_distinct

df.select(
    approx_count_distinct("user_id", rsd=0.05).alias("approx_distinct_users")
).show()

# Compare with exact (much slower on large data)
df.select(
    countDistinct("user_id").alias("exact_distinct_users")
).show()
```

---

## 4 · Count-Min Sketch — Frequency Estimation

Estimates the frequency of any item in a stream using a fixed-size matrix of counters.

```python
# Spark DataFrames do not expose CMS directly, but you can use it via RDDs
# or the datasketches library

# Pure Python sketch
import hashlib

class CountMinSketch:
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [i * 2654435761 for i in range(depth)]

    def _hashes(self, item: str):
        for seed in self.seeds:
            h = int(hashlib.md5(f"{seed}{item}".encode()).hexdigest(), 16)
            yield h % self.width

    def update(self, item: str, count: int = 1):
        for row, col in enumerate(self._hashes(item)):
            self.table[row][col] += count

    def query(self, item: str) -> int:
        return min(
            self.table[row][col]
            for row, col in enumerate(self._hashes(item))
        )


# Simulate a log stream
import random
words = ["spark", "hadoop", "python", "sql"] * 1000 + ["rare_term"] * 3
random.shuffle(words)

cms = CountMinSketch()
for word in words:
    cms.update(word)

print(f"spark  ≈ {cms.query('spark')}")      # ~1000
print(f"rare   ≈ {cms.query('rare_term')}")  # ~3
```

---

## 5 · Reservoir Sampling

Draw an unbiased random sample of size k from a stream of unknown length n — in a single pass, with O(k) memory.

```python
import random

def reservoir_sample(stream, k: int) -> list:
    """
    Returns a uniform random sample of k items from an iterable stream.
    O(k) memory, O(n) time, single pass.
    """
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


# Simulate a 10-million-row stream without loading it all into memory
def log_stream(n: int):
    for i in range(n):
        yield {"event_id": i, "user": f"user_{i % 5000}", "value": random.random()}

sample = reservoir_sample(log_stream(10_000_000), k=1000)
print(f"Sample size: {len(sample)}")
print(f"First record: {sample[0]}")
```

---

## 6 · Locality-Sensitive Hashing (LSH) for Similarity Search

Find approximate nearest neighbours in high-dimensional spaces without comparing all pairs (which is O(n²)).

```python
from pyspark.ml.feature import MinHashLSH, HashingTF, Tokenizer
from pyspark.ml.linalg import Vectors

# Example: find similar documents
sentences = spark.createDataFrame([
    (0, "big data is transforming industry"),
    (1, "industry is changed by data"),
    (2, "spark is a fast big data engine"),
    (3, "the weather today is sunny"),
], ["id", "text"])

tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_df = tokenizer.transform(sentences)

hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100)
featurized = hashingTF.transform(words_df)

mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(featurized)

# Find similar pairs (Jaccard distance < 0.6)
similar = model.approxSimilarityJoin(
    featurized, featurized, threshold=0.6, distCol="jaccard_dist"
)
similar.filter("datasetA.id < datasetB.id") \
       .select("datasetA.id", "datasetB.id", "jaccard_dist") \
       .orderBy("jaccard_dist") \
       .show()
```

---

## Exercises

### Exercise 4.1 — Bloom Filter False Positive Rate

1. Create a Bloom filter with `capacity=10_000`, `error_rate=0.05`
2. Insert 10,000 random strings
3. Query 10,000 strings that were **not** inserted
4. Measure the actual false positive rate — does it match the configured 5%?

### Exercise 4.2 — Stream Frequency Estimation

```python
# Generate a Zipf-distributed stream (realistic for text / user actions)
import numpy as np
n_events = 1_000_000
items     = np.random.zipf(a=1.5, size=n_events)
```

1. Count exact frequencies with a Python `Counter`
2. Count approximate frequencies with a `CountMinSketch`
3. For the top 20 items, compare exact vs. approximate counts and compute the relative error

### Exercise 4.3 — Reservoir Sampling Bias Check

Prove empirically that reservoir sampling is unbiased:
1. Create a stream of integers 0–99
2. Sample 10 items, repeat 10,000 times
3. Plot the frequency histogram of sampled items — it should be approximately uniform

---

## Further Reading

- *Mining of Massive Datasets* — Chapter 4 (Mining Data Streams) · [free PDF](http://www.mmds.org/)
- [Bloom Filters by Example](https://llimllib.github.io/bloomfilter-tutorial/)
- [HyperLogLog — Wikipedia](https://en.wikipedia.org/wiki/HyperLogLog)
- [Spark MLlib LSH Guide](https://spark.apache.org/docs/latest/ml-features#locality-sensitive-hashing)

---

*Previous: [Lesson 3 ← Spark Programming](./lesson-03-spark-coding.md) · Next: [Lesson 5 → Mathematics & Statistics](./lesson-05-math-stats.md)*
