# 040563105 · Statistical Analysis for Big Data

> **การวิเคราะห์เชิงสถิติสำหรับข้อมูลขนาดใหญ่**  
> King Mongkut's Institute of Technology North Bangkok (KMITNB)

---

## Table of Contents

- [Course Overview](#course-overview)
- [Prerequisites](#prerequisites)
- [Tools & Environment Setup](#tools--environment-setup)
- [Course Schedule](#course-schedule)
- [Grading](#grading)
- [Lessons](#lessons)
- [Class Project](#class-project)
- [Resources](#resources)
- [Contact](#contact)

---

## Course Overview

This course introduces the statistical foundations and computational tools required to work with large-scale data. Students will explore the full pipeline — from raw distributed storage to machine learning inference — using industry-standard frameworks.

**Topics covered:**

- Architecture and characteristics of big data systems
- Hadoop & Spark ecosystem
- Querying big data with SQL and DataFrames
- Mathematics and statistics at scale (PCA, sampling, hypothesis testing)
- Supervised and unsupervised machine learning on Spark
- Real-world use cases: Telco, IoT, Retail, Banking

| Field | Detail |
|---|---|
| Course Code | 040563105 |
| Credits | 3 (3-0-6) |
| Pre-requisite | None *(programming review recommended)* |
| Co-requisite | None |
| Instructor | Pasd Putthapipat |
| Contact | Pasd.Putthapipat@gmail.com |
| Curriculum Doc | [มคอ 3](https://drive.google.com/file/d/1Kvjokv5GkebTB9JEotrpBDENrYrkU5s0/view?usp=sharing) |

---

## Prerequisites

You don't need prior big-data experience, but you should be comfortable with:

- **Python basics** — variables, loops, functions, list comprehensions
- **SQL fundamentals** — `SELECT`, `WHERE`, `GROUP BY`, `JOIN`
- **Basic statistics** — mean, variance, probability, normal distribution

> **Recommended warmup:** Work through the [Python review notebook](https://github.com/pasdptt/PasdPublicNB/blob/master/Review_Python.ipynb) before the first class.

---

## Tools & Environment Setup

### 1 · Databricks (Free) Edition (primary)

The easiest way to run PySpark — no local install needed.

1. Sign up at <https://www.databricks.com/learn/free-edition>
2. Create an account.
3. Import notebooks.

### 2 · Google Colab (2nd option)

Run Spark inside a free Colab notebook - no local install needed also:

```python
# Install Java & PySpark in Colab
!apt-get install -y openjdk-11-jdk-headless -qq
!pip install pyspark --quiet

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("Colab").getOrCreate()
spark
```

### 3 · Local Setup (optional)

```bash
# macOS / Linux
brew install java          # or: sudo apt install default-jdk
pip install pyspark pandas matplotlib seaborn scikit-learn jupyter
```

---

## Course Schedule

| Week | Topic | Type |
|------|-------|------|
| 1 | Introduction to Big Data: Concepts & Architecture | Lecture |
| 2 | Modern Big Data — Hadoop & Spark Ecosystem | Lecture |
| 3 | Revisit: Python, SQL, and Bash | Lab |
| 4 | Querying Big Data | Lab |
| 5 | Mathematics for Big Data | Lecture + Lab |
| 6 | Algorithms for Big Data | Lecture + Lab |
| 7 | Big Data Visualisation | Lab |
| 8 | Project Proposal & Class Review | Workshop |
| **9** | **Midterm Exam** | **Exam** |
| 10 | Machine Learning for Big Data (1) — Supervised | Lecture + Lab |
| 11 | Machine Learning for Big Data (2) — Unsupervised | Lecture + Lab |
| 12 | Big Data Use Case: Telco | Case Study |
| 13 | Big Data Use Case: IoT | Case Study |
| 14 | Big Data Use Case: Retail & Banking | Case Study |
| 15 | Final Project Presentation | Presentation |
| 16 | Final Project Presentation (cont'd) | Presentation |
| **17** | **Final Exam** | **Exam** |

---

## Grading

| Component | Weight |
|-----------|--------|
| Midterm Exam | 20% |
| Final Exam | 35% |
| Class Project | 25% |
| Practical Assignments | 20% |
| **Total** | **100%** |

> Practical assignments are submitted as Jupyter notebooks (`.ipynb`) via the course platform.

---

## Lessons

Each lesson file lives in the [`lessons/`](./lessons/) folder and contains learning objectives, key concepts, code examples, and exercises.

| # | File | Topic |
|---|------|-------|
| 0 | [lesson-00-introduction.md](./lessons/lesson-00-introduction.md) | Introduction to Big Data |
| 1 | [lesson-01-hadoop-spark.md](./lessons/lesson-01-hadoop-spark.md) | Hadoop & Spark Ecosystem |
| 2 | [lesson-02-sql.md](./lessons/lesson-02-sql.md) | SQL & DataFrame Querying |
| 3 | [lesson-03-spark-coding.md](./lessons/lesson-03-spark-coding.md) | Spark DataFrame Programming |
| 4 | [lesson-04-algorithms.md](./lessons/lesson-04-algorithms.md) | Algorithms for Big Data |
| 5 | [lesson-05-math-stats.md](./lessons/lesson-05-math-stats.md) | Mathematics & Statistics for Big Data |
| 6 | [lesson-06-ml-bigdata.md](./lessons/lesson-06-ml-bigdata.md) | ML for Big Data — PCA, Testing, Sampling |
| 7 | [lesson-07-supervised.md](./lessons/lesson-07-supervised.md) | Supervised Learning on Spark |
| 8 | [lesson-08-unsupervised.md](./lessons/lesson-08-unsupervised.md) | Unsupervised Learning on Spark |
| 9 | [lesson-09-use-cases.md](./lessons/lesson-09-use-cases.md) | Use Cases: Telco, IoT, Retail & Banking |

---

## Class Project

The project is the centrepiece of the course. Working in groups, students apply the full pipeline — data ingestion, exploration, feature engineering, modelling, and storytelling — to a real large-scale dataset.

### Timeline

| Date | Milestone |
|------|-----------|
| TBD | Group declaration & dataset selection |
| TBD (Tentatively - before midterm exam) | In-class proposal presentation |
| TBC | Final presentation |
| TBC | Submit report + notebook |

### Suggested Datasets

| # | Dataset | Source |
|---|---------|--------|
| 1 | Trending Videos on YouTube | [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new/data) |
| 2 | User Anime List | [Kaggle](https://www.kaggle.com/datasets/tavuksuzdurum/user-animelist-dataset) |
| 3 | USA City Crime (LA) | [Kaggle](https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000) |
| 4 | Big Sales Data | [Kaggle](https://www.kaggle.com/datasets/pigment/big-sales-data) |
| 5 | Apple App Store Apps | [Kaggle](https://www.kaggle.com/datasets/gauthamp10/apple-appstore-apps) |
| 6 | Smart Meters in London | [Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) |
| 7 | Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/credit-card-fraud) |
| 8 | Credit Card Default | [Kaggle](https://www.kaggle.com/mishra5001/credit-card) |
| 9 | Scania Truck APS Failure | [Kaggle](https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set) |
| 10 | Airbnb New User Bookings | [Kaggle](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings) |
| 11 | NYC Taxi Trips | [Kaggle](https://www.kaggle.com/gopalkalpnde/nyc-tlc-data) |
| 12 | UK Accidents — 10 Years | [Kaggle](https://www.kaggle.com/benoit72/uk-accidents-10-years-history-with-many-variables) |
| 13 | BART Train Ridership | [Kaggle](https://www.kaggle.com/mrgeislinger/bart-ridership) |
| 14 | Google Play Store Apps | [Kaggle](https://www.kaggle.com/gauthamp10/google-playstore-apps) |
| 15 | Fraud Detection | [Kaggle](https://www.kaggle.com/datasets/bannourchaker/frauddetection) |

> You may bring your own dataset, but it must pass the **proposal review session** first.

### Deliverables

- **Proposal (Week 8):** 5-minute pitch — problem statement, dataset overview, planned analysis
- **Final Presentation (Week 15–16):** 15-minute demo of findings with visualisations
- **Report + Notebook:** Clean, reproducible Jupyter notebook with a written report

---

## Resources

### Course Notebooks

- 📓 [Pasd-BigData GitHub Repo](https://github.com/pasdptt/public_teaching_bdandspark) — public repo for this course
- 📓 [PasdPublicNB GitHub Repo](https://github.com/pasdptt/PasdPublicNB) — my other public repo

### External References

| Resource | Link |
|----------|------|
| PySpark Documentation | https://spark.apache.org/docs/latest/api/python/ |
| Databricks Community | https://community.cloud.databricks.com/ |
| W3Schools SQL Try-It | https://www.w3schools.com/sql/trymysql.asp |
| Kaggle (datasets & notebooks) | https://www.kaggle.com/ |
| MLlib Guide | https://spark.apache.org/docs/latest/ml-guide.html |

### Textbooks & Reading

- *Learning Spark* (2nd ed.) — Damji et al. (O'Reilly)
- *Spark: The Definitive Guide* — Chambers & Zaharia (O'Reilly)
- *Mining of Massive Datasets* — Leskovec, Rajaraman & Ullman ([free PDF](http://www.mmds.org/))

---

## Contact

| | |
|---|---|
| **Instructor** | Pasd Putthapipat |
| **Email** | [Pasd.Putthapipat@gmail.com](mailto:Pasd.Putthapipat@gmail.com) |
| **Office Hours** | By appointment |
| **Course Site** | TBD |
