# Scalable Hate Speech Detection using PySpark

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Distributed%20ML-red.svg)
![MLlib](https://img.shields.io/badge/Spark%20MLlib-Machine%20Learning-green.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

# Scalable Hate Speech Detection using PySpark

An end-to-end Natural Language Processing (NLP) project built using Apache Spark (PySpark) for large-scale hate speech detection.

This project demonstrates how distributed machine learning pipelines can be used to process thousands of social media posts efficiently and classify them into:

- Hate Speech (1)
- Not Hate Speech (0)

The project covers:

- Data Acquisition
- Exploratory Data Analysis (EDA)
- Text Cleaning
- NLP Preprocessing
- Feature Engineering
- Distributed Machine Learning
- Model Training
- Model Evaluation
- Production-Ready Inference Pipeline

The goal is to improve and grow practical Data Science and Big Data skills relevant for Data Science and Machine Learning domains roles such as:

- Insurance
- Banking
- FinTech
- Social Media Monitoring
- Risk Analytics
- Compliance Monitoring

---

# Business Problem

Insurance companies continuously monitor customer feedback, complaints, social media interactions, and public sentiment.

Automated hate speech detection can help:

- Flag abusive content
- Protect brand reputation
- Improve customer experience
- Support compliance monitoring
- Reduce manual moderation efforts

This project demonstrates how a scalable Spark pipeline can solve such NLP classification problems.

---

# Dataset Information

Dataset:

**Hate Speech and Offensive Language Dataset**

Author:

**Thomas Davidson**

Original Source:

https://github.com/t-davidson/hate-speech-and-offensive-language

Dataset Size:

Approximately 25,000 tweets

Original Classes:

| Class | Meaning |
|---------|---------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Neither |

Binary Conversion Used In This Project:

| Original Class | New Label |
|----------------|------------|
| 0 | 1 (Hate) |
| 1 | 0 (Not Hate) |
| 2 | 0 (Not Hate) |

---

# Tech Stack

## Programming

- Python

## Big Data

- Apache Spark
- PySpark
- Spark SQL

## Machine Learning

- Spark MLlib
- Logistic Regression
- Naive Bayes

## NLP

- Regex Cleaning
- Tokenization
- Stop Word Removal
- Bag of Words
- TF-IDF

## Visualization

- Matplotlib
- Seaborn
- WordCloud

## Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# Project Architecture

```text
Raw Dataset
     │
     ▼
EDA
     │
     ▼
Text Cleaning
     │
     ▼
Tokenization
     │
     ▼
Stop Word Removal
     │
     ▼
TF-IDF Features
     │
     ▼
Train/Test Split
     │
     ▼
Model Training
 ┌──────────────┐
 │ Logistic Reg │
 └──────────────┘
        │

 ┌──────────────┐
 │ Naive Bayes  │
 └──────────────┘

        │
        ▼

Model Evaluation

        │
        ▼

Prediction Pipeline
```

---

# Project Structure

```text
hate_speech_detection_pyspark/
│
├── data/
│   │
│   ├── raw/
│   │   └── hate_speech.csv
│   │
│   ├── processed/
│   │   ├── cleaned_data.parquet
│   │   ├── train_data.parquet
│   │   └── test_data.parquet
│   │
│   └── download_data.py
│
├── models/
│   │
│   ├── logistic_regression/
│   │
│   └── naive_bayes/
│
├── notebooks/
│   │
│   └── full_pipeline.ipynb
│
├── plots/
│   │
│   ├── class_distribution.png
│   ├── tweet_length_distribution.png
│   ├── word_count_distribution.png
│   ├── top_20_words.png
│   ├── hate_wordcloud.png
│   ├── non_hate_wordcloud.png
│   ├── roc_curve.png
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_nb.png
│
├── src/
│   │
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_logistic_regression.py
│   ├── 05_train_naive_bayes.py
│   ├── 06_evaluate_models.py
│   └── 07_predict_new_text.py
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/hate_speech_detection_pyspark.git

cd hate_speech_detection_pyspark
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Google Colab Setup

Install Spark

```python
!pip install pyspark==3.5.1
```

Verify Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Test") \
    .getOrCreate()

spark
```

---

# Running the Project

## Step 1: Download Dataset

```bash
python data/download_data.py
```

---

## Step 2: Perform EDA

```bash
python src/01_eda.py
```

Generated:

- Class Distribution Plot
- Length Distribution Plot
- Word Count Distribution Plot
- Top Words Plot
- Hate WordCloud
- Non-Hate WordCloud

---

## Step 3: Preprocess Data

```bash
python src/preprocessing.py
```

Generated:

```text
data/processed/cleaned_data.parquet
```

---

## Step 4: Feature Engineering

```bash
python src/feature_engineering.py
```

Generated:

```text
data/processed/train_data.parquet
data/processed/test_data.parquet
```

---

## Step 5: Train Logistic Regression

```bash
python src/train_logistic_regression.py
```

Generated:

```text
models/logistic_regression/
```

---

## Step 6: Train Naive Bayes

```bash
python src/train_naive_bayes.py
```

Generated:

```text
models/naive_bayes/
```

---

## Step 7: Evaluate Models

```bash
python src/evaluate_models.py
```

Generated:

```text
plots/roc_curve.png

plots/confusion_matrix_lr.png

plots/confusion_matrix_nb.png
```

---

## Step 8: Predict New Text

```bash
python src/predict_new_text.py
```

---

# Feature Engineering Pipeline

The project uses the following NLP pipeline:

```text
Raw Tweet
     │
     ▼

Regex Cleaning

     │
     ▼

Lowercase Conversion

     │
     ▼

Tokenizer

     │
     ▼

StopWordsRemover

     │
     ▼

CountVectorizer

     │
     ▼

TF-IDF

     │
     ▼

Feature Vector
```

---

# Machine Learning Models

## Logistic Regression

Advantages:

- Fast
- Interpretable
- Works well with sparse TF-IDF vectors
- Strong baseline model

---

## Naive Bayes

Advantages:

- Efficient on text classification
- Low computational cost
- Suitable for large datasets

---

# Handling Class Imbalance

Hate speech is a minority class.

To prevent bias toward majority classes:

Weighted classification is used.

Weight Calculation:

```python
weight = total_samples / (2 * class_count)
```

Spark Logistic Regression uses:

```python
weightCol="classWeight"
```

---

# Evaluation Metrics

The following metrics are reported:

| Metric | Purpose |
|----------|----------|
| Accuracy | Overall correctness |
| Precision | False positive control |
| Recall | False negative control |
| F1 Score | Balance between precision & recall |
| ROC-AUC | Classification quality |

---

# Example Results

(The exact values depend on train-test split.)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.92 | 0.84 | 0.79 | 0.81 | 0.94 |
| Naive Bayes | 0.89 | 0.76 | 0.74 | 0.75 | 0.90 |

---

# Generated Visualizations

EDA:

- Class Distribution
- Tweet Length Distribution
- Word Count Distribution
- Top 20 Common Words
- Hate WordCloud
- Non-Hate WordCloud

Evaluation:

- ROC Curve
- Logistic Regression Confusion Matrix
- Naive Bayes Confusion Matrix

---

# Example Predictions

Input:

```text
I hate all people from that community
```

Output:

```text
Prediction: Hate Speech
Confidence: 0.97
```

Input:

```text
Have a wonderful day everyone
```

Output:

```text
Prediction: Not Hate Speech
Confidence: 0.99
```

---

# Skills Demonstrated

This project demonstrates:

✅ PySpark

✅ Distributed Computing

✅ Spark SQL

✅ NLP

✅ Text Classification

✅ TF-IDF

✅ Feature Engineering

✅ Logistic Regression

✅ Naive Bayes

✅ Model Evaluation

✅ Data Visualization

✅ Production ML Pipelines

---

# Future Improvements

Possible extensions:

- Word2Vec embeddings
- FastText embeddings
- Spark XGBoost
- Hyperparameter Tuning
- MLflow Tracking
- Docker Deployment
- Real-time Spark Streaming Classification
- REST API using FastAPI

---

# Author

Chaitanya Dahale

Data Science Project

Built using:

- Python
- PySpark
- Apache Spark MLlib
- NLP
- Machine Learning
