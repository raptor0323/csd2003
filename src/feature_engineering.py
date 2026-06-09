"""
03_feature_engineering.py

Purpose
-------
Create NLP features using PySpark MLlib.

Pipeline:
---------
1. Load cleaned parquet data
2. CountVectorizer (Bag of Words)
3. TF-IDF feature generation
4. Stratified train/test split
5. Create class weights for imbalance handling
6. Save train/test parquet datasets

Outputs
-------
data/processed/train_data.parquet
data/processed/test_data.parquet

Columns Produced
----------------
label           -> Target variable
features        -> TF-IDF vector
classWeight     -> Weight for class imbalance
clean_text      -> Original cleaned text

Run
---
python src/03_feature_engineering.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    when
)

from pyspark.ml.feature import (
    CountVectorizer,
    IDF
)

# =============================================================================
# PATHS
# =============================================================================

INPUT_PARQUET = os.path.join(
    "data",
    "processed",
    "cleaned_data.parquet"
)

TRAIN_OUTPUT = os.path.join(
    "data",
    "processed",
    "train_data.parquet"
)

TEST_OUTPUT = os.path.join(
    "data",
    "processed",
    "test_data.parquet"
)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("HateSpeechFeatureEngineering")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("FEATURE ENGINEERING STARTED")
print("=" * 80)

# =============================================================================
# LOAD CLEANED DATA
# =============================================================================

print("\nLoading cleaned parquet...")

df = spark.read.parquet(INPUT_PARQUET)

print(f"\nRows Loaded: {df.count()}")

print("\nSchema:")
df.printSchema()

# =============================================================================
# COUNT VECTORIZER
# =============================================================================

print("\nBuilding CountVectorizer vocabulary...")

count_vectorizer = CountVectorizer(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    vocabSize=20000,
    minDF=5
)

cv_model = count_vectorizer.fit(df)

cv_df = cv_model.transform(df)

print("\nVocabulary Size:")
print(len(cv_model.vocabulary))

# =============================================================================
# TF-IDF
# =============================================================================

print("\nGenerating TF-IDF features...")

idf = IDF(
    inputCol="raw_features",
    outputCol="features"
)

idf_model = idf.fit(cv_df)

feature_df = idf_model.transform(cv_df)

# =============================================================================
# KEEP REQUIRED COLUMNS
# =============================================================================

feature_df = feature_df.select(
    "clean_text",
    "label",
    "features"
)

print("\nFeature DataFrame Created")

feature_df.show(
    5,
    truncate=False
)

# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

print("\nCalculating class distribution...")

class_distribution = (
    feature_df
    .groupBy("label")
    .count()
    .collect()
)

label_counts = {}

for row in class_distribution:
    label_counts[row["label"]] = row["count"]

total_records = sum(label_counts.values())

negative_count = label_counts.get(0, 1)
positive_count = label_counts.get(1, 1)

print(f"\nTotal Records: {total_records}")
print(f"Not Hate (0): {negative_count}")
print(f"Hate (1): {positive_count}")

# =============================================================================
# CLASS WEIGHTS
# =============================================================================

"""
Weight Formula

weight = total_samples / (num_classes * class_count)

For binary classification:

weight = N / (2 * class_count)
"""

negative_weight = total_records / (2 * negative_count)
positive_weight = total_records / (2 * positive_count)

print("\nClass Weights")
print(f"Weight (0): {negative_weight:.4f}")
print(f"Weight (1): {positive_weight:.4f}")

weighted_df = (
    feature_df
    .withColumn(
        "classWeight",
        when(
            col("label") == 1,
            lit(float(positive_weight))
        ).otherwise(
            lit(float(negative_weight))
        )
    )
)

# =============================================================================
# STRATIFIED TRAIN TEST SPLIT
# =============================================================================

"""
Spark does not provide a direct stratified train-test split.

We'll create it manually using sampleBy()
to maintain class proportions.
"""

print("\nCreating stratified train/test split...")

train_fraction = 0.80

train_df = weighted_df.sampleBy(
    "label",
    fractions={
        0: train_fraction,
        1: train_fraction
    },
    seed=42
)

# Test set = records not in train
test_df = weighted_df.subtract(train_df)

print("\nSplit Summary")
print("-" * 50)

print(f"Train Records: {train_df.count()}")
print(f"Test Records : {test_df.count()}")

print("\nTrain Distribution")
train_df.groupBy("label").count().show()

print("\nTest Distribution")
test_df.groupBy("label").count().show()

# =============================================================================
# REMOVE OLD OUTPUTS
# =============================================================================

for path in [TRAIN_OUTPUT, TEST_OUTPUT]:
    if os.path.exists(path):
        shutil.rmtree(path)

# =============================================================================
# SAVE TRAIN DATA
# =============================================================================

print("\nSaving train parquet...")

(
    train_df.write
    .mode("overwrite")
    .parquet(TRAIN_OUTPUT)
)

# =============================================================================
# SAVE TEST DATA
# =============================================================================

print("\nSaving test parquet...")

(
    test_df.write
    .mode("overwrite")
    .parquet(TEST_OUTPUT)
)

# =============================================================================
# VERIFY SAVED DATA
# =============================================================================

print("\nVerifying saved datasets...")

train_check = spark.read.parquet(TRAIN_OUTPUT)
test_check = spark.read.parquet(TEST_OUTPUT)

print(f"\nVerified Train Rows: {train_check.count()}")
print(f"Verified Test Rows : {test_check.count()}")

print("\nTrain Schema")
train_check.printSchema()

# =============================================================================
# COMPLETION
# =============================================================================

print("\nFeature Engineering Completed Successfully")

print("\nGenerated Files:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)

print("=" * 80)

spark.stop()