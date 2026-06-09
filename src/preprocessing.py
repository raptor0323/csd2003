"""
02_preprocessing.py

Purpose
-------
Preprocess raw tweets using PySpark.

Operations:
-----------
1. Initialize SparkSession
2. Load raw CSV dataset
3. Create binary target label
4. Clean tweet text
5. Tokenize text
6. Remove stop words
7. Save processed dataset as parquet

Output
------
data/processed/cleaned_data.parquet

Run
---
python src/02_preprocessing.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    regexp_replace,
    trim,
    udf
)
from pyspark.sql.types import StringType

from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover
)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

RAW_DATA_PATH = os.path.join(
    "data",
    "raw",
    "hate_speech.csv"
)

PROCESSED_DIR = os.path.join(
    "data",
    "processed"
)

OUTPUT_PARQUET = os.path.join(
    PROCESSED_DIR,
    "cleaned_data.parquet"
)

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("HateSpeechPreprocessing")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("SPARK SESSION STARTED")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RAW_DATA_PATH)
)

print("\nDataset Loaded Successfully")

print("\nSchema:")
df.printSchema()

print("\nTotal Rows:")
print(df.count())

# =============================================================================
# CREATE BINARY LABEL
# =============================================================================

"""
Original Dataset

0 = Hate Speech
1 = Offensive Language
2 = Neither

Binary Mapping

0 -> 1 (Hate Speech)
1 -> 0 (Not Hate)
2 -> 0 (Not Hate)
"""

df = df.withColumn(
    "label",
    (col("class") == 0).cast("integer")
)

print("\nBinary label created.")

# =============================================================================
# TEXT CLEANING FUNCTION
# =============================================================================

def clean_text(text):
    """
    Clean tweet text.

    Steps:
    ------
    1. Lowercase
    2. Remove URLs
    3. Remove mentions
    4. Remove hashtags
    5. Remove special characters
    6. Remove numbers
    7. Remove extra spaces
    """

    if text is None:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtags symbol
    text = re.sub(r"#", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


clean_text_udf = udf(
    clean_text,
    StringType()
)

# =============================================================================
# CLEAN TEXT
# =============================================================================

print("\nCleaning tweets...")

df = df.withColumn(
    "clean_text",
    clean_text_udf(col("tweet"))
)

# =============================================================================
# REMOVE EMPTY RECORDS
# =============================================================================

df = df.filter(
    col("clean_text").isNotNull()
)

df = df.filter(
    col("clean_text") != ""
)

# =============================================================================
# TOKENIZATION
# =============================================================================

print("\nTokenizing text...")

tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="tokens"
)

tokenized_df = tokenizer.transform(df)

# =============================================================================
# STOP WORD REMOVAL
# =============================================================================

print("\nRemoving stop words...")

remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)

processed_df = remover.transform(tokenized_df)

# =============================================================================
# OPTIONAL QUALITY CHECKS
# =============================================================================

print("\nSample Processed Records")

processed_df.select(
    "tweet",
    "clean_text",
    "filtered_tokens",
    "label"
).show(
    5,
    truncate=False
)

# =============================================================================
# SELECT FINAL COLUMNS
# =============================================================================

final_df = processed_df.select(
    "count",
    "hate_speech",
    "offensive_language",
    "neither",
    "class",
    "tweet",
    "clean_text",
    "tokens",
    "filtered_tokens",
    "label"
)

# =============================================================================
# SAVE PARQUET
# =============================================================================

print("\nSaving processed parquet...")

if os.path.exists(OUTPUT_PARQUET):
    import shutil
    shutil.rmtree(OUTPUT_PARQUET)

(
    final_df.write
    .mode("overwrite")
    .parquet(OUTPUT_PARQUET)
)

print("\nSaved Successfully:")
print(OUTPUT_PARQUET)

# =============================================================================
# SUMMARY
# =============================================================================

print("\nDataset Summary")
print("-" * 50)

print(f"Rows: {final_df.count()}")
print(f"Columns: {len(final_df.columns)}")

print("\nLabel Distribution")

final_df.groupBy(
    "label"
).count().show()

# =============================================================================
# STOP SPARK
# =============================================================================

spark.stop()

print("\nPreprocessing Completed Successfully")
print("=" * 80)