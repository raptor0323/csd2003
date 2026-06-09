"""
07_predict_new_text.py

Purpose
-------
Perform inference on new unseen text using the trained
PySpark Hate Speech Detection pipeline.

Pipeline
--------
Raw Text
    ↓
Cleaning
    ↓
Tokenization
    ↓
Stop Word Removal
    ↓
CountVectorizerModel
    ↓
IDFModel
    ↓
TF-IDF Features
    ↓
Logistic Regression Model
    ↓
Prediction

Run
---
python src/07_predict_new_text.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizerModel,
    IDFModel
)

from pyspark.ml.classification import (
    LogisticRegressionModel
)

# =============================================================================
# PATHS
# =============================================================================

LR_MODEL_PATH = os.path.join(
    "models",
    "logistic_regression"
)

CV_MODEL_PATH = os.path.join(
    "models",
    "count_vectorizer_model"
)

IDF_MODEL_PATH = os.path.join(
    "models",
    "idf_model"
)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("HateSpeechInference")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("HATE SPEECH DETECTION - INFERENCE")
print("=" * 80)

# =============================================================================
# TEXT CLEANING FUNCTION
# =============================================================================

def clean_text(text):
    """
    Same cleaning logic used during training.
    """

    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"#", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


clean_text_udf = udf(
    clean_text,
    StringType()
)

# =============================================================================
# LOAD TRAINED ARTIFACTS
# =============================================================================

print("\nLoading saved models...")

cv_model = CountVectorizerModel.load(
    CV_MODEL_PATH
)

idf_model = IDFModel.load(
    IDF_MODEL_PATH
)

lr_model = LogisticRegressionModel.load(
    LR_MODEL_PATH
)

print("All models loaded successfully.")

# =============================================================================
# SAMPLE INPUT TEXTS
# =============================================================================

sample_texts = [

    "I hate everyone from that community",

    "Have a wonderful day and enjoy your life",

    "You are all disgusting people and should disappear",

    "Thank you for helping me today",

    "I wish happiness and success for everyone"
]

# =============================================================================
# CREATE DATAFRAME
# =============================================================================

input_df = spark.createDataFrame(
    [(text,) for text in sample_texts],
    ["text"]
)

# =============================================================================
# CLEAN TEXT
# =============================================================================

input_df = input_df.withColumn(
    "clean_text",
    clean_text_udf(col("text"))
)

# =============================================================================
# TOKENIZATION
# =============================================================================

tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="tokens"
)

tokenized_df = tokenizer.transform(
    input_df
)

# =============================================================================
# STOP WORD REMOVAL
# =============================================================================

remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered_tokens"
)

processed_df = remover.transform(
    tokenized_df
)

# =============================================================================
# COUNT VECTORIZER TRANSFORM
# =============================================================================

cv_df = cv_model.transform(
    processed_df
)

# =============================================================================
# IDF TRANSFORM
# =============================================================================

feature_df = idf_model.transform(
    cv_df
)

# =============================================================================
# PREDICTION
# =============================================================================

predictions = lr_model.transform(
    feature_df
)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("\nPrediction Results")
print("=" * 80)

results = predictions.select(
    "text",
    "prediction",
    "probability"
).collect()

for row in results:

    prediction = int(row["prediction"])

    probability_vector = row["probability"]

    confidence = max(probability_vector)

    label = (
        "HATE SPEECH"
        if prediction == 1
        else "NOT HATE SPEECH"
    )

    print(f"\nInput Text:")
    print(f"{row['text']}")

    print(f"\nPrediction:")
    print(label)

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )

    print("-" * 80)

# =============================================================================
# COMPLETION
# =============================================================================

print("\nInference Completed Successfully")
print("=" * 80)

spark.stop()