"""
05_train_naive_bayes.py

Purpose
-------
Train a Naive Bayes classifier using PySpark MLlib.

Features
--------
1. Load training parquet dataset
2. Train Naive Bayes model
3. Handle class imbalance using weightCol
4. Save trained model
5. Display training information
6. Verify model loading

Input
-----
data/processed/train_data.parquet

Output
------
models/naive_bayes/

Run
---
python src/05_train_naive_bayes.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import shutil

from pyspark.sql import SparkSession

from pyspark.ml.classification import (
    NaiveBayes,
    NaiveBayesModel
)

# =============================================================================
# PATHS
# =============================================================================

TRAIN_DATA_PATH = os.path.join(
    "data",
    "processed",
    "train_data.parquet"
)

MODEL_DIR = os.path.join(
    "models",
    "naive_bayes"
)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("HateSpeechNaiveBayes")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("NAIVE BAYES TRAINING")
print("=" * 80)

# =============================================================================
# LOAD TRAINING DATA
# =============================================================================

print("\nLoading training dataset...")

train_df = spark.read.parquet(TRAIN_DATA_PATH)

print(f"\nTraining Records: {train_df.count()}")

print("\nSchema:")
train_df.printSchema()

# =============================================================================
# VERIFY FEATURES
# =============================================================================

print("\nChecking feature column...")

train_df.select(
    "features",
    "label",
    "classWeight"
).show(
    5,
    truncate=False
)

# =============================================================================
# INITIALIZE MODEL
# =============================================================================

"""
multinomial:
Best suited for text classification
using count or TF-IDF features.
"""

print("\nInitializing Naive Bayes model...")

nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    modelType="multinomial",
    smoothing=1.0
)

# =============================================================================
# TRAIN MODEL
# =============================================================================

print("\nTraining model...")

nb_model = nb.fit(train_df)

print("\nTraining completed successfully.")

# =============================================================================
# MODEL INFORMATION
# =============================================================================

print("\nModel Information")
print("-" * 60)

print(
    f"Model Type: {nb_model.getModelType()}"
)

print(
    f"Number of Classes: "
    f"{len(nb_model.pi)}"
)

print(
    f"Number of Features: "
    f"{nb_model.theta.numCols}"
)

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================

print("\nSample Predictions")
print("-" * 60)

sample_predictions = nb_model.transform(
    train_df.limit(10)
)

sample_predictions.select(
    "label",
    "prediction",
    "probability"
).show(
    truncate=False
)

# =============================================================================
# SAVE MODEL
# =============================================================================

print("\nSaving model...")

if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

nb_model.save(MODEL_DIR)

print(
    f"\nModel Saved Successfully:\n{MODEL_DIR}"
)

# =============================================================================
# VERIFY SAVED MODEL
# =============================================================================

print("\nReloading model for verification...")

loaded_model = NaiveBayesModel.load(
    MODEL_DIR
)

print(
    "\nModel reloaded successfully."
)

print(
    f"Reloaded Model Type: "
    f"{loaded_model.getModelType()}"
)

# =============================================================================
# DISPLAY CLASS PRIORS
# =============================================================================

print("\nClass Priors")
print("-" * 60)

for idx, prior in enumerate(loaded_model.pi):
    print(
        f"Class {idx}: {prior}"
    )

# =============================================================================
# COMPLETION
# =============================================================================

print("\nNaive Bayes Training Completed Successfully")
print("=" * 80)

spark.stop()