"""
04_train_logistic_regression.py

Purpose
-------
Train a Logistic Regression classifier using PySpark MLlib.

Features:
---------
1. Load train parquet dataset
2. Train Logistic Regression model
3. Handle class imbalance using weightCol
4. Save trained model
5. Print training summary metrics

Input
-----
data/processed/train_data.parquet

Output
------
models/logistic_regression/

Run
---
python src/04_train_logistic_regression.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import shutil

from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel

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
    "logistic_regression"
)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("HateSpeechLogisticRegression")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("LOGISTIC REGRESSION TRAINING")
print("=" * 80)

# =============================================================================
# LOAD TRAIN DATA
# =============================================================================

print("\nLoading training dataset...")

train_df = spark.read.parquet(TRAIN_DATA_PATH)

print(f"\nTraining Records: {train_df.count()}")

print("\nSchema:")
train_df.printSchema()

# =============================================================================
# TRAIN LOGISTIC REGRESSION
# =============================================================================

"""
Using weighted classification to address
class imbalance.

weightCol -> classWeight
"""

print("\nInitializing Logistic Regression Model...")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,
    standardization=True
)

print("\nTraining model...")

lr_model = lr.fit(train_df)

print("\nTraining Completed Successfully")

# =============================================================================
# MODEL COEFFICIENT INFORMATION
# =============================================================================

print("\nModel Information")
print("-" * 60)

print(f"Number of Features: {len(lr_model.coefficients)}")

print(f"Intercept: {lr_model.intercept}")

# =============================================================================
# TRAINING SUMMARY
# =============================================================================

summary = lr_model.summary

print("\nTraining Summary")
print("-" * 60)

print(f"Total Iterations: {summary.totalIterations}")

print(f"Objective History Length: {len(summary.objectiveHistory)}")

print("\nObjective History (Last 10 Values)")

for value in summary.objectiveHistory[-10:]:
    print(value)

# =============================================================================
# TRAINING METRICS
# =============================================================================

try:
    print("\nTraining Metrics")
    print("-" * 60)

    print(f"Area Under ROC : {summary.areaUnderROC:.4f}")

except Exception:
    print(
        "\nArea Under ROC unavailable "
        "(depends on Spark version)."
    )

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================

print("\nSample Predictions")
print("-" * 60)

sample_predictions = lr_model.transform(
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

lr_model.save(MODEL_DIR)

print(f"\nModel Saved Successfully:\n{MODEL_DIR}")

# =============================================================================
# VERIFY SAVED MODEL
# =============================================================================

print("\nVerifying saved model...")

loaded_model = LogisticRegressionModel.load(
    MODEL_DIR
)

print(
    "\nModel Reloaded Successfully"
)

print(
    f"Reloaded Intercept: "
    f"{loaded_model.intercept}"
)

# =============================================================================
# COMPLETION
# =============================================================================

print("\nTraining Completed Successfully")
print("=" * 80)

spark.stop()