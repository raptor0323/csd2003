"""
06_evaluate_models.py

Purpose
-------
Evaluate Logistic Regression and Naive Bayes models.

Features
--------
1. Load test dataset
2. Load trained models
3. Generate predictions
4. Calculate:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
5. Plot ROC Curves
6. Plot Confusion Matrices
7. Generate model comparison table

Outputs
-------
plots/roc_curve.png
plots/confusion_matrix_lr.png
plots/confusion_matrix_nb.png

Run
---
python src/06_evaluate_models.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from pyspark.sql import SparkSession

from pyspark.ml.classification import (
    LogisticRegressionModel,
    NaiveBayesModel
)

# =============================================================================
# PATHS
# =============================================================================

TEST_DATA_PATH = os.path.join(
    "data",
    "processed",
    "test_data.parquet"
)

LR_MODEL_PATH = os.path.join(
    "models",
    "logistic_regression"
)

NB_MODEL_PATH = os.path.join(
    "models",
    "naive_bayes"
)

PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# SPARK SESSION
# =============================================================================

spark = (
    SparkSession.builder
    .appName("ModelEvaluation")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading test dataset...")

test_df = spark.read.parquet(TEST_DATA_PATH)

print(f"\nTest Records: {test_df.count()}")

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")

lr_model = LogisticRegressionModel.load(
    LR_MODEL_PATH
)

nb_model = NaiveBayesModel.load(
    NB_MODEL_PATH
)

print("Models loaded successfully.")

# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

print("\nGenerating predictions...")

lr_predictions = lr_model.transform(test_df)

nb_predictions = nb_model.transform(test_df)

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def evaluate_model(
    prediction_df,
    model_name
):
    """
    Convert Spark predictions to Pandas
    and compute evaluation metrics.
    """

    print("\n" + "=" * 80)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 80)

    pdf = prediction_df.select(
        "label",
        "prediction",
        "probability"
    ).toPandas()

    y_true = pdf["label"]

    y_pred = pdf["prediction"]

    # Probability of positive class
    y_prob = pdf["probability"].apply(
        lambda x: float(x[1])
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4
        )
    )

    print("\nMetrics")
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC ROC  : {auc:.4f}")

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }

    return (
        metrics,
        y_true,
        y_pred,
        y_prob
    )

# =============================================================================
# EVALUATE LOGISTIC REGRESSION
# =============================================================================

(
    lr_metrics,
    lr_y_true,
    lr_y_pred,
    lr_y_prob
) = evaluate_model(
    lr_predictions,
    "Logistic Regression"
)

# =============================================================================
# EVALUATE NAIVE BAYES
# =============================================================================

(
    nb_metrics,
    nb_y_true,
    nb_y_pred,
    nb_y_prob
) = evaluate_model(
    nb_predictions,
    "Naive Bayes"
)

# =============================================================================
# ROC CURVES
# =============================================================================

print("\nGenerating ROC Curve...")

lr_fpr, lr_tpr, _ = roc_curve(
    lr_y_true,
    lr_y_prob
)

nb_fpr, nb_tpr, _ = roc_curve(
    nb_y_true,
    nb_y_prob
)

plt.figure(figsize=(10, 7))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC={lr_metrics['AUC']:.3f})"
)

plt.plot(
    nb_fpr,
    nb_tpr,
    label=f"Naive Bayes (AUC={nb_metrics['AUC']:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

roc_path = os.path.join(
    PLOTS_DIR,
    "roc_curve.png"
)

plt.tight_layout()
plt.savefig(
    roc_path,
    dpi=300
)
plt.close()

print(f"Saved: {roc_path}")

# =============================================================================
# CONFUSION MATRIX PLOT FUNCTION
# =============================================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    filename,
    title
):
    """
    Save confusion matrix heatmap.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    output_path = os.path.join(
        PLOTS_DIR,
        filename
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300
    )
    plt.close()

    print(f"Saved: {output_path}")

# =============================================================================
# LOGISTIC REGRESSION CONFUSION MATRIX
# =============================================================================

save_confusion_matrix(
    lr_y_true,
    lr_y_pred,
    "confusion_matrix_lr.png",
    "Logistic Regression Confusion Matrix"
)

# =============================================================================
# NAIVE BAYES CONFUSION MATRIX
# =============================================================================

save_confusion_matrix(
    nb_y_true,
    nb_y_pred,
    "confusion_matrix_nb.png",
    "Naive Bayes Confusion Matrix"
)

# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame(
    [
        lr_metrics,
        nb_metrics
    ]
)

comparison_df = comparison_df.round(4)

print("\n")
print(comparison_df)

# =============================================================================
# SAVE COMPARISON TABLE
# =============================================================================

comparison_csv = os.path.join(
    "plots",
    "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)

print(
    f"\nComparison table saved to:\n{comparison_csv}"
)

# =============================================================================
# BEST MODEL
# =============================================================================

best_model = comparison_df.sort_values(
    by="F1",
    ascending=False
).iloc[0]

print("\nBest Model Based on F1 Score")
print("-" * 60)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"F1 Score: {best_model['F1']:.4f}"
)

# =============================================================================
# COMPLETION
# =============================================================================

print("\nEvaluation Completed Successfully")
print("=" * 80)

spark.stop()