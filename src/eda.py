"""
01_eda.py

Exploratory Data Analysis for Hate Speech Detection Project

This script performs:

1. Dataset overview
2. Missing value analysis
3. Class distribution visualization
4. Tweet length analysis
5. Word count analysis
6. Top 20 most common words
7. Hate speech WordCloud
8. Non-hate WordCloud

All generated plots are saved to:

plots/

Run:
-----
python src/01_eda.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_PATH = os.path.join(
    "data",
    "raw",
    "hate_speech.csv"
)

PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text_for_eda(text: str) -> str:
    """
    Basic text cleaning used only for EDA visualizations.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    text = str(text)

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()

    return text


def save_plot(filename: str) -> None:
    """
    Save current matplotlib figure.

    Parameters
    ----------
    filename : str
    """

    output_path = os.path.join(PLOTS_DIR, filename)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")

df = pd.read_csv(RAW_DATA_PATH)

print("\nDataset Loaded Successfully")
print("-" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# =============================================================================
# DATA OVERVIEW
# =============================================================================

print("\nFirst 5 Rows")
print("-" * 60)
print(df.head())

print("\nDataset Information")
print("-" * 60)
print(df.info())

print("\nMissing Values")
print("-" * 60)
print(df.isnull().sum())


# =============================================================================
# BINARY LABEL CREATION
# =============================================================================

"""
Original Labels:

0 = Hate Speech
1 = Offensive Language
2 = Neither

Binary Conversion:

0 -> 1 (Hate)
1 -> 0 (Not Hate)
2 -> 0 (Not Hate)
"""

df["binary_label"] = np.where(df["class"] == 0, 1, 0)


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

print("\nGenerating Class Distribution Plot...")

class_counts = (
    df["binary_label"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=["Not Hate", "Hate"],
    y=class_counts.values
)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

save_plot("class_distribution.png")


# =============================================================================
# TWEET LENGTH ANALYSIS
# =============================================================================

print("\nGenerating Tweet Length Distribution...")

df["tweet_length"] = df["tweet"].astype(str).apply(len)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="tweet_length",
    bins=50,
    kde=True
)

plt.title("Tweet Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")

save_plot("tweet_length_distribution.png")


# =============================================================================
# WORD COUNT ANALYSIS
# =============================================================================

print("\nGenerating Word Count Distribution...")

df["word_count"] = (
    df["tweet"]
    .astype(str)
    .apply(lambda x: len(x.split()))
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="word_count",
    bins=40,
    kde=True
)

plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

save_plot("word_count_distribution.png")


# =============================================================================
# TEXT CLEANING FOR WORD ANALYSIS
# =============================================================================

print("\nCleaning text for word frequency analysis...")

df["clean_text"] = (
    df["tweet"]
    .astype(str)
    .apply(clean_text_for_eda)
)


# =============================================================================
# TOP 20 MOST COMMON WORDS
# =============================================================================

print("\nGenerating Top 20 Most Common Words Plot...")

all_words = " ".join(df["clean_text"])

word_list = all_words.split()

word_counts = Counter(word_list)

top_20 = word_counts.most_common(20)

top_words = [item[0] for item in top_20]
top_counts = [item[1] for item in top_20]

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_counts,
    y=top_words
)

plt.title("Top 20 Most Common Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

save_plot("top_20_words.png")


# =============================================================================
# WORD CLOUD - HATE SPEECH
# =============================================================================

print("\nGenerating Hate Speech WordCloud...")

hate_text = " ".join(
    df[df["binary_label"] == 1]["clean_text"]
)

hate_wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=300
).generate(hate_text)

plt.figure(figsize=(12, 6))

plt.imshow(
    hate_wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title("WordCloud - Hate Speech")

save_plot("hate_wordcloud.png")


# =============================================================================
# WORD CLOUD - NON HATE SPEECH
# =============================================================================

print("\nGenerating Non-Hate WordCloud...")

non_hate_text = " ".join(
    df[df["binary_label"] == 0]["clean_text"]
)

non_hate_wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=300
).generate(non_hate_text)

plt.figure(figsize=(12, 6))

plt.imshow(
    non_hate_wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title("WordCloud - Non Hate Speech")

save_plot("non_hate_wordcloud.png")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\nSummary Statistics")
print("-" * 60)

print("\nTweet Length Statistics")
print(df["tweet_length"].describe())

print("\nWord Count Statistics")
print(df["word_count"].describe())

print("\nBinary Class Distribution")
print(df["binary_label"].value_counts())

print("\nEDA Completed Successfully.")
print(f"\nAll plots saved to: {PLOTS_DIR}/")