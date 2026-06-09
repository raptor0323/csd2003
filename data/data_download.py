"""
download_data.py

Purpose:
--------
Downloads the Hate Speech and Offensive Language dataset directly from
the original GitHub repository and stores it locally.

Dataset:
https://github.com/t-davidson/hate-speech-and-offensive-language

Output:
-------
data/raw/hate_speech.csv

Run:
----
python data/download_data.py
"""

import os
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
)

RAW_DATA_DIR = os.path.join("data", "raw")

OUTPUT_FILE = os.path.join(
    RAW_DATA_DIR,
    "hate_speech.csv"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directory(directory_path: str) -> None:
    """
    Creates a directory if it does not already exist.

    Parameters
    ----------
    directory_path : str
        Directory path to create.
    """
    os.makedirs(directory_path, exist_ok=True)


def download_file(url: str, output_path: str) -> None:
    """
    Download file from URL.

    Parameters
    ----------
    url : str
        URL of the dataset.

    output_path : str
        Local destination path.
    """

    print("=" * 70)
    print("Downloading dataset...")
    print("=" * 70)

    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        raise Exception(
            f"Failed to download dataset.\n"
            f"HTTP Status Code: {response.status_code}"
        )

    with open(output_path, "wb") as file:
        file.write(response.content)

    print(f"\nDataset saved successfully:\n{output_path}")


def verify_file(file_path: str) -> None:
    """
    Verify file existence and size.

    Parameters
    ----------
    file_path : str
        Path of downloaded file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Downloaded file not found: {file_path}"
        )

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    print("\nVerification Successful")
    print(f"File Size: {file_size_mb:.2f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main execution function.
    """

    print("\nStarting Dataset Download...\n")

    # Create raw data folder
    create_directory(RAW_DATA_DIR)

    # Download dataset
    download_file(
        url=DATA_URL,
        output_path=OUTPUT_FILE
    )

    # Verify download
    verify_file(OUTPUT_FILE)

    print("\nDataset download completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()