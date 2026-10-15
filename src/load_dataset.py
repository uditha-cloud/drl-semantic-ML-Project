import pandas as pd
from pathlib import Path

from config import SENSOR_DATA_FILE, METADATA_FILE, SENSOR_COLUMNS


# ============================================================
# LOAD SENSOR DATA
# ============================================================

def load_sensor_data():
    """
    Load the HT Sensor Dataset.

    Returns:
        pandas.DataFrame: Sensor readings
    """

    print("\nLoading sensor dataset...")
    print("File:", SENSOR_DATA_FILE)

    if not SENSOR_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Sensor dataset not found:\n{SENSOR_DATA_FILE}"
        )

    # HT Sensor dataset is whitespace separated
    df = pd.read_csv(
        SENSOR_DATA_FILE,
        sep=r"\s+",
        header=None
    )

    print("Raw dataset shape:", df.shape)

    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    return df


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """
    Load HT Sensor metadata.

    Returns:
        pandas.DataFrame: Metadata information
    """

    print("\nLoading metadata...")
    print("File:", METADATA_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    # Metadata contains:
    # id, date, class, tc, dt
    metadata = pd.read_csv(
        METADATA_FILE,
        sep=r"\s+"
    )

    print("Metadata shape:", metadata.shape)

    print("\nMetadata columns:")
    print(metadata.columns.tolist())

    print("\nFirst 5 metadata rows:")
    print(metadata.head())

    return metadata


# ============================================================
# DATASET INFORMATION
# ============================================================

def display_dataset_info(sensor_data, metadata):
    """
    Display useful information about the loaded dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nSensor data:")
    print("Rows:", sensor_data.shape[0])
    print("Columns:", sensor_data.shape[1])

    print("\nMetadata:")
    print("Rows:", metadata.shape[0])
    print("Columns:", metadata.shape[1])

    print("\nMetadata classes:")

    if "class" in metadata.columns:
        print(metadata["class"].value_counts())

    print("\nMissing values in metadata:")
    print(metadata.isnull().sum())

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HT SENSOR DATASET LOADER")
    print("=" * 60)

    sensor_data = load_sensor_data()

    metadata = load_metadata()

    display_dataset_info(
        sensor_data,
        metadata
    )

    print("\nDataset loading completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()