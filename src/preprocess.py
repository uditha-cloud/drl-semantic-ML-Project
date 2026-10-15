import numpy as np
import pandas as pd

from config import (
    SENSOR_DATA_FILE,
    PROCESSED_DATA_DIR,
    WINDOW_SIZE,
    WINDOW_STEP,
    TRAIN_RATIO
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():
    print("\nLoading raw sensor data...")

    if not SENSOR_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{SENSOR_DATA_FILE}"
        )

    data = pd.read_csv(
        SENSOR_DATA_FILE,
        sep=r"\s+",
        header=None
    )

    print("Raw shape:", data.shape)

    return data


# ============================================================
# IDENTIFY NUMERIC SENSOR COLUMNS
# ============================================================

def get_numeric_columns(data):
    """
    Detect columns containing sensor measurements.
    """

    numeric_columns = []

    for column in data.columns:

        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.90:
            numeric_columns.append(column)

    return numeric_columns


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(data):
    print("\nCleaning dataset...")

    numeric_columns = get_numeric_columns(data)

    print("Detected numeric columns:")
    print(numeric_columns)

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric sensor columns were detected."
        )

    sensor_data = data[numeric_columns].copy()

    # Convert everything to numeric
    sensor_data = sensor_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity values
    sensor_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Count missing values before cleaning
    missing_before = sensor_data.isnull().sum().sum()

    print("Missing values before cleaning:", missing_before)

    # Interpolate missing values
    sensor_data = sensor_data.interpolate(
        method="linear",
        limit_direction="both"
    )

    # Fill any remaining missing values
    sensor_data = sensor_data.fillna(
        sensor_data.median()
    )

    missing_after = sensor_data.isnull().sum().sum()

    print("Missing values after cleaning:", missing_after)

    return sensor_data


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(data):
    """
    Min-Max normalization.

    Each sensor is converted approximately to [0, 1].
    """

    print("\nNormalizing sensor data...")

    minimum = data.min()
    maximum = data.max()

    denominator = maximum - minimum

    # Avoid division by zero
    denominator = denominator.replace(
        0,
        1
    )

    normalized = (
        data - minimum
    ) / denominator

    normalized = normalized.clip(
        0,
        1
    )

    return normalized


# ============================================================
# CREATE TIME WINDOWS
# ============================================================

def create_windows(data):
    """
    Divide continuous sensor readings into overlapping windows.
    """

    print("\nCreating time windows...")

    values = data.values

    windows = []

    total_rows = len(values)

    for start in range(
        0,
        total_rows - WINDOW_SIZE + 1,
        WINDOW_STEP
    ):

        end = start + WINDOW_SIZE

        window = values[start:end]

        windows.append(window)

    windows = np.array(windows)

    print("Window shape:", windows.shape)

    return windows


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(windows):
    """
    Split windows into training and testing sets.
    """

    print("\nSplitting dataset...")

    split_index = int(
        len(windows) * TRAIN_RATIO
    )

    train_data = windows[:split_index]

    test_data = windows[split_index:]

    print("Training windows:", len(train_data))
    print("Testing windows:", len(test_data))

    return train_data, test_data


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_data(train_data, test_data):

    print("\nSaving processed data...")

    train_path = (
        PROCESSED_DATA_DIR /
        "train_windows.npy"
    )

    test_path = (
        PROCESSED_DATA_DIR /
        "test_windows.npy"
    )

    np.save(
        train_path,
        train_data
    )

    np.save(
        test_path,
        test_data
    )

    print("Training data saved to:")
    print(train_path)

    print("\nTesting data saved to:")
    print(test_path)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("HT SENSOR DATA PREPROCESSING")
    print("=" * 60)

    # 1. Load
    raw_data = load_raw_data()

    # 2. Clean
    clean_sensor_data = clean_data(
        raw_data
    )

    # 3. Normalize
    normalized_data = normalize_data(
        clean_sensor_data
    )

    # 4. Create windows
    windows = create_windows(
        normalized_data
    )

    if len(windows) == 0:
        raise ValueError(
            "No windows were created. "
            "Check WINDOW_SIZE and dataset size."
        )

    # 5. Split
    train_data, test_data = split_data(
        windows
    )

    # 6. Save
    save_data(
        train_data,
        test_data
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()