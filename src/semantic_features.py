import numpy as np

from config import (
    PROCESSED_DATA_DIR,
    SEMANTIC_FEATURES
)


# ============================================================
# LOAD PROCESSED WINDOWS
# ============================================================

def load_windows():

    train_path = PROCESSED_DATA_DIR / "train_windows.npy"
    test_path = PROCESSED_DATA_DIR / "test_windows.npy"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{train_path}\n"
            "Run preprocess.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing data not found:\n{test_path}\n"
            "Run preprocess.py first."
        )

    train_windows = np.load(train_path)
    test_windows = np.load(test_path)

    print("Training windows:", train_windows.shape)
    print("Testing windows:", test_windows.shape)

    return train_windows, test_windows


# ============================================================
# CALCULATE TREND
# ============================================================

def calculate_trend(signal):
    """
    Calculate the approximate trend of a sensor signal.

    Positive value  -> increasing trend
    Negative value  -> decreasing trend
    Near zero       -> stable trend
    """

    x = np.arange(len(signal))

    if np.std(signal) == 0:
        return 0.0

    slope = np.polyfit(x, signal, 1)[0]

    return slope


# ============================================================
# EXTRACT FEATURES FROM ONE SENSOR
# ============================================================

def extract_sensor_features(signal):

    mean_value = np.mean(signal)

    std_value = np.std(signal)

    min_value = np.min(signal)

    max_value = np.max(signal)

    variance_value = np.var(signal)

    trend_value = calculate_trend(signal)

    range_value = max_value - min_value

    return [
        mean_value,
        std_value,
        min_value,
        max_value,
        variance_value,
        trend_value,
        range_value
    ]


# ============================================================
# EXTRACT FEATURES FROM ONE WINDOW
# ============================================================

def extract_window_features(window):

    """
    Extract semantic features from all sensors
    in one time window.
    """

    features = []

    # window shape:
    # [time_steps, sensors]

    number_of_sensors = window.shape[1]

    for sensor_index in range(number_of_sensors):

        signal = window[:, sensor_index]

        sensor_features = extract_sensor_features(
            signal
        )

        features.extend(sensor_features)

    return features


# ============================================================
# EXTRACT FEATURES FROM ALL WINDOWS
# ============================================================

def extract_features(windows):

    all_features = []

    print("\nExtracting semantic features...")

    for index, window in enumerate(windows):

        features = extract_window_features(
            window
        )

        all_features.append(features)

    features_array = np.array(
        all_features,
        dtype=np.float32
    )

    return features_array


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    train_features,
    test_features
):

    train_path = (
        PROCESSED_DATA_DIR /
        "train_semantic_features.npy"
    )

    test_path = (
        PROCESSED_DATA_DIR /
        "test_semantic_features.npy"
    )

    np.save(
        train_path,
        train_features
    )

    np.save(
        test_path,
        test_features
    )

    print("\nSemantic features saved:")

    print(train_path)

    print(test_path)


# ============================================================
# DISPLAY FEATURE INFORMATION
# ============================================================

def display_feature_information(
    train_features,
    test_features
):

    print("\n" + "=" * 60)
    print("SEMANTIC FEATURE INFORMATION")
    print("=" * 60)

    print(
        "Training feature shape:",
        train_features.shape
    )

    print(
        "Testing feature shape:",
        test_features.shape
    )

    print(
        "Features per window:",
        train_features.shape[1]
    )

    print("\nFirst training feature vector:")

    print(train_features[0])

    print("\nFeature types per sensor:")

    for feature in SEMANTIC_FEATURES:
        print(" -", feature)

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SEMANTIC FEATURE EXTRACTION")
    print("=" * 60)

    # 1. Load windows
    train_windows, test_windows = load_windows()

    # 2. Extract training features
    train_features = extract_features(
        train_windows
    )

    # 3. Extract testing features
    test_features = extract_features(
        test_windows
    )

    # 4. Save
    save_features(
        train_features,
        test_features
    )

    # 5. Display information
    display_feature_information(
        train_features,
        test_features
    )

    print("\nSemantic feature extraction completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()