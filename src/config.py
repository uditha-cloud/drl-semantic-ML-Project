from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folders
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

# Raw dataset files
SENSOR_DATA_FILE = RAW_DATA_DIR / "HT_Sensor_dataset" / "HT_Sensor_dataset.dat"
METADATA_FILE = RAW_DATA_DIR / "HT_Sensor_metadata.dat"

# Model and result folders
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


# ============================================================
# DATASET SETTINGS
# ============================================================

# Number of sensor readings used in one time window
WINDOW_SIZE = 50

# Step between consecutive windows
WINDOW_STEP = 25

# Train/test split
TRAIN_RATIO = 0.80


# ============================================================
# SENSOR SETTINGS
# ============================================================

# Sensor columns in the HT Sensor Dataset
SENSOR_COLUMNS = [
    "Temperature",
    "Humidity",
    "Light",
    "CO",
    "HumidityRatio"
]


# ============================================================
# SEMANTIC FEATURE SETTINGS
# ============================================================

# Features extracted from each sensor window
SEMANTIC_FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "variance",
    "trend",
    "range"
]


# ============================================================
# DRL ENVIRONMENT SETTINGS
# ============================================================

# Available compression actions
#
# 0 -> No compression
# 1 -> Low compression
# 2 -> Medium compression
# 3 -> High compression
#
NUM_ACTIONS = 4

COMPRESSION_LEVELS = {
    0: 1.00,
    1: 0.75,
    2: 0.50,
    3: 0.25
}


# ============================================================
# REWARD SETTINGS
# ============================================================

# Weight given to energy saving
ENERGY_WEIGHT = 0.4

# Weight given to semantic information preservation
SEMANTIC_WEIGHT = 0.5

# Weight given to compression distortion
DISTORTION_WEIGHT = 0.1


# ============================================================
# ENERGY MODEL
# ============================================================

# Approximate energy consumed for transmitting one unit of data
TRANSMISSION_ENERGY = 1.0

# Approximate energy consumed for compression processing
COMPRESSION_ENERGY = 0.2


# ============================================================
# DRL TRAINING SETTINGS
# ============================================================

RANDOM_SEED = 42

NUM_EPISODES = 100

LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.95

BATCH_SIZE = 32

MEMORY_SIZE = 10000

TARGET_UPDATE_FREQUENCY = 10


# ============================================================
# EXPLORATION SETTINGS
# ============================================================

EPSILON_START = 1.0

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.995


# ============================================================
# MODEL FILES
# ============================================================

DRL_MODEL_PATH = MODELS_DIR / "drl_compression_model.pth"

SCALER_PATH = MODELS_DIR / "scaler.pkl"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for directory in [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    GRAPHS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Sensor Dataset:", SENSOR_DATA_FILE)
    print("Metadata File:", METADATA_FILE)
    print("Models Directory:", MODELS_DIR)
    print("Results Directory:", RESULTS_DIR)
    print("Number of Actions:", NUM_ACTIONS)