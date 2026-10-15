import numpy as np

from config import COMPRESSION_LEVELS


# ============================================================
# COMPRESSION LEVEL
# ============================================================

def get_compression_ratio(action):
    """
    Return the amount of information retained
    for a given compression action.

    Action 0 -> 100% retained
    Action 1 -> 75% retained
    Action 2 -> 50% retained
    Action 3 -> 25% retained
    """

    if action not in COMPRESSION_LEVELS:
        raise ValueError(
            f"Invalid compression action: {action}"
        )

    return COMPRESSION_LEVELS[action]


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_data(data, levels):
    """
    Quantize continuous sensor features.

    Higher compression means fewer quantization levels.
    """

    data = np.asarray(data, dtype=np.float32)

    minimum = np.min(data)
    maximum = np.max(data)

    if maximum == minimum:
        return data.copy()

    normalized = (
        (data - minimum) /
        (maximum - minimum)
    )

    quantized = np.round(
        normalized * (levels - 1)
    ) / (levels - 1)

    reconstructed = (
        quantized *
        (maximum - minimum)
        + minimum
    )

    return reconstructed.astype(np.float32)


# ============================================================
# APPLY COMPRESSION
# ============================================================

def compress_data(data, action):
    """
    Apply compression according to the selected action.

    Action 0 -> Low/no compression
    Action 1 -> Mild compression
    Action 2 -> Medium compression
    Action 3 -> High compression
    """

    retention_ratio = get_compression_ratio(action)

    if action == 0:
        levels = 256

    elif action == 1:
        levels = 128

    elif action == 2:
        levels = 64

    elif action == 3:
        levels = 16

    else:
        raise ValueError(
            f"Invalid compression action: {action}"
        )

    compressed_data = quantize_data(
        data,
        levels
    )

    return compressed_data


# ============================================================
# CALCULATE COMPRESSION RATIO
# ============================================================

def calculate_compression_ratio(
    original_data,
    compressed_data,
    action
):
    """
    Estimate the effective compression ratio.
    """

    original_size = original_data.nbytes

    retention_ratio = get_compression_ratio(action)

    estimated_compressed_size = (
        original_size * retention_ratio
    )

    if estimated_compressed_size == 0:
        return 0.0

    compression_ratio = (
        original_size /
        estimated_compressed_size
    )

    return compression_ratio


# ============================================================
# CALCULATE DISTORTION
# ============================================================

def calculate_distortion(
    original_data,
    compressed_data
):
    """
    Calculate Mean Squared Error between
    original and compressed data.
    """

    original_data = np.asarray(
        original_data,
        dtype=np.float32
    )

    compressed_data = np.asarray(
        compressed_data,
        dtype=np.float32
    )

    mse = np.mean(
        (original_data - compressed_data) ** 2
    )

    return float(mse)


# ============================================================
# SEMANTIC PRESERVATION
# ============================================================

def calculate_semantic_similarity(
    original_data,
    compressed_data
):
    """
    Estimate how much semantic information
    is preserved after compression.

    Similarity is based on normalized MSE.
    """

    original_data = np.asarray(
        original_data,
        dtype=np.float32
    )

    compressed_data = np.asarray(
        compressed_data,
        dtype=np.float32
    )

    mse = np.mean(
        (original_data - compressed_data) ** 2
    )

    variance = np.var(original_data)

    if variance == 0:
        return 1.0

    similarity = 1.0 - (
        mse / (variance + 1e-8)
    )

    similarity = np.clip(
        similarity,
        0.0,
        1.0
    )

    return float(similarity)


# ============================================================
# ESTIMATE ENERGY
# ============================================================

def estimate_energy(
    original_data,
    action
):
    """
    Estimate transmission energy.

    More compression means less data transmitted,
    therefore lower transmission energy.
    """

    original_size = original_data.nbytes

    retention_ratio = get_compression_ratio(action)

    transmitted_size = (
        original_size *
        retention_ratio
    )

    energy = transmitted_size / original_size

    return float(energy)


# ============================================================
# COMPLETE COMPRESSION EVALUATION
# ============================================================

def evaluate_compression(
    original_data,
    action
):
    """
    Compress data and calculate all important
    metrics required by the DRL environment.
    """

    compressed_data = compress_data(
        original_data,
        action
    )

    compression_ratio = calculate_compression_ratio(
        original_data,
        compressed_data,
        action
    )

    distortion = calculate_distortion(
        original_data,
        compressed_data
    )

    semantic_similarity = calculate_semantic_similarity(
        original_data,
        compressed_data
    )

    energy = estimate_energy(
        original_data,
        action
    )

    return {
        "compressed_data": compressed_data,
        "compression_ratio": compression_ratio,
        "distortion": distortion,
        "semantic_similarity": semantic_similarity,
        "energy": energy
    }


# ============================================================
# TEST COMPRESSION
# ============================================================

def main():

    print("=" * 60)
    print("ADAPTIVE SEMANTIC COMPRESSION")
    print("=" * 60)

    # Example sensor feature vector
    sample_data = np.array(
        [
            0.15,
            0.32,
            0.47,
            0.51,
            0.63,
            0.72,
            0.81,
            0.92
        ],
        dtype=np.float32
    )

    print("\nOriginal data:")
    print(sample_data)

    print("\nTesting compression levels...\n")

    for action in range(4):

        result = evaluate_compression(
            sample_data,
            action
        )

        print("-" * 50)

        print(
            "Action:",
            action
        )

        print(
            "Compression ratio:",
            round(
                result["compression_ratio"],
                3
            )
        )

        print(
            "Semantic similarity:",
            round(
                result["semantic_similarity"],
                4
            )
        )

        print(
            "Distortion:",
            round(
                result["distortion"],
                6
            )
        )

        print(
            "Estimated energy:",
            round(
                result["energy"],
                3
            )
        )

    print("\nCompression module working successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()