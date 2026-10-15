import numpy as np
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DATA_DIR,
    GRAPHS_DIR
)


# ============================================================
# LOAD TRAINING REWARDS
# ============================================================

def load_training_rewards():

    rewards_path = (
        PROCESSED_DATA_DIR /
        "training_rewards.npy"
    )

    if not rewards_path.exists():
        raise FileNotFoundError(
            f"Training rewards not found:\n"
            f"{rewards_path}\n\n"
            "Run train_drl.py first."
        )

    return np.load(rewards_path)


# ============================================================
# PLOT TRAINING REWARD
# ============================================================

def plot_training_reward():

    rewards = load_training_rewards()

    episodes = np.arange(
        1,
        len(rewards) + 1
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        episodes,
        rewards
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Total Reward"
    )

    plt.title(
        "DQN Training Reward"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        GRAPHS_DIR /
        "training_reward.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# PLOT SMOOTHED REWARD
# ============================================================

def plot_smoothed_reward():

    rewards = load_training_rewards()

    window = min(
        10,
        len(rewards)
    )

    if window < 2:
        return

    smoothed = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    episodes = np.arange(
        window,
        len(rewards) + 1
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        episodes,
        smoothed
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.title(
        "Smoothed DQN Training Reward"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        GRAPHS_DIR /
        "smoothed_training_reward.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# CREATE COMPRESSION COMPARISON GRAPH
# ============================================================

def plot_compression_comparison():

    actions = [
        "No/Low",
        "Low",
        "Medium",
        "High"
    ]

    # Approximate retention levels used by
    # our compression module.
    energy = [
        1.00,
        0.75,
        0.50,
        0.25
    ]

    energy_saving = [
        0.00,
        0.25,
        0.50,
        0.75
    ]

    x = np.arange(
        len(actions)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        energy,
        width,
        label="Energy Consumption"
    )

    plt.bar(
        x + width / 2,
        energy_saving,
        width,
        label="Energy Saving"
    )

    plt.xticks(
        x,
        actions
    )

    plt.xlabel(
        "Compression Level"
    )

    plt.ylabel(
        "Normalized Value"
    )

    plt.title(
        "Compression Level vs Energy"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        GRAPHS_DIR /
        "compression_energy_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# PLOT SEMANTIC FEATURE DISTRIBUTION
# ============================================================

def plot_feature_distribution():

    feature_path = (
        PROCESSED_DATA_DIR /
        "train_semantic_features.npy"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Semantic features not found:\n"
            f"{feature_path}"
        )

    features = np.load(
        feature_path
    )

    # Average feature value for each
    # feature position.
    mean_features = np.mean(
        features,
        axis=0
    )

    feature_indices = np.arange(
        len(mean_features)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        feature_indices,
        mean_features,
        marker="o"
    )

    plt.xlabel(
        "Feature Index"
    )

    plt.ylabel(
        "Mean Normalized Value"
    )

    plt.title(
        "Semantic Feature Distribution"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        GRAPHS_DIR /
        "semantic_feature_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING PROJECT GRAPHS")
    print("=" * 60)

    # Training performance
    plot_training_reward()

    plot_smoothed_reward()

    # Compression and energy
    plot_compression_comparison()

    # Semantic features
    plot_feature_distribution()

    print("\n" + "=" * 60)
    print("ALL GRAPHS GENERATED SUCCESSFULLY")
    print("=" * 60)

    print("\nGraphs saved inside:")
    print(GRAPHS_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()