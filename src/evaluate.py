import numpy as np
import torch

from config import (
    PROCESSED_DATA_DIR,
    DRL_MODEL_PATH,
    NUM_ACTIONS
)

from environment import SemanticCompressionEnvironment
from train_drl import DQN


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    test_path = (
        PROCESSED_DATA_DIR /
        "test_semantic_features.npy"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test features not found:\n{test_path}\n\n"
            "Run semantic_features.py first."
        )

    return np.load(test_path)


# ============================================================
# LOAD TRAINED DQN
# ============================================================

def load_model(state_size):

    if not DRL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{DRL_MODEL_PATH}\n\n"
            "Run train_drl.py first."
        )

    model = DQN(
        state_size,
        NUM_ACTIONS
    )

    model.load_state_dict(
        torch.load(
            DRL_MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# ============================================================
# EVALUATE FIXED ACTION
# ============================================================

def evaluate_fixed_action(
    features,
    action
):

    env = SemanticCompressionEnvironment(
        features
    )

    state = env.reset()

    total_reward = 0.0

    total_energy = 0.0
    total_energy_saving = 0.0
    total_similarity = 0.0
    total_distortion = 0.0
    total_compression_ratio = 0.0

    count = 0

    done = False

    while not done:

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        total_reward += reward

        total_energy += info["energy"]

        total_energy_saving += (
            info["energy_saving"]
        )

        total_similarity += (
            info["semantic_similarity"]
        )

        total_distortion += (
            info["distortion"]
        )

        total_compression_ratio += (
            info["compression_ratio"]
        )

        count += 1

        state = next_state

    return {
        "reward": total_reward / count,
        "energy": total_energy / count,
        "energy_saving": total_energy_saving / count,
        "semantic_similarity": total_similarity / count,
        "distortion": total_distortion / count,
        "compression_ratio": (
            total_compression_ratio / count
        )
    }


# ============================================================
# EVALUATE DQN
# ============================================================

def evaluate_dqn(
    features,
    model
):

    env = SemanticCompressionEnvironment(
        features
    )

    state = env.reset()

    total_reward = 0.0

    total_energy = 0.0
    total_energy_saving = 0.0
    total_similarity = 0.0
    total_distortion = 0.0
    total_compression_ratio = 0.0

    action_counts = np.zeros(
        NUM_ACTIONS,
        dtype=int
    )

    count = 0

    done = False

    while not done:

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = model(
                state_tensor
            )

        action = int(
            torch.argmax(
                q_values
            ).item()
        )

        action_counts[action] += 1

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        total_reward += reward

        total_energy += info["energy"]

        total_energy_saving += (
            info["energy_saving"]
        )

        total_similarity += (
            info["semantic_similarity"]
        )

        total_distortion += (
            info["distortion"]
        )

        total_compression_ratio += (
            info["compression_ratio"]
        )

        count += 1

        state = next_state

    return {
        "reward": total_reward / count,
        "energy": total_energy / count,
        "energy_saving": total_energy_saving / count,
        "semantic_similarity": total_similarity / count,
        "distortion": total_distortion / count,
        "compression_ratio": (
            total_compression_ratio / count
        ),
        "action_counts": action_counts
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_result(
    name,
    result
):

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(
        "Average Reward:",
        round(result["reward"], 4)
    )

    print(
        "Energy Consumption:",
        round(result["energy"], 4)
    )

    print(
        "Energy Saving:",
        round(result["energy_saving"], 4)
    )

    print(
        "Semantic Similarity:",
        round(result["semantic_similarity"], 4)
    )

    print(
        "Distortion:",
        round(result["distortion"], 6)
    )

    print(
        "Compression Ratio:",
        round(result["compression_ratio"], 4)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DRL MODEL EVALUATION")
    print("ADAPTIVE SEMANTIC COMPRESSION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test features
    # --------------------------------------------------------

    features = load_test_data()

    print(
        "\nTest feature shape:",
        features.shape
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model(
        features.shape[1]
    )

    print(
        "Trained DQN loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluate fixed compression actions
    # --------------------------------------------------------

    results = {}

    for action in range(NUM_ACTIONS):

        result = evaluate_fixed_action(
            features,
            action
        )

        results[
            f"Action {action}"
        ] = result

        print_result(
            f"Fixed Compression - Action {action}",
            result
        )

    # --------------------------------------------------------
    # Evaluate DQN
    # --------------------------------------------------------

    dqn_result = evaluate_dqn(
        features,
        model
    )

    print_result(
        "DRL Adaptive Compression",
        dqn_result
    )

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DRL ACTION SELECTION")
    print("=" * 60)

    action_names = {
        0: "No / Low Compression",
        1: "Low Compression",
        2: "Medium Compression",
        3: "High Compression"
    }

    for action in range(NUM_ACTIONS):

        count = dqn_result[
            "action_counts"
        ][action]

        percentage = (
            count /
            len(features)
        ) * 100

        print(
            f"Action {action} "
            f"({action_names[action]}): "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()