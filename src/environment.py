import numpy as np

from config import (
    ENERGY_WEIGHT,
    SEMANTIC_WEIGHT,
    DISTORTION_WEIGHT,
    NUM_ACTIONS
)

from compression import evaluate_compression


# ============================================================
# DRL ENVIRONMENT
# ============================================================

class SemanticCompressionEnvironment:
    """
    Reinforcement Learning environment for
    adaptive semantic compression of IoT data.

    The agent observes a sensor feature vector,
    selects a compression action, and receives
    a reward based on:

        Energy Saving
        Semantic Preservation
        Compression Distortion
    """

    def __init__(self, feature_data):

        self.feature_data = np.asarray(
            feature_data,
            dtype=np.float32
        )

        if len(self.feature_data) == 0:
            raise ValueError(
                "Feature dataset is empty."
            )

        self.current_index = 0

        self.num_samples = len(
            self.feature_data
        )

        self.state_size = (
            self.feature_data.shape[1]
        )

        self.action_size = NUM_ACTIONS

    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(self):

        self.current_index = 0

        state = self.feature_data[
            self.current_index
        ]

        return state.copy()

    # ========================================================
    # GET CURRENT STATE
    # ========================================================

    def get_state(self):

        return self.feature_data[
            self.current_index
        ].copy()

    # ========================================================
    # CALCULATE REWARD
    # ========================================================

    def calculate_reward(
        self,
        energy,
        semantic_similarity,
        distortion
    ):
        """
        Reward function.

        Higher reward:
            - lower energy
            - higher semantic preservation
            - lower distortion
        """

        # Convert energy consumption
        # into energy saving
        energy_saving = 1.0 - energy

        # Normalize distortion approximately
        distortion_penalty = min(
            distortion,
            1.0
        )

        reward = (
            ENERGY_WEIGHT * energy_saving
            +
            SEMANTIC_WEIGHT * semantic_similarity
            -
            DISTORTION_WEIGHT * distortion_penalty
        )

        return float(reward)

    # ========================================================
    # TAKE ACTION
    # ========================================================

    def step(self, action):

        if action < 0 or action >= self.action_size:

            raise ValueError(
                f"Invalid action {action}. "
                f"Expected 0-{self.action_size - 1}."
            )

        # Current sensor feature vector
        original_data = self.feature_data[
            self.current_index
        ]

        # Apply selected compression
        result = evaluate_compression(
            original_data,
            action
        )

        # Extract metrics
        energy = result["energy"]

        semantic_similarity = (
            result["semantic_similarity"]
        )

        distortion = result["distortion"]

        compression_ratio = (
            result["compression_ratio"]
        )

        # Calculate reward
        reward = self.calculate_reward(
            energy,
            semantic_similarity,
            distortion
        )

        # Move to next sample
        self.current_index += 1

        done = (
            self.current_index
            >= self.num_samples
        )

        # Get next state
        if done:

            next_state = np.zeros(
                self.state_size,
                dtype=np.float32
            )

        else:

            next_state = self.feature_data[
                self.current_index
            ]

        # Information useful for evaluation
        info = {
            "energy": energy,
            "energy_saving": 1.0 - energy,
            "semantic_similarity": semantic_similarity,
            "distortion": distortion,
            "compression_ratio": compression_ratio
        }

        return (
            next_state.copy(),
            reward,
            done,
            info
        )

    # ========================================================
    # ENVIRONMENT INFORMATION
    # ========================================================

    def get_action_meanings(self):

        return {
            0: "No / Low Compression",
            1: "Low Compression",
            2: "Medium Compression",
            3: "High Compression"
        }


# ============================================================
# TEST ENVIRONMENT
# ============================================================

def main():

    print("=" * 60)
    print("SEMANTIC COMPRESSION DRL ENVIRONMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load semantic feature data
    # --------------------------------------------------------

    from config import PROCESSED_DATA_DIR

    feature_path = (
        PROCESSED_DATA_DIR /
        "train_semantic_features.npy"
    )

    if not feature_path.exists():

        raise FileNotFoundError(
            f"Semantic feature file not found:\n"
            f"{feature_path}\n\n"
            "Run semantic_features.py first."
        )

    features = np.load(
        feature_path
    )

    print(
        "\nFeature dataset shape:",
        features.shape
    )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = SemanticCompressionEnvironment(
        features
    )

    print(
        "State size:",
        env.state_size
    )

    print(
        "Number of actions:",
        env.action_size
    )

    print("\nAvailable actions:")

    for action, meaning in (
        env.get_action_meanings().items()
    ):

        print(
            f"{action} -> {meaning}"
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    state = env.reset()

    print(
        "\nInitial state shape:",
        state.shape
    )

    # --------------------------------------------------------
    # Test every action
    # --------------------------------------------------------

    print("\nTesting actions...")

    for action in range(
        env.action_size
    ):

        env.reset()

        next_state, reward, done, info = (
            env.step(action)
        )

        print("\n" + "-" * 50)

        print(
            "Action:",
            action
        )

        print(
            "Reward:",
            round(reward, 4)
        )

        print(
            "Energy:",
            round(
                info["energy"],
                4
            )
        )

        print(
            "Energy Saving:",
            round(
                info["energy_saving"],
                4
            )
        )

        print(
            "Semantic Similarity:",
            round(
                info["semantic_similarity"],
                4
            )
        )

        print(
            "Distortion:",
            round(
                info["distortion"],
                6
            )
        )

        print(
            "Compression Ratio:",
            round(
                info["compression_ratio"],
                4
            )
        )

    print("\n" + "=" * 60)
    print("ENVIRONMENT TEST COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()