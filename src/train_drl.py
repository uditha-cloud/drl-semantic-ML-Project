import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    PROCESSED_DATA_DIR,
    DRL_MODEL_PATH,
    RANDOM_SEED,
    NUM_EPISODES,
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    BATCH_SIZE,
    MEMORY_SIZE,
    TARGET_UPDATE_FREQUENCY,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    NUM_ACTIONS
)

from environment import SemanticCompressionEnvironment


# ============================================================
# SETTINGS FOR FASTER CPU TRAINING
# ============================================================

SAMPLES_PER_EPISODE = 1000


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DQN NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "\nTraining device:",
            self.device
        )

        # Policy network
        self.policy_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Target network
        self.target_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=LEARNING_RATE
        )

        # Replay memory
        self.memory = ReplayMemory(
            MEMORY_SIZE
        )

        self.epsilon = EPSILON_START

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.from_numpy(
            np.asarray(
                state,
                dtype=np.float32
            )
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return int(
            torch.argmax(q_values).item()
        )

    # ========================================================
    # TRAIN ONE BATCH
    # ========================================================

    def train_step(self):

        if len(self.memory) < BATCH_SIZE:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            BATCH_SIZE
        )

        states = torch.from_numpy(
            states
        ).to(self.device)

        actions = torch.from_numpy(
            actions
        ).to(self.device)

        rewards = torch.from_numpy(
            rewards
        ).to(self.device)

        next_states = torch.from_numpy(
            next_states
        ).to(self.device)

        dones = torch.from_numpy(
            dones
        ).to(self.device)

        # Current Q values
        current_q = (
            self.policy_network(states)
            .gather(
                1,
                actions.unsqueeze(1)
            )
            .squeeze(1)
        )

        # Target Q values
        with torch.no_grad():

            next_q = (
                self.target_network(next_states)
                .max(dim=1)[0]
            )

            target_q = (
                rewards
                +
                DISCOUNT_FACTOR
                * next_q
                * (1.0 - dones)
            )

        # Loss
        loss = nn.MSELoss()(
            current_q,
            target_q
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    # ========================================================
    # UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(self):

        DRL_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            self.policy_network.state_dict(),
            DRL_MODEL_PATH
        )

        print(
            "\nModel saved to:",
            DRL_MODEL_PATH
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    feature_path = (
        PROCESSED_DATA_DIR /
        "train_semantic_features.npy"
    )

    if not feature_path.exists():

        raise FileNotFoundError(
            f"Training features not found:\n"
            f"{feature_path}\n\n"
            "Run semantic_features.py first."
        )

    features = np.load(
        feature_path
    )

    return features


# ============================================================
# TRAIN DQN
# ============================================================

def train():

    print("=" * 60)
    print("DEEP REINFORCEMENT LEARNING")
    print("DQN-BASED ADAPTIVE SEMANTIC COMPRESSION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    features = load_training_data()

    print(
        "\nFull training data:",
        features.shape
    )

    # --------------------------------------------------------
    # Limit samples per episode
    # --------------------------------------------------------

    samples_per_episode = min(
        SAMPLES_PER_EPISODE,
        len(features)
    )

    print(
        "Samples per episode:",
        samples_per_episode
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = SemanticCompressionEnvironment(
        features[:samples_per_episode]
    )

    state_size = env.state_size
    action_size = env.action_size

    print(
        "State size:",
        state_size
    )

    print(
        "Action size:",
        action_size
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size,
        action_size
    )

    episode_rewards = []

    print("\nStarting faster DQN training...\n")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        # Randomly choose a starting point
        if len(features) > SAMPLES_PER_EPISODE:

            start_index = random.randint(
                0,
                len(features) - SAMPLES_PER_EPISODE
            )

            episode_features = features[
                start_index:
                start_index + SAMPLES_PER_EPISODE
            ]

            env = SemanticCompressionEnvironment(
                episode_features
            )

        state = env.reset()

        total_reward = 0.0
        losses = []

        done = False

        while not done:

            action = agent.select_action(
                state
            )

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            state = next_state

            total_reward += reward

        # Update target network
        if episode % TARGET_UPDATE_FREQUENCY == 0:

            agent.update_target_network()

        # Reduce exploration
        agent.epsilon = max(
            EPSILON_MIN,
            agent.epsilon * EPSILON_DECAY
        )

        episode_rewards.append(
            total_reward
        )

        average_loss = (
            np.mean(losses)
            if losses
            else 0.0
        )

        print(
            f"Episode {episode:03d} | "
            f"Reward: {total_reward:9.3f} | "
            f"Loss: {average_loss:.6f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    agent.save()

    # --------------------------------------------------------
    # Save rewards
    # --------------------------------------------------------

    rewards_path = (
        PROCESSED_DATA_DIR /
        "training_rewards.npy"
    )

    np.save(
        rewards_path,
        np.asarray(
            episode_rewards,
            dtype=np.float32
        )
    )

    print(
        "\nTraining rewards saved to:",
        rewards_path
    )

    print("\n" + "=" * 60)
    print("DQN TRAINING COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()