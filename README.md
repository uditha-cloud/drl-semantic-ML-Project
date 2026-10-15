# Deep Reinforcement Learning-Based Adaptive Semantic Compression for Energy-Constrained IoT Communication

## 1. Project Overview

This project implements a Deep Reinforcement Learning (DRL)-based adaptive semantic compression system for energy-constrained Internet of Things (IoT) communication.

The system processes sensor time-series data, extracts meaningful semantic features, and uses a reinforcement learning agent to select suitable compression strategies. The main goal is to reduce data transmission costs and energy consumption while preserving important information.

## 2. Objectives

- Reduce the amount of data transmitted by IoT devices.
- Implement adaptive semantic compression using Deep Reinforcement Learning.
- Minimize estimated communication energy consumption.
- Preserve important information in sensor data.
- Evaluate the performance of the trained DRL agent.

## 3. Project Structure

```text
DRL-Semantic-Compression/
├── dataset/
├── models/
├── results/
└── src/
    ├── compression.py
    ├── config.py
    ├── environment.py
    ├── evaluate.py
    ├── load_dataset.py
    ├── main.py
    ├── plots.py
    ├── preprocess.py
    ├── semantic_features.py
    └── train_drl.py
├── README.md
└── requirements.txt
```

## 4. Technologies Used

- **Programming Language:** Python
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **Reinforcement Learning:** Gymnasium, Stable-Baselines3
- **Visualization:** Matplotlib

## 5. System Workflow

1. Load the sensor dataset.
2. Preprocess and normalize the sensor data.
3. Divide the time-series data into fixed-size windows.
4. Extract semantic features from the windows.
5. Apply adaptive semantic compression.
6. Train a DRL agent to select compression actions.
7. Evaluate the compression and energy performance.
8. Generate plots and save the results.

## 6. Deep Reinforcement Learning

### State

The state may include sensor features, remaining energy, channel conditions, and previous compression performance.

### Actions

The agent selects an appropriate compression level based on the current state.

### Reward Function

The reward function balances energy consumption, transmission cost, and reconstruction error.

A possible formulation is:

`Reward = -(α × Energy Cost + β × Reconstruction Error + γ × Transmission Cost)`

Here, α, β, and γ are weights that control the importance of each objective.

## 7. Installation

Install Python and open the project directory in VS Code.

Install the required packages:

```bash
pip install -r requirements.txt
```

## 8. Running the Project

Run the scripts from the project root directory in the appropriate order:

```bash
python src/load_dataset.py
python src/preprocess.py
python src/semantic_features.py
python src/train_drl.py
python src/evaluate.py
```

If `src/main.py` executes the complete pipeline, use:

```bash
python src/main.py
```

The exact execution order depends on how the scripts are implemented.

## 9. Evaluation Metrics

The project can evaluate the following metrics:

- Compression ratio
- Reconstruction error
- Estimated energy consumption
- Transmission cost
- Average DRL reward
- Data preservation performance

## 10. Expected Results

The proposed system aims to select compression levels dynamically according to sensor data and communication conditions.

The performance is evaluated by comparing adaptive compression against fixed compression strategies. Actual improvements must be confirmed through experimental results.

## 11. Applications

- Wireless sensor networks
- Smart agriculture
- Industrial IoT
- Environmental monitoring
- Battery-powered IoT devices
- Remote sensor monitoring

## 12. Future Enhancements

- Integrate the system with ESP32-based IoT devices.
- Test using real wireless communication measurements.
- Optimize the DRL model for edge deployment.
- Evaluate performance on additional sensor datasets.
- Investigate lightweight semantic compression algorithms.

## 13. Conclusion

This project explores Deep Reinforcement Learning for adaptive semantic compression in energy-constrained IoT communication. It aims to balance data reduction, information preservation, and communication energy consumption through intelligent compression decisions.