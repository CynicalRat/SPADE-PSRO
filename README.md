# PSRO-PPO Project

## Overview
This project implements the Policy Space Response Oracles (PSRO) algorithm using Proximal Policy Optimization (PPO) as the best response strategy. The framework is designed to facilitate self-play in multi-agent environments, allowing for the evaluation and improvement of policies through iterative training.

## Project Structure
```
psro-ppo-project
├── src
│   ├── algorithms
│   │   ├── __init__.py 
│   │   ├── psro.py
│   │   └── ppo_oracle.py
│   ├── envs
│   │   ├── __init__.py
│   │   └── self_play_wrapper.py  
│   ├── utils
│   │   ├── __init__.py
│   │   ├── payoff_table.py
│   │   └── meta_solver.py
│   └── main.py
├── configs
│   └── psro_config.yaml
├── results
│   └── metrics
├── tests
│   └── __init__.py
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd psro-ppo-project
pip install -r requirements.txt
```

## Usage
To run the PSRO algorithm, execute the main script:

```bash
python src/main.py
```

## Configuration
The configuration for the PSRO algorithm can be found in `configs/psro_config.yaml`. Modify this file to adjust parameters for the PPO oracle, training settings, and environment specifications.

## Algorithms
- **PSRO**: Implemented in `src/algorithms/psro.py`, this class manages the overall PSRO process, including policy evaluation and updating the policy pool.
- **PPO Oracle**: Implemented in `src/algorithms/ppo_oracle.py`, this class integrates with Stable Baselines3 to provide a PPO-based policy for the PSRO algorithm.

## Environments
- **Self-Play Wrapper**: Defined in `src/envs/self_play_wrapper.py`, this class manages interactions between the agent and its opponent, supporting self-play.

## Utilities
- **Payoff Table**: Managed in `src/utils/payoff_table.py`, this class handles the storage and retrieval of payoff information for different strategies.
- **Meta Solver**: Implemented in `src/utils/meta_solver.py`, this class is responsible for solving the meta-game and finding optimal strategies based on the current policies.

## Testing
Unit tests can be added in the `tests` directory. Ensure to run tests to validate the implementation.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Stable Baselines3 for providing a robust reinforcement learning library.
- Contributions from the community for enhancing the PSRO framework.