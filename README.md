

# Optimal Trade Execution with Deep Reinforcement Learning

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Project Status:** **Finished.** The final PPO agent successfully outperforms the TWAP benchmark in 87.4% of Bear Market scenarios on out-of-sample data (2024).

## Overview

This project implements a **Deep Reinforcement Learning (DRL)** agent designed for the optimal execution of large institutional orders (liquidation) in cryptocurrency markets (BTC/USDT).

The goal is to solve the classic **Impact-Time Risk trade-off**:
*   **Selling too fast** incurs high market impact (slippage).
*   **Selling too slow** exposes the portfolio to market volatility and potential crashes.

Unlike traditional algorithmic trading strategies (like TWAP) that follow a rigid schedule, our AI agent adapts dynamically to real-time market conditions (liquidity, volatility, and price trends) to minimize **Implementation Shortfall (IS)** and reduce **Tail Risk (CVaR)**.

## Key Results

The final model (`ppo_execution_best_win_rate_s1_b.pth`) was evaluated on a full year of out-of-sample data (2024).

| Metric | TWAP (Benchmark) | AI Agent (PPO) | Improvement |
| :--- | :--- | :--- | :--- |
| **Bear Market Win Rate** | N/A | **87.41%** | The Agent beats TWAP in 9 out of 10 crashes. |
| **Tail Risk (CVaR 5%)** | -79.82 bps | **-12.62 bps** | **6.3x reduction** in extreme losses. |
| **Avg Impact Cost** | 31.64 bps | **31.12 bps** | Lower impact despite faster execution. |
| **General Win Rate** | N/A | 34.89% | The agent pays a small "insurance premium" in bull markets. |

> **Conclusion:** The agent behaves like a "Smart Insurer." It accepts a negligible opportunity cost in calm markets to provide massive downside protection during liquidity crises.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/rl-trade-execution.git
    cd rl-trade-execution
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    The project includes a script to fetch Binance data.
    ```bash
    python main.py
    ```

## Usage Guide

### 1. Interactive Demo (Recommended)
To visualize the agent in action against specific scenarios (Flash Crash, Pump & Dump, Low Liquidity), run the Jupyter Notebook:
```bash
jupyter notebook notebooks/demo_execution.ipynb
```

### 2. Evaluate the Model
To reproduce the results from the report on the 2024 Test Set:
```bash
# Run full sequential backtest on real data
python scripts/eval-tr.py --model models/ppo_execution_best_win_rate_s1_b.pth --full --real
```
*   `--real`: Uses historical data instead of GARCH simulation.
*   `--full`: Runs a sliding window over the entire year (2,000+ episodes).

### 3. Train a New Agent
To train the PPO agent from scratch using the GARCH Simulator:
```bash
python scripts/train_ppo.py --episodes 10000 --lr 0.00005
```

## Project Structure

```
rl-trade-execution/
├── .gitignore
├── LICENSE
├── main.py                         # Entry point for downloading Binance data
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── data/
│   └── raw/                        # Historical market data (CSV)
│       ├── BTCUSDT_1m_test_2024... # 2024 Out-of-sample Test Data
│       └──BTCUSDT_1m_train_2023...# 2023 Training Data
├── models/                         # Trained PyTorch Models
│   ├── dqn_execution_v2.pth        # Baseline DQN Agent
│   ├── ppo_execution_best_median_cap0.pth
│   └── ppo_execution_best_win_rate_s1_b.pth # FINAL CHAMPION MODEL
├── notebooks/                      # Interactive Analysis
│   └── demo_execution.ipynb        # Interactive Agent Demo (Visualization)
│   
├── report/                         # Final Report
│   ├── Final Report.pdf            # Final Report
│   └── sample/                     # Generated figures for the report
├── sample/                         # Training Artifacts
│   
├── scripts/                        # Executable Scripts
│   ├── analyze_volume.py           # Market Microstructure & Volume Analysis
│   ├── eval-tr.py                  #  Evaluation Pipeline (PPO)
│   ├── eval-trdqn.py               # Evaluation Pipeline (DQN)
│   ├── generate_data_analysis_plots.py
│   ├── test_garch.py               # GARCH Simulator Unit Tests
│   ├── test_impact.py              # Market Impact Model Verification
│   ├── train_dqn.py                # DQN Training Loop
│   ├── train_ppo.py                # PPO Training Loop
│   └── logs/                       # Detailed Evaluation Logs            
│       
└── src/                            # Core Logic Library
    ├── data/
    │   └── collectors/
    │       └── binance_data.py     # Binance API Data Collector
    ├── environment/                # RL Environments
    │   ├── execution_env.py        # Main Trading Environment (PPO)
    │   ├── execution_envdqn.py     # Environment variant for DQN
    │   └── garch_simulator.py      # "Nightmare" Market Simulator
    └── models/                     # Neural Network Architectures
        ├── dqn_agent.py            # Deep Q-Network Implementation
        └── ppo_agent.py            # PPO Actor-Critic Implementation
```

## Methodology

### 1. The Environment (execution_env.py)
We modeled the execution problem as a Markov Decision Process (MDP):
* **State Space ($S_t$):** 9-dimensional vector including Inventory, Time Remaining, Liquidity Score, Price Trends, and **Volatility Lags** (to detect regime shifts).
* **Action Space ($A_t$):** Discrete percentages of remaining inventory to sell $\{0\\%, 1\\%, \dots, 100\\%\}$.
* **Market Impact:** Modeled using the **Square-Root Law** calibrated on Binance order book depth ($\lambda=0.003$).
* **Reward Function:** A **Symmetric Reward** minimizing Tracking Error against the Arrival Price. This penalizes both holding inventory too long (risk) and selling too fast (impact).

### 2. The "Nightmare" Simulator (`src/environment/garch_simulator.py`)
To prevent overfitting to historical dates, we trained the agent on a **GARCH(1,1) Simulator** with Student-t innovations.
*   **Purpose:** Generates infinite synthetic market data with "fat tails" (extreme crashes) and volatility clustering.
*   **Result:** The agent trained on this "nightmare" data generalized perfectly to the unseen real-world data of 2024.

### 3. The Algorithm (`src/models/ppo_agent.py`)
We used **Proximal Policy Optimization (PPO)**, an Actor-Critic method, for its stability in stochastic environments.
*   **Actor:** Decides the execution schedule.
*   **Critic:** Estimates the expected cost of the current state.
*   **Curriculum Learning:** We used random start times during training to force the agent to learn end-of-day liquidation logic early on.

## Evolution of the Agent

The project went through several iterations detailed in the report:
1.  **DQN Baseline:** Learned to sell, but lacked stability.
2.  **"Lazy" Agent:** Used a Capped Reward ($min(0, P - P_0)$). It learned to do nothing in bull markets, failing to beat TWAP.
3.  **Final Agent:** Used **Symmetric Tracking Error**. This forced the agent to treat "Time as Toxic," leading to the robust, risk-averse behavior observed in the final results.

## Requirements

*   Python 3.8+
*   PyTorch
*   Gymnasium
*   Pandas / NumPy
*   Matplotlib / Seaborn
*   Arch (for GARCH modeling)

See `requirements.txt` for exact versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Authors: Cameron Mouangue, Guicheney Jacques - Institut Polytechnique de Paris*
