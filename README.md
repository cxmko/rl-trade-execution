# rl-trade-execution

Optimal trade execution framework using reinforcement learning, trained on GARCH-simulated cryptocurrency price data. Minimizes market impact and price risk when executing large orders in volatile markets.

# Optimal Trade Execution with Reinforcement Learning

## Project overview

This project implements a reinforcement learning (RL) approach for optimal execution of large orders on cryptocurrency markets. The goal is to minimize market impact and maximize the value obtained when executing a large order by adapting the execution strategy to real-time market conditions.

## Problem statement

When a trader executes a large order, they face a trade-off:

* Executing quickly risks creating large market impact and degrading the execution price.
* Executing slowly exposes the trader to market risk (the price may move unfavorably).

This project aims to find the optimal compromise between these two risks using reinforcement learning.

## Approach

Our approach is characterized by:

1. **GARCH market modeling**: We use a GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) model calibrated on real data to simulate realistic price paths that capture the important phenomenon of volatility clustering.

2. **Training on simulated data**: To avoid data scarcity and allow safe exploration, the RL agent is trained on simulated price trajectories.

3. **Validation on real data**: Performance is then evaluated on historical real market data.

## Implemented components

### 1. Data collection and preprocessing

* Binance data collector to download BTCUSDT history
* Preprocessing and feature engineering

### 2. GARCH model and simulation

* Calibration of a GARCH(1,1) model on historical data
* Price path generator that reproduces the market’s essential statistical properties
* Calculation of realized volatility as a consistent measure across simulated and real data

## Theoretical foundations and design choices

### GARCH model

#### Mathematical formulation

The GARCH model (Generalized AutoRegressive Conditional Heteroskedasticity) is an extension of ARCH models developed by Bollerslev (1986). In its GARCH(1,1) form, the model is defined as:

For returns:
$$r_t = \mu + \sigma_t \cdot Z_t$$

For the conditional variance:
$$\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

Where:

* $r_t$ is the return at time $t$
* $\mu$ is the mean return (usually close to zero for high-frequency data)
* $\sigma_t$ is the conditional volatility at time $t$
* $Z_t$ is standardized white noise, typically distributed as $N(0,1)$
* $\omega, \alpha, \beta$ are parameters to estimate, with $\omega > 0$, $\alpha, \beta \ge 0$ and $\alpha + \beta < 1$ to ensure stationarity

#### Calibration

Parameter estimation is performed by maximum likelihood:

1. Returns are centered ($r_t - \mu$)
2. The log-likelihood is defined as:
   $$L(\omega, \alpha, \beta) = -\frac{1}{2}\sum_{t=1}^T \left(\log(\sigma_t^2) + \frac{(r_t-\mu)^2}{\sigma_t^2}\right)$$
3. The optimal parameters maximize this log-likelihood

We use 5,000 data points for calibration, which balances:

* Statistical accuracy (enough data)
* Local relevance (captures recent market regime)
* Computational efficiency (reasonable compute time)

#### Trajectory simulation

To generate price trajectories from the calibrated model:

1. Initialization:

   * Initial variance: $\sigma_0^2 = \omega/(1-\alpha-\beta)$ (unconditional variance)
   * Initial price: last observed value $P_0$

2. For each time step $t = 1, 2, \ldots$:

   * Sample $Z_t \sim N(0,1)$
   * Compute $\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$
   * Compute $r_t = \mu + \sigma_t \cdot Z_t$
   * Compute the new price: $P_t = P_{t-1} \cdot \exp(r_t)$

#### Realized volatility

To ensure consistency between the simulated environment and real data, we use an identical realized volatility measure in both contexts:

$$\text{realized volatility}*t = \sqrt{\frac{1}{n-1}\sum*{i=t-n+1}^{t} (r_i - \bar{r})^2}$$

Where:

* $n$ is the window size (typically 20 observations)
* $r_i$ are log returns
* $\bar{r}$ is the mean return within the window

### Absence of directional momentum

In our GARCH implementation there is no directional momentum in simulated prices for several reasons:

1. **Consistency with financial theory**: At intraday scales, liquid markets such as BTC/USDT show little autocorrelation in returns (weak-form market efficiency).

2. **Avoid overfitting**: An RL agent trained on data with artificial momentum might learn strategies that fail on real markets.

3. **Model robustness**: GARCH(1,1) effectively captures volatility clustering without introducing serial dependence in returns.

This approach was validated empirically by comparing the distributions of simulated and real returns, confirming that the model faithfully reproduces the market’s essential statistical properties.

### State structure for the RL agent

After analysis, we identified the most informative state variables for the RL agent:

1. **Normalized inventory** (remaining quantity to sell / total quantity)
2. **Normalized remaining time** (time left / total horizon)
3. **Realized volatility** (standard deviation of returns over a rolling window)
4. **Relative price** (current price / initial price)

This minimal yet complete representation avoids adding artificial features that would not be informative in a pure GARCH setting.

## Next steps

1. **RL environment development**:

   * Implement a Gymnasium-compatible environment
   * Model market impact proportional to volatility
   * Design the reward function

2. **Agent implementation**:

   * DQN or PPO architecture for learning
   * Action parameterization (percentage of inventory to execute)
   * Appropriate exploration strategies

3. **Evaluation and optimization**:

   * Comparison with benchmark strategies (TWAP, VWAP)
   * Sensitivity analysis to model parameters
   * Out-of-sample tests on market data

## Project structure

```
rl-trade-execution/
├── data/                 # Raw and processed data
├── src/
│   ├── data/             # Collection and preprocessing modules
│   ├── environment/      # GARCH simulator and RL environment
│   └── models/           # RL agent implementations
├── scripts/              # Analysis and test scripts
├── sample/               # Example simulated data
└── notebooks/            # Exploratory analyses
```

