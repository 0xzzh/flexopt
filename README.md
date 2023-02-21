# FlexOpt

Valuations and optimizations of energy flexibility options in Python.

The FlexOpt code base is clean and simple. It is written in the Functional Programming style. Users can easily compose functions in the library for the tasks of valuations and optimizations.

## Installation

```bash
pip install flexopt
```

## Features

Price Simulations

| Price Model | Scenarios | Multi-asset | Fundamental Variables |
|---|:---:|---|---|
| [Geometric Brownian motion](src/flexopt/price/geometric_brownian_motion.py) | :heavy_check_mark: |  |
| [Mean reversion process](src/flexopt/price/ou_process.py) | :heavy_check_mark: | |  |
| Correlated geometric Brownian motion  | :heavy_check_mark: | Correlation of log returns |  |
| VECM (Vector Error Correction Model) | :heavy_check_mark: | Cointegrated time series |  |
| ARIMAX | :heavy_check_mark: |  | Exogenous variables |

Valuations and optimizations

| Flexibility | Valuation | Dispatch Schedule | Dynamic Programming | Reinforcement Learning |
|---|:---:|:---:|:---:|:---:|
| [Swing options with variable volumes](src/flexopt/flex/swing_option.py) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| BESS (Battery Energy Storage System) | :heavy_check_mark: | :heavy_check_mark: |  |  |

## Example

```python
from flexopt.price import geometric_brownian_motion

price = 100
interest_rate = 0.05
sigma = 0.25
steps_per_year = 365
steps = 730
trials = 100000
random_seed = 1

sim_prices = geometric_brownian_motion.simulate(
    s=price, 
    r=interest_rate, 
    sigma=sigma, 
    steps_per_year=steps_per_year, 
    steps=steps, 
    trials=trials, 
    random_seed=random_seed
)
```

## More Examples

Examples with detailed explanations and graphs are availalbe on GitHub: https://github.com/0xzzh/flexopt-examples
