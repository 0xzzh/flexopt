import numpy as np

from scipy.stats import linregress


def calibrate_parameters(x, steps_per_year):
    dt = 1.0 / steps_per_year

    curr_x = x[:-1]
    next_x = x[1:]
    
    beta, alpha, _, _, _ = linregress(curr_x, next_x)
    
    kappa = -np.log(beta) / dt
    theta = alpha / (1 - beta)
    
    residuals = next_x - (alpha + beta * curr_x)
    sigma = np.std(residuals, ddof=2) * np.sqrt(2 * kappa / (1 - beta**2))

    return kappa, theta, sigma


def simulate(s, kappa, theta, sigma, steps_per_year, steps, trials, random_seed):
    """Simulates price with Ornstein-Uhlenbeck process

    Args:
        s: Current price.
        kappa: Mean reversion speed.
        theta: Long term mean reversion level.
        steps_per_year: Trading days (or other time units, for example hours) per year.
        steps: Simulation steps.
        trials: Scenario count.
        random_seed: Random seed to generate random numbers.
        
    Returns:
        A numpy array of shape (T+1, N) to corresponding simulated prices.

        Each row represents one time step. The first row is the current time step with current price s
      
        Each column represents one price scenario.

    Raises:
        ValueError: An error occurred when model input paramters are out of range.
    """
    if kappa <= 0:
        raise ValueError('Paramter incorrect: kappa (mean reversion speed) = {kappa}. It should be > 0'.format(kappa=kappa))

    if sigma <= 0:
        raise ValueError('Paramter incorrect: sigma (annualised volatility) = {sigma}. It should be > 0'.format(sigma=sigma))

    np.random.seed(random_seed)

    dt = 1.0 / steps_per_year

    epsilon = np.random.randn(steps, trials)
    
    x = np.zeros(shape=(steps + 1, trials))
    x[0, :] = s

    df = np.exp(-kappa * dt)
    std = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    
    for t in range(0, steps):
        x[t+1, :] = theta + df * (x[t, :] - theta) + std * epsilon[t, :]

    return x
