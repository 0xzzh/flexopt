import numpy as np


def simulate(s, r, sigma, steps_per_year, steps, trials, random_seed):
    """Simulate prices with geometric Brownian motion

    Args:
        s: Current price.
        r: Long term percent drift.
        sigma: Annualised volatility.
        steps_per_year: Trading days (or other time units, for example hours) per year.
        steps: Simulation steps.
        trials: Scenario count.
        random_seed: Random seed to generate random numbers.
      
    Returns:
        A numpy array of [T+1, N] size to corresponding simulated prices.

        Each row represents one time step. The first row is the current time step with current price s
      
        Each column represents one price scenario.

    Raises:
        ValueError: An error occurred when model input paramters are out of range.
    """
    if s <= 0:
        raise ValueError('Paramter incorrect: s (current price) = {s}. It should be > 0'.format(s=s))

    if sigma <= 0:
        raise ValueError('Paramter incorrect: sigma (annualised volatility) = {sigma}. It should be > 0'.format(sigma=sigma))

    if steps_per_year <= 0:
        raise ValueError('Paramter incorrect: steps_per_year = {steps_per_year}. It should be > 0'.format(steps_per_year=steps_per_year))

    if steps <= 0:
        raise ValueError('Paramter incorrect: steps = {steps}. It should be > 0'.format(steps=steps))

    if trials <= 0:
        raise ValueError('Paramter incorrect: trials = {trials}. It should be > 0'.format(trials=trials))

    if random_seed < 0 or random_seed > 2**32 - 1:
        raise ValueError('Paramter incorrect: random_seed = {random_seed}. It should be between 0 and 2**32 - 1'.format(random_seed=random_seed))

    np.random.seed(random_seed)

    dt = 1.0 / steps_per_year
    
    drift = r - sigma * sigma * 0.5 

    epsilon = np.random.randn(steps, trials)

    return np.cumprod(np.vstack((s * np.ones(shape=(1, trials)), np.exp(drift * dt + sigma * np.sqrt(dt) * epsilon))), axis=0)
