import unittest
import math

import numpy as np

from flexopt.price import geometric_brownian_motion


class TestGeometricBrownianMotion(unittest.TestCase):

    def test_simulation(self):
        s = 100
        r = 0.05
        sigma = 0.25
        steps_per_year = 365
        
        tau_in_years = 2
        steps = tau_in_years * steps_per_year
        trials = 100000
        random_seed = 1

        sim_prices = geometric_brownian_motion.simulate(s=s, r=r, sigma=sigma, steps_per_year=steps_per_year, steps=steps, trials=trials, random_seed=random_seed)

        # Data shape
        rows, cols = sim_prices.shape
        self.assertEqual(rows, steps + 1)
        self.assertEqual(cols, trials)

        # Stats: distribution mean
        test_mean = s * math.exp(r * tau_in_years)
        sim_mean = sim_prices[-1, :].mean()
        
        mean_percent_err = abs(sim_mean - test_mean) / test_mean
        
        self.assertTrue(mean_percent_err < 0.01)
        
        # Stats: sigma
        sim_prices_log_diff = np.diff(np.log(sim_prices), n=1, axis=0)
        sim_sigma = (sim_prices_log_diff.std(axis=0) * math.sqrt(steps_per_year)).mean()

        sigma_abs_err = abs(sim_sigma - sigma)

        self.assertTrue(sigma_abs_err < 0.001)


if __name__ == '__main__':
    unittest.main()