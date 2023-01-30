import unittest

from flexopt.price import ou_process


class TestOUProcess(unittest.TestCase):

    def test_simulation(self):
        s = 100
        kappa = 2
        theta = 90
        sigma = 0.30

        steps_per_year = 365
        steps = 730
        trials = 10000

        sim_prices = ou_process.simulate(
            s=s, 
            kappa=kappa, 
            theta=theta, 
            sigma=sigma, 
            steps_per_year=steps_per_year, 
            steps=steps, 
            trials=trials, 
            random_seed=1
        )

        # Data shape
        rows, cols = sim_prices.shape
        self.assertEqual(rows, steps + 1)
        self.assertEqual(cols, trials)

        sim_kappa = 0
        sim_theta = 0
        sim_sigma = 0

        for sim_ind in range(0, trials):
            curr_kappa, curr_theta, curr_sigma = ou_process.calibrate_parameters(x=sim_prices[:, sim_ind], steps_per_year=steps_per_year)
            
            sim_kappa += curr_kappa
            sim_theta += curr_theta
            sim_sigma += curr_sigma

        sim_kappa = sim_kappa / trials
        sim_theta = sim_theta / trials
        sim_sigma = sim_sigma / trials

        # Stats: kappa
        mean_percent_err = abs(sim_kappa - kappa) / kappa
        self.assertTrue(mean_percent_err < 0.01)

        # Stats: theta
        mean_percent_err = abs(sim_theta - theta) / theta
        self.assertTrue(mean_percent_err < 0.01)

        # Stats: sigma
        abs_err = abs(sim_sigma - sigma)
        self.assertTrue(abs_err < 0.001)


if __name__ == '__main__':
    unittest.main()