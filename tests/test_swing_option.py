import unittest

import numpy as np

from flexopt.flex import swing_option


class TestSwingOption(unittest.TestCase):

    def test_dynamic_programming_optimization_base(self):
        sim_prices = np.array([
            [50, 50, 50, 50, 50],
            [20, 35, 50, 65, 80],
            [30, 40, 50, 60, 70],
        ])

        swing_value, exercise_schedules = swing_option.dynamic_programming_optimization(
            prices=sim_prices,
            strike=50,
            r=0,
            steps_per_year=365,
            polyfit_degree=1,
            number_of_exercises=1,
            dcq=1,
            min_dcq=0,
            max_dcq=2,
        )

        self.assertEqual(swing_value, 18)
        self.assertTrue(np.array_equal(exercise_schedules[:, 0], np.array([-0.5, -0.5])))
        self.assertTrue(np.array_equal(exercise_schedules[:, 1], np.array([-0.5, -0.5])))
        self.assertTrue(np.array_equal(exercise_schedules[:, 2], np.array([0, 0])))
        self.assertTrue(np.array_equal(exercise_schedules[:, 3], np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(exercise_schedules[:, 4], np.array([0.5, 0.5])))


if __name__ == '__main__':
    unittest.main()