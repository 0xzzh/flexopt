import numpy as np

from numpy.polynomial import Polynomial


def _validate_contract_spec(number_of_exercises, dcq, min_dcq, max_dcq):
    if number_of_exercises < 0:
        raise ValueError('Paramter incorrect: number_of_exercises = {number_of_exercises}. It should be >= 0'.format(number_of_exercises=number_of_exercises))

    if dcq < 0:
        raise ValueError('Paramter incorrect: dcq (daily contract quantities) = {dcq}. It should be >= 0'.format(dcq=dcq))

    if min_dcq < 0:
        raise ValueError('Paramter incorrect: min_dcq (minimum daily contract quantities) = {min_dcq}. It should be >= 0'.format(min_dcq=min_dcq))

    if max_dcq < 0:
        raise ValueError('Paramter incorrect: max_dcq (maximium daily contract quantities) = {max_dcq}. It should be >= 0'.format(max_dcq=max_dcq))

    if not min_dcq <= dcq <= max_dcq:
        raise ValueError('Paramter incorrect: min_dcq = {min_dcq}, dcq = {dcq}, max_dcq = {max_dcq}. The relationship should be min_dcq <= dcq <= max_dcq'.format(min_dcq=min_dcq, dcq=dcq, max_dcq=max_dcq))


def dynamic_programming_optimization(sim_prices, strike, df_per_time_step, polyfit_degree, number_of_exercises, dcq, min_dcq, max_dcq):
    """Calculate swing price and dispatching plan by backward dynamic programming optimization from time T (contract expiry time) to time 0 (contract start time)

    Args:
        sim_prices: Simulated prices scenarios of shape [steps x trials].
        strike: The exercise price of the swing contract.
        df_per_time_step: Discount factor of continuation values per time step
        polyfit_degree: Degree of fitting the polynomial model.
        number_of_exercises: The number of exercise opportunities during the contract period.
        dcq: Daily contract quantities.
        min_dcq: Minimum daily contract quantities.
        max_dcq: Maximum daily contract quantities.

    Returns:
        The first return value: Swing price
        The second return value: A numpy array of shape [steps x trials] to corresponding exercising / dispatching plan.

    Raises:
        ValueError: An error occurred when input paramters are out of range.

    References:
        Boogert, A., de Jong, C.: Gas storage valuation using a Monte Carlo method. The Journal of Derivatives pp. 81-98 (2008)
        Longstaff, F.A., Schwartz, E.S.: Valuing American options by simulation: A simple leastsquares approach. The Review of Financial Studies 14(1), 113-147 (2001)
    """
    _validate_contract_spec(number_of_exercises=number_of_exercises, dcq=dcq, min_dcq=min_dcq, max_dcq=max_dcq)
    
    steps, trials = sim_prices.shape
    
    # Actions and states
    qty_action_arr = np.array([min_dcq - dcq, max_dcq - dcq])  # [dispatch min, dispatch max]
    remaining_exercise_state_list = list(range(0, number_of_exercises + 1))  # [0, ..., number_of_exercises]
    
    # Table to update in each time step
    value_table = np.zeros(shape=(len(remaining_exercise_state_list), trials))  # Shape = (number_of_exercises+1, trials)

    for t in range(steps-1, 0, -1):  # From steps-1 (t=expiry) to 1 (t=start + 1)
        curr_prices = sim_prices[t, :]

        # Calculate optimal exercise values at the current time step
        payoff_arr = np.zeros(shape=(len(qty_action_arr), trials))  # Payoffs based on chosen qty; Shape = (qty_action_arr=2, trials)
        
        for action_ind, action_qty in enumerate(qty_action_arr):
            payoff_arr[action_ind, :] = action_qty * (curr_prices - strike)
        
        exercise_value_arr = np.nanmax(payoff_arr, axis=0)  # Shap = (trials, 1)

        # Fit current prices to the values of the next time step
        next_value_table = df_per_time_step * np.array(value_table) # The value table of t+1 discounted to t
        fitted_next_value_table = np.zeros(shape=next_value_table.shape)

        if t < steps - 1:
            for remaining_exercise_state in remaining_exercise_state_list:
                model_coef = Polynomial.fit(x=curr_prices, y=next_value_table[remaining_exercise_state, :], deg=polyfit_degree)
                fitted_next_value_table[remaining_exercise_state, :] = model_coef(curr_prices)

        # Update value table based on optimal decision
        for remaining_exercise_state in remaining_exercise_state_list:
            value_table[remaining_exercise_state, :] = next_value_table[remaining_exercise_state, :]  

            if remaining_exercise_state > 0:  # Compare rewards of exercising and not exercising for optimal decision
                exercise_ind = (exercise_value_arr + fitted_next_value_table[remaining_exercise_state-1, :] > fitted_next_value_table[remaining_exercise_state, :])
                value_table[remaining_exercise_state, exercise_ind] = exercise_value_arr[exercise_ind] + next_value_table[remaining_exercise_state-1, exercise_ind]
                
    return df_per_time_step * np.mean(value_table)
