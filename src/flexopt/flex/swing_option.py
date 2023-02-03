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


def dynamic_programming_optimization(prices, strike, number_of_exercises, dcq, min_dcq, max_dcq, r, steps_per_year, polyfit_degree):
    """Calculates swing value and exercise schedule by the approximate dynamic programming method.
    
    Retrieves price scenarios, swing option specifications, and optimization parameters. Loops backwards from swing maturity day
    to swing start day. 
    
    At each time step compares the following two statements to decide whether to exercise:
    - Exercise value of the current time step + continuation value of the next time step with one fewer exercise opportunity;
    - No exercise at the current time step (value = 0) + continuation value of the next time step with the same exercise opportunity. 

    Args:
        prices: Price scenarios of shape (contract period time steps + 1, trials). 
          The prices at time step 0 is prior to the contract period and not counted in the exercise schedule
        strike: The exercise price of the swing contract.
        number_of_exercises: The number of exercise opportunities during the contract period.
        dcq: Daily contract quantities.
        min_dcq: Minimum daily contract quantities.
        max_dcq: Maximum daily contract quantities.
        r: Risk-free rate.
        steps_per_year: Trading days (or other time units, for example hours) per year.
        polyfit_degree: Degree of fitting the polynomial model at each time step.
        
    Returns:
        The first return value: swing_value
        The second return value: exercise_schedules of shape (contract period time steps, trials), exercising amount per time step per price scenario.

    Raises:
        ValueError: An error occurred when input paramters are out of range.

    References:
        Boogert, A., de Jong, C.: Gas storage valuation using a Monte Carlo method. The Journal of Derivatives pp. 81-98 (2008)
        Longstaff, F.A., Schwartz, E.S.: Valuing American options by simulation: A simple leastsquares approach. The Review of Financial Studies 14(1), 113-147 (2001)
    """
    _validate_contract_spec(number_of_exercises=number_of_exercises, dcq=dcq, min_dcq=min_dcq, max_dcq=max_dcq)
    
    dt = 1.0 / steps_per_year
    df_per_time_step = np.exp(-r * dt)
    
    steps, trials = prices.shape
    
    # Actions and states
    qty_actions = np.array([min_dcq - dcq, max_dcq - dcq])  # [dispatch min, dispatch max]
    remaining_exercise_states = list(range(0, number_of_exercises + 1))  # [0, ..., number_of_exercises]
    
    # Table to update in each time step
    value_table = np.zeros(shape=(len(remaining_exercise_states), trials))  # Shape = (number_of_exercises+1, trials)

    for t in range(steps-1, 0, -1):  # From steps-1 (t=expiry) to 1 (t=start + 1)
        curr_prices = prices[t, :]

        # Calculate optimal exercise values at the current time step
        payoffs = np.zeros(shape=(len(qty_actions), trials))  # Payoffs based on chosen qty; Shape = (qty_actions=2, trials)
        
        for action_ind, action_qty in enumerate(qty_actions):
            payoffs[action_ind, :] = action_qty * (curr_prices - strike)
        
        exercise_values = np.nanmax(payoffs, axis=0)  # Shap = (trials, 1)

        # Fit current prices to the values of the next time step
        next_value_table = df_per_time_step * np.array(value_table) # The value table of t+1 discounted to t
        fitted_next_value_table = np.zeros(shape=next_value_table.shape)

        if t < steps - 1:
            for remaining_exercise_state in remaining_exercise_states:
                model_coef = Polynomial.fit(x=curr_prices, y=next_value_table[remaining_exercise_state, :], deg=polyfit_degree)
                fitted_next_value_table[remaining_exercise_state, :] = model_coef(curr_prices)

        # Update value table based on optimal decision
        for remaining_exercise_state in remaining_exercise_states:
            value_table[remaining_exercise_state, :] = next_value_table[remaining_exercise_state, :]  

            if remaining_exercise_state > 0:  # Compare rewards of exercising and not exercising for optimal decision
                exercise_ind = (exercise_values + fitted_next_value_table[remaining_exercise_state-1, :] > fitted_next_value_table[remaining_exercise_state, :])
                value_table[remaining_exercise_state, exercise_ind] = exercise_values[exercise_ind] + next_value_table[remaining_exercise_state-1, exercise_ind]
                
    return df_per_time_step * np.mean(value_table)
