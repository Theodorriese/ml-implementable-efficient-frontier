import numpy as np

def ewma_volatility(x, lambda_, start):
    """
    Calculate the EWMA volatility of a time series.

    Parameters:
        x (array-like): Input time series data (numpy array or list).
        lambda_ (float): Smoothing parameter (decay factor) between 0 and 1.
        start (int): Warm-up period to calculate initial variance.

    Returns:
        np.ndarray: EWMA volatility values (standard deviation) at each time point.
    """
    x = np.array(x, dtype=np.float64)
    n = len(x)
    var_vec = np.full(n, np.nan)  # Initialize result array with NaN
    na_count = 0
    initial_value = 0.0

    if n <= start:
        return var_vec  # Return NaN array if data is too short

    # Calculate initial variance over the warm-up period
    for i in range(start):
        if np.isnan(x[i]):
            na_count += 1  # Count missing values
        else:
            initial_value += x[i] ** 2
    initial_value /= (start - 1 - na_count)

    # Initialize EWMA variance
    var_vec[start] = initial_value

    # Calculate EWMA iteratively
    for j in range(start + 1, n):
        if np.isnan(x[j - 1]):
            var_vec[j] = var_vec[j - 1]  # Carry forward previous value if missing
        else:
            var_vec[j] = lambda_ * var_vec[j - 1] + (1 - lambda_) * x[j - 1] ** 2

    # Return the square root of the variance (volatility)
    return np.sqrt(var_vec)

# Example Usage
# x = [0.1, 0.2, -0.1, np.nan, 0.3, -0.2, 0.1, 0.2, -0.1, 0.1]
# lambda_ = 0.94  # Decay factor
# start = 5       # Warm-up period
#
# volatility = ewma_volatility(x, lambda_, start)
# print(volatility)
