"""Module for utility functions."""

from __future__ import annotations

import numpy as np
from scipy import stats


def calculate_confidence_intervals(
    data: np.ndarray, confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate confidence intervals for each row of a 2D numpy array.

    Args:
        data: A 2D numpy array where each row represents a variable
                           and each column represents an observation.
        confidence_level: The confidence level for the interval (default: 0.95 for 95% CI).

    Returns:
        tuple: A tuple containing three 1D numpy arrays:
               - mean: The mean value for each row.
               - lower_bound: The lower bound of the confidence interval for each row.
               - upper_bound: The upper bound of the confidence interval for each row.
    """
    if data.ndim != 2:
        msg = "Input data must be a 2D numpy array."
        raise ValueError(msg)

    n = data.shape[1]  # number of observations

    mean = np.mean(data, axis=1)
    std_error = np.std(data, axis=1, ddof=1) / np.sqrt(n)

    # Calculate the t-value
    t_value = stats.t.ppf((1.0 + confidence_level) / 2.0, n - 1)

    # Calculate the margin of error
    margin_of_error = t_value * std_error

    # Calculate lower and upper bounds
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return mean, lower_bound, upper_bound
