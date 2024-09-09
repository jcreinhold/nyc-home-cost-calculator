"""Functions for calculating financial measures."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import newton

TRADING_DAYS_PER_YEAR = 252


def calculate_returns(values: pd.Series[float]) -> pd.Series:
    """Calculate period-to-period returns.

    Args:
        values: Array of asset values.

    Returns:
        Array of period-to-period returns.
    """
    return values.pct_change()


def arithmetic_mean(returns: pd.Series[float]) -> float:
    """Calculate the arithmetic mean of returns.

    Args:
        returns: Array of returns.

    Returns:
        Arithmetic mean of returns.
    """
    return returns.mean()


def geometric_mean(returns: pd.Series[float]) -> float:
    """Calculate the geometric mean of returns.

    Args:
        returns: Array of returns.

    Returns:
        Geometric mean of returns.
    """
    _returns = returns.to_numpy()
    return (1.0 + _returns).prod() ** (1.0 / len(_returns)) - 1.0


def time_weighted_return(returns: pd.Series) -> float:
    """Calculate the Time-Weighted Return (TWR).

    Args:
        returns: Array of returns.

    Returns:
        Time-Weighted Return.
    """
    return (1.0 + returns.to_numpy()).prod() - 1.0


def money_weighted_return(values: pd.Series[float]) -> float:
    """Calculate the Money-Weighted Return (IRR).

    Args:
        values: Array of asset values.

    Returns:
        Money-Weighted Return (IRR).
    """

    def npv(rate: float, cashflows: pd.Series[float]) -> float:
        """Calculate Net Present Value."""
        if rate == -1:
            return np.inf if np.any(cashflows > 0) else -np.inf

        time = np.arange(len(cashflows))
        discount_factors = np.exp(-np.log1p(rate) * time)
        return (cashflows * discount_factors).sum()

    cashflows = values.diff().fillna(values.iloc[0])
    try:
        # Use Newton's method to find the root of the NPV function
        return newton(lambda r: npv(r, cashflows), x0=0.1)
    except RuntimeError:
        # If Newton's method fails to converge, return NaN
        return np.nan


def sortino_ratio(returns: pd.Series[float], target_return: float = 0) -> float:
    """Calculate the Sortino Ratio.

    Args:
        returns: Array of returns.
        target_return: Target return (default is 0).

    Returns:
        Sortino Ratio.
    """
    # Calculate downside returns (returns below target)
    downside_returns = returns[returns < target_return]
    # Calculate downside risk (standard deviation of downside returns)
    downside_risk = np.sqrt((downside_returns**2).mean())
    # Calculate Sortino Ratio
    return (returns.mean() - target_return) / downside_risk if downside_risk != 0 else np.nan


def value_at_risk(returns: pd.Series[float], confidence: float = 0.95) -> float:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Array of returns.
        confidence: Confidence level (default is 0.95 for 95% VaR).

    Returns:
        Value at Risk.
    """
    return returns.quantile(1.0 - confidence)


def expected_shortfall(returns: pd.Series[float], confidence: float = 0.95) -> float:
    """Calculate Expected Shortfall (ES).

    Args:
        returns: Array of returns.
        confidence: Confidence level (default is 0.95 for 95% ES).

    Returns:
        Expected Shortfall.
    """
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def standard_deviation(returns: pd.Series[float]) -> float:
    """Calculate the standard deviation of returns (volatility).

    Args:
        returns: Array of returns.

    Returns:
        Standard deviation of returns.
    """
    return returns.std()


def sharpe_ratio(returns: pd.Series[float], risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio.

    Args:
        returns: Array of returns.
        risk_free_rate: Risk-free rate (annualized).

    Returns:
        Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR  # Assuming daily returns
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()


def maximum_drawdown(values: pd.Series[float]) -> float:
    """Calculate the Maximum Drawdown.

    Args:
        values: Array of asset values.

    Returns:
        Maximum Drawdown.
    """
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return drawdown.min()


def beta(portfolio_returns: pd.Series[float], market_returns: pd.Series[float]) -> float:
    """Calculate the Beta of the portfolio.

    Args:
        portfolio_returns: Array of portfolio returns.
        market_returns: Array of market returns.

    Returns:
        Beta of the portfolio.
    """
    covariance = portfolio_returns.cov(market_returns)
    market_variance = cast(float, market_returns.var())
    if market_variance == 0.0:
        return float("nan")
    result = covariance / market_variance
    if isinstance(result, complex):
        return float(result.real)
    return float(result)


def calculate_measures(values: pd.Series[float], market_values: pd.Series[float] | None = None) -> pd.DataFrame:
    """Calculate financial measures for rolling periods of n points.

    Args:
        values: Array of asset values.
        market_values: Array of market index values.

    Returns:
        DataFrame containing calculated measures for each rolling period.
    """
    returns = calculate_returns(values)

    results = pd.DataFrame(
        {
            "arithmetic_mean": [arithmetic_mean(returns)],
            "geometric_mean": [geometric_mean(returns)],
            "twr": [time_weighted_return(returns)],
            "irr": [money_weighted_return(values)],
            "sortino_ratio": [sortino_ratio(returns)],
            "var_95": [value_at_risk(returns)],
            "es_95": [expected_shortfall(returns)],
            "standard_deviation": [standard_deviation(returns)],
            "sharpe_ratio": [sharpe_ratio(returns)],
            "maximum_drawdown": [maximum_drawdown(values)],
        }
    )

    if market_values is not None:
        market_returns = calculate_returns(market_values)
        results["beta"] = beta(returns, market_returns)

    results.index = pd.Index([values.index[-1]])
    return results
