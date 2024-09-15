"""Functions for calculating financial measures."""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS_PER_YEAR = 252
logger = logging.getLogger(__name__)


def calculate_returns(values: pd.Series) -> pd.Series:
    """Calculate period-to-period returns.

    Computes the percentage change between consecutive periods:

    r_t = (V_t - V_(t-1)) / V_(t-1)

    Where:
    V_t = value at time t

    Args:
        values: Series of asset values over time.

    Returns:
        Series of period-to-period returns.

    Caveats:
    - Assumes equally spaced time periods.
    - First value will be NaN due to lack of previous value.
    - Does not account for dividends or other cash flows.
    """
    return values.pct_change()


def arithmetic_mean(returns: pd.Series) -> float:
    """Calculate the arithmetic mean of returns.

    Defined as the sum of returns divided by the number of observations:

    AM = (Σ r_i) / n

    Where:
    r_i = individual returns
    n = number of returns

    Args:
        returns: Series of returns.

    Returns:
        Arithmetic mean of returns.

    Caveats:
    - Overestimates expected return for volatile assets due to compounding.
    - Does not account for the order of returns.
    - Assumes each return has equal weight, regardless of the asset's value.
    """
    return returns.mean()


def geometric_mean(returns: pd.Series) -> float:
    """Calculate the geometric mean of returns.

    The geometric mean represents the compounded growth rate of an investment.
    It's defined as:

    G = (∏(1 + r_i))^(1/n) - 1

    Where:
    r_i = individual returns
    n = number of returns

    Args:
        returns: Series of returns (as decimals, not percentages).

    Returns:
        Geometric mean of returns.

    Interpretation:
    - Provides the constant rate of return that would yield the same final value.
    - More appropriate than arithmetic mean for compounded growth scenarios.
    - Accounts for volatility drag, unlike arithmetic mean.
    """
    returns_array = returns.to_numpy()
    return (1.0 + returns_array).prod() ** (1.0 / len(returns_array)) - 1.0


def time_weighted_return(returns: pd.Series) -> float:
    """Calculate the Time-Weighted Return (TWR).

    TWR measures the compound growth rate of a portfolio independent of
    external cash flows. It's defined as:

    TWR = ∏(1 + r_i) - 1

    Where:
    r_i = individual period returns

    Args:
        returns: Series of period returns (as decimals, not percentages).

    Returns:
        Time-Weighted Return.

    Interpretation:
    - Eliminates the impact of cash flows on performance measurement.
    - Useful for comparing performance across portfolios or to benchmarks.
    - Assumes equal weighting of each period, regardless of portfolio value.
    """
    return np.prod(1.0 + returns.to_numpy()) - 1.0


def money_weighted_return(  # noqa: C901
    values: pd.Series[float],
    *,
    initial_investment: float | None = None,
    period: Literal["D", "M", "Q", "Y"] = "D",
) -> float:
    """Calculate the money-weighted return (MWR) of an investment.

    Money-weighted return, also known as the internal rate of return (IRR), measures
    the performance of an investment taking into account the size and timing of cash flows.
    It is the discount rate that makes the net present value of all cash flows equal to zero.

    Calculation:
        The MWR (r) is the solution to the equation:
        0 = -initial_investment + Σ((Vᵢ - Vᵢ₋₁) / (1 + r)^tᵢ)
        Where:
        Vᵢ = Asset value at time i
        tᵢ = Time in years from the start

    This function uses numerical methods to solve for r.

    Interpretation:
    - MWR is expressed as an annual percentage return.
    - It accounts for the impact of cash flows on performance, unlike time-weighted return.
    - A higher MWR indicates better performance, considering both investment growth and timing of cash flows.
    - MWR can be compared to other investments or benchmarks to assess relative performance.
    - It's particularly useful for assessing performance when significant cash flows occur during the investment period.

    Note:
    MWR assumes that all cash flows are reinvested at the same rate of return, which may not always reflect reality.

    Args:
        values: Series of asset values over time. Index can be datetime or numeric.
        initial_investment (optional): Initial amount invested. If None, uses the first value in the series.
        period (optional): For non-datetime indices, specifies the period between values.
                           Options: 'D' (daily), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly). Default is 'D'.

    Returns:
        float: The money-weighted return as a decimal (e.g., 0.1 for 10% return).
    """
    if initial_investment is None:
        initial_investment = values.iloc[0]

    # Calculate cash flows
    cash_flows = values.diff()
    cash_flows.iloc[0] = -initial_investment
    cash_flows.iloc[-1] += values.iloc[-1]

    # Calculate times based on index type
    if pd.api.types.is_datetime64_any_dtype(values.index):
        times = (values.index - values.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    else:
        # Assume uniform intervals and convert to years based on the period
        times = np.arange(len(values))
        if period == "D":
            times /= 365.25
        elif period == "M":
            times /= 12.0
        elif period == "Q":
            times /= 4.0
        # 'Y' doesn't need adjustment

    def npv(rate: float) -> float:
        return (cash_flows / (1.0 + rate) ** times).sum()

    def npv_derivative(rate: float) -> float:
        return (-times * cash_flows / (1.0 + rate) ** (times + 1.0)).sum()

    # Intelligent initial guess
    total_return = (values.iloc[-1] - initial_investment) / initial_investment
    time_period = times[-1] - times[0]
    rate_guess = (1.0 + total_return) ** (1.0 / time_period) - 1.0

    try:
        result = optimize.newton(npv, x0=rate_guess, fprime=npv_derivative, maxiter=1000, tol=1e-6)
    except RuntimeError:
        # Fallback to a more robust but potentially slower method if newton fails
        try:
            result = optimize.brentq(npv, a=-0.999, b=1000.0, maxiter=1000)
        except ValueError:
            logger.debug("Failed to converge on a solution for money-weighted return.")
            return np.nan
        else:
            return result
    else:
        return result


def sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculate the Sortino Ratio.

    The Sortino ratio is a risk-adjusted performance measure that modifies
    the Sharpe ratio by considering only downside volatility. It's defined as:

    S = (R - T) / σd

    Where:
        R = portfolio's average return
        T = target or required rate of return
        σd = standard deviation of downside returns

    Args:
        returns: Series of returns.
        target_return: Target return (default is 0).

    Returns:
        Sortino Ratio. Higher values indicate better risk-adjusted performance.

    Interpretation:
    - > 2: Excellent
    - 1-2: Good
    - 0.5-1: Fair
    - < 0.5: Poor
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    downside_risk = np.sqrt((downside_returns**2).sum() / len(returns))
    return excess_returns.mean() / downside_risk if downside_risk != 0 else np.nan


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    *,
    interpolation: Literal["linear", "lower", "higher", "midpoint", "nearest"] = "linear",
) -> float:
    """Calculate Value at Risk (VaR).

    VaR estimates the maximum potential loss in value of a portfolio over
    a defined period for a given confidence interval. It's defined as:

    VaR = -quantile(returns, 1 - confidence)

    Args:
        returns: Series of returns (as decimals, not percentages).
        confidence: Confidence level (default is 0.95 for 95% VaR).
        interpolation: Method used to interpolate quantiles (default is 'linear').

    Returns:
        Value at Risk as a positive number.

    Interpretation:
    - With (confidence * 100)% certainty, the maximum loss will not exceed VaR.
    - Does not provide information about the severity of losses beyond VaR.
    - Higher VaR indicates higher potential for loss.
    """
    return -returns.quantile(1.0 - confidence, interpolation=interpolation)


def expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    *,
    interpolation: Literal["linear", "lower", "higher", "midpoint", "nearest"] = "linear",
) -> float:
    """Calculate Expected Shortfall (ES), also known as Conditional VaR (CVaR).

    ES measures the expected loss given that the loss is greater than the VaR.
    It's defined as:

    ES = E[X | X ≤ -VaR(X)]

    Where:
    X = returns
    VaR = Value at Risk

    Args:
        returns: Series of returns (as decimals, not percentages).
        confidence: Confidence level (default is 0.95 for 95% ES).
        interpolation: Method used to interpolate quantiles (default is 'linear').

    Returns:
        Expected Shortfall as a positive number.

    Interpretation:
    - Average loss in the worst (1 - confidence)% of cases.
    - More sensitive to tail risk than VaR.
    - Considered a coherent risk measure, unlike VaR.
    """
    var = value_at_risk(returns, confidence, interpolation=interpolation)
    return -returns[returns <= -var].mean()


def standard_deviation(returns: pd.Series[float]) -> float:
    """Calculate the standard deviation of returns (volatility).

    Standard deviation measures the dispersion of returns from their mean.
    For a sample, it's defined as:

    σ = sqrt(Σ(r_i - μ)^2 / (n - 1))

    Where:
    r_i = individual returns
    μ = mean return
    n = number of returns

    Args:
        returns: Series of returns.

    Returns:
        Standard deviation of returns.

    Interpretation:
    - Higher values indicate greater volatility.
    - Approximately 68% of returns fall within ±1σ of the mean, assuming normal distribution.

    Caveats:
    - Assumes returns are normally distributed.
    - Equally weights upside and downside deviations.
    - Does not distinguish between irregular large deviations and frequent small ones.
    """
    return returns.std()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio.

    The Sharpe ratio measures the risk-adjusted return of an investment.
    It's defined as:

    S = (R_p - R_f) / σ_p

    Where:
    R_p = annualized portfolio return
    R_f = risk-free rate
    σ_p = annualized standard deviation of portfolio excess returns

    Args:
        returns: Series of daily returns (as decimals, not percentages).
        risk_free_rate: Annualized risk-free rate (default is 0.0).

    Returns:
        Sharpe Ratio.

    Interpretation:
    - Higher values indicate better risk-adjusted performance.
    - < 1: Poor, 1-2: Adequate, 2-3: Very good, > 3: Excellent.
    - Assumes returns are normally distributed (a potential limitation).
    """
    daily_risk_free_rate = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_risk_free_rate
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()


def maximum_drawdown(values: pd.Series) -> float:
    """Calculate the Maximum Drawdown.

    Maximum Drawdown measures the largest peak-to-trough decline in the value of a portfolio.
    It's defined as:

    MDD = min((V_t - V_peak) / V_peak)

    Where:
    V_t = portfolio value at time t
    V_peak = peak portfolio value observed up to time t

    Args:
        values: Series of asset values over time.

    Returns:
        Maximum Drawdown as a positive decimal value.

    Interpretation:
    - Measures downside risk and worst-case loss from a peak.
    - Smaller (closer to 0) is better; -1 implies complete loss.
    - Doesn't consider frequency or timing of drawdowns.
    """
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return abs(drawdown.min())


def beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate the Beta of the portfolio.

    Beta measures the volatility of a portfolio in relation to the market.
    It's defined as:

    β = Cov(R_p, R_m) / Var(R_m)

    Where:
    R_p = portfolio returns
    R_m = market returns

    Args:
        portfolio_returns: Series of portfolio returns.
        market_returns: Series of market returns.

    Returns:
        Beta of the portfolio.

    Interpretation:
    - β = 1: Portfolio moves with the market.
    - β > 1: Portfolio is more volatile than the market.
    - β < 1: Portfolio is less volatile than the market.
    - β < 0: Portfolio moves opposite to the market.
    """
    covariance = portfolio_returns.cov(market_returns)
    market_variance = cast(float, market_returns.var())
    if market_variance == 0.0:
        return float("nan")
    result = covariance / market_variance
    return float(result.real) if isinstance(result, complex) else float(result)


def omega_ratio(returns: pd.Series[float], threshold: float = 0.0) -> float:
    """Calculate the Omega ratio for a given series of returns.

    The Omega ratio is defined as:
    Ω(r) = ∫[r,∞] (1 - F(x))dx / ∫[-∞,r] F(x)dx

    where F(x) is the cumulative distribution function of returns,
    and r is the threshold return.

    Args:
        returns: A pandas Series of returns.
        threshold: The threshold return (default is 0).

    Returns:
        The calculated Omega ratio.
    """
    excess_returns = returns - threshold
    positive_returns = excess_returns[excess_returns > 0.0]
    negative_returns = excess_returns[excess_returns <= 0.0]

    return positive_returns.sum() / negative_returns.abs().sum()


def modified_var(returns: pd.Series[float], confidence: float = 0.95) -> float:
    """Calculate the Modified Value at Risk (MVaR) for a given series of returns.

    MVaR adjusts the traditional VaR for non-normal distributions by
    incorporating skewness and kurtosis:

    MVaR = μ - σ * (z_α + (z_α^2 - 1) * S / 6 + (z_α^3 - 3z_α) * (K - 3) / 24 - (2z_α^3 - 5z_α) * S^2 / 36)

    where:
    μ is the mean return
    σ is the standard deviation of returns
    z_α is the z-score for the given confidence level
    S is the skewness of returns
    K is the kurtosis of returns

    Args:
        returns: A pandas Series of returns.
        confidence: Confidence level (default is 0.95 for 95% MVaR).

    Returns:
        The calculated Modified VaR.
    """
    mu: float = returns.mean()
    sigma: float = returns.std()
    skew = cast(float, returns.skew())
    excess_kurt = cast(float, returns.kurtosis())
    z_score: float = stats.norm.ppf(1.0 - confidence)

    return mu - sigma * (
        z_score
        + (z_score**2.0 - 1.0) * skew / 6.0
        + (z_score**3.0 - 3.0 * z_score) * excess_kurt / 24.0
        - (2.0 * z_score**3.0 - 5.0 * z_score) * skew**2.0 / 36.0
    )


def calmar_ratio(returns: pd.Series, period: int = 252) -> float:
    """Calculate the Calmar ratio for a given series of returns.

    Calmar ratio = Annualized Return / Maximum Drawdown

    Args:
        returns: A pandas Series of returns.
        period: Number of periods in a year for annualization (default is 252 for daily returns).

    Returns:
        The calculated Calmar ratio.
    """
    values = (1.0 + returns).cumprod()
    max_drawdown = maximum_drawdown(values)
    annualized_return = geometric_mean(returns) * period
    return annualized_return / max_drawdown if max_drawdown != 0.0 else float("inf")


def calculate_measures(values: pd.Series[float], market_values: pd.Series[float] | None = None) -> pd.DataFrame:
    """Calculate financial measures for rolling periods of n points.

    Args:
        values: Array of asset values.
        market_values: Array of market index values.

    Returns:
        DataFrame containing calculated measures for each rolling period.
    """
    returns = calculate_returns(values)

    results = pd.DataFrame({
        "arithmetic_mean": [arithmetic_mean(returns)],
        "geometric_mean": [geometric_mean(returns)],
        "twr": [time_weighted_return(returns)],
        "irr": [money_weighted_return(values)],
        "sortino_ratio": [sortino_ratio(returns)],
        "var_95": [value_at_risk(returns)],
        "es_95": [expected_shortfall(returns)],
        "mvar_95": [modified_var(returns)],
        "standard_deviation": [standard_deviation(returns)],
        "sharpe_ratio": [sharpe_ratio(returns)],
        "omega_ratio": [omega_ratio(returns)],
        "maximum_drawdown": [maximum_drawdown(values)],
        "calmar_ratio": [calmar_ratio(returns)],
    })

    if market_values is not None:
        market_returns = calculate_returns(market_values)
        results["beta"] = beta(returns, market_returns)

    results.index = pd.Index([values.index[-1]])
    return results
