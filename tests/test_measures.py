import numpy as np
import pandas as pd
import pytest

from nyc_home_cost_calculator.measures import (
    arithmetic_mean,
    beta,
    calculate_measures,
    calculate_returns,
    expected_shortfall,
    geometric_mean,
    maximum_drawdown,
    money_weighted_return,
    sharpe_ratio,
    sortino_ratio,
    standard_deviation,
    time_weighted_return,
    value_at_risk,
)


@pytest.fixture
def sample_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
    values = pd.Series([100.0, 102.0, 99.0, 101.0, 103.0], index=dates)
    market_values = pd.Series([1000.0, 1010.0, 990.0, 1005.0, 1015.0], index=dates)
    returns = values.pct_change().dropna()
    market_returns = market_values.pct_change().dropna()
    return values, market_values, returns, market_returns


def test_calculate_returns(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test calculate_returns function.

    Derivation:
    r_t = (V_t - V_(t-1)) / V_(t-1)
    r_1 = (102 - 100) / 100 = 0.02
    r_2 = (99 - 102) / 102 = -0.02941176
    r_3 = (101 - 99) / 99 = 0.02020202
    r_4 = (103 - 101) / 101 = 0.01980198
    """
    values, *_ = sample_data
    expected = pd.Series([np.nan, 0.02, -0.02941176, 0.02020202, 0.01980198], index=values.index)
    pd.testing.assert_series_equal(calculate_returns(values), expected, atol=1e-8)


def test_arithmetic_mean(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test arithmetic_mean function.

    Derivation:
    AM = (Σ r_i) / n
    AM = (0.02 + (-0.02941176) + 0.02020202 + 0.01980198) / 4 = 0.00764806
    """
    _, _, returns, _ = sample_data
    assert arithmetic_mean(returns) == pytest.approx(0.00764806, abs=1e-8)


def test_geometric_mean(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test geometric_mean function.

    Derivation:
    G = (∏(1 + r_i))^(1/n) - 1
    G = ((1 + 0.02) * (1 - 0.02941176) * (1 + 0.02020202) * (1 + 0.01980198))^(1/4) - 1
    G ≈ 0.0074170729
    """
    _, _, returns, _ = sample_data
    assert geometric_mean(returns) == pytest.approx(0.0074170729, abs=1e-8)


def test_time_weighted_return(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test time_weighted_return function.

    Derivation:
    TWR = ∏(1 + r_i) - 1
    TWR = (1 + 0.02) * (1 - 0.02941176) * (1 + 0.02020202) * (1 + 0.01980198) - 1
    TWR = 1.03 - 1 = 0.03
    """
    _, _, returns, _ = sample_data
    assert time_weighted_return(returns) == pytest.approx(0.03, abs=1e-8)


def test_money_weighted_return(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test money_weighted_return function.

    Derivation:
    MWR is the rate r that solves:
    0 = -100 + 2/(1+r)^(1/365) + (-3)/(1+r)^(2/365) + 2/(1+r)^(3/365) + 105/(1+r)^(4/365)

    This equation is solved numerically, resulting in r ≈ 208.14 (annualized)
    """
    values, _, _, _ = sample_data
    assert money_weighted_return(values) == pytest.approx(208.14, abs=1.0)


def test_sortino_ratio(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test sortino_ratio function.

    Derivation:
    S = (R - T) / σd
    R = 0.00764806 (arithmetic mean of returns)
    T = 0 (default target return)
    σd = sqrt(mean of squared negative returns) = sqrt((0.02941176^2) / 4) ≈ 0.01470588

    S = 0.00764806 / 0.01470588 ≈ 0.520068
    """
    _, _, returns, _ = sample_data
    assert sortino_ratio(returns) == pytest.approx(0.520068, abs=1e-6)


def test_value_at_risk(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test value_at_risk function.

    Derivation:
    VaR is the 5th percentile of returns (for 95% confidence)
    With only 4 return values, it corresponds to the minimum return
    VaR = -min(returns) = -(-0.02941176) = 0.02941176
    """
    _, _, returns, _ = sample_data
    assert value_at_risk(returns, interpolation="nearest") == pytest.approx(0.02941176, abs=1e-8)


def test_expected_shortfall(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test expected_shortfall function.

    Derivation:
    ES is the mean of returns below VaR
    With only 4 return values and 95% confidence, ES equals VaR
    ES = 0.02941176
    """
    _, _, returns, _ = sample_data
    assert expected_shortfall(returns, interpolation="nearest") == pytest.approx(0.02941176, abs=1e-8)


def test_standard_deviation(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test standard_deviation function.

    Derivation:
    σ = sqrt(Σ(r_i - μ)^2 / (n - 1))
    μ = 0.00764806 (arithmetic mean)
    σ = sqrt(((0.02 - μ)^2 + (-0.02941176 - μ)^2 + (0.02020202 - μ)^2 + (0.01980198 - μ)^2) / 3)
    σ ≈ 0.02470708645
    """
    _, _, returns, _ = sample_data
    assert standard_deviation(returns) == pytest.approx(0.02470708645, abs=1e-8)


def test_sharpe_ratio(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test sharpe_ratio function.

    Derivation:
    S = (R_p - R_f) * sqrt(252) / σ_p
    R_p = 0.00764806 (arithmetic mean of returns)
    R_f = 0 (default risk-free rate)
    σ_p = 0.02470708645 (standard deviation of returns)

    S = 0.00764806 * sqrt(252) / 0.02470708645 ≈ 4.91394
    """
    _, _, returns, _ = sample_data
    assert sharpe_ratio(returns) == pytest.approx(4.91394, abs=1e-6)


def test_maximum_drawdown(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test maximum_drawdown function.

    Derivation:
    MDD = max((peak_value - value) / peak_value)
    Peak value up to each point: [100, 102, 102, 102, 103]
    Drawdowns: [0, 0, (102-99)/102, (102-101)/102, 0]
    Maximum drawdown = (102-99)/102 = 0.02941176
    """
    values, *_ = sample_data
    assert maximum_drawdown(values) == pytest.approx(0.02941176, abs=1e-8)


def test_beta(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test beta function.

    Derivation:
    β = Cov(R_p, R_m) / Var(R_m)

    Covariance matrix of portfolio and market returns:
    Cov(R_p, R_m) = 0.0003895077929945128
    Var(R_m) = 0.000254057812022996

    β ≈ 1.53
    """
    _, _, returns, market_returns = sample_data
    assert beta(returns, market_returns) == pytest.approx(1.53, abs=1e-2)


def test_calculate_measures(sample_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series]) -> None:
    """Test calculate_measures function.

    This test checks if the function returns a DataFrame with the expected columns.
    The actual values are tested in individual measure tests.
    """
    values, market_values, _, _ = sample_data
    result = calculate_measures(values, market_values)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns) == {
        "arithmetic_mean",
        "geometric_mean",
        "twr",
        "irr",
        "sortino_ratio",
        "var_95",
        "es_95",
        "standard_deviation",
        "sharpe_ratio",
        "maximum_drawdown",
        "beta",
    }
