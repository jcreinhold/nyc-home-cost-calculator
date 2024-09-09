from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nyc_home_cost_calculator.investment import InvestmentCalculator
from nyc_home_cost_calculator.portfolio import Portfolio
from nyc_home_cost_calculator.simulate import SimulationEngine


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def default_calculator(rng: np.random.Generator) -> InvestmentCalculator:
    return InvestmentCalculator(simulations=1_000, rng=rng)


def test_initialization(default_calculator: InvestmentCalculator) -> None:
    assert default_calculator.initial_investment == 100_000.0
    assert default_calculator.monthly_contribution == 1_000.0
    assert default_calculator.total_years == 30
    assert default_calculator.mean_return_rate == 0.07
    assert default_calculator.volatility == 0.15
    assert default_calculator.degrees_of_freedom == 5
    assert default_calculator.simulations == 1_000


def test_get_input_parameters(default_calculator: InvestmentCalculator) -> None:
    params = default_calculator._get_input_parameters()
    assert len(params) == 7
    assert params[0] == ("Initial Investment", "$100,000.00")
    assert params[1] == ("Monthly Contribution", "$1,000.00")
    assert params[2] == ("Total Years", "30")
    assert params[3] == ("Mean Annual Return Rate", "7.00%")
    assert params[4] == ("Annual Volatility", "15.00%")
    assert params[5] == ("Degrees of Freedom", "5")
    assert params[6] == ("Number of Simulations", "1000")


def test_simulate_vectorized_shape(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()
    assert results.monthly_costs.shape == (360, 1000)
    assert results.profit_loss.shape == (360, 1000)
    assert results.portfolio_values is not None
    assert results.investment_returns is not None
    assert results.portfolio_values.shape == (360, 1000)
    assert results.investment_returns.shape == (360, 1000)
    assert results.extra["monthly_log_returns"].shape == (360, 1000)


def test_simulate_vectorized_initial_values(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()
    assert results.portfolio_values is not None
    assert results.extra is not None
    initial_log_returns = results.extra["monthly_log_returns"][0]
    expected_initial_values = 100_000.0 * np.exp(initial_log_returns)
    np.testing.assert_allclose(results.portfolio_values[0], expected_initial_values, rtol=1e-6)


def test_simulate_vectorized_final_values(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()

    assert results.portfolio_values is not None
    assert results.investment_returns is not None
    assert np.all(results.investment_returns[-1] != 1.0)  # Should have changed


def test_simulate_vectorized_monotonic_contributions(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()
    cumulative_contributions = np.cumsum(-results.monthly_costs, axis=0)
    assert np.all(np.diff(cumulative_contributions, axis=0) >= 0)


def test_simulate_vectorized_return_distribution(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()
    assert results.extra is not None
    monthly_log_returns = results.extra["monthly_log_returns"]
    expected_mean = (
        np.log(1 + default_calculator.mean_return_rate / 12) - 0.5 * (default_calculator.volatility / np.sqrt(12)) ** 2
    )
    _, p_value = stats.ttest_1samp(monthly_log_returns.flatten(), expected_mean)
    assert p_value > 0.05  # Log returns should be consistent with specified mean


def test_zero_volatility(rng: np.random.Generator) -> None:
    calculator = InvestmentCalculator(
        initial_investment=100_000,
        monthly_contribution=0,
        total_years=30,
        mean_return_rate=0.07,
        volatility=0.0,
        degrees_of_freedom=5,
        simulations=1_000,
        rng=rng,
    )
    results = calculator.simulate()
    monthly_return_rate = 1 + 0.07 / 12
    months = 30 * 12
    expected_final_value = 100_000 * monthly_return_rate**months
    assert results.portfolio_values is not None
    np.testing.assert_allclose(results.portfolio_values[-1], expected_final_value, rtol=1e-6)


def test_negative_returns(rng: np.random.Generator) -> None:
    calculator = InvestmentCalculator(
        initial_investment=100_000,
        monthly_contribution=1_000,
        total_years=30,
        mean_return_rate=-0.05,
        volatility=0.15,
        degrees_of_freedom=5,
        simulations=1_000,
        rng=rng,
    )
    engine = SimulationEngine(360, 1000, calculator.rng)
    results = engine.run_simulation(calculator._simulate_vectorized)

    assert results.extra is not None
    assert results.portfolio_values is not None
    assert np.mean(results.extra["monthly_log_returns"]) < 0
    assert np.mean(results.portfolio_values[-1]) < 100_000 * 360  # Less than total contributions


@pytest.fixture
def mock_yf_download(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(42)

    def _mock_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "QQQ": rng.standard_normal(100).cumsum() + 100.0,
                "SPY": rng.standard_normal(100).cumsum() + 100.0,
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_product([["Adj Close"], data.columns])
        return data

    monkeypatch.setattr("yfinance.download", _mock_download)


@pytest.fixture
def sample_portfolio(mock_yf_download: pytest.MonkeyPatch) -> Portfolio:
    return Portfolio(["SPY", "QQQ"], [0.5, 0.5])


def test_investment_calculator_from_portfolio(sample_portfolio: Portfolio) -> None:
    # Initialize InvestmentCalculator from the sample portfolio
    calculator = InvestmentCalculator.from_portfolio(sample_portfolio)

    # Test initial investment
    assert calculator.initial_investment == pytest.approx(sample_portfolio.data.iloc[0].sum())

    # Test total years
    expected_years = len(sample_portfolio.data) // 252
    assert calculator.total_years == expected_years

    # Test mean return rate (annualized)
    expected_mean_return = sample_portfolio.metrics["arithmetic_mean"]
    assert calculator.mean_return_rate == pytest.approx(expected_mean_return)

    # Test volatility (annualized)
    expected_volatility = sample_portfolio.metrics["volatility"]
    assert calculator.volatility == pytest.approx(expected_volatility)

    # Test default values
    assert calculator.monthly_contribution == 0
    assert calculator.degrees_of_freedom == 5
    assert calculator.simulations == 5000

    # Test overriding parameters
    custom_calculator = InvestmentCalculator.from_portfolio(
        sample_portfolio, monthly_contribution=1000, degrees_of_freedom=7, simulations=10000
    )
    assert custom_calculator.monthly_contribution == 1000
    assert custom_calculator.degrees_of_freedom == 7
    assert custom_calculator.simulations == 10000
