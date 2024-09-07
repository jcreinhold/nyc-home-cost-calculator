import numpy as np
import pytest
from scipy import stats

from nyc_home_cost_calculator.investment import InvestmentCalculator
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
    assert results.extra["monthly_returns"].shape == (360, 1000)


def test_simulate_vectorized_initial_values(default_calculator: InvestmentCalculator) -> None:
    results = default_calculator.simulate()
    assert results.portfolio_values is not None
    np.testing.assert_almost_equal(results.portfolio_values[0], 100_000.0)


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
    monthly_returns = results.extra["monthly_returns"]
    _, p_value = stats.ttest_1samp(monthly_returns.flatten(), default_calculator.mean_return_rate / 12)
    assert p_value > 0.05  # Returns should be consistent with specified mean


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
    np.testing.assert_almost_equal(results.portfolio_values[-1], expected_final_value, decimal=2)


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
    assert np.mean(results.extra["monthly_returns"]) < 0
    assert np.mean(results.portfolio_values[-1]) < 100_000 * 360  # Less than total contributions
