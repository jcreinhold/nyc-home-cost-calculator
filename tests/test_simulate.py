import numpy as np
import pandas as pd
import pytest

from nyc_home_cost_calculator.simulate import SimulationEngine, SimulationResults


@pytest.fixture
def sample_results() -> SimulationResults:
    return SimulationResults(
        monthly_costs=np.array([[100.0, 200.0], [150.0, 250.0]]),
        profit_loss=np.array([[1000.0, 2000.0], [1500.0, 2500.0]]),
        total_years=2,
        simulations=2,
        home_values=np.array([[300000.0, 350000.0], [310000.0, 360000.0]]),
        personal_income=np.array([[5000.0, 6000.0], [5100.0, 6100.0]]),
        marital_status=np.array([[0, 1], [0, 1]]),
        extra={"test": np.array([1, 2])},
    )


def test_simulation_results_initialization(sample_results: SimulationResults) -> None:
    assert sample_results.monthly_costs.shape == (2, 2)
    assert sample_results.profit_loss.shape == (2, 2)
    assert sample_results.home_values is not None
    assert sample_results.personal_income is not None
    assert sample_results.marital_status is not None
    assert "test" in sample_results.extra


def test_simulation_results_get_percentiles(sample_results: SimulationResults) -> None:
    percentiles = sample_results.get_percentiles("monthly_costs", [25, 50, 75])
    assert percentiles.shape == (3, 2)


def test_simulation_results_get_mean(sample_results: SimulationResults) -> None:
    mean = sample_results.get_mean("profit_loss")
    assert mean.shape == (2,)
    np.testing.assert_almost_equal(mean, np.array([1500, 2000]))


def test_simulation_results_get_std_dev(sample_results: SimulationResults) -> None:
    std_dev = sample_results.get_std_dev("home_values")
    assert std_dev.shape == (2,)


def test_simulation_results_addition(sample_results: SimulationResults) -> None:
    result = sample_results + sample_results
    assert np.array_equal(result.monthly_costs, sample_results.monthly_costs * 2)


def test_simulation_results_subtraction(sample_results: SimulationResults) -> None:
    result = sample_results - sample_results
    assert np.array_equal(result.monthly_costs, np.zeros_like(sample_results.monthly_costs))


def test_simulation_results_multiplication(sample_results: SimulationResults) -> None:
    result = sample_results * sample_results
    assert np.array_equal(result.monthly_costs, sample_results.monthly_costs**2)


def test_simulation_results_division(sample_results: SimulationResults) -> None:
    result = sample_results / sample_results
    assert np.array_equal(result.monthly_costs, np.ones_like(sample_results.monthly_costs))


def test_simulation_results_zeros_like(sample_results: SimulationResults) -> None:
    zeros = SimulationResults.zeros_like(sample_results)
    assert np.array_equal(zeros.monthly_costs, np.zeros_like(sample_results.monthly_costs))
    assert zeros.extra == {}


@pytest.fixture
def sample_engine() -> SimulationEngine:
    return SimulationEngine(total_months=120, simulations=1000, rng=np.random.default_rng(42))


def test_simulation_engine_initialization(sample_engine: SimulationEngine) -> None:
    assert sample_engine.total_months == 120
    assert sample_engine.simulations == 1000
    assert isinstance(sample_engine.rng, np.random.Generator)


def test_simulation_engine_run_simulation(sample_engine: SimulationEngine) -> None:
    def mock_simulate(months: np.ndarray) -> SimulationResults:
        return SimulationResults(
            monthly_costs=np.ones_like(months),
            profit_loss=np.cumsum(np.ones_like(months), axis=0),
            total_years=10,
            simulations=1000,
        )

    results = sample_engine.run_simulation(mock_simulate)
    assert results.monthly_costs.shape == (120, 1000)
    assert results.profit_loss.shape == (120, 1000)
    assert np.array_equal(results.monthly_costs, np.ones((120, 1000)))
    assert np.array_equal(results.profit_loss[-1], np.full((1000,), 120))


def test_simulation_results_invalid_field(sample_results: SimulationResults) -> None:
    with pytest.raises(AttributeError):
        sample_results.get_mean("invalid_field")


def test_simulation_results_invalid_addition(sample_results: SimulationResults) -> None:
    with pytest.raises(TypeError):
        sample_results + 5  # type: ignore[operator]


def test_simulation_results_to_dataframe() -> None:
    # Create a sample SimulationResults object
    results = SimulationResults(
        monthly_costs=np.array([[100, 200], [150, 250], [200, 300]]),
        profit_loss=np.array([[-50, -100], [50, 100], [150, 200]]),
        total_years=0.25,  # 3 months
        simulations=2,
        home_values=np.array([[300000, 310000], [320000, 330000], [340000, 350000]]),
        extra={"nested": {"value": 42, "array": np.array([1, 2, 3])}},
    )

    # Convert to DataFrame
    df = results.to_dataframe()

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Three time steps
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "monthly_costs" in df.columns
    assert "profit_loss" in df.columns
    assert "home_values" in df.columns
    assert "total_years" in df.columns
    assert "simulations" in df.columns
    assert "extra_nested_value" in df.columns
    assert "extra_nested_array" in df.columns
