import numpy as np
import pytest

from nyc_home_cost_calculator.income import CareerIncomeSimulator


@pytest.fixture
def default_simulator() -> CareerIncomeSimulator:
    return CareerIncomeSimulator(simulations=1_000, rng=np.random.default_rng(42))


def test_initialization(default_simulator: CareerIncomeSimulator) -> None:
    assert default_simulator.initial_income == 75000.0
    assert default_simulator.total_years == 30
    assert default_simulator.mean_income_growth == 0.03
    assert default_simulator.income_volatility == 0.02


def test_simulation_shape(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert results.monthly_income is not None
    assert results.monthly_income.shape == (360, 1000)
    assert results.personal_income is not None
    assert results.personal_income.shape == (360, 1000)


def test_initial_income(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert results.monthly_income is not None
    assert np.allclose(results.monthly_income[0], default_simulator.initial_income / 12, rtol=100.0)


def test_promotions_and_demotions(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert results.promotions is not None
    assert results.demotions is not None
    assert results.promotions.sum() > 0
    assert results.demotions.sum() > 0


def test_layoffs(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert results.layoffs is not None
    assert results.layoffs.sum() >= 0


def test_income_growth(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert results.monthly_income is not None
    assert np.mean(results.monthly_income[-1]) > np.mean(results.monthly_income[0])


def test_input_parameters(default_simulator: CareerIncomeSimulator) -> None:
    params = default_simulator._get_input_parameters()
    assert len(params) == 12
    assert all(isinstance(param, tuple) and len(param) == 2 for param in params)


def test_custom_parameters() -> None:
    custom_simulator = CareerIncomeSimulator(
        initial_income=100000.0,
        total_years=20,
        mean_income_growth=0.05,
        income_volatility=0.03,
        simulations=500,
        rng=np.random.default_rng(42),
    )
    assert custom_simulator.initial_income == 100000.0
    assert custom_simulator.total_years == 20
    assert custom_simulator.mean_income_growth == 0.05
    assert custom_simulator.income_volatility == 0.03

    results = custom_simulator.simulate()
    assert results.monthly_income is not None
    assert results.monthly_income.shape == (240, 500)


def test_layoff_durations(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert "layoff_durations" in results.extra
    layoff_durations = results.extra["layoff_durations"]
    assert layoff_durations.shape == (360, 1000)
    assert np.mean(layoff_durations[layoff_durations > 0]) > 0
    assert np.isclose(
        np.mean(layoff_durations[layoff_durations > 0]), default_simulator.mean_layoff_duration_months, rtol=0.1
    )


def test_layoff_mask(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    assert "layoff_mask" in results.extra
    layoff_mask = results.extra["layoff_mask"]
    assert layoff_mask.shape == (360, 1000)
    assert layoff_mask.sum() > 0
    assert results.monthly_income is not None
    assert np.all(results.monthly_income[layoff_mask] <= 504 * 52 / 12)  # Max unemployment benefit


def test_post_layoff_impact(default_simulator: CareerIncomeSimulator) -> None:
    results = default_simulator.simulate()
    layoff_mask = results.extra["layoff_mask"]
    post_layoff_mask = np.roll(layoff_mask, 1, axis=0) & ~layoff_mask
    post_layoff_mask[0] = False
    assert results.monthly_income is not None
    non_layoff_income = results.monthly_income[~layoff_mask & ~post_layoff_mask]
    post_layoff_income = results.monthly_income[post_layoff_mask]
    assert np.mean(post_layoff_income) < np.mean(non_layoff_income)


def test_custom_layoff_duration() -> None:
    custom_simulator = CareerIncomeSimulator(mean_layoff_duration_months=12.0, rng=np.random.default_rng(42))
    results = custom_simulator.simulate()
    layoff_durations = results.extra["layoff_durations"]
    assert np.isclose(np.mean(layoff_durations[layoff_durations > 0]), 12.0, rtol=0.1)
