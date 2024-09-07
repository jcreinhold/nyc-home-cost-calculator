from typing import TYPE_CHECKING

import numpy as np
import pytest

from nyc_home_cost_calculator.income import CareerIncomeSimulator
from nyc_home_cost_calculator.life import FinancialLifeSimulator
from nyc_home_cost_calculator.tax import FilingStatus

if TYPE_CHECKING:
    from nyc_home_cost_calculator.simulate import SimulationResults


@pytest.fixture
def career_simulator() -> CareerIncomeSimulator:
    return CareerIncomeSimulator(
        initial_income=100_000.0,
        total_years=30,
        mean_income_growth=0.03,
        income_volatility=0.02,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def default_life_simulator(career_simulator: CareerIncomeSimulator) -> FinancialLifeSimulator:
    return FinancialLifeSimulator(
        career_simulator=career_simulator,
        initial_age=25,
        marriage_probability=0.05,
        divorce_probability=0.02,
        partner_income_ratio=1.0,
        divorce_cost=50_000.0,
        initial_marriage_status=FilingStatus.SINGLE,
        simulations=1000,
        rng=np.random.default_rng(42),
    )


def test_initialization(default_life_simulator: FinancialLifeSimulator) -> None:
    assert default_life_simulator.initial_age == 25
    assert default_life_simulator.marriage_probability == 0.05
    assert default_life_simulator.divorce_probability == 0.02
    assert default_life_simulator.partner_income_ratio == 1.0
    assert default_life_simulator.divorce_cost == 50_000.0
    assert default_life_simulator.initial_marriage_status == FilingStatus.SINGLE
    assert default_life_simulator.simulations == 1000


def test_simulation_shape(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert results.monthly_costs.shape == (360, 1000)
    assert results.profit_loss.shape == (360, 1000)


def test_initial_marital_status(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert results.marital_status is not None
    assert np.all(results.marital_status[0] == 0)  # All start as single


def test_marital_status_changes(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert results.marital_status is not None
    marital_status = results.marital_status
    assert np.any(marital_status[1:] != marital_status[0])  # Some status changes occur


def test_household_income(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert results.household_income is not None
    assert results.personal_income is not None
    assert np.all(results.household_income >= results.personal_income)


def test_divorce_costs(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert "divorce_costs" in results.extra
    assert np.any(results.extra["divorce_costs"] > 0)  # Some divorces occur
    assert np.all(results.extra["divorce_costs"][results.extra["divorce_costs"] > 0] == 50_000.0)


def test_partner_income(default_life_simulator: FinancialLifeSimulator) -> None:
    results: SimulationResults = default_life_simulator.simulate()
    assert results.partner_income is not None
    assert results.marital_status is not None
    partner_income = results.partner_income
    marital_status = results.marital_status
    assert np.all(partner_income[marital_status == 0] == 0)  # No partner income when single
    assert np.any(partner_income[marital_status == 1] > 0)  # Some partner income when married


def test_initial_married_status() -> None:
    career_sim = CareerIncomeSimulator(rng=np.random.default_rng(42))
    life_sim = FinancialLifeSimulator(
        career_simulator=career_sim, initial_marriage_status=FilingStatus.MARRIED_JOINT, rng=np.random.default_rng(42)
    )
    results: SimulationResults = life_sim.simulate()
    assert results.marital_status is not None
    assert np.all(results.marital_status[0] == 1)  # All start as married


def test_input_parameters(default_life_simulator: FinancialLifeSimulator) -> None:
    params = default_life_simulator._get_input_parameters()
    assert len(params) > 0
    assert all(isinstance(param, tuple) and len(param) == 2 for param in params)


@pytest.mark.parametrize(("marriage_prob", "divorce_prob"), [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
def test_extreme_probabilities(
    career_simulator: CareerIncomeSimulator, marriage_prob: float, divorce_prob: float
) -> None:
    life_sim = FinancialLifeSimulator(
        career_simulator=career_simulator,
        marriage_probability=marriage_prob,
        divorce_probability=divorce_prob,
        rng=np.random.default_rng(42),
    )
    results: SimulationResults = life_sim.simulate()
    assert results.marital_status is not None
    marital_status = results.marital_status

    if marriage_prob == 0.0 and divorce_prob == 0.0:
        assert np.all(marital_status == 0)  # All remain single
    elif marriage_prob == 1.0 and divorce_prob == 0.0:
        assert np.all(marital_status[-1] == 1)  # All end up married
    elif marriage_prob == 0.0 and divorce_prob == 1.0:
        assert np.all(marital_status == 0)  # All remain single (can't divorce if never married)
