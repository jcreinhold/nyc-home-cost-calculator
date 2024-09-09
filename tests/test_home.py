from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from nyc_home_cost_calculator.home import FilingStatus, NYCHomeCostCalculator
from nyc_home_cost_calculator.simulate import SimulationResults


@pytest.fixture(scope="session")
def default_calculator() -> NYCHomeCostCalculator:
    return NYCHomeCostCalculator(
        home_price=1_000_000,
        down_payment=200_000,
        mortgage_rate=0.03,
        loan_term=30,
        initial_income=150_000,
        hoa_fee=500,
        insurance_rate=0.005,
        maintenance_rate=0.01,
        property_tax_rate=0.01,
        mean_appreciation_rate=0.03,
        appreciation_volatility=0.05,
        mean_inflation_rate=0.02,
        inflation_volatility=0.01,
        mean_income_change_rate=0.02,
        income_change_volatility=0.03,
        retirement_contribution_rate=0.15,
        filing_status=FilingStatus.SINGLE,
        simulations=100,
        rng=np.random.default_rng(42),
    )


@pytest.fixture(scope="session")
def default_results(default_calculator: NYCHomeCostCalculator) -> SimulationResults:
    return default_calculator.simulate()


def test_initialization(default_calculator: NYCHomeCostCalculator) -> None:
    assert default_calculator.home_price == 1_000_000
    assert default_calculator.down_payment == 200_000
    assert default_calculator.mortgage_rate == 0.03
    assert default_calculator.loan_term == 30
    assert default_calculator.initial_income == 150_000
    assert default_calculator.hoa_fee == 500
    assert default_calculator.insurance_rate == 0.005
    assert default_calculator.maintenance_rate == 0.01
    assert default_calculator.property_tax_rate == 0.01
    assert default_calculator.mean_appreciation_rate == 0.03
    assert default_calculator.appreciation_volatility == 0.05
    assert default_calculator.mean_inflation_rate == 0.02
    assert default_calculator.inflation_volatility == 0.01
    assert default_calculator.mean_income_change_rate == 0.02
    assert default_calculator.income_change_volatility == 0.03
    assert default_calculator.retirement_contribution_rate == 0.15
    assert default_calculator.simulations == 100


def test_calculate_monthly_payment(default_calculator: NYCHomeCostCalculator) -> None:
    payment = default_calculator.calculate_monthly_payment(800_000, 0.03, 360)
    assert 3_000 < payment < 4_000  # Reasonable range for given parameters


def test_simulate_costs_over_time(
    default_calculator: NYCHomeCostCalculator, default_results: SimulationResults
) -> None:
    costs = default_results.cumulative_costs
    assert costs is not None
    assert len(costs) == (12 * default_calculator.loan_term), costs.shape
    assert len(costs[0]) == default_calculator.simulations, costs.shape
    assert all(isinstance(cost, float) for year_costs in costs for cost in year_costs)


def test_get_cost_statistics(default_results: SimulationResults) -> None:
    stats = default_results.get_cost_statistics()
    assert set(stats.keys()) == {"mean", "median", "std_dev", "percentile_5", "percentile_95"}
    assert all(isinstance(value, float) for value in stats.values())


@pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 300})
def test_plot_costs_over_time(default_results: SimulationResults) -> None:
    default_results.plot()


def test_export_to_excel(default_results: SimulationResults, tmp_path: Path) -> None:
    filename = tmp_path / "test_export.xlsx"
    default_results.export_to_excel(str(filename))
    assert filename.exists()


def test_edge_cases() -> None:
    # Test with minimum down payment
    calculator = NYCHomeCostCalculator(
        home_price=500_000,
        down_payment=1,  # Extremely low down payment
        mortgage_rate=0.05,
        loan_term=15,
        initial_income=50_000,
        hoa_fee=100,
        insurance_rate=0.01,
        maintenance_rate=0.02,
        property_tax_rate=0.02,
        mean_appreciation_rate=0.01,
        appreciation_volatility=0.1,
        mean_inflation_rate=0.03,
        inflation_volatility=0.02,
        mean_income_change_rate=0.01,
        income_change_volatility=0.05,
        retirement_contribution_rate=0.05,
        filing_status=FilingStatus.MARRIED_JOINT,
        simulations=100,
    )
    results = calculator.simulate()
    stats = results.get_cost_statistics()
    assert all(isinstance(value, float) for value in stats.values())

    # Test with very high appreciation and income growth
    calculator = NYCHomeCostCalculator(
        home_price=1000000,
        down_payment=500000,
        mortgage_rate=0.02,
        loan_term=30,
        initial_income=200000,
        hoa_fee=1000,
        insurance_rate=0.003,
        maintenance_rate=0.005,
        property_tax_rate=0.005,
        mean_appreciation_rate=0.1,  # Very high appreciation
        appreciation_volatility=0.02,
        mean_inflation_rate=0.01,
        inflation_volatility=0.005,
        mean_income_change_rate=0.08,  # Very high income growth
        income_change_volatility=0.01,
        retirement_contribution_rate=0.2,
        filing_status=FilingStatus.MARRIED_SEPARATE,
        simulations=100,
    )
    results = calculator.simulate()
    stats = results.get_cost_statistics()
    assert all(isinstance(value, float) for value in stats.values())
