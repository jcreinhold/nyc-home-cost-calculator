import math
import random

import pytest

from nyc_home_cost_calculator.calculator import NYCHomeCostCalculator


@pytest.fixture()
def default_calculator() -> NYCHomeCostCalculator:
    return NYCHomeCostCalculator(
        home_price=1000000,
        down_payment=200000,
        mortgage_rate=0.03,
        loan_term=30,
        initial_income=150000,
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
        simulations=1000,
        rng=random.Random(42),
    )


def test_initialization(default_calculator: NYCHomeCostCalculator) -> None:
    assert default_calculator.home_price == 1000000
    assert default_calculator.down_payment == 200000
    assert default_calculator.mortgage_rate == 0.03
    assert default_calculator.loan_term == 30
    assert default_calculator.initial_income == 150000
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
    assert default_calculator.simulations == 1000


def test_calculate_tax(default_calculator: NYCHomeCostCalculator) -> None:
    income = 100000
    brackets = [(0, 50000, 0.1), (50000, 100000, 0.2), (100000, float("inf"), 0.3)]
    expected_tax = 50000 * 0.1 + 50000 * 0.2
    assert math.isclose(default_calculator.calculate_tax(income, brackets), expected_tax)


def test_generate_random_rates(default_calculator: NYCHomeCostCalculator) -> None:
    rates = default_calculator.generate_random_rates()
    assert len(rates) == 6
    assert all(isinstance(rate, float) for rate in rates)
    assert -0.5 <= rates[2] <= 1.0  # income_change_rate


def test_calculate_effective_tax_rates(default_calculator: NYCHomeCostCalculator) -> None:
    rates = default_calculator.calculate_effective_tax_rates(150000, 0, 0, 0)
    assert len(rates) == 3
    assert all(0 <= rate <= 1 for rate in rates)


def test_calculate_tax_deduction(default_calculator: NYCHomeCostCalculator) -> None:
    deduction = default_calculator.calculate_tax_deduction(10000, 5000, 0.2, 0.05, 0.03)
    assert deduction >= 0


def test_calculate_monthly_payment(default_calculator: NYCHomeCostCalculator) -> None:
    payment = default_calculator.calculate_monthly_payment(800000, 0.03, 360)
    assert 3000 < payment < 4000  # Reasonable range for given parameters


def test_simulate_costs_over_time(default_calculator: NYCHomeCostCalculator) -> None:
    costs = default_calculator.simulate_costs_over_time()
    assert len(costs) == default_calculator.loan_term
    assert len(costs[0]) == default_calculator.simulations
    assert all(isinstance(cost, float) for year_costs in costs for cost in year_costs)


def test_get_cost_statistics(default_calculator: NYCHomeCostCalculator) -> None:
    stats = default_calculator.get_cost_statistics()
    assert set(stats.keys()) == {"mean", "median", "std_dev", "percentile_5", "percentile_95"}
    assert all(isinstance(value, float) for value in stats.values())


@pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 300})
def test_plot_costs_over_time(default_calculator: NYCHomeCostCalculator) -> None:
    fig = default_calculator.plot_costs_over_time()
    return fig


def test_export_to_excel(default_calculator: NYCHomeCostCalculator, tmp_path) -> None:
    filename = tmp_path / "test_export.xlsx"
    default_calculator.export_to_excel(str(filename))
    assert filename.exists()


def test_edge_cases() -> None:
    # Test with minimum down payment
    calculator = NYCHomeCostCalculator(
        home_price=500000,
        down_payment=1,  # Extremely low down payment
        mortgage_rate=0.05,
        loan_term=15,
        initial_income=50000,
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
        simulations=100,
    )
    stats = calculator.get_cost_statistics()
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
        simulations=100,
    )
    stats = calculator.get_cost_statistics()
    assert all(isinstance(value, float) for value in stats.values())
