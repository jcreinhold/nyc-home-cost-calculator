import numpy as np
import pytest

from nyc_home_cost_calculator.tax import Bracket, EffectiveTaxRate, FilingStatus, TaxCalculator


@pytest.fixture
def tax_calculator() -> TaxCalculator:
    return TaxCalculator()


def test_get_mortgage_interest_deduction_limit(tax_calculator: TaxCalculator) -> None:
    mortgage_rates = np.array([0.03, 0.04, 0.05])
    filing_status = np.array([FilingStatus.SINGLE, FilingStatus.MARRIED_JOINT, FilingStatus.MARRIED_SEPARATE])
    limits = tax_calculator.get_mortgage_interest_deduction_limit(mortgage_rates, filing_status)
    expected = np.array([750_000 * 0.03, 750_000 * 0.04, 375_000 * 0.05])
    np.testing.assert_almost_equal(limits, expected)


def test_get_standard_deduction(tax_calculator: TaxCalculator) -> None:
    filing_status = np.array([FilingStatus.SINGLE, FilingStatus.MARRIED_JOINT, FilingStatus.MARRIED_SEPARATE])
    deductions = tax_calculator.get_standard_deduction(filing_status)
    expected = np.array([13_850.0, 27_700.0, 13_850.0])
    np.testing.assert_almost_equal(deductions, expected)


def test_get_salt_deduction_limit(tax_calculator: TaxCalculator) -> None:
    filing_status = np.array([FilingStatus.SINGLE, FilingStatus.MARRIED_JOINT, FilingStatus.MARRIED_SEPARATE])
    limits = tax_calculator.get_salt_deduction_limit(filing_status)
    expected = np.array([10_000.0, 10_000.0, 5_000.0])
    np.testing.assert_almost_equal(limits, expected)


def test_calculate_tax_deduction(tax_calculator: TaxCalculator) -> None:
    ages = np.array([45, 55])
    mortgage_rates = np.array([0.03, 0.04])
    mortgage_interests = np.array([15000.0, 20000.0])
    property_taxes = np.array([8000.0, 12000.0])
    federal_rates = np.array([0.22, 0.24])
    state_rates = np.array([0.06, 0.07])
    local_rates = np.array([0.035, 0.04])
    retirement_contributions = np.array([19500.0, 26000.0])
    annual_incomes = np.array([100000.0, 150000.0])
    filing_status = np.array([FilingStatus.SINGLE, FilingStatus.MARRIED_JOINT])

    deductions = tax_calculator.calculate_tax_deduction(
        ages,
        mortgage_rates,
        mortgage_interests,
        property_taxes,
        federal_rates,
        state_rates,
        local_rates,
        retirement_contributions,
        annual_incomes,
        filing_status,
    )

    # The exact expected values would depend on the specific implementation
    # Here we're just checking that the deductions are non-negative and less than the income
    assert np.all(deductions >= 0)
    assert np.all(deductions < annual_incomes)


def test_calculate_tax(tax_calculator: TaxCalculator) -> None:
    incomes = np.array([50000.0, 100000.0, 200000.0])
    brackets = [Bracket(0.0, 10000.0, 0.10), Bracket(10000.0, 50000.0, 0.15), Bracket(50000.0, float("inf"), 0.25)]
    taxes = tax_calculator.calculate_tax(incomes, brackets)

    # Calculate expected taxes manually
    expected = np.array([
        10000 * 0.10 + 40000 * 0.15,
        10000 * 0.10 + 40000 * 0.15 + 50000 * 0.25,
        10000 * 0.10 + 40000 * 0.15 + 150000 * 0.25,
    ])

    np.testing.assert_almost_equal(taxes, expected)

    # Add a test for a simple case
    simple_income = np.array([5000.0])
    simple_brackets = [Bracket(0.0, 10000.0, 0.10)]
    simple_tax = tax_calculator.calculate_tax(simple_income, simple_brackets)
    np.testing.assert_almost_equal(simple_tax, np.array([500.0]))


def test_calculate_effective_tax_rates(tax_calculator: TaxCalculator) -> None:
    incomes = np.array([50000.0, 100000.0, 200000.0])
    filing_status = np.array([FilingStatus.SINGLE, FilingStatus.MARRIED_JOINT, FilingStatus.MARRIED_SEPARATE])
    rates = tax_calculator.calculate_effective_tax_rates(incomes, filing_status)

    assert isinstance(rates, EffectiveTaxRate)
    assert rates.federal_rate.shape == (3,)
    assert rates.state_rate.shape == (3,)
    assert rates.local_rate.shape == (3,)
    assert np.all(rates.federal_rate >= 0)
    assert np.all(rates.federal_rate <= 1)
    assert np.all(rates.state_rate >= 0)
    assert np.all(rates.state_rate <= 1)
    assert np.all(rates.local_rate >= 0)
    assert np.all(rates.local_rate <= 1)


def test_adjust_brackets_for_inflation(tax_calculator: TaxCalculator) -> None:
    brackets = [Bracket(0.0, 10000.0, 0.10), Bracket(10000.0, float("inf"), 0.20)]
    inflation_rates = np.array([0.02, 0.03])
    adjusted = tax_calculator._adjust_brackets_for_inflation(brackets, inflation_rates)

    lower0 = adjusted[0].lower
    upper0 = adjusted[0].upper
    lower1 = adjusted[1].lower
    upper1 = adjusted[1].upper

    assert isinstance(lower0, np.ndarray)
    assert isinstance(lower1, np.ndarray)
    assert isinstance(upper0, np.ndarray)
    assert isinstance(upper1, np.ndarray)

    lower00, upper00 = lower0[0], upper0[0]
    lower01, upper01 = lower0[1], upper0[1]
    lower10, upper10 = lower1[0], upper1[0]
    lower11, upper11 = lower1[1], upper1[1]

    assert isinstance(lower00, float)
    assert isinstance(lower01, float)
    assert isinstance(lower10, float)
    assert isinstance(lower11, float)
    assert isinstance(upper00, float)
    assert isinstance(upper01, float)
    assert isinstance(upper10, float)
    assert isinstance(upper11, float)

    assert lower00 == 0
    assert upper00 == 10200
    assert lower01 == 0
    assert upper01 == 10300
    assert lower10 == 10200
    assert np.isinf(upper10)
    assert lower11 == 10300
    assert np.isinf(upper11)


def test_get_tax_brackets(tax_calculator: TaxCalculator) -> None:
    incomes = np.array([5000.0, 15000.0, 50000.0])
    brackets = [Bracket(0.0, 10000.0, 0.10), Bracket(10000.0, 30000.0, 0.15), Bracket(30000.0, float("inf"), 0.25)]
    rates = tax_calculator.get_tax_brackets(incomes, brackets)
    expected = np.array([0.10, 0.15, 0.25])
    np.testing.assert_almost_equal(rates, expected)
