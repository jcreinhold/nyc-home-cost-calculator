"""Tax Calculator class."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np


class FilingStatus(Enum):
    """Enumeration for tax filing status."""

    SINGLE = "single"
    MARRIED_JOINT = "married_filing_jointly"
    MARRIED_SEPARATE = "married_filing_separately"


class EffectiveTaxRate(NamedTuple):
    """NamedTuple representing the effective tax rates."""

    federal_rate: np.ndarray
    state_rate: np.ndarray
    local_rate: np.ndarray


class Bracket(NamedTuple):
    """NamedTuple representing a tax bracket."""

    lower: float | np.ndarray
    upper: float | np.ndarray
    rate: float | np.ndarray


class TaxCalculator:
    """Tax calculator class for calculating tax deductions and rates based on income and filing status."""

    def __init__(self):
        """Initialize the tax calculator."""
        self.federal_brackets = {
            FilingStatus.SINGLE: [
                Bracket(0.0, 11_000.0, 0.10),
                Bracket(11_000.0, 44_725.0, 0.12),
                Bracket(44_725.0, 95_375.0, 0.22),
                Bracket(95_375.0, 182_100.0, 0.24),
                Bracket(182_100.0, 231_250.0, 0.32),
                Bracket(231_250.0, 578_125.0, 0.35),
                Bracket(578_125.0, float("inf"), 0.37),
            ],
            FilingStatus.MARRIED_JOINT: [
                Bracket(0.0, 22_000.0, 0.10),
                Bracket(22_000.0, 89_450.0, 0.12),
                Bracket(89_450.0, 190_750.0, 0.22),
                Bracket(190_750.0, 364_200.0, 0.24),
                Bracket(364_200.0, 462_500.0, 0.32),
                Bracket(462_500.0, 693_750.0, 0.35),
                Bracket(693_750.0, float("inf"), 0.37),
            ],
            FilingStatus.MARRIED_SEPARATE: [
                Bracket(0.0, 11_000.0, 0.10),
                Bracket(11_000.0, 44_725.0, 0.12),
                Bracket(44_725.0, 95_375.0, 0.22),
                Bracket(95_375.0, 182_100.0, 0.24),
                Bracket(182_100.0, 231_250.0, 0.32),
                Bracket(231_250.0, 346_875.0, 0.35),
                Bracket(346_875.0, float("inf"), 0.37),
            ],
        }
        self.ny_state_brackets = {
            FilingStatus.SINGLE: [
                Bracket(0.0, 8_500.0, 0.04),
                Bracket(8_500.0, 11_700.0, 0.045),
                Bracket(11_700.0, 13_900.0, 0.0525),
                Bracket(13_900.0, 80_650.0, 0.0585),
                Bracket(80_650.0, 215_400.0, 0.0625),
                Bracket(215_400.0, 1_077_550.0, 0.0685),
                Bracket(1_077_550.0, float("inf"), 0.0882),
            ],
            FilingStatus.MARRIED_JOINT: [
                Bracket(0.0, 17150.0, 0.04),
                Bracket(17_150.0, 23_600.0, 0.045),
                Bracket(23_600.0, 27_900.0, 0.0525),
                Bracket(27_900.0, 161_550.0, 0.0585),
                Bracket(161_550.0, 323_200.0, 0.0625),
                Bracket(323_200.0, 2_155_350.0, 0.0685),
                Bracket(2_155_350.0, float("inf"), 0.0882),
            ],
            FilingStatus.MARRIED_SEPARATE: [
                Bracket(0.0, 8_500.0, 0.04),
                Bracket(8_500.0, 11_800.0, 0.045),
                Bracket(11_800.0, 13_950.0, 0.0525),
                Bracket(13_950.0, 80_800.0, 0.0585),
                Bracket(80_800.0, 161_550.0, 0.0625),
                Bracket(161_550.0, 1_077_550.0, 0.0685),
                Bracket(1_077_550.0, float("inf"), 0.0882),
            ],
        }
        self.nyc_local_rate = 0.03876
        self.standard_deduction = {
            FilingStatus.SINGLE: 13_850.0,
            FilingStatus.MARRIED_JOINT: 27_700.0,
            FilingStatus.MARRIED_SEPARATE: 13_850.0,
        }
        self.salt_limit = {
            FilingStatus.SINGLE: 10_000.0,
            FilingStatus.MARRIED_JOINT: 10_000.0,
            FilingStatus.MARRIED_SEPARATE: 5_000.0,
        }

    @staticmethod
    def get_mortgage_interest_deduction_limit(mortgage_rates: np.ndarray, filing_status: np.ndarray) -> np.ndarray:
        """Get the mortgage interest deduction limit based on filing status."""
        return np.where(
            filing_status == FilingStatus.MARRIED_SEPARATE, 375_000.0 * mortgage_rates, 750_000.0 * mortgage_rates
        )

    @staticmethod
    def get_standard_deduction(filing_status: np.ndarray) -> np.ndarray:
        """Get the standard deduction based on the filing status."""
        return np.where(filing_status == FilingStatus.MARRIED_JOINT, 27_700.0, 13_850.0)

    @staticmethod
    def get_salt_deduction_limit(filing_status: np.ndarray) -> np.ndarray:
        """Get the SALT deduction limit based on filing status."""
        return np.where(filing_status == FilingStatus.MARRIED_SEPARATE, 5_000.0, 10_000.0)

    def calculate_tax_deduction(
        self,
        ages: np.ndarray,
        mortgage_rates: np.ndarray,
        mortgage_interests: np.ndarray,
        property_taxes: np.ndarray,
        federal_rates: np.ndarray,
        state_rates: np.ndarray,
        local_rates: np.ndarray,
        retirement_contributions: np.ndarray,
        annual_incomes: np.ndarray,
        filing_status: np.ndarray,
    ) -> np.ndarray:
        """Calculate tax deductions (vectorized for multiple inputs)."""
        # Calculate retirement contribution limits
        base_contribution_limit = 22_500.0  # 2023 limit for 401(k)
        catch_up_contribution = np.where(ages >= 50, 7_500.0, 0.0)  # 2023 catch-up contribution for age 50+
        retirement_contribution_limits = np.minimum(base_contribution_limit + catch_up_contribution, annual_incomes)

        # Calculate standard deductions
        standard_deductions = self.get_standard_deduction(filing_status)

        # Cap the retirement contributions
        capped_retirement_contributions = np.minimum(retirement_contributions, retirement_contribution_limits)

        # Calculate SALT deductions
        salt_deductions = np.minimum(self.get_salt_deduction_limit(filing_status), property_taxes)

        # Calculate mortgage interest deductions
        mortgage_interest_limits = self.get_mortgage_interest_deduction_limit(mortgage_rates, filing_status)
        mortgage_interest_deductions = np.minimum(mortgage_interests, mortgage_interest_limits)

        # Federal itemized deductions
        federal_itemized_deductions = salt_deductions + mortgage_interest_deductions + capped_retirement_contributions

        # State itemized deductions (New York doesn't limit SALT deductions)
        state_itemized_deductions = property_taxes + mortgage_interests + capped_retirement_contributions

        # Calculate deductions
        federal_deductions = np.maximum(0, federal_itemized_deductions - standard_deductions) * federal_rates
        state_deductions = np.maximum(0, state_itemized_deductions - standard_deductions) * (state_rates + local_rates)

        return federal_deductions + state_deductions

    @staticmethod
    def _adjust_brackets_for_inflation(brackets: list[Bracket], inflation_rates: np.ndarray) -> list[Bracket]:
        """Adjust tax brackets for inflation."""
        adjusted_brackets = []
        for lower, upper, rate in brackets:
            adjusted_upper = (
                np.full_like(inflation_rates, np.inf) if np.isinf(upper) else upper * (1.0 + inflation_rates)
            )
            adjusted_lower = lower * (1.0 + inflation_rates)
            adjusted_brackets.append(Bracket(adjusted_lower, adjusted_upper, rate))
        return adjusted_brackets

    def calculate_tax(
        self, incomes: np.ndarray, brackets: list[Bracket], inflation_rates: np.ndarray | None = None
    ) -> np.ndarray:
        """Calculate the tax amount based on income and tax brackets (vectorized for multiple incomes)."""
        tax = np.zeros_like(incomes)
        if inflation_rates is not None:
            adjusted_brackets = self._adjust_brackets_for_inflation(brackets, inflation_rates)
        else:
            adjusted_brackets = brackets

        for lower, upper, rate in adjusted_brackets:
            tax += np.maximum(0, np.minimum(incomes - lower, upper - lower)) * rate
        return tax

    def calculate_effective_tax_rates(
        self,
        incomes: np.ndarray,
        filing_status: np.ndarray,
        federal_adj: np.ndarray | None = None,
        state_adj: np.ndarray | None = None,
        local_adj: np.ndarray | None = None,
        inflation_rates: np.ndarray | None = None,
    ) -> EffectiveTaxRate:
        """Calculate effective tax rates based on income and random adjustments (vectorized for multiple incomes)."""
        if federal_adj is None:
            federal_adj = np.zeros_like(incomes)
        if state_adj is None:
            state_adj = np.zeros_like(incomes)
        if local_adj is None:
            local_adj = np.zeros_like(incomes)

        federal_tax = np.zeros_like(incomes)
        state_tax = np.zeros_like(incomes)

        for status in FilingStatus:
            mask = filing_status == status
            federal_tax[mask] = self.calculate_tax(
                incomes[mask],
                self.federal_brackets[status],
                inflation_rates[mask] if inflation_rates is not None else None,
            )
            state_tax[mask] = self.calculate_tax(
                incomes[mask],
                self.ny_state_brackets[status],
                inflation_rates[mask] if inflation_rates is not None else None,
            )

        local_tax = incomes * self.nyc_local_rate

        # Calculate effective rates and apply adjustments
        federal_rate = (federal_tax / incomes) + federal_adj
        state_rate = (state_tax / incomes) + state_adj
        local_rate = (local_tax / incomes) + local_adj

        # Ensure rates are non-negative
        return EffectiveTaxRate(np.maximum(0.0, federal_rate), np.maximum(0.0, state_rate), np.maximum(0.0, local_rate))

    @staticmethod
    def get_tax_brackets(incomes: np.ndarray, brackets: list[Bracket]) -> np.ndarray:
        """Get the tax brackets for given incomes (vectorized)."""
        rates = np.zeros_like(incomes)
        for lower, _, rate in brackets:
            rates = np.where(incomes > lower, rate, rates)
        return rates

    def get_federal_tax_brackets(self, incomes: np.ndarray) -> np.ndarray:
        """Get the federal tax brackets for given incomes (vectorized)."""
        return self.get_tax_brackets(incomes, self.federal_brackets)

    def get_state_tax_brackets(self, incomes: np.ndarray) -> np.ndarray:
        """Get the state tax brackets for given incomes (vectorized)."""
        return self.get_tax_brackets(incomes, self.ny_state_brackets)
