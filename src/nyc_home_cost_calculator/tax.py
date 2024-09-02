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


class TaxCalculator:
    """Tax calculator class for calculating tax deductions and rates based on income and filing status."""

    def __init__(self, filing_status: FilingStatus):
        """Initialize the tax calculator with the given filing status."""
        self.filing_status = filing_status
        self.federal_brackets = self.get_federal_brackets()
        self.ny_state_brackets = self.get_ny_state_brackets()
        self.nyc_local_rate = 0.03876
        self.standard_deduction = self.get_standard_deduction()
        self.salt_limit = self.get_salt_deduction_limit()

    def get_federal_brackets(self) -> list[tuple[float, float, float]]:
        """Get the federal tax brackets based on the filing status."""
        if self.filing_status == FilingStatus.SINGLE:
            brackets = [
                (0.0, 11_000.0, 0.10),
                (11_000.0, 44_725.0, 0.12),
                (44_725.0, 95_375.0, 0.22),
                (95_375.0, 182_100.0, 0.24),
                (182_100.0, 231_250.0, 0.32),
                (231_250.0, 578_125.0, 0.35),
                (578_125.0, float("inf"), 0.37),
            ]
        elif self.filing_status == FilingStatus.MARRIED_JOINT:
            brackets = [
                (0.0, 22_000.0, 0.10),
                (22_000.0, 89_450.0, 0.12),
                (89_450.0, 190_750.0, 0.22),
                (190_750.0, 364_200.0, 0.24),
                (364_200.0, 462_500.0, 0.32),
                (462_500.0, 693_750.0, 0.35),
                (693_750.0, float("inf"), 0.37),
            ]
        elif self.filing_status == FilingStatus.MARRIED_SEPARATE:
            brackets = [
                (0.0, 11_000.0, 0.10),
                (11_000.0, 44_725.0, 0.12),
                (44_725.0, 95_375.0, 0.22),
                (95_375.0, 182_100.0, 0.24),
                (182_100.0, 231_250.0, 0.32),
                (231_250.0, 346_875.0, 0.35),
                (346_875.0, float("inf"), 0.37),
            ]
        else:
            msg = "Invalid FilingStatus."
            raise ValueError(msg)
        return brackets

    def get_ny_state_brackets(self) -> list[tuple[float, float, float]]:
        """Get the New York state tax brackets based on the filing status."""
        if self.filing_status == FilingStatus.SINGLE:
            brackets = [
                (0.0, 8_500.0, 0.04),
                (8_500.0, 11_700.0, 0.045),
                (11_700.0, 13_900.0, 0.0525),
                (13_900.0, 80_650.0, 0.0585),
                (80_650.0, 215_400.0, 0.0625),
                (215_400.0, 1_077_550.0, 0.0685),
                (1_077_550.0, float("inf"), 0.0882),
            ]
        elif self.filing_status == FilingStatus.MARRIED_JOINT:
            brackets = [
                (0.0, 17150.0, 0.04),
                (17_150.0, 23_600.0, 0.045),
                (23_600.0, 27_900.0, 0.0525),
                (27_900.0, 161_550.0, 0.0585),
                (161_550.0, 323_200.0, 0.0625),
                (323_200.0, 2_155_350.0, 0.0685),
                (2_155_350.0, float("inf"), 0.0882),
            ]
        elif self.filing_status == FilingStatus.MARRIED_SEPARATE:
            brackets = [
                (0.0, 8_500.0, 0.04),
                (8_500.0, 11_800.0, 0.045),
                (11_800.0, 13_950.0, 0.0525),
                (13_950.0, 80_800.0, 0.0585),
                (80_800.0, 161_550.0, 0.0625),
                (161_550.0, 1_077_550.0, 0.0685),
                (1_077_550.0, float("inf"), 0.0882),
            ]
        else:
            msg = "Invalid FilingStatus."
            raise ValueError(msg)
        return brackets

    def get_standard_deduction(self) -> float:
        """Get the standard deduction based on the filing status."""
        if self.filing_status == FilingStatus.SINGLE:
            standard_deduction = 13_850.0
        elif self.filing_status == FilingStatus.MARRIED_JOINT:
            standard_deduction = 27_700.0
        elif self.filing_status == FilingStatus.MARRIED_SEPARATE:
            standard_deduction = 13_850.0
        else:
            msg = "Invalid FilingStatus."
            raise ValueError(msg)
        return standard_deduction

    def get_salt_deduction_limit(self) -> float:
        """Get the SALT deduction limit based on filing status."""
        if self.filing_status == FilingStatus.MARRIED_SEPARATE:
            return 5_000.0
        return 10_000.0

    def get_mortgage_interest_deduction_limit(self, mortgage_rate: np.ndarray) -> np.ndarray:
        """Get the mortgage interest deduction limit based on filing status."""
        if self.filing_status == FilingStatus.MARRIED_SEPARATE:
            return 375_000.0 * mortgage_rate
        return 750_000.0 * mortgage_rate

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
    ) -> np.ndarray:
        """Calculate tax deductions (vectorized for multiple inputs)."""
        # Calculate retirement contribution limits
        base_contribution_limit = 22_500  # 2023 limit for 401(k)
        catch_up_contribution = np.where(ages >= 50, 7_500, 0)  # 2023 catch-up contribution for age 50+
        retirement_contribution_limits = np.minimum(base_contribution_limit + catch_up_contribution, annual_incomes)

        # Cap the retirement contributions
        capped_retirement_contributions = np.minimum(retirement_contributions, retirement_contribution_limits)

        # Calculate SALT deductions
        salt_deductions = np.minimum(self.salt_limit, property_taxes)

        # Calculate mortgage interest deductions
        mortgage_interest_limits = self.get_mortgage_interest_deduction_limit(mortgage_rates)
        mortgage_interest_deductions = np.minimum(mortgage_interests, mortgage_interest_limits)

        # Federal itemized deductions
        federal_itemized_deductions = salt_deductions + mortgage_interest_deductions + capped_retirement_contributions

        # State itemized deductions (New York doesn't limit SALT deductions)
        state_itemized_deductions = property_taxes + mortgage_interests + capped_retirement_contributions

        # Calculate deductions
        federal_deductions = np.maximum(0, federal_itemized_deductions - self.standard_deduction) * federal_rates
        state_deductions = np.maximum(0, state_itemized_deductions - self.standard_deduction) * (
            state_rates + local_rates
        )

        return federal_deductions + state_deductions

    def calculate_tax(self, incomes: np.ndarray, brackets: list[tuple[float, float, float]]) -> np.ndarray:
        """Calculate the tax amount based on income and tax brackets (vectorized for multiple incomes)."""
        tax = np.zeros_like(incomes)
        for lower, upper, rate in brackets:
            tax += np.clip(incomes - lower, 0.0, upper - lower) * rate
        return tax

    def calculate_effective_tax_rates(
        self,
        incomes: np.ndarray,
        federal_adj: np.ndarray | None = None,
        state_adj: np.ndarray | None = None,
        local_adj: np.ndarray | None = None,
    ) -> EffectiveTaxRate:
        """Calculate effective tax rates based on income and random adjustments (vectorized for multiple incomes)."""
        if federal_adj is None:
            federal_adj = np.zeros_like(incomes)
        if state_adj is None:
            state_adj = np.zeros_like(incomes)
        if local_adj is None:
            local_adj = np.zeros_like(incomes)

        federal_tax = self.calculate_tax(incomes, self.federal_brackets)
        state_tax = self.calculate_tax(incomes, self.ny_state_brackets)
        local_tax = incomes * self.nyc_local_rate

        # Calculate effective rates and apply adjustments
        federal_rate = (federal_tax / incomes) + federal_adj
        state_rate = (state_tax / incomes) + state_adj
        local_rate = (local_tax / incomes) + local_adj

        # Ensure rates are non-negative
        return EffectiveTaxRate(np.maximum(0.0, federal_rate), np.maximum(0.0, state_rate), np.maximum(0.0, local_rate))

    def get_tax_brackets(self, incomes: np.ndarray, brackets: list[tuple[float, float, float]]) -> np.ndarray:
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
