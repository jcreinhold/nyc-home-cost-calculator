"""This module provides classes and functions for simulating home costs in NYC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    T = TypeVar("T")


@dataclass
class SimulationResults:
    """Represents the results of various financial simulations."""

    # Common fields
    monthly_costs: np.ndarray
    profit_loss: np.ndarray

    # Home ownership specific
    home_values: np.ndarray | None = None
    remaining_mortgage_balance: np.ndarray | None = None
    property_taxes: np.ndarray | None = None
    insurance_costs: np.ndarray | None = None
    maintenance_costs: np.ndarray | None = None

    # Investment specific
    portfolio_values: np.ndarray | None = None
    investment_returns: np.ndarray | None = None

    # Career and life event specific
    personal_income: np.ndarray | None = None
    monthly_income: np.ndarray | None = None
    promotions: np.ndarray | None = None
    demotions: np.ndarray | None = None
    layoffs: np.ndarray | None = None
    marital_status: np.ndarray | None = None
    partner_income: np.ndarray | None = None
    household_income: np.ndarray | None = None

    # Tax related
    tax_deductions: np.ndarray | None = None
    federal_effective_tax_rate: np.ndarray | None = None
    state_effective_tax_rate: np.ndarray | None = None
    local_effective_tax_rate: np.ndarray | None = None
    after_tax_income: np.ndarray | None = None

    # Misc financial
    cumulative_costs: np.ndarray | None = None
    cumulative_savings: np.ndarray | None = None

    # Additional fields for flexibility
    extra: dict = field(default_factory=dict)

    def __repr__(self):
        """Provide a concise representation of the SimulationResults."""
        fields = [
            f"{k}={v.shape if isinstance(v, np.ndarray) else v}" for k, v in asdict(self).items() if v is not None
        ]
        return f"SimulationResults({', '.join(fields)})"

    def get_percentiles(self, field: str, percentiles: list[float]) -> np.ndarray:
        """Calculate percentiles for a given field across all simulations.

        Args:
            field: The name of the field to calculate percentiles for.
            percentiles: List of percentiles to calculate (e.g., [0.05, 0.5, 0.95]).

        Returns:
            Array of percentile values for each time step.
        """
        data = getattr(self, field)
        if data is None:
            msg = f"Field '{field}' not found in simulation results."
            raise ValueError(msg)
        return np.percentile(data, percentiles, axis=1)

    def get_mean(self, field: str) -> np.ndarray:
        """Calculate the mean for a given field across all simulations.

        Args:
            field: The name of the field to calculate the mean for.

        Returns:
            Array of mean values for each time step.
        """
        data = getattr(self, field)
        if data is None:
            msg = f"Field '{field}' not found in simulation results."
            raise ValueError(msg)
        return np.mean(data, axis=1)

    def get_std_dev(self, field: str) -> np.ndarray:
        """Calculate the standard deviation for a given field across all simulations.

        Args:
            field: The name of the field to calculate the standard deviation for.

        Returns:
            Array of standard deviation values for each time step.
        """
        data = getattr(self, field)
        if data is None:
            msg = f"Field '{field}' not found in simulation results."
            raise ValueError(msg)
        return np.std(data, axis=1)

    def __add__(self, other: SimulationResults) -> SimulationResults:
        """Add two SimulationResults objects together.

        Args:
            other: The SimulationResults object to add.

        Returns:
            The combined SimulationResults object.
        """
        return self._combine(other, np.add)

    def __sub__(self, other: SimulationResults) -> SimulationResults:
        """Subtract two SimulationResults objects.

        Args:
            other: The SimulationResults object to subtract.

        Returns:
            The subtracted SimulationResults object.
        """
        return self._combine(other, np.subtract)

    def __mul__(self, other: SimulationResults) -> SimulationResults:
        """Multiply two SimulationResults objects together.

        Args:
            other: The SimulationResults object to multiply.

        Returns:
            The multiplied SimulationResults object.
        """
        return self._combine(other, np.multiply)

    def __truediv__(self, other: SimulationResults) -> SimulationResults:
        """Divide two SimulationResults objects element-wise.

        Args:
            other: The SimulationResults object to divide.

        Returns:
            The divided SimulationResults object.
        """
        return self._combine(other, np.divide)

    def _combine(self, other: SimulationResults, operation: Callable) -> SimulationResults:
        if not isinstance(other, SimulationResults):
            msg = f"Unsupported operand type for SimulationResults: {type(other)}"
            raise TypeError(msg)

        def _combine_arrays(a: T, b: T) -> T | dict | None:
            if a is None:
                return b
            if b is None:
                return a
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return operation(a, b)
            if isinstance(a, dict) and isinstance(b, dict):
                return {**a, **b}
            return None

        combined_dict: dict[str, np.ndarray | dict | None] = {}
        for _field in self.__dataclass_fields__:
            if _field == "extra":
                combined_dict[_field] = _combine_arrays(self.extra, other.extra)
            else:
                combined_dict[_field] = _combine_arrays(getattr(self, _field), getattr(other, _field))

        return SimulationResults(**combined_dict)  # type: ignore[arg-type]

    @classmethod
    def zeros_like(cls, other: SimulationResults) -> SimulationResults:
        """Create a new SimulationResults instance with zero-filled arrays like another instance."""
        zero_dict: dict[str, Any] = {}
        for _field, value in asdict(other).items():
            if isinstance(value, np.ndarray):
                zero_dict[_field] = np.zeros_like(value)
            elif _field == "extra":
                zero_dict[_field] = {}
            else:
                zero_dict[_field] = None
        return cls(**zero_dict)


class SimulationEngine:
    """Represents a simulation engine for calculating home costs in NYC."""

    def __init__(self, total_months: int, simulations: int, rng: np.random.Generator | None = None):
        """Initialize the SimulationEngine object.

        Args:
            total_months: The total number of months for the simulation.
            simulations: The number of simulations to run.
            rng: The random number generator to use.
        """
        self.total_months = total_months
        self.simulations = simulations
        self.rng = rng or np.random.default_rng()

    def run_simulation(self, simulate_vectorized: Callable[[np.ndarray], SimulationResults]) -> SimulationResults:
        """Run the simulation using the provided vectorized simulation function.

        Args:
            simulate_vectorized: A vectorized simulation function that takes an array of months as input and returns
                the monthly costs and cumulative costs.

        Returns:
            The results of the simulation, including the monthly costs and profit/loss.
        """
        months = np.arange(self.total_months)[:, np.newaxis]
        months_matrix = np.tile(months, (1, self.simulations))
        return simulate_vectorized(months_matrix)
