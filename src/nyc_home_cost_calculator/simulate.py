"""This module provides classes and functions for simulating home costs in NYC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class SimulationResults(NamedTuple):
    """Represents the results of a simulation."""

    monthly_costs: np.ndarray
    profit_loss: np.ndarray
    extra: dict[str, Any] | None = None


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
