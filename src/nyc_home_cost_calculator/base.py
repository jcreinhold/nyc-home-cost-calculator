"""This module contains the base class for the NYC Home/Rent Cost Calculator."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from nyc_home_cost_calculator.simulate import SimulationEngine, SimulationResults

logger = logging.getLogger(__name__)


class AbstractSimulatorBase(ABC):
    """Abstract base class for all financial simulators."""

    def __init__(
        self,
        total_years: int,
        simulations: int,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the simulator base.

        Args:
            total_years: The total number of years to simulate.
            simulations: The number of Monte Carlo simulations to run.
            rng: Custom random number generator. If None, use default numpy RNG.
        """
        self.total_years = total_years
        self.simulations = simulations
        self.rng = rng or np.random.default_rng()
        self.simulation_engine = SimulationEngine(total_years * 12, simulations, self.rng)

    @abstractmethod
    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        """Run the vectorized simulation.

        This method should be implemented by subclasses to perform the actual
        simulation calculations.

        Args:
            months: A 2D numpy array of shape (total_months, num_simulations)
                    representing the months to simulate.

        Returns:
            A SimulationResults object containing the results of the simulation.
        """

    @abstractmethod
    def _get_input_parameters(self) -> list[tuple[str, Any]]:
        """Get the input parameters for the simulator.

        This method should be implemented by subclasses to return a list of
        tuples, where each tuple contains the name of a parameter and its value.

        Returns:
            A list of tuples, each containing a parameter name and its value.
        """

    def simulate(self) -> SimulationResults:
        """Run the simulation and return the results.

        This method sets up the simulation environment and calls the
        _simulate_vectorized method to perform the actual calculations.

        Returns:
            A SimulationResults object containing the results of the simulation.
        """
        return self.simulation_engine.run_simulation(self._simulate_vectorized)


class AbstractNYCCostCalculator(AbstractSimulatorBase):
    """Abstract base class for the NYC Home/Rent Cost Calculator."""

    def __init__(
        self,
        initial_cost: float,
        total_years: int,
        mean_inflation_rate: float,
        inflation_volatility: float,
        simulations: int,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the NYC Home/Rent Cost Calculator.

        Args:
            initial_cost: The initial cost of the home or rent.
            total_years: The total number of years for the calculation.
            mean_inflation_rate: The mean inflation rate.
            inflation_volatility: The inflation volatility.
            simulations: The number of simulations to run.
            rng: The random number generator.
        """
        super().__init__(total_years, simulations, rng)
        self.initial_cost = initial_cost
        self.mean_inflation_rate = mean_inflation_rate
        self.inflation_volatility = inflation_volatility
