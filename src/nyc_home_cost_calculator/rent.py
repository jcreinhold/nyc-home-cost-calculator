"""Module for calculating and simulating the long-term costs of renting in NYC."""

from __future__ import annotations

import numpy as np

from nyc_home_cost_calculator.base import AbstractNYCCostCalculator, LocScaleRV
from nyc_home_cost_calculator.simulate import SimulationResults


class NYCRentalCostCalculator(AbstractNYCCostCalculator):
    """A class to calculate and simulate the long-term costs renting in NYC.

    This calculator takes into account various factors such as moving costs,
    income changes, market volatility, taxes, and other associated costs of renting.
    It uses Monte Carlo simulation to project potential financial outcomes over time.
    """

    def __init__(
        self,
        initial_rent: float = 4_500.0,
        lease_term: float = 1.0,
        total_years: int = 30,
        utility_cost: float = 100.0,
        renters_insurance: float = 200.0,
        moving_cost: float = 2_500.0,
        mean_rent_increase_rate: float = 0.03,
        rent_increase_volatility: float = 0.02,
        mean_inflation_rate: float = 0.02,
        inflation_volatility: float = 0.01,
        broker_fee_rate: float = 0.15,
        move_probability: float = 0.1,
        rent_increase_move_threshold: float = 0.1,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the NYCRentalCostCalculator with rental and financial parameters.

        Args:
            initial_rent: The initial monthly rent. Defaults to $4,500.
            lease_term: Length of the lease term in years. Defaults to 1 year.
            total_years: Length of the time renting in years. Defaults to 30 years.
            initial_income: Initial annual income of the renter. Defaults to $150,000.
            utility_cost: Monthly utility cost. Defaults to $100 per month.
            renters_insurance: Annual renters insurance premium. Defaults to $200 per year.
            moving_cost: Initial moving costs. Defaults to $2,500.
            mean_rent_increase_rate: Expected annual rent increase rate. Defaults to 3%.
            rent_increase_volatility: Volatility of the rent increase rate. Defaults to 2%.
            mean_inflation_rate: Expected annual inflation rate. Defaults to 2% a year.
            inflation_volatility: Volatility of the inflation rate. Defaults to 1%.
            broker_fee_rate: Broker fee as a percentage of annual rent. Defaults to 15%.
            move_probability: The probability of moving after a lease is up. Defaults to 10%.
            rent_increase_move_threshold: The percentage threshold at which you move. Defaults to 10%.
            simulations: Number of Monte Carlo simulations to run. Defaults to 10,000.
            rng: Custom random number generator. If None, use default numpy RNG.
        """
        super().__init__(
            initial_cost=initial_rent,
            total_years=total_years,
            mean_inflation_rate=mean_inflation_rate,
            inflation_volatility=inflation_volatility,
            simulations=simulations,
            rng=rng,
        )
        self.initial_rent = self.initial_cost
        self.lease_term = lease_term
        self.utility_cost = utility_cost
        self.renters_insurance = renters_insurance
        self.moving_cost = moving_cost
        self.mean_rent_increase_rate = mean_rent_increase_rate
        self.rent_increase_volatility = rent_increase_volatility
        self.broker_fee_rate = broker_fee_rate
        self.move_probability = move_probability
        self.rent_increase_move_threshold = rent_increase_move_threshold

        self.rent_increase_rate = LocScaleRV(self.mean_rent_increase_rate, self.rent_increase_volatility, self.rng)
        self.inflation_rate = LocScaleRV(self.mean_inflation_rate, self.inflation_volatility, self.rng)

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        total_months, num_simulations = shape = months.shape

        # Generate random rates
        rent_increase_rates, inflation_rates = self.rent_increase_rate(shape), self.inflation_rate(shape)

        # Initialize arrays
        rents = np.full(shape, self.initial_rent)
        utility_costs = np.full(shape, self.utility_cost)
        moving_costs = np.full(shape, self.moving_cost)
        monthly_costs = np.zeros(shape)

        # Calculate cumulative rates
        cumulative_rent_increase = np.cumprod(1.0 + rent_increase_rates / 12.0, axis=0)
        cumulative_inflation = np.cumprod(1.0 + inflation_rates / 12.0, axis=0)

        # Update values over time
        rents *= cumulative_rent_increase
        utility_costs *= cumulative_inflation
        moving_costs *= cumulative_inflation

        # Calculate lease end months
        lease_end_months = np.arange(0, total_months, int(self.lease_term * 12))

        for i, lease_end in enumerate(lease_end_months):
            if i == 0:
                continue  # Skip the first lease end (initial move-in)

            # Check if we should move
            rent_increase = (rents[lease_end] - rents[lease_end - 1]) / rents[lease_end - 1]
            should_move = (rent_increase > self.rent_increase_move_threshold) | (
                self.rng.random(num_simulations) < self.move_probability
            )
            # Calculate new rent after move
            previous_rent = rents[lease_end - 1]
            market_rent = rents[lease_end]
            monthly_moving_cost = moving_costs[lease_end] / 12.0

            # Ensure the lower bound is always less than the upper bound
            lower_bound = np.minimum(previous_rent, market_rent - monthly_moving_cost)
            upper_bound = np.maximum(previous_rent, market_rent)

            new_rent_after_move = np.where(should_move, self.rng.uniform(lower_bound, upper_bound), market_rent)

            # Apply move effects
            move_costs = np.where(
                should_move, moving_costs[lease_end] + (new_rent_after_move * 12 * self.broker_fee_rate), 0
            )
            monthly_costs[lease_end] += move_costs

            # Correct the broadcasting for rent updates
            rent_mask = np.broadcast_to(should_move[:, np.newaxis], (num_simulations, total_months - lease_end)).T
            new_rents = np.broadcast_to(
                new_rent_after_move[:, np.newaxis], (num_simulations, total_months - lease_end)
            ).T
            rents[lease_end:] = np.where(rent_mask, new_rents, rents[lease_end:])

        # Calculate monthly costs
        monthly_insurance = self.renters_insurance / 12.0
        monthly_costs += rents + utility_costs + monthly_insurance

        # Calculate cumulative costs
        cumulative_costs = np.cumsum(monthly_costs, axis=0)

        # Add initial move-in costs
        cumulative_costs += self.moving_cost + (self.initial_rent * 12 * self.broker_fee_rate)

        return SimulationResults(monthly_costs=monthly_costs, profit_loss=-cumulative_costs, extra={})

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        return [
            ("Initial Monthly Rent", f"${self.initial_rent:,.2f}"),
            ("Lease Term", f"{self.lease_term} years"),
            ("Total Years", f"{self.total_years} years"),
            ("Monthly Utility Cost", f"${self.utility_cost:,.2f}"),
            ("Annual Renters Insurance", f"${self.renters_insurance:,.2f}"),
            ("Moving Cost", f"${self.moving_cost:,.2f}"),
            ("Mean Rent Increase Rate", f"{self.mean_rent_increase_rate:.2%}"),
            ("Rent Increase Volatility", f"{self.rent_increase_volatility:.2%}"),
            ("Mean Inflation Rate", f"{self.mean_inflation_rate:.2%}"),
            ("Inflation Volatility", f"{self.inflation_volatility:.2%}"),
            ("Broker Fee Rate", f"{self.broker_fee_rate:.2%}"),
            ("Move Probability", f"{self.move_probability:.2%}"),
            ("Rent Increase Move Threshold", f"{self.rent_increase_move_threshold:.2%}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]
