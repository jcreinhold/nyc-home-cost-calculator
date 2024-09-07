"""Calculate investment returns/costs."""

from __future__ import annotations

import numpy as np
from scipy import stats

from nyc_home_cost_calculator.base import AbstractNYCCostCalculator
from nyc_home_cost_calculator.simulate import SimulationResults


class InvestmentCalculator(AbstractNYCCostCalculator):
    """Calculates investment-related costs for NYC."""

    def __init__(
        self,
        initial_investment: float = 100_000.0,
        monthly_contribution: float = 1_000.0,
        total_years: int = 30,
        mean_return_rate: float = 0.07,
        volatility: float = 0.15,
        degrees_of_freedom: int = 5,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the InvestmentCalculator object.

        Args:
            initial_investment: The initial investment amount. Defaults to 100_000.0.
            monthly_contribution: The monthly contribution amount. Defaults to 1_000.0.
            total_years: The total number of years for the investment. Defaults to 30.
            mean_return_rate: The mean annual return rate. Defaults to 0.07.
            volatility: The annual volatility. Defaults to 0.15.
            degrees_of_freedom: The degrees of freedom for the Student's t-distribution. Defaults to 5.
            simulations: The number of simulations to run. Defaults to 5_000.
            rng: The random number generator. Defaults to None.
        """
        super().__init__(
            initial_cost=initial_investment,
            total_years=total_years,
            mean_inflation_rate=0,  # Not used in this class
            inflation_volatility=0,  # Not used in this class
            simulations=simulations,
            rng=rng,
        )
        self.initial_investment = initial_investment
        self.monthly_contribution = monthly_contribution
        self.mean_return_rate = mean_return_rate
        self.volatility = volatility
        self.degrees_of_freedom = degrees_of_freedom

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        total_months, num_simulations = shape = months.shape

        # Generate monthly returns using Student's t-distribution
        monthly_returns = stats.t.rvs(
            df=self.degrees_of_freedom,
            loc=self.mean_return_rate / 12.0,
            scale=self.volatility / np.sqrt(12.0),
            size=shape,
            random_state=self.rng,
        )

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + monthly_returns, axis=0)

        # Create a matrix of monthly contributions
        contributions = np.full(shape, self.monthly_contribution)
        contributions[0] += self.initial_investment

        # Calculate portfolio values
        portfolio_values = np.zeros(shape)
        portfolio_values[0] = self.initial_investment

        # Vectorized calculation of portfolio values
        portfolio_values[1:] = (
            self.initial_investment * cumulative_returns[1:]
            + self.monthly_contribution
            * (cumulative_returns[1:] * np.cumprod(1 + monthly_returns[:-1], axis=0) - 1.0)
            / monthly_returns[:-1]
        )

        # Handle the case where monthly return is zero
        zero_returns = monthly_returns[:-1] == 0.0
        if np.any(zero_returns):
            contribution_periods = np.arange(1, total_months)[:, np.newaxis]
            portfolio_values[1:] = np.where(
                zero_returns,
                self.initial_investment * cumulative_returns[1:] + self.monthly_contribution * contribution_periods,
                portfolio_values[1:],
            )

        # Calculate cumulative contributions
        cumulative_contributions = np.cumsum(contributions, axis=0)

        return SimulationResults(
            monthly_costs=-contributions,
            profit_loss=portfolio_values - cumulative_contributions,
            portfolio_values=portfolio_values,
            investment_returns=cumulative_returns,
            extra={
                "monthly_returns": monthly_returns,
            },
        )

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        return [
            ("Initial Investment", f"${self.initial_investment:,.2f}"),
            ("Monthly Contribution", f"${self.monthly_contribution:,.2f}"),
            ("Total Years", f"{self.total_years}"),
            ("Mean Annual Return Rate", f"{self.mean_return_rate:.2%}"),
            ("Annual Volatility", f"{self.volatility:.2%}"),
            ("Degrees of Freedom", f"{self.degrees_of_freedom}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]
